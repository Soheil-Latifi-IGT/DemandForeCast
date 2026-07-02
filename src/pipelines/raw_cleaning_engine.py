from __future__ import annotations

from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.data.load_raw import get_loader
from src.data.normalize_raw import (
    apply_cabinet_mapping,
    apply_ownership_mapping,
    clean_cabinet_names,
    filter_include_values,
    filter_min_date,
    fix_complementary_columns,
    fix_missing_values,
    fix_supplier_names,
    hierarchical_impute,
    normalize_columns,
    set_majority_supplier_per_game,
)
from src.data.performance_supplier_validation import (
    build_cabinet_supplier_reference,
    validate_performance_supplier_matches,
)
from src.pipelines.config_paths import project_root_from_config, resolve_project_path

logger = logging.getLogger(__name__)


def _summarize_step(step_name: str, before: pd.DataFrame, after: pd.DataFrame) -> dict[str, Any]:
    common_cols = sorted(set(before.columns).intersection(after.columns))
    comparable_cells = before.shape[0] * len(common_cols)

    changed_cells: int | None = None
    if comparable_cells <= 250_000 and before.index.equals(after.index):
        before_common = before[common_cols].astype("object").where(before[common_cols].notna(), "<NA>")
        after_common = after[common_cols].astype("object").where(after[common_cols].notna(), "<NA>")
        changed_cells = int((before_common != after_common).sum().sum())

    return {
        "step": step_name,
        "rows_before": int(before.shape[0]),
        "rows_after": int(after.shape[0]),
        "cols_before": int(before.shape[1]),
        "cols_after": int(after.shape[1]),
        "missing_before": int(before.isna().sum().sum()),
        "missing_after": int(after.isna().sum().sum()),
        "changed_cells": changed_cells,
        "added_columns": sorted(set(after.columns) - set(before.columns)),
        "dropped_columns": sorted(set(before.columns) - set(after.columns)),
    }


def _run_step(
    step_name: str,
    df: pd.DataFrame,
    fn,
    stats: list[dict[str, Any]],
    verbose: bool = False,
) -> pd.DataFrame:
    before = df.copy()
    after = fn(df)
    step_stats = _summarize_step(step_name, before, after)
    stats.append(step_stats)

    if verbose:
        logger.info(
            "[%s] rows %s->%s, cols %s->%s, missing %s->%s, changed_cells=%s",
            step_name,
            step_stats["rows_before"],
            step_stats["rows_after"],
            step_stats["cols_before"],
            step_stats["cols_after"],
            step_stats["missing_before"],
            step_stats["missing_after"],
            step_stats["changed_cells"]
            if step_stats["changed_cells"] is not None
            else "skipped_large_frame",
        )
    return after


def _write_run_log(
    stats: list[dict[str, Any]],
    cfg: dict[str, Any],
    input_path: str,
    output_path: str | None,
    initial: pd.DataFrame,
    final: pd.DataFrame,
) -> Path | None:
    logging_cfg = cfg.get("logging", {})
    if not logging_cfg.get("enabled", True):
        return None

    log_dir = Path(logging_cfg.get("dir", "logs/raw_cleaning"))
    keep_last = int(logging_cfg.get("keep_last", 4))
    log_dir.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_path = log_dir / f"run_{run_id}.json"
    payload = {
        "run_id": run_id,
        "input_path": input_path,
        "output_path": output_path,
        "initial_shape": list(initial.shape),
        "final_shape": list(final.shape),
        "columns_before": list(initial.columns),
        "columns_after": list(final.columns),
        "missing_cells_before": int(initial.isna().sum().sum()),
        "missing_cells_after": int(final.isna().sum().sum()),
        "added_columns": sorted(set(final.columns) - set(initial.columns)),
        "dropped_columns": sorted(set(initial.columns) - set(final.columns)),
        "steps": stats,
    }
    log_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    existing_logs = sorted(log_dir.glob("run_*.json"), key=lambda path: path.name, reverse=True)
    for old_log in existing_logs[keep_last:]:
        old_log.unlink()
    return log_path


def _resolve_supplier_validation_paths(
    cfg: dict[str, Any],
    project_root: Path,
) -> dict[str, Any]:
    supplier_validation_cfg = dict(cfg.get("supplier_consistency_validation", {}))
    if not supplier_validation_cfg:
        return cfg

    supplier_validation_cfg["report_dir"] = str(
        resolve_project_path(
            supplier_validation_cfg.get(
                "report_dir",
                "Data/processed/reports/performance_supplier_validation",
            ),
            project_root,
        )
    )
    external_sources = []
    for source in supplier_validation_cfg.get("external_sources", []):
        resolved_source = dict(source)
        if resolved_source.get("path"):
            resolved_source["path"] = str(resolve_project_path(resolved_source["path"], project_root))
        external_sources.append(resolved_source)
    supplier_validation_cfg["external_sources"] = external_sources
    cfg["supplier_consistency_validation"] = supplier_validation_cfg
    return cfg


def _load_raw_frame(input_path: str, cfg: dict[str, Any]) -> pd.DataFrame:
    loader_cfg = cfg.get("loader", "performance")
    if isinstance(loader_cfg, dict):
        loader_name = loader_cfg.get("name", "csv")
        read_options = loader_cfg.get("read_options", cfg.get("read_options"))
    else:
        loader_name = loader_cfg
        read_options = cfg.get("read_options")

    loader = get_loader(loader_name)
    return loader(path=input_path, batch_read=False, read_options=read_options)


def _write_output(df: pd.DataFrame, output_path: str | None) -> None:
    if not output_path:
        return

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def _supplier_columns(cfg: dict[str, Any], df: pd.DataFrame) -> list[str]:
    configured = cfg.get("supplier_columns")
    if configured:
        return [col for col in configured if col in df.columns]
    candidates = ["supplier", "parent_supplier", "manufacturer", "parent_manufacturer"]
    return [col for col in candidates if col in df.columns]


def run_raw_cleaning(config_path: str) -> pd.DataFrame:
    """Run the primary config-driven raw cleaning pipeline."""
    with open(config_path, "r", encoding="utf-8") as file:
        cfg = yaml.safe_load(file)

    project_root = project_root_from_config(config_path)
    input_path = str(resolve_project_path(cfg["input_path"], project_root))
    output_path = (
        str(resolve_project_path(cfg["output_path"], project_root))
        if cfg.get("output_path")
        else None
    )

    logging_cfg = dict(cfg.get("logging", {}))
    logging_cfg["dir"] = str(resolve_project_path(logging_cfg.get("dir", "logs/raw_cleaning"), project_root))
    cfg["logging"] = logging_cfg
    cfg = _resolve_supplier_validation_paths(cfg, project_root)

    df = _load_raw_frame(input_path, cfg)
    initial_df = df.copy()
    step_stats: list[dict[str, Any]] = []
    verbose = bool(cfg.get("verbose", False))

    date_filter_cfg = cfg.get("date_filter", {})
    if date_filter_cfg.get("enabled", False):
        df = _run_step(
            "filter_min_date",
            df,
            lambda frame: filter_min_date(
                frame,
                date_col=date_filter_cfg.get("column", "yearmonth"),
                min_date=date_filter_cfg["min_date"],
                date_format=date_filter_cfg.get("date_format", cfg.get("date_format")),
            ),
            step_stats,
            verbose,
        )

    include_values_filter_cfg = cfg.get("include_values_filter", {})
    if include_values_filter_cfg.get("enabled", False):
        df = _run_step(
            "filter_include_values",
            df,
            lambda frame: filter_include_values(
                frame,
                column=include_values_filter_cfg["column"],
                values=include_values_filter_cfg["values"],
                case_sensitive=include_values_filter_cfg.get("case_sensitive", False),
            ),
            step_stats,
            verbose,
        )

    df = _run_step(
        "normalize_columns",
        df,
        lambda frame: normalize_columns(
            frame,
            rename_map=cfg.get("rename_map"),
            standardize=cfg.get("standardize_columns", True),
            lower_case=cfg.get("lower_case_columns", True),
        ),
        step_stats,
        verbose,
    )

    df = _run_step(
        "fix_missing_values",
        df,
        lambda frame: fix_missing_values(
            frame,
            numeric_cols=cfg.get("numeric_cols"),
            typo_maps=cfg.get("typo_maps"),
            missing_tokens=cfg.get("missing_tokens"),
            string_cols_numeric_to_nan=cfg.get("string_cols"),
            date_cols=cfg.get("date_cols"),
            date_format=cfg.get("date_format"),
            lower_case=cfg.get("lower_case", True),
        ),
        step_stats,
        verbose,
    )

    df = _run_step(
        "fix_complementary_columns",
        df,
        lambda frame: fix_complementary_columns(
            frame,
            column_groups=cfg.get("complementary_columns", {}),
            drop_fallbacks=cfg.get("drop_complementary_fallbacks", False),
        ),
        step_stats,
        verbose,
    )

    for supplier_col in _supplier_columns(cfg, df):
        df = _run_step(
            f"fix_{supplier_col}_names",
            df,
            lambda frame, col=supplier_col: fix_supplier_names(
                frame,
                supplier_col=col,
                supplier_map=cfg.get("supplier_map", {}),
                verbose=False,
            ),
            step_stats,
            verbose,
        )

    df = _run_step(
        "apply_ownership_mapping",
        df,
        lambda frame: apply_ownership_mapping(frame, config=cfg.get("ownership_mapping")),
        step_stats,
        verbose,
    )

    ownership_status_filter_cfg = cfg.get("ownership_status_filter", {})
    if ownership_status_filter_cfg.get("enabled", False):
        df = _run_step(
            "filter_ownership_status",
            df,
            lambda frame: filter_include_values(
                frame,
                column=ownership_status_filter_cfg.get("column", "own_status"),
                values=ownership_status_filter_cfg.get("values", ["owned"]),
                case_sensitive=ownership_status_filter_cfg.get("case_sensitive", False),
            ),
            step_stats,
            verbose,
        )

    cabinet_cleaning_cfg = cfg.get("cabinet_cleaning", {})
    supplier_validation_cfg = cfg.get("supplier_consistency_validation", {})
    if cabinet_cleaning_cfg.get("enabled", False):
        df = _run_step(
            "clean_cabinet_names",
            df,
            lambda frame: clean_cabinet_names(
                frame,
                cabinet_col=cabinet_cleaning_cfg.get("column", "slot_cabinet_name"),
                artifact_map=cabinet_cleaning_cfg.get("artifact_map"),
                lower_case=cabinet_cleaning_cfg.get("lower_case", True),
            ),
            step_stats,
            verbose,
        )

    if supplier_validation_cfg.get("enabled", False):
        cabinet_col = supplier_validation_cfg.get("cabinet_col", "slot_cabinet_name")
        if not cabinet_cleaning_cfg.get("enabled", False):
            df = _run_step(
                "clean_cabinet_names",
                df,
                lambda frame: clean_cabinet_names(
                    frame,
                    cabinet_col=cabinet_col,
                    artifact_map=cfg.get("cabinet_name_artifact_map"),
                ),
                step_stats,
                verbose,
            )

        reference = build_cabinet_supplier_reference(
            df,
            validation_config=supplier_validation_cfg,
            supplier_map=cfg.get("supplier_map", {}),
            explicit_cabinet_supplier_map=cfg.get("cabinet_supplier_map", {}),
        )
        clean, mismatches, mismatch_summary = validate_performance_supplier_matches(
            df,
            reference,
            validation_config=supplier_validation_cfg,
        )

        report_dir = Path(supplier_validation_cfg["report_dir"])
        report_dir.mkdir(parents=True, exist_ok=True)
        reference.to_csv(report_dir / "cabinet_supplier_reference.csv", index=False)
        mismatches.to_csv(report_dir / "supplier_mismatch_rows.csv", index=False)
        mismatch_summary.to_csv(report_dir / "supplier_mismatch_summary.csv", index=False)
        reason_summary = (
            pd.concat([clean.assign(supplier_validation_reason="matched"), mismatches], ignore_index=True)[
                "supplier_validation_reason"
            ]
            .value_counts(dropna=False)
            .rename_axis("supplier_validation_reason")
            .reset_index(name="rows")
        )
        reason_summary.to_csv(report_dir / "supplier_validation_totals.csv", index=False)

        before_validation = df
        df = clean if supplier_validation_cfg.get("strict", True) else pd.concat(
            [clean, mismatches],
            ignore_index=True,
        )
        step_stats.append(_summarize_step("validate_supplier_consistency", before_validation, df))
    else:
        df = _run_step(
            "apply_cabinet_mapping",
            df,
            lambda frame: apply_cabinet_mapping(
                frame,
                cabinet_supplier_map=cfg.get("cabinet_supplier_map"),
                cabinet_col=cfg.get("cabinet_col", "slot_cabinet_name"),
                supplier_col=cfg.get("supplier_col", "supplier"),
            ),
            step_stats,
            verbose,
        )

    majority_supplier_cfg = cfg.get("majority_supplier_per_game", {})
    if majority_supplier_cfg.get("enabled", False) and not supplier_validation_cfg.get("enabled", False):
        df = _run_step(
            "set_majority_supplier_per_game",
            df,
            lambda frame: set_majority_supplier_per_game(
                frame,
                game_col=majority_supplier_cfg.get("game_col", "game_name"),
                game_cols=majority_supplier_cfg.get("game_cols"),
                cabinet_col=majority_supplier_cfg.get("cabinet_col", "slot_cabinet_name"),
                supplier_col=majority_supplier_cfg.get("supplier_col", "supplier"),
            ),
            step_stats,
            verbose,
        )

    imputation_cfg = cfg.get("hierarchical_imputation", {})
    if imputation_cfg.get("enabled", False):
        impute_cols = imputation_cfg.get("cols") or cfg.get("numeric_cols", [])
        df = _run_step(
            "hierarchical_impute",
            df,
            lambda frame: hierarchical_impute(
                frame,
                cols=impute_cols,
                group_hierarchy=imputation_cfg.get("group_hierarchy", []),
                add_missing_flags=imputation_cfg.get("add_missing_flags", False),
                numeric_strategy=imputation_cfg.get("numeric_strategy", "median"),
                categorical_strategy=imputation_cfg.get("categorical_strategy", "mode"),
            ),
            step_stats,
            verbose,
        )

    _write_output(df, output_path)
    log_path = _write_run_log(
        stats=step_stats,
        cfg=cfg,
        input_path=input_path,
        output_path=output_path,
        initial=initial_df,
        final=df,
    )
    if verbose and log_path:
        logger.info("[raw_cleaning] wrote run log to %s", log_path)
    return df
