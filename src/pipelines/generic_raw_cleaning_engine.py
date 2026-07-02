from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.data.load_raw import get_loader
from src.data.normalize_raw import (
    apply_cabinet_mapping,
    clean_cabinet_names,
    encode_topgames_title_ranks,
    fix_missing_values,
    fix_supplier_names,
    normalize_columns,
)
from src.pipelines.config_paths import project_root_from_config, resolve_project_path


def _write_output(df: pd.DataFrame, output_path: str | None) -> None:
    if not output_path:
        return

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def clean_generic_raw_frame(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Apply the shared generic raw-cleaning sequence for secondary datasets."""
    output = normalize_columns(
        df,
        rename_map=cfg.get("rename_map"),
        standardize=cfg.get("standardize_columns", True),
        lower_case=cfg.get("lower_case_columns", True),
    )
    output = fix_missing_values(
        output,
        numeric_cols=cfg.get("numeric_cols"),
        typo_maps=cfg.get("typo_maps"),
        missing_tokens=cfg.get("missing_tokens"),
        string_cols_numeric_to_nan=cfg.get("string_cols"),
        date_cols=cfg.get("date_cols"),
        date_format=cfg.get("date_format"),
        lower_case=cfg.get("lower_case", True),
    )

    supplier_col = cfg.get("supplier_col", "supplier")
    if supplier_col in output.columns:
        output = fix_supplier_names(
            output,
            supplier_col=supplier_col,
            supplier_map=cfg.get("supplier_map", {}),
            verbose=False,
        )

    cabinet_col = cfg.get("cabinet_col", "cabinet_name")
    if cfg.get("clean_cabinet_names", False):
        output = clean_cabinet_names(
            output,
            cabinet_col=cabinet_col,
            artifact_map=cfg.get("cabinet_name_artifact_map"),
        )

    if cfg.get("cabinet_supplier_map"):
        output = apply_cabinet_mapping(
            output,
            cabinet_supplier_map=cfg.get("cabinet_supplier_map"),
            cabinet_col=cabinet_col,
            supplier_col=supplier_col,
        )

    if cfg.get("encode_topgames_title_ranks", False):
        output = encode_topgames_title_ranks(
            output,
            month_col=cfg.get("topgames_month_col", "data_month"),
            game_col=cfg.get("topgames_game_col", "game_name"),
            supplier_col=cfg.get("topgames_supplier_col", "supplier"),
            title_col=cfg.get("topgames_title_col", "title"),
            rank_col=cfg.get("topgames_rank_col", "rank"),
            prefix=cfg.get("topgames_title_prefix", "rank_title_"),
        )

    return output


def run_generic_raw_cleaning(config_path: str) -> dict[str, pd.DataFrame]:
    """Run the generic cleaner for one or more configured raw datasets."""
    with open(config_path, "r", encoding="utf-8") as file:
        cfg = yaml.safe_load(file)

    project_root = project_root_from_config(config_path)
    shared_cfg = cfg.get("shared", {})
    cleaned: dict[str, pd.DataFrame] = {}

    for dataset_cfg in cfg.get("datasets", []):
        dataset_cfg = dict(dataset_cfg)
        dataset_cfg["input_path"] = str(
            resolve_project_path(dataset_cfg["input_path"], project_root)
        )
        if dataset_cfg.get("output_path"):
            dataset_cfg["output_path"] = str(
                resolve_project_path(dataset_cfg["output_path"], project_root)
            )

        merged_cfg = {**shared_cfg, **dataset_cfg}
        loader = get_loader(dataset_cfg.get("loader", "csv"))
        df = loader(
            path=dataset_cfg["input_path"],
            batch_read=False,
            read_options=dataset_cfg.get("read_options"),
        )
        df = clean_generic_raw_frame(df, merged_cfg)
        _write_output(df, dataset_cfg.get("output_path"))
        cleaned[dataset_cfg["name"]] = df

    return cleaned
