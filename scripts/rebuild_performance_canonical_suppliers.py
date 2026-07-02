from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipelines.config_paths import project_root_from_config, resolve_project_path
from src.pipelines.performance_canonical_supplier import (
    apply_canonical_suppliers,
    build_canonical_supplier_map,
    build_cross_supplier_reports,
    build_sales_cabinet_reference,
    filter_excluded_games,
)
from src.pipelines.raw_cleaning_engine import run_raw_cleaning


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild performance with canonical suppliers.")
    parser.add_argument(
        "--config",
        default=str(ROOT / "config" / "raw_cleaning" / "performance.yml"),
        help="Path to the primary raw-cleaning YAML config.",
    )
    parser.add_argument("--corrected-sales-path", help="UTF-16 corrected cabinet-sales file.")
    parser.add_argument("--original-sales-path", help="UTF-16 original cabinet-sales file.")
    parser.add_argument("--output-path", help="Canonical cleaned performance CSV output.")
    parser.add_argument("--report-dir", help="Directory for canonical supplier reports.")
    return parser.parse_args()


def _base_clean(config: dict, root: Path) -> pd.DataFrame:
    base_config = dict(config)
    base_config["output_path"] = None
    base_config["cabinet_supplier_map"] = {}
    base_config["supplier_consistency_validation"] = {
        **config.get("supplier_consistency_validation", {}),
        "enabled": False,
    }
    base_config["majority_supplier_per_game"] = {
        **config.get("majority_supplier_per_game", {}),
        "enabled": False,
    }
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".yml",
            prefix="canonical_supplier_",
            dir=root,
            delete=False,
            encoding="utf-8",
        ) as file:
            yaml.safe_dump(base_config, file, sort_keys=False)
            temp_path = Path(file.name)
        return run_raw_cleaning(str(temp_path))
    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink()


def _write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    root = project_root_from_config(config_path)
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    canonical_cfg = config.get("canonical_supplier", {})
    corrected_sales_path = args.corrected_sales_path or canonical_cfg.get("corrected_sales_path")
    original_sales_path = args.original_sales_path or canonical_cfg.get("original_sales_path")
    output_path = args.output_path or canonical_cfg.get(
        "output_path",
        "Data/processed/performance_canonical_suppliers.csv",
    )
    report_dir = args.report_dir or canonical_cfg.get(
        "report_dir",
        "Data/processed/reports/performance_canonical_supplier",
    )
    if not corrected_sales_path or not original_sales_path:
        raise ValueError(
            "Set canonical_supplier.corrected_sales_path and original_sales_path, "
            "or pass --corrected-sales-path and --original-sales-path."
        )

    corrected_sales_path = resolve_project_path(corrected_sales_path, root)
    original_sales_path = resolve_project_path(original_sales_path, root)
    output_path = resolve_project_path(output_path, root)
    report_dir = resolve_project_path(report_dir, root)

    performance = _base_clean(config, root)
    input_rows = len(performance)
    performance, excluded = filter_excluded_games(performance)
    sales_reference = build_sales_cabinet_reference(
        corrected_sales_path=corrected_sales_path,
        original_sales_path=original_sales_path,
        supplier_map=config.get("supplier_map", {}),
    )
    canonical_map, cabinet_evidence = build_canonical_supplier_map(performance, sales_reference)
    performance = apply_canonical_suppliers(performance, canonical_map, sales_reference)
    pair_summary, game_summary, monthly_detail = build_cross_supplier_reports(performance)

    report_dir.mkdir(parents=True, exist_ok=True)
    _write_csv_atomic(canonical_map, report_dir / "canonical_game_supplier_map.csv")
    _write_csv_atomic(cabinet_evidence, report_dir / "canonical_game_cabinet_evidence.csv")
    _write_csv_atomic(sales_reference, report_dir / "sales_cabinet_supplier_reference.csv")
    _write_csv_atomic(excluded, report_dir / "excluded_games_summary.csv")
    _write_csv_atomic(pair_summary, report_dir / "cross_supplier_game_cabinet_summary.csv")
    _write_csv_atomic(game_summary, report_dir / "cross_supplier_game_summary.csv")
    _write_csv_atomic(monthly_detail, report_dir / "cross_supplier_game_cabinet_monthly.csv")

    unresolved = canonical_map.loc[canonical_map["canonical_supplier_status"].ne("resolved")]
    _write_csv_atomic(unresolved, report_dir / "unresolved_canonical_supplier_games.csv")
    _write_csv_atomic(performance, output_path)

    manifest = {
        "input_rows_after_base_clean": input_rows,
        "excluded_rows": int(input_rows - len(performance)),
        "output_rows": int(len(performance)),
        "output_columns": int(len(performance.columns)),
        "games": int(performance["game_name"].nunique()),
        "canonical_multi_supplier_games": int(
            canonical_map["observed_supplier_count"].gt(1).sum()
        ),
        "unresolved_canonical_supplier_games": int(len(unresolved)),
        "cross_supplier_rows": int(performance["cross_supplier_cabinet"].sum()),
        "cross_supplier_games": int(pair_summary["game_name"].nunique()),
        "cross_supplier_game_cabinet_pairs": int(len(pair_summary)),
    }
    (report_dir / "rebuild_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, default=str))


if __name__ == "__main__":
    main()
