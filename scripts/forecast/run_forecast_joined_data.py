from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipelines.forecast_joined_data import (
    load_forecast_joined_data_config,
    run_forecast_joined_data_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the clean joined forecast modeling dataset."
    )
    parser.add_argument(
        "--config",
        default=str(ROOT / "config" / "forecast_joined_data" / "clean.yml"),
        help="Path to a forecast joined-data YAML config.",
    )
    parser.add_argument("--performance-path", help="Override performance input path.")
    parser.add_argument("--mapping-path", help="Override mapping input path.")
    parser.add_argument("--game-attr-path", help="Override game attribute input path.")
    parser.add_argument("--sales-path", help="Override sales input path.")
    parser.add_argument("--input-format", help="csv or parquet.")
    parser.add_argument("--output-path", help="Override joined output path.")
    parser.add_argument("--sales-aggregated-output-path", help="Override sales aggregate path.")
    parser.add_argument("--output-format", help="csv or parquet.")
    parser.add_argument("--report-dir", help="Directory for manifest reports.")
    return parser.parse_args()


def first_non_empty(*values: Any) -> str | None:
    for value in values:
        if value is None:
            continue
        value_text = str(value).strip()
        if value_text:
            return value_text
    return None


def merge_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    output = dict(config)
    for key in (
        "performance_path",
        "mapping_path",
        "game_attr_path",
        "sales_path",
        "input_format",
        "output_path",
        "sales_aggregated_output_path",
        "output_format",
        "report_dir",
    ):
        override = first_non_empty(getattr(args, key, None))
        if override is not None:
            output[key] = override
    return output


def main() -> int:
    args = parse_args()
    config, _ = load_forecast_joined_data_config(args.config)
    config = merge_overrides(config, args)
    result = run_forecast_joined_data_config(config)
    print(json.dumps(result.manifest, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
