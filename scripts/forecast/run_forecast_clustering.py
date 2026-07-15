from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipelines.forecast_clustering import (
    load_forecast_clustering_config,
    run_forecast_clustering_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cluster forecast game profiles and produce like-game matches."
    )
    parser.add_argument(
        "--config",
        default=str(ROOT / "config" / "forecast_clustering" / "default.yml"),
        help="Path to a forecast clustering YAML config.",
    )
    parser.add_argument("--input-path", help="Override joined-data input path.")
    parser.add_argument("--input-format", help="csv or parquet.")
    parser.add_argument("--output-format", help="csv or parquet.")
    parser.add_argument("--profiles-output-path", help="Override game profile output path.")
    parser.add_argument("--feature-matrix-output-path", help="Override feature output path.")
    parser.add_argument("--like-games-output-path", help="Override like-game output path.")
    parser.add_argument("--dendrogram-output-path", help="Override dendrogram image path.")
    parser.add_argument("--report-dir", help="Directory for clustering reports.")
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
        "input_path",
        "input_format",
        "output_format",
        "profiles_output_path",
        "feature_matrix_output_path",
        "like_games_output_path",
        "dendrogram_output_path",
        "report_dir",
    ):
        override = first_non_empty(getattr(args, key, None))
        if override is not None:
            output[key] = override
    return output


def main() -> int:
    args = parse_args()
    config, _ = load_forecast_clustering_config(args.config)
    config = merge_overrides(config, args)
    result = run_forecast_clustering_config(config)
    print(json.dumps(result.manifest, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
