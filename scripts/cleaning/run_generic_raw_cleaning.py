from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipelines.generic_raw_cleaning_engine import run_generic_raw_cleaning


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run generic raw cleaning for secondary tables.")
    parser.add_argument(
        "--config",
        default=str(ROOT / "config" / "raw_cleaning" / "generic.yml"),
        help="Path to a generic raw-cleaning YAML config.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cleaned = run_generic_raw_cleaning(args.config)
    for dataset_name, frame in cleaned.items():
        print(dataset_name, frame.shape)
        print(frame.head())
