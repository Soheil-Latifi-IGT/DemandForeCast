from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipelines.raw_cleaning_engine import run_raw_cleaning


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the primary raw-cleaning pipeline.")
    parser.add_argument(
        "--config",
        default=str(ROOT / "config" / "raw_cleaning" / "performance.yml"),
        help="Path to a raw-cleaning YAML config.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cleaned = run_raw_cleaning(args.config)
    print(cleaned.shape)
    print(cleaned.head())
