from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipelines.bronze_cleaning import clean_csv_bronze


DEFAULT_INPUT = ROOT / "Data" / "raw sample" / "sales.csv"
DEFAULT_OUTPUT = ROOT / "Data" / "processed" / "sales_clean.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bronze-clean raw sales CSV headers and text values."
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help="Raw sales CSV path.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Cleaned bronze CSV output path.",
    )
    parser.add_argument(
        "--preserve-case",
        action="store_true",
        help="Trim text but do not lowercase cell values.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = clean_csv_bronze(
        args.input,
        args.output,
        lower_case_values=not args.preserve_case,
    )
    print(f"Wrote {result.output_path}")
    print(f"Rows cleaned: {result.rows_cleaned}")
    print("Columns:", ", ".join(result.columns))
