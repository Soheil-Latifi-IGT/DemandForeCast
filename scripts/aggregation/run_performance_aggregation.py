from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipelines.performance_aggregation import (
    aggregate_performance_spark_frame,
    load_performance_aggregation_config,
    run_performance_aggregation_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate cleaned Eilers performance data.")
    parser.add_argument(
        "--config",
        help="Path to a performance aggregation YAML config.",
    )
    parser.add_argument("--engine", choices=["auto", "pandas", "spark"], help="Execution engine.")
    parser.add_argument("--input-path", help="Override configured input path.")
    parser.add_argument("--input-format", help="csv, parquet, or delta.")
    parser.add_argument("--output-path", help="Override configured path output.")
    parser.add_argument("--output-table", help="Databricks table output for table writes.")
    parser.add_argument("--output-format", help="csv, parquet, or delta.")
    parser.add_argument("--write-target", choices=["path", "table"], help="Spark write target.")
    parser.add_argument("--write-mode", help="Spark write mode, usually overwrite or append.")
    parser.add_argument("--report-dir", help="Directory for aggregation reports.")
    return parser.parse_args()


def is_databricks_runtime() -> bool:
    return bool(os.environ.get("DATABRICKS_RUNTIME_VERSION"))


def get_spark_session() -> Any | None:
    spark = globals().get("spark")
    if spark is not None:
        return spark

    try:
        from pyspark.sql import SparkSession
    except ImportError:
        return None

    return SparkSession.builder.getOrCreate()


def get_dbutils(spark: Any) -> Any | None:
    dbutils = globals().get("dbutils")
    if dbutils is not None:
        return dbutils

    try:
        from pyspark.dbutils import DBUtils
    except ImportError:
        return None

    try:
        return DBUtils(spark)
    except Exception:
        return None


def databricks_parameters(dbutils: Any | None) -> dict[str, str]:
    if dbutils is None:
        return {}

    defaults = {
        "config": "",
        "engine": "",
        "input_path": "",
        "input_format": "",
        "output_path": "",
        "output_table": "",
        "output_format": "",
        "write_target": "",
        "write_mode": "",
        "report_dir": "",
    }
    params: dict[str, str] = {}
    for name, default in defaults.items():
        try:
            dbutils.widgets.text(name, default)
        except Exception:
            pass
        try:
            value = dbutils.widgets.get(name).strip()
        except Exception:
            value = ""
        if value:
            params[name] = value
    return params


def first_non_empty(*values: Any) -> str | None:
    for value in values:
        if value is None:
            continue
        value_text = str(value).strip()
        if value_text:
            return value_text
    return None


def merge_overrides(config: dict[str, Any], args: argparse.Namespace, params: dict[str, str]) -> dict[str, Any]:
    output = dict(config)
    for key in (
        "engine",
        "input_path",
        "input_format",
        "output_path",
        "output_table",
        "output_format",
        "write_target",
        "write_mode",
        "report_dir",
    ):
        override = first_non_empty(
            getattr(args, key.replace("-", "_"), None),
            params.get(key),
        )
        if override is not None:
            output[key] = override
    return output


def _read_spark_frame(spark: Any, config: dict[str, Any]) -> Any:
    input_format = str(config.get("input_format", "csv")).lower()
    input_path = config["input_path"]
    if input_format == "csv":
        return (
            spark.read.option("header", True)
            .option("inferSchema", True)
            .option("quote", '"')
            .option("escape", '"')
            .csv(input_path)
        )
    if input_format == "parquet":
        return spark.read.parquet(input_path)
    if input_format == "delta":
        return spark.read.format("delta").load(input_path)
    raise ValueError(f"Unsupported Spark input_format: {input_format}")


def _write_spark_frame(frame: Any, config: dict[str, Any]) -> str:
    output_format = str(config.get("output_format", "delta")).lower()
    write_target = str(config.get("write_target", "table")).lower()
    write_mode = str(config.get("write_mode", "overwrite")).lower()
    writer = frame.write.mode(write_mode)

    if write_target == "table":
        output_table = config.get("output_table")
        if not output_table:
            raise ValueError("Spark table writes require output_table.")
        if output_format:
            writer = writer.format(output_format)
        if output_format == "delta" and write_mode == "overwrite":
            writer = writer.option("overwriteSchema", True)
        writer.saveAsTable(output_table)
        return str(output_table)

    if write_target == "path":
        output_path = config.get("output_path")
        if not output_path:
            raise ValueError("Spark path writes require output_path.")
        if output_format == "csv":
            writer.option("header", True).csv(output_path)
        elif output_format == "parquet":
            writer.parquet(output_path)
        else:
            writer.format(output_format).option("overwriteSchema", True).save(output_path)
        return str(output_path)

    raise ValueError("write_target must be table or path.")


def _write_spark_reports(excluded: Any, config: dict[str, Any]) -> None:
    report_dir = config.get("report_dir")
    if not report_dir or excluded is None:
        return

    (
        excluded.write.mode("overwrite")
        .option("header", True)
        .csv(str(report_dir).rstrip("/\\") + "/excluded_games_summary")
    )


def run_with_spark(spark: Any, config: dict[str, Any]) -> dict[str, Any]:
    frame = _read_spark_frame(spark, config)
    aggregated, excluded = aggregate_performance_spark_frame(frame, config)
    output = _write_spark_frame(aggregated, config)
    _write_spark_reports(excluded, config)
    rows = aggregated.count()
    result = {
        "engine": "spark",
        "rows": int(rows),
        "columns": len(aggregated.columns),
        "output": output,
        "group_columns": config.get("group_columns"),
    }
    print(json.dumps(result, indent=2, default=str))
    return result


def main() -> int:
    args = parse_args()
    spark = get_spark_session()
    dbutils = get_dbutils(spark) if spark is not None else None
    db_params = databricks_parameters(dbutils) if is_databricks_runtime() or dbutils else {}
    config_path = first_non_empty(
        args.config,
        db_params.get("config"),
        ROOT / "config" / "performance_aggregation" / "monthly.yml",
    )
    if config_path is None:
        raise ValueError("Missing aggregation config path.")

    config, _ = load_performance_aggregation_config(config_path)
    config = merge_overrides(config, args, db_params)
    engine = str(config.get("engine", "auto")).lower()
    use_spark = engine == "spark" or (
        engine == "auto" and spark is not None and (is_databricks_runtime() or dbutils is not None)
    )

    if use_spark:
        if spark is None:
            raise ValueError("Spark engine requested but no Spark session is available.")
        run_with_spark(spark, config)
        return 0

    result = run_performance_aggregation_config(config)
    print(json.dumps({"engine": "pandas", **result.manifest}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



