"""Clean Eilers performance data and write a full Databricks silver table."""

from __future__ import annotations

import argparse
import re
from collections.abc import Mapping
from datetime import datetime, timezone
from functools import reduce
from operator import or_
from pathlib import Path
from typing import Any
from uuid import uuid4


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load the YAML config for the Eilers cleaning job."""

    import yaml

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return config


def standardize_column_name(column_name: str, lower_case: bool = True) -> str:
    """Return the normalized column-name form used in the exploration notebook."""

    value = str(column_name).strip()
    if lower_case:
        value = value.lower()
    value = re.sub(r"\s+", "_", value)
    value = re.sub(r"[-/]+", "_", value)
    value = re.sub(r"[^\w_]", "", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_")


def normalize_columns(
    df: Any,
    rename_map: Mapping[str, str] | None = None,
    standardize: bool = True,
    lower_case: bool = True,
) -> Any:
    """Apply explicit renames, optional standardization, and duplicate checks."""

    columns = list(df.columns)
    if rename_map:
        columns = [rename_map.get(column_name, column_name) for column_name in columns]

    if standardize:
        columns = [
            standardize_column_name(column_name, lower_case=lower_case)
            for column_name in columns
        ]

    duplicates = sorted({column_name for column_name in columns if columns.count(column_name) > 1})
    if duplicates:
        raise ValueError(f"Duplicate columns after normalization: {duplicates}")

    for old_name, new_name in zip(df.columns, columns):
        if old_name != new_name:
            df = df.withColumnRenamed(old_name, new_name)

    return df


def apply_filters(df: Any, filters: Mapping[str, Any] | None = None) -> Any:
    """Apply equality filters before column normalization."""

    if not filters:
        return df

    from pyspark.sql import functions as F

    for column_name, filter_value in filters.items():
        if column_name not in df.columns:
            raise ValueError(f"Filter column not found: {column_name}")
        df = df.filter(F.col(column_name) == filter_value)
    return df


def _quote_spark_column(column_name: str) -> str:
    escaped_name = column_name.replace("`", "``")
    return f"`{escaped_name}`"


def fix_missing_values_spark(
    df: Any,
    numeric_cols: list[str] | None = None,
    typo_maps: Mapping[str, Mapping[Any, Any]] | None = None,
    missing_tokens: list[str] | None = None,
    string_cols: list[str] | None = None,
    date_cols: list[str] | None = None,
    date_format: str | None = None,
    lower_case: bool = True,
) -> Any:
    """Clean missing tokens, numeric columns, strings, dates, and typo maps."""

    from pyspark.sql import functions as F

    missing_tokens = missing_tokens or ["-", "--", "", " ", "N/A", "n/a", "NULL", "null"]
    df = df.replace(to_replace=list(missing_tokens), value=None)

    for column_name in numeric_cols or []:
        if column_name not in df.columns:
            continue

        quoted_column = _quote_spark_column(column_name)
        numeric_expression = f"""
            try_cast(
                regexp_replace(
                    regexp_replace(trim(cast({quoted_column} AS string)), ',', ''),
                    '%',
                    ''
                )
                AS double
            )
        """
        df = df.withColumn(column_name, F.expr(numeric_expression))

    numeric_string_pattern = r"^\d+(\.\d+)?$"
    normalized_missing_tokens = [
        str(token).strip().lower() if lower_case else str(token).strip()
        for token in missing_tokens
    ]

    for column_name in string_cols or []:
        if column_name not in df.columns:
            continue

        cleaned_value = F.trim(F.col(column_name).cast("string"))
        if lower_case:
            cleaned_value = F.lower(cleaned_value)

        df = df.withColumn(
            column_name,
            F.when(
                cleaned_value.rlike(numeric_string_pattern)
                | cleaned_value.isin(normalized_missing_tokens),
                F.lit(None).cast("string"),
            ).otherwise(cleaned_value),
        )

    for column_name in date_cols or []:
        if column_name not in df.columns:
            continue

        quoted_column = _quote_spark_column(column_name)
        if date_format:
            safe_date_format = date_format.replace("'", "''")
            date_expression = f"to_date(try_to_timestamp({quoted_column}, '{safe_date_format}'))"
        else:
            date_expression = f"to_date(try_cast({quoted_column} AS timestamp))"
        df = df.withColumn(column_name, F.expr(date_expression))

    for column_name, mapping in (typo_maps or {}).items():
        if column_name in df.columns:
            df = df.replace(to_replace=dict(mapping), subset=[column_name])

    return df


def fix_complementary_columns(
    df: Any,
    column_groups: Mapping[str, list[str]] | None,
    drop_fallbacks: bool = False,
) -> Any:
    """Fill primary columns from configured fallback columns."""

    if not column_groups:
        return df

    from pyspark.sql import functions as F

    for primary_col, fallback_cols in column_groups.items():
        existing_cols = [
            column_name
            for column_name in [primary_col, *(fallback_cols or [])]
            if column_name in df.columns
        ]
        if not existing_cols:
            continue

        df = df.withColumn(primary_col, F.coalesce(*[F.col(name) for name in existing_cols]))
        if drop_fallbacks:
            df = df.drop(*[name for name in fallback_cols if name in df.columns])

    return df


def fix_supplier_names_spark(
    df: Any,
    supplier_col: str,
    supplier_map: Mapping[str, str] | None,
    *,
    lower_case: bool = True,
    verbose: bool = True,
) -> Any:
    """Standardize supplier names using a configured mapping."""

    from pyspark.sql import functions as F

    if supplier_col not in df.columns:
        raise ValueError(f"Supplier column not found: {supplier_col}")

    normalized_map = {
        str(key).strip().lower() if lower_case else str(key).strip(): value
        for key, value in (supplier_map or {}).items()
    }

    original_value = F.col(supplier_col)
    cleaned_value = F.trim(original_value.cast("string"))
    if lower_case:
        cleaned_value = F.lower(cleaned_value)

    if normalized_map:
        mapping_expression = F.create_map(
            *[
                item
                for key, value in normalized_map.items()
                for item in (F.lit(key), F.lit(value))
            ]
        )
        mapped_value = F.element_at(mapping_expression, cleaned_value)
        final_value = (
            F.when(original_value.isNull(), F.lit(None).cast("string"))
            .when(cleaned_value.isin(list(normalized_map)), mapped_value)
            .otherwise(cleaned_value)
        )
    else:
        final_value = F.when(original_value.isNull(), F.lit(None).cast("string")).otherwise(
            cleaned_value
        )

    if verbose:
        changed_rows = df.filter(~original_value.eqNullSafe(final_value)).count()
        print(f"[fix_supplier_names_spark] updated {changed_rows:,} rows")

    return df.withColumn(supplier_col, final_value)


def apply_ownership_mapping(df: Any, config: Mapping[str, Any] | None = None) -> Any:
    """Map raw ownership values to owned, leased, or null."""

    if not config:
        return df

    from pyspark.sql import functions as F

    input_col = config["input_col"]
    output_col = config.get("output_col", input_col)
    if input_col not in df.columns:
        raise ValueError(f"Ownership column not found: {input_col}")

    value = F.lower(F.trim(F.col(input_col).cast("string")))

    def contains_any(values: list[str] | None) -> Any:
        checks = [value.contains(str(item).lower()) for item in values or []]
        return reduce(or_, checks, F.lit(False))

    owned_exact = [str(item).lower() for item in config.get("owned_exact", [])]
    leased_exact = [str(item).lower() for item in config.get("leased_exact", [])]

    return df.withColumn(
        output_col,
        F.when(value.isin(owned_exact), "owned")
        .when(value.isin(leased_exact), "leased")
        .when(contains_any(config.get("leased_contains", [])), "leased")
        .when(contains_any(config.get("owned_contains", [])) & ~value.contains("lease"), "owned")
        .otherwise(F.lit(None).cast("string")),
    )


def clean_eilers_dataframe(df: Any, config: Mapping[str, Any]) -> Any:
    """Apply the full cleaning flow from the exploration notebook."""

    column_config = config.get("columns", {})
    df = normalize_columns(
        df,
        rename_map=column_config.get("rename_map"),
        standardize=column_config.get("standardize", True),
        lower_case=column_config.get("lower_case", True),
    )

    df = fix_missing_values_spark(df, **(config.get("cleaning") or {}))
    df = fix_complementary_columns(
        df,
        column_groups=config.get("complementary_columns"),
        drop_fallbacks=config.get("drop_complementary_fallbacks", False),
    )

    supplier_config = config.get("supplier") or {}
    if supplier_config.get("column"):
        df = fix_supplier_names_spark(
            df,
            supplier_col=supplier_config["column"],
            supplier_map=supplier_config.get("map"),
            lower_case=supplier_config.get("lower_case", True),
            verbose=supplier_config.get("verbose", True),
        )

    return apply_ownership_mapping(df, config.get("ownership_mapping"))


def build_missing_summary(df: Any) -> Any:
    """Build the same missing summary displayed in the exploration notebook."""

    from pyspark.sql import functions as F
    from pyspark.sql.types import DoubleType, FloatType, StringType

    total_rows = df.count()
    if total_rows == 0:
        rows = [
            (field.name, field.dataType.simpleString(), 0, 0, 0)
            for field in df.schema.fields
        ]
        return df.sparkSession.createDataFrame(
            rows,
            [
                "column_name",
                "data_type",
                "missing_count",
                "missing_percent",
                "non_missing_count",
            ],
        )

    missing_expressions = []

    for field in df.schema.fields:
        column = F.col(field.name)
        missing_condition = column.isNull()
        if isinstance(field.dataType, (FloatType, DoubleType)):
            missing_condition = missing_condition | F.isnan(column)
        if isinstance(field.dataType, StringType):
            missing_condition = missing_condition | (F.trim(column) == "")

        missing_expressions.append(
            F.sum(F.when(missing_condition, 1).otherwise(0)).alias(field.name)
        )

    missing_counts = df.agg(*missing_expressions).first().asDict()
    rows = [
        (
            field.name,
            field.dataType.simpleString(),
            int(missing_counts[field.name]),
            round(missing_counts[field.name] / total_rows * 100, 2) if total_rows else 0,
            total_rows - int(missing_counts[field.name]),
        )
        for field in df.schema.fields
    ]

    return df.sparkSession.createDataFrame(
        rows,
        ["column_name", "data_type", "missing_count", "missing_percent", "non_missing_count"],
    )


def _as_list(value: Any, *, default: list[str] | None = None) -> list[str]:
    if value is None:
        return list(default or [])
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def _quote_sql_identifier(identifier: str) -> str:
    escaped = str(identifier).replace("`", "``")
    return f"`{escaped}`"


def _quote_table_name(table_name: str) -> str:
    return ".".join(_quote_sql_identifier(part) for part in str(table_name).split("."))


def _validate_data_source_format(format_name: str) -> str:
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", format_name):
        raise ValueError(f"Invalid Spark data source format: {format_name}")
    return format_name


def _output_table_name(config: Mapping[str, Any]) -> str:
    output = config.get("output") or {}
    output_table = output.get("table")
    if not output_table:
        raise ValueError("Set output.table in the YAML config.")
    return str(output_table)


def _watermark_table_name(config: Mapping[str, Any]) -> str:
    incremental = config.get("incremental") or {}
    watermark_table = incremental.get("watermark_table")
    if not watermark_table:
        raise ValueError("Set incremental.watermark_table in the YAML config.")
    return str(watermark_table)


def _incremental_job_name(config: Mapping[str, Any]) -> str:
    incremental = config.get("incremental") or {}
    return str(incremental.get("job_name") or _output_table_name(config))


def _incremental_watermark_column(config: Mapping[str, Any]) -> str:
    incremental = config.get("incremental") or {}
    return str(incremental.get("watermark_column") or "databricks_date_created")


def _table_exists(spark: Any, table_name: str) -> bool:
    try:
        if spark.catalog.tableExists(table_name):
            return True
    except Exception:
        pass

    try:
        spark.table(table_name).limit(0).collect()
    except Exception:
        return False
    return True


def _resolve_column_name(df: Any, column_name: str, label: str) -> str:
    if column_name in df.columns:
        return column_name

    matches = [name for name in df.columns if name.lower() == column_name.lower()]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(f"Ambiguous {label}: {column_name}. Matches: {matches}")
    raise ValueError(f"{label} not found: {column_name}")


def _resolve_configured_columns(df: Any, columns: list[str], label: str) -> list[str]:
    return [_resolve_column_name(df, column_name, label) for column_name in columns]


def _key_columns(df: Any, incremental: Mapping[str, Any]) -> list[str]:
    configured_keys = _as_list(
        incremental.get("key_columns", incremental.get("merge_key_columns"))
    )
    key_columns = _resolve_configured_columns(df, configured_keys, "key_columns")
    if not key_columns:
        raise ValueError(
            "Set incremental.key_columns so matched silver rows can be updated."
        )
    return key_columns


def _update_columns(
    df: Any,
    incremental: Mapping[str, Any],
    merge_keys: list[str],
) -> list[str]:
    configured_updates = _as_list(incremental.get("update_columns"))
    if configured_updates:
        return _resolve_configured_columns(df, configured_updates, "update_columns")

    merge_key_set = set(merge_keys)
    return [column_name for column_name in df.columns if column_name not in merge_key_set]


def _max_watermark_value(df: Any, watermark_column: str) -> Any | None:
    from pyspark.sql import functions as F

    row = df.select(
        F.max(F.col(watermark_column).cast("timestamp")).alias("watermark_value")
    ).first()
    if row is None:
        return None
    return row["watermark_value"]


def _filter_after_watermark(df: Any, watermark_column: str, watermark_value: Any) -> Any:
    from pyspark.sql import functions as F

    return df.filter(
        F.col(watermark_column).cast("timestamp")
        > F.lit(watermark_value).cast("timestamp")
    )


def _deduplicate_merge_rows(
    df: Any,
    merge_keys: list[str],
    watermark_column: str,
) -> Any:
    from pyspark.sql import functions as F
    from pyspark.sql.window import Window

    rank_column = "__clean_eilers_merge_rank"
    while rank_column in df.columns:
        rank_column = f"_{rank_column}"

    window = Window.partitionBy(*[F.col(column_name) for column_name in merge_keys]).orderBy(
        F.col(watermark_column).cast("timestamp").desc_nulls_last()
    )
    return (
        df.withColumn(rank_column, F.row_number().over(window))
        .filter(F.col(rank_column) == 1)
        .drop(rank_column)
    )


def ensure_watermark_table(spark: Any, config: Mapping[str, Any]) -> str:
    """Create the incremental watermark table when it is missing."""

    output = config.get("output") or {}
    incremental = config.get("incremental") or {}
    watermark_table = _watermark_table_name(config)
    table_format = _validate_data_source_format(
        str(incremental.get("watermark_format") or output.get("format") or "delta")
    )

    spark.sql(
        f"""
        CREATE TABLE IF NOT EXISTS {_quote_table_name(watermark_table)} (
            job_name STRING,
            watermark_column STRING,
            watermark_value TIMESTAMP,
            source_table STRING,
            target_table STRING,
            rows_processed BIGINT,
            operation STRING,
            updated_at TIMESTAMP
        )
        USING {table_format}
        """
    )
    return watermark_table


def read_latest_watermark(spark: Any, config: Mapping[str, Any]) -> Any | None:
    """Return the latest saved watermark for this cleaning job."""

    from pyspark.sql import functions as F

    watermark_table = _watermark_table_name(config)
    if not _table_exists(spark, watermark_table):
        return None

    job_name = _incremental_job_name(config)
    watermark_column = _incremental_watermark_column(config)
    rows = (
        spark.table(watermark_table)
        .filter(
            (F.col("job_name") == job_name)
            & (F.col("watermark_column") == watermark_column)
        )
        .orderBy(F.col("updated_at").desc_nulls_last())
        .select("watermark_value")
        .limit(1)
        .collect()
    )
    if not rows:
        return None
    return rows[0]["watermark_value"]


def append_watermark(
    spark: Any,
    config: Mapping[str, Any],
    *,
    source_table: str,
    rows_processed: int,
    operation: str,
    watermark_value: Any | None,
) -> None:
    """Append a successful-run watermark record."""

    if watermark_value is None:
        return

    from pyspark.sql.types import (
        LongType,
        StringType,
        StructField,
        StructType,
        TimestampType,
    )

    watermark_table = _watermark_table_name(config)
    schema = StructType(
        [
            StructField("job_name", StringType(), False),
            StructField("watermark_column", StringType(), False),
            StructField("watermark_value", TimestampType(), True),
            StructField("source_table", StringType(), True),
            StructField("target_table", StringType(), True),
            StructField("rows_processed", LongType(), True),
            StructField("operation", StringType(), True),
            StructField("updated_at", TimestampType(), False),
        ]
    )
    row = [
        (
            _incremental_job_name(config),
            _incremental_watermark_column(config),
            watermark_value,
            source_table,
            _output_table_name(config),
            int(rows_processed),
            operation,
            datetime.now(timezone.utc).replace(tzinfo=None),
        )
    ]
    spark.createDataFrame(row, schema).write.mode("append").saveAsTable(watermark_table)


def write_clean_table(
    df: Any,
    config: Mapping[str, Any],
    *,
    mode: str | None = None,
) -> str:
    """Write the cleaned data as a full-refresh silver table."""

    output = config.get("output") or {}
    output_table = _output_table_name(config)
    output_format = output.get("format", "delta")
    write_mode = mode or output.get("mode", "overwrite")
    if write_mode == "merge":
        write_mode = "overwrite"

    writer = df.write.mode(write_mode).format(output_format)
    if output_format == "delta" and output.get("overwrite_schema", True):
        writer = writer.option("overwriteSchema", "true")

    writer.saveAsTable(output_table)
    return output_table


def merge_clean_table(
    df: Any,
    config: Mapping[str, Any],
    *,
    watermark_column: str,
) -> dict[str, Any]:
    """Merge changed cleaned rows into the configured silver table."""

    spark = df.sparkSession
    incremental = config.get("incremental") or {}
    output_table = _output_table_name(config)
    merge_keys = _key_columns(df, incremental)
    deduped_df = _deduplicate_merge_rows(df, merge_keys, watermark_column)
    update_columns = _update_columns(deduped_df, incremental, merge_keys)

    source_view = f"clean_eilers_merge_{uuid4().hex}"
    deduped_df.createOrReplaceTempView(source_view)

    on_condition = " AND ".join(
        f"target.{_quote_sql_identifier(column_name)} "
        f"<=> source.{_quote_sql_identifier(column_name)}"
        for column_name in merge_keys
    )
    insert_columns = ", ".join(
        _quote_sql_identifier(column_name) for column_name in deduped_df.columns
    )
    insert_values = ", ".join(
        f"source.{_quote_sql_identifier(column_name)}" for column_name in deduped_df.columns
    )

    matched_clause = ""
    if update_columns:
        update_assignments = ", ".join(
            f"target.{_quote_sql_identifier(column_name)} = "
            f"source.{_quote_sql_identifier(column_name)}"
            for column_name in update_columns
        )
        matched_clause = f"WHEN MATCHED THEN UPDATE SET {update_assignments}"

    merge_sql = f"""
        MERGE INTO {_quote_table_name(output_table)} AS target
        USING {_quote_sql_identifier(source_view)} AS source
        ON {on_condition}
        {matched_clause}
        WHEN NOT MATCHED THEN INSERT ({insert_columns})
        VALUES ({insert_values})
    """

    try:
        spark.sql(merge_sql)
    finally:
        spark.catalog.dropTempView(source_view)

    return {
        "table": output_table,
        "operation": "merge",
        "key_columns": merge_keys,
        "update_columns": update_columns,
    }


def write_incremental_clean_table(
    df: Any,
    config: Mapping[str, Any],
    *,
    source_table: str,
    rows_processed: int,
    max_watermark_value: Any | None,
    full_refresh: bool = False,
) -> dict[str, Any]:
    """Create or incrementally merge the cleaned silver table."""

    spark = df.sparkSession
    incremental = config.get("incremental") or {}
    output_table = _output_table_name(config)
    watermark_table = ensure_watermark_table(spark, config)
    target_exists = _table_exists(spark, output_table)
    clean_watermark_column = _resolve_column_name(
        df,
        _incremental_watermark_column(config),
        "cleaned watermark_column",
    )
    merge_keys = _key_columns(df, incremental)
    deduped_df = _deduplicate_merge_rows(df, merge_keys, clean_watermark_column)

    if full_refresh or not target_exists:
        operation = "full_refresh" if target_exists else "create"
        write_clean_table(deduped_df, config, mode="overwrite")
    elif rows_processed == 0:
        operation = "skip"
    else:
        merge_result = merge_clean_table(
            df,
            config,
            watermark_column=clean_watermark_column,
        )
        operation = merge_result["operation"]

    if operation != "skip":
        append_watermark(
            spark,
            config,
            source_table=source_table,
            rows_processed=rows_processed,
            operation=operation,
            watermark_value=max_watermark_value,
        )

    return {
        "table": output_table,
        "operation": operation,
        "watermark_table": watermark_table,
        "new_watermark": max_watermark_value,
        "key_columns": merge_keys,
    }


def get_spark_session() -> Any:
    """Return the active Spark session."""

    try:
        spark = globals().get("spark")
        if spark is not None:
            return spark
    except NameError:
        pass

    from pyspark.sql import SparkSession

    return SparkSession.builder.getOrCreate()


def run_clean_eilers(
    config_path: str | Path = "config/clean_eilers.yml",
    *,
    spark: Any | None = None,
    write: bool = True,
    full_refresh: bool = False,
) -> dict[str, Any]:
    """Read the source table, clean it, and write or merge the silver table."""

    config = load_config(config_path)
    spark = spark or get_spark_session()

    source = config.get("source") or {}
    source_table = source.get("table")
    if not source_table:
        raise ValueError("Set source.table in the YAML config.")
    source_table = str(source_table)

    output_table = _output_table_name(config)
    incremental = config.get("incremental") or {}
    incremental_enabled = bool(incremental.get("enabled", False))
    target_exists = _table_exists(spark, output_table)

    df = spark.table(source_table)
    df = apply_filters(df, source.get("filters"))

    previous_watermark = None
    max_watermark = None
    watermark_table = incremental.get("watermark_table")
    if incremental_enabled:
        source_watermark_column = _resolve_column_name(
            df,
            _incremental_watermark_column(config),
            "source watermark_column",
        )
        if write:
            watermark_table = ensure_watermark_table(spark, config)
        watermark_exists = watermark_table and _table_exists(spark, str(watermark_table))
        if not full_refresh and target_exists and watermark_exists:
            previous_watermark = read_latest_watermark(spark, config)
            if previous_watermark is not None:
                df = _filter_after_watermark(df, source_watermark_column, previous_watermark)
        max_watermark = _max_watermark_value(df, source_watermark_column)

    clean_df = clean_eilers_dataframe(df, config)

    diagnostics = config.get("diagnostics") or {}
    if diagnostics.get("cache_output", True):
        clean_df = clean_df.cache()

    row_count = clean_df.count()
    write_result: dict[str, Any] = {"operation": "no_write", "table": None}
    if write:
        if incremental_enabled:
            write_result = write_incremental_clean_table(
                clean_df,
                config,
                source_table=source_table,
                rows_processed=row_count,
                max_watermark_value=max_watermark,
                full_refresh=full_refresh,
            )
        else:
            output_table = write_clean_table(clean_df, config)
            write_result = {"operation": "full_refresh", "table": output_table}

    missing_summary_df = (
        build_missing_summary(clean_df) if diagnostics.get("missing_summary", True) else None
    )

    ownership_counts_df = None
    ownership_column = diagnostics.get("ownership_counts_column")
    if ownership_column and ownership_column in clean_df.columns:
        from pyspark.sql import functions as F

        ownership_counts_df = clean_df.groupBy(ownership_column).count().orderBy(F.desc("count"))

    return {
        "config": config,
        "clean_df": clean_df,
        "row_count": row_count,
        "output_table": write_result.get("table"),
        "write_operation": write_result.get("operation"),
        "watermark_table": write_result.get("watermark_table") or watermark_table,
        "previous_watermark": previous_watermark,
        "new_watermark": write_result.get("new_watermark", max_watermark),
        "key_columns": write_result.get("key_columns"),
        "missing_summary_df": missing_summary_df,
        "ownership_counts_df": ownership_counts_df,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean Eilers performance data.")
    parser.add_argument("--config", default="config/clean_eilers.yml")
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--full-refresh", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = run_clean_eilers(
        args.config,
        write=not args.no_write,
        full_refresh=args.full_refresh,
    )
    print(f"Rows: {result['row_count']:,}")
    print(f"Operation: {result['write_operation']}")
    print(f"Output table: {result['output_table']}")
    print(f"Watermark table: {result['watermark_table']}")
    print(f"Previous watermark: {result['previous_watermark']}")
    print(f"New watermark: {result['new_watermark']}")
