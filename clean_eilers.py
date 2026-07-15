"""Clean Eilers performance data and write a full Databricks silver table."""

from __future__ import annotations

import argparse
import re
from collections.abc import Mapping
from functools import reduce
from operator import or_
from pathlib import Path
from typing import Any


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


def write_clean_table(df: Any, config: Mapping[str, Any]) -> str:
    """Overwrite the configured silver table."""

    output = config.get("output") or {}
    output_table = output.get("table")
    if not output_table:
        raise ValueError("Set output.table in the YAML config.")

    output_format = output.get("format", "delta")
    mode = output.get("mode", "overwrite")
    writer = df.write.mode(mode).format(output_format)

    if output_format == "delta" and output.get("overwrite_schema", True):
        writer = writer.option("overwriteSchema", "true")

    writer.saveAsTable(output_table)
    return output_table


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
) -> dict[str, Any]:
    """Read the source table, clean it, and write the configured silver table."""

    config = load_config(config_path)
    spark = spark or get_spark_session()

    source = config.get("source") or {}
    source_table = source.get("table")
    if not source_table:
        raise ValueError("Set source.table in the YAML config.")

    df = spark.table(source_table)
    df = apply_filters(df, source.get("filters"))
    clean_df = clean_eilers_dataframe(df, config)

    diagnostics = config.get("diagnostics") or {}
    if diagnostics.get("cache_output", True):
        clean_df = clean_df.cache()

    row_count = clean_df.count()
    output_table = write_clean_table(clean_df, config) if write else None

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
        "output_table": output_table,
        "missing_summary_df": missing_summary_df,
        "ownership_counts_df": ownership_counts_df,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean Eilers performance data.")
    parser.add_argument("--config", default="config/clean_eilers.yml")
    parser.add_argument("--no-write", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = run_clean_eilers(args.config, write=not args.no_write)
    print(f"Rows: {result['row_count']:,}")
    print(f"Output table: {result['output_table']}")
