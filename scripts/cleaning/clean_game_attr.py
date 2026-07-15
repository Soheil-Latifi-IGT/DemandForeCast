"""Clean the game attribute source file for modeling.

The script is intentionally runnable in two places:
- Locally, it reads paths from `.env` and uses pandas to write a CSV sample.
- On Databricks, it reads widget/job parameters plus the selected YAML and uses Spark.

Environment-specific paths and transformation policy live in YAML so the script can be
automated without changing Python code for dev/prod differences.
"""

from __future__ import annotations

import argparse
import ast
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RunConfig:
    """Fully resolved runtime settings from CLI, widgets, .env, and YAML."""

    config_env: str
    input_path: str
    output_path: str | None
    output_table: str | None
    output_format: str
    write_target: str
    write_mode: str
    game_matrix_column: str
    explicit_drop_columns: tuple[str, ...]
    impute_columns: tuple[str, ...]
    impute_groups: tuple[str, ...]
    null_strings: tuple[str, ...]


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse command-line overrides for local runs or Databricks script tasks."""

    parser = argparse.ArgumentParser(description="Clean game_attr data.")
    parser.add_argument("--config-env", help="Config name, usually dev or prod.")
    parser.add_argument("--config-dir", help="Directory containing dev.yml and prod.yml.")
    parser.add_argument("--env-file", default=".env", help="Local env file path.")
    parser.add_argument("--input-path", help="Input CSV path.")
    parser.add_argument("--output-path", help="Output path for CSV/parquet/delta writes.")
    parser.add_argument("--output-table", help="Output Databricks table for table writes.")
    parser.add_argument("--output-format", help="csv, parquet, or delta.")
    parser.add_argument("--write-target", help="path or table.")
    parser.add_argument("--write-mode", help="overwrite, append, etc.")
    return parser.parse_args(argv)


def is_databricks_runtime() -> bool:
    """Return True when Databricks sets its runtime marker environment variable."""

    return bool(os.environ.get("DATABRICKS_RUNTIME_VERSION"))


def first_non_empty(*values: Any) -> str | None:
    """Pick the first non-empty value from a precedence list."""

    for value in values:
        if value is None:
            continue
        value_text = str(value).strip()
        if value_text:
            return value_text
    return None


def strip_config_value(value: str) -> str:
    """Normalize scalar values from our small .env/YAML files."""

    return value.split("#", 1)[0].strip().strip('"').strip("'")


def load_env_file(path: str | Path) -> dict[str, str]:
    """Load KEY=VALUE pairs for local runs without requiring python-dotenv."""

    env_path = Path(path)
    if not env_path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        if key:
            values[key] = strip_config_value(value)
    return values


def load_simple_yaml(path: str | Path) -> dict[str, Any]:
    """Load the flat YAML shape used by this job.

    The project does not depend on PyYAML. This parser supports scalar keys and
    top-level list keys, which is enough for the dev/prod job configuration.
    """

    yaml_path = Path(path)
    if not yaml_path.exists():
        return {}

    values: dict[str, Any] = {}
    current_list_key: str | None = None
    for raw_line in yaml_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if line.startswith("- "):
            if current_list_key is None:
                raise ValueError(f"List item without a key in {yaml_path}: {raw_line}")
            values.setdefault(current_list_key, []).append(strip_config_value(line[2:]))
            continue

        if ":" not in line:
            raise ValueError(f"Unsupported YAML line in {yaml_path}: {raw_line}")

        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not value:
            values[key] = []
            current_list_key = key
        else:
            values[key] = strip_config_value(value)
            current_list_key = None

    return values


def required_config_value(config: dict[str, Any], key: str) -> str:
    """Read a required scalar from YAML with a clear error when it is missing."""

    value = config.get(key)
    if value is None or isinstance(value, list) or not str(value).strip():
        raise ValueError(f"Missing required YAML value: {key}")
    return str(value).strip()


def required_config_list(config: dict[str, Any], key: str) -> tuple[str, ...]:
    """Read a required list from YAML.

    A comma-delimited scalar is accepted as a convenience, but the checked-in
    YAML uses block lists because they are easier to review.
    """

    value = config.get(key)
    if isinstance(value, list):
        items = [str(item) for item in value]
    elif isinstance(value, str) and value.strip():
        items = [item.strip() for item in value.split(",")]
    else:
        raise ValueError(f"Missing required YAML list: {key}")

    return tuple(item for item in items if item is not None)


def get_spark_session() -> Any | None:
    """Return the active Spark session when Spark is available."""

    spark = globals().get("spark")
    if spark is not None:
        return spark

    try:
        from pyspark.sql import SparkSession
    except ImportError:
        return None

    return SparkSession.builder.getOrCreate()


def get_dbutils(spark: Any) -> Any | None:
    """Return Databricks dbutils when the script is running in Databricks."""

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
    """Read Databricks widget/job parameters.

    These are operational overrides only. Transformation attributes live in YAML
    so dev/prod policy changes are versioned outside the script.
    """

    if dbutils is None:
        return {}

    defaults = {
        "config_env": "",
        "config_dir": "config/game_attr_cleaning",
        "input_path": "",
        "output_path": "",
        "output_table": "",
        "output_format": "",
        "write_target": "",
        "write_mode": "",
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


def build_config(
    args: argparse.Namespace,
    db_params: dict[str, str],
    local_env: dict[str, str],
    running_on_databricks: bool,
) -> RunConfig:
    """Resolve runtime configuration.

    Precedence for paths/options is CLI > Databricks params > local .env > YAML.
    Cleaning policy is YAML-only on purpose, so schema decisions are reviewed in
    `config/game_attr_cleaning/dev.yml` and `prod.yml` instead of hidden here.
    """

    config_env = first_non_empty(
        args.config_env,
        db_params.get("config_env"),
        local_env.get("GAME_ATTR_CONFIG_ENV") if not running_on_databricks else None,
        os.environ.get("GAME_ATTR_CONFIG_ENV") if not running_on_databricks else None,
        "dev",
    )
    config_dir = first_non_empty(
        args.config_dir,
        db_params.get("config_dir"),
        local_env.get("GAME_ATTR_CONFIG_DIR") if not running_on_databricks else None,
        os.environ.get("GAME_ATTR_CONFIG_DIR") if not running_on_databricks else None,
        "config/game_attr_cleaning",
    )

    yaml_config = load_simple_yaml(Path(config_dir) / f"{config_env}.yml")
    env_config = {
        "input_path": first_non_empty(
            local_env.get("GAME_ATTR_INPUT_PATH"),
            os.environ.get("GAME_ATTR_INPUT_PATH"),
        ),
        "output_path": first_non_empty(
            local_env.get("GAME_ATTR_OUTPUT_PATH"),
            os.environ.get("GAME_ATTR_OUTPUT_PATH"),
        ),
        "output_table": first_non_empty(
            local_env.get("GAME_ATTR_OUTPUT_TABLE"),
            os.environ.get("GAME_ATTR_OUTPUT_TABLE"),
        ),
        "output_format": first_non_empty(
            local_env.get("GAME_ATTR_OUTPUT_FORMAT"),
            os.environ.get("GAME_ATTR_OUTPUT_FORMAT"),
        ),
        "write_target": first_non_empty(
            local_env.get("GAME_ATTR_WRITE_TARGET"),
            os.environ.get("GAME_ATTR_WRITE_TARGET"),
        ),
        "write_mode": first_non_empty(
            local_env.get("GAME_ATTR_WRITE_MODE"),
            os.environ.get("GAME_ATTR_WRITE_MODE"),
        ),
    }
    if running_on_databricks:
        env_config = {}

    input_path = first_non_empty(
        args.input_path,
        db_params.get("input_path"),
        env_config.get("input_path"),
        yaml_config.get("input_path"),
    )
    output_path = first_non_empty(
        args.output_path,
        db_params.get("output_path"),
        env_config.get("output_path"),
        yaml_config.get("output_path"),
    )
    output_table = first_non_empty(
        args.output_table,
        db_params.get("output_table"),
        env_config.get("output_table"),
        yaml_config.get("output_table"),
    )
    output_format = first_non_empty(
        args.output_format,
        db_params.get("output_format"),
        env_config.get("output_format"),
        yaml_config.get("output_format"),
        "delta",
    )
    write_target = first_non_empty(
        args.write_target,
        db_params.get("write_target"),
        env_config.get("write_target"),
        yaml_config.get("write_target"),
        "table",
    )
    write_mode = first_non_empty(
        args.write_mode,
        db_params.get("write_mode"),
        env_config.get("write_mode"),
        yaml_config.get("write_mode"),
        "overwrite",
    )

    if not input_path:
        raise ValueError(
            "Missing input path. Set GAME_ATTR_INPUT_PATH locally or pass input_path on Databricks."
        )

    return RunConfig(
        config_env=str(config_env),
        input_path=input_path,
        output_path=output_path,
        output_table=output_table,
        output_format=str(output_format).lower(),
        write_target=str(write_target).lower(),
        write_mode=str(write_mode).lower(),
        game_matrix_column=required_config_value(yaml_config, "game_matrix_column"),
        explicit_drop_columns=required_config_list(yaml_config, "explicit_drop_columns"),
        impute_columns=required_config_list(yaml_config, "impute_columns"),
        impute_groups=required_config_list(yaml_config, "impute_groups"),
        null_strings=required_config_list(yaml_config, "null_strings"),
    )


def safe_flag_column(prefix: str, value: str) -> str:
    """Create a stable boolean flag name from a matrix value."""

    suffix = re.sub(r"[^0-9a-zA-Z]+", "_", value.strip().lower()).strip("_")
    return f"{prefix}_{suffix or 'blank'}"


def parse_matrix_value(raw_value: Any) -> list[str]:
    """Parse the source's list-like string, for example "['Bingo', 'Lottery']"."""

    parsed = ast.literal_eval(str(raw_value))
    if not isinstance(parsed, list):
        raise ValueError(f"Expected a list-like game matrix value, got: {raw_value}")
    return [str(value) for value in parsed]


def clean_with_pandas(config: RunConfig) -> dict[str, Any]:
    """Run the same transformation locally with pandas for fast sample validation."""

    import pandas as pd

    if config.output_format != "csv":
        raise ValueError("Local pandas runs only support GAME_ATTR_OUTPUT_FORMAT=csv.")
    if not config.output_path:
        raise ValueError("Local pandas runs require GAME_ATTR_OUTPUT_PATH.")

    raw_df = pd.read_csv(
        config.input_path,
        na_values=list(config.null_strings),
        keep_default_na=True,
    )

    # Drop both configured unwanted columns and columns that are fully null in the input file.
    explicit_drops = set(config.explicit_drop_columns)
    all_null_drops = set(raw_df.columns[raw_df.isna().all()])
    drop_columns = sorted((explicit_drops | all_null_drops).intersection(raw_df.columns))
    clean_df = raw_df.drop(columns=drop_columns)

    # Expand the list-like matrix attribute into one boolean column per observed value.
    game_matrix_flags: dict[str, str] = {}
    matrix_col = config.game_matrix_column
    if matrix_col in clean_df.columns:
        values = sorted(
            {
                matrix_value
                for raw_value in clean_df[matrix_col].dropna()
                for matrix_value in parse_matrix_value(raw_value)
            }
        )
        used_columns = set(clean_df.columns)
        for matrix_value in values:
            flag_col = safe_flag_column(matrix_col, matrix_value)
            base_col = flag_col
            suffix = 2
            while flag_col in used_columns:
                flag_col = f"{base_col}_{suffix}"
                suffix += 1

            clean_df[flag_col] = clean_df[matrix_col].apply(
                lambda raw, value=matrix_value: False
                if pd.isna(raw)
                else value in parse_matrix_value(raw)
            )
            game_matrix_flags[matrix_value] = flag_col
            used_columns.add(flag_col)

    # Learn imputation modes from observed data only, then fill in configured hierarchy order.
    reference_df = clean_df.copy()
    for value_col in config.impute_columns:
        if value_col not in clean_df.columns:
            continue
        for group_col in config.impute_groups:
            if group_col not in clean_df.columns:
                continue

            counts = (
                reference_df.dropna(subset=[group_col, value_col])
                .groupby([group_col, value_col])
                .size()
                .reset_index(name="value_count")
                .sort_values(
                    [group_col, "value_count", value_col],
                    ascending=[True, False, True],
                )
            )
            mode_map = counts.drop_duplicates(group_col).set_index(group_col)[value_col]
            fill_values = clean_df[group_col].map(mode_map)
            fill_mask = clean_df[value_col].isna() & fill_values.notna()
            clean_df.loc[fill_mask, value_col] = fill_values[fill_mask]

    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(output_path, index=False)

    return {
        "engine": "pandas",
        "rows": len(clean_df),
        "columns": len(clean_df.columns),
        "dropped_columns": drop_columns,
        "game_matrix_flags": game_matrix_flags,
        "output": str(output_path),
        "remaining_target_nulls": {
            col: int(clean_df[col].isna().sum())
            for col in config.impute_columns
            if col in clean_df.columns
        },
    }


def clean_with_spark(spark: Any, config: RunConfig) -> dict[str, Any]:
    """Run the production Databricks/Spark transformation and write the result."""

    from pyspark.sql import Window
    from pyspark.sql import functions as F
    from pyspark.sql.types import ArrayType, StringType

    def normalize_string_nulls(df: Any) -> Any:
        """Convert configured string placeholders to Spark nulls in string columns."""

        projections = []
        for field in df.schema.fields:
            col = F.col(field.name)
            if isinstance(field.dataType, StringType):
                projections.append(
                    F.when(F.trim(col).isin(list(config.null_strings)), F.lit(None))
                    .otherwise(col)
                    .alias(field.name)
                )
            else:
                projections.append(col)
        return df.select(*projections)

    def fully_null_columns(df: Any) -> list[str]:
        """Return columns with no non-null values."""

        if not df.columns:
            return []

        counts_row = df.select(
            [
                F.sum(F.when(F.col(col_name).isNotNull(), F.lit(1)).otherwise(F.lit(0))).alias(
                    col_name
                )
                for col_name in df.columns
            ]
        ).first()
        return [
            col_name
            for col_name, non_null_count in counts_row.asDict().items()
            if non_null_count == 0
        ]

    def expand_game_matrix(df: Any, source_col: str) -> tuple[Any, dict[str, str]]:
        """Create boolean columns for every observed value in the matrix array."""

        if source_col not in df.columns:
            return df, {}

        parsed_col = "_game_matrix_values_raw"
        values_col = "_game_matrix_values"

        # The raw CSV stores Python-style list strings, so replace single quotes before JSON parsing.
        json_like = F.regexp_replace(F.col(source_col), "'", '"')
        with_values = df.withColumn(parsed_col, F.from_json(json_like, ArrayType(StringType())))

        bad_parse_count = (
            with_values.where(F.col(source_col).isNotNull() & F.col(parsed_col).isNull())
            .limit(1)
            .count()
        )
        if bad_parse_count:
            raise ValueError(f"Unable to parse one or more non-null {source_col} values.")

        empty_array = F.array().cast(ArrayType(StringType()))
        with_values = with_values.withColumn(values_col, F.coalesce(F.col(parsed_col), empty_array))

        matrix_values = [
            row["matrix_value"]
            for row in (
                with_values.select(F.explode_outer(F.col(values_col)).alias("matrix_value"))
                .where(F.col("matrix_value").isNotNull() & (F.length(F.trim("matrix_value")) > 0))
                .distinct()
                .orderBy("matrix_value")
                .collect()
            )
        ]

        flag_map: dict[str, str] = {}
        used_columns = set(with_values.columns)
        expanded = with_values
        for matrix_value in matrix_values:
            flag_col = safe_flag_column(source_col, matrix_value)
            base_col = flag_col
            suffix = 2
            while flag_col in used_columns:
                flag_col = f"{base_col}_{suffix}"
                suffix += 1

            expanded = expanded.withColumn(flag_col, F.array_contains(F.col(values_col), matrix_value))
            flag_map[matrix_value] = flag_col
            used_columns.add(flag_col)

        return expanded.drop(parsed_col, values_col), flag_map

    def mode_lookup(df: Any, group_col: str, value_col: str) -> Any:
        """Build one mode row per group with deterministic tie-breaking."""

        mode_col = f"__{value_col}_mode_by_{group_col}"
        rank_window = Window.partitionBy(group_col).orderBy(
            F.desc("value_count"),
            F.asc(F.col(value_col).cast("string")),
        )
        return (
            df.where(F.col(group_col).isNotNull() & F.col(value_col).isNotNull())
            .groupBy(group_col, value_col)
            .agg(F.count(F.lit(1)).alias("value_count"))
            .withColumn("__mode_rank", F.row_number().over(rank_window))
            .where(F.col("__mode_rank") == 1)
            .select(group_col, F.col(value_col).alias(mode_col))
        )

    def impute_hierarchical_modes(df: Any) -> Any:
        """Fill configured columns by each configured grouping level in order."""

        imputed = df
        reference = df
        for value_col in config.impute_columns:
            if value_col not in imputed.columns:
                continue
            for group_col in config.impute_groups:
                if group_col not in imputed.columns:
                    continue

                lookup = mode_lookup(reference, group_col, value_col)
                mode_col = f"__{value_col}_mode_by_{group_col}"
                imputed = (
                    imputed.join(F.broadcast(lookup), on=group_col, how="left")
                    .withColumn(value_col, F.coalesce(F.col(value_col), F.col(mode_col)))
                    .drop(mode_col)
                )
        return imputed

    raw_df = (
        spark.read.option("header", True)
        .option("inferSchema", True)
        .option("quote", '"')
        .option("escape", '"')
        .option("nullValue", "null")
        .csv(config.input_path)
    )
    normalized_df = normalize_string_nulls(raw_df)

    explicit_drops = set(config.explicit_drop_columns)
    all_null_drops = set(fully_null_columns(normalized_df))
    drop_columns = sorted((explicit_drops | all_null_drops).intersection(normalized_df.columns))
    clean_df = normalized_df.drop(*drop_columns)
    clean_df, game_matrix_flags = expand_game_matrix(clean_df, config.game_matrix_column)

    # Preserve the post-expansion column order because Spark joins can reorder join keys.
    final_column_order = clean_df.columns
    clean_df = impute_hierarchical_modes(clean_df).select(*final_column_order)
    clean_df.cache()

    if config.write_target == "table":
        if not config.output_table:
            raise ValueError("Databricks table writes require output_table.")

        writer = clean_df.write.mode(config.write_mode)
        if config.output_format:
            writer = writer.format(config.output_format)
        if config.output_format == "delta" and config.write_mode == "overwrite":
            writer = writer.option("overwriteSchema", True)
        writer.saveAsTable(config.output_table)
        output = config.output_table
    elif config.write_target == "path":
        if not config.output_path:
            raise ValueError("Databricks path writes require output_path.")

        writer = clean_df.write.mode(config.write_mode)
        if config.output_format == "csv":
            writer.option("header", True).csv(config.output_path)
        elif config.output_format == "parquet":
            writer.parquet(config.output_path)
        else:
            writer.format(config.output_format).option("overwriteSchema", True).save(config.output_path)
        output = config.output_path
    else:
        raise ValueError("write_target must be table or path.")

    missing_summary = clean_df.select(
        [
            F.sum(F.when(F.col(col_name).isNull(), F.lit(1)).otherwise(F.lit(0))).alias(col_name)
            for col_name in config.impute_columns
            if col_name in clean_df.columns
        ]
    ).first()

    return {
        "engine": "spark",
        "rows": clean_df.count(),
        "columns": len(clean_df.columns),
        "dropped_columns": drop_columns,
        "game_matrix_flags": game_matrix_flags,
        "output": output,
        "remaining_target_nulls": missing_summary.asDict() if missing_summary else {},
    }


def main(argv: list[str] | None = None) -> int:
    """Entrypoint used by both local Python and Databricks script tasks."""

    args = parse_args(argv or sys.argv[1:])
    spark = get_spark_session()
    dbutils = get_dbutils(spark) if spark is not None else None
    running_on_databricks = is_databricks_runtime() or dbutils is not None
    local_env = {} if running_on_databricks else load_env_file(args.env_file)
    db_params = databricks_parameters(dbutils) if running_on_databricks else {}
    config = build_config(args, db_params, local_env, running_on_databricks)

    result = (
        clean_with_spark(spark, config)
        if running_on_databricks and spark is not None
        else clean_with_pandas(config)
    )

    print(f"Engine: {result['engine']}")
    print(f"Config env: {config.config_env}")
    print(f"Rows: {result['rows']}")
    print(f"Columns: {result['columns']}")
    print(f"Dropped columns: {result['dropped_columns']}")
    print(f"Game matrix flags: {result['game_matrix_flags']}")
    print(f"Remaining target nulls: {result['remaining_target_nulls']}")
    print(f"Output: {result['output']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())