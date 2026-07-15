# DemandForeCast

## Game Attribute Cleaning

Local runs read paths from `.env` and use pandas to write a cleaned CSV:

```powershell
python scripts\cleaning\clean_game_attr.py
```

Databricks runs use Spark. Cleaning policy lists such as drop columns, imputation columns, imputation groups, null tokens, and the matrix source column live in the selected YAML. Set `config_env` to `dev` or `prod` through a Databricks widget/job parameter, and keep environment-specific paths in:

- `config/game_attr_cleaning/dev.yml`
- `config/game_attr_cleaning/prod.yml`

Useful Databricks job parameters:

- `config_env`: `dev` or `prod`
- `config_dir`: directory containing the YAML files
- `input_path`: optional override for the YAML input path
- `output_table`: optional override for the YAML output table
- `output_path`: optional override when `write_target` is `path`
- `output_format`: `delta`, `parquet`, or `csv`
- `write_target`: `table` or `path`
- `write_mode`: usually `overwrite`

Example Databricks script task parameters:

```text
--config-env dev --config-dir config/game_attr_cleaning
```

## Config-Driven Raw Cleaning

Reusable raw-cleaning pipelines live under `src/` and are configured with YAML:

```powershell
python scripts\cleaning\run_raw_cleaning.py --config config\raw_cleaning\performance.yml
python scripts\cleaning\run_generic_raw_cleaning.py --config config\raw_cleaning\generic.yml
```

The primary cleaner supports early filters, column normalization, missing/text/numeric/date cleanup, supplier and ownership standardization, cabinet/product cleanup, strict supplier consistency reports, majority supplier fallback, hierarchical imputation, CSV/parquet output, and JSON run logs.

The generic cleaner is for secondary tables such as sales, mapping, game attributes, and ranked-title reference inputs.

## Performance Aggregation

Aggregate the cleaned Eilers performance rows to the configured monthly game/cabinet grain:

```powershell
python scripts\aggregation\run_performance_aggregation.py --config config\performance_aggregation\monthly.yml
```

The default config reads `Data/processed/performance_clean.csv` and writes `Data/processed/performance_aggregated.csv`. It uses `no_of_slots` as the weight for Eilers index metrics, sums additive unit/casino counts, keeps first stable descriptors, counts states, and applies the same excluded-game rules used by the Eilers supplier analysis.

On Databricks, run the same script with Spark by using job parameters or CLI overrides such as:

```text
--config config/performance_aggregation/monthly.yml --engine spark --input-path dbfs:/.../performance_clean --input-format delta --output-table catalog.schema.performance_aggregated --output-format delta --write-target table --write-mode overwrite
```

## Forecast Joined Data

Build the clean joined modeling table from the processed performance, mapping, game attribute, and sales files:

```powershell
python scripts\forecast\run_forecast_joined_data.py --config config\forecast_joined_data\clean.yml
```

The default config joins performance to mapping on game name plus cabinet, enriches game attributes by normalized
`theme_name` when a safe name match exists, pre-aggregates sales by `theme_key`, and writes
`Data/processed/forecast_joined_clean.csv` plus a diagnostics manifest. Attribute enrichment is a left join by
default because the current sales-backed rows do not have safe matches in `game_attr_cleaned.csv`; the manifest reports
both row retention and actual attribute match counts.

## Forecast Clustering

Cluster the joined game profiles and generate like-game recommendations for the configured target releases:

```powershell
python scripts\forecast\run_forecast_clustering.py --config config\forecast_clustering\default.yml
```

The clustering config builds one profile per game/cabinet, applies weighted one-hot and numeric features, runs
complete-linkage hierarchical clustering, walks the dendrogram to find qualified historical like games, and writes
`Data/processed/forecast_like_games.csv` plus `Data/processed/reports/forecast_clustering/dendrogram.png`.

For an interactive presentation of the same function calls, open:

```text
notebooks/forecast_clustering_demo.ipynb
```
