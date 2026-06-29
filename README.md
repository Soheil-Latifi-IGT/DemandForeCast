# DemandForeCast

## Game Attribute Cleaning

Local runs read paths from `.env` and use pandas to write a cleaned CSV:

```powershell
python scripts\clean_game_attr.py
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