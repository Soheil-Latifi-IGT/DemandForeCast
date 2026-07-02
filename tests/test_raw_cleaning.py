from __future__ import annotations

import pandas as pd
import pytest
import yaml

from src.data.normalize_raw import (
    apply_ownership_mapping,
    fix_missing_values,
    fix_supplier_names,
    hierarchical_impute,
    normalize_columns,
    set_majority_supplier_per_game,
)
from src.data.performance_supplier_validation import (
    build_cabinet_supplier_reference,
    validate_performance_supplier_matches,
)
from src.pipelines.generic_raw_cleaning_engine import (
    clean_generic_raw_frame,
    run_generic_raw_cleaning,
)


def test_column_normalization_detects_duplicates() -> None:
    frame = pd.DataFrame([[1, 2]], columns=["Game Name", "game-name"])

    with pytest.raises(ValueError, match="Duplicate columns"):
        normalize_columns(frame)


def test_missing_tokens_and_numeric_coercion() -> None:
    frame = pd.DataFrame(
        {
            "coin_in": ["1,200", "45%", "bad", "-"],
            "supplier": ["N/A", "IGT", "12345", "Not Provided"],
        }
    )

    out = fix_missing_values(
        frame,
        numeric_cols=["coin_in"],
        string_cols_numeric_to_nan=["supplier"],
        missing_tokens=["-", "N/A", "Not Provided"],
    )

    assert out["coin_in"].tolist()[:2] == [1200.0, 45.0]
    assert pd.isna(out.loc[2, "coin_in"])
    assert pd.isna(out.loc[3, "coin_in"])
    assert pd.isna(out.loc[0, "supplier"])
    assert out.loc[1, "supplier"] == "igt"
    assert pd.isna(out.loc[2, "supplier"])
    assert pd.isna(out.loc[3, "supplier"])


def test_mojibake_and_unicode_whitespace_cleanup() -> None:
    frame = pd.DataFrame(
        {
            "game_name": [
                "Dragon\u00c3\u201a\u00a0Link\u2007",
                "Dragon\u00c3\u0192\u00e2\u20ac\u0161\u00a0Link",
                "\ufeff  ",
            ]
        }
    )

    out = fix_missing_values(frame, string_cols_numeric_to_nan=["game_name"])

    assert out.loc[0, "game_name"] == "dragon link"
    assert out.loc[1, "game_name"] == "dragon link"
    assert pd.isna(out.loc[2, "game_name"])


def test_ownership_mapping_checks_leased_patterns_before_owned_substrings() -> None:
    frame = pd.DataFrame({"own_status": ["owned", "not owned", "daily fee", "lease cap", "mystery"]})
    config = {
        "input_col": "own_status",
        "output_col": "own_status",
        "owned_exact": ["owned", "own", "yes"],
        "leased_exact": ["leased", "lease", "not owned"],
        "owned_contains": ["own"],
        "leased_contains": ["lease", "daily", "cap"],
    }

    out = apply_ownership_mapping(frame, config)

    assert out["own_status"].tolist()[:4] == ["owned", "leased", "leased", "leased"]
    assert pd.isna(out.loc[4, "own_status"])


def test_supplier_name_mapping() -> None:
    frame = pd.DataFrame(
        {"supplier": ["Scientific Games", "Light and Wonder", "Aristocrat Gaming", None]}
    )

    out = fix_supplier_names(
        frame,
        supplier_col="supplier",
        supplier_map={
            "scientific games": "light & wonder",
            "light and wonder": "light & wonder",
            "aristocrat gaming": "aristocrat",
        },
        verbose=False,
    )

    assert out["supplier"].tolist()[:3] == ["light & wonder", "light & wonder", "aristocrat"]
    assert pd.isna(out.loc[3, "supplier"])


def test_cabinet_supplier_validation_builds_reference_and_flags_rows(tmp_path) -> None:
    sales_path = tmp_path / "sales.csv"
    pd.DataFrame(
        {
            "supplier": ["konami", "konami", "wrong"],
            "cabinet": ["dimension 49", "dimension 49", "dimension 49"],
        }
    ).to_csv(sales_path, index=False)
    performance = pd.DataFrame(
        {
            "slot_cabinet_name": ["dimension 49", "dimension 49", "unknown"] + ["legacy"] * 20,
            "supplier": ["konami", "igt", "igt"] + ["igt"] * 20,
            "game_name": ["match", "mismatch", "fallback"] + ["fallback"] * 20,
        }
    )

    reference = build_cabinet_supplier_reference(
        performance,
        validation_config={
            "external_sources": [
                {
                    "name": "sales",
                    "path": str(sales_path),
                    "supplier_col": "supplier",
                    "cabinet_col": "cabinet",
                    "minimum_share": 0.6,
                }
            ],
            "performance_minimum_share": 0.95,
            "performance_minimum_rows": 20,
        },
        supplier_map={},
        explicit_cabinet_supplier_map={},
    )
    clean, mismatch, summary = validate_performance_supplier_matches(
        performance.iloc[:3],
        reference,
        validation_config={},
    )

    assert set(reference["cabinet_key"]) == {"dimension 49", "legacy"}
    assert clean["game_name"].tolist() == ["match"]
    assert set(mismatch["supplier_validation_reason"]) == {
        "supplier_mismatch",
        "unresolved_cabinet_supplier",
    }
    assert summary["rows"].sum() == 2


def test_hierarchical_imputation_uses_group_then_global_fallback() -> None:
    frame = pd.DataFrame(
        {
            "supplier": ["igt", "igt", "konami", "konami", "everi"],
            "cabinet": ["peak", "peak", "dim", "dim", "dynasty"],
            "coin_in": [10.0, None, 30.0, None, None],
            "segment": ["core", "core", "premium", None, None],
        }
    )

    out = hierarchical_impute(
        frame,
        cols=["coin_in", "segment"],
        group_hierarchy=[["supplier", "cabinet"], ["supplier"]],
        add_missing_flags=True,
    )

    assert out.loc[1, "coin_in"] == 10.0
    assert out.loc[3, "segment"] == "premium"
    assert out.loc[4, "coin_in"] == 20.0
    assert out.loc[4, "segment"] == "core"
    assert out["coin_in_missing"].tolist() == [0, 1, 0, 1, 1]


def test_majority_supplier_per_game_uses_cabinet_mode_for_ties() -> None:
    frame = pd.DataFrame(
        {
            "game_name": [
                "tied game",
                "tied game",
                "cabinet anchor",
                "cabinet anchor",
                "other anchor",
                "other anchor",
            ],
            "slot_cabinet_name": [
                "cab igt",
                "cab konami",
                "cab igt",
                "cab igt",
                "cab konami",
                "cab konami",
            ],
            "supplier": ["igt", "konami", "igt", "igt", "konami", "konami"],
        }
    )

    out = set_majority_supplier_per_game(frame)

    tied = out.loc[out["game_name"].eq("tied game"), "supplier"].tolist()
    assert tied == ["igt", "konami"]


def test_generic_cleaner_outputs_rank_features() -> None:
    frame = pd.DataFrame(
        {
            "Data Month": ["2024-01-01", "2024-01-01", "2024-01-01"],
            "Game Name": ["game one", "game one", "game two"],
            "Supplier": ["Scientific Games", "Scientific Games", "IGT Gaming"],
            "Title": ["group a", "group b", "group c"],
            "Rank": ["2", "1", "3"],
        }
    )

    out = clean_generic_raw_frame(
        frame,
        {
            "rename_map": {
                "Data Month": "data_month",
                "Game Name": "game_name",
                "Supplier": "supplier",
                "Title": "title",
                "Rank": "rank",
            },
            "numeric_cols": ["rank"],
            "string_cols": ["game_name", "supplier", "title"],
            "supplier_map": {
                "scientific games": "light & wonder",
                "igt gaming": "igt",
            },
            "encode_topgames_title_ranks": True,
        },
    )

    game_one = out.set_index("game_name").loc["game one"]
    assert game_one["supplier"] == "light & wonder"
    assert game_one["topgames_ranked_count"] == 2
    assert game_one["topgames_title_group_count"] == 2
    assert game_one["topgames_min_rank"] == 1


def test_run_generic_cleaning_writes_configured_output(tmp_path) -> None:
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.csv"
    config_path = tmp_path / "generic.yml"
    pd.DataFrame({"Supplier": ["International Game Technology"], "Value": ["1,000"]}).to_csv(
        input_path,
        index=False,
    )
    config_path.write_text(
        yaml.safe_dump(
            {
                "shared": {
                    "supplier_map": {"international game technology": "igt"},
                    "missing_tokens": ["-"],
                },
                "datasets": [
                    {
                        "name": "sample",
                        "loader": "csv",
                        "input_path": str(input_path),
                        "output_path": str(output_path),
                        "rename_map": {"Supplier": "supplier", "Value": "value"},
                        "supplier_col": "supplier",
                        "string_cols": ["supplier"],
                        "numeric_cols": ["value"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    cleaned = run_generic_raw_cleaning(str(config_path))

    assert cleaned["sample"].loc[0, "supplier"] == "igt"
    assert cleaned["sample"].loc[0, "value"] == 1000
    assert output_path.exists()
