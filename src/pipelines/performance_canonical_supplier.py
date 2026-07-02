from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.normalize_raw import clean_cabinet_names, fix_supplier_names


EXCLUDED_GAME_NAME_PATTERN = (
    r"multigame|multi[\s-]?game|roulette|roullete|vlt"
)
VLT_COLUMNS = (
    "game_classification",
    "game_category",
    "operator_type",
    "cabinet_type",
)


def _normalize_supplier(
    values: pd.Series,
    supplier_map: dict[str, str],
) -> pd.Series:
    frame = pd.DataFrame(
        {
            "supplier": (
                values.astype("string").str.strip().str.lower()
            )
        }
    )
    frame = fix_supplier_names(
        frame,
        supplier_col="supplier",
        supplier_map=supplier_map,
        verbose=False,
    )
    return frame["supplier"].replace(
        {
            "": pd.NA,
            "-": pd.NA,
            "nan": pd.NA,
            "none": pd.NA,
            "unknown": pd.NA,
            "other": pd.NA,
        }
    )


def filter_excluded_games(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    game_name = frame["game_name"].astype("string")
    missing_game = game_name.isna()
    excluded_name = game_name.str.contains(
        EXCLUDED_GAME_NAME_PATTERN,
        case=False,
        regex=True,
        na=False,
    )
    vlt_metadata = pd.Series(False, index=frame.index)
    for column in VLT_COLUMNS:
        if column in frame.columns:
            vlt_metadata |= (
                frame[column]
                .astype("string")
                .str.contains("vlt", case=False, regex=False, na=False)
            )

    reason = pd.Series(pd.NA, index=frame.index, dtype="string")
    reason.loc[missing_game] = "missing_game_name"
    reason.loc[excluded_name] = "excluded_game_name"
    reason.loc[vlt_metadata] = "vlt_metadata"
    excluded_mask = reason.notna()

    excluded = (
        pd.DataFrame(
            {
                "exclusion_reason": reason.loc[excluded_mask],
                "game_name": game_name.loc[excluded_mask],
            }
        )
        .groupby(
            ["exclusion_reason", "game_name"],
            dropna=False,
            as_index=False,
        )
        .size()
        .rename(columns={"size": "rows"})
        .sort_values(["rows", "game_name"], ascending=[False, True])
    )
    return frame.loc[~excluded_mask].copy(), excluded


def _normalize_sales(
    path: Path,
    *,
    corrected_layout: bool,
    supplier_map: dict[str, str],
    source_name: str,
) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        encoding="utf-16",
        sep="\t",
        low_memory=False,
    )
    if corrected_layout:
        rename = {
            "Unnamed: 0": "data_month",
            "Unnamed: 1": "supplier",
            "Unnamed: 2": "cabinet_name",
            "Sold Cabinets": "sold_cabinets",
        }
    else:
        rename = {
            "Unnamed: 0": "data_month",
            "Unnamed: 1": "cabinet_name",
            "Unnamed: 2": "supplier",
            "Sold Cabinets": "sold_cabinets",
        }
    frame = frame.rename(columns=rename)
    frame = frame[
        ["data_month", "supplier", "cabinet_name", "sold_cabinets"]
    ].copy()
    frame["month"] = pd.to_datetime(
        frame["data_month"],
        format="%B %Y",
        errors="coerce",
    )
    frame = frame.loc[frame["month"].ge("2023-01-01")].copy()
    frame["supplier"] = _normalize_supplier(
        frame["supplier"],
        supplier_map,
    )
    frame["cabinet_name"] = (
        frame["cabinet_name"].astype("string").str.strip().str.lower()
    )
    frame = clean_cabinet_names(frame, cabinet_col="cabinet_name")
    frame["sold_cabinets"] = pd.to_numeric(
        frame["sold_cabinets"]
        .astype("string")
        .str.replace(",", "", regex=False),
        errors="coerce",
    ).fillna(0)
    frame["reference_source"] = source_name
    return frame.dropna(subset=["supplier", "cabinet_name"])


def _dominant_sales_supplier(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        frame.groupby(
            ["cabinet_name", "supplier", "reference_source"],
            as_index=False,
        )
        .agg(
            supplier_sold_cabinets=("sold_cabinets", "sum"),
            sales_rows=("sold_cabinets", "size"),
            sales_month_start=("month", "min"),
            sales_month_end=("month", "max"),
        )
    )
    grouped["cabinet_total_sold_cabinets"] = grouped.groupby(
        ["cabinet_name", "reference_source"]
    )["supplier_sold_cabinets"].transform("sum")
    grouped["reference_confidence"] = (
        grouped["supplier_sold_cabinets"]
        / grouped["cabinet_total_sold_cabinets"].replace(0, pd.NA)
    )
    grouped["candidate_suppliers"] = grouped.groupby(
        ["cabinet_name", "reference_source"]
    )["supplier"].transform("nunique")
    return (
        grouped.sort_values(
            [
                "cabinet_name",
                "supplier_sold_cabinets",
                "sales_rows",
                "supplier",
            ],
            ascending=[True, False, False, True],
        )
        .drop_duplicates(["cabinet_name", "reference_source"])
        .rename(columns={"supplier": "cabinet_supplier"})
    )


def build_sales_cabinet_reference(
    *,
    corrected_sales_path: Path,
    original_sales_path: Path,
    supplier_map: dict[str, str],
) -> pd.DataFrame:
    corrected = _dominant_sales_supplier(
        _normalize_sales(
            corrected_sales_path,
            corrected_layout=True,
            supplier_map=supplier_map,
            source_name="corrected_cabinet_sales",
        )
    )
    original = _dominant_sales_supplier(
        _normalize_sales(
            original_sales_path,
            corrected_layout=False,
            supplier_map=supplier_map,
            source_name="original_cabinet_sales",
        )
    )
    original_fallback = original.loc[
        ~original["cabinet_name"].isin(corrected["cabinet_name"])
    ]
    return (
        pd.concat([corrected, original_fallback], ignore_index=True)
        .sort_values("cabinet_name")
        .reset_index(drop=True)
    )


def _latest_and_peak(
    monthly: pd.DataFrame,
    group_columns: list[str],
) -> pd.DataFrame:
    latest = (
        monthly.sort_values("yearmonth")
        .groupby(group_columns, as_index=False)
        .tail(1)
        .rename(
            columns={
                "yearmonth": "latest_performance_month",
                "owned_slots": "latest_owned_slots",
            }
        )
    )
    peak = (
        monthly.groupby(group_columns, as_index=False)["owned_slots"]
        .max()
        .rename(columns={"owned_slots": "peak_owned_slots"})
    )
    return latest.merge(peak, on=group_columns, how="left")


def build_canonical_supplier_map(
    frame: pd.DataFrame,
    sales_reference: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = [
        "game_name",
        "supplier",
        "slot_cabinet_name",
        "yearmonth",
        "no_of_slots",
    ]
    working = frame[columns].dropna(
        subset=["game_name", "supplier", "slot_cabinet_name"]
    ).copy()
    working["yearmonth"] = (
        pd.to_datetime(working["yearmonth"], errors="coerce")
        .dt.to_period("M")
        .dt.to_timestamp()
    )
    working["no_of_slots"] = pd.to_numeric(
        working["no_of_slots"],
        errors="coerce",
    ).fillna(0)

    observed = (
        working.groupby(["game_name", "supplier"], as_index=False)
        .agg(
            performance_rows=("no_of_slots", "size"),
            unit_months=("no_of_slots", "sum"),
        )
    )
    observed["observed_supplier_count"] = observed.groupby(
        "game_name"
    )["supplier"].transform("nunique")
    observed_names = (
        observed.groupby("game_name", as_index=False)
        .agg(
            observed_supplier_count=("supplier", "nunique"),
            observed_suppliers=(
                "supplier",
                lambda values: " | ".join(sorted(set(values))),
            ),
            performance_rows=("performance_rows", "sum"),
        )
    )

    monthly_cabinet = (
        working.groupby(
            ["game_name", "slot_cabinet_name", "yearmonth"],
            as_index=False,
        )["no_of_slots"]
        .sum()
        .rename(columns={"no_of_slots": "owned_slots"})
    )
    cabinet_metrics = _latest_and_peak(
        monthly_cabinet,
        ["game_name", "slot_cabinet_name"],
    )
    cabinet_rows = (
        working.groupby(
            ["game_name", "slot_cabinet_name"],
            as_index=False,
        )
        .size()
        .rename(columns={"size": "cabinet_performance_rows"})
    )
    cabinet_metrics = cabinet_metrics.merge(
        cabinet_rows,
        on=["game_name", "slot_cabinet_name"],
        how="left",
    )
    sales = sales_reference.rename(
        columns={"cabinet_name": "slot_cabinet_name"}
    )
    cabinet_metrics = cabinet_metrics.merge(
        sales,
        on="slot_cabinet_name",
        how="left",
        validate="many_to_one",
    )
    cabinet_metrics = cabinet_metrics.sort_values(
        [
            "game_name",
            "latest_owned_slots",
            "peak_owned_slots",
            "cabinet_performance_rows",
            "slot_cabinet_name",
        ],
        ascending=[True, False, False, False, True],
    )
    major = cabinet_metrics.drop_duplicates("game_name").copy()
    mapped = (
        cabinet_metrics.loc[cabinet_metrics["cabinet_supplier"].notna()]
        .drop_duplicates("game_name")
        .copy()
    )

    supplier_monthly = (
        working.groupby(
            ["game_name", "supplier", "yearmonth"],
            as_index=False,
        )["no_of_slots"]
        .sum()
        .rename(columns={"no_of_slots": "owned_slots"})
    )
    supplier_metrics = _latest_and_peak(
        supplier_monthly,
        ["game_name", "supplier"],
    )
    supplier_metrics = supplier_metrics.merge(
        observed[
            ["game_name", "supplier", "performance_rows", "unit_months"]
        ],
        on=["game_name", "supplier"],
        how="left",
    )
    supplier_fallback = (
        supplier_metrics.sort_values(
            [
                "game_name",
                "latest_owned_slots",
                "peak_owned_slots",
                "performance_rows",
                "supplier",
            ],
            ascending=[True, False, False, False, True],
        )
        .drop_duplicates("game_name")
        .rename(columns={"supplier": "fallback_observed_supplier"})
    )

    major_columns = {
        "slot_cabinet_name": "major_cabinet",
        "latest_performance_month": "major_cabinet_latest_month",
        "latest_owned_slots": "major_cabinet_latest_owned_slots",
        "peak_owned_slots": "major_cabinet_peak_owned_slots",
        "cabinet_supplier": "major_cabinet_sales_supplier",
        "reference_source": "major_cabinet_sales_source",
        "reference_confidence": "major_cabinet_supplier_confidence",
    }
    mapped_columns = {
        "slot_cabinet_name": "selected_sales_mapped_cabinet",
        "cabinet_supplier": "selected_sales_supplier",
        "reference_source": "selected_sales_source",
        "reference_confidence": "selected_sales_confidence",
        "latest_owned_slots": "selected_cabinet_latest_owned_slots",
    }
    canonical = observed_names.merge(
        major[
            ["game_name"] + list(major_columns)
        ].rename(columns=major_columns),
        on="game_name",
        how="left",
    )
    canonical = canonical.merge(
        mapped[
            ["game_name"] + list(mapped_columns)
        ].rename(columns=mapped_columns),
        on="game_name",
        how="left",
    )
    canonical = canonical.merge(
        supplier_fallback[
            ["game_name", "fallback_observed_supplier"]
        ],
        on="game_name",
        how="left",
    )

    single_supplier = canonical["observed_supplier_count"].eq(1)
    canonical["canonical_game_supplier"] = canonical[
        "selected_sales_supplier"
    ].where(~single_supplier, canonical["fallback_observed_supplier"])
    canonical["canonical_game_supplier"] = canonical[
        "canonical_game_supplier"
    ].fillna(canonical["fallback_observed_supplier"])
    canonical["canonical_supplier_basis"] = np.select(
        [
            single_supplier,
            canonical["selected_sales_supplier"].notna()
            & canonical["selected_sales_mapped_cabinet"].eq(
                canonical["major_cabinet"]
            ),
            canonical["selected_sales_supplier"].notna(),
        ],
        [
            "single_observed_supplier",
            "major_cabinet_sales_manufacturer",
            "largest_sales_mapped_cabinet_manufacturer",
        ],
        default="dominant_observed_supplier_no_sales_mapping",
    )
    canonical["canonical_supplier_confidence"] = np.where(
        single_supplier,
        1.0,
        canonical["selected_sales_confidence"],
    )
    canonical["canonical_supplier_status"] = np.where(
        canonical["canonical_supplier_basis"].eq(
            "dominant_observed_supplier_no_sales_mapping"
        ),
        "sales_mapping_unresolved_fallback_used",
        "resolved",
    )
    return canonical.sort_values("game_name"), cabinet_metrics


def apply_canonical_suppliers(
    frame: pd.DataFrame,
    canonical_map: pd.DataFrame,
    sales_reference: pd.DataFrame,
) -> pd.DataFrame:
    output = frame
    output["slot_cabinet_name"] = (
        output["slot_cabinet_name"]
        .astype("string")
        .str.strip()
        .str.lower()
    )
    output = clean_cabinet_names(
        output,
        cabinet_col="slot_cabinet_name",
    )
    output["reported_game_supplier"] = output["supplier"]

    canonical_index = canonical_map.set_index("game_name")
    for column in (
        "canonical_game_supplier",
        "canonical_supplier_basis",
        "canonical_supplier_confidence",
        "canonical_supplier_status",
        "observed_supplier_count",
        "observed_suppliers",
    ):
        output[column] = output["game_name"].map(canonical_index[column])
    output["game_supplier"] = output["canonical_game_supplier"]
    output["supplier"] = output["canonical_game_supplier"]

    sales_index = sales_reference.set_index("cabinet_name")
    output["cabinet_supplier"] = output["slot_cabinet_name"].map(
        sales_index["cabinet_supplier"]
    )
    output["reference_source"] = output["slot_cabinet_name"].map(
        sales_index["reference_source"]
    )
    output["reference_confidence"] = output["slot_cabinet_name"].map(
        sales_index["reference_confidence"]
    )
    output["supplier_rows"] = output["slot_cabinet_name"].map(
        sales_index["sales_rows"]
    )
    output["candidate_suppliers"] = output["slot_cabinet_name"].map(
        sales_index["candidate_suppliers"]
    )
    comparable = (
        output["game_supplier"].notna()
        & output["cabinet_supplier"].notna()
    )
    output["supplier_match"] = (
        comparable
        & output["game_supplier"].eq(output["cabinet_supplier"])
    )
    output["cross_supplier_cabinet"] = (
        comparable & ~output["supplier_match"]
    )
    output["supplier_validation_reason"] = np.select(
        [
            output["game_supplier"].isna(),
            output["cabinet_supplier"].isna(),
            output["cross_supplier_cabinet"],
        ],
        [
            "missing_canonical_game_supplier",
            "unresolved_cabinet_supplier",
            "cross_supplier_cabinet",
        ],
        default="matched",
    )
    return output


def _month_span(start: pd.Series, end: pd.Series) -> pd.Series:
    return (
        (end.dt.year - start.dt.year) * 12
        + (end.dt.month - start.dt.month)
        + 1
    )


def build_cross_supplier_reports(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cross = frame.loc[frame["cross_supplier_cabinet"]].copy()
    cross["yearmonth"] = (
        pd.to_datetime(cross["yearmonth"], errors="coerce")
        .dt.to_period("M")
        .dt.to_timestamp()
    )
    cross["no_of_slots"] = pd.to_numeric(
        cross["no_of_slots"],
        errors="coerce",
    ).fillna(0)
    group_columns = [
        "game_name",
        "game_supplier",
        "slot_cabinet_name",
        "cabinet_supplier",
        "reference_source",
    ]
    monthly = (
        cross.groupby(group_columns + ["yearmonth"], as_index=False)
        .agg(
            owned_units=("no_of_slots", "sum"),
            performance_rows=("no_of_slots", "size"),
            states=("state", "nunique"),
        )
        .sort_values(group_columns + ["yearmonth"])
    )
    pair = (
        monthly.groupby(group_columns, as_index=False)
        .agg(
            first_observed_month=("yearmonth", "min"),
            latest_observed_month=("yearmonth", "max"),
            months_observed_on_cabinet=("yearmonth", "nunique"),
            peak_owned_units=("owned_units", "max"),
            mean_monthly_owned_units=("owned_units", "mean"),
            total_unit_months=("owned_units", "sum"),
            performance_rows=("performance_rows", "sum"),
            states_observed=("states", "max"),
        )
    )
    latest = (
        monthly.groupby(group_columns, as_index=False)
        .tail(1)[group_columns + ["yearmonth", "owned_units"]]
        .rename(
            columns={
                "yearmonth": "latest_units_month",
                "owned_units": "latest_owned_units",
            }
        )
    )
    pair = pair.merge(latest, on=group_columns, how="left")
    pair["cabinet_calendar_span_months"] = _month_span(
        pair["first_observed_month"],
        pair["latest_observed_month"],
    )

    game_dates = (
        frame.groupby(["game_name", "game_supplier"], as_index=False)
        .agg(
            game_first_observed_month=("yearmonth", "min"),
            game_latest_observed_month=("yearmonth", "max"),
            game_release_date=("game_release_date", "min"),
        )
    )
    for column in (
        "game_first_observed_month",
        "game_latest_observed_month",
        "game_release_date",
    ):
        game_dates[column] = (
            pd.to_datetime(game_dates[column], errors="coerce")
            .dt.to_period("M")
            .dt.to_timestamp()
        )
    valid_release = (
        game_dates["game_release_date"].notna()
        & game_dates["game_release_date"].le(
            game_dates["game_latest_observed_month"]
        )
    )
    game_dates["market_start_month"] = (
        game_dates["game_release_date"]
        .where(valid_release)
        .fillna(game_dates["game_first_observed_month"])
    )
    game_dates["market_age_basis"] = np.where(
        valid_release,
        "game_release_date",
        "first_observed_month",
    )
    game_dates["months_on_market"] = _month_span(
        game_dates["market_start_month"],
        game_dates["game_latest_observed_month"],
    )
    pair = pair.merge(
        game_dates,
        on=["game_name", "game_supplier"],
        how="left",
    )
    pair["months_from_market_start_to_cabinet"] = _month_span(
        pair["market_start_month"],
        pair["first_observed_month"],
    )
    pair = pair.sort_values(
        ["latest_owned_units", "months_observed_on_cabinet"],
        ascending=False,
    )

    game_summary = (
        pair.groupby(["game_name", "game_supplier"], as_index=False)
        .agg(
            other_supplier_cabinet_count=(
                "slot_cabinet_name",
                "nunique",
            ),
            other_cabinet_supplier_count=(
                "cabinet_supplier",
                "nunique",
            ),
            other_cabinet_suppliers=(
                "cabinet_supplier",
                lambda values: " | ".join(sorted(set(values))),
            ),
            first_cross_supplier_month=("first_observed_month", "min"),
            latest_cross_supplier_month=("latest_observed_month", "max"),
            longest_other_cabinet_tenure_months=(
                "months_observed_on_cabinet",
                "max",
            ),
            latest_owned_units_on_other_cabinets=(
                "latest_owned_units",
                "sum",
            ),
            peak_owned_units_on_one_other_cabinet=(
                "peak_owned_units",
                "max",
            ),
            months_on_market=("months_on_market", "max"),
        )
        .sort_values(
            [
                "latest_owned_units_on_other_cabinets",
                "longest_other_cabinet_tenure_months",
            ],
            ascending=False,
        )
    )
    return pair, game_summary, monthly
