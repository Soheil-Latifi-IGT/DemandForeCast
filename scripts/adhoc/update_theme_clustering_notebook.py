import json
from pathlib import Path


NOTEBOOK_PATH = Path("notebooks/holdout_like_game_clustering.ipynb")


def set_cell(nb, index, source):
    nb["cells"][index]["source"] = source.splitlines(True)


def main():
    nb = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))

    set_cell(nb, 0, """# Attr-Only Elbow Like-Game Clustering

This notebook clusters games using structured `game_attr` attributes only. It intentionally excludes cabinet-name features, performance-derived features, `game_type`, raw theme-name text/token features, and duplicate flattened game-matrix indicators.

A clean semantic theme category is included as one derived attr feature. It is built only from true theme/name fields (`theme_name`, `theme_name_friendly`, `family_name`, and `brand_name`), so product-format labels such as `Format Non Theme` are not used.

The relationship map is used only to keep accepted attr-to-historical-game links and to produce the final like-game mapping table. Low-confidence review/fuzzy/no-match relationship rows are dropped before matching.

Cluster count is selected with an elbow view. The selected cluster count is 4 for this run. The elbow table is still generated for review.

The final feature audit CSV lists every feature actually used in clustering. Flattened game-matrix indicators, raw theme-name text tokens, and duplicate `product_line` are not used, so the same signal is not counted twice and raw name text does not drive clusters. Own status and game class from the accepted performance mapping are included as clustering features.

Cluster description charts show the top 2 semantic theme categories plus the top 4 other positive-lift attr features for each cluster, with user-readable labels.
""")

    cell1 = "".join(nb["cells"][1]["source"])
    cell1 = cell1.replace(
        'CLUSTER_SUMMARY_OUTPUT_PATH = ROOT / "Data" / "processed" / "attr_only_elbow_cluster_summary.csv"\nELBOW_OUTPUT_PATH = OUTPUT_DIR / "elbow_scores.csv"',
        'CLUSTER_SUMMARY_OUTPUT_PATH = ROOT / "Data" / "processed" / "attr_only_elbow_cluster_summary.csv"\nTHEME_CATEGORY_OUTPUT_PATH = ROOT / "Data" / "processed" / "attr_only_elbow_theme_categories.csv"\nELBOW_OUTPUT_PATH = OUTPUT_DIR / "elbow_scores.csv"',
    )
    set_cell(nb, 1, cell1)

    cell4 = "".join(nb["cells"][4]["source"])
    cell4 = cell4.replace(
        'MISSING_FEATURE_VALUES = {"__missing__", "__other__", "missing", "nan", "none", "null", ""}',
        'MISSING_FEATURE_VALUES = {"__missing__", "__other__", "uncategorized_theme", "missing", "nan", "none", "null", ""}\nTHEME_CATEGORY_COLUMN = "semantic_theme_bucket"\nTHEME_TEXT_COLUMNS = ["theme_name", "theme_name_friendly", "family_name", "brand_name"]',
    )
    cell4 = cell4.replace(
        '    "attr_theme_bucket": "Theme bucket",\n    "release_age_bucket": "Release age bucket",',
        '    "attr_theme_bucket": "Theme bucket",\n    "semantic_theme_bucket": "Theme category",\n    "semantic_theme_keywords": "Theme keywords",\n    "release_age_bucket": "Release age bucket",',
    )
    cell4 = cell4.replace(
        "\n\ndef is_missing_feature_value(value):\n",
        """

FRIENDLY_VALUE_LABELS = {
    "ancient_civilizations": "Ancient Civilizations",
    "fantasy_mythology": "Fantasy / Mythology",
    "nature_animals": "Nature / Animals",
    "culture_regional": "Culture / Regional",
    "entertainment_licensed": "Entertainment / Licensed",
    "adventure_exploration": "Adventure / Exploration",
    "seasonal_holiday": "Seasonal / Holiday",
    "sports_games": "Sports / Games",
    "food_drink": "Food / Drink",
    "vehicles_speed": "Vehicles / Speed",
    "mystery_scifi": "Mystery / Sci-Fi",
    "luxury_classic": "Luxury / Classic",
}


def is_missing_feature_value(value):
""",
    )
    cell4 = cell4.replace(
        'def friendly_value(value):\n    text = clean_category(value)\n    if text in MISSING_FEATURE_VALUES:\n        return ""\n    return title_preserving_acronyms(text)\n',
        'def friendly_value(value):\n    text = clean_category(value)\n    if text in MISSING_FEATURE_VALUES:\n        return ""\n    if text in FRIENDLY_VALUE_LABELS:\n        return FRIENDLY_VALUE_LABELS[text]\n    return title_preserving_acronyms(text)\n',
    )
    set_cell(nb, 4, cell4)

    set_cell(nb, 6, """THEME_BUCKET_KEYWORDS = {
    "ancient_civilizations": [
        "egypt", "egyptian", "pharaoh", "pyramid", "nile", "cleopatra", "ramosis", "ramses",
        "roman", "rome", "caesar", "gladiator", "greek", "zeus", "athena", "hercules", "odyssey",
        "mayan", "aztec", "inca", "incan", "babylon", "babylonian", "emperor", "dynasty",
    ],
    "fantasy_mythology": [
        "dragon", "phoenix", "fairy", "fairies", "genie", "magic", "wizard", "witch", "myth",
        "mythology", "thor", "valkyrie", "unicorn", "mermaid", "siren", "goddess", "god", "legend",
    ],
    "nature_animals": [
        "animal", "buffalo", "wolf", "tiger", "lion", "leopard", "panther", "eagle", "bear", "horse",
        "stallion", "dolphin", "shark", "fish", "frog", "turtle", "panda", "monkey", "gorilla", "bird",
        "flamingo", "jungle", "forest", "ocean", "sea", "island", "mountain", "river", "lotus",
        "orchid", "flower", "garden", "safari",
    ],
    "culture_regional": [
        "asian", "china", "chinese", "japan", "japanese", "samurai", "ninja", "irish", "ireland",
        "italy", "italian", "latin", "mexico", "mexican", "fiesta", "western", "cowboy", "native",
        "indian", "australia", "australian", "africa", "african", "country", "mardi gras",
    ],
    "entertainment_licensed": [
        "movie", "tv", "television", "music", "show", "rock", "concert", "celebrity", "cartoon",
        "character", "beerfest", "casper", "karate kid", "willie", "nelson", "press your luck", "hollywood",
    ],
    "adventure_exploration": [
        "pirate", "viking", "treasure", "quest", "voyage", "adventure", "expedition", "frontier",
        "explorer", "journey", "trail", "wild", "outlaw",
    ],
    "seasonal_holiday": [
        "winter", "christmas", "holiday", "halloween", "spooky", "fright", "haunted", "harvest",
        "valentine", "easter", "fiesta", "firework", "fireworks",
    ],
    "sports_games": [
        "football", "baseball", "basketball", "golf", "derby", "racing", "race", "nascar", "boxing",
        "soccer", "hockey", "touchdown", "stadium", "sports",
    ],
    "food_drink": [
        "chili", "chile", "pepper", "fruit", "cherry", "lemon", "orange", "grape", "beer", "wine",
        "cocktail", "tequila", "candy", "cake", "sugar", "sweet",
    ],
    "vehicles_speed": [
        "car", "cars", "motorcycle", "bike", "truck", "train", "locomotive", "plane", "aircraft",
        "speed", "speedway", "road", "highway",
    ],
    "mystery_scifi": [
        "space", "galaxy", "cosmic", "planet", "alien", "moon", "starship", "sci fi", "scifi",
        "mystery", "horror", "zombie", "mystic", "supernova",
    ],
    "luxury_classic": [
        "gold", "golden", "diamond", "jewel", "jewels", "ruby", "emerald", "pearl", "cash", "money",
        "wealth", "wealthy", "riches", "fortune", "royal", "king", "queen", "crown", "classic", "casino",
        "jackpot", "lucky", "luck", "win", "winner", "million", "millionaire",
    ],
}


def normalized_theme_search_text(row):
    pieces = []
    for column in THEME_TEXT_COLUMNS:
        value = row.get(column, pd.NA)
        if pd.notna(value):
            text = str(value).strip().lower()
            if text and text != "nan":
                pieces.append(text)
    text = " ".join(pieces)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\\s+", " ", text).strip()


def theme_keyword_pattern(keyword):
    normalized = re.sub(r"[^a-z0-9]+", " ", str(keyword).lower()).strip()
    escaped = re.escape(normalized).replace(r"\\ ", r"\\s+")
    return rf"(?<![a-z0-9]){escaped}s?(?![a-z0-9])"


def derive_semantic_theme_bucket(row):
    text = normalized_theme_search_text(row)
    if not text:
        return "__missing__", ""
    for bucket, keywords in THEME_BUCKET_KEYWORDS.items():
        matched = [keyword for keyword in keywords if re.search(theme_keyword_pattern(keyword), text)]
        if matched:
            return bucket, ", ".join(dict.fromkeys(matched[:6]))
    return "__missing__", ""


def first_release_date(row):
    candidates = []
    for column in ["na_release_date", "lac_release_date", "emea_release_date", "apac_release_date", "matched_release_date"]:
        if column in row and pd.notna(row[column]):
            candidates.append(row[column])
    if not candidates:
        return pd.NaT
    return min(candidates)


attr_work = attr.copy()
attr_work["join_attr_theme_name"] = attr_work["theme_name_friendly"].map(norm_text)
attr_work = attr_work[attr_work["join_attr_theme_name"].notna()].copy()
if "audit_is_deleted" in attr_work.columns:
    attr_work = attr_work[~bool_series(attr_work["audit_is_deleted"]).fillna(0).astype(bool)].copy()

for column in ["na_release_date", "lac_release_date", "emea_release_date", "apac_release_date"]:
    if column in attr_work.columns:
        attr_work[column] = pd.to_datetime(attr_work[column], errors="coerce")

profiles = attr_work.merge(rel_profile, on="join_attr_theme_name", how="inner")
profiles = profiles.reset_index(drop=True)
profiles.insert(0, "profile_id", np.arange(len(profiles), dtype=int))

profiles["profile_release_date"] = profiles.apply(first_release_date, axis=1)
reference_date = profiles["profile_release_date"].max()
if pd.isna(reference_date):
    reference_date = pd.Timestamp.today().normalize()
profiles["release_age_months"] = profiles["profile_release_date"].map(lambda value: months_between(reference_date, value)).clip(lower=0)
profiles["release_age_bucket"] = profiles["release_age_months"].map(release_age_bucket)
semantic_theme = profiles.apply(
    lambda row: pd.Series(derive_semantic_theme_bucket(row), index=["semantic_theme_bucket", "semantic_theme_keywords"]),
    axis=1,
)
profiles = pd.concat([profiles, semantic_theme], axis=1)
profiles["semantic_theme_label"] = profiles["semantic_theme_bucket"].map(friendly_value)

for matrix_token in ["bingo", "hhr", "lottery", "prize", "reels"]:
    profiles[f"attr_matrix_token_{matrix_token}"] = profiles["game_matrix"].map(
        lambda value, token=matrix_token: float(any(token in item.lower() for item in parse_listish(value)))
    )

print(f"Attr-only profiles after accepted relationship join: {len(profiles):,}")
print("Semantic theme category coverage:")
print(profiles["semantic_theme_label"].replace("", "No theme category").value_counts().head(20).to_string())
display(profiles.head(5))
""")

    set_cell(nb, 8, """# Clustering features are structured attr-only.
# Exclusions:
# - game_type is not used.
# - cabinet-name and performance fields are not used.
# - raw theme-name text/token features are not used.
# - flattened game_matrix indicators are not used with raw game_matrix, so the same signal is not counted twice.
# - one clean semantic theme category is included from true theme/name fields only.
ATTR_CATEGORICAL_WEIGHTS = {
    "source_company_cd": 0.7,
    "source_system_cd": 0.35,
    "family_name": 1.7,
    "brand_name": 1.5,
    "semantic_theme_bucket": 1.4,
    "product_family": 1.0,
    "parent_own_status": 1.3,
    "game_classification": 1.5,
    "game_matrix": 2.5,
    "product_segment": 1.8,
    "business_segment": 1.4,
    "gaming_channel_cd": 1.0,
    "volatility_cd": 1.2,
    "progressive_tiers": 1.1,
    "cert_family_ref": 0.9,
    "release_age_bucket": 0.30,
}

ATTR_BOOLEAN_WEIGHTS = {
    "is_wap": 2.2,
    "is_poker": 1.9,
    "is_mlp": 1.6,
    "is_multi_game": 2.6,
    "is_tournament_capable": 1.2,
    "is_multi_denom": 1.9,
}

ATTR_NUMERIC_WEIGHTS = {
    "rtp_default": 0.9,
    "max_bet": 1.2,
    "min_bet": 0.9,
    "lines": 1.0,
    "ways": 1.0,
    "release_age_months": 0.30,
}


def add_categorical_block(frame, column, weight, max_levels=180, min_count=2, drop_missing_features=False):
    if column not in frame.columns:
        return None, []
    series = frame[column].map(clean_category).astype("string")
    counts = series.value_counts(dropna=False)
    if len(counts) <= 1:
        return None, []
    keep = set(counts[counts >= min_count].head(max_levels).index)
    bucketed = series.where(series.isin(keep), "__other__")
    dummies = pd.get_dummies(bucketed, prefix=f"cat__{column}", dtype=float)
    prefix = f"cat__{column}_"
    feature_values = {
        feature: feature[len(prefix):] if feature.startswith(prefix) else feature
        for feature in dummies.columns
    }
    if drop_missing_features:
        keep_features = [
            feature for feature, value in feature_values.items()
            if not is_missing_feature_value(value)
        ]
        dummies = dummies[keep_features]
    if dummies.shape[1] == 0:
        return None, []
    weighted = dummies * float(weight) * ATTR_FEATURE_MULTIPLIER
    rows = []
    for feature in weighted.columns:
        rows.append({
            "feature_column": feature,
            "source_column": column,
            "feature_type": "categorical",
            "feature_value": feature_values[feature],
            "base_weight": float(weight),
            "multiplier": ATTR_FEATURE_MULTIPLIER,
            "final_weight": float(weight) * ATTR_FEATURE_MULTIPLIER,
        })
    return weighted, rows


def add_boolean_block(frame, column, weight):
    if column not in frame.columns:
        return None, []
    values = bool_series(frame[column]).fillna(0.0)
    if values.nunique(dropna=False) <= 1:
        return None, []
    feature = f"bool__{column}"
    block = pd.DataFrame({feature: values * float(weight) * ATTR_FEATURE_MULTIPLIER}, index=frame.index)
    rows = [{
        "feature_column": feature,
        "source_column": column,
        "feature_type": "boolean",
        "feature_value": "true",
        "base_weight": float(weight),
        "multiplier": ATTR_FEATURE_MULTIPLIER,
        "final_weight": float(weight) * ATTR_FEATURE_MULTIPLIER,
    }]
    return block, rows


def add_numeric_block(frame, column, weight):
    if column not in frame.columns:
        return None, []
    values = numeric_series(frame[column])
    if any(token in column for token in ["bet", "lines", "ways", "months"]):
        values = np.log1p(values.clip(lower=0))
    scaled = robust_scale_01(values)
    if scaled is None or scaled.nunique(dropna=False) <= 1:
        return None, []
    feature = f"num__{column}"
    block = pd.DataFrame({feature: scaled * float(weight) * ATTR_FEATURE_MULTIPLIER}, index=frame.index)
    rows = [{
        "feature_column": feature,
        "source_column": column,
        "feature_type": "numeric",
        "feature_value": "",
        "base_weight": float(weight),
        "multiplier": ATTR_FEATURE_MULTIPLIER,
        "final_weight": float(weight) * ATTR_FEATURE_MULTIPLIER,
    }]
    return block, rows


def build_weighted_feature_matrix(frame):
    blocks = []
    feature_sources = {}
    metadata_rows = []

    for column, weight in ATTR_CATEGORICAL_WEIGHTS.items():
        block, rows = add_categorical_block(
            frame,
            column,
            weight,
            drop_missing_features=column == THEME_CATEGORY_COLUMN,
        )
        if block is not None:
            blocks.append(block)
            metadata_rows.extend(rows)
            source_family = "semantic_theme_attr" if column == THEME_CATEGORY_COLUMN else "attr"
            feature_sources.update({feature: source_family for feature in block.columns})

    for column, weight in ATTR_BOOLEAN_WEIGHTS.items():
        block, rows = add_boolean_block(frame, column, weight)
        if block is not None:
            blocks.append(block)
            metadata_rows.extend(rows)
            feature_sources.update({feature: "attr" for feature in block.columns})

    for column, weight in ATTR_NUMERIC_WEIGHTS.items():
        block, rows = add_numeric_block(frame, column, weight)
        if block is not None:
            blocks.append(block)
            metadata_rows.extend(rows)
            feature_sources.update({feature: "attr" for feature in block.columns})

    if not blocks:
        raise ValueError("No usable structured attr clustering features were created.")

    matrix = pd.concat(blocks, axis=1).fillna(0.0)
    matrix["__profile_bias"] = PROFILE_BIAS_WEIGHT
    feature_sources["__profile_bias"] = "bias"
    metadata_rows.append({
        "feature_column": "__profile_bias",
        "source_column": "profile_bias",
        "feature_type": "bias",
        "feature_value": "",
        "base_weight": PROFILE_BIAS_WEIGHT,
        "multiplier": 1.0,
        "final_weight": PROFILE_BIAS_WEIGHT,
    })
    metadata = pd.DataFrame(metadata_rows)
    return matrix, feature_sources, metadata


feature_matrix, feature_sources, feature_usage = build_weighted_feature_matrix(profiles)
feature_usage["friendly_source_name"] = feature_usage["source_column"].map(friendly_source_name)
feature_usage["friendly_feature_name"] = feature_usage.apply(
    lambda row: friendly_feature_label(row["source_column"], row["feature_type"], row["feature_value"]),
    axis=1,
)
feature_usage["feature_group"] = np.where(
    feature_usage["source_column"].eq(THEME_CATEGORY_COLUMN),
    "semantic_theme_category",
    feature_usage["feature_type"],
)
feature_usage["is_missing_or_other"] = feature_usage["feature_value"].map(is_missing_feature_value)
feature_usage["is_reportable"] = ~feature_usage["is_missing_or_other"] & feature_usage["feature_type"].ne("bias")
feature_usage["active_count"] = [
    int((feature_matrix[column] > 0).sum()) for column in feature_usage["feature_column"]
]
feature_usage["active_rate"] = feature_usage["active_count"] / len(feature_matrix)
feature_usage["mean_weighted_value"] = [
    float(feature_matrix[column].mean()) for column in feature_usage["feature_column"]
]
feature_usage["max_weighted_value"] = [
    float(feature_matrix[column].max()) for column in feature_usage["feature_column"]
]
feature_usage = feature_usage[
    [
        "feature_column",
        "friendly_feature_name",
        "source_column",
        "friendly_source_name",
        "feature_group",
        "feature_type",
        "feature_value",
        "base_weight",
        "multiplier",
        "final_weight",
        "active_count",
        "active_rate",
        "mean_weighted_value",
        "max_weighted_value",
        "is_missing_or_other",
        "is_reportable",
    ]
].sort_values(["feature_group", "feature_type", "source_column", "feature_value"]).reset_index(drop=True)

for forbidden in ["game_type", "cabinet", "performance", "perf_", "slot_cabinet", "theme_name_text", "token__", "attr_theme_bucket", "format_non_theme", "product_line"]:
    bad = [column for column in feature_matrix.columns if forbidden in column.lower()]
    if bad:
        raise ValueError(f"Forbidden clustering features found for {forbidden}: {bad[:10]}")

print(f"Feature matrix shape: {feature_matrix.shape[0]:,} profiles x {feature_matrix.shape[1]:,} structured attr-only weighted features")
print(pd.Series(feature_sources).value_counts().to_string())
print(f"Zero-norm rows: {(np.linalg.norm(feature_matrix.to_numpy(dtype=float), axis=1) == 0).sum()}")
print(f"Reportable clustering features: {int(feature_usage['is_reportable'].sum()):,}")
display(feature_usage.head(25))
display(feature_matrix.head(3))
""")

    cell10 = "".join(nb["cells"][10]["source"])
    cell10 = cell10.replace(
        '                "target_theme_name_friendly": target.get("theme_name_friendly"),\n                "target_product_family": target.get("product_family"),',
        '                "target_theme_name_friendly": target.get("theme_name_friendly"),\n                "target_semantic_theme_bucket": target.get("semantic_theme_bucket"),\n                "target_semantic_theme_label": target.get("semantic_theme_label"),\n                "target_product_family": target.get("product_family"),',
    )
    cell10 = cell10.replace(
        '                "like_theme_name_friendly": like.get("theme_name_friendly"),\n                "like_product_family": like.get("product_family"),',
        '                "like_theme_name_friendly": like.get("theme_name_friendly"),\n                "like_semantic_theme_bucket": like.get("semantic_theme_bucket"),\n                "like_semantic_theme_label": like.get("semantic_theme_label"),\n                "like_product_family": like.get("product_family"),',
    )
    cell10 = cell10.replace(
        '        "target_theme_name_friendly",\n        "target_product_family",',
        '        "target_theme_name_friendly",\n        "target_semantic_theme_bucket",\n        "target_semantic_theme_label",\n        "target_product_family",',
    )
    cell10 = cell10.replace(
        '        like_attr_themes=("like_theme_name_friendly", lambda values: " | ".join(values.astype(str))),\n        like_historical_games=',
        '        like_attr_themes=("like_theme_name_friendly", lambda values: " | ".join(values.astype(str))),\n        like_theme_categories=("like_semantic_theme_label", lambda values: " | ".join(values.fillna("").astype(str))),\n        like_historical_games=',
    )
    set_cell(nb, 10, cell10)

    set_cell(nb, 11, """def top_values(series, n=3):
    cleaned = series.map(clean_category)
    cleaned = cleaned[~cleaned.isin(MISSING_FEATURE_VALUES)]
    counts = cleaned.value_counts(normalize=True).head(n)
    if counts.empty:
        return ""
    return "; ".join(f"{friendly_value(idx)} ({share:.0%})" for idx, share in counts.items())


def reportable_lift_from(feature_lift_frame, cluster_id, n, source_column=None, exclude_source_column=None, important_only=True):
    usage = feature_usage[feature_usage["is_reportable"]].copy()
    if source_column is not None:
        usage = usage[usage["source_column"].eq(source_column)]
    if exclude_source_column is not None:
        usage = usage[~usage["source_column"].eq(exclude_source_column)]
    reportable = usage["feature_column"].tolist()
    if not reportable:
        return pd.Series(dtype=float)
    lifted_all = feature_lift_frame.loc[cluster_id, reportable].sort_values(ascending=False)
    lifted_all = lifted_all[lifted_all > 0]
    if lifted_all.empty:
        return lifted_all

    if not important_only:
        return lifted_all.head(n)

    top_lift = float(lifted_all.iloc[0])
    threshold = max(
        CLUSTER_FEATURE_MIN_ABSOLUTE_LIFT,
        top_lift * CLUSTER_FEATURE_MIN_RELATIVE_TO_TOP,
    )
    important = lifted_all[lifted_all >= threshold]
    if len(important) >= CLUSTER_FEATURE_MIN_COUNT:
        return important.head(n)

    filler = lifted_all.drop(index=important.index, errors="ignore")
    combined = pd.concat([important, filler]).head(min(n, CLUSTER_FEATURE_MIN_COUNT))
    return combined


def cluster_description_lifts(cluster_id, theme_count=2, attr_count=4, feature_lift_frame=None):
    if feature_lift_frame is None:
        feature_lift_frame = feature_lift
    theme_lift = reportable_lift_from(
        feature_lift_frame,
        cluster_id,
        theme_count,
        source_column=THEME_CATEGORY_COLUMN,
        important_only=False,
    )
    attr_lift = reportable_lift_from(
        feature_lift_frame,
        cluster_id,
        attr_count,
        exclude_source_column=THEME_CATEGORY_COLUMN,
        important_only=False,
    )
    combined = pd.concat([theme_lift, attr_lift])
    combined = combined[~combined.index.duplicated(keep="first")]
    return theme_lift, attr_lift, combined


def build_cluster_summary(profiles, feature_matrix):
    feature_view = feature_matrix.drop(columns=["__profile_bias"], errors="ignore")
    cluster_means = feature_view.groupby(profiles["cluster_id"]).mean()
    global_means = feature_view.mean()
    feature_lift = cluster_means.subtract(global_means, axis=1)

    summary = (
        profiles.groupby("cluster_id", as_index=False)
        .agg(
            profile_count=("profile_id", "size"),
            holdout_count=("is_holdout", "sum"),
            historical_count=("is_qualified_historical", "sum"),
            min_release_date=("profile_release_date", "min"),
            max_release_date=("profile_release_date", "max"),
            median_release_age_months=("release_age_months", "median"),
        )
        .sort_values("cluster_id")
    )

    characteristic_columns = [
        "semantic_theme_bucket",
        "product_family",
        "parent_own_status",
        "game_classification",
        "game_matrix",
        "product_segment",
        "family_name",
        "brand_name",
        "business_segment",
        "volatility_cd",
        "is_multi_denom",
        "is_wap",
        "is_poker",
    ]
    for column in characteristic_columns:
        if column in profiles.columns:
            values = profiles.groupby("cluster_id")[column].apply(top_values).rename(f"top_{column}")
            summary = summary.merge(values.reset_index(), on="cluster_id", how="left")

    top_theme_features = []
    top_attr_features = []
    top_features = []
    for cluster_id in summary["cluster_id"]:
        theme_lift, attr_lift, combined_lift = cluster_description_lifts(cluster_id, feature_lift_frame=feature_lift)
        theme_text = "; ".join(pretty_feature_name(name) for name in theme_lift.index[:2])
        attr_text = "; ".join(pretty_feature_name(name) for name in attr_lift.index[:4])
        top_theme_features.append(theme_text)
        top_attr_features.append(attr_text)
        parts = []
        if theme_text:
            parts.append(f"Theme categories: {theme_text}")
        if attr_text:
            parts.append(f"Attr features: {attr_text}")
        if not parts:
            parts.append("No positive-lift reportable features")
        top_features.append(" | ".join(parts))
    summary["top_theme_category_features"] = top_theme_features
    summary["top_structured_attr_features"] = top_attr_features
    summary["top_distinguishing_features"] = top_features
    return summary, feature_lift


cluster_summary, feature_lift = build_cluster_summary(profiles, feature_matrix)
display(cluster_summary)
""")

    set_cell(nb, 13, """def save_cluster_feature_chart(cluster_id):
    theme_lift, attr_lift, lifted = cluster_description_lifts(cluster_id)
    labels = [pretty_feature_name(name) for name in lifted.index]
    values = lifted.to_numpy()
    sources = feature_usage.set_index("feature_column").reindex(lifted.index)["source_column"] if len(lifted) else pd.Series(dtype="object")
    colors = ["#7b5ea7" if source == THEME_CATEGORY_COLUMN else "#3568a8" for source in sources]
    cluster_rows = profiles[profiles["cluster_id"].eq(cluster_id)]
    summary_row = cluster_summary[cluster_summary["cluster_id"].eq(cluster_id)].iloc[0]

    fig, ax = plt.subplots(figsize=(12, 6.5))
    if len(values):
        y = np.arange(len(values))
        ax.barh(y, values, color=colors)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=9)
        ax.invert_yaxis()
    else:
        ax.text(0.5, 0.5, "No positive-lift reportable features", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("Feature lift vs global mean")
    ax.text(
        0.99,
        0.02,
        "Purple = top theme categories; blue = top other attr features",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="#555555",
    )
    ax.set_title(
        f"Cluster {cluster_id}: Top 2 Theme Categories + Top 4 Attr Features "
        f"({len(cluster_rows)} games, {int(summary_row['holdout_count'])} holdout)"
    )
    ax.grid(axis="x", alpha=0.20)

    text_lines = [
        f"Top semantic themes: {summary_row.get('top_semantic_theme_bucket', '')}",
        f"Top product family: {summary_row.get('top_product_family', '')}",
        f"Top own status: {summary_row.get('top_parent_own_status', '')}",
        f"Top game class: {summary_row.get('top_game_classification', '')}",
        f"Top game matrix: {summary_row.get('top_game_matrix', '')}",
    ]
    fig.text(0.02, 0.02, "\\n".join([line for line in text_lines if not line.endswith(': ')]), ha="left", va="bottom", fontsize=9)
    path = CLUSTER_CHART_DIR / f"cluster_{int(cluster_id):02d}_features.png"
    fig.subplots_adjust(left=0.37, right=0.98, top=0.88, bottom=0.20)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


for old_chart in CLUSTER_CHART_DIR.glob("*.png"):
    old_chart.unlink()

cluster_chart_paths = []
for cluster_id in sorted(profiles["cluster_id"].unique()):
    cluster_chart_paths.append(save_cluster_feature_chart(cluster_id))

print(f"Saved {len(cluster_chart_paths)} cluster feature charts to {CLUSTER_CHART_DIR}")
display(pd.DataFrame({"cluster_id": sorted(profiles["cluster_id"].unique()), "chart_path": [str(p) for p in cluster_chart_paths]}))
""")

    cell14 = "".join(nb["cells"][14]["source"])
    cell14 = cell14.replace(
        '    text = (\n        f"Holdout: {row.get(\'theme_name_friendly\')}\\n"\n        f"Assigned cluster: {int(row.get(\'cluster_id\'))}\\n"',
        '    theme_label = friendly_value(row.get(THEME_CATEGORY_COLUMN)) or "No theme category"\n    theme_keywords = row.get("semantic_theme_keywords") or ""\n    text = (\n        f"Holdout: {row.get(\'theme_name_friendly\')}\\n"\n        f"Assigned cluster: {int(row.get(\'cluster_id\'))} | Theme category: {theme_label}\\n"\n        f"Theme keywords: {theme_keywords}\\n"',
    )
    cell14 = cell14.replace(
        '        "theme_name_friendly": row.get("theme_name_friendly"),\n        "cluster_id": int(row.get("cluster_id")),',
        '        "theme_name_friendly": row.get("theme_name_friendly"),\n        "semantic_theme_label": row.get("semantic_theme_label"),\n        "cluster_id": int(row.get("cluster_id")),',
    )
    set_cell(nb, 14, cell14)

    set_cell(nb, 15, """theme_category_columns = [
    "profile_id",
    "theme_sk",
    "theme_name_friendly",
    "family_name",
    "brand_name",
    "semantic_theme_bucket",
    "semantic_theme_label",
    "semantic_theme_keywords",
    "cluster_id",
    "is_holdout",
    "is_qualified_historical",
]
theme_category_audit = profiles[[column for column in theme_category_columns if column in profiles.columns]].copy()

theme_category_audit.to_csv(THEME_CATEGORY_OUTPUT_PATH, index=False)
profiles.to_csv(PROFILES_OUTPUT_PATH, index=False)
feature_matrix.to_csv(FEATURE_MATRIX_OUTPUT_PATH, index=False)
feature_usage.to_csv(FEATURE_USAGE_OUTPUT_PATH, index=False)
like_games.to_csv(MAPPING_OUTPUT_PATH, index=False)
grouped_like_games.to_csv(GROUPED_MAPPING_OUTPUT_PATH, index=False)
cluster_summary.to_csv(CLUSTER_SUMMARY_OUTPUT_PATH, index=False)
elbow_scores.to_csv(ELBOW_OUTPUT_PATH, index=False)
holdout_chart_index.to_csv(CHART_INDEX_OUTPUT_PATH, index=False)

manifest = {
    "inputs": {
        "attr_path": str(ATTR_PATH),
        "attr_raw_path": str(ATTR_RAW_PATH),
        "rel_map_path": str(REL_MAP_PATH),
    },
    "profile_rows": int(len(profiles)),
    "feature_columns": int(feature_matrix.shape[1]),
    "feature_source_counts": {str(k): int(v) for k, v in pd.Series(feature_sources).value_counts().items()},
    "reportable_feature_columns": int(feature_usage["is_reportable"].sum()),
    "holdout_profiles": int(profiles["is_holdout"].sum()),
    "historical_profiles": int(profiles["is_qualified_historical"].sum()),
    "mapped_holdout_profiles": int(like_games["target_profile_id"].nunique()),
    "mapping_rows": int(len(like_games)),
    "auto_elbow_k": int(auto_elbow_k),
    "selected_clusters": int(profiles["cluster_id"].nunique()),
    "selected_clusters_setting": int(ELBOW_SELECTED_N_CLUSTERS),
    "excluded_match_methods": sorted(EXCLUDED_MATCH_METHODS),
    "distance_metric": DISTANCE_METRIC,
    "linkage_method": LINKAGE_METHOD,
    "included_derived_features": [
        "semantic_theme_bucket_from_theme_name_family_brand_only",
    ],
    "excluded_feature_families": [
        "cabinet_name",
        "performance",
        "game_type",
        "flattened_game_matrix_duplicates",
        "math_model_family_duplicate",
        "product_line_duplicate_of_product_family",
        "raw_theme_text_tokens",
        "attr_theme_bucket_format_non_theme",
    ],
    "outputs": {
        "profiles": str(PROFILES_OUTPUT_PATH),
        "theme_categories": str(THEME_CATEGORY_OUTPUT_PATH),
        "feature_matrix": str(FEATURE_MATRIX_OUTPUT_PATH),
        "feature_usage": str(FEATURE_USAGE_OUTPUT_PATH),
        "mapping": str(MAPPING_OUTPUT_PATH),
        "grouped_mapping": str(GROUPED_MAPPING_OUTPUT_PATH),
        "cluster_summary": str(CLUSTER_SUMMARY_OUTPUT_PATH),
        "elbow_scores": str(ELBOW_OUTPUT_PATH),
        "chart_index": str(CHART_INDEX_OUTPUT_PATH),
        "cluster_chart_dir": str(CLUSTER_CHART_DIR),
        "holdout_chart_dir": str(HOLDOUT_CHART_DIR),
        "report_dir": str(OUTPUT_DIR),
    },
}
MANIFEST_OUTPUT_PATH.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

print("Saved outputs:")
for path in [
    THEME_CATEGORY_OUTPUT_PATH,
    PROFILES_OUTPUT_PATH,
    FEATURE_MATRIX_OUTPUT_PATH,
    FEATURE_USAGE_OUTPUT_PATH,
    MAPPING_OUTPUT_PATH,
    GROUPED_MAPPING_OUTPUT_PATH,
    CLUSTER_SUMMARY_OUTPUT_PATH,
    ELBOW_OUTPUT_PATH,
    CHART_INDEX_OUTPUT_PATH,
    MANIFEST_OUTPUT_PATH,
]:
    print(f"- {path}")
""")

    NOTEBOOK_PATH.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"Updated {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
