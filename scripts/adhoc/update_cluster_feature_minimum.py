from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK = Path("notebooks/holdout_like_game_clustering.ipynb")


nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))

constants = "".join(nb["cells"][2]["source"])
if "CLUSTER_FEATURE_MIN_COUNT" not in constants:
    constants = constants.replace(
        "TOP_CLUSTER_FEATURES = 14\n",
        "TOP_CLUSTER_FEATURES = 14\n"
        "CLUSTER_FEATURE_MIN_COUNT = 5\n",
    )
nb["cells"][2]["source"] = constants.splitlines(keepends=True)

summary_src = "".join(nb["cells"][11]["source"])
old_helper = '''def reportable_lift_from(feature_lift_frame, cluster_id, n, important_only=True):
    reportable = feature_usage.loc[feature_usage["is_reportable"], "feature_column"].tolist()
    lifted = feature_lift_frame.loc[cluster_id, reportable].sort_values(ascending=False)
    lifted = lifted[lifted > 0]
    if important_only and not lifted.empty:
        top_lift = float(lifted.iloc[0])
        threshold = max(
            CLUSTER_FEATURE_MIN_ABSOLUTE_LIFT,
            top_lift * CLUSTER_FEATURE_MIN_RELATIVE_TO_TOP,
        )
        important = lifted[lifted >= threshold]
        if not important.empty:
            lifted = important
    return lifted.head(n)
'''
new_helper = '''def reportable_lift_from(feature_lift_frame, cluster_id, n, important_only=True):
    reportable = feature_usage.loc[feature_usage["is_reportable"], "feature_column"].tolist()
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
'''
if old_helper not in summary_src:
    raise RuntimeError("Could not find the cluster feature helper to replace.")
summary_src = summary_src.replace(old_helper, new_helper)
nb["cells"][11]["source"] = summary_src.splitlines(keepends=True)

chart_src = "".join(nb["cells"][13]["source"])
chart_src = chart_src.replace(
    'f"Showing lift >= {CLUSTER_FEATURE_MIN_ABSOLUTE_LIFT:g} and >= {CLUSTER_FEATURE_MIN_RELATIVE_TO_TOP:.0%} of top lift",',
    'f"Showing strongest features; minimum {CLUSTER_FEATURE_MIN_COUNT} per cluster",',
)
nb["cells"][13]["source"] = chart_src.splitlines(keepends=True)

intro = "".join(nb["cells"][0]["source"])
if "Each cluster summary and chart includes at least 5 reportable attributes." not in intro:
    intro += "\nEach cluster summary and chart includes at least 5 reportable attributes, while still prioritizing the strongest lift features.\n"
nb["cells"][0]["source"] = intro.splitlines(keepends=True)

NOTEBOOK.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"Updated {NOTEBOOK}")
