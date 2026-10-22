# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# Databricks notebook source
# Stage 1: build separate, reviewable canonical dictionaries for themes and
# mechanics. This notebook does not score PDFs or write to the source table.
# Run this after extraction; use a later process to apply reviewed mappings to
# the final game dataset and to aggregate prominence scores.

# COMMAND ----------

import hashlib
import json
import re
from collections import defaultdict

from pyspark.sql import functions as F


# ---------- Configuration ----------

SOURCE_TABLE = "development_042_silver_sandbox.demand_forecast.overview_ctc_line_reel_theme_raw"
RUN_TAG = "v1"  # Change when you want a new set of output/cache tables.
TEXT_MODEL = "system.ai.llama-4-maverick"

# All output tables are derived development tables and are replaced on rerun.
# The source extraction table is read-only in this notebook.
OUTPUT_PREFIX = SOURCE_TABLE.rsplit(".", 1)[0] + ".game_taxonomy_unification_" + RUN_TAG

TAXONOMY_BATCH_SIZE = 70
TAXONOMY_BATCH_CHARS = 18000
MAX_CONTEXT_CHARS = 320
MAX_RECONCILIATION_ROUNDS = 8

assert re.fullmatch(r"[A-Za-z0-9_]+", RUN_TAG)


def qtable(name):
    return ".".join("`" + part.replace("`", "``") + "`" for part in name.split("."))


def sql_string(value):
    return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"


def normalize_label(value):
    return re.sub(r"\s+", " ", str(value).strip()).casefold()


def write_table(df, suffix):
    name = OUTPUT_PREFIX + suffix
    df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(name)
    return spark.table(name)


def object_schema(properties):
    return {"type": "object", "properties": properties,
            "required": list(properties), "additionalProperties": False}


def ai_cached(requests, suffix, schema):
    """Cache each text request in Delta so a rerun does not repeat completed inference."""
    cache_table = OUTPUT_PREFIX + suffix
    response_format = json.dumps({"type": "json_schema", "json_schema": {
        "name": "taxonomy_result", "strict": True, "schema": schema}})
    fingerprint = TEXT_MODEL + response_format
    requests = requests.withColumn(
        "request_id", F.sha2(F.concat(F.lit(fingerprint), F.col("request")), 256))
    unique = requests.dropDuplicates(["request_id"])

    if spark.catalog.tableExists(cache_table):
        pending = unique.join(
            spark.table(cache_table).select("request_id"), "request_id", "left_anti")
    else:
        pending = unique

    pending_count = pending.count()
    print(f"{suffix}: {pending_count} new AI requests")
    if pending_count:
        expression = (
            f"ai_query({sql_string(TEXT_MODEL)}, request, failOnError => false, "
            f"modelParameters => named_struct('temperature', 0.0, 'max_tokens', 8192), "
            f"responseFormat => {sql_string(response_format)})"
        )
        pending.select("request_id", F.expr(expression).alias("ai")) \
            .write.format("delta").mode("append").saveAsTable(cache_table)

    return requests.join(spark.table(cache_table), "request_id", "left")


def decode_answer(value):
    if hasattr(value, "asDict"):
        value = value.asDict(recursive=True)
    elif hasattr(value, "toPython"):
        value = value.toPython()
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8")
    if isinstance(value, str):
        value = json.loads(value.strip().lstrip("\ufeff"))
    if isinstance(value, dict) and "response" in value:
        return decode_answer(value["response"])
    if isinstance(value, dict) and "result" in value:
        error = value.get("errorMessage") or value.get("error_message")
        if error:
            raise ValueError(f"AI endpoint error: {str(error)[:400]}")
        return decode_answer(value["result"])
    if not isinstance(value, dict):
        raise ValueError(f"AI response must be a JSON object; got {type(value).__name__}")
    error = value.get("errorMessage") or value.get("error_message")
    if error:
        raise ValueError(f"AI endpoint error: {str(error)[:400]}")
    return value


def validate_json(value, schema, path="$"):
    kind = schema.get("type")
    checks = {
        "object": lambda x: isinstance(x, dict),
        "array": lambda x: isinstance(x, list),
        "string": lambda x: isinstance(x, str),
        "integer": lambda x: type(x) is int,
    }
    kinds = kind if isinstance(kind, list) else [kind]
    if not any(k in checks and checks[k](value) for k in kinds):
        raise ValueError(f"{path}: expected {kind}; got {type(value).__name__}")
    if isinstance(value, dict):
        props = schema.get("properties", {})
        missing = set(schema.get("required", [])) - set(value)
        extra = set(value) - set(props)
        if missing or (schema.get("additionalProperties") is False and extra):
            raise ValueError(f"{path}: missing={sorted(missing)}, unexpected={sorted(extra)}")
        for key, sub_schema in props.items():
            if key in value:
                validate_json(value[key], sub_schema, path + "." + key)
    elif isinstance(value, list):
        for i, item in enumerate(value):
            validate_json(item, schema["items"], f"{path}[{i}]")


def as_list(value):
    """Accept Spark arrays/structs and JSON strings from the extraction output."""
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            return [value]
    return value if isinstance(value, (list, tuple)) else [value]


# COMMAND ----------

# ---------- 1. Snapshot the extracted fields used for taxonomy preparation ----------

source = spark.table(SOURCE_TABLE)
required = {"game_name", "themes", "mechanics", "mechanic_families", "mechanic_source_names"}
missing = required - set(source.columns)
if missing:
    raise ValueError(
        f"Source table is missing {sorted(missing)}. Point SOURCE_TABLE to the output "
        "from the extraction script that includes theme and mechanic fields."
    )

snapshot_fields = ["game_name", "themes", "theme_description", "mechanics",
                   "mechanic_families", "mechanic_source_names", "mechanic_notes",
                   "game_attributes", "additional_features", "additional_feature_families",
                   "feature_notes", "game_file", "source_path", "extracted_at"]
snapshot_fields = [c for c in snapshot_fields if c in source.columns]
snapshot = write_table(source.select(*snapshot_fields), "_input_snapshot")
records = [row.asDict(recursive=True) for row in snapshot.collect()]

if not records:
    raise ValueError(f"No rows found in {SOURCE_TABLE}")
if any(not row.get("game_name") or not str(row["game_name"]).strip() for row in records):
    raise ValueError("Resolve empty game_name values before taxonomy preparation.")

print(f"Snapshotted {len(records)} extracted game rows from {SOURCE_TABLE}.")


# COMMAND ----------

# ---------- 2. Build distinct raw-label inventories with provenance/context ----------

def build_inventory(taxonomy, rows):
    """Return one row per normalized raw label while retaining every source spelling."""
    # key -> source spelling variants, game set, and short source-grounded context
    collected = defaultdict(lambda: {"variants": set(), "games": set(), "contexts": set()})

    def add(raw_field, raw_label, game_key, context=""):
        if raw_label is None:
            return
        label = str(raw_label).strip()
        if not label:
            return
        key = (raw_field, normalize_label(label))
        collected[key]["variants"].add(label)
        collected[key]["games"].add(game_key)
        if context:
            collected[key]["contexts"].add(re.sub(r"\s+", " ", str(context)).strip()[:MAX_CONTEXT_CHARS])

    for row in rows:
        game_key = normalize_label(row["game_name"])
        if taxonomy == "theme":
            description = row.get("theme_description") or ""
            for label in as_list(row.get("themes")):
                if isinstance(label, str):
                    add("themes", label, game_key, description)
        else:
            # Include extracted family labels and the wording used in the PDF.
            # The source fields remain distinct so the later mapping can retain provenance.
            for field in ("mechanic_families", "mechanic_source_names"):
                for label in as_list(row.get(field)):
                    if isinstance(label, str):
                        add(field, label, game_key, row.get("mechanic_notes") or "")

            # Pull names/families and descriptions from the detailed mechanic records.
            for mechanic in as_list(row.get("mechanics")):
                if hasattr(mechanic, "asDict"):
                    mechanic = mechanic.asDict(recursive=True)
                if not isinstance(mechanic, dict):
                    continue
                context = " | ".join(str(mechanic.get(k) or "") for k in
                                     ("mechanic_description", "variant_notes", "evidence"))
                for field, key in (("mechanics.source_name", "source_name"),
                                   ("mechanics.mechanic_family", "mechanic_family")):
                    add(field, mechanic.get(key), game_key, context)

    rows_out = []
    for raw_id, ((raw_field, normalized), data) in enumerate(sorted(collected.items())):
        variants = sorted(data["variants"])
        contexts = sorted(data["contexts"])
        rows_out.append({
            "raw_id": raw_id,
            "raw_field": raw_field,
            "normalized_label": normalized,
            "raw_label": variants[0],
            "raw_variants_json": json.dumps(variants, ensure_ascii=False),
            "game_count": len(data["games"]),
            "context": contexts[0][:MAX_CONTEXT_CHARS] if contexts else "",
        })
    return rows_out


inventories = {
    "theme": build_inventory("theme", records),
    "mechanic": build_inventory("mechanic", records),
}
for taxonomy, inventory in inventories.items():
    if not inventory:
        print(f"Warning: no raw {taxonomy} labels found.")
        continue
    write_table(spark.createDataFrame(inventory), f"_{taxonomy}_raw_inventory")
    print(f"{taxonomy.title()}: {len(inventory)} distinct source-field/label entries")


# COMMAND ----------

# ---------- 3. Discover candidates, globally reconcile, then map every raw label ----------

category_schema = object_schema({
    "categories": {"type": "array", "items": object_schema({
        "name": {"type": "string"}, "definition": {"type": "string"}})}
})

assignment_schema = object_schema({
    "assignments": {"type": "array", "items": object_schema({
        "raw_id": {"type": "integer"},
        "category_ids": {"type": "array", "items": {"type": "integer"}},
        "reason": {"type": "string"}})}
})


def pack_batches(items, max_items=TAXONOMY_BATCH_SIZE, max_chars=TAXONOMY_BATCH_CHARS):
    batches, batch, size = [], [], 2
    for item in items:
        item_size = len(json.dumps(item, ensure_ascii=False)) + 2
        if item_size + 2 > max_chars:
            raise ValueError(f"One taxonomy input record is over {max_chars} characters: {item}")
        if batch and (len(batch) >= max_items or size + item_size > max_chars):
            batches.append(batch)
            batch, size = [], 2
        batch.append(item)
        size += item_size
    if batch:
        batches.append(batch)
    return batches


def collect_categories(answers):
    unique = {}
    for answer in answers:
        for entry in answer.get("categories", []):
            name = str(entry.get("name", "")).strip()
            definition = str(entry.get("definition", "")).strip()
            if not name or not definition:
                raise ValueError("AI returned a category with an empty name or definition.")
            unique[(normalize_label(name), normalize_label(definition))] = {
                "name": name, "definition": definition}
    return [unique[k] for k in sorted(unique)]


def query_batches(batches, instruction, schema, suffix):
    requests = [(i, instruction + "\nDATA:\n" + json.dumps(batch, ensure_ascii=False))
                for i, batch in enumerate(batches)]
    frame = spark.createDataFrame(requests, "batch_id int, request string")
    output = ai_cached(frame, suffix, schema).orderBy("batch_id").collect()
    if len(output) != len(batches) or {r.batch_id for r in output} != set(range(len(batches))):
        raise ValueError(f"Missing/duplicate AI response batches in {OUTPUT_PREFIX + suffix}")
    answers = []
    for row in output:
        answer = decode_answer(row.ai)
        validate_json(answer, schema)
        answers.append(answer)
    return answers


def taxonomy_rules(taxonomy):
    if taxonomy == "theme":
        return """Build a taxonomy of game THEMES only: setting, subject, characters, narrative,
and recurring visual motifs. Merge genuine synonyms while preserving distinct subjects.
Exclude mechanics, game titles, supplier/cabinet/product names, generic casino interface,
and unsupported or ambiguous labels. Keep a useful level of detail; do not force a fixed
number of categories. A game may have multiple themes. Do not invent unsupported concepts.
"""
    return """Build a taxonomy of SLOT GAME MECHANICS only: rules and repeatable gameplay
systems/features. Merge labels only when the mechanics are equivalent in operation. Keep
meaningful variants separate when triggers, symbol behavior, respins, collection, or outcomes
differ; a variant may be described in its definition. Do not merge merely related features.
Exclude themes, game titles, supplier/cabinet/product names, and generic marketing claims.
Use concise, recognizable mechanic names and clear definitions. Do not force a fixed number.
"""


def normalize_candidate_round(answers):
    candidates = collect_categories(answers)
    if not candidates:
        raise ValueError("AI returned no taxonomy categories.")
    return candidates


def make_taxonomy(taxonomy, inventory):
    suffix_root = "_" + taxonomy
    compact_inventory = [{
        "raw_id": x["raw_id"], "raw_field": x["raw_field"],
        "label": x["raw_label"], "other_spellings": json.loads(x["raw_variants_json"]),
        "games": x["game_count"], "context": x["context"],
    } for x in inventory]
    input_batches = pack_batches(compact_inventory)
    print(f"{taxonomy.title()} discovery: {len(input_batches)} batches")

    rules = taxonomy_rules(taxonomy)
    discovery_instruction = rules + """
Propose a compact set of provisional categories that covers the supported concepts in this
batch. Return short category names and definitions of at most 35 words. If a label does not
belong in this taxonomy, do not create a catch-all category for it.
"""
    candidates = normalize_candidate_round(query_batches(
        input_batches, discovery_instruction, category_schema, suffix_root + "_discover_ai"))

    merge_instruction = rules + """
The input contains provisional categories from several batches. Reconcile them into ONE
consistent dictionary. Merge true synonyms/overlaps, resolve accidental parent/child
duplicates, and preserve distinct meanings. Do not add concepts absent from the input.
Return each final category once with one concise definition.
"""
    for round_number in range(MAX_RECONCILIATION_ROUNDS):
        if not candidates:
            raise ValueError(f"{taxonomy} reconciliation returned no candidates.")

        # Shift the batch boundaries on each round so categories previously
        # separated into different batches can be compared.
        shift = (round_number * 50) % len(candidates)
        ordered_candidates = candidates[shift:] + candidates[:shift]

        merge_batches = pack_batches(
            ordered_candidates,
            max_items=100,
            max_chars=26000
        )

        print(
            f"{taxonomy.title()} reconciliation {round_number + 1}: "
            f"{len(candidates)} candidates in {len(merge_batches)} batches"
        )

        merged = normalize_candidate_round(query_batches(
            merge_batches,
            merge_instruction,
            category_schema,
            suffix_root + f"_reconcile_{round_number + 1}_ai"
        ))

        if len(merge_batches) == 1:
            candidates = merged
            break

        if len(merged) >= len(candidates):
            print(
                f"No reduction this round ({len(candidates)} candidates); "
                "shifting batch boundaries next round."
            )

        candidates = merged
    else:
        raise ValueError(
            f"{taxonomy} taxonomy did not consolidate after "
            f"{MAX_RECONCILIATION_ROUNDS} rounds. Inspect the latest "
            f"reconciliation AI table."
        )

    # Stable IDs for a given dictionary content; no score-based ranking or top-N truncation.
    candidates = sorted(candidates, key=lambda x: (normalize_label(x["name"]), x["definition"]))
    dictionary = [{
        "category_index": index,
        "category_id": f"{taxonomy[:1]}_" + hashlib.sha256(
            (normalize_label(c["name"]) + "|" + normalize_label(c["definition"])).encode()
        ).hexdigest()[:12],
        "category_name": c["name"],
        "definition": c["definition"],
    } for index, c in enumerate(candidates)]

    write_table(spark.createDataFrame(dictionary), suffix_root + "_dictionary")

    mapping_instruction = rules + """
Map EVERY raw_id in DATA to the FIXED dictionary below. Return every raw_id exactly once.
Map a label only when the supplied evidence supports it. Use the integer category_index
values from the dictionary in category_ids. Assign [] when it is outside the taxonomy,
ambiguous, or not covered; give a short reason for every empty assignment. A
compound label may map to multiple categories only when it clearly expresses multiple
distinct concepts. Never create or rename dictionary categories.

FIXED DICTIONARY:
""" + json.dumps(dictionary, ensure_ascii=False)

    assignment_batches = input_batches
    assignments = []
    for batch_no, batch in enumerate(assignment_batches):
        answer = query_batches([batch], mapping_instruction, assignment_schema,
                               suffix_root + f"_map_batch_{batch_no}_ai")[0]
        entries = answer.get("assignments")
        if not isinstance(entries, list):
            raise ValueError(f"{taxonomy} mapping batch {batch_no} has no assignments array.")
        expected_ids = {item["raw_id"] for item in batch}
        seen = set()
        for entry in entries:
            raw_id = entry.get("raw_id")
            ids = entry.get("category_ids")
            reason = entry.get("reason")
            if type(raw_id) is not int:
                raise ValueError(f"{taxonomy}: non-integer raw_id {raw_id!r} in batch {batch_no}")
            if raw_id not in expected_ids or raw_id in seen:
                print(f"Warning: {taxonomy} batch {batch_no}: skipping unexpected/duplicate raw_id {raw_id!r}")
                continue
            if not isinstance(ids, list):
                print(f"Warning: {taxonomy} raw_id={raw_id}: non-list category_ids, treating as unmapped")
                ids = []
            else:
                valid_ids = [i for i in ids if type(i) is int and 0 <= i < len(dictionary)]
                if len(valid_ids) != len(ids):
                    print(f"Warning: {taxonomy} raw_id={raw_id}: filtered {len(ids) - len(valid_ids)} invalid category IDs")
                ids = valid_ids
            ids = sorted(set(ids))
            if not isinstance(reason, str):
                reason = ""
            seen.add(raw_id)
            assignments.append({"raw_id": raw_id, "category_ids": ids, "reason": reason})
        if seen != expected_ids:
            print(f"Warning: {taxonomy} batch {batch_no} omitted raw IDs {sorted(expected_ids - seen)}")

    by_id = {x["raw_id"]: x for x in inventory}
    by_category_id = {i: item for i, item in enumerate(dictionary)}
    mapping_rows, unmapped_rows = [], []
    category_games = defaultdict(set)
    category_label_count = defaultdict(set)
    for assignment in assignments:
        item = by_id[assignment["raw_id"]]
        for category_index in assignment["category_ids"]:
            category = by_category_id[category_index]
            mapping_rows.append({
                "raw_id": item["raw_id"], "raw_field": item["raw_field"],
                "normalized_label": item["normalized_label"], "raw_label": item["raw_label"],
                "raw_variants_json": item["raw_variants_json"],
                "category_id": category["category_id"],
                "category_name": category["category_name"],
                "definition": category["definition"], "mapping_reason": assignment["reason"],
            })
            category_games[category_index].add(item["normalized_label"])
            category_label_count[category_index].add(item["raw_id"])
        if not assignment["category_ids"]:
            unmapped_rows.append({
                "raw_id": item["raw_id"], "raw_field": item["raw_field"],
                "raw_label": item["raw_label"], "raw_variants_json": item["raw_variants_json"],
                "game_count": item["game_count"], "review_reason": assignment["reason"],
            })

    # Write empty tables with explicit schemas where a taxonomy has no rows.
    mapping_schema = ("raw_id int, raw_field string, normalized_label string, raw_label string, "
                      "raw_variants_json string, category_id string, category_name string, "
                      "definition string, mapping_reason string")
    unmapped_schema = ("raw_id int, raw_field string, raw_label string, raw_variants_json string, "
                       "game_count int, review_reason string")
    mapping_df = spark.createDataFrame(mapping_rows, mapping_schema)
    unmapped_df = spark.createDataFrame(unmapped_rows, unmapped_schema)
    write_table(mapping_df, suffix_root + "_mapping")
    write_table(unmapped_df, suffix_root + "_unmapped_review")

    print(f"{taxonomy.title()}: {len(dictionary)} canonical categories; "
          f"{len(mapping_rows)} mapped rows; {len(unmapped_rows)} labels to review.")
    display(spark.table(OUTPUT_PREFIX + suffix_root + "_dictionary"))
    display(spark.table(OUTPUT_PREFIX + suffix_root + "_unmapped_review"))


# Run the taxonomies independently. No themes are mixed into mechanics or vice versa.
for taxonomy in ("theme", "mechanic"):
    if inventories[taxonomy]:
        make_taxonomy(taxonomy, inventories[taxonomy])


# COMMAND ----------

# ---------- Outputs for the later scoring/dataset process ----------
#   ..._theme_dictionary       canonical theme names and definitions
#   ..._theme_mapping          raw theme labels/spellings -> canonical theme IDs
#   ..._theme_unmapped_review  labels intentionally left unmapped with reasons
#   ..._mechanic_dictionary    canonical mechanic names and definitions
#   ..._mechanic_mapping       raw mechanic
# /family/source names -> canonical IDs
#   ..._mechanic_unmapped_review
#   ..._input_snapshot         unchanged extracted evidence/score fields for lineage
#
# This stage does not:
#   * score PDFs or recalculate prominence;
#   * rank or truncate categories (there is no TOP_N);
#   * merge taxonomy mappings into the final game-level dataset;
#   * update or overwrite SOURCE_TABLE.
#
# The next process should join the reviewed mapping tables to the original extraction
# rows, preserve the original source labels/evidence and mechanic prominence fields, and
# then build game-level features/scores from the canonical IDs.


# COMMAND ----------

review_tables = [
    "development_042_silver_sandbox.demand_forecast.game_taxonomy_unification_v1_input_snapshot",
    "development_042_silver_sandbox.demand_forecast.game_taxonomy_unification_v1_theme_raw_inventory",
    "development_042_silver_sandbox.demand_forecast.game_taxonomy_unification_v1_mechanic_raw_inventory",
    "development_042_silver_sandbox.demand_forecast.game_taxonomy_unification_v1_theme_dictionary",
    "development_042_silver_sandbox.demand_forecast.game_taxonomy_unification_v1_theme_mapping",
    "development_042_silver_sandbox.demand_forecast.game_taxonomy_unification_v1_theme_unmapped_review",
    "development_042_silver_sandbox.demand_forecast.game_taxonomy_unification_v1_mechanic_dictionary",
    "development_042_silver_sandbox.demand_forecast.game_taxonomy_unification_v1_mechanic_mapping",
    "development_042_silver_sandbox.demand_forecast.game_taxonomy_unification_v1_mechanic_unmapped_review",
]

# Set this to the table number you want to review
table_number = 7

if table_number < 1 or table_number > len(review_tables):
    raise ValueError(f"Choose a number from 1 to {len(review_tables)}.")

selected_table = review_tables[table_number - 1]

print(f"Displaying table {table_number}: {selected_table}")
display(spark.table(selected_table))

# COMMAND ----------

table = (
    "development_042_silver_sandbox.demand_forecast."
    "game_taxonomy_unification_v1_theme_reconcile_5_ai"
)

display(spark.table(table).select("ai"))
