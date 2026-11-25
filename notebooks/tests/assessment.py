"""Validate model evidence and project a game assessment into the CSV feature layout."""
import hashlib
import json
import re
from pathlib import Path

import yaml


def read_yaml(path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_config(root):
    root = Path(root)
    config_dir = root / "config" if (root / "config" / "config.yaml").is_file() else root
    settings = read_yaml(config_dir / "config.yaml")
    taxonomy = read_yaml(config_dir / "theme.yaml")
    cfg = settings["run"]
    cfg.setdefault("validate_citations", True)
    if type(cfg["validate_citations"]) is not bool:
        raise ValueError("Use true or false for run.validate_citations")
    contract = settings["output_schema"]
    rules = settings["instructions"]
    catalog = {}
    for kind in ("mechanics", "themes"):
        entries = taxonomy[kind]
        catalog[kind] = {entry["id"]: entry for entry in entries}
        if not entries or len(catalog[kind]) != len(entries):
            raise ValueError(f"Empty catalog or duplicate IDs in {kind}")
        for entry in entries:
            if not re.fullmatch(r"[a-z0-9_]+", entry["id"]) or not entry["description"].strip():
                raise ValueError(f"Invalid category: {entry}")
    cols = list(contract["legacy_columns"])
    if len(set(cols)) != len(cols):
        raise ValueError("Duplicate legacy column names")
    for kind, items in catalog.items():
        for item in items:
            prefix = "mechanic" if kind == "mechanics" else "theme"
            suffixes = ("score", "present") if kind == "mechanics" else ("score",)
            for suffix in suffixes:
                col = f"{prefix}_{item}_{suffix}"
                if col not in cols:
                    cols.append(col)
    expected = {f"theme_{key}_score" for key in catalog["themes"]}
    expected.update(f"mechanic_{key}_{s}" for key in catalog["mechanics"] for s in ("present", "score"))
    legacy_features = {c for c in cols if c.startswith(("theme_", "mechanic_")) and c != "theme_presence_score"}
    retired_ids = taxonomy.get("retired_theme_ids", [])
    retired = [f"theme_{key}_score" for key in retired_ids]
    if len(set(retired)) != len(retired) or set(retired_ids) & catalog["themes"].keys():
        raise ValueError("A retired theme must have a unique ID and cannot also be active")
    if not set(retired) <= set(contract["legacy_columns"]):
        raise ValueError("Each retired theme must correspond to an original CSV column")
    contract["retired_theme_columns"] = retired
    missing = legacy_features - expected - set(retired)
    if missing:
        raise ValueError(f"Missing definitions for output columns: {sorted(missing)}")
    limit = cfg["game_limit"]
    if type(limit) is not int or (limit < 1 and limit != -1) or type(cfg["overwrite"]) is not bool:
        raise ValueError("Use game_limit: -1 or a positive integer, and a boolean overwrite flag")
    if cfg["categories_per_call"] < 1 or cfg["max_output_tokens"] < 1 or cfg["validation_retries"] < 0:
        raise ValueError("Invalid batching or retry settings")
    tables = [cfg[k] for k in ("source_table", "output_table", "attempts_table")]
    if len(set(tables)) != 3 or any(not re.fullmatch(r"\w+\.\w+\.\w+", x) for x in tables):
        raise ValueError("Use three different fully qualified Unity Catalog table names")
    settings = [catalog, rules, contract, cfg["model_endpoint"], cfg["taxonomy_version"],
                cfg["categories_per_call"], cfg["max_output_tokens"], cfg["validate_citations"]]
    code = (root / "assessment.py").read_bytes()
    digest = hashlib.sha256(code + json.dumps(settings, sort_keys=True).encode()).hexdigest()
    return cfg, contract, rules, catalog, cols, digest


def column_types(columns, extra):
    def legacy_type(name):
        if name == "saved_at":
            return "TIMESTAMP"
        if name == "theme_presence_score" or name.endswith("_present"):
            return "INT"
        if name.endswith("_score"):
            return "DOUBLE"
        return "STRING"
    return {**{c: legacy_type(c) for c in columns}, **extra}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def call_checked(spark, cfg, rules, prompt, data, validate, image_path=None):
    """Ask Maverick via Databricks SQL and validate the JSON before accepting it."""
    args = {"model": cfg["model_endpoint"], "max_tokens": cfg["max_output_tokens"]}
    options = "modelParameters => named_struct('temperature', 0.0, 'max_tokens', :max_tokens)"
    if image_path:
        query = (f"SELECT ai_query(:model, :prompt, {options}, files => content) AS answer "
                 "FROM read_files(:image_path, format => 'binaryFile')")
        args["image_path"] = image_path
    else:
        query = f"""SELECT ai_query(:model, :prompt, {options},
                    responseFormat => '{{"type":"json_object"}}') AS answer"""
    instructions = "\n\n".join([rules["system"], prompt, json.dumps(data, ensure_ascii=False)])
    args["prompt"] = instructions
    for attempt in range(cfg["validation_retries"] + 1):
        row = spark.sql(query, args=args).first()
        if row is None:
            raise ValueError(f"No request row returned for {image_path or 'text input'}")
        try:
            body = (row["answer"] or "").strip()
            body = re.sub(r"^\x60\x60\x60(?:json)?\s*|\s*\x60\x60\x60$", "", body, flags=re.I)
            return validate(json.loads(body))
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            if attempt == cfg["validation_retries"]:
                raise ValueError(f"Model output failed validation: {exc}") from exc
            args["prompt"] = f"{instructions}\n\nRegenerate the complete JSON. Validation issue: {str(exc)[:400]}"


def collect_evidence(spark, cfg, rules, pages):
    sources = {"filename": pages[0]["file_name"]}
    for page in pages:
        p = page["page_number"]
        ref = f"p{p}.text"
        sources[ref] = page["page_text"] or ""
        path = page["page_image_path"]
        require(isinstance(path, str) and path.strip(), f"Missing rendered image path for page {p}")

        def parse_visual(raw):
            observations = raw["observations"]
            require(isinstance(observations, list) and all(isinstance(x, str) and x.strip() for x in observations),
                    "Expected a list of visual descriptions")
            return observations
        page_data = {"file_name": sources["filename"], "page": p, "text": sources[ref]}
        visuals = call_checked(spark, cfg, rules, rules["page"], page_data, parse_visual, image_path=path)
        for index, detail in enumerate(visuals, start=1):
            sources[f"p{p}.v{index}"] = detail
    return sources


def citations(values, sources, allow_filename=False, *, validate=True):
    if not validate:
        items = values if isinstance(values, list) else ([] if values is None else [values])
        result = []
        for item in items:
            ref = item.get("ref") if isinstance(item, dict) else item
            quote = item.get("quote") if isinstance(item, dict) else None
            ref_text = ref if isinstance(ref, str) else ""
            match = re.fullmatch(r"p([0-9]{1,9})\.(text|v[0-9]+)", ref_text)
            kind = "unknown"
            page = None
            if match:
                page = int(match[1])
                kind = "pdf_text" if match[2] == "text" else "page_image"
            elif ref_text and ref_text in ("filename", sources.get("filename")):
                kind = "filename"
            if quote is None and kind == "page_image":
                quote = sources.get(ref_text)
            if quote is None and kind == "filename":
                quote = sources.get("filename")
            result.append({"page_number": page, "source_type": kind, "model_citation": item,
                           "excerpt_or_visual_description": quote, "review_status": "unchecked"})
        return result

    require(isinstance(values, list), "Evidence must be a list")
    result = []
    for item in values:
        ref = item["ref"]
        require(ref in sources, f"Unknown source {ref}")
        is_filename = ref == "filename"
        is_visual = ".v" in ref
        require(allow_filename or not is_filename, "Filename evidence only supports a franchise lead")
        excerpt = sources[ref] if is_visual else item.get("quote")
        require(isinstance(excerpt, str) and excerpt.strip(), f"Missing excerpt for {ref}")
        normalize = lambda s: " ".join(s.split()).casefold()
        require(normalize(excerpt) in normalize(sources[ref]), f"Excerpt not found at {ref}: {excerpt[:80]}")
        number = None if is_filename else int(ref[1:].split(".")[0])
        kind = "filename" if is_filename else ("page_image" if is_visual else "pdf_text")
        result.append({"page_number": number, "source_type": kind, "excerpt_or_visual_description": excerpt})
    return result


SCALAR_FIELDS = ("game_format", "reel_configuration", "number_of_reels", "symbol_count")
MONEY_FIELDS = ("denomination", "minimum_bet", "maximum_bet", "cost_to_cover")
EMPTY = {"value": None, "evidence": []}
MONEY = {**EMPTY, "currency": None}
FACT_TEMPLATE = {
    **{key: EMPTY for key in SCALAR_FIELDS}, "win_configurations": [],
    "denomination_configurations": [{key: MONEY for key in MONEY_FIELDS}],
    "overall_minimum_bet": {**MONEY, "denomination": None},
    "overall_maximum_bet": {**MONEY, "denomination": None},
    "persistence": {"status": "unknown", "scope": "unknown", "definition": "", "evidence": []},
    "perceived_persistence": {"score": 0, "definition": "", "evidence": []},
    "art_style": {"value": "unknown", "evidence": []}, "franchise_candidate": EMPTY,
    "franchise_status": "unknown", "extraction_notes": [],
}


def validated_facts(raw, sources, *, validate_citations=True):
    require(set(FACT_TEMPLATE) <= set(raw), "Missing fact fields")
    def fact(obj, optional=(), filename=False):
        value = obj["value"]
        require(value is None or isinstance(value, (int, float, str)) and str(value).strip(),
                "Expected a scalar value or null")
        out = {"value": None if value is None else str(value),
               "evidence": citations(obj.get("evidence"), sources, allow_filename=filename, validate=validate_citations)}
        if validate_citations:
            require(out["value"] in (None, "unknown") or out["evidence"], f"Missing evidence for {value}")
        for key in optional:
            out[key] = obj[key]
            require(out[key] is None or isinstance(out[key], str), f"Invalid {key}")
        return out

    result = {key: fact(raw[key]) for key in SCALAR_FIELDS}
    for key in ("number_of_reels", "symbol_count"):
        value = result[key]["value"]
        require(value is None or (value.isdigit() and int(value) > 0), f"Expected positive count or null in {key}")
    result["win_configurations"] = []
    for win in raw["win_configurations"]:
        require(win["kind"] in ("ways", "paylines", "other") and win["scope"] in ("base", "option", "bonus"),
                "Unexpected win kind or scope")
        require(type(win["count"]) is int and win["count"] > 0, "Expected a positive integer win count")
        evidence = citations(win.get("evidence"), sources, validate=validate_citations)
        if validate_citations:
            require(evidence, "Every win configuration needs a page citation")
        result["win_configurations"].append({k: win[k] for k in ("kind", "scope", "count")} | {"evidence": evidence})
    result["denomination_configurations"] = [
        {k: fact(row[k], ("currency",)) for k in MONEY_FIELDS} for row in raw["denomination_configurations"]]
    for key in ("overall_minimum_bet", "overall_maximum_bet"):
        result[key] = fact(raw[key], ("currency", "denomination"))
    for key in ("persistence", "perceived_persistence"):
        obj = raw[key]
        evidence = citations(obj.get("evidence"), sources, validate=validate_citations)
        require(isinstance(obj["definition"], str), f"Missing explanation for {key}")
        if key == "persistence":
            status = obj["status"]
            require(status in ("present", "explicitly_absent", "unknown"), "Unknown persistence status")
            score = {"present": 1, "explicitly_absent": 0, "unknown": None}[status]
            if validate_citations:
                require(status == "unknown" or evidence, "Persistence decision needs evidence")
            scope = obj["scope"]
            require(scope in ("base_spins", "bonus_rounds", "sessions", "multiple", "unknown"), "Unknown carryover scope")
            out = {"status": status, "scope": scope}
        else:
            score = obj["score"]
            require(type(score) is int and 0 <= score <= 5, "Perceived persistence score must be 0–5")
            if validate_citations:
                require(score == 0 or evidence, "Perceived progress needs evidence")
            out = {}
        result[key] = {**out, "score": score, "definition": obj["definition"], "evidence": evidence}
    style = result["art_style"] = fact(raw["art_style"])
    require(style["value"] in ("realistic", "cartoon", "unknown"), "Unexpected art style")
    if validate_citations:
        require(style["value"] == "unknown" or any(e["source_type"] == "page_image" for e in style["evidence"]),
                "An art style decision requires page-image evidence")
    franchise = result["franchise_candidate"] = fact(raw["franchise_candidate"], filename=True)
    status = result["franchise_status"] = raw["franchise_status"]
    require(status in ("unknown", "candidate", "confirmed"), "Unexpected franchise status")
    require(status == "unknown" or franchise["value"], "Missing franchise name")
    if validate_citations:
        require(status != "confirmed" or any(e["page_number"] is not None for e in franchise["evidence"]),
                "A confirmed franchise requires page evidence")
    notes = raw["extraction_notes"]
    require(isinstance(notes, list) and all(isinstance(x, str) for x in notes), "Expected extraction notes array")
    result["extraction_notes"] = notes
    if not validate_citations:
        result["citation_review_status"] = "unchecked"
    return result


def assess_catalog(spark, cfg, rules, catalog, sources, facts):
    assessed = {}
    for kind, definitions in catalog.items():
        results = {}
        entries = list(definitions.values())
        batch_size = cfg["categories_per_call"]
        for start in range(0, len(entries), batch_size):
            batch = entries[start:start + batch_size]
            expected = {entry["id"] for entry in batch}
            def validate(raw):
                rows = raw["items"]
                require(len(rows) == len(expected) and {r["id"] for r in rows} == expected,
                        "Return each requested category exactly once")
                out = {}
                for row in rows:
                    key, status, score = row["id"], row["status"], row["score"]
                    require(status in ("present", "not_evidenced", "uncertain"), "Unexpected category status")
                    evidence = citations(row.get("evidence"), sources, validate=cfg["validate_citations"])
                    if status == "present":
                        bound = 1 if kind == "mechanics" else 5
                        require(type(score) is int and 1 <= score <= bound, f"Invalid positive {key}")
                        if cfg["validate_citations"]:
                            require(evidence, f"Missing evidence for {key}")
                    else:
                        require(score is None, f"Use null for a non-positive {key}")
                    if kind == "themes" and key == "pop_culture":
                        franchise_status = facts["franchise_status"]
                        require((franchise_status == "confirmed") == (status == "present"),
                                "Pop culture must agree with the franchise decision")
                        require(franchise_status != "candidate" or status == "uncertain",
                                "Preserve a candidate franchise for review")
                    rationale = row["rationale"]
                    require(isinstance(rationale, str) and (status != "present" or rationale.strip()),
                            f"Missing rationale for {key}")
                    out[key] = {"id": key, "name": definitions[key]["name"],
                                "definition": definitions[key]["description"], "status": status,
                                "score": score, "rationale": rationale, "evidence": evidence}
                return out
            instructions = {"kind": kind, "assess": batch, "franchise_status": facts["franchise_status"],
                            "franchise": facts["franchise_candidate"], "sources": sources}
            if any(entry["id"].startswith("other_") for entry in batch):
                instructions["entire_catalog_for_other"] = {k: v["description"] for k, v in definitions.items()}
            result = call_checked(spark, cfg, rules, rules["evidence"] + rules["category"],
                                  instructions, validate)
            results.update(result)
        assessed[kind] = results
    return assessed


def assess_game(spark, cfg, rules, catalog, pages):
    require(pages, "No pages for this game")
    numbers = [int(p["page_number"]) for p in pages]
    require(numbers == list(range(1, len(numbers) + 1)), f"Duplicate or missing page numbers: {numbers}")
    for page in pages:
        if page["parser_errors_json"] and json.loads(page["parser_errors_json"]):
            raise ValueError(f"Parser reported errors for {page['file_name']}")
    sources = collect_evidence(spark, cfg, rules, pages)
    facts = call_checked(spark, cfg, rules, rules["evidence"] + rules["facts"],
                         {"template": FACT_TEMPLATE, "sources": sources},
                         lambda raw: validated_facts(raw, sources, validate_citations=cfg["validate_citations"]))
    return facts, assess_catalog(spark, cfg, rules, catalog, sources, facts)


def project(facts, categories, retired_theme_columns=()):
    def legacy_values(value):
        if isinstance(value, list):
            return [legacy_values(item) for item in value]
        if isinstance(value, dict):
            return {k: str(v) if k in {"score", "page_number"} and v is not None else legacy_values(v)
                    for k, v in value.items() if k != "scope"}
        return value

    def encoded(value):
        return json.dumps(legacy_values(value), ensure_ascii=False, separators=(",", ":"))
    row = {k: encoded(facts[k]) for k in (*SCALAR_FIELDS, "overall_minimum_bet", "overall_maximum_bet",
                                          "denomination_configurations", "persistence", "perceived_persistence",
                                          "extraction_notes")}
    configs = [w for w in facts["win_configurations"] if w["scope"] == "base"]
    values = {(w["kind"], w["count"]) for w in configs}
    chosen = configs[0] if len(values) == 1 else None
    row["paylines_or_ways_count"] = encoded({"value": str(chosen["count"]), "evidence": chosen["evidence"]}
                                            if chosen else EMPTY)
    row["win_count_type"] = chosen["kind"] if chosen else None
    row["base_win_count"] = chosen["count"] if chosen else None
    value = facts["number_of_reels"]["value"]
    row["base_reel_count"] = int(value) if value is not None else None
    row["persistence_present"] = facts["persistence"]["score"]
    row["persistence_scope"] = facts["persistence"]["scope"]
    row["perceived_persistence_score"] = facts["perceived_persistence"]["score"]
    for kind, items in categories.items():
        positive = [v for v in items.values() if v["status"] == "present"]
        row[kind] = encoded([{k: v[k] for k in ("name", "definition", "score", "evidence")} for v in positive])
        for key, item in items.items():
            if kind == "mechanics":
                row[f"mechanic_{key}_score"] = float(item["score"]) if item["score"] else None
                row[f"mechanic_{key}_present"] = 1 if item["status"] == "present" else None
            else:
                row[f"theme_{key}_score"] = float(item["score"]) if item["score"] else None
        if kind == "themes":
            row["theme_presence_score"] = max((item["score"] for item in positive), default=0)
    row.update(art_style=facts["art_style"]["value"], art_style_evidence=encoded(facts["art_style"]["evidence"]),
               franchise_candidate=facts["franchise_candidate"]["value"], franchise_status=facts["franchise_status"])
    row["assessment_yaml"] = yaml.safe_dump({**facts, **categories}, allow_unicode=True, sort_keys=False)
    row.update(dict.fromkeys(retired_theme_columns))
    return row
