"""Validate model evidence and project a game assessment into the CSV feature layout."""
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import re
from pathlib import Path
from time import perf_counter

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
    cfg.setdefault("allow_partial_results", False)
    if type(cfg["allow_partial_results"]) is not bool:
        raise ValueError("Use true or false for run.allow_partial_results")
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
    cfg.setdefault("inference_rows_per_query", 8)
    if type(cfg["inference_rows_per_query"]) is not int or cfg["inference_rows_per_query"] < 1:
        raise ValueError("Use a positive integer for run.inference_rows_per_query")
    tables = [cfg[k] for k in ("source_table", "output_table", "attempts_table")]
    if len(set(tables)) != 3 or any(not re.fullmatch(r"\w+\.\w+\.\w+", x) for x in tables):
        raise ValueError("Use three different fully qualified Unity Catalog table names")
    settings = [catalog, rules, contract, cfg["model_endpoint"], cfg["taxonomy_version"],
                cfg["categories_per_call"], cfg["max_output_tokens"], cfg["validate_citations"], cfg["allow_partial_results"]]
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


def read_reply(text):
    """Accept an isolated JSON object or one fenced JSON object with surrounding prose."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        fenced = re.fullmatch(r"[^{}\[\]\x60]*\x60{3}(?:json)?\s*(.*?)\s*\x60{3}[^{}\[\]\x60]*", text,
                              flags=re.IGNORECASE | re.DOTALL)
        if fenced:
            return json.loads(fenced[1])
        raise


@dataclass
class ModelRequest:
    stage: str
    prompt: str
    validate: Callable
    fallback: object
    image_path: str | None = None


def model_prompt(rules, instruction, data):
    return "\n\n".join([rules["system"], instruction, json.dumps(data, ensure_ascii=False)])


def query_answers(spark, cfg, requests, prompts):
    """Run one SQL action with one independent Maverick response per input row."""
    args = {"model": cfg["model_endpoint"], "max_tokens": cfg["max_output_tokens"]}
    options = "modelParameters => named_struct('temperature', 0.0, 'max_tokens', :max_tokens)"
    with_images = requests[0].image_path is not None
    if with_images:
        options += ", files => content"
    else:
        options += """, responseFormat => '{"type":"json_object"}'"""

    # Keep the original single-request query for facts, isolated retries and serial mode.
    if len(requests) == 1:
        args["prompt"] = prompts[0]
        query = f"SELECT ai_query(:model, :prompt, {options}) AS answer"
        if with_images:
            args["image_path"] = requests[0].image_path
            query += " FROM read_files(:image_path, format => 'binaryFile')"
        row = spark.sql(query, args=args).first()
        require(row is not None, f"No response row for {requests[0].stage}")
        return [row["answer"]]

    parts = []
    for index, (request, prompt) in enumerate(zip(requests, prompts)):
        args[f"prompt_{index}"] = prompt
        if with_images:
            args[f"path_{index}"] = request.image_path
            parts.append(f"SELECT {index} AS request_id, :prompt_{index} AS prompt, content "
                         f"FROM read_files(:path_{index}, format => 'binaryFile')")
        else:
            parts.append(f"({index}, :prompt_{index})")
    if with_images:
        source = "(" + " UNION ALL ".join(parts) + ") AS requests"
    else:
        source = "VALUES " + ", ".join(parts) + " AS requests(request_id, prompt)"
    query = f"SELECT request_id, ai_query(:model, prompt, {options}) AS answer FROM {source}"
    rows = spark.sql(query, args=args).collect()
    by_id = {row["request_id"]: row["answer"] for row in rows}
    require(len(rows) == len(requests) and set(by_id) == set(range(len(requests))),
            "Expected one Maverick response for each request in the SQL batch")
    return [by_id[index] for index in range(len(requests))]


def checked_requests(spark, cfg, requests, review, stage):
    """Batch independent requests, then retry only replies that failed validation."""
    size = cfg.get("inference_rows_per_query", 8)
    results = [None] * len(requests)
    for start in range(0, len(requests), size):
        stop = min(start + size, len(requests))
        pending = {i: requests[i].prompt for i in range(start, stop)}
        for attempt in range(cfg["validation_retries"] + 1):
            current = [requests[i] for i in pending]
            print(f"  {stage} {start + 1}–{stop}/{len(requests)} | "
                  f"{len(current)} request(s) | attempt {attempt + 1}", flush=True)
            answers = query_answers(spark, cfg, current, list(pending.values()))
            retry = {}
            for index, answer in zip(pending, answers):
                request = requests[index]
                try:
                    results[index] = request.validate(read_reply(answer or ""))
                except (ValueError, KeyError, TypeError, AttributeError) as exc:
                    if attempt < cfg["validation_retries"]:
                        retry[index] = (f"{request.prompt}\n\nYour previous response failed: {str(exc)[:400]}. "
                                        "Return exactly one complete JSON object with double-quoted keys and commas "
                                        "between items. Do not append text or a second object.")
                    elif review is None:
                        raise ValueError(f"{request.stage} failed validation: {exc}") from exc
                    else:
                        review.append({"field": request.stage, "reason": str(exc), "last_reply": answer})
                        results[index] = request.fallback
            if not retry:
                break
            pending = retry
    return results


def call_checked(spark, cfg, rules, prompt, data, validate, image_path=None, *, review=None, stage=None, fallback=None):
    label = stage or "Model output"
    request = ModelRequest(label, model_prompt(rules, prompt, data), validate, fallback, image_path)
    return checked_requests(spark, cfg, [request], review, label)[0]


def collect_evidence(spark, cfg, rules, pages, review=None):
    def parse_visual(raw):
        observations = raw["observations"]
        require(isinstance(observations, list) and all(isinstance(x, str) and x.strip() for x in observations),
                "Expected a list of visual descriptions")
        return observations

    sources = {"filename": pages[0]["file_name"]}
    requests = []
    for page in pages:
        p = page["page_number"]
        path = page["page_image_path"]
        require(isinstance(path, str) and path.strip(), f"Missing rendered image path for page {p}")
        data = {"file_name": sources["filename"], "page": p, "text": page["page_text"] or ""}
        requests.append(ModelRequest(f"page {p} image observations", model_prompt(rules, rules["page"], data),
                                     parse_visual, [], path))

    visuals = checked_requests(spark, cfg, requests, review, "Page images")
    for page, observations in zip(pages, visuals):
        p = page["page_number"]
        sources[f"p{p}.text"] = page["page_text"] or ""
        for index, detail in enumerate(observations, start=1):
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


def empty_facts(validate_citations):
    result = deepcopy(FACT_TEMPLATE)
    result["denomination_configurations"] = []
    result["perceived_persistence"]["score"] = None
    result["persistence"]["score"] = None
    if not validate_citations:
        result["citation_review_status"] = "unchecked"
    return result


def positive_count(value, maximum):
    if type(value) is float and value.is_integer():
        value = int(value)
    if isinstance(value, str) and re.fullmatch(r"[0-9]{1,19}", value.strip()):
        value = int(value.strip())
    require(type(value) is int and 0 < value <= maximum, f"Expected a positive integer count, received {value!r}")
    return value


def review_value(field, value, parser, fallback, review):
    try:
        return parser(value)
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        if review is None:
            raise
        review.append({"field": field, "reason": str(exc), "model_value": value})
        return fallback


def validated_facts(raw, sources, *, validate_citations=True, review=None):
    require(isinstance(raw, dict) and any(k in raw for k in FACT_TEMPLATE),
            "Expected an object with named fact fields")
    missing = sorted(set(FACT_TEMPLATE) - raw.keys())
    if missing and review is None:
        raise ValueError(f"Missing fact fields: {missing}")

    def fact(obj, optional=(), filename=False, path="fact"):
        require(isinstance(obj, dict), "Expected a value/evidence object")
        value = obj["value"]
        require(value is None or type(value) in (int, float, str) and str(value).strip(), "Invalid scalar value")
        evidence = citations(obj.get("evidence"), sources, allow_filename=filename, validate=validate_citations)
        if validate_citations:
            require(value in (None, "unknown") or evidence, f"Missing evidence for {value}")
        out = {"value": str(value) if value is not None else None, "evidence": evidence}
        for key in optional:
            v = obj.get(key)
            if key == "denomination" and type(v) in (int, float):
                v = str(v)
            if v is not None and not isinstance(v, str):
                if review is None:
                    raise ValueError(f"Invalid {key}")
                review.append({"field": f"{path}.{key}", "reason": f"Invalid {key}", "model_value": v})
                v = None
            out[key] = v
        return out

    def scalar(obj, key):
        out = fact(obj)
        if key in ("number_of_reels", "symbol_count") and out["value"] is not None:
            out["value"] = str(positive_count(out["value"], 2**31 - 1))
        return out

    def win(obj):
        require(isinstance(obj, dict), "Expected a win configuration")
        require(obj.get("kind") in ("ways", "paylines", "other") and obj.get("scope") in ("base", "option", "bonus"),
                "Unexpected win kind or scope")
        count = positive_count(obj.get("count"), 2**63 - 1)
        evidence = citations(obj.get("evidence"), sources, validate=validate_citations)
        if validate_citations:
            require(evidence, "Every win configuration needs a page citation")
        return {"kind": obj["kind"], "scope": obj["scope"], "count": count, "evidence": evidence}

    def denomination(obj):
        require(isinstance(obj, dict) and any(k in obj for k in MONEY_FIELDS), "Expected a denomination configuration")
        return {k: fact(obj.get(k, MONEY), ("currency",), path=f"denomination_configurations.{k}")
                for k in MONEY_FIELDS}

    def persistence(obj, real):
        evidence = citations(obj.get("evidence"), sources, validate=validate_citations)
        definition = obj["definition"]
        require(isinstance(definition, str), "Expected a progression explanation")
        if real:
            status, scope = obj["status"], obj["scope"]
            scores = {"present": 1, "explicitly_absent": 0, "unknown": None}
            require(status in scores, "Unknown persistence status")
            require(scope in ("base_spins", "bonus_rounds", "sessions", "multiple", "unknown"), "Unknown carryover scope")
            if validate_citations:
                require(status == "unknown" or evidence, "Persistence decision needs evidence")
            return dict(status=status, scope=scope, score=scores[status], definition=definition, evidence=evidence)
        score = obj["score"]
        require(type(score) is int and 0 <= score <= 5, "Perceived persistence score must be 0–5")
        if validate_citations:
            require(score == 0 or evidence, "Perceived progress needs evidence")
        return dict(score=score, definition=definition, evidence=evidence)

    def style(obj):
        out = fact(obj)
        require(out["value"] in ("realistic", "cartoon", "unknown"), "Unexpected art style")
        if validate_citations:
            require(out["value"] == "unknown" or any(e["source_type"] == "page_image" for e in out["evidence"]),
                    "An art style decision requires page-image evidence")
        return out

    def franchise_status(status):
        require(status in ("unknown", "candidate", "confirmed"), "Unexpected franchise status")
        candidate = result["franchise_candidate"]
        require(status == "unknown" or candidate["value"], "Missing franchise name")
        if validate_citations:
            require(status != "confirmed" or any(e["page_number"] is not None for e in candidate["evidence"]),
                    "A confirmed franchise requires page evidence")
        return status

    def notes(values):
        require(isinstance(values, list) and all(isinstance(x, str) for x in values), "Expected extraction notes array")
        return list(values)

    result = empty_facts(validate_citations)
    if missing:
        review.append({"field": "facts", "reason": "Missing fields: " + ", ".join(missing), "missing": missing})
    def field(key, parser):
        if key in raw:
            result[key] = review_value(key, raw[key], parser, result[key], review)

    for key in SCALAR_FIELDS:
        field(key, lambda v: scalar(v, key))
    for key, parser in (("win_configurations", win), ("denomination_configurations", denomination)):
        if key not in raw:
            continue
        values = raw[key]
        if not isinstance(values, list):
            if review is None:
                raise ValueError(f"Expected a list in {key}")
            review.append({"field": key, "reason": "Expected a list", "model_value": values})
            continue
        for i, value in enumerate(values):
            parsed = review_value(f"{key}[{i}]", value, parser, None, review)
            if parsed is not None:
                result[key].append(parsed)
    for key in ("overall_minimum_bet", "overall_maximum_bet"):
        field(key, lambda v: fact(v, ("currency", "denomination"), path=key))
    field("persistence", lambda v: persistence(v, True))
    field("perceived_persistence", lambda v: persistence(v, False))
    field("art_style", style)
    field("franchise_candidate", lambda v: fact(v, filename=True))
    field("franchise_status", franchise_status)
    field("extraction_notes", notes)
    return result


def category_request(cfg, rules, kind, definitions, batch, sources, facts, review):
    def validate_item(row):
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
            franchise = facts["franchise_status"]
            require((franchise == "confirmed") == (status == "present"),
                    f"Pop Culture {status} conflicts with franchise {franchise}")
            require(franchise != "candidate" or status == "uncertain", "Candidate franchise needs review")
        rationale = row["rationale"]
        require(isinstance(rationale, str) and (status != "present" or rationale.strip()),
                f"Missing rationale for {key}")
        return dict(id=key, name=definitions[key]["name"], definition=definitions[key]["description"],
                    status=status, score=score, rationale=rationale, evidence=evidence)

    blank = {d["id"]: dict(id=d["id"], name=d["name"], definition=d["description"], score=None,
                           status="not_assessed", rationale="See review_issues.", evidence=[]) for d in batch}

    def validate(raw):
        rows = raw["items"]
        require(isinstance(rows, list) and rows, "Expected non-empty category items")
        by_id = {}
        for row in rows:
            key = row.get("id") if isinstance(row, dict) else None
            if not isinstance(key, str) or key not in blank:
                if review is None:
                    raise ValueError(f"Unexpected category ID {key}")
                review.append({"field": kind, "reason": "Unexpected category", "model_value": row})
                continue
            by_id.setdefault(key, []).append(row)
        out = blank.copy()
        for key in blank:
            matches = by_id.get(key, [])
            if len(matches) != 1:
                if review is None:
                    raise ValueError(f"Expected one {key}, received {len(matches)}")
                review.append({"field": f"{kind}.{key}", "reason": f"Expected one item, received {len(matches)}",
                               "model_value": matches})
            else:
                out[key] = review_value(f"{kind}.{key}", matches[0], validate_item, blank[key], review)
        return out

    instructions = {"kind": kind, "assess": batch, "franchise_status": facts["franchise_status"],
                    "franchise": facts["franchise_candidate"], "sources": sources}
    if any(entry["id"].startswith("other_") for entry in batch):
        instructions["entire_catalog_for_other"] = {k: v["description"] for k, v in definitions.items()}
    prompt = model_prompt(rules, rules["evidence"] + rules["category"], instructions)
    return ModelRequest(f"{kind}: {', '.join(blank)}", prompt, validate, blank)


def assess_catalog(spark, cfg, rules, catalog, sources, facts, review=None):
    requests = []
    kinds = []
    size = cfg["categories_per_call"]
    for kind, definitions in catalog.items():
        entries = list(definitions.values())
        for start in range(0, len(entries), size):
            batch = entries[start:start + size]
            requests.append(category_request(cfg, rules, kind, definitions, batch, sources, facts, review))
            kinds.append(kind)
    assessed = {kind: {} for kind in catalog}
    for kind, result in zip(kinds, checked_requests(spark, cfg, requests, review, "Category batches")):
        assessed[kind].update(result)
    return assessed


def assess_game(spark, cfg, rules, catalog, pages):
    require(pages, "No pages for this game")
    numbers = [int(p["page_number"]) for p in pages]
    require(numbers == list(range(1, len(numbers) + 1)), f"Duplicate or missing page numbers: {numbers}")
    for page in pages:
        if page["parser_errors_json"] and json.loads(page["parser_errors_json"]):
            raise ValueError(f"Parser reported errors for {page['file_name']}")
    review = [] if cfg.get("allow_partial_results", False) else None
    started = perf_counter()
    print(f"{pages[0]['file_name']} | assessing {len(pages)} page(s)", flush=True)
    sources = collect_evidence(spark, cfg, rules, pages, review)
    empty = empty_facts(cfg["validate_citations"])
    facts = call_checked(spark, cfg, rules, rules["evidence"] + rules["facts"],
                         {"template": FACT_TEMPLATE, "sources": sources},
                         lambda raw: validated_facts(raw, sources, validate_citations=cfg["validate_citations"],
                                                     review=review),
                         review=review, stage="facts", fallback=empty)
    categories = assess_catalog(spark, cfg, rules, catalog, sources, facts, review)
    if review:
        has_decisions = any(item["status"] != "not_assessed"
                            for items in categories.values() for item in items.values())
        if not has_decisions and facts == empty:
            raise ValueError(f"No usable game assessment; first issue in {review[0]['field']}: {review[0]['reason']}")
        facts["review_issues"] = review
        facts["assessment_status"] = "partial"
        facts["extraction_notes"].extend(f"{issue['field']}: {issue['reason']}" for issue in review)
    print(f"  Assessment finished in {perf_counter() - started:.1f}s", flush=True)
    return facts, categories


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
    for issue in facts.get("review_issues", []):
        field = issue["field"]
        if field == "win_configurations" or field.startswith("win_configurations["):
            raw = issue.get("model_value")
            if not isinstance(raw, dict) or raw.get("scope") not in ("bonus", "option"):
                chosen = None
                break
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
            highest = max((item["score"] for item in positive), default=0)
            incomplete = any(item["status"] == "not_assessed" for item in items.values())
            row["theme_presence_score"] = None if incomplete and highest < 5 else highest
    row.update(art_style=facts["art_style"]["value"], art_style_evidence=encoded(facts["art_style"]["evidence"]),
               franchise_candidate=facts["franchise_candidate"]["value"], franchise_status=facts["franchise_status"])
    row["assessment_yaml"] = yaml.safe_dump({**facts, **categories}, allow_unicode=True, sort_keys=False)
    row.update(dict.fromkeys(retired_theme_columns))
    return row
