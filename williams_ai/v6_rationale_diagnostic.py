"""Fail-closed non-scoring Base-vs-LoRA reasoning receipt audit.

No network, model invocation, trading, gold generation, or training.
This checks evidence completeness, not actual correctness of a book signal.
"""
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import argparse
import json
from pathlib import Path
import re

SCHEMA = "WILLIAMS_V6_IMMEDIATE_REAL_RETRO_4BOOK_CPU_SMOKE_V1"
BOOKS = {"TC1", "NTD", "TC2", "COURSE"}
ROLES = ("matched_base", "candidate_v5")
FAMILIES = {
    "ENTRY": {"LONG_STOP_ENTRY", "SHORT_STOP_ENTRY", "NO_SETUP", "INSUFFICIENT_DATA", "ABSTAIN"},
    "EXIT": {"EXIT_LONG", "EXIT_SHORT", "HOLD", "INSUFFICIENT_DATA", "ABSTAIN"},
    "WAVE": {"WAVE_CONFIRMED", "WAVE_HYPOTHESIS", "WAVE_INVALID", "INSUFFICIENT_DATA", "ABSTAIN"},
}
_HEX = re.compile(r"[a-f0-9]{64}\Z")
_DATETIME = re.compile(r"20\d\d-\d\d-\d\dT\d\d:\d\d:\d\dZ")
_RULE = re.compile(r"\b(?:TC1|NTD|TC2|COURSE)\.[A-Z0-9_.]+\b")


def need(ok: bool, why: str) -> None:
    if not ok:
        raise ValueError("V6_REASON_AUDIT_" + why)


def is_sha(value: object) -> bool:
    return isinstance(value, str) and _HEX.fullmatch(value) is not None


def utc(value: object) -> datetime:
    need(type(value) is str and _DATETIME.fullmatch(value) is not None, "BAD_UTC")
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError as exc:
        raise ValueError("V6_REASON_AUDIT_BAD_UTC") from exc


def audit(data: dict) -> dict:
    """Reject incomparable experiment receipts; mark all unproved dimensions."""
    need(type(data) is dict and data.get("schema") == SCHEMA, "BAD_SCHEMA")
    need(data.get("cases_total") == 4 and
         data.get("no_expert_labels") is True and
         data.get("real_market_book_accuracy") is None and
         data.get("research_execution_allowed") is False and
         data.get("gpu_training_allowed") is False, "FALSE_EXPERT_OR_TRADE_CLAIM")
    for key in ("protocol_sha256", "base_gguf_sha256",
                "v5_original_raw_adapter_sha256_attested",
                "v5_converted_lora_gguf_sha256"):
        need(is_sha(data.get(key)), "UNVERIFIED_" + key.upper())
    need(data["base_gguf_sha256"] != data["v5_converted_lora_gguf_sha256"],
         "BASE_AND_LORA_NOT_DISTINCT")
    sources = data.get("sources")
    need(type(sources) is list and len(sources) == 4, "INCOMPLETE_SOURCE_RECEIPTS")
    by_edition = {}
    for row in sources:
        need(type(row) is dict and row.get("edition") in BOOKS and
             type(row.get("symbol")) is str and is_sha(row.get("chart_sha256")),
             "INVALID_SOURCE_RECEIPT")
        need(row["edition"] not in by_edition, "DUPLICATE_EDITION")
        by_edition[row["edition"]] = row
    need(set(by_edition) == BOOKS, "MISSING_BOOK")
    obs = data.get("observed")
    need(type(obs) is dict and set(obs) == set(ROLES), "MISSING_CONTROL_MODEL")
    indexed = {}
    for role in ROLES:
        rows = obs[role]
        need(type(rows) is list and len(rows) == 4, "INCOMPLETE_PREDICTIONS")
        index = {}
        for row in rows:
            need(type(row) is dict and type(row.get("id")) is str and
                 row.get("edition") in BOOKS and
                 row.get("family") in FAMILIES and
                 row.get("decision") in FAMILIES[row["family"]] and
                 row.get("finish_reason") in {"stop", "length", "error"} and
                 is_sha(row.get("prompt_sha256")) and is_sha(row.get("chart_sha256")) and
                 type(row.get("retrieved_rule_ids")) is list and
                 type(row.get("source_missing")) is bool, "MALFORMED_MODEL_ROW")
            need(row["id"] not in index, "DUPLICATE_CASE_ID")
            source = by_edition[row["edition"]]
            need(row.get("symbol") == source["symbol"] and
                 row["chart_sha256"] == source["chart_sha256"], "CHART_SOURCE_MISMATCH")
            ids = row["retrieved_rule_ids"]
            need(len(ids) == len(set(ids)) and
                 all(type(x) is str and x.startswith(row["edition"] + ".")
                     for x in ids), "CROSS_EDITION_RAG")
            need(row["source_missing"] is (not bool(ids)), "SOURCE_MISSING_FLAG")
            if row["finish_reason"] != "stop":
                need(row["decision"] == "ABSTAIN", "NONSTOP_AS_DECISION")
            if "model_reason" in row:
                need(type(row["model_reason"]) is str and
                     1 <= len(row["model_reason"]) <= 240 and
                     row["finish_reason"] == "stop", "INVALID_SAVED_REASON")
            index[row["id"]] = row
        need({x["edition"] for x in rows} == BOOKS, "FOUR_BOOKS_REQUIRED")
        indexed[role] = index
    need(set(indexed[ROLES[0]]) == set(indexed[ROLES[1]]), "CASE_SETS_DIFFER")
    checks, changed = {r: [] for r in ROLES}, []
    for cid, base in indexed["matched_base"].items():
        trained = indexed["candidate_v5"][cid]
        same = ("edition", "family", "symbol", "cutoff_utc", "chart_sha256",
                "prompt_sha256", "retrieved_rule_ids", "source_missing")
        need(all(base.get(k) == trained.get(k) for k in same),
             "DIFFERENT_INPUTS_BETWEEN_MODELS")
        cutoff = utc(base["cutoff_utc"])
        for role, row in (("matched_base", base), ("candidate_v5", trained)):
            rationale = row.get("model_reason")
            if rationale is None:
                explanation = {"status": "BLOCKED_REASON_NOT_CAPTURED",
                               "unretrieved_rule_assertions": [],
                               "future_timestamp_mentioned": False}
            else:
                asserted = set(_RULE.findall(rationale))
                timestamps = [_date for value in _DATETIME.findall(rationale)
                              if (_date := utc(value)) > cutoff]
                explanation = {
                    "status": "TEXT_PRESENT_NOT_VALIDATED_AS_BOOK_REASONING",
                    "reason_sha256": sha256(rationale.encode("utf-8")).hexdigest(),
                    "reason_length": len(rationale),
                    "rule_ids_asserted": sorted(asserted),
                    "unretrieved_rule_assertions": sorted(
                        asserted - set(row["retrieved_rule_ids"])),
                    "future_timestamp_mentioned": bool(timestamps),
                    "explicit_pdf_page_mention": bool(re.search(
                        r"\b(?:PDF|стр(?:аниц[а-я]*)?|page|p\.)\s*#?\s*\d+",
                        rationale, re.I)),
                }
            checks[role].append({
                "id": cid, "edition": row["edition"],
                "decision": row["decision"],
                "A_source_grounding": "BLOCKED_NO_INDEPENDENT_PAGE_CHECK",
                "B_market_evidence": "BLOCKED_NO_BAR_LEVEL_PRICE_EVIDENCE_IN_RECEIPT",
                "C_causal_consistency": ("FAIL_FUTURE_TIMESTAMP_IN_EXPLANATION"
                    if explanation["future_timestamp_mentioned"]
                    else "PARTIAL_HASH_ONLY_NOT_FULL_CAUSAL_AUDIT"),
                "D_contradiction_detection": ("REQUIRES_POSITION_STATE_FOR_EXIT"
                    if row["family"] == "EXIT" and row["decision"] in
                    {"HOLD", "EXIT_LONG", "EXIT_SHORT"} else
                    "BLOCKED_NO_INDEPENDENT_MARKET_RECONSTRUCTION"),
                "E_signal_validation": "BLOCKED_NO_DETERMINISTIC_SIGNAL_PROOF",
                "F_uncertainty_calibration": "NOT_SCOREABLE_WITHOUT_EXPERT_GOLD",
                "G_counterfactual_test": "BLOCKED_NO_PAIRED_INPUT_PERTURBATIONS",
                "H_explanation_quality": explanation,
            })
        if base["decision"] != trained["decision"]:
            changed.append({"id": cid, "edition": base["edition"],
                            "base": base["decision"], "v5": trained["decision"]})
    return {
        "schema": "WILLIAMS_V6_NONSCORING_EVIDENCE_AUDIT_V1",
        "source_protocol_sha256": data["protocol_sha256"],
        "cases_count": 4, "matched_inputs": True,
        "changed_decisions": changed,
        "reason_text_available": {role: sum(
            r.get("model_reason") is not None for r in indexed[role].values())
            for role in ROLES},
        "checks": checks,
        "independent_accuracy": None,
        "book_reasoning_proven": False,
        "lora_caused_improvement_proven": False,
        "expert_exam_status": "BLOCKED",
        "trading_allowed": False, "training_allowed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(json.loads(args.input.read_text(encoding="utf-8")))
    with args.output.open("x", encoding="utf-8") as file:
        json.dump(result, file, sort_keys=True, ensure_ascii=False, indent=2)
        file.write("\n")


if __name__ == "__main__":
    main()
