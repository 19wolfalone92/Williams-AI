"""Fail-closed aggregation of actual matched Qwen Base + v5 LoRA 10x10 CPU shards.

No independent trading correctness or accuracy can be inferred from these rows.
"""
from __future__ import annotations
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import argparse

SCHEMA = "WILLIAMS_V6_HIST100_ACTUAL_MATCHED_BASE_V5_SHARD_CPU_NO_EXPERT_GOLD_V1"
OUT_SCHEMA = "WILLIAMS_V6_HIST100_VERIFIED_MATCHED_REAL_MODEL_ROWS_NO_EXPERT_GOLD_V1"
ROLES = ("matched_base", "candidate_v5")
BOOKS = ("TC1", "NTD", "TC2", "COURSE")
FILES = 10


def need(ok, message):
    if not ok:
        raise ValueError("HIST100_AGGREGATION_" + message)


def aggregate(directory: Path):
    inputs = sorted(directory.glob("hist100_shard_*.json"))
    need(len(inputs) == FILES, "MISSING_OR_EXTRA_SHARDS")
    full = {role: [] for role in ROLES}
    used = set()
    original_plan_sha = protocol = base = adapter = None
    for file in inputs:
        data = json.loads(file.read_bytes())
        need(data["schema"] == SCHEMA and data["complete"] is True and
             data["total_original_cases"] == 100 and data["cases_total"] == 10 and
             data["completed_model_rows"] == 20 and
             data["no_expert_labels"] is True and
             data["real_market_book_accuracy"] is None and
             data["research_execution_allowed"] is False and
             data["prospective_exam"] is False and
             data["training_overlap_audited"] is False, "MISLEADING_SHARD_RECEIPT")
        i = data["shard_index"]
        need(type(i) is int and 0 <= i < FILES and
             data["shard_count"] == FILES and i not in used,
             "DUPLICATE_OR_WRONG_SHARD_INDEX")
        used.add(i)
        identity = (data["market_selection_plan_sha256"], data["protocol_sha256"],
                    data["base_gguf_sha256"], data["v5_converted_lora_gguf_sha256"])
        need(all(isinstance(x, str) and len(x) == 64 for x in identity),
             "UNPINNED_MODEL_OR_PLAN")
        if original_plan_sha is None:
            original_plan_sha, protocol, base, adapter = identity
        else:
            need(identity == (original_plan_sha, protocol, base, adapter),
                 "NON_IDENTICAL_INPUT_OR_MODEL_PROTOCOL")
        rows = data["observed"]
        need(set(rows) == set(ROLES) and
             all(len(rows[k]) == 10 for k in ROLES), "WRONG_MODEL_ROW_COUNTS")
        for pos, (b, v) in enumerate(zip(rows["matched_base"], rows["candidate_v5"], strict=True)):
            need(b["id"] == v["id"], "MISMATCHED_CASE")
            need(all(b[key] == v[key] for key in
                     ("edition", "family", "symbol", "cutoff_utc",
                      "chart_sha256", "prompt_sha256", "retrieved_rule_ids",
                      "source_missing")), "MISMATCHED_MODEL_INPUT")
            need(b["edition"] in BOOKS and
                 b["finish_reason"] in {"stop", "length", "error"} and
                 v["finish_reason"] in {"stop", "length", "error"},
                 "BAD_STOP_REASON_OR_BOOK")
            # The deterministic slot is encoded in HIST100-{symbol-index:02d}-{date-index}.
            suffix = b["id"].split("-")
            need(len(suffix) == 3 and suffix[0] == "HIST100" and
                 1 <= int(suffix[1]) <= 25 and
                 1 <= int(suffix[2]) <= 4 and
                 ((int(suffix[1]) - 1) * 4 + int(suffix[2]) - 1) % FILES == i,
                 "CASE_SHARD_SLOT_CHANGED")
            need((b["id"], b["chart_sha256"]) not in
                 {(r["id"], r["chart_sha256"]) for r in full["matched_base"]},
                 "DUPLICATE_CASE")
        for role in ROLES:
            full[role].extend(rows[role])
    need(used == set(range(FILES)), "MISSING_SHARD_INDICES")
    for role in ROLES:
        need(len(full[role]) == 100 and len({r["id"] for r in full[role]}) == 100,
             "NOT_100_REAL_MODEL_ROWS")
        need(all(sum(r["edition"] == book for r in full[role]) == 25 for book in BOOKS),
             "UNBALANCED_BOOKS")
    for role in ROLES:
        full[role].sort(key=lambda x: x["id"])
    counts = {}
    for role in ROLES:
        rows = full[role]
        counts[role] = {
            "total": len(rows),
            "decisions": dict(Counter(r["decision"] for r in rows)),
            "finish_reasons": dict(Counter(r["finish_reason"] for r in rows)),
            "decisions_by_book": {book: dict(Counter(r["decision"] for r in rows if r["edition"] == book)) for book in BOOKS},
            "reasons_at_length_240": sum(len(r.get("model_reason", "")) == 240 for r in rows),
            "decision_reason_label_contradictions": sum(
                r.get("model_reason", "").strip().upper() == "INSUFFICIENT_DATA" and
                r["decision"] not in {"INSUFFICIENT_DATA", "ABSTAIN"}
                for r in rows),
        }
    diffs = [r["id"] for r, v in zip(full["matched_base"], full["candidate_v5"], strict=True)
             if r["decision"] != v["decision"]]
    return {
        "schema": OUT_SCHEMA,
        "complete": True,
        "real_chart_count": 100,
        "real_base_inference_rows": 100,
        "real_v5_inference_rows": 100,
        "matched_inputs": True,
        "official_archive_sha_verified_upstream": True,
        "gold_labels": None,
        "accuracy": None,
        "trading_qualification": False,
        "model_train_overlap_independently_audited": False,
        "prospective_holdout": False,
        "market_selection_plan_sha256": original_plan_sha,
        "protocol_sha256": protocol,
        "base_model_sha256": base,
        "trained_lora_sha256": adapter,
        "decision_difference_count_NOT_AN_ACCURACY_SCORE": len(diffs),
        "difference_case_ids": diffs,
        "summaries": counts,
        "observed": full,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    need(not args.out.exists(), "NO_OVERWRITE")
    result = aggregate(args.input)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in ("observed", "difference_case_ids")},
                     ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
