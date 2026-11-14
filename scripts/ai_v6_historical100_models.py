"""Run genuine pinned Qwen3-4B BASE + genuine v5 LoRA on 100 REAL H1 charts.

Strictly development-only, no human/expert gold, no exchange orders, no
training, no future_24h access. Output is actual model decisions, NOT accuracy.
Partial progress is saved to disk, never misrepresented as a complete exam.
"""
from __future__ import annotations
import argparse
from collections import Counter
from hashlib import sha256
from pathlib import Path
from datetime import datetime, timezone
import gc
import json
import os
import re

from williams_ai.independent_exam_v6 import canonical
from williams_ai.v6_expert_model_runner import (
    decide, prompt_for_case, protocol_sha256,
)
from williams_ai.v6_v5_lora_binding import (
    BASE_SHA, RAW_ADAPTER_SHA, verified_gguf, verified_receipt,
)

SCHEMA = "WILLIAMS_V6_HIST100_ACTUAL_MATCHED_BASE_V5_CPU_NO_EXPERT_GOLD_V1"
QUESTIONS = {
    "TC1": "Торговый Хаос 1: первый действительный фрактал, волна, AO и аллигатор. Можно ли утверждать о подтверждении стоп-входа только по закрытым барам?",
    "NTD": "Новые измерения в биржевой торговле: AO AC и зона, условия подтверждения входа в рынок по закрытым барам.",
    "TC2": "Trading Chaos 2: когда допустим обратный сигнал мудреца и разворот гипотетической длинной позиции? Биржевой статус неизвестен.",
    "COURSE": "Profitunity Course: 5/34/5 AO zero line минимальные критерии волны 4 и подтверждение завершения волны, если оно имеется.",
}
FAMILIES = {"TC1": "ENTRY", "NTD": "ENTRY", "TC2": "EXIT", "COURSE": "WAVE"}
ROLES = ("matched_base", "candidate_v5")


def need(ok: bool, reason: str):
    if not ok:
        raise ValueError("HIST100_AI_" + reason)


def gather(plan_path: Path, corpus: Path):
    plan = json.loads(plan_path.read_bytes())
    receipt = json.loads((corpus / "market_evidence.json").read_bytes())
    need(plan["schema"] == "WILLIAMS_V6_HISTORICAL100_FIXED_SELECTION_V1", "PLAN_INVALID")
    need(receipt["schema"] == "WILLIAMS_V6_REAL_HISTORICAL_100_MARKET_RECEIPT_V1", "CORPUS_INVALID")
    need(receipt["selection_plan_sha256"] == sha256(plan_path.read_bytes()).hexdigest(), "PLAN_SHA_MISMATCH")
    need(receipt["distinct_chart_windows"] == len(plan["cases"]) == 100 and
         receipt["real_monthly_archives_verified"] == 200 and
         receipt["historical_H1_bars_verified"] == 102400 and
         receipt["future_24h_bars_separately_verified"] == 2400 and
         receipt["no_future_in_model_chart"] is True and
         receipt["expert_gold_present"] is False and
         receipt["base_model_inferences"] == 0 and
         receipt["v5_model_inferences"] == 0, "FAKE_OR_INCOMPLETE_CORPUS")
    meta = {r["id"]: r for r in receipt["cases"]}
    need(len(meta) == 100, "BAD_RECEIPT_CASE_IDS")
    selected = []
    for case in plan["cases"]:
        need(case["id"] in meta and meta[case["id"]]["expert_label"] is None, "MISSING_OR_LEAKING_GOLD")
        b = (corpus / "charts" / (case["id"] + ".json")).read_bytes()
        chart = json.loads(b)
        need(set(chart) == {"symbol", "bars"} and
             chart["symbol"] == case["symbol"] and
             len(chart["bars"]) == case["h1_bars"] == 1024, "UNEXPECTED_CHART_FIELDS")
        need(sha256(canonical(chart)).hexdigest() == meta[case["id"]]["chart_sha256"],
             "CHART_HASH_MISMATCH")
        cutoff = datetime.fromisoformat(case["cutoff_utc"].replace("Z", "+00:00"))
        need(datetime.fromisoformat(chart["bars"][-1]["close_utc"].replace("Z", "+00:00")) == cutoff,
             "CHART_BAD_CUTOFF")
        need(all(datetime.fromisoformat(bar["close_utc"].replace("Z", "+00:00")) <= cutoff
                 for bar in chart["bars"]), "FUTURE_LEAKAGE_IN_CHART")
        model_case = {"id": case["id"], "edition": case["edition"],
                      "family": case["family"], "question": QUESTIONS[case["edition"]],
                      "cutoff_utc": case["cutoff_utc"]}
        need(FAMILIES[case["edition"]] == case["family"], "BOOK_FAMILY_MISMATCH")
        _, ev = prompt_for_case(model_case, chart)
        need(ev["source_missing"] is False, "NO_REVIEWED_BOOK_RULE")
        selected.append((model_case, chart, ev, meta[case["id"]]["chart_sha256"]))
    need(len(selected) == 100 and
         len({x[3] for x in selected}) == 100, "NOT_100_DISTINCT_INPUTS")
    return selected, receipt["selection_plan_sha256"]


def write_receipt(path: Path, state: dict):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, sort_keys=True, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def actual_model_inference(cases, model_path: Path, adapter: Path | None,
                           role: str, out: Path, receipt: dict):
    from llama_cpp import Llama
    options = {"model_path": str(model_path.resolve()),
               "n_ctx": 8192, "n_gpu_layers": 0, "verbose": False}
    if adapter is not None:
        options.update(lora_path=str(adapter.resolve()), lora_scale=1.0)
    model = Llama(**options)
    rows = []
    try:
        for i, (case, chart, evidence_before, chart_sha) in enumerate(cases, start=1):
            answer, ev = decide(model, case, chart, include_model_reason=True)
            need(ev == evidence_before, "CHANGED_SOURCE_IN_MODEL_RUN")
            need(answer["finish_reason"] in {"stop", "length", "error"}, "UNKNOWN_FINISH")
            need(answer["finish_reason"] != "stop" or
                 isinstance(answer.get("model_reason"), str) and
                 1 <= len(answer["model_reason"]) <= 240,
                 "MISSING_MODEL_REASON")
            rows.append({
                **answer, "edition": case["edition"], "family": case["family"],
                "symbol": chart["symbol"], "cutoff_utc": case["cutoff_utc"],
                "chart_sha256": chart_sha,
                "prompt_sha256": ev["prompt_sha256"],
                "retrieved_rule_ids": ev["retrieved_rule_ids"],
                "source_missing": ev["source_missing"],
            })
            if i % 5 == 0 or i == len(cases):
                receipt["observed"][role] = rows
                receipt["complete"] = False
                receipt["completed_model_rows"] = sum(
                    len(receipt["observed"].get(x, [])) for x in ROLES)
                write_receipt(out, receipt)
                print("REAL_HIST100_AI_INFERENCE_PROGRESS", role, i, len(cases), flush=True)
    finally:
        del model
        gc.collect()
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--base-gguf", type=Path, required=True)
    p.add_argument("--trained-lora-gguf", type=Path, required=True)
    p.add_argument("--training-receipt", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    need(not a.out.exists(), "NO_OVERWRITE")
    verified_receipt(a.training_receipt)
    need(verified_gguf(a.base_gguf) == BASE_SHA, "BASE_WEIGHT_IDENTITY_MISMATCH")
    adapter_sha = verified_gguf(a.trained_lora_gguf)
    selected, dataset_sha = gather(a.plan, a.corpus)
    state = {
        "schema": SCHEMA, "cases_total": 100,
        "market_selection_plan_sha256": dataset_sha,
        "protocol_sha256": protocol_sha256(),
        "base_gguf_sha256": BASE_SHA,
        "v5_original_raw_adapter_sha256_attested": RAW_ADAPTER_SHA,
        "v5_converted_lora_gguf_sha256": adapter_sha,
        "expected_models": list(ROLES), "observed": {},
        "complete": False, "completed_model_rows": 0,
        "no_expert_labels": True, "real_market_book_accuracy": None,
        "training_overlap_audited": False,
        "prospective_exam": False, "gpu_training_allowed": False,
        "no_live_broker_trading": True, "research_execution_allowed": False,
    }
    write_receipt(a.out, state)
    for role, adapter in (("matched_base", None),
                          ("candidate_v5", a.trained_lora_gguf)):
        state["observed"][role] = actual_model_inference(
            selected, a.base_gguf, adapter, role, a.out, state)
    before, after = state["observed"]["matched_base"], state["observed"]["candidate_v5"]
    need(len(before) == len(after) == 100, "INCOMPLETE_MODEL_COUNTS")
    for b, v in zip(before, after, strict=True):
        need(all(b[k] == v[k] for k in ("id", "edition", "family", "symbol",
                                       "cutoff_utc", "chart_sha256",
                                       "prompt_sha256", "retrieved_rule_ids")),
             "MODEL_PROMPTS_NOT_MATCHED")
    stats = {}
    for role in ROLES:
        rows = state["observed"][role]
        stats[role] = {
            "total": len(rows),
            "finish_stop": sum(x["finish_reason"] == "stop" for x in rows),
            "decision_counts": dict(Counter(x["decision"] for x in rows)),
            "cases_by_book": {edition: dict(Counter(x["decision"] for x in rows
                if x["edition"] == edition)) for edition in QUESTIONS},
            "short_uncertainty_reason_with_non_uncertainty_decision": sum(
                x.get("model_reason", "").strip().upper() == "INSUFFICIENT_DATA"
                and x["decision"] not in {"INSUFFICIENT_DATA", "ABSTAIN"}
                for x in rows),
            "reason_exactly_240_chars": sum(
                len(x.get("model_reason", "")) == 240 for x in rows),
        }
    state["results_not_accuracy"] = {
        "matched_input_case_count": 100,
        "decision_differences": sum(b["decision"] != v["decision"]
                                    for b, v in zip(before, after)),
        "model_summary": stats,
    }
    state["complete"] = True
    state["completed_model_rows"] = 200
    write_receipt(a.out, state)
    print("ACTUAL_HIST100_REAL_BASE_AND_V5_CPU_COMPLETE_NO_GOLD",
          json.dumps(state["results_not_accuracy"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
