"""Execute REAL pinned Qwen3-4B/base vs genuine v5 LoRA on archived USD-M H1.

Immediate RETROSPECTIVE 4-book CPU technical smoke: NOT prospectively sealed,
NOT expert-labelled, NOT a trading qualification, NOT a test accuracy score.
Reads only official SHA256-verified PUBLIC monthly Binance archive data.
NEVER accesses exchange trading endpoints, places orders, retrains or exports
model text/weights/keys. Any incompleteness or identity mismatch fails closed.
"""
from __future__ import annotations

import argparse
import gc
from hashlib import sha256
import json
from pathlib import Path

from scripts.ai_v6_public_ohlcv_smoke import official_monthly_klines
from williams_ai.independent_exam_v6 import canonical
from williams_ai.v6_public_chart_collector import BARS, MS, normalized_chart
from williams_ai.v6_expert_model_runner import (
    decide, prompt_for_case, protocol_sha256,
)
from williams_ai.v6_v5_lora_binding import (
    BASE_SHA, RAW_ADAPTER_SHA, verified_gguf, verified_receipt,
)

CUTOFF = "2026-11-01T00:00:00Z"
CUT_MS = 1793491200000
BASE_SHA256 = BASE_SHA
SCHEMA = "WILLIAMS_V6_IMMEDIATE_REAL_RETRO_4BOOK_CPU_SMOKE_V1"
CASES = (
    ("RETRO-TC1-BTC-ENTRY", "BTCUSDT", "TC1", "ENTRY",
     "Торговый Хаос 1: первый действительный фрактал, волна, AO и аллигатор. Можно ли утверждать о подтверждении стоп-входа только по закрытым барам?"),
    ("RETRO-NTD-ETH-ENTRY", "ETHUSDT", "NTD", "ENTRY",
     "Новые измерения в биржевой торговле: AO AC и зона, условия подтверждения входа в рынок по закрытым барам."),
    ("RETRO-TC2-SOL-EXIT", "SOLUSDT", "TC2", "EXIT",
     "Trading Chaos 2: когда допустим обратный сигнал мудреца и разворот гипотетической длинной позиции? Биржевой статус неизвестен."),
    ("RETRO-COURSE-XRP-WAVE", "XRPUSDT", "COURSE", "WAVE",
     "Profitunity Course: 5/34/5 AO zero line минимальные критерии волны 4 и подтверждение завершения волны, если оно имеется."),
)


def build_cases():
    from datetime import datetime, timezone
    end = int(datetime(2026, 11, 1, tzinfo=timezone.utc).timestamp()*1000)
    if end != CUT_MS:
        raise ValueError("RETRO_CUTOFF_CONSTANT_MISMATCH")
    results = []
    for cid, symbol, edition, family, question in CASES:
        raw, official = official_monthly_klines(
            symbol, end-BARS*MS, end-1)
        chart = normalized_chart(
            symbol, datetime(2026, 11, 1, tzinfo=timezone.utc), raw)
        case = {"id": cid, "edition": edition, "family": family,
                "cutoff_utc": CUTOFF, "question": question}
        messages, meta = prompt_for_case(case, chart)
        results.append((case, chart, meta, official))
    if len(results) != 4 or len({x[0]["edition"] for x in results}) != 4:
        raise ValueError("RETRO_NOT_ALL_FOUR_BOOKS")
    return results


def model_results(*, cases, base: Path, lora: Path | None):
    from llama_cpp import Llama
    options = {"model_path": str(base.resolve()),
               "n_ctx": 8192, "n_gpu_layers": 0, "verbose": False}
    if lora is not None:
        options.update(lora_path=str(lora.resolve()), lora_scale=1.0)
    model = Llama(**options)
    rows = []
    try:
        for case, chart, original_meta, source in cases:
            _, meta = prompt_for_case(case, chart)
            if meta != original_meta:
                raise ValueError("RETRO_SOURCE_OR_PROMPT_CHANGED_DURING_RUN")
            answer, evidence = decide(model, case, chart, include_model_reason=True)
            if evidence != original_meta:
                raise ValueError("RETRO_MODEL_INPUT_NOT_SAME_AS_SOURCE")
            rows.append({**answer,
                "edition": case["edition"], "symbol": chart["symbol"],
                "family": case["family"], "cutoff_utc": CUTOFF,
                "chart_sha256": sha256(canonical(chart)).hexdigest(),
                "prompt_sha256": evidence["prompt_sha256"],
                "retrieved_rule_ids": evidence["retrieved_rule_ids"],
                "source_missing": evidence["source_missing"],
            })
    finally:
        del model
        gc.collect()
    return rows


def run(*, base: Path, adapter: Path, receipt_file: Path) -> dict:
    verified_receipt(receipt_file)
    if verified_gguf(base) != BASE_SHA256:
        raise ValueError("RETRO_WRONG_BASE_MODEL_BYTES")
    adapter_sha = verified_gguf(adapter)
    samples = build_cases()
    report = {}
    for label, lora in (("matched_base", None), ("candidate_v5", adapter)):
        report[label] = model_results(cases=samples, base=base, lora=lora)
    for role, rows in report.items():
        for row in rows:
            if row["finish_reason"] == "stop" and (
                not isinstance(row.get("model_reason"), str) or
                not 1 <= len(row["model_reason"]) <= 240
            ):
                raise ValueError("RETRO_MODEL_REASON_NOT_RETAINED")
    for a, b in zip(report["matched_base"], report["candidate_v5"], strict=True):
        if (a["id"] != b["id"] or a["prompt_sha256"] != b["prompt_sha256"] or
            a["chart_sha256"] != b["chart_sha256"] or
            a["retrieved_rule_ids"] != b["retrieved_rule_ids"]):
            raise ValueError("RETRO_BASE_V5_PROTOCOL_NOT_EQUAL")
    return {
        "schema": SCHEMA, "cutoff_utc": CUTOFF,
        "data_period_before_v5_training": True,
        "cases_total": 4, "four_books": ["TC1", "NTD", "TC2", "COURSE"],
        "protocol_sha256": protocol_sha256(),
        "base_gguf_sha256": BASE_SHA256,
        "v5_original_raw_adapter_sha256_attested": RAW_ADAPTER_SHA,
        "v5_converted_lora_gguf_sha256": adapter_sha,
        "sources": [
            {"symbol": chart["symbol"], "edition": case["edition"],
             "chart_sha256": sha256(canonical(chart)).hexdigest(),
             "official_archive_provenance": original}
            for case, chart, _, original in samples],
        "observed": report,
        "no_expert_labels": True, "real_market_book_accuracy": None,
        "comparison_is_only_model_decisions_not_correctness": True,
        "model_generated_rationales_available_if_stop": True,
        "model_reason_not_expert_gold_or_verified_causal_attribution": True,
        "research_execution_allowed": False, "gpu_training_allowed": False,
        "no_live_broker_trading": True, "no_model_weight_export": True,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-gguf", type=Path, required=True)
    p.add_argument("--trained-lora-gguf", type=Path, required=True)
    p.add_argument("--training-receipt", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError("RETRO_NO_OVERWRITE")
    result = run(base=a.base_gguf, adapter=a.trained_lora_gguf,
                 receipt_file=a.training_receipt)
    with a.out.open("x", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, sort_keys=True, indent=2)
        f.write("\n")
    print(json.dumps({
        "schema": SCHEMA,
        "cases": len(result["observed"]["candidate_v5"]),
        "each_model_has_four_book_real_market_decisions": True,
        "base_stop": sum(x["finish_reason"]=="stop" for x in result["observed"]["matched_base"]),
        "v5_stop": sum(x["finish_reason"]=="stop" for x in result["observed"]["candidate_v5"]),
        "base_abstentions": sum(x["decision"]=="ABSTAIN" for x in result["observed"]["matched_base"]),
        "v5_abstentions": sum(x["decision"]=="ABSTAIN" for x in result["observed"]["candidate_v5"]),
        "expert_accuracy": None, "execution_allowed": False,
    }, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
