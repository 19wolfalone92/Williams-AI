"""Real PUBLIC market-data smoke test; never produces expert gold or AI scores.

Fixed historical UTC cutoff to avoid selection after observing returns. Validate
Binance USD-M H1 for four symbols, deterministic H4/D1 construction and 4-book
gold-free Qwen prompts. No secrets, GPU, training, trading or main merge.
"""
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import argparse
import json
from pathlib import Path

from williams_ai.v6_public_chart_collector import (
    BARS, MS, normalized_chart, read_only_public_klines,
)
from williams_ai.v6_history_context import causal_context
from williams_ai.v6_expert_model_runner import prompt_for_case, protocol_sha256
from williams_ai.independent_exam_v6 import canonical, PDF_SHA

CUTOFF = "2026-11-12T00:00:00Z"
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")
BOOKS = ("TC1", "NTD", "TC2", "COURSE")
QUESTIONS = {
    "TC1": "How are the Williams first Trading Chaos fractal and five waves checked from completed candles?",
    "NTD": "Under New Trading Dimensions what are Alligator Teeth, zone, AO and AC requirements?",
    "TC2": "How does Trading Chaos second edition establish a valid wise man reversal with angulation?",
    "COURSE": "In Profitunity Course what establishes the fourth wave minimum and zero-line crossing?",
}


def run(fetch=read_only_public_klines) -> dict:
    cutoff = datetime.fromisoformat(CUTOFF.replace("Z", "+00:00"))
    if not cutoff < datetime.now(timezone.utc):
        raise ValueError("SMOKE_REQUIRES_FULLY_PAST_CUTOFF")
    end = int(cutoff.timestamp() * 1000)
    start = end - BARS * MS
    rows = []
    for symbol, edition in zip(SYMBOLS, BOOKS):
        raw = fetch(symbol, start, end - 1)
        chart = normalized_chart(symbol, cutoff, raw)
        context = causal_context(chart, cutoff_utc=CUTOFF)
        chart_hash = sha256(canonical(chart)).hexdigest()
        case = {"edition": edition, "family": "ENTRY", "cutoff_utc": CUTOFF,
                "question": QUESTIONS[edition]}
        msgs, meta = prompt_for_case(case, chart)
        if len(msgs) != 2 or not meta["retrieved_rule_ids"]:
            raise ValueError("SMOKE_MISSING_BOOK_SOURCE_OR_MODEL_PROMPT")
        tfs = context["timeframes"]
        if any(x["latest_close_utc"] > CUTOFF for x in tfs.values()):
            raise ValueError("SMOKE_FUTURE_CLOSE_LEAK")
        if any(tfs[x]["ao_ac_median_price"]["ao"] is None
               or tfs[x]["ao_ac_median_price"]["ac"] is None
               for x in ("H1", "H4", "D1")):
            raise ValueError("SMOKE_INSUFFICIENT_CONTEXT_FOR_AO_AC")
        rows.append({
            "symbol": symbol, "edition": edition,
            "data_source": "Binance public USD-M futures 1h closed klines",
            "cutoff_utc": CUTOFF,
            "history_first_utc": chart["bars"][0]["open_utc"],
            "history_last_close_utc": chart["bars"][-1]["close_utc"],
            "bars": len(chart["bars"]),
            "chart_sha256": chart_hash,
            "verified_pdf_source_sha256": PDF_SHA[edition],
            "H1_bars": tfs["H1"]["closed_bars"],
            "H4_bars": tfs["H4"]["closed_bars"],
            "D1_bars": tfs["D1"]["closed_bars"],
            "retrieved_rule_ids": meta["retrieved_rule_ids"],
            "prompt_sha256": meta["prompt_sha256"],
        })
    if len(rows) != 4 or any(r["bars"] != BARS for r in rows):
        raise ValueError("SMOKE_PARTIAL_PUBLIC_MARKET_INTAKE")
    return {
        "schema": "WILLIAMS_V6_REAL_PUBLIC_OHLCV_SMOKE_V1",
        "protocol_sha256": protocol_sha256(),
        "cases": rows,
        "cutoff_preselected_not_return_optimized": True,
        "no_expert_labels_or_model_outputs": True,
        "not_independent_exam": True,
        "test_accuracy": None,
        "weights_loaded": False,
        "GPU_used": False,
        "execution_allowed": False,
        "main_merged": False,
    }


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():
        raise FileExistsError("SMOKE_NEVER_OVERWRITE")
    result=run()
    with a.out.open("x",encoding="utf-8") as fd:
        json.dump(result,fd,indent=2,ensure_ascii=False,sort_keys=True)
        fd.write("\n")
    print("PUBLIC_H1_FOUR_BOOK_SMOKE_PASS",
          json.dumps({ "cases":len(result["cases"]),
            "books":[x["edition"] for x in result["cases"]],
            "symbols":[x["symbol"] for x in result["cases"]],
            "H1_min":min(x["H1_bars"] for x in result["cases"]),
            "H4_min":min(x["H4_bars"] for x in result["cases"]),
            "D1_min":min(x["D1_bars"] for x in result["cases"]),
            "not_independent_exam":True},sort_keys=True),flush=True)


if __name__=="__main__":
    main()
