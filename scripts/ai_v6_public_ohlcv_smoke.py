"""REAL public Binance USD-M H1 format smoke, never an exam or trading call.

No cohort, no labels, no GPU, no credentials, no model calls. The only
artifact is a numeric/summed SHA-256 receipt of public closed bars.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
from pathlib import Path

from williams_ai.independent_exam_v6 import canonical
from williams_ai.v6_history_context import causal_context
from williams_ai.v6_public_chart_collector import (
    ALLOWED, BARS, MS, normalized_chart, read_only_public_klines,
)


def collect_smoke(*, now:datetime|None=None, fetcher=read_only_public_klines) -> dict:
    if now is None:
        now = datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise ValueError("SMOKE_CLOCK_MUST_BE_UTC_AWARE")
    cutoff = now.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0) - timedelta(hours=2)
    end = int(cutoff.timestamp() * 1000)
    start = end - BARS * MS
    results = []
    for symbol in sorted(ALLOWED):
        raw = fetcher(symbol, start, end - 1)
        chart = normalized_chart(symbol, cutoff, raw)
        context = causal_context(chart, cutoff_utc=cutoff.strftime("%Y-%m-%dT%H:%M:%SZ"))
        frames = context["timeframes"]
        if not all(frames[t]["closed_bars"]>=34 for t in ("H1","H4","D1")):
            raise ValueError("SMOKE_MISSING_HISTORY_FOR_H1_H4_D1")
        results.append({
            "symbol":symbol,"bars":len(chart["bars"]),
            "first_open_utc":chart["bars"][0]["open_utc"],
            "last_close_utc":chart["bars"][-1]["close_utc"],
            "closed_candle_payload_sha256":sha256(canonical(chart)).hexdigest(),
            "completed_H1":frames["H1"]["closed_bars"],
            "completed_H4":frames["H4"]["closed_bars"],
            "completed_D1":frames["D1"]["closed_bars"],
            "D1_AO_available":frames["D1"]["ao_ac_median_price"]["ao"] is not None,
            "D1_AC_available":frames["D1"]["ao_ac_median_price"]["ac"] is not None,
        })
    return {"schema":"WILLIAMS_V6_REAL_PUBLIC_OHLCV_SMOKE_V1",
            "cutoff_utc":cutoff.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "market_data_endpoint":"Binance public USD-M GET /fapi/v1/klines",
            "auth_keys_used":False,"model_inference_performed":False,
            "expert_gold_present":False,"training_allowed":False,
            "execution_allowed":False,"not_independent_forward_holdout":True,
            "market_data_result":results}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    if args.out.exists():
        raise FileExistsError("SMOKE_DO_NOT_OVERWRITE")
    data=collect_smoke()
    with args.out.open("x",encoding="utf-8") as f:
        json.dump(data,f,sort_keys=True,indent=2)
        f.write("\n")
    print(json.dumps({"cutoff":data["cutoff_utc"],
        "symbols_verified":len(data["market_data_result"]),
        "market_data_only":True,"no_model_accuracy_claims":True}))


if __name__=="__main__":
    main()
