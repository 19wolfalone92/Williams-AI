"""One-time REAL public USD-M H1 data smoke test; NEVER an expert holdout.

Retrieves and verifies four 1024-bar UTC historical series through the
existing no-key collector transport. No model calls, labels, Binance keys,
positions, order endpoints, GPU or promotion. The as-of is intentionally
BEFORE the fully forward research cohort: DO NOT USE AS EXAM GOLD.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

from .independent_exam_v6 import canonical
from .v6_history_context import causal_context
from .v6_public_chart_collector import (
    BARS, MS, URL, normalized_chart, read_only_public_klines,
)

CUTOFF = datetime(2026, 11, 12, tzinfo=timezone.utc)
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")
LABEL = "NON_HOLDOUT_PUBLIC_MARKET_TRANSPORT_SMOKE_NOT_EXPERT_EXAM"


def smoke(fetcher=read_only_public_klines, *, now=None):
    if now is None:
        now = datetime.now(timezone.utc)
    if now < CUTOFF or now.tzinfo is None:
        raise ValueError("SMOKE_HISTORIC_CUTOFF_NOT_YET_AVAILABLE")
    end = int(CUTOFF.timestamp() * 1000)
    start = end - BARS * MS
    receipts, chart_blobs = {}, {}
    for symbol in SYMBOLS:
        raw = fetcher(symbol, start, end - 1)
        chart = normalized_chart(symbol, CUTOFF, raw)
        context = causal_context(chart, cutoff_utc="2026-11-12T00:00:00Z")
        frames = context["timeframes"]
        if (frames["H1"]["closed_bars"] != BARS
                or frames["H4"]["closed_bars"] != BARS // 4
                or frames["D1"]["closed_bars"] < 40
                or frames["D1"]["ao_ac_median_price"]["ac"] is None
                or any(f["latest_close_utc"] > "2026-11-12T00:00:00Z"
                       for f in frames.values())):
            raise ValueError("SMOKE_REAL_MULTIFRAME_CAUSAL_OR_COVERAGE_MISMATCH")
        chart_blob = canonical(chart)
        chart_blobs[f"charts/{symbol}_H1_1024_NON_HOLDOUT.json"] = chart_blob
        receipts[symbol] = {
            "chart_sha256": sha256(chart_blob).hexdigest(),
            "raw_exchange_response_sha256": sha256(canonical(raw)).hexdigest(),
            "completed_H1_bars": frames["H1"]["closed_bars"],
            "completed_H4_bars": frames["H4"]["closed_bars"],
            "completed_D1_bars": frames["D1"]["closed_bars"],
            "latest_H1_close_utc": frames["H1"]["latest_close_utc"],
            "latest_D1_close_utc": frames["D1"]["latest_close_utc"],
            "last_price": chart["bars"][-1]["close"],
            "H1_AO": frames["H1"]["ao_ac_median_price"]["ao"],
            "H1_AC": frames["H1"]["ao_ac_median_price"]["ac"],
            "H4_AO": frames["H4"]["ao_ac_median_price"]["ao"],
            "D1_AO": frames["D1"]["ao_ac_median_price"]["ao"],
            "D1_AC": frames["D1"]["ao_ac_median_price"]["ac"],
            "book_rule_verified": False, "expert_label_available": False,
        }
    report = {
        "schema": "WILLIAMS_V6_REAL_BINANCE_SOURCE_SMOKE_V1",
        "purpose": LABEL,
        "venue": "BINANCE_PUBLIC_USDM_PERPETUAL_H1",
        "endpoint": URL, "cutoff_utc": "2026-11-12T00:00:00Z",
        "source_is_live_public_api_not_replayed_mock": True,
        "four_book_expert_gold": False,
        "new_unseen_heldout_market_data": False,
        "future_final_cohort_touched": False,
        "lora_inference_performed": False, "gpu_used": False,
        "trading_keys_used": False, "execution_allowed": False,
        "results": receipts,
    }
    return report, chart_blobs


def archive(out: Path, *, fetcher=read_only_public_klines, now=None) -> dict:
    if out.exists() or out.is_symlink():
        raise FileExistsError("SMOKE_OUTPUT_EXISTS_NO_OVERWRITE")
    report, charts = smoke(fetcher=fetcher, now=now)
    with ZipFile(out, "x", compression=ZIP_DEFLATED) as z:
        z.writestr("NON_HOLDOUT_real_market_receipt.json", canonical(report))
        for file, content in sorted(charts.items()):
            z.writestr(file, content)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--acknowledge-non-holdout", action="store_true")
    args = parser.parse_args()
    if not args.acknowledge_non_holdout:
        raise ValueError("SMOKE_MUST_EXPLICITLY_ACKNOWLEDGE_NOT_EXPERT_HOLDOUT")
    report = archive(args.out)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
