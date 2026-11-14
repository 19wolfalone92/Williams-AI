"""Acquire 100 real historical public USD-M H1 windows, verify official ZIP SHA256.

Research dataset: no Qwen/LoRA inference, no gold, no orders, no training.
Each model prompt must take ONLY charts/*.json, never future24/*.json.
"""
from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from io import BytesIO, TextIOWrapper
import json
from pathlib import Path
from time import sleep
from urllib.request import Request, urlopen
import zipfile

from williams_ai.independent_exam_v6 import canonical

BASE = "https://data.binance.vision/data/futures/um/monthly/klines/"
HOUR_MS = 3_600_000
HISTORY = 1024
FUTURE = 24
EXPECTED_ED = {"TC1", "NTD", "TC2", "COURSE"}
SOURCE = "BINANCE_OFFICIAL_MONTHLY_USDM_H1_ZIP_SHA256_VERIFIED"


def require(condition: bool, code: str):
    if not condition:
        raise ValueError("HIST100_" + code)


def download_bytes(url: str, max_bytes: int) -> bytes:
    last = None
    for attempt in range(3):
        try:
            request = Request(url, headers={"User-Agent": "williams-ai-v6-historical100-readonly/1.0"})
            with urlopen(request, timeout=40) as response:
                payload = response.read(max_bytes + 1)
            require(0 < len(payload) <= max_bytes, "ARCHIVE_SIZE_UNSAFE")
            return payload
        except Exception as exc:
            last = exc
            if attempt < 2:
                sleep(attempt + 1)
    raise RuntimeError(f"Historical source not available: {url}: {last}") from last


def normalize_ms(value: str) -> int:
    n = int(value)
    if n > 10**14:  # Future-proof Binance's possible microsecond timestamps.
        require(n % 1000 == 0, "MICROSECOND_NON_HOURLY")
        n //= 1000
    return n


def verified_month(symbol: str, year_month: str):
    name = f"{symbol}-1h-{year_month}.zip"
    url = f"{BASE}{symbol}/1h/{name}"
    checksum = download_bytes(url + ".CHECKSUM", 4000).decode("ascii").split()[0].lower()
    require(len(checksum) == 64 and all(c in "0123456789abcdef" for c in checksum), "INVALID_CHECKSUM")
    packed = download_bytes(url, 40_000_000)
    digest = sha256(packed).hexdigest()
    require(digest == checksum, "ZIP_SHA256_MISMATCH")
    records = []
    with zipfile.ZipFile(BytesIO(packed)) as z:
        require(len(z.namelist()) == 1, "ZIP_UNEXPECTED_MEMBERS")
        info = z.infolist()[0]
        require(info.file_size <= 150_000_000, "ZIP_EXPANSION_LIMIT")
        with z.open(info) as f:
            for row in csv.reader(TextIOWrapper(f, encoding="utf-8-sig")):
                if not row or row[0].strip().lower() in {"open_time", "open time"}:
                    continue
                require(len(row) >= 7, "INVALID_CSV_COLUMN_COUNT")
                timestamp = normalize_ms(row[0])
                x = [float(row[n]) for n in range(1, 6)]
                require(all(__import__("math").isfinite(n) for n in x) and
                        x[0] > 0 and x[2] > 0 and x[4] >= 0 and
                        x[1] >= max(x[0], x[3]) and
                        x[2] <= min(x[0], x[3]), "INVALID_OHLCV")
                close_time = normalize_ms(row[6])
                require(close_time == timestamp + HOUR_MS - 1, "UNEXPECTED_CANDLE_CLOSE")
                records.append((timestamp, *x))
    require(24 * 28 <= len(records) <= 24 * 31, "BAD_MONTH_H1_ROW_COUNT")
    for i, record in enumerate(records[1:], start=1):
        require(record[0] == records[i - 1][0] + HOUR_MS, "MONTH_DATA_GAP")
    return (symbol, year_month), records, {
        "source_url": url,
        "zip_sha256": digest,
        "official_checksum_verified": True,
        "records": len(records),
    }


def months_for(start_ms: int, end_ms: int):
    d = datetime.fromtimestamp(start_ms / 1000, timezone.utc).date().replace(day=1)
    stop = datetime.fromtimestamp(end_ms / 1000, timezone.utc).date().replace(day=1)
    result = []
    while d <= stop:
        result.append(d.strftime("%Y-%m"))
        d = (d.replace(day=28) + timedelta(days=4)).replace(day=1)
    return result


def derived_ao_ac(series):
    median = [(float(x["high"]) + float(x["low"])) / 2 for x in series]
    aos = [sum(median[i - 4:i + 1]) / 5 -
           sum(median[i - 33:i + 1]) / 34
           for i in range(33, len(median))]
    require(len(aos) >= 6, "NO_AO_AC_CONTEXT")
    return {
        "ao": round(aos[-1], 10),
        "ao_previous": round(aos[-2], 10),
        "ac": round(aos[-1] - sum(aos[-5:]) / 5, 10),
        "ac_previous": round(aos[-2] - sum(aos[-6:-1]) / 5, 10),
    }


def aggregate(series, hours):
    require(hours in (4, 24), "UNSUPPORTED_TIMEFRAME")
    first = datetime.fromisoformat(series[0]["open_utc"].replace("Z", "+00:00"))
    skip = (hours - first.hour % hours) % hours
    bars = series[skip:]
    result = []
    for index in range(0, len(bars) - hours + 1, hours):
        batch = bars[index:index + hours]
        result.append({
            "open": batch[0]["open"],
            "high": max(b["high"] for b in batch),
            "low": min(b["low"] for b in batch),
            "close": batch[-1]["close"],
            "volume": sum(b["volume"] for b in batch),
        })
    return result


def utc(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def chart_for(case, by_month):
    cutoff = int(datetime.fromisoformat(case["cutoff_utc"].replace("Z", "+00:00")).timestamp() * 1000)
    begin, last = cutoff - HISTORY * HOUR_MS, cutoff + FUTURE * HOUR_MS - 1
    items = []
    receipts = []
    for ym in months_for(begin, last):
        rows, receipt = by_month[(case["symbol"], ym)]
        items += rows
        receipts.append(receipt)
    selected = [x for x in items if begin <= x[0] < cutoff + FUTURE * HOUR_MS]
    require(len(selected) == HISTORY + FUTURE, "MISSING_WINDOW_H1")
    for i, row in enumerate(selected):
        require(row[0] == begin + i * HOUR_MS, "NONCAUSAL_H1_OR_GAP")
    historic, future = selected[:HISTORY], selected[HISTORY:]
    require(historic[-1][0] + HOUR_MS == cutoff and
            future[0][0] == cutoff, "LOOKAHEAD_SPLIT")

    def convert(x):
        return {
            "open_utc": utc(x[0]), "close_utc": utc(x[0] + HOUR_MS),
            "open": x[1], "high": x[2], "low": x[3], "close": x[4], "volume": x[5],
        }

    chart = {"symbol": case["symbol"], "bars": list(map(convert, historic))}
    next24 = {"symbol": case["symbol"], "cutoff_utc": case["cutoff_utc"],
              "evaluation_only_not_in_model_prompt": True,
              "bars": list(map(convert, future))}
    h1 = derived_ao_ac(chart["bars"])
    h4 = derived_ao_ac(aggregate(chart["bars"], 4))
    d1 = derived_ao_ac(aggregate(chart["bars"], 24))
    price = chart["bars"][-1]["close"]
    delta_pct = (future[-1][4] / price - 1) * 100
    return chart, next24, {
        "id": case["id"], "symbol": case["symbol"],
        "cutoff_utc": case["cutoff_utc"],
        "edition": case["edition"], "family": case["family"],
        "historical_H1": HISTORY, "evaluation_next24_H1": FUTURE,
        "H1_ao_ac": h1, "H4_ao_ac": h4, "D1_ao_ac": d1,
        "next24h_market_close_pct_NOT_AI_PNL": round(delta_pct, 6),
        "next24h_market_high_pct_NOT_AI_PNL": round(
            (max(x[2] for x in future) / price - 1) * 100, 6),
        "next24h_market_low_pct_NOT_AI_PNL": round(
            (min(x[3] for x in future) / price - 1) * 100, 6),
        "chart_sha256": sha256(canonical(chart)).hexdigest(),
        "archive_sha256_receipts": receipts,
        "model_prediction": None, "model_reason": None,
        "expert_label": None, "trade_execution_allowed": False,
    }


def run(plan_path: Path, out: Path):
    raw = plan_path.read_bytes()
    plan = json.loads(raw)
    cases = plan["cases"]
    require(plan["schema"] == "WILLIAMS_V6_HISTORICAL100_FIXED_SELECTION_V1" and
            len(cases) == 100 and len({(x["symbol"], x["cutoff_utc"]) for x in cases}) == 100,
            "CHANGED_SELECTION")
    require(len({x["symbol"] for x in cases}) == 25 and
            {x["edition"] for x in cases} == EXPECTED_ED and
            all(sum(x["edition"] == book for x in cases) == 25 for book in EXPECTED_ED),
            "UNBALANCED_BOOKS")
    jobs = set()
    for c in cases:
        require(c["h1_bars"] == HISTORY and
                c["postcutoff_evaluation_h1"] == FUTURE, "BAD_BAR_CONTRACT")
        cutoff = int(datetime.fromisoformat(c["cutoff_utc"].replace("Z", "+00:00")).timestamp() * 1000)
        for ym in months_for(cutoff - HISTORY * HOUR_MS, cutoff + FUTURE * HOUR_MS - 1):
            jobs.add((c["symbol"], ym))
    fetched = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        tasks = {pool.submit(verified_month, sym, ym): (sym, ym) for sym, ym in sorted(jobs)}
        for future in as_completed(tasks):
            k, records, evidence = future.result()
            require(k not in fetched, "DUPLICATE_MONTHLY_DATA")
            fetched[k] = (records, evidence)
    require(len(fetched) == len(jobs), "MISSING_OFFICIAL_MONTHS")
    out.mkdir(parents=True, exist_ok=False)
    (out / "charts").mkdir()
    (out / "future24").mkdir()
    rows = []
    for case in cases:
        chart, future, report = chart_for(case, fetched)
        (out / "charts" / f'{case["id"]}.json').write_bytes(canonical(chart))
        (out / "future24" / f'{case["id"]}.json').write_bytes(canonical(future))
        rows.append(report)
    require(len(rows) == 100 and len({r["chart_sha256"] for r in rows}) == 100,
            "CHARTS_NOT_100_DISTINCT")
    summary = {
        "schema": "WILLIAMS_V6_REAL_HISTORICAL_100_MARKET_RECEIPT_V1",
        "selection_plan_sha256": sha256(raw).hexdigest(),
        "source": SOURCE, "real_monthly_archives_verified": len(fetched),
        "distinct_symbols": 25, "distinct_chart_windows": 100,
        "historical_H1_bars_verified": 100 * HISTORY,
        "future_24h_bars_separately_verified": 100 * FUTURE,
        "no_future_in_model_chart": True, "expert_gold_present": False,
        "base_model_inferences": 0, "v5_model_inferences": 0,
        "training_overlap_audited": False,
        "independent_exam_passed": False,
        "live_trading_allowed": False,
        "next24h_up_market_count": sum(r["next24h_market_close_pct_NOT_AI_PNL"] > 0 for r in rows),
        "next24h_down_market_count": sum(r["next24h_market_close_pct_NOT_AI_PNL"] < 0 for r in rows),
        "edition_counts": {e: sum(r["edition"] == e for r in rows) for e in sorted(EXPECTED_ED)},
        "cases": rows,
    }
    (out / "market_evidence.json").write_bytes(canonical(summary))
    print(json.dumps({k: v for k, v in summary.items() if k != "cases"}, indent=2), flush=True)
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--plan", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    run(a.plan, a.out)


if __name__ == "__main__":
    main()
