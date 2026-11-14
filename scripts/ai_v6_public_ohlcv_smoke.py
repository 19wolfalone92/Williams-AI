"""Real source-verified Binance H1 market-format SMOKE via official CSV archives.

The Binance Futures API returns HTTP 451 from some GitHub hosted runners.
Use only the OFFICIAL data.binance.vision monthly public futures archives
and check the provider's SHA-256 .CHECKSUM for EACH ZIP. Never substitute
a proxy, change geographic restrictions, trade, train, or claim expert gold.

This is deliberately a PRE-TRAINING HISTORICAL format/causality smoke,
not an independent forward chart exam and not a neural inference test.
"""
from __future__ import annotations

import argparse
import calendar
import csv
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import io
import json
from pathlib import Path
import re
from urllib.request import Request, urlopen
from zipfile import ZipFile

from williams_ai.independent_exam_v6 import canonical
from williams_ai.v6_history_context import causal_context
from williams_ai.v6_public_chart_collector import (
    ALLOWED, BARS, MS, normalized_chart,
)

ROOT = "https://data.binance.vision/data/futures/um/monthly/klines/"
_MAX_CHECKSUM_BYTES = 512
_MAX_ZIP_BYTES = 3_000_000


def _require(ok: bool, name: str) -> None:
    if not ok:
        raise ValueError("V6_OFFICIAL_ARCHIVE_" + name)


def _previous_month(start: datetime) -> datetime:
    previous_end = start - timedelta(days=1)
    return previous_end.replace(day=1, hour=0, minute=0, second=0,
                                microsecond=0)


def _cutoff_for_completed_month(now: datetime) -> datetime:
    _require(now.tzinfo is not None, "UTC_AWARE_NOW_REQUIRED")
    utc = now.astimezone(timezone.utc)
    start = utc.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    # Monthly zip may not exist immediately on the first day of new month.
    # If a month is just beginning, fall back to the PRIOR completed pair.
    return start if utc >= start + timedelta(days=3) else _previous_month(start)


def _fetch(url: str, max_bytes: int) -> bytes:
    _require(url.startswith(ROOT) and
             max_bytes in (_MAX_CHECKSUM_BYTES, _MAX_ZIP_BYTES),
             "UNAPPROVED_PUBLIC_ARCHIVE_URL")
    request = Request(url, method="GET", headers={
        "User-Agent": "Williams-AI-v6-official-public-archive-research"})
    with urlopen(request, timeout=25) as response:
        _require(response.status == 200, "OFFICIAL_ARCHIVE_HTTP_FAILURE")
        raw = response.read(max_bytes + 1)
    _require(0 < len(raw) <= max_bytes, "ARCHIVE_SIZE_OUT_OF_BOUNDS")
    return raw


def _month_rows(symbol: str, month: datetime, *, fetch=_fetch) -> tuple[list, dict]:
    _require(symbol in ALLOWED and month.tzinfo is not None
             and month.day == 1 and month.hour == 0,
             "INVALID_SYMBOL_OR_MONTH")
    m = month.strftime("%Y-%m")
    filename = f"{symbol}-1h-{m}.zip"
    url = f"{ROOT}{symbol}/1h/{filename}"
    checksum = fetch(url + ".CHECKSUM", _MAX_CHECKSUM_BYTES).decode("ascii").strip()
    expected = checksum.split()[0] if checksum else ""
    _require(re.fullmatch(r"[0-9a-fA-F]{64}", expected) is not None,
             "INVALID_OFFICIAL_SHA256_CHECKSUM")
    archive = fetch(url, _MAX_ZIP_BYTES)
    actual = sha256(archive).hexdigest()
    _require(actual.lower() == expected.lower(), "OFFICIAL_SHA256_MISMATCH")
    with ZipFile(io.BytesIO(archive)) as z:
        names = z.namelist()
        _require(len(names) == 1 and names[0] == filename[:-4] + ".csv",
                 "UNEXPECTED_ARCHIVE_CSV_NAME")
        _require(z.getinfo(names[0]).file_size <= 10_000_000,
                 "UNBOUNDED_UNCOMPRESSED_CSV")
        with z.open(names[0]) as fd:
            reader = csv.reader(io.TextIOWrapper(fd, encoding="utf-8-sig", newline=""))
            rows = list(reader)
    if rows and rows[0][0] == "open_time":
        rows.pop(0)
    expected_len = 24 * calendar.monthrange(month.year, month.month)[1]
    _require(len(rows) == expected_len, "MISSING_OR_DUPLICATE_MONTHLY_H1")
    first = int(month.timestamp() * 1000)
    normalized = []
    for i, row in enumerate(rows):
        _require(len(row) >= 7, "BAD_MONTHLY_CSV_WIDTH")
        try:
            op, close = int(row[0]), int(row[6])
        except (ValueError, TypeError) as exc:
            raise ValueError("V6_OFFICIAL_ARCHIVE_INVALID_TIMESTAMP") from exc
        _require(op == first + i * MS and close == op + MS - 1,
                 "GAP_OR_FUTURE_BAR_IN_OFFICIAL_MONTH")
        normalized.append([op, *row[1:6], close])
    return normalized, {
        "symbol": symbol, "month_utc": m,
        "official_archive_zip_sha256": actual,
        "official_checksum_verified": True,
        "hourly_rows": expected_len,
        "source": url,
    }


def official_monthly_klines(symbol: str, start_ms: int, end_ms: int,
                            *, fetch=_fetch) -> tuple[list, list]:
    _require(symbol in ALLOWED and type(start_ms) is int and
             type(end_ms) is int and 0 < start_ms < end_ms and
             end_ms - start_ms + 1 == BARS * MS,
             "INVALID_1024_H1_WINDOW")
    cutoff = datetime.fromtimestamp((end_ms + 1) / 1000, tz=timezone.utc)
    _require(cutoff.hour == cutoff.minute == cutoff.second == 0 and cutoff.day == 1,
             "SMOKE_MONTH_BOUNDARY_REQUIRED")
    month2 = _previous_month(cutoff)
    month1 = _previous_month(month2)
    assembled, receipts = [], []
    for month in (month1, month2):
        rows, provenance = _month_rows(symbol, month, fetch=fetch)
        assembled.extend(rows)
        receipts.append(provenance)
    selected = [row for row in assembled if start_ms <= row[0] <= end_ms]
    _require(len(selected) == BARS, "INCOMPLETE_CLOSED_1024_HOUR_SELECTION")
    return selected, receipts


def collect_smoke(*, now: datetime | None = None, fetcher=official_monthly_klines) -> dict:
    if now is None:
        now = datetime.now(timezone.utc)
    cutoff = _cutoff_for_completed_month(now)
    end = int(cutoff.timestamp() * 1000)
    start = end - BARS * MS
    results = []
    for symbol in sorted(ALLOWED):
        material = fetcher(symbol, start, end - 1)
        if isinstance(material, tuple):
            raw, source_receipts = material
        else:
            raw, source_receipts = material, []
        chart = normalized_chart(symbol, cutoff, raw)
        context = causal_context(chart, cutoff_utc=cutoff.strftime("%Y-%m-%dT%H:%M:%SZ"))
        frames = context["timeframes"]
        _require(all(frames[t]["closed_bars"] >= 34 for t in ("H1", "H4", "D1")),
                 "INSUFFICIENT_H1_H4_D1_HISTORICAL_CONTEXT")
        results.append({
            "symbol": symbol, "bars": len(chart["bars"]),
            "first_open_utc": chart["bars"][0]["open_utc"],
            "last_close_utc": chart["bars"][-1]["close_utc"],
            "closed_candle_payload_sha256": sha256(canonical(chart)).hexdigest(),
            "completed_H1": frames["H1"]["closed_bars"],
            "completed_H4": frames["H4"]["closed_bars"],
            "completed_D1": frames["D1"]["closed_bars"],
            "D1_AO_available": frames["D1"]["ao_ac_median_price"]["ao"] is not None,
            "D1_AC_available": frames["D1"]["ao_ac_median_price"]["ac"] is not None,
            "verified_source_months": source_receipts,
        })
    return {
        "schema": "WILLIAMS_V6_REAL_PUBLIC_OHLCV_SMOKE_V1",
        "cutoff_utc": cutoff.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "market_data_endpoint": "Official Binance data.binance.vision UM monthly CSV ZIP + SHA256",
        "futures_api_451_not_bypassed_or_proxied": True,
        "auth_keys_used": False, "model_inference_performed": False,
        "expert_gold_present": False, "training_allowed": False,
        "execution_allowed": False, "not_independent_forward_holdout": True,
        "market_data_result": results,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError("SMOKE_DO_NOT_OVERWRITE")
    data = collect_smoke()
    with args.out.open("x", encoding="utf-8") as f:
        json.dump(data, f, sort_keys=True, indent=2)
        f.write("\n")
    print(json.dumps({
        "cutoff": data["cutoff_utc"],
        "symbols_verified": len(data["market_data_result"]),
        "market_data_only": True,
        "no_model_accuracy_claims": True,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
