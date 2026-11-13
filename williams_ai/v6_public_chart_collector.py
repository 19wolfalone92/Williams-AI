"""Collect prospective UNLABELED H1 USD-M public candles; zero keys/orders/GPU.

Only run after all scheduled UTC cutoffs have fully closed. Reject partial,
missing, noncontiguous, malformed or not-yet-closed Binance kline responses.
This output is a market-data CANDIDATE, not an independent expert exam.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zipfile import ZipFile, ZIP_DEFLATED

from .independent_exam_v6 import (
    BOOKS, FAMILIES, PDF_SHA, SCHEMA, _utc, canonical, validate_cases,
)

SCHEME = "WILLIAMS_V6_FORWARD_CHART_PLAN_V1"
URL = "https://fapi.binance.com/fapi/v1/klines"
MS = 3600 * 1000
BARS = 128
ALLOWED = frozenset(("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"))


def need(ok: bool, code: str):
    if not ok:
        raise ValueError("V6_PUBLIC_MARKET_INTAKE_" + code)


def check_plan(plan: dict, *, now: datetime | None = None):
    need(type(plan) is dict and
         set(plan) == {
             "schema", "dataset_id", "created_utc",
             "training_cutoff_reference_sha", "examination_labels_present",
             "prospective_chart_source", "data_provenance_not_independent_expert_review",
             "source_overlap_audit", "selection_algorithm", "max_bars_per_case", "cases"},
         "PLAN_SCHEMA_CHANGED")
    need(plan["schema"] == SCHEME and plan["prospective_chart_source"] == URL
         and plan["examination_labels_present"] is False
         and plan["data_provenance_not_independent_expert_review"] is True
         and plan["source_overlap_audit"] == "EXTERNAL_REVIEW_REQUIRED"
         and plan["training_cutoff_reference_sha"] ==
             "e3f1b4d34059f10adb1b2791267bb11b79ec67c1"
         and plan["max_bars_per_case"] == BARS, "PLAN_PROVENANCE_INVALID")
    c = plan["cases"]
    need(type(c) is list and len(c) == 32, "EXPECTED_32_FROZEN_FORWARD_CASES")
    ids, windows, days, books = set(), set(), set(), set()
    allowed_keys = {"id", "edition", "family", "symbol", "cutoff_utc",
                    "source_pdf_pages", "question", "provenance_id",
                    "venue", "source_csv_optional"}
    for row in c:
        need(type(row) is dict and set(row) == allowed_keys,
             "PLAN_CASE_FIELDS_CHANGED")
        need(type(row["id"]) is str and row["id"].startswith("E6-FWD-")
             and row["id"].replace("-", "").isalnum()
             and row["id"] not in ids, "INVALID_OR_DUPLICATE_CASE_ID")
        ids.add(row["id"])
        need(row["edition"] in BOOKS and row["family"] in FAMILIES,
             "INVALID_BOOK_OR_TASK")
        need(row["symbol"] in ALLOWED
             and row["venue"] == "BINANCE_PUBLIC_USDM_PERPETUAL_H1"
             and row["source_csv_optional"] is None,
             "UNAUTHORIZED_MARKET_SOURCE")
        cutoff = _utc(row["cutoff_utc"])
        need(cutoff.minute == cutoff.second == cutoff.microsecond == 0,
             "NON_HOURLY_CUTOFF")
        # The fixed plan MUST be created before the forward market data.
        need(cutoff > datetime(2026, 11, 13, tzinfo=timezone.utc),
             "NOT_POST_TRAINING_FORWARD_CASE")
        days.add(cutoff.date().isoformat())
        books.add(row["edition"])
        key = (row["symbol"], row["cutoff_utc"])
        need(key not in windows, "DUPLICATE_MARKET_WINDOW")
        windows.add(key)
        if now is not None:
            need(cutoff + timedelta(hours=2) <= now,
                 "PROSPECTIVE_CANDLE_NOT_YET_CLOSED")
    need(len(days) == 8 and books == set(BOOKS), "MISSING_BOOK_OR_DATE")
    need(all(sum(r["edition"] == b for r in c) == 8 for b in BOOKS),
         "BOOK_SAMPLE_SIZE_CHANGED")
    need(all(sum(r["family"] == f for r in c) >= 4 for f in FAMILIES),
         "INSUFFICIENT_FAMILY_DIVERSITY")


def read_only_public_klines(symbol: str, start_ms: int, end_ms: int):
    need(symbol in ALLOWED and type(start_ms) is int and type(end_ms) is int
         and 0 < start_ms < end_ms, "INVALID_PUBLIC_KLINE_PARAMS")
    query = urlencode({"symbol": symbol, "interval": "1h",
                       "startTime": start_ms, "endTime": end_ms,
                       "limit": BARS})
    request = Request(URL + "?" + query, method="GET",
                      headers={"User-Agent": "Williams-AI-v6-public-readonly-research"})
    with urlopen(request, timeout=15) as response:
        need(response.status == 200, "PUBLIC_MARKET_HTTP_FAILURE")
        body = response.read(300_000)
    need(len(body) < 300_000, "PUBLIC_MARKET_BODY_TOO_LARGE")
    return json.loads(body)


def normalized_chart(symbol: str, cutoff: datetime, raw: list):
    need(type(raw) is list and len(raw) == BARS, "INCOMPLETE_H1_HISTORY")
    end_ms = int(cutoff.timestamp() * 1000)
    start_ms = end_ms - BARS * MS
    out = []
    for i, kline in enumerate(raw):
        need(type(kline) is list and len(kline) >= 7,
             "MALFORMED_KLINE")
        open_ms = kline[0]
        close_ms = kline[6]
        need(type(open_ms) is int and type(close_ms) is int and
             open_ms == start_ms + i * MS and close_ms == open_ms + MS - 1,
             "FUTURE_OR_DISCONTINUOUS_H1")
        nums = []
        for val in kline[1:6]:
            need(type(val) in (str, int, float), "NONNUMERIC_KLINE_FIELD")
            try:
                number = float(val)
            except ValueError as exc:
                raise ValueError("V6_PUBLIC_MARKET_INTAKE_BAD_PRICE") from exc
            need(math.isfinite(number), "NONFINITE_KLINE_PRICE")
            nums.append(number)
        op, high, low, close, volume = nums
        need(0 < low <= min(op, close) <= max(op, close) <= high
             and volume >= 0, "INVALID_KLINE_OHLC")
        t1 = datetime.fromtimestamp(open_ms/1000, tz=timezone.utc)
        t2 = t1 + timedelta(hours=1)
        out.append({
            "open_utc": t1.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "close_utc": t2.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "open": op, "high": high, "low": low,
            "close": close, "volume": volume})
    return {"symbol": symbol, "bars": out}


def collect(plan: dict, *, fetcher=read_only_public_klines,
            now: datetime | None = None) -> tuple[dict, dict, dict]:
    """Injectable read-only fetcher; no outgoing network during offline tests."""
    if now is None:
        now = datetime.now(timezone.utc)
    check_plan(plan, now=now)
    metadata, provenance, chart_bytes = [], [], {}
    for planned in plan["cases"]:
        cutoff = _utc(planned["cutoff_utc"])
        end = int(cutoff.timestamp() * 1000)
        start = end - BARS * MS
        raw = fetcher(planned["symbol"], start, end - 1)
        chart = normalized_chart(planned["symbol"], cutoff, raw)
        blob = canonical(chart)
        file = "charts/" + planned["id"] + ".json"
        chart_bytes[file] = blob
        metadata.append({
            "id": planned["id"], "edition": planned["edition"],
            "family": planned["family"], "chart_file": file,
            "chart_sha256": sha256(blob).hexdigest(),
            "cutoff_utc": planned["cutoff_utc"],
            "source_pdf_sha256": PDF_SHA[planned["edition"]],
            "pdf_pages": planned["source_pdf_pages"],
            "question": planned["question"],
            "provenance_id": planned["provenance_id"],
        })
        provenance.append({
            "id": planned["id"], "symbol": planned["symbol"],
            "endpoint": URL, "interval": "1h",
            "startTime": start, "endTime": end - 1,
            "raw_response_sha256": sha256(canonical(raw)).hexdigest(),
            "market_source_not_expert_label": True,
        })
    cases = {
        "schema": SCHEMA, "dataset_id": plan["dataset_id"],
        "provenance_note": (
            "Forward-scheduled PUBLIC Binance USD-M H1 bars, no trading keys; "
            "external novelty and source correctness NOT independently audited, "
            "no expert-approved market labels or completed AI evaluation"),
        "cases": metadata,
    }
    return cases, {"schema": "WILLIAMS_V6_MARKET_SOURCE_RECEIPT_V1",
                   "rows": provenance, "data_only_unreviewed": True,
                   "chart_labels_authorized": False}, chart_bytes


def build_archive(plan: dict, out: Path, *, fetcher=read_only_public_klines,
                  now: datetime | None = None) -> dict:
    need(not out.exists(), "NO_OVERWRITE_PUBLIC_MARKET_PACKET")
    cases, receipts, charts = collect(plan, fetcher=fetcher, now=now)
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        for name, blob in charts.items():
            file = root / name
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_bytes(blob)
        manifest = validate_cases(cases, root)
        receipts["frozen_dataset_sha256"] = manifest["dataset_sha256"]
        with ZipFile(out, "x", compression=ZIP_DEFLATED) as archive:
            archive.writestr("cases.json", canonical(cases))
            archive.writestr("frozen_manifest.json", canonical(manifest))
            archive.writestr("provenance.json", canonical(receipts))
            for name, blob in sorted(charts.items()):
                archive.writestr(name, blob)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    manifest = build_archive(plan, args.out)
    print(json.dumps({"file": str(args.out), "manifest": manifest},
                     ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
