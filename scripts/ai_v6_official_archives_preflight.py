"""Standalone read-only 200-resource Binance preflight for the immutable pilot.

Never mutates the pilot request, trains LoRA, exports private data or places orders.
The explicit --probe downloads only official SHA-verified public market archives
and validates all frozen 1024-closed-H1 windows; no substitute candles allowed.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from hashlib import sha1
import json
from pathlib import Path

from williams_ai.v6_official_archive_collector import (
    ROOT, HOUR_MS, _cached_daily, read_only_public_archives_klines,
)

PINNED_PILOT_PLAN_BLOB = "b5ca8bb4168ff192e510bb2d1df640921f933ef7"
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")
CUTOFF_DATES = tuple(f"2026-11-{d:02}T00:00:00Z" for d in range(15, 23))
SCHEMA = "WILLIAMS_AI_V6_BINANCE_READONLY_PREFLIGHT_V2"


def load_pinned_plan(path: Path) -> dict:
    raw = path.read_bytes()
    git_blob = sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
    if git_blob != PINNED_PILOT_PLAN_BLOB:
        raise ValueError("V6_PREFLIGHT_UNPINNED_PLAN_NO_NETWORK")
    plan = json.loads(raw)
    if (type(plan) is not dict or
            plan.get("schema") != "WILLIAMS_V6_FORWARD_CHART_PLAN_V1" or
            plan.get("max_bars_per_case") != 1024):
        raise ValueError("V6_PREFLIGHT_UNEXPECTED_PLAN")
    return plan


def required_sources(plan: dict) -> list[dict]:
    cases = plan.get("cases")
    if type(cases) is not list or len(cases) != 32:
        raise ValueError("V6_PREFLIGHT_EXPECTED_FROZEN_32_CASES")
    seen, required = set(), set()
    for case in cases:
        if (type(case) is not dict or case.get("symbol") not in SYMBOLS or
                case.get("cutoff_utc") not in CUTOFF_DATES):
            raise ValueError("V6_PREFLIGHT_UNEXPECTED_CASE_OR_TIME")
        key = (case["symbol"], case["cutoff_utc"])
        if key in seen:
            raise ValueError("V6_PREFLIGHT_DUPLICATE_CASE")
        seen.add(key)
        cutoff = datetime.fromisoformat(case["cutoff_utc"].replace("Z", "+00:00"))
        cursor = (cutoff-timedelta(hours=1024)).date()
        last = (cutoff-timedelta(hours=1)).date()
        while cursor <= last:
            required.add((case["symbol"], cursor))
            cursor += timedelta(days=1)
    if len(required) != 200 or len(seen) != 32:
        raise ValueError("V6_PREFLIGHT_UNEXPECTED_SOURCE_COVERAGE")
    return [{"symbol": sym, "utc_date": day.isoformat(),
             "zip_url": ROOT+f"{sym}/1h/{sym}-1h-{day}.zip",
             "checksum_url": ROOT+f"{sym}/1h/{sym}-1h-{day}.zip.CHECKSUM",
             "fingerprint": f"BINANCE_ARCHIVE|{sym}|{day}"}
            for sym, day in sorted(required)]


def _cached_fetch(symbol: str, day: date):
    return _cached_daily(symbol, day.isoformat())


def probe_one(row: dict, *, fetch=_cached_fetch) -> dict:
    """Real SHA/24h verification; HTTP status and resource retained on failure."""
    try:
        _, receipt = fetch(row["symbol"], date.fromisoformat(row["utc_date"]))
        if (receipt.get("source") != row["zip_url"] or
                receipt.get("official_checksum_verified") is not True or
                receipt.get("H1_count") != 24):
            raise ValueError("V6_PREFLIGHT_RECEIPT_MISMATCH")
        return {**row, "status": "VERIFIED_24_H1_SHA256",
                "zip_sha256": receipt["sha256"]}
    except Exception as exc:  # Any unexpected archive failure remains BLOCKED.
        return {**row, "status": "BLOCKED", "error_type": type(exc).__name__,
                "reason": str(exc)[:600]}


def evaluate(plan: dict, rows: list[dict], *, fetch=_cached_fetch,
             verify_windows=read_only_public_archives_klines) -> dict:
    """Bounded concurrent source audit; only verify windows if all days pass."""
    from functools import partial
    with ThreadPoolExecutor(max_workers=8) as pool:
        outcomes = list(pool.map(partial(probe_one, fetch=fetch), rows))
    failures = [r for r in outcomes if r["status"] != "VERIFIED_24_H1_SHA256"]
    verified_windows = []
    if not failures:
        for case in plan["cases"]:
            cutoff = datetime.fromisoformat(case["cutoff_utc"].replace("Z", "+00:00"))
            end = int(cutoff.timestamp()*1000)
            start = end-1024*HOUR_MS
            try:
                bars, provenance = verify_windows(case["symbol"], start, end-1)
                if (len(bars) != 1024 or
                        provenance.get("official_checksum_verified_all") is not True):
                    raise ValueError("V6_PREFLIGHT_WINDOW_OR_PROVENANCE_MISMATCH")
                verified_windows.append({"case_id": case["id"], "symbol": case["symbol"],
                                         "cutoff_utc": case["cutoff_utc"],
                                         "closed_h1_bars": 1024, "status": "VERIFIED"})
            except Exception as exc:  # No unchecked archive/window can pass.
                failures.append({"case_id": case["id"], "status": "BLOCKED",
                                 "reason": str(exc)[:600]})
    return {"schema": SCHEMA, "plan_git_blob": PINNED_PILOT_PLAN_BLOB,
            "expected_unique_sources": len(rows), "actual_source_probes": len(outcomes),
            "verified_daily_archives": len(outcomes) -
                 sum(r["status"] != "VERIFIED_24_H1_SHA256" for r in outcomes),
            "verified_1024_h1_windows": len(verified_windows),
            "blocked_count": len(failures), "pass": not failures and
                 len(verified_windows) == 32, "sources": outcomes,
            "windows": verified_windows, "failures": failures,
            "training_started": False, "pilot_triggered": False,
            "independent_expert_review_done": False}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--probe", action="store_true",
                   help="Explicit read-only GET of 200 official archives; no pilot rerun")
    args = p.parse_args()
    if args.out.exists() or args.out.is_symlink():
        raise ValueError("V6_PREFLIGHT_NO_OVERWRITE")
    plan = load_pinned_plan(args.plan)
    rows = required_sources(plan)
    if args.probe:
        report = evaluate(plan, rows)
    else:
        report = {"schema": SCHEMA, "plan_git_blob": PINNED_PILOT_PLAN_BLOB,
                  "expected_unique_sources": len(rows), "actual_source_probes": 0,
                  "verified_daily_archives": 0, "verified_1024_h1_windows": 0,
                  "blocked_count": len(rows), "pass": False,
                  "sources": [{**r, "status": "NOT_PROBED"} for r in rows],
                  "windows": [], "failures": [],
                  "training_started": False, "pilot_triggered": False,
                  "independent_expert_review_done": False}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as output:
        json.dump(report, output, indent=2, sort_keys=True, ensure_ascii=False)
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("sources", "windows", "failures")}, sort_keys=True))
    return 0 if report["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
