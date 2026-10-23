"""Previously unused, SHA-256 verified public Binance USD-M 2025 H1 cohort.

This loader imports ONLY 2025-03 and 2025-04 monthly USD-M Futures archives.
No REST-account endpoint, keys, private files, model weights, fitting, or
external trading calls. A failure to fetch authentic continuous closed bars
FAILS THE TEST, and is never substituted with old in-sample research data.

2025-03/04 are predeclared here before the first run. Once model outputs have
been inspected, this cohort is "SEEN" and must never be called untouched again.
Binance archives are external public historical data, not an independent
expert Elliott wave ground-truth.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import zipfile
from datetime import datetime,timezone
from pathlib import Path
from typing import Any,Callable
from urllib.request import Request,urlopen

from .chart_reader import validate_closed_bars

SYMBOLS=(
    "ADAUSDT","AVAXUSDT","BCHUSDT","BNBUSDT","BTCUSDT",
    "DOGEUSDT","DOTUSDT","ETHUSDT","LINKUSDT","LTCUSDT",
    "NEARUSDT","SOLUSDT","SUIUSDT","TRXUSDT","XRPUSDT",
)
MONTHS=("2025-03","2025-04")
HOST="https://data.binance.vision"
HOUR_MS=3_600_000
KNOWN_PRIOR_FIRST_DATE="2026-07-14"
MAX_CSV_BYTES=500_000
MAX_ZIP_BYTES=350_000


def _fetch(url:str)->bytes:
    if not url.startswith(
        HOST+"/data/futures/um/monthly/klines/"
    ) or not (url.endswith(".zip") or url.endswith(".zip.CHECKSUM")):
        raise ValueError("Not an explicitly approved public USD-M archive URL")
    req=Request(url,headers={"User-Agent":"Williams-AI-2025-blind-H1-research",
                             "Accept":"application/octet-stream"})
    with urlopen(req,timeout=25) as response:
        buf=response.read(MAX_ZIP_BYTES+1)
    if len(buf)>MAX_ZIP_BYTES:
        raise ValueError("Public archive exceeds capped download size")
    return buf


def fetch_month(symbol:str,month:str, *,
                downloader:Callable[[str],bytes]=_fetch
                )->tuple[list[dict[str,Any]],dict[str,Any]]:
    if symbol not in SYMBOLS or month not in MONTHS:
        raise ValueError("Unapproved symbol or month; no retrospective cherry-picking")
    name=f"{symbol}-1h-{month}.zip"
    base=f"{HOST}/data/futures/um/monthly/klines/{symbol}/1h/{name}"
    zip_bytes=downloader(base)
    checksum=downloader(base+".CHECKSUM")
    if not isinstance(zip_bytes,bytes) or not isinstance(checksum,bytes):
        raise ValueError("Expected exact binary ZIP and CHECKSUM")
    if len(checksum)>512:
        raise ValueError("Oversize archive checksum")
    m=re.fullmatch(
        rb"([a-fA-F0-9]{64})\s+\*?"+re.escape(name.encode())+rb"\s*",
        checksum
    )
    if m is None:
        raise ValueError("Malformed remote Binance CHECKSUM")
    actual=hashlib.sha256(zip_bytes).hexdigest()
    if actual!=m.group(1).decode().lower():
        raise ValueError("BINANCE_ARCHIVE_SHA256_MISMATCH")
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        members=archive.infolist()
        if len(members)!=1 or members[0].filename!=name[:-4]+".csv" or (
            members[0].file_size>MAX_CSV_BYTES or
            members[0].compress_size>MAX_ZIP_BYTES
        ):
            raise ValueError("Unexpected archive members or decompressed file size")
        text=archive.read(members[0]).decode("utf-8-sig")
    rows=[]
    for parts in csv.reader(io.StringIO(text)):
        if not parts:continue
        if parts[0].lower().replace(" ","_") in ("open_time","opentime"):
            continue
        if len(parts)<7:
            raise ValueError("Malformed public Binance futures CSV row")
        open_ms=int(parts[0])
        close_ms=int(parts[6])
        if open_ms%HOUR_MS or close_ms!=open_ms+HOUR_MS-1:
            raise ValueError("Public H1 timestamp misaligned")
        candle=[float(parts[i]) for i in (1,2,3,4,5)]
        op,hi,lo,cl,vol=candle
        if not all(math.isfinite(x) for x in candle) or (
            op<=0 or hi<=0 or lo<=0 or cl<=0 or vol<0
            or lo>min(op,cl) or hi<max(op,cl)
        ):
            raise ValueError("Malformed public H1 OHLC or base volume")
        rows.append({
            "close_time":datetime.fromtimestamp(close_ms/1000,timezone.utc).isoformat(),
            "open":op,"high":hi,"low":lo,"close":cl,
            "volume":vol,"closed":True,
        })
    if len(rows)<650 or len(rows)>745:
        raise ValueError("Unexpected monthly H1 count")
    start=datetime.fromisoformat(rows[0]["close_time"])
    if start.strftime("%Y-%m")!=month or start.day!=1:
        raise ValueError("Public monthly first H1 candle mismatched")
    expected=int(month.split("-")[0])*12+int(month.split("-")[1])
    for a,b in zip(rows,rows[1:]):
        t0=datetime.fromisoformat(a["close_time"])
        t1=datetime.fromisoformat(b["close_time"])
        if int(round((t1-t0).total_seconds()))!=3600:
            raise ValueError("Hourly archive contains a gap or duplicate")
    return rows,{
        "symbol":symbol,"month":month,"archive_filename":name,
        "archive_sha256":actual,
        "closed_h1_rows":len(rows),
        "venue":"BINANCE_USDT_M_FUTURES_PUBLIC_ARCHIVE",
        "base_volume_is_not_tick_count":True,
    }


def make_never_used_cohort(*,
                           downloader:Callable[[str],bytes]=_fetch
                           )->dict[str,Any]:
    markets={}
    receipts=[]
    for symbol in SYMBOLS:
        all_rows=[]
        for month in MONTHS:
            batch,source=fetch_month(symbol,month,downloader=downloader)
            receipts.append(source)
            all_rows.extend(batch)
        last=all_rows[-1]["close_time"]
        candles,gaps=validate_closed_bars(all_rows,"H1",last)
        if len(candles)!=len(all_rows) or gaps:
            raise ValueError("Market has incomplete source history")
        if len(all_rows)<1200:
            raise ValueError("Need at least 1200 previously unused H1 candles")
        for a,b in zip(candles,candles[1:]):
            if int(round((b.at-a.at).total_seconds()))!=3600:
                raise ValueError("Market OHLCH1 must have exact cadence")
        if any(x["close_time"][:7] not in MONTHS for x in all_rows):
            raise ValueError("Out-of-cohort candle leaked")
        markets[symbol]={
            "symbol":symbol,"tick_size":None,"bars":all_rows,
        }
    summary={
        "schema":"WILLIAMS_AI_EXPLICIT_2025_MAR_APR_UNSEEN_COHORT_V1",
        "sources":receipts,
        "markets":markets,
        "intentionally_predeclared_dates":list(MONTHS),
        "was_in_previous_2026_project_datasets":False,
        "model_training_on_this_cohort":False,
        "independent_expert_wave_labels":None,
        "source_volumes_are_NOT_TC1_tick_count":True,
        "trade_action":None,"execution_allowed":False,
    }
    digest=hashlib.sha256(json.dumps(summary,sort_keys=True,
                         separators=(",",":"),allow_nan=False).encode()).hexdigest()
    return {**summary,"source_manifest_sha256":digest}


def main():
    p=argparse.ArgumentParser(description="Retrieve new 2025 Binance USD-M H1 research cohort")
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    data=make_never_used_cohort()
    if args.out.exists():
        raise FileExistsError("Never silently overwrite a pinned source manifest")
    with args.out.open("x",encoding="utf-8") as f:
        json.dump(data,f,ensure_ascii=False,separators=(",",":"))
    print("NEW_2025_BINANCE_COHORT",
          json.dumps({
              "sha":data["source_manifest_sha256"],
              "symbols":len(data["markets"]),
              "monthly_zip_files":len(data["sources"]),
              "months":list(MONTHS),
              "closed_h1_bars":sum(len(m["bars"]) for m in data["markets"].values()),
              "real_trades":0,
          },sort_keys=True),flush=True)


if __name__=="__main__":
    main()
