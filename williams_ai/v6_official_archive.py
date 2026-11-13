"""Verified Binance's OFFICIAL public USD-M daily H1 archive, no API keys.

HTTP451 on api.binance.com is an IP/legal availability blocker in Actions.
The exchange publishes read-only daily CSV ZIPs at data.binance.vision.
This is a distinct published distribution channel, NOT a proxy for private
API calls; do not pretend archive prices are independently expert-reviewed.
"""
from __future__ import annotations

import csv
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from hashlib import sha256
from io import BytesIO, TextIOWrapper
import re
from urllib.request import Request, urlopen
from zipfile import ZipFile

BASE = "https://data.binance.vision/data/futures/um/daily/klines"
MS = 3600 * 1000
BARS = 1024
SYMBOLS = frozenset(("BTCUSDT","ETHUSDT","SOLUSDT","XRPUSDT"))
_DAY = re.compile(r"\d{4}-\d{2}-\d{2}\Z")


def need(test: bool, code: str):
    if not test:
        raise ValueError("V6_BINANCE_OFFICIAL_ARCHIVE_" + code)


def verified_day(symbol: str, day: str, *, http_get=None) -> list[list]:
    """Fetch .CHECKSUM and exact ZIP, SHA256-verify, parse/validate 24 H1 bars."""
    need(symbol in SYMBOLS and bool(_DAY.fullmatch(day)), "INVALID_SOURCE_COORDINATE")
    parsed = datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    filename = f"{symbol}-1h-{day}.zip"
    url = f"{BASE}/{symbol}/1h/{filename}"
    if http_get is None:
        def http_get(u, limit):
            with urlopen(Request(u,headers={"User-Agent":"Williams-v6-readonly"}),timeout=20) as r:
                need(r.status==200, "BAD_HTTP_STATUS")
                raw = r.read(limit + 1)
            need(len(raw)<=limit, "SOURCE_DOWNLOAD_TOO_LARGE")
            return raw
    checksum_text = http_get(url+".CHECKSUM", 512).decode("ascii").strip()
    tokens = checksum_text.split()
    need(len(tokens)==2 and bool(re.fullmatch("[0-9a-fA-F]{64}",tokens[0])) and
         tokens[1].lstrip("*") == filename, "INVALID_OFFICIAL_CHECKSUM_LINE")
    data = http_get(url, 1_000_000)
    need(sha256(data).hexdigest().lower()==tokens[0].lower(), "ARCHIVE_SHA256_MISMATCH")
    with ZipFile(BytesIO(data)) as z:
        members=z.infolist()
        need(len(members)==1 and members[0].filename==filename[:-4]+".csv"
             and not members[0].is_dir() and members[0].file_size<150_000,
             "UNEXPECTED_CSV_ARCHIVE_MEMBER")
        with z.open(members[0]) as fp:
            raw=list(csv.reader(TextIOWrapper(fp,encoding="utf-8-sig",newline="")))
    if raw and (raw[0][0] in ("open_time","Open time","openTime")):
        raw=raw[1:]
    need(len(raw)==24,"EXPECTED_24_H1_BARS_IN_PUBLISHED_DAY")
    base=int(parsed.timestamp()*1000)
    result=[]
    for n,row in enumerate(raw):
        need(len(row)>=7,"INCOMPLETE_DAILY_CSV_ROW")
        try:
            at=int(row[0])
            close_at=int(row[6])
        except (TypeError,ValueError) as exc:
            raise ValueError("V6_BINANCE_OFFICIAL_ARCHIVE_BAD_TIMESTAMP") from exc
        need(at==base+n*MS and close_at==at+MS-1,
             "DISCONTIGUOUS_OR_WRONG_TIMEFRAME")
        # Preserve the original open,high,low,close,volume strings.
        result.append([at,*row[1:6],close_at])
    return result


@lru_cache(maxsize=256)
def _day(symbol: str, date: str) -> tuple[tuple, ...]:
    return tuple(tuple(x) for x in verified_day(symbol, date))


def official_archive_klines(symbol: str, start_ms: int, end_ms: int) -> list[list]:
    """Read a strict consecutive 1024-H1 window from verified daily snapshots."""
    need(symbol in SYMBOLS and type(start_ms) is int and type(end_ms) is int
         and start_ms>0 and start_ms%MS==0
         and end_ms == start_ms+BARS*MS-1,"WINDOW_NOT_1024_CLOSED_HOURS")
    start=datetime.fromtimestamp(start_ms/1000,tz=timezone.utc)
    last=datetime.fromtimestamp((end_ms-1)/1000,tz=timezone.utc)
    date=start.date()
    days=[]
    while date<=last.date():
        days.extend(_day(symbol,date.isoformat()))
        date+=timedelta(days=1)
    window=[list(r) for r in days if start_ms<=r[0]<=end_ms]
    need(len(window)==BARS and
         all(r[0]==start_ms+i*MS for i,r in enumerate(window)),
         "MISSING_OR_NONCONTIGUOUS_ARCHIVED_H1_BARS")
    return window


ARCHIVE_SOURCE_ID="https://data.binance.vision/data/futures/um/daily/klines;SHA256_CSV_ZIP"
