"""Verify official Binance USD-M H1 historical ZIP + SHA256 checksum fallback.

Use ONLY if the public REST klines endpoint explicitly returns HTTP 451/403.
No exchange API credentials, trading, model loading, training or case labels.
Fails closed on any archive integrity/timestamp ambiguity. ZIP paths are fixed.
"""
from __future__ import annotations

import csv
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import io
import re
from urllib.request import Request, urlopen
from zipfile import ZipFile, BadZipFile

BASE="https://data.binance.vision/data/futures/um/"
HOUR_MS=3600*1000
ALLOWED=("BTCUSDT","ETHUSDT","SOLUSDT","XRPUSDT")
_HEX=re.compile(r"(?<![A-Fa-f0-9])([a-fA-F0-9]{64})(?![A-Fa-f0-9])")
_CACHE={}  # Never persisted; bounded to one ephemeral 4-symbol workflow run.


def _request(url, *, max_bytes):
    if not url.startswith(BASE) or len(url)>280:
        raise ValueError("BINANCE_ARCHIVE_HOST_NOT_ALLOWLISTED")
    with urlopen(Request(url,headers={"User-Agent":"Williams-AI-v6-public-readonly"}),timeout=18) as r:
        if r.status != 200:
            raise ValueError("BINANCE_ARCHIVE_HTTP_NOT_200")
        raw=r.read(max_bytes+1)
    if not 0<len(raw)<=max_bytes:
        raise ValueError("BINANCE_ARCHIVE_SIZE_INVALID")
    return raw


def _month_after(y,m):
    return (y+1,1) if m==12 else (y,m+1)


def _archive_parts(start_ms,end_ms):
    first=datetime.fromtimestamp(start_ms/1000,tz=timezone.utc)
    last=datetime.fromtimestamp(end_ms/1000,tz=timezone.utc)
    day=first.replace(hour=0,minute=0,second=0,microsecond=0)
    endday=last.replace(hour=0,minute=0,second=0,microsecond=0)
    while day<=endday:
        y,m=day.year,day.month
        ny,nm=_month_after(y,m)
        next_month=day.replace(year=ny,month=nm,day=1)
        current_month=day.replace(day=1)
        # Entire month falls within the requested window? Prefer monthly ZIP,
        # avoiding ~30 daily HTTP calls and reducing 451 fallback cost.
        if day==current_month and day+timedelta(days=1)!=next_month and \
           first<=current_month and last>=next_month-timedelta(milliseconds=1):
            yield "monthly",f"{y:04d}-{m:02d}",next_month
            day=next_month
        else:
            yield "daily",day.strftime("%Y-%m-%d"),day+timedelta(days=1)
            day+=timedelta(days=1)


def _load_csv_zip(symbol,frequency,part_id):
    key=(symbol,frequency,part_id)
    if key in _CACHE:
        return _CACHE[key]
    name=f"{symbol}-1h-{part_id}.zip"
    path=f"{frequency}/klines/{symbol}/1h/{name}"
    checksum=_request(BASE+path+".CHECKSUM",max_bytes=1024).decode("ascii","strict")
    match=_HEX.search(checksum)
    if not match:
        raise ValueError("BINANCE_ARCHIVE_CHECKSUM_FORMAT_INVALID")
    zip_raw=_request(BASE+path,max_bytes=6_000_000)
    if sha256(zip_raw).hexdigest().lower()!=match.group(1).lower():
        raise ValueError("BINANCE_ARCHIVE_SHA256_MISMATCH")
    try:
        with ZipFile(io.BytesIO(zip_raw)) as archive:
            files=archive.infolist()
            if len(files)!=1 or files[0].filename!=name[:-4]+".csv" or files[0].file_size>3_500_000:
                raise ValueError("BINANCE_ARCHIVE_WRONG_CSV_MEMBER")
            data=archive.read(files[0]).decode("utf-8-sig","strict")
    except BadZipFile as exc:
        raise ValueError("BINANCE_ARCHIVE_BAD_ZIP") from exc
    rows=[]
    for cells in csv.reader(io.StringIO(data)):
        if not cells:
            continue
        if cells[0].strip().lower() in ("open_time","opentime"):
            continue
        if len(cells)!=12:
            raise ValueError("BINANCE_ARCHIVE_INVALID_KLINE_COLUMNS")
        try:
            opening=int(cells[0])
            closing=int(cells[6])
        except ValueError as exc:
            raise ValueError("BINANCE_ARCHIVE_BAD_TIMESTAMP") from exc
        if opening<0 or closing-opening!=HOUR_MS-1:
            raise ValueError("BINANCE_ARCHIVE_BAD_HOUR_MILLISECONDS")
        rows.append([opening,*cells[1:6],closing,*cells[7:]])
    if not rows:
        raise ValueError("BINANCE_ARCHIVE_NO_KLINES")
    if len(_CACHE)>=150:
        _CACHE.clear()  # fixed bound; never treat eviction as dropping data
    _CACHE[key]=rows
    return rows


def official_archive_klines(symbol:str,start_ms:int,end_ms:int)->list:
    """Return EXACT 1024 1h klines after independent checksum verification."""
    if symbol not in ALLOWED or type(start_ms) is not int or type(end_ms) is not int or \
       start_ms<=0 or end_ms<start_ms or start_ms%HOUR_MS or \
       (end_ms+1)%HOUR_MS or (end_ms+1-start_ms)//HOUR_MS!=1024:
        raise ValueError("BINANCE_ARCHIVE_REJECTED_WINDOW")
    requested={}
    for frequency,part_id,_ in _archive_parts(start_ms,end_ms):
        for row in _load_csv_zip(symbol,frequency,part_id):
            opening=row[0]
            if start_ms<=opening<=end_ms:
                if opening in requested:
                    raise ValueError("BINANCE_ARCHIVE_OVERLAP_DUPLICATE")
                requested[opening]=row
    expected=[start_ms+i*HOUR_MS for i in range(1024)]
    if sorted(requested)!=expected:
        raise ValueError("BINANCE_ARCHIVE_INCOMPLETE_CONTIGUOUS_HISTORY")
    return [requested[t] for t in expected]
