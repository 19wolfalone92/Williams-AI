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
import math
import re
from urllib.parse import urlsplit
from urllib.error import HTTPError, URLError
from urllib.request import Request, HTTPRedirectHandler, build_opener
from zipfile import ZipFile, BadZipFile

BASE="https://data.binance.vision/data/futures/um/"
HOUR_MS=3600*1000
ALLOWED=("BTCUSDT","ETHUSDT","SOLUSDT","XRPUSDT")
_HEX=re.compile(r"[A-Fa-f0-9]{64}$")
_RESOURCE=re.compile(r"^/data/futures/um/(daily|monthly)/klines/([A-Z0-9]+)/1h/([A-Z0-9]+)-1h-([0-9]{4}-[0-9]{2}(?:-[0-9]{2})?)[.]zip([.]CHECKSUM)?$")
_CACHE={}  # Never persisted; bounded to one ephemeral 4-symbol workflow run.


class _RejectArchiveRedirect(HTTPRedirectHandler):
    """Never issue an HTTP request to a redirected archive URL."""
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError(
            "BINANCE_ARCHIVE_REDIRECT_BLOCKED status="
            + str(code) + " resource=" + req.full_url
        )


def _safe_open(request, *, timeout):
    """Build a dedicated opener: the redirect must be rejected pre-follow."""
    return build_opener(_RejectArchiveRedirect()).open(request, timeout=timeout)


def _request(url, *, max_bytes):
    """Read ONLY the requested official USD-M daily/monthly resource.

    urllib follows redirects by default. Refuse ANY changed final resource,
    including a same-host redirect to another symbol/date or another archive.
    Never turn a network failure into synthetic data.
    """
    if type(url) is not str:
        raise ValueError("BINANCE_ARCHIVE_HOST_NOT_ALLOWLISTED")
    try:
        parts=urlsplit(url)
    except (ValueError,TypeError) as exc:
        raise ValueError("BINANCE_ARCHIVE_HOST_NOT_ALLOWLISTED") from exc
    match=_RESOURCE.fullmatch(parts.path)
    if (len(url)>280 or parts.scheme!="https" or
            parts.netloc!="data.binance.vision" or parts.query or
            parts.fragment or match is None or
            max_bytes not in (1024,6_000_000)):
        raise ValueError("BINANCE_ARCHIVE_HOST_NOT_ALLOWLISTED")
    frequency,directory,filename_symbol,stamp,suffix=match.groups()
    if (directory not in ALLOWED or filename_symbol!=directory or
            (frequency=="daily" and len(stamp)!=10) or
            (frequency=="monthly" and len(stamp)!=7) or
            (suffix is None and max_bytes!=6_000_000) or
            (suffix is not None and max_bytes!=1024)):
        raise ValueError("BINANCE_ARCHIVE_INVALID_RESOURCE_IDENTITY")
    try:
        datetime.strptime(stamp,"%Y-%m-%d" if frequency=="daily" else "%Y-%m")
    except ValueError as exc:
        raise ValueError("BINANCE_ARCHIVE_INVALID_CALENDAR_DATE") from exc
    try:
        with _safe_open(Request(url,headers={"User-Agent":"Williams-AI-v6-public-readonly"}),timeout=18) as r:
            if r.geturl()!=url:
                raise ValueError("BINANCE_ARCHIVE_REDIRECT_RESOURCE_CHANGED resource="+url)
            if r.status!=200:
                raise ValueError(f"BINANCE_ARCHIVE_HTTP_{r.status} resource={url}")
            raw=r.read(max_bytes+1)
    except HTTPError as exc:
        raise ValueError(f"BINANCE_ARCHIVE_HTTP_{exc.code} resource={url}") from exc
    except URLError as exc:
        raise ValueError(f"BINANCE_ARCHIVE_NETWORK_{type(exc.reason).__name__} resource={url}") from exc
    if not 0<len(raw)<=max_bytes:
        raise ValueError("BINANCE_ARCHIVE_SIZE_INVALID resource="+url)
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
    fields=checksum.strip().split()
    if (len(fields)!=2 or _HEX.fullmatch(fields[0]) is None or
            fields[1] not in (name,"*"+name)):
        raise ValueError("BINANCE_ARCHIVE_CHECKSUM_FILENAME_OR_FORMAT_INVALID")
    zip_raw=_request(BASE+path,max_bytes=6_000_000)
    if sha256(zip_raw).hexdigest().lower()!=fields[0].lower():
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
        try:
            op,high,low,close,volume=(float(v) for v in cells[1:6])
        except (TypeError,ValueError,OverflowError) as exc:
            raise ValueError("BINANCE_ARCHIVE_BAD_OHLCV_NUMBER") from exc
        if (not all(math.isfinite(v) for v in (op,high,low,close,volume)) or
                not 0<low<=min(op,close)<=max(op,close)<=high or volume<0):
            raise ValueError("BINANCE_ARCHIVE_INVALID_OHLCV_RANGE")
        rows.append([opening,*cells[1:6],closing,*cells[7:]])
    if not rows:
        raise ValueError("BINANCE_ARCHIVE_NO_KLINES")
    day=datetime.strptime(part_id,"%Y-%m-%d" if frequency=="daily" else "%Y-%m").replace(tzinfo=timezone.utc)
    if frequency=="monthly":
        y,m=_month_after(day.year,day.month)
        span_end=day.replace(year=y,month=m,day=1)
    else:
        span_end=day+timedelta(days=1)
    required=int((span_end-day).total_seconds()//3600)
    first_ms=int(day.timestamp()*1000)
    if len(rows)!=required or any(row[0]!=first_ms+i*HOUR_MS for i,row in enumerate(rows)):
        raise ValueError("BINANCE_ARCHIVE_DAY_MONTH_H1_INTEGRITY")
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
    return [requested[t].copy() for t in expected]
