"""Read-only, SHA-verified official Binance USD-M daily H1 archives.

Github-hosted runners cannot access Binance Futures REST in some regions
(HTTP 451). Use Binance's separate publicly accessible OFFICIAL daily
historical archive delivery, not a proxy. Never use trading credentials,
executable orders, adjusted candle cutoffs or unverified substitute data.

The exact experimental cutoffs remain frozen by the caller. If any daily
source is missing, abort; do not silently fall back to synthetic candles.
"""
from __future__ import annotations

import csv
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
from hashlib import sha256
import io
import json
import math
import re
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, build_opener, HTTPRedirectHandler
from zipfile import ZipFile

ROOT = "https://data.binance.vision/data/futures/um/daily/klines/"
HOUR_MS = 3_600_000
MAX_ZIP_BYTES = 2_000_000
_MAX_CSV_BYTES = 4_000_000
_SUPPORTED = frozenset({"BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"})


def _need(condition: bool, code: str):
    if not condition:
        raise ValueError("OFFICIAL_BINANCE_H1_" + code)


_URL_PATH = re.compile(
    r"/data/futures/um/daily/klines/([A-Z0-9]+)/1h/"
    r"([A-Z0-9]+)-1h-(\d{4}-\d{2}-\d{2})\.zip(\.CHECKSUM)?\Z"
)


def _valid_official_url(url: str) -> bool:
    """Allow exactly one public Binance USD-M daily H1 resource."""
    if not isinstance(url, str):
        return False
    try:
        parts = urlsplit(url)
    except ValueError:
        return False
    if (parts.scheme != "https" or parts.netloc != "data.binance.vision"
            or parts.query or parts.fragment or parts.username or parts.password):
        return False
    match = _URL_PATH.fullmatch(parts.path)
    if not match or match[1] != match[2] or match[1] not in _SUPPORTED:
        return False
    try:
        date.fromisoformat(match[3])
    except ValueError:
        return False
    return True


class _OfficialOnlyRedirect(HTTPRedirectHandler):
    """Prevent redirecting an approved request to a different archive."""
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _need(newurl == req.full_url and _valid_official_url(newurl),
              f"UNTRUSTED_ARCHIVE_REDIRECT url={req.full_url} actual={newurl}")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _open_official(request: Request, timeout: int):
    return build_opener(_OfficialOnlyRedirect()).open(request, timeout=timeout)


def _download(url: str, limit: int) -> bytes:
    _need(_valid_official_url(url) and
          ((url.endswith(".CHECKSUM") and limit == 512) or
           (url.endswith(".zip") and limit == MAX_ZIP_BYTES)),
          "INVALID_OFFICIAL_URL")
    request = Request(url, method="GET", headers={
        "User-Agent": "Williams-AI-v6-official-public-historic-H1"})
    try:
        with _open_official(request, timeout=20) as response:
            _need(response.status == 200,
                  f"ARCHIVE_HTTP_FAILURE status={response.status} url={url}")
            _need(response.geturl() == url,
                  f"UNTRUSTED_ARCHIVE_REDIRECT url={url} actual={response.geturl()}")
            data = response.read(limit + 1)
    except HTTPError as exc:
        raise ValueError(
            f"OFFICIAL_BINANCE_H1_ARCHIVE_HTTP_FAILURE status={exc.code} url={url}"
        ) from exc
    except (URLError, TimeoutError) as exc:
        raise ValueError(
            f"OFFICIAL_BINANCE_H1_ARCHIVE_NETWORK_FAILURE url={url} reason={type(exc).__name__}"
        ) from exc
    _need(0 < len(data) <= limit, f"INVALID_ARCHIVE_SIZE url={url}")
    return data

def verified_daily(symbol: str, day: date, *,
                   fetch=_download) -> tuple[list, dict]:
    _need(symbol in _SUPPORTED and type(day) is date, "INVALID_SYMBOL_OR_DATE")
    filename = f"{symbol}-1h-{day.isoformat()}.zip"
    url = ROOT + f"{symbol}/1h/{filename}"
    text = fetch(url + ".CHECKSUM", 512).decode("ascii").strip()
    fields = text.split()
    _need(len(fields) in (1, 2), "INVALID_CHECKSUM_RECORD")
    expected = fields[0]
    _need(re.fullmatch(r"[0-9a-fA-F]{64}", expected) is not None,
          "NO_PUBLISHED_SIGNED_CHECKSUM")
    if len(fields) == 2:
        _need(fields[1].lstrip("*") == filename, "CHECKSUM_FILENAME_MISMATCH")
    packed = fetch(url, MAX_ZIP_BYTES)
    digest = sha256(packed).hexdigest()
    _need(digest.lower() == expected.lower(), "OFFICIAL_CHECKSUM_MISMATCH")
    with ZipFile(io.BytesIO(packed)) as archive:
        names = archive.namelist()
        expected_csv = filename[:-4] + ".csv"
        _need(len(names) == 1 and names[0] == expected_csv,
              "UNEXPECTED_ARCHIVE_MEMBER")
        _need(archive.getinfo(expected_csv).file_size <= _MAX_CSV_BYTES,
              "UNBOUNDED_ARCHIVE_CONTENT")
        with archive.open(expected_csv) as fd:
            records = list(csv.reader(io.TextIOWrapper(
                fd, encoding="utf-8-sig", newline="")))
    if records and records[0] and records[0][0].strip() == "open_time":
        records.pop(0)
    _need(len(records) == 24, "INCOMPLETE_OFFICIAL_UTC_DAY")
    first = int(datetime(day.year, day.month, day.day,
                         tzinfo=timezone.utc).timestamp() * 1000)
    series = []
    for i, row in enumerate(records):
        _need(type(row) is list and len(row) >= 7, "MALFORMED_KLINE_CSV")
        try:
            start, close = int(row[0]), int(row[6])
        except (ValueError, TypeError) as exc:
            raise ValueError("OFFICIAL_BINANCE_H1_INVALID_TIME_FIELD") from exc
        _need(start == first + HOUR_MS * i and
              close == start + HOUR_MS - 1, "DISCONTINUOUS_OFFICIAL_H1")
        try:
            opening, high, low, closing, volume = map(float, row[1:6])
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError("OFFICIAL_BINANCE_H1_INVALID_OHLCV") from exc
        _need(all(math.isfinite(v) for v in (opening, high, low, closing, volume))
              and low > 0 and volume >= 0 and low <= opening <= high
              and low <= closing <= high,
              "INVALID_OHLCV_RANGE")
        series.append([start, *row[1:6], close])
    return series, {
        "utc_date": day.isoformat(), "sha256": digest,
        "official_checksum_verified": True, "source": url,
        "H1_count": len(series),
    }


@lru_cache(maxsize=512)
def _cached_daily(symbol: str, date_iso: str) -> tuple[list, dict]:
    day = date.fromisoformat(date_iso)
    return verified_daily(symbol, day)


def read_only_public_archives_klines(
        symbol: str, start_ms: int, end_ms: int) -> tuple[list, dict]:
    """Return 1024 closed H1 klines plus independently verified ZIP provenance.

    Network calls are cached per symbol/date for overlapping frozen cases;
    no recent REST requests or credential-based endpoints are issued.
    """
    _need(symbol in _SUPPORTED and type(start_ms) is int
          and type(end_ms) is int and start_ms % HOUR_MS == 0
          and end_ms % HOUR_MS == HOUR_MS - 1
          and end_ms - start_ms + 1 == 1024 * HOUR_MS,
          "INVALID_CLOSED_H1_WINDOW")
    begin = datetime.fromtimestamp(start_ms/1000, timezone.utc).date()
    last = datetime.fromtimestamp(end_ms/1000, timezone.utc).date()
    data = []
    receipts = []
    d = begin
    while d <= last:
        rows, receipt = _cached_daily(symbol, d.isoformat())
        data.extend(r for r in rows if start_ms <= r[0] <= end_ms)
        receipts.append(receipt)
        d += timedelta(days=1)
    _need(len(data) == 1024, "MISSING_CLOSED_H1_BARS")
    _need(all(data[i][0] == start_ms + i*HOUR_MS
              and data[i][6] == start_ms + (i+1)*HOUR_MS - 1
              for i in range(1024)), "NONCONTIGUOUS_CLOSED_H1_BARS")
    return data, {
        "archive_data_source": ROOT,
        "official_checksum_verified_all": True,
        "daily_sources": receipts,
        "public_data_only_not_expert_gold": True,
    }
