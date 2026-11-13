"""Official Binance PUBLIC USD-M 1h ZIP+CHECKSUM source (read-only, no keys).

Alternative to the Binance Futures REST endpoint which returns HTTP 451
from some GitHub-hosted runners. Binance documents this separate public
archive at data.binance.vision. Only reads published archive packages;
never accesses account APIs or attempts to circumvent private endpoints.
Official ZIP+CHECKSUM are checked byte-for-byte, but are NOT signatures.
"""
from __future__ import annotations

import csv
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from io import BytesIO, StringIO
from pathlib import Path
import re
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from zipfile import ZipFile

from .v6_public_chart_collector import ALLOWED, BARS, MS

PUBLIC_ARCHIVE = "https://data.binance.vision/data/futures/um"
MAX_ARCHIVE_BYTES = 2_000_000
MAX_CSV_BYTES = 3_000_000


def require(ok: bool, why: str) -> None:
    if not ok:
        raise ValueError("V6_OFFICIAL_PUBLIC_ARCHIVE_" + why)


def public_get(url: str, limit: int) -> bytes:
    require(url.startswith(PUBLIC_ARCHIVE + "/")
            and re.fullmatch(r"https://data[.]binance[.]vision/data/futures/um/"
                             r"(?:monthly|daily)/klines/[A-Z0-9]+/1h/"
                             r"[A-Z0-9]+-1h-[0-9-]+[.]zip(?:[.]CHECKSUM)?", url),
            "INVALID_PUBLIC_ARCHIVE_URL")
    request = Request(url, method="GET",
                      headers={"User-Agent": "Williams-AI-v6-read-only-official-archive"})
    try:
        with urlopen(request, timeout=22) as r:
            require(r.status == 200, "HTTP_NOT_SUCCESS")
            require(r.url.startswith(PUBLIC_ARCHIVE + "/"), "REDIRECT_OUTSIDE_OFFICIAL_HOST")
            body = r.read(limit + 1)
            require(len(body) <= limit, "ZIP_OR_CHECKSUM_TOO_LARGE")
            return body
    except HTTPError as err:
        if err.code == 404:
            raise FileNotFoundError("OFFICIAL_ARCHIVE_NOT_PUBLISHED_YET") from err
        # HTTP 451 or other failures remain explicit: never hide or invent bars.
        raise


def _url(symbol: str, grouping: str, date_part: str) -> tuple[str, str]:
    require(symbol in ALLOWED and grouping in ("monthly", "daily")
            and re.fullmatch(
                r"[0-9]{4}-[0-9]{2}" if grouping == "monthly" else
                r"[0-9]{4}-[0-9]{2}-[0-9]{2}", date_part),
            "UNAUTHORIZED_ARCHIVE_PACKAGE")
    base = f"{symbol}-1h-{date_part}"
    return (f"{PUBLIC_ARCHIVE}/{grouping}/klines/{symbol}/1h/{base}.zip", base)


def verified_rows(symbol: str, grouping: str, date_part: str, *,
                  fetch=public_get) -> tuple[list[list], dict]:
    url, name = _url(symbol, grouping, date_part)
    checklist = fetch(url + ".CHECKSUM", 300).decode("ascii").strip()
    parts = checklist.split()
    require(len(parts) == 2 and
            re.fullmatch(r"[A-Fa-f0-9]{64}", parts[0]) is not None and
            parts[1].lstrip("*") == name + ".zip", "INVALID_CHECKSUM_MANIFEST")
    archive = fetch(url, MAX_ARCHIVE_BYTES)
    checksum = sha256(archive).hexdigest()
    require(checksum == parts[0].lower(), "PUBLISHED_SHA256_MISMATCH")
    with ZipFile(BytesIO(archive)) as zipped:
        files = zipped.infolist()
        require(len(files) == 1 and
                files[0].filename == name + ".csv" and
                0 < files[0].file_size <= MAX_CSV_BYTES and
                not files[0].is_dir(), "UNSAFE_OR_UNEXPECTED_ARCHIVE_MEMBER")
        csv_bytes = zipped.read(files[0])
    require(len(csv_bytes) <= MAX_CSV_BYTES, "DECOMPRESSED_ARCHIVE_TOO_LARGE")
    rows = list(csv.reader(StringIO(csv_bytes.decode("utf-8-sig"))))
    if rows and rows[0][0].strip().lower().replace(" ", "_") in (
            "open_time", "opentime"):
        rows = rows[1:]
    normalized = []
    for row in rows:
        require(len(row) >= 7 and row[0].isdigit() and row[6].isdigit(),
                "INVALID_BINANCE_UM_CSV_ROW")
        # USD-M archive uses millisecond timestamps, not spot microseconds.
        a, b = int(row[0]), int(row[6])
        require(1_000_000_000_000 < a < 10_000_000_000_000
                and b == a + MS - 1 and a % MS == 0,
                "UM_1H_TIMESTAMP_UNITS_OR_CLOSURE_INVALID")
        open_time = datetime.fromtimestamp(a / 1000, tz=timezone.utc)
        require(open_time.strftime(
            "%Y-%m" if grouping == "monthly" else "%Y-%m-%d") == date_part,
            "BAR_NOT_IN_EXPECTED_OFFICIAL_PACKAGE")
        # Keep exact archive decimal strings for forensic provenance.
        normalized.append([a, *row[1:6], b, *row[7:]])
    return normalized, {
        "source_url": url, "checksum_url": url + ".CHECKSUM",
        "verified_archive_sha256": checksum,
        "raw_csv_sha256": sha256(csv_bytes).hexdigest(),
        "rows": len(normalized),
        "kind": grouping, "sha256_is_official_published_not_digital_signature": True,
    }


class OfficialUmArchiveFetcher:
    """Caches checksum-verified public monthly/daily packages across 32 cases."""
    origin = PUBLIC_ARCHIVE + "/monthly-or-daily/klines"
    def __init__(self, *, fetch=public_get) -> None:
        self.fetch = fetch
        self.cache: dict[tuple[str, str, str], tuple[list, dict]] = {}
        self.archive_provenance: dict[str, dict] = {}

    def _one(self, symbol: str, grouping: str, datepart: str) -> list[list]:
        key = (symbol, grouping, datepart)
        if key not in self.cache:
            rows, proof = verified_rows(symbol, grouping, datepart, fetch=self.fetch)
            self.cache[key] = (rows, proof)
            self.archive_provenance[proof["source_url"]] = proof
        return self.cache[key][0]

    def __call__(self, symbol: str, start_ms: int, end_ms: int) -> list[list]:
        require(symbol in ALLOWED and type(start_ms) is int and type(end_ms) is int
                and 0 < start_ms < end_ms and
                end_ms - start_ms + 1 == BARS * MS and
                start_ms % MS == 0 and (end_ms + 1) % MS == 0,
                "INVALID_1024H_WINDOW")
        start = datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc)
        end = datetime.fromtimestamp((end_ms + 1) / 1000, tz=timezone.utc)
        by_open: dict[int, list] = {}
        month = start.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        while month < end:
            following = (month.replace(year=month.year+1, month=1)
                         if month.month == 12 else month.replace(month=month.month+1))
            month_name = month.strftime("%Y-%m")
            try:
                chunks = [self._one(symbol, "monthly", month_name)]
            except FileNotFoundError:
                # Official daily ZIPs are normally published next day.
                chunks = []
                day = max(month, start.replace(hour=0, minute=0, second=0, microsecond=0))
                while day < min(following, end):
                    chunks.append(self._one(symbol, "daily", day.strftime("%Y-%m-%d")))
                    day += timedelta(days=1)
            for chunk in chunks:
                for row in chunk:
                    open_ms = row[0]
                    if start_ms <= open_ms <= end_ms:
                        if open_ms in by_open:
                            require(by_open[open_ms] == row, "CONFLICTING_ARCHIVE_CANDLES")
                        by_open[open_ms] = row
            month = following
        series = [by_open[x] for x in sorted(by_open)]
        require(len(series) == BARS, "MISSING_FINAL_CLOSED_ARCHIVE_H1_HISTORY")
        for i, bar in enumerate(series):
            require(bar[0] == start_ms + i*MS and bar[6] == bar[0]+MS-1,
                    "FUTURE_OR_GAPPED_OFFICIAL_ARCHIVE")
        return series
