"""Fetch contiguous closed Binance USD-M H1 candles with publisher checksums."""
from __future__ import annotations

from calendar import monthrange
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
from decimal import Decimal
from hashlib import sha256
from io import StringIO
import json
from pathlib import Path
import time
from urllib.request import urlopen
from zipfile import ZipFile

ARCHIVE = "https://data.binance.vision/data/futures/um/monthly/klines"
HOUR = 3_600_000
SPLITS = {
    "train": (2023, range(1, 6), ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "LINKUSDT", "DOGEUSDT")),
    "validation": (2024, range(6, 11), ("BTCUSDT", "ETHUSDT", "LINKUSDT")),
    "test": (2026, range(5, 11), ("BTCUSDT", "ETHUSDT", "OPUSDT", "APTUSDT", "NEARUSDT", "SUIUSDT")),
}


def _read(url: str, maximum: int) -> bytes:
    for attempt in range(3):
        try:
            with urlopen(url, timeout=20) as response:
                raw = response.read(maximum + 1)
            if not raw or len(raw) > maximum:
                raise ValueError("EMPTY_OR_OVERSIZE_BINANCE_RESPONSE")
            return raw
        except OSError:
            if attempt == 2:
                raise
            time.sleep(attempt + 1)
    raise AssertionError("unreachable")


def _archive(directory: Path, symbol: str, year: int, month: int):
    name = f"{symbol}-1h-{year}-{month:02d}"
    first = int(datetime(year, month, 1, tzinfo=timezone.utc).timestamp() * 1000)
    count = monthrange(year, month)[1] * 24
    connector_file = directory / (name + ".api.json")
    if connector_file.is_file():
        rows = json.loads(connector_file.read_bytes())
        if len(rows) != count or any(
            len(row) != 6 or row[0] != first + (i + 1) * HOUR - 1
            for i, row in enumerate(rows)
        ):
            raise ValueError("INCOMPLETE_BINANCE_API_MONTH")
        return None, rows
    archive = directory / (name + ".zip")
    checksum = directory / (name + ".CHECKSUM")
    url = f"{ARCHIVE}/{symbol}/1h/{name}.zip"
    if not checksum.exists():
        checksum.write_bytes(_read(url + ".CHECKSUM", 512))
    fields = checksum.read_text(encoding="ascii").split()
    if len(fields) != 2 or fields[1] != archive.name or len(fields[0]) != 64:
        raise ValueError("BINANCE_CHECKSUM_FILENAME_MISMATCH")
    expected = fields[0]
    if not archive.exists() or sha256(archive.read_bytes()).hexdigest() != expected:
        raw = _read(url, 524_288)
        if sha256(raw).hexdigest() != expected:
            raise ValueError("BINANCE_ARCHIVE_SHA256_MISMATCH")
        archive.write_bytes(raw)
    with ZipFile(archive) as z:
        if z.namelist() != [name + ".csv"] or z.infolist()[0].file_size > 1_048_576:
            raise ValueError("UNEXPECTED_BINANCE_CSV_ARCHIVE")
        rows = list(csv.reader(StringIO(z.read(name + ".csv").decode("ascii"))))
    if rows and rows[0][0] == "open_time":
        rows.pop(0)
    if len(rows) != count:
        raise ValueError("PARTIAL_BINANCE_MONTH")
    result = []
    for i, row in enumerate(rows):
        opened = first + i * HOUR
        if len(row) != 12 or int(row[0]) != opened or int(row[6]) != opened + HOUR - 1:
            raise ValueError("NONCONTIGUOUS_BINANCE_CANDLE")
        result.append([int(row[6]), *row[1:6]])
    return expected, result


def fetch_market_history(directory: Path) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    work = [(split, symbol, year, month) for split, (year, months, symbols) in SPLITS.items()
            for symbol in symbols for month in months]

    def read(item):
        split, symbol, year, month = item
        expected, rows = _archive(directory, symbol, year, month)
        return split, symbol, f"{year}-{month:02d}", expected, rows

    result = {s: {} for s in SPLITS}
    with ThreadPoolExecutor(max_workers=8) as pool:
        for split, symbol, month, expected, rows in pool.map(read, work):
            series = result[split].setdefault(symbol, {"archives": {}, "bars": []})
            series["archives"][month] = expected
            series["bars"].extend(
                [row[0], *(format(Decimal(x).normalize(), "f") for x in row[1:])]
                for row in rows
            )
    return result


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        raise SystemExit("usage: ai_market_reasoning_data.py CACHE_DIRECTORY")
    series = fetch_market_history(Path(sys.argv[1]))
    print(json.dumps({s: {symbol: len(row["bars"]) for symbol, row in p.items()}
                      for s, p in series.items()}, sort_keys=True))
