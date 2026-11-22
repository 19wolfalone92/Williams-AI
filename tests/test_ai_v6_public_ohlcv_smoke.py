"""Official Binance MONTHLY archive smoke: offline checksum and 1024 H1 contracts."""
from __future__ import annotations

import calendar
from datetime import datetime, timezone
from hashlib import sha256
import io
from zipfile import ZipFile, ZIP_DEFLATED
import unittest

from scripts.ai_v6_public_ohlcv_smoke import (
    ROOT, _MAX_CHECKSUM_BYTES, _MAX_ZIP_BYTES, _cutoff_for_completed_month,
    _previous_month, _month_rows, official_monthly_klines, collect_smoke,
)
from williams_ai.v6_public_chart_collector import MS, BARS


class BinanceSignedArchiveTest(unittest.TestCase):
    def setUp(self):
        self.now = datetime(2026, 11, 13, tzinfo=timezone.utc)
        self.cutoff = _cutoff_for_completed_month(self.now)
        self.assertEqual(self.cutoff.isoformat(), "2026-11-01T00:00:00+00:00")
        self.end_ms = int(self.cutoff.timestamp()*1000)
        self.start_ms = self.end_ms - BARS*MS
        self.cache = {}

    def archive(self, filename):
        if filename in self.cache:
            return self.cache[filename]
        symbol, interval, year, month = filename.removesuffix(".zip").split("-")
        year, month = int(year), int(month)
        first = datetime(year, month, 1, tzinfo=timezone.utc)
        base_ms = int(first.timestamp()*1000)
        lines = ["open_time,open,high,low,close,volume,close_time"]
        for k in range(24*calendar.monthrange(year, month)[1]):
            start = base_ms + k*MS
            lines.append(f"{start},100,103,99,101,25,{start+MS-1}")
        buf = io.BytesIO()
        with ZipFile(buf, "w", compression=ZIP_DEFLATED) as z:
            z.writestr(filename.removesuffix(".zip")+".csv",
                       "\n".join(lines)+"\n")
        raw = buf.getvalue()
        self.cache[filename] = raw
        return raw

    def fake_fetch(self, url, limit):
        self.assertTrue(url.startswith(ROOT))
        name = url.split("/")[-1]
        if name.endswith(".CHECKSUM"):
            self.assertEqual(limit, _MAX_CHECKSUM_BYTES)
            archive_name = name.removesuffix(".CHECKSUM")
            raw = self.archive(archive_name)
            return f"{sha256(raw).hexdigest()}  {archive_name}\n".encode()
        self.assertEqual(limit, _MAX_ZIP_BYTES)
        return self.archive(name)

    def test_source_sha256_and_no_future_closure_across_two_months(self):
        a, receipts = official_monthly_klines("BTCUSDT", self.start_ms,
                    self.end_ms-1, fetch=self.fake_fetch)
        self.assertEqual(len(a), BARS)
        self.assertEqual(len(receipts), 2)
        self.assertTrue(all(x["official_checksum_verified"] for x in receipts))
        self.assertEqual(a[0][0], self.start_ms)
        self.assertEqual(a[-1][6], self.end_ms-1)
        self.assertTrue(all(a[k][0]+MS == a[k+1][0]
                            for k in range(len(a)-1)))

    def test_tampered_archive_sha256_fails(self):
        def corrupt(url, limit):
            if url.endswith(".CHECKSUM"):
                return ("0"*64+"  archive.zip\n").encode("ascii")
            return self.fake_fetch(url, limit)
        with self.assertRaisesRegex(ValueError, "OFFICIAL_SHA256_MISMATCH"):
            official_monthly_klines("BTCUSDT", self.start_ms, self.end_ms-1,
                                    fetch=corrupt)

    def test_missing_archive_month_is_not_replaced_by_future_bars(self):
        def missing(url, limit):
            if "2026-09" in url and url.endswith(".zip"):
                raise FileNotFoundError("Month archive missing")
            return self.fake_fetch(url, limit)
        with self.assertRaises(FileNotFoundError):
            official_monthly_klines("BTCUSDT", self.start_ms, self.end_ms-1,
                                    fetch=missing)

    def test_real_smoke_schema_never_reports_model_or_expert_accuracy(self):
        def source(symbol, start_ms, end_ms):
            return official_monthly_klines(symbol, start_ms, end_ms,
                                           fetch=self.fake_fetch)
        receipt = collect_smoke(now=self.now, fetcher=source)
        self.assertEqual(receipt["cutoff_utc"], "2026-11-01T00:00:00Z")
        self.assertEqual(len(receipt["market_data_result"]), 4)
        self.assertFalse(receipt["auth_keys_used"])
        self.assertFalse(receipt["model_inference_performed"])
        self.assertFalse(receipt["expert_gold_present"])
        self.assertTrue(receipt["not_independent_forward_holdout"])
        self.assertTrue(all(x["bars"]==1024 and x["completed_D1"]>=40 and
                            x["D1_AO_available"] and x["D1_AC_available"]
                            for x in receipt["market_data_result"]))

    def test_cutoff_reduces_if_first_day_archives_not_yet_ready(self):
        t = datetime(2026, 11, 1, tzinfo=timezone.utc)
        self.assertEqual(_cutoff_for_completed_month(t),
                         datetime(2026, 10, 1, tzinfo=timezone.utc))


if __name__ == "__main__":
    unittest.main()
