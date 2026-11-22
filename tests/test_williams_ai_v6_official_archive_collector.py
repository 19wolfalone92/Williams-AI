"""Offline tests for the exact official daily 1h archive source contract."""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from hashlib import sha256
import io
from unittest.mock import patch
import unittest
from zipfile import ZipFile, ZIP_DEFLATED

from williams_ai.v6_official_archive_collector import (
    ROOT, HOUR_MS, MAX_ZIP_BYTES, verified_daily,
    read_only_public_archives_klines,
)


def day_bars(day: date):
    start = int(datetime(day.year, day.month, day.day,
                         tzinfo=timezone.utc).timestamp() * 1000)
    return [[start+i*HOUR_MS, "100", "102", "99", "101",
             "20", start+(i+1)*HOUR_MS-1] for i in range(24)]


def fixture_zip(symbol: str, day: date):
    filename=f"{symbol}-1h-{day.isoformat()}"
    csv = ["open_time,open,high,low,close,volume,close_time"]
    for row in day_bars(day):
        csv.append(",".join(map(str,row)))
    data=io.BytesIO()
    with ZipFile(data,"w",ZIP_DEFLATED) as z:
        z.writestr(filename+".csv","\n".join(csv)+"\n")
    return data.getvalue()


class OfficialDailyArchiveTests(unittest.TestCase):
    def setUp(self):
        self.day=date(2026,11,11)
        self.symbol="BTCUSDT"
        self.zipbytes=fixture_zip(self.symbol,self.day)

    def fetch(self, url: str, limit: int):
        self.assertTrue(url.startswith(ROOT))
        if url.endswith(".CHECKSUM"):
            self.assertEqual(limit,512)
            return (sha256(self.zipbytes).hexdigest()+
                    f"  {self.symbol}-1h-{self.day.isoformat()}.zip\n").encode("ascii")
        self.assertEqual(limit, MAX_ZIP_BYTES)
        return self.zipbytes

    def test_24_real_format_hours_and_official_sha256_receipt(self):
        rows,rcpt=verified_daily(self.symbol,self.day,fetch=self.fetch)
        self.assertEqual(len(rows),24)
        self.assertEqual(rows[0][0],day_bars(self.day)[0][0])
        self.assertEqual(rows[-1][6],day_bars(self.day)[-1][6])
        self.assertEqual(rcpt["sha256"],sha256(self.zipbytes).hexdigest())
        self.assertTrue(rcpt["official_checksum_verified"])

    def test_checksum_wrong_must_abort_without_fake_data(self):
        def wrong(url,limit):
            if url.endswith(".CHECKSUM"):
                return ("0"*64+"  btc.zip\n").encode("ascii")
            return self.fetch(url,limit)
        with self.assertRaisesRegex(ValueError,"OFFICIAL_CHECKSUM_MISMATCH"):
            verified_daily(self.symbol,self.day,fetch=wrong)

    def test_missing_archive_does_not_trigger_fallback_fapi(self):
        def absent(url,limit):
            raise FileNotFoundError("the official daily ZIP is not published")
        with self.assertRaises(FileNotFoundError):
            verified_daily(self.symbol,self.day,fetch=absent)

    def test_exact_1024_from_43_verified_days_no_future_fill(self):
        cutoff=datetime(2026,11,22,tzinfo=timezone.utc)
        finish=int(cutoff.timestamp()*1000)
        start=finish-1024*HOUR_MS
        def cached(symbol, day_iso):
            day=date.fromisoformat(day_iso)
            return day_bars(day),{
                "utc_date":day_iso,"sha256":"a"*64,
                "official_checksum_verified":True,
                "source":ROOT+symbol+"/1h/"+day_iso,"H1_count":24}
        with patch("williams_ai.v6_official_archive_collector._cached_daily",
                   side_effect=cached) as mocked:
            candles,provenance=read_only_public_archives_klines(
                "BTCUSDT",start,finish-1)
        self.assertEqual(len(candles),1024)
        self.assertEqual(candles[0][0],start)
        self.assertEqual(candles[-1][6],finish-1)
        self.assertGreaterEqual(mocked.call_count,43)
        self.assertTrue(provenance["official_checksum_verified_all"])
        self.assertTrue(all(p["H1_count"]==24
                            for p in provenance["daily_sources"]))

    def test_tampered_day_gaps_fail_closed(self):
        cutoff=datetime(2026,11,22,tzinfo=timezone.utc)
        finish=int(cutoff.timestamp()*1000)
        start=finish-1024*HOUR_MS
        def missing(symbol, day_iso):
            day=date.fromisoformat(day_iso)
            raw=day_bars(day)
            if day_iso=="2026-11-11": raw=raw[:-1]
            return raw, {
                "utc_date":day_iso,"sha256":"a"*64,
                "official_checksum_verified":True,
                "source":ROOT+symbol+"/1h/"+day_iso,"H1_count":len(raw)}
        with patch("williams_ai.v6_official_archive_collector._cached_daily",
                   side_effect=missing):
            with self.assertRaisesRegex(ValueError,"MISSING_CLOSED_H1_BARS"):
                read_only_public_archives_klines("BTCUSDT",start,finish-1)


if __name__=="__main__":
    unittest.main()
