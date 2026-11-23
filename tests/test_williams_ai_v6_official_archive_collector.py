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


class P0ArchiveTransportAndIntegrityTests(unittest.TestCase):
    """Negative tests must not access the network or use real order credentials."""

    def setUp(self):
        from williams_ai import v6_official_archive_collector as module
        self.m = module
        self.day = date(2026, 11, 11)
        self.zipbytes = fixture_zip("BTCUSDT", self.day)
        self.url = ROOT + "BTCUSDT/1h/BTCUSDT-1h-2026-11-11.zip.CHECKSUM"

    def test_missing_checksum_404_reports_exact_symbol_day_url(self):
        from urllib.error import HTTPError
        with patch.object(self.m, "_open_official",
                          side_effect=HTTPError(self.url, 404, "Not Found", {}, None)):
            with self.assertRaisesRegex(
                    ValueError, "status=404.*BTCUSDT-1h-2026-11-11.zip.CHECKSUM"):
                self.m._download(self.url, 512)

    def test_451_and_transport_fail_closed(self):
        from urllib.error import HTTPError, URLError
        for failure, expected in ((HTTPError(self.url, 451, "Unavailable", {}, None),
                                   "status=451"), (URLError("dns"), "NETWORK_FAILURE")):
            with self.subTest(failure=type(failure).__name__):
                with patch.object(self.m, "_open_official", side_effect=failure):
                    with self.assertRaisesRegex(ValueError, expected):
                        self.m._download(self.url, 512)

    def test_url_allowlist_rejects_date_symbol_host_port_and_encoding(self):
        urls = (self.url+"?a=1", self.url+"#fragment",
                self.url.replace("BTCUSDT-1h", "ETHUSDT-1h"),
                self.url.replace("2026-11-11", "2026-02-30"),
                self.url.replace("data.binance.vision", "data.binance.vision.evil.tld"),
                self.url.replace("https:", "http:"),
                self.url.replace("data.binance.vision", "data.binance.vision:443"),
                self.url+"%2f", self.url.replace("BTCUSDT", "DOGEUSDT"))
        for url in urls:
            with self.subTest(url=url):
                self.assertFalse(self.m._valid_official_url(url))
        self.assertTrue(self.m._valid_official_url(self.url))

    def test_limit_must_match_resource_type(self):
        for url, limit in ((self.url, MAX_ZIP_BYTES),
                           (self.url.removesuffix(".CHECKSUM"), 512)):
            with self.assertRaisesRegex(ValueError, "INVALID_OFFICIAL_URL"):
                self.m._download(url, limit)

    def test_cross_host_and_cross_day_redirect_blocked_before_following(self):
        from urllib.request import Request
        handler = self.m._OfficialOnlyRedirect()
        for target in ("https://evil.example/target",
                       self.url.replace("2026-11-11", "2026-11-10"),
                       self.url.replace("BTCUSDT", "ETHUSDT")):
            with self.subTest(target=target):
                with self.assertRaisesRegex(ValueError, "UNTRUSTED_ARCHIVE_REDIRECT"):
                    handler.redirect_request(Request(self.url), None, 302, "Moved",
                                             {}, target)

    def test_completed_response_must_match_exact_requested_url(self):
        class Response(io.BytesIO):
            status = 200
            def geturl(self):
                return self_url.replace("BTCUSDT", "ETHUSDT")
        self_url = self.url
        with patch.object(self.m, "_open_official", return_value=Response(b"a"*64)):
            with self.assertRaisesRegex(ValueError, "UNTRUSTED_ARCHIVE_REDIRECT"):
                self.m._download(self.url, 512)

    def test_checksum_filename_is_bound_to_zip_not_only_digest(self):
        wrong = (sha256(self.zipbytes).hexdigest() +
                 "  ETHUSDT-1h-2026-11-11.zip\n").encode()
        def fetch(url, limit):
            return wrong if url.endswith(".CHECKSUM") else self.zipbytes
        with self.assertRaisesRegex(ValueError, "CHECKSUM_FILENAME_MISMATCH"):
            verified_daily("BTCUSDT", self.day, fetch=fetch)

    def test_corrupted_ohlcv_nan_and_negative_volume_rejected(self):
        with ZipFile(io.BytesIO(self.zipbytes)) as source:
            name = source.namelist()[0]
            original = source.read(name).decode()
        for replacement in ("100,102,99,nan,20", "100,102,99,104,20",
                            "100,102,99,101,-1", "100,102,99,101,inf"):
            with self.subTest(replacement=replacement):
                changed = original.replace("100,102,99,101,20", replacement, 1)
                stream = io.BytesIO()
                with ZipFile(stream, "w", ZIP_DEFLATED) as output:
                    output.writestr(name, changed)
                raw = stream.getvalue()
                def fetch(url, limit):
                    return (sha256(raw).hexdigest().encode()
                            if url.endswith(".CHECKSUM") else raw)
                with self.assertRaisesRegex(ValueError, "INVALID_OHLCV_RANGE"):
                    verified_daily("BTCUSDT", self.day, fetch=fetch)

    def test_missing_daily_archive_never_becomes_1024_fake_bars(self):
        cutoff = datetime(2026, 11, 15, tzinfo=timezone.utc)
        finish = int(cutoff.timestamp()*1000)
        start = finish-1024*HOUR_MS
        def fake(symbol, day_iso):
            if day_iso == "2026-10-28":
                raise ValueError("OFFICIAL_BINANCE_H1_ARCHIVE_HTTP_FAILURE status=404")
            return day_bars(date.fromisoformat(day_iso)), {
                "utc_date": day_iso, "official_checksum_verified": True}
        with patch.object(self.m, "_cached_daily", side_effect=fake):
            with self.assertRaisesRegex(ValueError, "status=404"):
                read_only_public_archives_klines("BTCUSDT", start, finish-1)


if __name__=="__main__":
    unittest.main()
