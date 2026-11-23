"""Offline fixtures for verified Binance archive fallback; NO network/keys."""
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from io import BytesIO
import csv
import zipfile
from urllib.error import HTTPError
from unittest.mock import patch
import unittest

from williams_ai import v6_public_binance_archive as arc
from williams_ai.v6_public_chart_collector import (
    MS, read_only_public_klines, normalized_chart,
)

CUTOFF=datetime(2026,11,12,tzinfo=timezone.utc)
END=int(CUTOFF.timestamp()*1000)
START=END-1024*MS


class BinanceOfficialArchiveContracts(unittest.TestCase):
    def setUp(self):
        arc._CACHE.clear()
        self.addCleanup(arc._CACHE.clear)
        self.files={}
        self.calls=[]
        self.error_mode=None

    def _zip(self,url):
        name=url.split("/")[-1]
        part=name[:-4].split("-1h-")[-1]
        if len(part)==7:
            start=datetime.strptime(part,"%Y-%m").replace(tzinfo=timezone.utc)
            if start.month==12:
                end=start.replace(year=start.year+1,month=1)
            else:
                end=start.replace(month=start.month+1)
        else:
            start=datetime.strptime(part,"%Y-%m-%d").replace(tzinfo=timezone.utc)
            end=start+timedelta(days=1)
        csvbuf=BytesIO()
        rows=[]
        t=start
        while t<end:
            ms=int(t.timestamp()*1000)
            rows.append([str(ms),"100","103","99","101","50",
                         str(ms+MS-1),"0","30","0","0","0"])
            t+=timedelta(hours=1)
        text="\n".join(",".join(x) for x in rows)+"\n"
        raw=BytesIO()
        with zipfile.ZipFile(raw,"w",zipfile.ZIP_DEFLATED) as out:
            out.writestr(name[:-4]+".csv",text.encode("utf-8"))
        return raw.getvalue()

    def fake_request(self,url,*,max_bytes):
        self.calls.append(url)
        if url.endswith(".CHECKSUM"):
            zip_url=url[:-9]
            content=self.files.setdefault(zip_url,self._zip(zip_url))
            digest=sha256(content).hexdigest()
            if self.error_mode=="bad_checksum":
                digest="0"*64
            return (digest+" *"+zip_url.rsplit("/",1)[-1]+"\n").encode()
        return self.files.setdefault(url,self._zip(url))

    def test_all_1024_real_shape_candles_spanning_months(self):
        with patch.object(arc,"_request",side_effect=self.fake_request):
            rows=arc.official_archive_klines("BTCUSDT",START,END-1)
        self.assertEqual(len(rows),1024)
        chart=normalized_chart("BTCUSDT",CUTOFF,rows)
        self.assertEqual(len(chart["bars"]),1024)
        self.assertEqual(chart["bars"][0]["open_utc"],"2026-09-30T08:00:00Z")
        self.assertEqual(chart["bars"][-1]["close_utc"],"2026-11-12T00:00:00Z")
        self.assertTrue(any("/monthly/" in x for x in self.calls))
        self.assertTrue(any("/daily/" in x for x in self.calls))
        self.assertEqual(len([x for x in self.calls if x.endswith(".CHECKSUM")]),
                         13)
        self.assertLessEqual(len(arc._CACHE),150)

    def test_corrupt_checksum_blocked_not_silently_accepted(self):
        self.error_mode="bad_checksum"
        with patch.object(arc,"_request",side_effect=self.fake_request):
            with self.assertRaisesRegex(ValueError,"SHA256_MISMATCH"):
                arc.official_archive_klines("BTCUSDT",START,END-1)

    def test_modified_zip_bytes_fail_even_with_original_checksum(self):
        def corrupted_request(url, *, max_bytes):
            raw=self.fake_request(url,max_bytes=max_bytes)
            if not url.endswith(".CHECKSUM"):
                raw=raw[:-1]+bytes([raw[-1] ^ 1])
            return raw
        with patch.object(arc,"_request",side_effect=corrupted_request):
            with self.assertRaisesRegex(ValueError,"SHA256_MISMATCH"):
                arc.official_archive_klines("BTCUSDT",START,END-1)

    def test_unapproved_symbol_and_nonhour_window_rejected(self):
        for sym,begin,end in (("FAKEUSDT",START,END-1),
                              ("BTCUSDT",START+1,END-1),
                              ("BTCUSDT",START,END)):
            with self.subTest(sym=sym):
                with self.assertRaisesRegex(ValueError,"REJECTED_WINDOW"):
                    arc.official_archive_klines(sym,begin,end)

    def test_rest_451_falls_back_only_to_verified_official_archive(self):
        def rejected(*a,**kw):
            raise HTTPError(a[0],451,"Unavailable for legal reasons",{},None)
        with patch("williams_ai.v6_public_chart_collector.urlopen",
                   side_effect=rejected), \
             patch.object(arc,"official_archive_klines",
                          return_value=[[START+i*MS,"100","101","99",
                            "100.5","1",START+(i+1)*MS-1,"0",0,"0","0","0"]
                                        for i in range(1024)]) as fallback:
            got=read_only_public_klines("BTCUSDT",START,END-1)
            self.assertEqual(len(got),1024)
            self.assertEqual(read_only_public_klines.last_transport,
                             "BINANCE_OFFICIAL_PUBLIC_ARCHIVE_SHA256")
            fallback.assert_called_once()

    def test_rest_429_not_papered_over_as_valid_history(self):
        with patch("williams_ai.v6_public_chart_collector.urlopen",
                   side_effect=HTTPError("x",429,"rate limit",{},None)), \
             patch.object(arc,"official_archive_klines") as fallback:
            with self.assertRaises(HTTPError):
                read_only_public_klines("BTCUSDT",START,END-1)
            fallback.assert_not_called()

    def test_official_zip_request_rejects_malformed_urls_before_network(self):
        good=arc.BASE+"daily/klines/BTCUSDT/1h/BTCUSDT-1h-2026-11-11.zip.CHECKSUM"
        invalid=[
            good.replace("https://","http://"),
            good.replace("data.binance.vision","data.binance.vision.evil"),
            good.replace("/BTCUSDT/1h/","/ETHUSDT/1h/"),
            good.replace("BTCUSDT-1h","ETHUSDT-1h"),
            good.replace("2026-11-11","2026-02-30"),
            good.replace("2026-11-11","2026-11"),
            good.replace("BTCUSDT","DOGEUSDT"),
            good+"?download=1",
            good+"#fragment",
            good.replace("/1h/","/1h/%2e%2e/"),
            good.replace("https://","https://user:pass@"),
        ]
        with patch.object(arc,"urlopen") as network:
            for bad in invalid:
                with self.subTest(url=bad):
                    with self.assertRaisesRegex(ValueError,"BINANCE_ARCHIVE_"):
                        arc._request(bad,max_bytes=1024)
            with self.assertRaisesRegex(ValueError,"INVALID_RESOURCE_IDENTITY"):
                arc._request(good,max_bytes=6_000_000)
            network.assert_not_called()

    def test_exact_official_request_keeps_original_final_url(self):
        address=arc.BASE+"daily/klines/BTCUSDT/1h/BTCUSDT-1h-2026-11-11.zip.CHECKSUM"
        class Response:
            status=200
            def __enter__(self): return self
            def __exit__(self,*a): return False
            def geturl(self): return address
            def read(self,n): return b"abc"
        with patch.object(arc,"urlopen",return_value=Response()) as network:
            self.assertEqual(arc._request(address,max_bytes=1024),b"abc")
            network.assert_called_once()

    def test_archive_redirect_to_other_date_or_symbol_is_rejected(self):
        address=arc.BASE+"daily/klines/BTCUSDT/1h/BTCUSDT-1h-2026-11-11.zip"
        for final in [
            address.replace("2026-11-11","2026-11-10"),
            address.replace("BTCUSDT","ETHUSDT"),
            address.replace("data.binance.vision","evil.invalid"),
        ]:
            class Response:
                status=200
                def __enter__(self): return self
                def __exit__(self,*a): return False
                def geturl(self): return final
                def read(self,n): return b"zip"
            with self.subTest(final=final),patch.object(arc,"urlopen",return_value=Response()):
                with self.assertRaisesRegex(ValueError,"REDIRECT_RESOURCE_CHANGED"):
                    arc._request(address,max_bytes=6_000_000)

    def test_archive_http_error_reports_exact_source_and_status(self):
        address=arc.BASE+"daily/klines/BTCUSDT/1h/BTCUSDT-1h-2026-11-11.zip.CHECKSUM"
        with patch.object(arc,"urlopen",side_effect=HTTPError(address,404,"missing",{},None)):
            with self.assertRaisesRegex(ValueError,"HTTP_404.*BTCUSDT-1h-2026-11-11.zip.CHECKSUM"):
                arc._request(address,max_bytes=1024)

    def test_checksum_filename_mismatch_rejected_before_zip_read(self):
        def wrong(url, *, max_bytes):
            if url.endswith(".CHECKSUM"):
                target=url[:-9]
                return (sha256(self._zip(target)).hexdigest()+
                        " *ETHUSDT-1h-2026-11-11.zip\\n").encode()
            self.fail("ZIP SHOULD NOT BE FETCHED AFTER BAD CHECKSUM NAME")
        with patch.object(arc,"_request",side_effect=wrong):
            with self.assertRaisesRegex(ValueError,"CHECKSUM_FILENAME_OR_FORMAT_INVALID"):
                arc._load_csv_zip("BTCUSDT","daily","2026-11-11")

    def test_checksum_missing_filename_or_trailing_tokens_rejected(self):
        for line in [b"a"*64,b"a"*64+b" *BTCUSDT-1h-2026-11-11.zip extra",b"invalid *BTCUSDT-1h-2026-11-11.zip"]:
            def wrong(url, *, max_bytes):
                if url.endswith(".CHECKSUM"):return line
                self.fail("ZIP SHOULD NOT BE FETCHED AFTER BAD RECEIPT")
            with self.subTest(line=line[:16]),patch.object(arc,"_request",side_effect=wrong):
                with self.assertRaisesRegex(ValueError,"CHECKSUM_FILENAME_OR_FORMAT_INVALID"):
                    arc._load_csv_zip("BTCUSDT","daily","2026-11-11")

    def _tampered_daily(self, transform):
        target=arc.BASE+"daily/klines/BTCUSDT/1h/BTCUSDT-1h-2026-11-11.zip"
        original=self._zip(target)
        with zipfile.ZipFile(BytesIO(original)) as z:
            name=z.namelist()[0]
            lines=z.read(name).decode().splitlines()
        transform(lines)
        buf=BytesIO()
        with zipfile.ZipFile(buf,"w",zipfile.ZIP_DEFLATED) as z:
            z.writestr(name,("\\n".join(lines)+"\\n").encode())
        raw=buf.getvalue()
        def fetch(url, *, max_bytes):
            if url.endswith(".CHECKSUM"):
                return (sha256(raw).hexdigest()+" *"+target.rsplit("/",1)[-1]+"\\n").encode()
            self.assertEqual(url,target)
            return raw
        with patch.object(arc,"_request",side_effect=fetch):
            return arc._load_csv_zip("BTCUSDT","daily","2026-11-11")

    def test_signed_incomplete_day_still_fails_closed(self):
        with self.assertRaisesRegex(ValueError,"DAY_MONTH_H1_INTEGRITY"):
            self._tampered_daily(lambda lines: lines.pop(-1))

    def test_signed_duplicate_utc_hour_still_fails_closed(self):
        def duplicate(lines):
            lines[1]=lines[0]
        with self.assertRaisesRegex(ValueError,"DAY_MONTH_H1_INTEGRITY"):
            self._tampered_daily(duplicate)

    def test_signed_nonfinite_or_inverted_ohlcv_is_rejected(self):
        for index,val in [(4,"NaN"),(4,"104"),(5,"-1"),(3,"102")]:
            def mutate(lines):
                first=lines[0].split(",")
                first[index]=val
                lines[0]=",".join(first)
            with self.subTest(index=index,value=val):
                with self.assertRaisesRegex(ValueError,"INVALID_OHLCV_RANGE"):
                    self._tampered_daily(mutate)



if __name__=="__main__":
    unittest.main()
