"""Offline adversarial 15-native-timeframe Binance Futures market capture tests."""
import tempfile
import unittest
from pathlib import Path
from urllib.parse import urlencode
from datetime import datetime,timezone

from williams_ai.binance_wave_history import (
    VENUES,INTERVAL_MS,ANALYSIS_TFS,
    floor_open_ms,next_open_ms,parse_closed_candle,
    DurableWaveHistory,HistoryUnavailable,
)

# UTC known market snapshot, no external HTTP in CI or credentials ever.
NOW=1_800_000_000_000
SYM="ETHUSDT"

def _raw(op,interval,*,close="100",volume="10"):
    end=next_open_ms(op,interval)-1
    return [op,"100","105","95",close,volume,end,"1000",100,
            "4","400","0"]

class Venue:
    def __init__(self,interval="1h",n=60,*,venue="PUBLIC_USDM",start=1_700_000_000_000):
        self.interval=interval
        self.start=floor_open_ms(start,interval)
        self.bars=[]
        open_ms=self.start
        for _ in range(n):
            self.bars.append(_raw(open_ms,interval))
            open_ms=next_open_ms(open_ms,interval)
        self.venue=venue
        self.calls=[]
        self.inject_gap=False
        self.bad_overlap=False
    def __call__(self,venue,path,params):
        self.calls.append((venue,path,params))
        if venue!=self.venue:
            raise AssertionError("silent venue switching forbidden")
        if path=="/fapi/v1/time":
            return {"serverTime":NOW}
        if path=="/fapi/v1/exchangeInfo":
            return {"symbols":[{"symbol":SYM,"status":"TRADING","filters":[
                {"filterType":"PRICE_FILTER","tickSize":"0.01"}]}]}
        if path=="/fapi/v1/klines":
            assert params["symbol"]==SYM and params["interval"]==self.interval
            start=int(params["startTime"])
            end=int(params["endTime"])
            size=int(params["limit"])
            rows=[x for x in self.bars if start<=x[0] and x[6]<=end]
            rows=rows[:size]
            if self.bad_overlap and rows:
                import copy
                rows=copy.deepcopy(rows)
                rows[0][4]="101"
            if self.inject_gap and len(rows)>4:
                rows=rows[:3]+rows[4:]
            return rows
        raise AssertionError("unapproved URL")

class WaveHistoryTests(unittest.TestCase):
    def test_all_15_native_binance_futures_intervals(self):
        self.assertEqual(len(INTERVAL_MS),15)
        self.assertEqual(len(ANALYSIS_TFS),15)
        self.assertEqual(set(VENUES),{"DEMO_USDM","PUBLIC_USDM"})
        self.assertEqual(next_open_ms(floor_open_ms(1_730_000_000_000,"1M"),"1M"),
                         1_730_419_200_000)
        # Monday anchor for UTC Binance 1w.
        monday=floor_open_ms(1_730_000_000_000,"1w")
        self.assertEqual(datetime.fromtimestamp(monday/1000,timezone.utc).weekday(),0)
        for tf in INTERVAL_MS:
            with self.subTest(tf=tf):
                origin=floor_open_ms(1_730_000_000_000,tf)
                bar=_raw(origin,tf)
                self.assertEqual(parse_closed_candle(bar,tf,server_ms=NOW)["close_ms"],
                                 next_open_ms(origin,tf)-1)

    def test_rejects_unfinished_invalid_prices_and_holes(self):
        hour=floor_open_ms(NOW,"1h")
        self.assertIsNone(parse_closed_candle(_raw(hour,"1h"),"1h",server_ms=NOW))
        p=_raw(hour-2*3600000,"1h")
        p[2]="89"
        with self.assertRaises(HistoryUnavailable):
            parse_closed_candle(p,"1h",server_ms=NOW)
        p=_raw(hour-2*3600000,"1h")
        p[9]="50"
        with self.assertRaisesRegex(HistoryUnavailable,"TAKER_VOLUME"):
            parse_closed_candle(p,"1h",server_ms=NOW)
        p=_raw(hour-2*3600000,"1h")
        p[6]-=1
        with self.assertRaisesRegex(HistoryUnavailable,"BOUNDARIES"):
            parse_closed_candle(p,"1h",server_ms=NOW)

    def test_incremental_pagination_exact_replay_and_persistence(self):
        source=Venue(n=62)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"history.sqlite3"
            db=DurableWaveHistory(path)
            first=db.sync(venue="PUBLIC_USDM",symbol=SYM,interval="1h",
                          since_ms=source.start,fetch=source,now_ms=NOW,
                          page_size=20,max_pages=2,sleep_seconds=0)
            self.assertEqual(first["stored_closed_bars"],40)
            self.assertEqual(first["sync_state"],"PAGE_BUDGET_EXHAUSTED")
            self.assertFalse(first["complete_download_claimed"])
            reopened=DurableWaveHistory(path)
            second=reopened.sync(venue="PUBLIC_USDM",symbol=SYM,interval="1h",
                                since_ms=source.start,fetch=source,now_ms=NOW,
                                page_size=20,max_pages=4,sleep_seconds=0)
            self.assertEqual(second["stored_closed_bars"],62)
            self.assertEqual(second["new_closed_bars_added"],22)
            self.assertEqual(second["sync_state"],"AT_LATEST_CLOSED_BAR")
            third=reopened.sync(venue="PUBLIC_USDM",symbol=SYM,interval="1h",
                               since_ms=source.start,fetch=source,now_ms=NOW,
                               page_size=20,max_pages=1,sleep_seconds=0)
            self.assertEqual(third["new_closed_bars_added"],0)
            self.assertEqual(third["stored_closed_bars"],62)
            self.assertEqual(len(reopened.recent("PUBLIC_USDM",SYM,"1h")),62)
            self.assertEqual(reopened.recent("PUBLIC_USDM",SYM,"1h")[-1]["number_of_trades"],100)
            self.assertFalse(third["execution_allowed"])

    def test_rejected_modified_old_candle_does_not_corrupt_db(self):
        source=Venue(n=20)
        with tempfile.TemporaryDirectory() as d:
            db=DurableWaveHistory(Path(d)/"bars.db")
            db.sync(venue="PUBLIC_USDM",symbol=SYM,interval="1h",
                    since_ms=source.start,fetch=source,now_ms=NOW,
                    max_pages=2,page_size=20,sleep_seconds=0)
            source.bad_overlap=True
            with self.assertRaisesRegex(HistoryUnavailable,"CHANGED_QUARANTINE"):
                db.sync(venue="PUBLIC_USDM",symbol=SYM,interval="1h",
                        since_ms=source.start,fetch=source,now_ms=NOW,
                        max_pages=1,page_size=20,sleep_seconds=0)
            self.assertEqual(db.status("PUBLIC_USDM",SYM,"1h")["stored_closed_bars"],20)

    def test_gap_inside_batch_is_hard_error_and_atomic(self):
        source=Venue(n=45)
        source.inject_gap=True
        with tempfile.TemporaryDirectory() as d:
            db=DurableWaveHistory(Path(d)/"h.db")
            with self.assertRaisesRegex(HistoryUnavailable,"PAGE_GAPPED"):
                db.sync(venue="PUBLIC_USDM",symbol=SYM,interval="1h",
                        since_ms=source.start,fetch=source,now_ms=NOW,
                        max_pages=2,page_size=20,sleep_seconds=0)
            self.assertEqual(db.status("PUBLIC_USDM",SYM,"1h")["stored_closed_bars"],0)

    def test_public_and_demo_are_physically_separate(self):
        market=Venue(n=22,venue="PUBLIC_USDM")
        demo=Venue(n=24,venue="DEMO_USDM")
        with tempfile.TemporaryDirectory() as d:
            db=DurableWaveHistory(Path(d)/"separate.db")
            a=db.sync(venue="PUBLIC_USDM",symbol=SYM,interval="1h",
                      since_ms=market.start,fetch=market,now_ms=NOW,
                      max_pages=3,page_size=20,sleep_seconds=0)
            b=db.sync(venue="DEMO_USDM",symbol=SYM,interval="1h",
                      since_ms=demo.start,fetch=demo,now_ms=NOW,
                      max_pages=3,page_size=20,sleep_seconds=0)
            self.assertEqual(a["stored_closed_bars"],22)
            self.assertEqual(b["stored_closed_bars"],24)
            self.assertEqual(a["venue"],"PUBLIC_USDM")
            self.assertEqual(b["venue"],"DEMO_USDM")
            self.assertFalse(a["venue_switched_automatically"])
            self.assertFalse(b["venue_switched_automatically"])

    def test_failed_http_451_remains_failure_no_spot_fallback(self):
        def blocked(venue,path,params):
            raise HistoryUnavailable("PUBLIC_USDM_PUBLIC_HTTP_451")
        with tempfile.TemporaryDirectory() as d:
            db=DurableWaveHistory(Path(d)/"blocked.db")
            with self.assertRaisesRegex(HistoryUnavailable,"451"):
                db.sync(venue="PUBLIC_USDM",symbol=SYM,interval="1h",
                        fetch=blocked,now_ms=NOW,sleep_seconds=0)
            self.assertEqual(db.status("PUBLIC_USDM",SYM,"1h")["stored_closed_bars"],0)

    def test_earlier_backfill_cannot_be_silently_claimed(self):
        source=Venue(n=20)
        with tempfile.TemporaryDirectory() as d:
            db=DurableWaveHistory(Path(d)/"bars.db")
            db.sync(venue="PUBLIC_USDM",symbol=SYM,interval="1h",
                    since_ms=source.start,fetch=source,now_ms=NOW,
                    page_size=20,max_pages=2,sleep_seconds=0)
            with self.assertRaisesRegex(HistoryUnavailable,"EARLIER_HISTORY"):
                db.sync(venue="PUBLIC_USDM",symbol=SYM,interval="1h",
                        since_ms=0,fetch=source,now_ms=NOW,
                        max_pages=1,page_size=20,sleep_seconds=0)

    def test_no_cached_secrets_and_private_permissions(self):
        import stat
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"hist.db"
            db=DurableWaveHistory(path)
            self.assertEqual(stat.S_IMODE(path.stat().st_mode),0o600)
            self.assertFalse(db.status("PUBLIC_USDM",SYM,"1h")["execution_allowed"])
            wrong=Path(d)/"hist2.db"
            wrong.symlink_to(path)
            with self.assertRaises(HistoryUnavailable):
                DurableWaveHistory(wrong)


if __name__=="__main__":
    unittest.main()
