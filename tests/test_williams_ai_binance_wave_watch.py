"""Periodic reader cannot mistake missing data for a safe live trade."""
import tempfile
import unittest
from pathlib import Path

from williams_ai.binance_wave_watch import one_watch_cycle
from williams_ai.binance_wave_history import DurableWaveHistory,HistoryUnavailable
from test_williams_ai_binance_wave_history import Venue,NOW,SYM


class NativeBinanceWatchTests(unittest.TestCase):
    def test_one_pair_foreground_auto_sync_incremental_no_orders(self):
        source=Venue(interval="1h",n=75)
        with tempfile.TemporaryDirectory() as d:
            db=DurableWaveHistory(Path(d)/"watch.db")
            first=one_watch_cycle(
                db,venue="PUBLIC_USDM",symbols=(SYM,),
                intervals=("1h",),max_pages=3,fetch=source,now_ms=NOW,
            )
            self.assertEqual(first["symbols_checked"],1)
            self.assertTrue(first["all_caught_up"])
            self.assertTrue(first["per_symbol"][0]["history_usable_on_every_timeframe_this_cycle"])
            self.assertEqual(first["per_symbol"][0]["wave_study"]["observed_intervals"],1)
            self.assertFalse(first["execution_allowed"])
            self.assertFalse(first["model_can_place_orders"])
            self.assertFalse(first["production_trading_enabled"])
            self.assertFalse(any(
                method not in ("/fapi/v1/time","/fapi/v1/exchangeInfo","/fapi/v1/klines")
                for _,method,_ in source.calls))
            second=one_watch_cycle(
                db,venue="PUBLIC_USDM",symbols=(SYM,),
                intervals=("1h",),max_pages=3,fetch=source,now_ms=NOW,
            )
            self.assertEqual(second["per_symbol"][0]["sources"][0]["new_closed_bars"],0)
            self.assertEqual(db.status("PUBLIC_USDM",SYM,"1h")["stored_closed_bars"],75)

    def test_network_rejection_halts_claim_of_all_timeframes_ready(self):
        calls=[]
        def rejected(venue,path,params):
            calls.append((venue,path))
            raise HistoryUnavailable("DEMO_USDM_PUBLIC_HTTP_451")
        with tempfile.TemporaryDirectory() as d:
            db=DurableWaveHistory(Path(d)/"watch.db")
            outcome=one_watch_cycle(
                db,venue="DEMO_USDM",symbols=(SYM,),
                intervals=("1m","1h","4h","1d"),fetch=rejected,now_ms=NOW,
            )
            self.assertFalse(outcome["all_caught_up"])
            self.assertFalse(outcome["execution_allowed"])
            self.assertEqual(outcome["per_symbol"][0]["sources"][0]["sync_state"],
                             "FAILED_UNVERIFIED_DATA")
            self.assertEqual(len(calls),1)
            self.assertNotIn("wave_study",outcome["per_symbol"][0])
            self.assertEqual(outcome["per_symbol"][0]["timeframes_checked"],1)

    def test_15_pairs_allowed_but_global_binance_page_budget_enforced(self):
        pairs=tuple("TEST%02dUSDT"%i for i in range(15))
        def fetch(venue,path,params):
            self.assertEqual(venue,"PUBLIC_USDM")
            if path=="/fapi/v1/time":
                return {"serverTime":NOW}
            if path=="/fapi/v1/exchangeInfo":
                return {"symbols":[{"symbol":name,
                    "status":"TRADING","filters":[
                        {"filterType":"PRICE_FILTER","tickSize":"0.01"}]}
                        for name in pairs]}
            if path=="/fapi/v1/klines":
                return []
            raise AssertionError("No account calls")
        with tempfile.TemporaryDirectory() as d:
            db=DurableWaveHistory(Path(d)/"all15.db")
            with self.assertRaisesRegex(ValueError,"WEIGHT_LIMIT"):
                one_watch_cycle(
                    db,venue="PUBLIC_USDM",symbols=pairs,
                    intervals=("1m","5m","15m","1h","4h","1d","1M"),
                    max_pages=20,fetch=fetch,now_ms=NOW,
                )
            # 15 independent symbols pass validation, then fail at a
            # real public market-data absence; never a fake LIVE/READY.
            result=one_watch_cycle(
                db,venue="PUBLIC_USDM",symbols=pairs,
                intervals=("1h",),max_pages=1,
                fetch=fetch,now_ms=NOW,
                archive_checkpoints_per_tf=0,
            )
            self.assertEqual(result["symbols_checked"],15)
            self.assertFalse(result["all_caught_up"])
            self.assertFalse(result["execution_allowed"])
            self.assertTrue(all(
                x["sources"][0]["sync_state"]=="FAILED_UNVERIFIED_DATA"
                for x in result["per_symbol"]))
            self.assertTrue(all("wave_study" not in x
                                for x in result["per_symbol"]))

    def test_duplicate_sources_and_unknown_symbol_invalid(self):
        with tempfile.TemporaryDirectory() as d:
            db=DurableWaveHistory(Path(d)/"w.db")
            for args in (
                {"venue":"SPOT","symbols":(SYM,),"intervals":("1h",)},
                {"venue":"PUBLIC_USDM","symbols":(SYM,SYM),"intervals":("1h",)},
                {"venue":"PUBLIC_USDM","symbols":(SYM,),"intervals":("1h","1h")},
            ):
                with self.assertRaises(ValueError):
                    one_watch_cycle(db,fetch=lambda *args:None,now_ms=NOW,**args)


if __name__=="__main__":
    unittest.main()
