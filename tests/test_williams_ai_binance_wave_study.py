"""Independent closed OHLCV per timeframe; cross-degree stays conjectural."""
import tempfile
import unittest
from pathlib import Path
from datetime import datetime
from williams_ai.binance_wave_history import (
    INTERVAL_MS,ANALYSIS_TFS,DurableWaveHistory,HistoryUnavailable,
)
from williams_ai.binance_wave_study import analyze_all_cached
from test_williams_ai_binance_wave_history import Venue,NOW,SYM


class FakeExplanation:
    def __init__(self):
        self.context_seen=None
    def propose_chart_reasoning(self,chart,edition="TC2",**kwargs):
        self.context_seen=chart
        return {
            "symbol":chart["symbol"],"as_of":chart["as_of"],
            "edition":edition,"setup":"NO_SETUP","side":"NONE",
            "entry_state":"NO_SETUP","rule_ids":[],"exit_basis":"NONE",
            "reason":"No current trade verified.",
            "trade_action":None,"execution_allowed":False,
        }


class FifteenTimeframeResearchTests(unittest.TestCase):
    def test_true_separately_sourced_15_native_intervals(self):
        with tempfile.TemporaryDirectory() as d:
            store=DurableWaveHistory(Path(d)/"research.db")
            for interval in INTERVAL_MS:
                source=Venue(interval=interval,n=65,start=(
                    1_785_000_000_000 if interval=="1h"
                    else 1_500_000_000_000))
                result=store.sync(
                    venue="PUBLIC_USDM",symbol=SYM,interval=interval,
                    since_ms=source.start,max_pages=6,page_size=20,
                    fetch=source,now_ms=NOW,sleep_seconds=0,
                )
                self.assertEqual(result["stored_closed_bars"],65)
                self.assertEqual(result["sync_state"],"AT_LATEST_CLOSED_BAR")
            model=FakeExplanation()
            report=analyze_all_cached(store,venue="PUBLIC_USDM",symbol=SYM,
                                      model=model,window=140)
            self.assertEqual(report["coverage_native_intervals"],15)
            self.assertEqual(report["observed_intervals"],15)
            self.assertTrue(report["all_intervals_have_usable_wave_indicators"])
            self.assertEqual(len(report["matrix"]),15)
            self.assertEqual(len(report["adjacent_cross_scale_competing_hypotheses"]),14)
            self.assertEqual(report["gguf_independent_source_audit"]["setup"],"NO_SETUP")
            self.assertEqual(len(report["matrix"]["1M"]["closed_visible_bars"].__str__())>0,True)
            self.assertFalse(report["wave_degree_expert_verified"])
            self.assertFalse(report["execution_allowed"])
            self.assertIsNone(report["trade_action"])
            self.assertIsNotNone(model.context_seen)
            native=model.context_seen["all_binance_timeframes_compact"]
            as_of=datetime.fromisoformat(model.context_seen["as_of"])
            self.assertIn("1h",native)
            for tf,observation in native.items():
                self.assertLessEqual(datetime.fromisoformat(observation["close"]),as_of,
                                     msg="Future native TF passed to older H1 context")
                self.assertFalse(observation["wave_degree_verified"])
            self.assertEqual(report["gguf_independent_source_audit"]["reviewed_native_intervals_visible_to_llm"],
                             sorted(native))
            for frame in report["matrix"].values():
                self.assertEqual(frame["wave_degree"],"UNKNOWN")
                self.assertFalse(frame["execution_allowed"])

    def test_no_fake_ready_claim_if_only_h1_downloaded(self):
        source=Venue(interval="1h",n=65)
        with tempfile.TemporaryDirectory() as d:
            store=DurableWaveHistory(Path(d)/"only_h1.db")
            store.sync(venue="DEMO_USDM",symbol=SYM,interval="1h",
                       since_ms=source.start,max_pages=5,page_size=20,
                       fetch=Venue(interval="1h",n=65,venue="DEMO_USDM"),
                       now_ms=NOW,sleep_seconds=0)
            report=analyze_all_cached(store,venue="DEMO_USDM",symbol=SYM)
            self.assertEqual(report["observed_intervals"],1)
            self.assertFalse(report["all_intervals_have_usable_wave_indicators"])
            self.assertEqual(report["matrix"]["1d"]["study_status"],"INSUFFICIENT_HISTORY")
            self.assertEqual(report["matrix"]["1M"]["study_status"],"INSUFFICIENT_HISTORY")
            self.assertFalse(report["full_historical_backfill_completed_for_every_tf"])
            self.assertFalse(report["execution_allowed"])

    def test_h1_required_to_explain_full_matrix(self):
        with tempfile.TemporaryDirectory() as d:
            store=DurableWaveHistory(Path(d)/"empty.db")
            with self.assertRaisesRegex(HistoryUnavailable,"H1_REQUIRED"):
                analyze_all_cached(
                    store,venue="PUBLIC_USDM",symbol=SYM,
                    intervals=("15m",),model=FakeExplanation(),
                )


if __name__=="__main__":
    unittest.main()
