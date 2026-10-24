"""One H1 decision must not see M1 candles from later that same day/hour."""
import copy
import tempfile
import unittest
from pathlib import Path
from datetime import datetime

from williams_ai.binance_wave_history import DurableWaveHistory
from williams_ai.binance_wave_study import analyze_all_cached
from test_williams_ai_binance_wave_history import Venue,NOW,SYM
from test_williams_ai_binance_wave_study import FakeExplanation


class AllTimeframeCausalTimestampTest(unittest.TestCase):
    def _prep(self,db):
        h1=Venue(interval="1h",n=75)
        h1_start=h1.start
        final_h1_open=h1.bars[-1][0]
        m1=Venue(interval="1m",n=240,
                 start=final_h1_open-120*60_000)
        # H1 is 75h old, M1 includes 120 more minutes than last complete H1.
        db.sync(venue="PUBLIC_USDM",symbol=SYM,interval="1h",
                since_ms=h1_start,fetch=h1,now_ms=NOW,
                max_pages=1,page_size=500,sleep_seconds=0)
        db.sync(venue="PUBLIC_USDM",symbol=SYM,interval="1m",
                since_ms=m1.start,fetch=m1,now_ms=NOW,
                max_pages=1,page_size=500,sleep_seconds=0)
        return h1,m1

    def test_older_h1_decision_recovers_past_m1_but_never_future_m1(self):
        with tempfile.TemporaryDirectory() as d:
            db=DurableWaveHistory(Path(d)/"times.db")
            h1,m1=self._prep(db)
            model=FakeExplanation()
            report=analyze_all_cached(db,venue="PUBLIC_USDM",symbol=SYM,
                intervals=("1m","1h"),model=model,window=140)
            self.assertEqual(report["observed_intervals"],2)
            self.assertEqual(report["matrix"]["1m"]["closed_visible_bars"],120)
            self.assertEqual(report["matrix"]["1h"]["closed_visible_bars"],75)
            self.assertEqual(report["shared_historical_as_of_ms"],h1.bars[-1][6])
            self.assertEqual(report["shared_historical_clock_source"],
                             "LATEST_CACHED_H1_CLOSE")
            anchor=model.context_seen
            native=anchor["all_binance_timeframes_compact"]
            self.assertEqual(len(native),2)
            self.assertLessEqual(
                datetime.fromisoformat(native["1m"]["close"]),
                datetime.fromisoformat(anchor["as_of"]))
            self.assertFalse(report["execution_allowed"])
            self.assertIsNone(report["trade_action"])

    def test_future_m1_changes_do_not_change_old_h1_wave_analysis(self):
        with tempfile.TemporaryDirectory() as d:
            db=DurableWaveHistory(Path(d)/"times.db")
            h1,m1=self._prep(db)
            before=analyze_all_cached(db,venue="PUBLIC_USDM",symbol=SYM,
                intervals=("1m","1h"),window=140)
            # Mutate ONLY bars that close AFTER old H1 close. This emulates
            # additional market development; read-only as-of inference must
            # not depend on these bars, even when SQLite already contains them.
            with db._connect() as conn:
                conn.execute("BEGIN IMMEDIATE")
                conn.execute(
                    "UPDATE candles SET close='101',high='105',low='95' "
                    "WHERE venue=? AND symbol=? AND interval='1m' AND close_ms>?",
                    ("PUBLIC_USDM",SYM,h1.bars[-1][6]))
                conn.execute("COMMIT")
            after=analyze_all_cached(db,venue="PUBLIC_USDM",symbol=SYM,
                intervals=("1m","1h"),window=140)
            self.assertEqual(before,after)
            self.assertEqual(after["observed_intervals"],2)
            self.assertFalse(after["wave_degree_expert_verified"])


if __name__=="__main__":
    unittest.main()
