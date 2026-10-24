"""Causal archive-wide Williams wave reviews do NOT train weights or trade."""
import json
import tempfile
import unittest
from pathlib import Path

from test_williams_ai_binance_wave_history import Venue,NOW,SYM
from williams_ai.binance_wave_history import DurableWaveHistory
from williams_ai.wave_history_notebook import WaveArchiveNotebook


class ArchiveWaveNotebookTests(unittest.TestCase):
    def test_across_300_real_shape_closed_rows_resumes_after_restart(self):
        source=Venue(n=300,interval="1h")
        with tempfile.TemporaryDirectory() as d:
            db=DurableWaveHistory(Path(d)/"study.db")
            sync=db.sync(venue="PUBLIC_USDM",symbol=SYM,interval="1h",
                  since_ms=source.start,max_pages=2,page_size=1000,
                  fetch=source,now_ms=NOW,sleep_seconds=0)
            self.assertEqual(sync["stored_closed_bars"],300)
            notebook=WaveArchiveNotebook(db)
            first=notebook.review(venue="PUBLIC_USDM",symbol=SYM,
                                  interval="1h",checkpoint_budget=2,
                                  stride_bars=120,window_bars=1400)
            self.assertEqual(first["new_checkpoints"],2)
            self.assertEqual(first["checkpoints_stored"],2)
            self.assertEqual(first["study_status"],"ARCHIVE_MORE_CLOSED_HISTORY_TO_REVIEW")
            # Simulate process restart: same SQLite, no duplicate source/wave review.
            restarted=WaveArchiveNotebook(DurableWaveHistory(Path(d)/"study.db"))
            later=restarted.review(venue="PUBLIC_USDM",symbol=SYM,
                                   interval="1h",checkpoint_budget=2,
                                   stride_bars=120,window_bars=1400)
            self.assertEqual(later["new_checkpoints"],1)
            self.assertEqual(later["checkpoints_stored"],3)
            final=restarted.review(venue="PUBLIC_USDM",symbol=SYM,
                                   interval="1h",checkpoint_budget=1,
                                   stride_bars=120,window_bars=1400)
            self.assertEqual(final["new_checkpoints"],0)
            self.assertEqual(final["checkpoints_stored"],3)
            with db._connect() as conn:
                notes=[json.loads(x[0]) for x in conn.execute(
                    "SELECT review_json FROM wave_review WHERE interval='1h' "
                    "ORDER BY asof_open_ms").fetchall()]
            self.assertEqual([x["closed_bars_visible_at_cutoff"] for x in notes],
                             [45,165,285])
            for note in notes:
                self.assertFalse(note["execution_allowed"])
                self.assertIsNone(note["trade_action"])
                self.assertEqual(note["wave_degree"],"UNKNOWN")
                self.assertTrue(note["future_bars_excluded"])
                self.assertFalse(note["model_weights_changed"])
                self.assertEqual(note["venue"],"PUBLIC_USDM")

    def test_monthly_and_hourly_journals_are_segregated(self):
        with tempfile.TemporaryDirectory() as d:
            db=DurableWaveHistory(Path(d)/"allframes.db")
            for interval in ("1M","1h"):
                f=Venue(interval=interval,n=90,start=1_500_000_000_000)
                db.sync(venue="DEMO_USDM",symbol=SYM,interval=interval,
                         since_ms=f.start,max_pages=2,page_size=1000,
                         fetch=Venue(interval=interval,n=90,
                                     start=1_500_000_000_000,venue="DEMO_USDM"),
                         now_ms=NOW,sleep_seconds=0)
            learner=WaveArchiveNotebook(db)
            a=learner.review(venue="DEMO_USDM",symbol=SYM,interval="1M",
                              checkpoint_budget=2,stride_bars=30)
            b=learner.review(venue="DEMO_USDM",symbol=SYM,interval="1h",
                              checkpoint_budget=2,stride_bars=30)
            self.assertEqual(a["checkpoints_stored"],2)
            self.assertEqual(b["checkpoints_stored"],2)
            self.assertFalse(a["execution_allowed"])
            self.assertFalse(b["execution_allowed"])
            self.assertEqual(a["interval"],"1M")
            self.assertEqual(b["interval"],"1h")

    def test_insufficient_bars_does_not_invent_wave_review(self):
        with tempfile.TemporaryDirectory() as d:
            db=DurableWaveHistory(Path(d)/"empty.db")
            learner=WaveArchiveNotebook(db)
            r=learner.review(venue="DEMO_USDM",symbol=SYM,interval="1M")
            self.assertEqual(r["status"],"INSUFFICIENT_CACHED_BARS")
            self.assertEqual(r["new_checkpoints"],0)
            self.assertFalse(r["execution_allowed"])
            with self.assertRaises(ValueError):
                learner.review(venue="DEMO_USDM",symbol=SYM,interval="1h",
                               stride_bars=0)


if __name__=="__main__":
    unittest.main()
