"""Explicit chronological cutoffs do not become a fabricated blind experiment."""
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import unittest

from williams_ai.continual_trader import Outcome, PrivateExperience
from williams_ai.trader_marathon import marathon

START=datetime(2025, 1, 1, tzinfo=timezone.utc)
def iso(hour):
    return (START + timedelta(hours=hour)).isoformat()

def historical_candidates(symbol, rows, tf, *, tick_size, horizon):
    outcomes=[
        Outcome(symbol,"H1","LONG" if i%2 else "SHORT",iso(i),iso(i+3),
                (1.0,(i%10-5)*0.1,0.2,0.5,0.1,0.7),
                0.008 if i%4<3 else -0.018,i%4==3,"f"*64,
                "WM3","TC2.WM3")
        for i in range(1,186)
    ]
    return {
        "outcomes":outcomes,
        "source_sha256":"f"*64,
        "research_approximated_wm_candidates":185,
        "candidates_no_next_bar_trigger":0,
        "other_rejections":0,
        "setup_counts":{"WM3":185},
    }

class TemporalCutoffProvenanceTests(unittest.TestCase):
    def test_explicit_cutoff_and_retrospective_cutoff_are_differentiated(self):
        markets={symbol:{"H1":{"rows":[],"tick_size":.01}}
                 for symbol in ("ETHUSDT","BTCUSDT","SOLUSDT")}
        with TemporaryDirectory() as d:
            journal=PrivateExperience(Path(d)/"private.db")
            with patch("williams_ai.trader_marathon.signal_conditioned_outcomes",
                       side_effect=historical_candidates):
                explicit=marathon(
                    journal,markets,
                    frozen_cutoffs=(iso(100),iso(145),iso(190)),
                    min_train=30,min_validation=12,goal=10000000)
                exploratory=marathon(
                    journal,markets,min_train=30,min_validation=12,goal=10000000)
            self.assertEqual(explicit["temporal_split_cutoffs"],
                             [iso(100),iso(145),iso(190)])
            self.assertIn("OPERATOR_SPECIFIED",
                          explicit["temporal_split_provenance"])
            self.assertIn("RETROSPECTIVE",
                          exploratory["temporal_split_provenance"])
            self.assertFalse(explicit["execution_allowed"])
            self.assertFalse(exploratory["execution_allowed"])
            self.assertTrue(explicit["repeated_batches_not_new_independent_holdouts"])

    def test_invalid_time_cutoffs_fail_closed(self):
        with TemporaryDirectory() as d:
            journal=PrivateExperience(Path(d)/"private.db")
            with self.assertRaisesRegex(ValueError,"CUTOFF_ORDER"):
                marathon(journal,{},frozen_cutoffs=(iso(100),iso(90),iso(190)))
            with self.assertRaisesRegex(ValueError,"ALL_THREE"):
                marathon(journal,{},frozen_cutoffs=(iso(100),iso(190)))
            with self.assertRaises(ValueError):
                marathon(journal,{},frozen_cutoffs=(
                    "2025-01-01T00:00:00",iso(100),iso(190)))

if __name__=="__main__":
    unittest.main()
