"""Practical comprehension requires closed facts, uncertainty and no order authority."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import unittest

from williams_ai.practical_market_teacher import rehearse_trade_reasoning


def make_history(count=180):
    epoch=datetime(2026,9,1,tzinfo=timezone.utc)
    out=[]
    for j in range(count):
        mid=200+0.08*j+(j%17-8)*0.45
        out.append({
            "close_time":(epoch+timedelta(hours=j+1)).isoformat(),
            "open":mid-0.4,"high":mid+1.1,"low":mid-1.4,
            "close":mid+0.2,"volume":2+j/20,"closed":True,
        })
    return out


def learn(bars,ed="TC2"):
    return rehearse_trade_reasoning("ETHUSDT","H1",bars,
                                   edition=ed,tick_size=0.01)


class WilliamsPracticalMarketTeacherTests(unittest.TestCase):
    def setUp(self):
        self.bars=make_history()

    def test_all_four_edition_profiles_rehearse_both_sides_and_abstention(self):
        for book in ("TC1","NTD","TC2","COURSE"):
            with self.subTest(book=book):
                lesson=learn(self.bars,book)
                self.assertEqual(lesson["book"],book)
                self.assertFalse(lesson["execution_allowed"])
                self.assertIsNone(lesson["trade_action"])
                self.assertFalse(lesson["model_weights_modified_here"])
                self.assertFalse(lesson["elliott_wave_degree_certified"])
                self.assertFalse(lesson["future_bars_read"])
                self.assertEqual(len(lesson["five_competing_falsifiable_scenarios"]),5)
                self.assertEqual({x["hypothesis"] for x in lesson[
                    "five_competing_falsifiable_scenarios"]},{
                    "CONTINUATION_UP","CONTINUATION_DOWN",
                    "CORRECTION_OR_COMPLEX_RANGE",
                    "POSSIBLE_REVERSAL_UP","POSSIBLE_REVERSAL_DOWN"})
                self.assertTrue(all(
                    s["prediction_probability"] is None and
                    s["wave_degree_confirmed"] is False
                    for s in lesson["five_competing_falsifiable_scenarios"]))
                self.assertTrue(all(x["source_rule"].startswith(book+".")
                    for x in lesson["same_edition_current_signal_observations_not_fills"]))
                self.assertEqual(len(lesson["source_learning_objectives"]),3)

    def test_future_append_never_changes_old_prefix_lesson(self):
        old=learn(self.bars[:-1])
        appended=deepcopy(self.bars)
        appended[-1]["close"]+=0.1
        appended[-1]["high"]+=0.2
        new=learn(appended[:-1])
        self.assertEqual(new,old)
        self.assertNotEqual(learn(self.bars)["review_prefix_sha256"],
                            old["review_prefix_sha256"])

    def test_ntd_ao_cannot_skip_first_fractal_admission(self):
        out=learn(self.bars,"NTD")
        self.assertEqual(out["edition_specific_entry_gate"]["status"],
            "INITIAL_OUTSIDE_ALLIGATOR_FRACTAL_TRIGGER_NOT_VERIFIED")
        self.assertTrue(out["edition_specific_entry_gate"][
            "no_independent_initial_AO_AC_ZONE_BALANCE_ENTRY"])
        self.assertTrue(all(not x["source_rule"].startswith("TC2")
                            for x in out["same_edition_current_signal_observations_not_fills"]))

    def test_tc2_opposite_wise_man_not_close_and_book_example_not_risk(self):
        out=learn(self.bars,"TC2")
        self.assertEqual(out["edition_specific_entry_gate"]["book"],"TC2")
        self.assertIn("opposite Wise Man",out["edition_specific_entry_gate"]["exit_notes"])
        self.assertIsNone(out["claimed_win_rate"])
        self.assertFalse(out["autonomous_execution_allowed"])

    def test_no_future_gaps_unclosed_candle_or_bad_inputs(self):
        bad=deepcopy(self.bars)
        bad[-1]["closed"]=False
        with self.assertRaises(ValueError):
            learn(bad)
        bad=deepcopy(self.bars)
        bad[-1]["close_time"]=(datetime.fromisoformat(bad[-1]["close_time"])+
                                 timedelta(hours=2)).isoformat()
        with self.assertRaisesRegex(ValueError,"GAPPED"):
            learn(bad)
        bad=deepcopy(self.bars)
        bad[-1]["high"]<bad[-1]["close"]
        bad[-1]["high"]=0.0
        with self.assertRaises(ValueError):
            learn(bad)
        with self.assertRaises(ValueError):
            learn(self.bars,"TC2+NTD")
        with self.assertRaises(ValueError):
            learn(self.bars[:42])


if __name__=="__main__":
    unittest.main()
