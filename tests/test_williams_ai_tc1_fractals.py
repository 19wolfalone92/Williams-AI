"""Book-checked initiating fractal patterns in TC1, distinct from TC2."""
from datetime import datetime,timezone,timedelta
from types import SimpleNamespace
import unittest

from williams_ai.tc1_fractals import tc1_initiating_fractal_evidence
from williams_ai.algorithm_guard import audit_reasoning
from williams_ai.causal_paper_lab import frozen_markets
from williams_ai.chart_reader import analyze_chart


def bars(heights, lows=None):
    lowvalues=[2.0]*len(heights) if lows is None else lows
    origin=datetime(2026,10,1,tzinfo=timezone.utc)
    return [SimpleNamespace(high=float(h),low=float(lo),
            at=origin+timedelta(hours=i))
            for i,(h,lo) in enumerate(zip(heights,lowvalues))]


def select(view,pivot,side):
    return [x for x in view["recent_formations"]
            if x["pivot_index"]==pivot and x["side"]==side]


class TC1FractalProofTests(unittest.TestCase):
    def test_equal_neighbor_skipped_and_delayed_confirmation(self):
        heights=[8,9,7,8,9,10,10,8,7,9,8]
        early=tc1_initiating_fractal_evidence(bars(heights[:8]))
        self.assertEqual(select(early,5,"LONG"),[])
        confirmed=tc1_initiating_fractal_evidence(bars(heights[:9]))
        c=select(confirmed,5,"LONG")
        self.assertEqual(len(c),1)
        self.assertEqual(c[0]["confirmed_at_index"],8)
        self.assertTrue(c[0]["same_level_bars_skipped"])
        self.assertEqual(c[0]["state"],"EQUAL_LEVEL_PLATEAU_REQUIRES_VISUAL_REVIEW")
        self.assertFalse(c[0]["can_prove_price_fill"])
        self.assertIsNone(c[0]["trade_action"])
        late=tc1_initiating_fractal_evidence(bars(heights))
        self.assertEqual(select(late,5,"LONG")[0]["confirmed_at"],c[0]["confirmed_at"])

    def test_strict_five_bar_normal_fractal_t_plus_two(self):
        heights=[8,9,7,8,9,11,10,8,7,9,8]
        early=tc1_initiating_fractal_evidence(bars(heights[:7]))
        self.assertEqual(select(early,5,"LONG"),[])
        when=tc1_initiating_fractal_evidence(bars(heights[:8]))
        got=select(when,5,"LONG")[0]
        self.assertEqual(got["confirmed_at_index"],7)
        self.assertFalse(got["same_level_bars_skipped"])
        self.assertEqual(got["state"],"CONFIRMED_CLOSED_PRICE_FORMATION_NOT_ENTRY")

    def test_higher_right_bar_invalidates_and_no_future_evidence(self):
        highs=[8,9,7,8,9,10,10,12,7,9,8]
        view=tc1_initiating_fractal_evidence(bars(highs))
        self.assertEqual(select(view,5,"LONG"),[])
        with self.assertRaises(ValueError):
            tc1_initiating_fractal_evidence(bars(highs),lookback=2)

    def test_sell_mirrors_equal_low_bar_case(self):
        heights=[16.0]*11
        lows=[9,8,10,9,8,7,7,9,10,8,9]
        view=tc1_initiating_fractal_evidence(bars(heights,lows))
        found=select(view,5,"SHORT")[0]
        self.assertEqual(found["confirmed_at_index"],8)
        self.assertEqual(found["level"],7)
        self.assertTrue(found["same_level_bars_skipped"])

    def test_only_tc1_can_claim_this_shape_while_neither_can_trade(self):
        market=frozen_markets()["ETHUSDT"]
        candle_rows=market["bars"][:310]
        as_of=candle_rows[-1]["close_time"]
        chart=analyze_chart("ETHUSDT","H1",candle_rows,as_of=as_of,
                            tick_size=market["tick_size"])
        self.assertFalse(chart["execution_allowed"])
        self.assertIn("TC1",chart["edition_evidence"])
        research=chart["edition_evidence"]["TC1"]
        self.assertFalse(research["execution_allowed"])
        self.assertIsNone(research["trade_action"])
        observation={**chart,"edition_evidence":{**chart["edition_evidence"],
          "TC1":{"recent_formations":[{"type":"TC1_INITIATING_FRACTAL",
          "side":"LONG","same_level_bars_skipped":True}]}}}
        idea={"symbol":"ETHUSDT","as_of":as_of,"edition":"TC1",
              "setup":"TC1_FRACTAL","side":"LONG","rule_ids":["TC1.FRACTAL"],
              "trade_action":None,"execution_allowed":False}
        evaluated=audit_reasoning(observation,idea)
        self.assertEqual(evaluated["verdict"],"PROVISIONAL_REASONING_ONLY")
        self.assertIn("EQUAL_LEVEL_FRACTAL_REQUIRES_VISUAL_INSPECTION",
                      evaluated["uncertainties"])
        bad=audit_reasoning(observation,{**idea,"edition":"TC2"})
        self.assertIn("TC1_INITIATING_FRACTAL_CANNOT_USE_OTHER_EDITION",bad["hard_failures"])
        self.assertIn("CROSS_EDITION_RULE_MIXING",bad["hard_failures"])


if __name__=="__main__":
    unittest.main()
