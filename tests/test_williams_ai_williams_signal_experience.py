"""TC2 signal-conditioned research samples do not fabricate actual trades."""
import copy
from datetime import datetime,timedelta,timezone
import unittest
from unittest.mock import patch

from williams_ai.williams_signal_experience import signal_conditioned_outcomes
from williams_ai.causal_paper_lab import frozen_markets


def candles(count=260):
    begin=datetime(2024,1,1,tzinfo=timezone.utc)
    return [{"close_time":(begin+timedelta(hours=i)).isoformat(),
             "open":100.0,"high":101.0,"low":99.0,
             "close":100.0,"volume":100,"closed":True}
            for i in range(count)]

def candidate(bars,ind,mode,i,tick):
    if mode!="WM3" or i%10: return []
    return [{"formed":i,"side":"LONG","trigger":100.5,
             "stop":98.0,"type":"WM3","source_rule":"TC2.WM3"}]

class SignalLearningTests(unittest.TestCase):
    def test_wm3_requires_post_close_price_touch_and_provenance(self):
        data=candles()
        with patch("williams_ai.williams_signal_experience._candidate",side_effect=candidate):
            result=signal_conditioned_outcomes(
                "ETHUSDT",data,"H1",tick_size=.01,horizon=10,start_index=80)
        self.assertGreater(result["research_approximated_wm_candidates"],0)
        self.assertGreater(len(result["outcomes"]),0)
        self.assertEqual(result["real_gguf_decisions_made"],0)
        self.assertEqual(result["exchange_fills_observed"],0)
        for row in result["outcomes"]:
            self.assertEqual(row.setup,"WM3")
            self.assertEqual(row.rule_id,"TC2.WM3")
            self.assertLess(row.decision_at,row.resolved_at)
            self.assertFalse(result["execution_allowed"])
        data=[dict(r) for r in data]
        for row in data[140:]:
            row["high"]=100.1
        with patch("williams_ai.williams_signal_experience._candidate",side_effect=candidate):
            next_version=signal_conditioned_outcomes(
                "ETHUSDT",data,"H1",tick_size=.01,horizon=10,start_index=80)
        initial=[o for o in result["outcomes"] if
                 o.decision_at<data[120]["close_time"]]
        changed=[o for o in next_version["outcomes"] if
                 o.decision_at<data[120]["close_time"]]
        self.assertEqual(initial,changed,
            "Later exchange prices must not rewrite already resolved learning")

    def test_real_frozen_binance_examples_are_real_candle_derived_not_gguf_claims(self):
        data=frozen_markets()
        for symbol in ("ETHUSDT","BTCUSDT","SOLUSDT"):
            rows=data[symbol]["bars"]
            outcome=signal_conditioned_outcomes(
                symbol,rows,"H1",tick_size=data[symbol]["tick_size"],horizon=12)
            self.assertGreaterEqual(outcome["research_approximated_wm_candidates"],0)
            self.assertEqual(outcome["source_sha256"].__len__(),64)
            self.assertFalse(outcome["canonical_4book_wave_interpretations_verified"])
            for s in outcome["outcomes"]:
                self.assertTrue(s.rule_id.startswith("TC2."))
                self.assertIn(s.setup,("WM1","WM2","WM3"))
                self.assertLessEqual(s.resolved_at,rows[-1]["close_time"])

    def test_gapped_future_no_candle_is_rejected(self):
        rows=candles()
        rows[50]["close_time"]=(datetime.fromisoformat(rows[50]["close_time"])+
                                 timedelta(hours=2)).isoformat()
        with self.assertRaises(ValueError):
            signal_conditioned_outcomes(
                "ETHUSDT",rows,"H1",tick_size=.01)
        rows=candles()
        rows[-1]["closed"]=False
        with self.assertRaises(ValueError):
            signal_conditioned_outcomes(
                "ETHUSDT",rows,"H1",tick_size=.01)

if __name__=="__main__":
    unittest.main()
