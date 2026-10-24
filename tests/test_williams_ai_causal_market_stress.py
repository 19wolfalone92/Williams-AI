"""Real OHLC source-level metamorphic / anti-lookahead market analysis proof."""
import copy
import unittest
from williams_ai.causal_paper_lab import frozen_markets
from williams_ai.causal_market_stress import (
    CUTOFFS,EXPECTED_HARD_FAILURES,_false_model_claim,stress_one,
)
from williams_ai.chart_reader import analyze_chart
from williams_ai.algorithm_guard import audit_reasoning


class ClosedHistoricalCausality(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.markets=frozen_markets()

    def test_h1_h4_d1_and_book_critic_detect_future_and_fake_execution(self):
        sample=self.markets["BTCUSDT"]
        outcome=stress_one("BTCUSDT",sample["bars"],
                           sample["tick_size"],899)
        self.assertTrue(outcome["future_poison_has_no_effect"])
        self.assertTrue(outcome["passing_future_to_analyzer_hard_rejected"])
        self.assertTrue(outcome["all_four_book_fake_executions_rejected"])
        self.assertEqual(set(outcome["book_guard_falsification_codes"]),
                         {"TC1","NTD","TC2","COURSE"})
        for edition,failures in outcome["book_guard_falsification_codes"].items():
            with self.subTest(edition=edition):
                self.assertEqual(set(failures),EXPECTED_HARD_FAILURES)
        self.assertFalse(outcome["execution_allowed"])
        self.assertIsNone(outcome["trade_action"])
        self.assertFalse(outcome["real_GGUF_was_queried"])
        self.assertEqual(len(outcome["top_down_hypotheses"]),2)
        self.assertTrue(all(x["not_confirmed"]
                            for x in outcome["top_down_hypotheses"]))

    def test_all_15_pairs_causal_sample_no_future(self):
        self.assertEqual(len(self.markets),15)
        for sym in sorted(self.markets):
            sample=self.markets[sym]
            result=stress_one(sym,sample["bars"],
                              sample["tick_size"],CUTOFFS[0])
            with self.subTest(sym=sym):
                self.assertEqual(result["at_index"],699)
                self.assertEqual(result["current_h1_facts"]["wave_degree_verified"],
                                 False)
                self.assertTrue(result["future_poison_has_no_effect"])
                self.assertTrue(result["all_four_book_fake_executions_rejected"])

    def test_unfinished_future_not_allowed_to_shift_historical_cutoff(self):
        sample=self.markets["ETHUSDT"]
        bars=sample["bars"]
        i=899
        existing=analyze_chart("ETHUSDT","H1",bars[:i+1],
                               as_of=bars[i]["close_time"],
                               tick_size=sample["tick_size"])
        self.assertFalse(existing["execution_allowed"])
        poisoned=copy.deepcopy(bars[:i+2])
        poisoned[-1]["closed"]=False
        with self.assertRaisesRegex(ValueError,"Only explicitly finalized|NO_LOOKAHEAD"):
            analyze_chart("ETHUSDT","H1",poisoned,
                          as_of=bars[i]["close_time"],
                          tick_size=sample["tick_size"])

    def test_guard_fails_invalid_realistic_golden_chart_source_claim(self):
        sample=self.markets["LINKUSDT"]
        bars=sample["bars"][:1100]
        observed=analyze_chart("LINKUSDT","H1",bars,
                               as_of=bars[-1]["close_time"],
                               tick_size=sample["tick_size"])
        for ed in ("TC1","NTD","TC2","COURSE"):
            hallucination=_false_model_claim("LINKUSDT",observed["as_of"],ed)
            outcome=audit_reasoning(observed,hallucination)
            self.assertTrue(EXPECTED_HARD_FAILURES.issubset(outcome["hard_failures"]))
            self.assertEqual(outcome["verdict"],"UNSUPPORTED_CLAIM")
            self.assertFalse(outcome["execution_allowed"])


if __name__=="__main__":
    unittest.main()
