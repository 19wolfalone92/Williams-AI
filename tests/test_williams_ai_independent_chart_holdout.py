"""Real frozen Binance 15-asset H1 prospective-as-of evaluation contracts."""
import copy
import json
from pathlib import Path
import unittest

from williams_ai.independent_chart_holdout import (
    build_blind_market_cases, grade_blind_predictions
)

DATA=Path(__file__).resolve().parents[1]/"examples"/"ai_binance_15pair_h1_prior_holdout_20260902.json"


class IndependentRealMarketHoldoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.archive=json.loads(DATA.read_text(encoding="utf-8"))
        cls.questions,cls.outcomes,cls.seal=build_blind_market_cases(cls.archive)

    def test_two_nonoverlapping_future_blind_real_h1_windows_for_15_assets(self):
        self.assertEqual(len(self.questions),30)
        self.assertEqual(len(self.outcomes),30)
        self.assertFalse(self.seal["future_market_bars_in_inference_prompt"])
        for q in self.questions:
            self.assertEqual(len(q["market_context"]["bars"]),240)
            self.assertFalse(q["future_data_in_prompt"])
            self.assertIsNone(q["trade_action"])
            self.assertNotIn("true_class",q)
            self.assertNotIn("future_return",q)

    def test_changing_only_future_preserves_precommit_context_and_key_changes(self):
        changed=copy.deepcopy(self.archive)
        pair=changed["pairs"]["BTCUSDT"]["klines"]
        # cutoff 420; the outcome is bar 444 (0-indexed).
        future=pair[444]
        future[1] *= 1.11
        future[2] *= 1.11
        future[3] *= 1.11
        future[4] *= 1.11
        q,o,seal=build_blind_market_cases(changed)
        self.assertEqual(q,self.questions)
        self.assertEqual(seal["market_context_sha256"],self.seal["market_context_sha256"])
        self.assertNotEqual(o,self.outcomes)
        self.assertNotEqual(seal["result_keys_sha256"],self.seal["result_keys_sha256"])

    def test_random_uniform_probs_are_finite_and_no_profit_claim(self):
        rows=[{"case_id":o["case_id"],
              "probabilities":{"UP":1/3,"DOWN":1/3,"RANGE":1/3}}
              for o in self.outcomes]
        result=grade_blind_predictions(self.outcomes,rows)
        self.assertEqual(result["cases"],30)
        self.assertTrue(result["trading_profit_not_measured"])
        self.assertAlmostEqual(result["multiclass_brier"],2/9)

    def test_fake_prognoses_must_not_use_unknown_labels_or_partial_cohort(self):
        with self.assertRaisesRegex(ValueError,"EXACT_COHORT"):
            grade_blind_predictions(self.outcomes,[
                {"case_id":self.outcomes[0]["case_id"],
                 "probabilities":{"UP":1,"DOWN":0,"RANGE":0}}])

if __name__=="__main__":
    unittest.main()
