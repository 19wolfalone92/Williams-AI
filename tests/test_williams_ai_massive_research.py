"""Contract tests for gigantic counterfactual stress and real-history learning."""
import unittest

from williams_ai.ten_million_stress import ten_million_stress
from williams_ai.many_token_learning import (
    TRAIN_MONTHS, TRAIN_SYMBOLS, obtain_predeclared_training, learn_and_evaluate,
)


class MassiveResearchContracts(unittest.TestCase):
    def test_6000_distinct_counterfactuals_cannot_create_fake_trades(self):
        output=ten_million_stress(count=6000)
        self.assertEqual(output["checks_executed"],6000)
        self.assertTrue(output["all_checked_without_error"])
        self.assertEqual(output["unique_scenario_ids"],6000)
        self.assertEqual(sum(output["checks_by_kind"].values()),6000)
        self.assertEqual(set(output["checks_by_kind"]),{
            "positive_price_scale_invariance",
            "long_short_mirror_symmetry",
            "no_future_candle_contamination_of_confirmed_pivot",
            "two_right_closed_bar_confirmation_and_direction_scaling",
        })
        self.assertEqual(output["source_historical_coins"],15)
        self.assertEqual(output["actual_GGUF_inferences"],0)
        self.assertIsNone(output["trade_action"])
        self.assertFalse(output["execution_allowed"])
        self.assertTrue(output["repeated_real_historical_windows_used_with_different_transformations"])

    def test_invalid_massive_counts_are_rejected(self):
        for invalid in (-1,0,True,10_000_001):
            with self.subTest(value=invalid):
                with self.assertRaises(ValueError):
                    ten_million_stress(count=invalid)

    def test_training_cannot_pretend_missing_or_fake_data_are_verified(self):
        with self.assertRaises(ValueError):
            learn_and_evaluate({
                "markets":{},"available":0,"requested":51,
                "missing":{},"raw_closed_h1_bars":0,
            })
        with self.assertRaises(ValueError):
            obtain_predeclared_training(workers=0)

    def test_predeclared_symbol_universe_and_months(self):
        self.assertEqual(TRAIN_MONTHS,("2025-01","2025-02"))
        self.assertGreaterEqual(len(TRAIN_SYMBOLS),40)
        self.assertEqual(len(TRAIN_SYMBOLS),len(set(TRAIN_SYMBOLS)))


if __name__=="__main__":
    unittest.main()
