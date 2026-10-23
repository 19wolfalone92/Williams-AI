"""A model that always abstains is safe but has NOT learned Williams trading."""
import unittest

from williams_ai.real_gguf_history_exam import run_actual_history_exam


class AlwaysAbstain:
    def propose_chart_reasoning(self,chart,*,edition):
        return {
            "symbol":chart["symbol"],"as_of":chart["as_of"],
            "edition":edition,"setup":"NO_SETUP","side":"NONE",
            "entry_state":"NO_SETUP","exit_basis":"NONE",
            "rule_ids":[],"reason":"Unverified evidence; no trade.",
            "trade_action":None,"execution_allowed":False,
            "exchange_fill_confirmed":False,"exchange_sl_confirmed":False,
            "wave_degree_verified":False,
        }


class SafeAbstentionIsNotMastery(unittest.TestCase):
    def test_all_eight_safe_no_setup_is_not_certified_as_williams_mastery(self):
        report=run_actual_history_exam(
            AlwaysAbstain(),
            verified_weights={"verified_model_magic":"GGUF","model_sha256":"TEST_MOCK"},
        )
        self.assertEqual(report["model_supported_reasoning_cases"],8)
        self.assertEqual(report["model_no_setup_responses"],8)
        self.assertEqual(report["source_consistent_substantive_cases"],0)
        self.assertTrue(report["abstentions_are_not_book_mastery"])
        self.assertFalse(report["actual_trading_qualification"])
        self.assertEqual(
            report["strict_safety_qualification"],
            "CONSISTENT_RESEARCH_ONLY_NOT_TRADING_QUALIFIED",
        )
        self.assertFalse(report["execution_allowed"])


if __name__=="__main__":
    unittest.main()
