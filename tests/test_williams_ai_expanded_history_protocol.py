"""Validate future-blind frozen historical exam design, not mocked competence."""
import unittest
from collections import Counter

from williams_ai.real_gguf_expanded_history_exam import (
    SOURCE_CASES,DIRECTION_CUT,MODEL_EDITION_COUNTS,run_real_expanded_history,
)
from williams_ai.causal_paper_lab import frozen_markets


class HistoricalExamIntegrity(unittest.TestCase):
    def test_predeclared_book_cases_balanced_and_known_closed_cutoffs(self):
        markets=frozen_markets()
        self.assertEqual(len(markets),15)
        self.assertEqual(len(SOURCE_CASES),16)
        self.assertEqual(dict(Counter(x[2] for x in SOURCE_CASES)),
                         MODEL_EDITION_COUNTS)
        self.assertEqual(len({(s,i,edition) for s,i,edition in SOURCE_CASES}),16)
        for symbol,index,edition in SOURCE_CASES:
            self.assertIn(symbol,markets)
            self.assertLess(index+24,len(markets[symbol]["bars"]))
            self.assertIn(edition,("TC1","NTD","TC2","COURSE"))
        self.assertLess(DIRECTION_CUT+24,1200)

    def test_fake_or_missing_weight_verification_never_runs_exam(self):
        class NoExecutionModel:
            def propose_chart_reasoning(self,*args,**kwargs):
                self.fail=True
                raise RuntimeError("should never be called")
        model=NoExecutionModel()
        for invalid in (
            {},
            {"verified_model_magic":"FAKE"},
            {"verified_model_magic":"GGUF","model_sha256":"too-short"},
        ):
            with self.assertRaises(ValueError):
                run_real_expanded_history(model,verified_weights=invalid)
            self.assertFalse(getattr(model,"fail",False))

    def test_model_that_fails_all_reasoning_is_not_counted_as_good(self):
        # Testing on one script's entire sealed cohort is slow but deterministic.
        # A bad model must be shown as 16 explicit failures and no real entry,
        # not misleading 100%-accuracy NO_SETUP responses.
        class FailingModel:
            def propose_chart_reasoning(self,*args,**kwargs):
                raise ValueError("Intentionally invalid model output")
            def propose_24h_research_forecast(self,*args,**kwargs):
                raise ValueError("Intentionally invalid directional output")
        result=run_real_expanded_history(
            FailingModel(),verified_weights={
                "verified_model_magic":"GGUF",
                "model_sha256":"a"*64,
            },code_id="mock-only-unit-test",
        )
        self.assertEqual(result["source_audit_verdicts"],
                         {"MODEL_INVALID_OR_RUNTIME_FAILURE":16})
        self.assertEqual(result["source_consistent_cases"],0)
        self.assertEqual(result["directional_settled_cases"],0)
        self.assertEqual(len(result["directional_inference_failures"]),15)
        self.assertIsNone(result["directional_report"])
        self.assertFalse(result["independent_blind_holdout"])
        self.assertEqual(result["real_trading_qualification"],"NOT_QUALIFIED")
        self.assertFalse(result["execution_allowed"])
        self.assertIsNone(result["trade_action"])


if __name__=="__main__":
    unittest.main()
