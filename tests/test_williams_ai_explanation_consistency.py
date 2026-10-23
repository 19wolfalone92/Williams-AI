"""Regression from REAL Qwen3 1.7B: JSON setup and prose must agree.

In the real run the model selected WM3, WM2 and TC1_FRACTAL in structured
JSON while writing 'NO_SETUP' or 'No setup found'. That is NOT book mastery,
even if numeric setup happened to match a closed-candle candidate.
"""
import tempfile
import unittest
from pathlib import Path

from williams_ai.algorithm_guard import audit_reasoning
from williams_ai.causal_paper_lab import frozen_markets
from williams_ai.chart_reader import analyze_chart
from williams_ai.guided_retry import GuidedReadOnlyModel,safe_error_code
from williams_ai.local_model import LocalWilliamsModel


def chart():
    market=frozen_markets()["DOGEUSDT"]
    rows=market["bars"][:924]
    return analyze_chart("DOGEUSDT","H1",rows,as_of=rows[-1]["close_time"],
                         tick_size=market["tick_size"])


def decision(observation,setup="NO_SETUP",reason="NO_SETUP",side="NONE"):
    return {
        "symbol":observation["symbol"],"as_of":observation["as_of"],
        "edition":"TC2","setup":setup,"side":side,
        "reason":reason,"entry_state":"NO_SETUP","rule_ids":[],
        "exit_basis":"NONE","trade_action":None,"execution_allowed":False,
    }


class ContradictoryMarketReasoningTests(unittest.TestCase):
    def test_structured_wm2_and_explicit_no_setup_explanation_rejected(self):
        obs=chart()
        for reason in ("NO_SETUP", "no_setup.", "No setup found.",
                       "No confirmed setup detected.",
                       "Нет подтвержденного сигнала.",
                       "Нет подтверждённого паттерна"):
            with self.subTest(reason=reason):
                report=audit_reasoning(obs,decision(obs,"WM2",reason,"LONG"))
                self.assertIn("EXPLANATION_CONTRADICTS_SELECTED_SETUP",
                              report["hard_failures"])
                self.assertEqual(report["verdict"],"UNSUPPORTED_CLAIM")
                self.assertFalse(report["execution_allowed"])

    def test_actual_book_candidate_does_not_rescue_contradictory_explanation(self):
        obs=chart()
        obs["candidates"].append({
            "type":"WM2_SUPER_AO","side":"LONG",
            "state":"MOMENTUM_CANDIDATE_NOT_ORDER",
        })
        proposal=decision(obs,"WM2","NO_SETUP","LONG")
        proposal["rule_ids"]=["TC2.WM2"]
        report=audit_reasoning(obs,proposal)
        self.assertIn("EXPLANATION_CONTRADICTS_SELECTED_SETUP",
                      report["hard_failures"])
        self.assertNotIn("WM2_SUPER_AO_NOT_OBSERVED",report["hard_failures"])

    def test_abstention_and_non_executed_setup_distinguished(self):
        obs=chart()
        self.assertNotIn("EXPLANATION_CONTRADICTS_SELECTED_SETUP",
                          audit_reasoning(obs,decision(obs))["hard_failures"])
        obs["candidates"].append({
            "type":"WM2_SUPER_AO","side":"LONG",
            "state":"MOMENTUM_CANDIDATE_NOT_ORDER",
        })
        prop=decision(obs,"WM2",
                      "A Super AO candidate exists but NO order was executed.",
                      "LONG")
        prop["rule_ids"]=["TC2.WM2"]
        self.assertNotIn("EXPLANATION_CONTRADICTS_SELECTED_SETUP",
                          audit_reasoning(obs,prop)["hard_failures"])

    def test_correct_same_edition_lesson_on_model_prose_conflict(self):
        class Mock:
            def __init__(self):self.calls=[]
            def propose_chart_reasoning(self,obs,*,edition,prior_lessons=None):
                self.calls.append(prior_lessons)
                if prior_lessons is None:
                    return decision(obs,"WM2","NO_SETUP","LONG")
                return decision(obs,"NO_SETUP","No signal confirmed")
        mock=Mock()
        guided=GuidedReadOnlyModel(mock)
        self.assertEqual(guided.propose_chart_reasoning(chart())["setup"],"NO_SETUP")
        self.assertEqual(len(mock.calls),2)
        self.assertIn("EXPLANATION_CONTRADICTS_SELECTED_SETUP",
                      guided.latest["first_attempt_failures"])
        self.assertEqual(mock.calls[1][0]["source"],"TC2:RULE:TC2.WM2")
        self.assertTrue(guided.latest["retry_is_safe_abstention"])
        self.assertFalse(guided.latest["retry_supported_substantive_claim"])

    def test_runtime_llama_error_has_stable_non_private_diagnostic(self):
        class Throws:
            def __init__(self,**kwargs):pass
            def create_chat_completion(self,**kwargs):
                raise ValueError("n_ctx requested context exceeded: secret /private/model")
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"dummy.gguf";p.write_bytes(b"fake-test")
            model=LocalWilliamsModel(p,loader=Throws,use_decision_schema=True)
            with self.assertRaisesRegex(
                ValueError,"^GGUF_RUNTIME_CONTEXT_WINDOW_EXCEEDED$"
            ):
                model.propose_chart_reasoning(chart(),edition="TC2")
        self.assertEqual(
            safe_error_code(ValueError("GGUF_RUNTIME_CONTEXT_WINDOW_EXCEEDED")),
            "GGUF_RUNTIME_CONTEXT_WINDOW_EXCEEDED",
        )
        self.assertEqual(safe_error_code(ValueError("/private/secret")),
                         "UNCLASSIFIED_MODEL_ERROR")


if __name__=="__main__":
    unittest.main()
