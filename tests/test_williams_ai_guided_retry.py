"""Test supervised correction of wrong book claims, not mock 'AI mastery'."""
import unittest

from williams_ai.chart_reader import analyze_chart
from williams_ai.causal_paper_lab import frozen_markets
from williams_ai.guided_retry import GuidedReadOnlyModel


def observed():
    market=frozen_markets()["DOGEUSDT"]
    rows=market["bars"][:924]
    return analyze_chart("DOGEUSDT","H1",rows,as_of=rows[-1]["close_time"],
                         tick_size=market["tick_size"])


def answer(obs,setup,side="NONE",refs=()):
    return {"symbol":obs["symbol"],"as_of":obs["as_of"],"edition":"TC2",
            "setup":setup,"side":side,"entry_state":"NO_SETUP",
            "exit_basis":"NONE","rule_ids":list(refs),
            "execution_allowed":False,"trade_action":None}


class GuidedCorrectionTests(unittest.TestCase):
    def test_unsupported_wm2_retries_with_exact_source_without_orders(self):
        class Fake:
            def __init__(self):self.calls=[]
            def propose_chart_reasoning(self,obs,*,edition,prior_lessons=None):
                self.calls.append((edition,prior_lessons))
                if prior_lessons is None:
                    return answer(obs,"WM2","LONG",("TC2.WM2",))
                return answer(obs,"NO_SETUP")
        model=Fake()
        wrapper=GuidedReadOnlyModel(model)
        result=wrapper.propose_chart_reasoning(observed(),edition="TC2")
        self.assertEqual(result["setup"],"NO_SETUP")
        self.assertEqual(len(model.calls),2)
        report=wrapper.latest
        self.assertEqual(report["first_attempt_verdict"],"SOURCE_UNSUPPORTED")
        self.assertIn("WM2_SUPER_AO_NOT_OBSERVED",report["first_attempt_failures"])
        self.assertEqual(report["retry_verdict"],"PROVISIONAL_REASONING_ONLY")
        self.assertTrue(report["retry_is_safe_abstention"])
        self.assertFalse(report["retry_supported_substantive_claim"])
        self.assertTrue(report["no_future_bars_or_live_orders"])
        self.assertEqual(model.calls[1][1][0]["source"],"TC2:RULE:TC2.WM2")
        self.assertNotIn("price",model.calls[1][1][0])

    def test_initial_valid_abstention_is_not_rewarded_with_more_inference(self):
        class Fake:
            def __init__(self):self.calls=0
            def propose_chart_reasoning(self,obs,*,edition,prior_lessons=None):
                self.calls+=1
                return answer(obs,"NO_SETUP")
        fake=Fake()
        wrapper=GuidedReadOnlyModel(fake)
        self.assertEqual(wrapper.propose_chart_reasoning(observed())["setup"],"NO_SETUP")
        self.assertEqual(fake.calls,1)
        self.assertFalse(wrapper.latest["retry_performed"])
        self.assertTrue(wrapper.latest["first_attempt_is_safe_abstention"])

    def test_failed_retry_cannot_be_relabelled_a_good_trade(self):
        class Fake:
            def __init__(self):self.calls=0
            def propose_chart_reasoning(self,obs,*,edition,prior_lessons=None):
                self.calls+=1
                return answer(obs,"WM2","LONG",("TC2.WM2",))
        fake=Fake()
        wrap=GuidedReadOnlyModel(fake)
        report_answer=wrap.propose_chart_reasoning(observed())
        self.assertEqual(report_answer["setup"],"WM2")
        self.assertEqual(fake.calls,2)
        self.assertEqual(wrap.latest["retry_verdict"],"UNSUPPORTED_CLAIM")
        self.assertIn("WM2_SUPER_AO_NOT_OBSERVED",
                      wrap.latest["retry_hard_failures"])
        self.assertFalse(wrap.latest["retry_supported_substantive_claim"])

    def test_corrupt_first_model_output_is_explicitly_accounted_for(self):
        class Fake:
            def __init__(self):self.calls=[]
            def propose_chart_reasoning(self,obs,*,edition,prior_lessons=None):
                self.calls.append(prior_lessons)
                if prior_lessons is None:
                    raise ValueError("GGUF_EMPTY_DECISION_RESPONSE")
                return answer(obs,"NO_SETUP")
        f=Fake()
        wrap=GuidedReadOnlyModel(f)
        self.assertEqual(wrap.propose_chart_reasoning(observed())["setup"],"NO_SETUP")
        self.assertEqual(wrap.latest["first_attempt_verdict"],"MODEL_ERROR")
        self.assertEqual(wrap.latest["first_attempt_failures"],[
            "GGUF_EMPTY_DECISION_RESPONSE"
        ])
        self.assertTrue(wrap.latest["retry_is_safe_abstention"])

    def test_no_future_chart_prices_or_private_book_snippets_in_feedback(self):
        class Fake:
            def __init__(self):self.feedback=None
            def propose_chart_reasoning(self,obs,*,edition,prior_lessons=None):
                if prior_lessons is not None:
                    self.feedback=prior_lessons
                    return answer(obs,"NO_SETUP")
                return answer(obs,"WM2","LONG",("TC2.WM2",))
        student=Fake()
        wrap=GuidedReadOnlyModel(student)
        wrap.propose_chart_reasoning(observed())
        self.assertLessEqual(len(student.feedback),2)
        self.assertTrue(all("source" in s and "excerpt" in s for s in student.feedback))
        self.assertTrue(all(s["source"].startswith("TC2:RULE:")
                            for s in student.feedback))
        self.assertFalse(any("private" in s["excerpt"].lower()
                             for s in student.feedback))


if __name__=="__main__":
    unittest.main()
