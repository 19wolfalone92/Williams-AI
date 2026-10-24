"""Book theory -> actual CLOSED chart -> Qwen -> independent reasoning audit."""
import unittest

from williams_ai.reasoning_trial import evaluate_historical_thought
from test_williams_ai_practical_market_teacher import make_history


class ObservingModel:
    def __init__(self):
        self.calls=[]
    def propose_chart_reasoning(self,analysis,*,edition,**kwargs):
        self.calls.append((edition,analysis))
        return {
            "symbol":analysis["symbol"],"as_of":analysis["as_of"],
            "edition":edition,"setup":"NO_SETUP","side":"NONE",
            "entry_state":"NO_SETUP","exit_basis":"NONE",
            "rule_ids":[],"trade_action":None,"execution_allowed":False,
            "reason":"No confirmed price break or exchange fill.",
        }


class PracticalKnowledgeToInferenceTests(unittest.TestCase):
    def test_closed_prefix_teaching_reaches_every_four_book_model(self):
        rows=make_history(170)
        for edition in ("TC1","NTD","TC2","COURSE"):
            with self.subTest(edition=edition):
                model=ObservingModel()
                report=evaluate_historical_thought(
                    "ETHUSDT",rows,tick_size=0.01,bar_index=100,
                    edition=edition,model=model,include_topdown=False)
                self.assertEqual(len(model.calls),1)
                sent=model.calls[0][1]
                lesson=sent["practical_market_teacher"]
                self.assertEqual(lesson["book"],edition)
                self.assertEqual(lesson["as_of"],rows[100]["close_time"])
                self.assertEqual(lesson["closed_bar_count"],101)
                self.assertEqual(len(lesson["five_competing_falsifiable_scenarios"]),5)
                self.assertFalse(lesson["execution_allowed"])
                self.assertEqual(lesson["source_learning_objectives"][0]["rule_ids"][0][:3],
                                 edition[:3])
                self.assertTrue(report["independent_practical_decision_teacher_supplied"])
                self.assertEqual(report["competing_market_hypotheses_rehearsed"],5)
                self.assertFalse(report["model_saw_future_bars"])
                self.assertFalse(report["model_weights_trained_here"])
                self.assertFalse(report["execution_allowed"])

    def test_no_model_no_pseudo_qwen_skill_claim(self):
        rows=make_history()
        proposal={"symbol":"ETHUSDT","as_of":rows[120]["close_time"],
                  "edition":"TC2","setup":"NO_SETUP","side":"NONE",
                  "entry_state":"NO_SETUP","exit_basis":"NONE",
                  "rule_ids":[],"trade_action":None,"execution_allowed":False,
                  "reason":"Waiting for complete confirmation."}
        report=evaluate_historical_thought("ETHUSDT",rows,
                 tick_size=0.01,bar_index=120,edition="TC2",
                 proposal=proposal,include_topdown=False)
        self.assertFalse(report["independent_practical_decision_teacher_supplied"])
        self.assertEqual(report["competing_market_hypotheses_rehearsed"],0)


if __name__=="__main__":
    unittest.main()
