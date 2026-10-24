"""Genuine previously resolved experience enters later *GGUF* chart context."""
from datetime import datetime,timedelta,timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest

from williams_ai.continual_trader import Outcome,PrivateExperience
from williams_ai.causal_paper_lab import frozen_markets
from williams_ai.reasoning_trial import evaluate_historical_thought
from williams_ai.chart_reader import analyze_chart
from williams_ai.local_model import LocalWilliamsModel


class InspectingBookModel:
    def __init__(self):
        self.experience=None
    def propose_chart_reasoning(self,chart,*,edition="TC2",**kwargs):
        self.experience=chart.get("past_experience")
        return {"symbol":chart["symbol"],"as_of":chart["as_of"],
                "edition":edition,"setup":"NO_SETUP","side":"NONE",
                "rule_ids":[],"entry_state":"NO_SETUP","exit_basis":"NONE",
                "wave_degree_verified":False,
                "exchange_fill_confirmed":False,"exchange_sl_confirmed":False,
                "reason":"No confirmed current trade.",
                "trade_action":None,"execution_allowed":False}


class LearningMustBeCausal(unittest.TestCase):
    def build_memory(self,db):
        start=datetime(2025,1,1,tzinfo=timezone.utc)
        history=[]
        for i in range(24):
            dt=start+timedelta(hours=i)
            history.append(Outcome(
                "ETHUSDT","H1","LONG",dt.isoformat(),
                (dt+timedelta(hours=2)).isoformat(),
                (1.,float(i%3),0.1,0.1,0.5,1.0),
                .01 if i%2 else -.02,i%2==0,
                "b"*64,"WM3","TC2.WM3"))
        future=datetime(2035,1,1,tzinfo=timezone.utc)
        history.append(Outcome(
            "ETHUSDT","H1","LONG",future.isoformat(),
            (future+timedelta(hours=1)).isoformat(),
            (1.,1.,0.,0.,0.,0.),4.0,False,"c"*64,
            "WM3","TC2.WM3"))
        db.append(history,knowledge_as_of=(future+timedelta(hours=1)).isoformat())

    def test_past_settled_feedback_influences_next_model_view_only(self):
        with TemporaryDirectory() as d:
            db=PrivateExperience(Path(d)/"lessons.db")
            self.build_memory(db)
            market=frozen_markets()["ETHUSDT"]
            model=InspectingBookModel()
            result=evaluate_historical_thought(
                "ETHUSDT",market["bars"],tick_size=market["tick_size"],
                bar_index=310,edition="TC2",model=model,
                experience_journal=Path(d)/"lessons.db")
            self.assertEqual(result["past_learning_feedback_used"],24)
            self.assertEqual(model.experience["past_examples_used"],24)
            group=model.experience["by_wise_man_and_direction"]["WM3:LONG"]
            self.assertEqual(group["count"],24)
            self.assertEqual(group["past_positive_fraction"],0.5)
            self.assertEqual(group["past_paper_profit_factor"],0.5)
            self.assertEqual(group["past_payoff_break_even_win_rate"],0.66667)
            self.assertEqual(group["distinct_utc_decision_dates_not_independent"],1)
            self.assertTrue(group["not_actual_portfolio_pnl"])
            self.assertFalse(result["execution_allowed"])
            self.assertIsNone(result["trade_action"])
            # Earlier 2024 view sees NO lessons from 2025 or 2035.
            past=db.past_feedback(symbol="ETHUSDT",as_of="2024-12-31T22:00:00Z")
            self.assertEqual(past["past_examples_used"],0)
            self.assertEqual(past["by_wise_man_and_direction"]["WM3:LONG"]["status"],
                             "INSUFFICIENT_RESOLVED_EXPERIENCE")

    def test_actual_gguf_prompt_receives_private_experience_as_nonbook_evidence(self):
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as temp:
            model_path=Path(temp)/"fake.gguf"
            model_path.write_bytes(b"synthetic mock only, not real GGUF")
            class CapturingRuntime:
                def __init__(self,**kwargs):
                    self.calls=[]
                def create_chat_completion(self,**kwargs):
                    self.calls.append(kwargs)
                    return {"choices":[{"message":{"content":
                        '{"symbol":"ETHUSDT","as_of":"2026-10-10T07:59:59.999000+00:00",' +
                        '"edition":"TC2","setup":"NO_SETUP","side":"NONE",' +
                        '"reason":"No proven setup.","trade_action":null,"execution_allowed":false}'}}]}
            m=LocalWilliamsModel(model_path,loader=CapturingRuntime)
            rows=frozen_markets()["ETHUSDT"]["bars"][:900]
            chart=analyze_chart("ETHUSDT","H1",rows,
                as_of=rows[-1]["close_time"],tick_size=.01)
            market_at=chart["as_of"]
            chart["past_experience"]={
                "symbol":"ETHUSDT","as_of":market_at,"execution_allowed":False,
                "past_examples_used":26,
                "by_wise_man_and_direction":{"WM3:LONG":{
                    "count":26,"status":"PAST_CAUSAL_PAPER_OUTCOMES_UNVERIFIED_FOR_TRADING",
                    "past_positive_fraction":.46,"past_mean_net_return":-.001}},
            }
            m.propose_chart_reasoning(chart,edition="TC2")
            user=json.loads(m._model.calls[-1]["messages"][1]["content"])
            context=user["chart"]["prior_resolved_private_TC2_paper_experience"]
            self.assertEqual(context["evidence_count"],26)
            self.assertEqual(context["by_setup"]["WM3:LONG"]["count"],26)
            self.assertIn("past_paper_profit_factor",
                          context["by_setup"]["WM3:LONG"])
            self.assertTrue(context["by_setup"]["WM3:LONG"]
                            ["past_returns_not_next_trade_expectancy"])
            self.assertTrue(context["not_book_confirmation"])
            self.assertTrue(context["no_execution"])

if __name__=="__main__":
    unittest.main()
