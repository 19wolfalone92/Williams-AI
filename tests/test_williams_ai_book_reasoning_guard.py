"""Book-source reasoning gate: falsifiable decisions, not self-awarded mastery."""
from datetime import datetime,timedelta,timezone
from pathlib import Path
import json
import tempfile
import unittest

from williams_ai.algorithm_guard import audit_reasoning
from williams_ai.causal_paper_lab import frozen_markets
from williams_ai.chart_reader import analyze_chart
from williams_ai.local_model import LocalWilliamsModel
from williams_ai.reasoning_trial import evaluate_historical_thought


class GateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x=frozen_markets()["ETHUSDT"]
        cls.rows=cls.x["bars"]
        cls.cutoff_index=410
        cls.as_of=cls.rows[cls.cutoff_index]["close_time"]
        cls.observed=analyze_chart("ETHUSDT","H1",
            cls.rows[:cls.cutoff_index+1],
            as_of=cls.as_of,tick_size=cls.x["tick_size"])

    def prop(self,**modifications):
        valid={
            "symbol":"ETHUSDT","as_of":self.as_of,
            "edition":"TC2","setup":"NO_SETUP","side":"NONE",
            "entry_state":"NO_SETUP","exit_basis":"NONE",
            "rule_ids":[],"trade_action":None,"execution_allowed":False,
            "exchange_fill_confirmed":False,
            "exchange_sl_confirmed":False,
            "wave_degree_verified":False,"reason":"Нет подтверждённого входа",
        }
        valid.update(modifications)
        return valid

    def test_no_setup_is_provisional_not_mastery(self):
        x=audit_reasoning(self.observed,self.prop())
        self.assertEqual(x["verdict"],"PROVISIONAL_REASONING_ONLY")
        self.assertIsNone(x["actual_book_mastery_percent"])
        self.assertFalse(x["execution_allowed"])
        self.assertIsNone(x["trade_action"])

    def test_future_past_candle_mismatch_forbidden(self):
        dt=datetime.fromisoformat(self.as_of)
        for offset in (-1,1):
            wrong=(dt+timedelta(hours=offset)).isoformat()
            x=audit_reasoning(self.observed,self.prop(as_of=wrong))
            self.assertIn("DECISION_CHART_CUTOFF_MISMATCH",x["hard_failures"])

    def test_exchange_execution_and_stop_claims_never_inferred(self):
        x=audit_reasoning(self.observed,self.prop(
            entry_state="EXCHANGE_FILLED",exchange_fill_confirmed=True,
            exchange_sl_confirmed=True,exit_basis="SL_CONFIRMED",
            trade_action={"side":"BUY"}))
        self.assertIn("NO_EXCHANGE_FILL_LEDGER",x["hard_failures"])
        self.assertIn("CANNOT_CONFIRM_EXCHANGE_FILL_FROM_OHLC",x["hard_failures"])
        self.assertIn("CANNOT_CONFIRM_EXCHANGE_SL_FROM_OHLC",x["hard_failures"])
        self.assertIn("NO_EXCHANGE_CLOSE_LEDGER_WAS_PROVIDED",x["hard_failures"])
        self.assertIn("EXECUTION_REQUEST_FORBIDDEN",x["hard_failures"])
        self.assertIsNone(x["trade_action"])

    def test_substantive_wise_man_must_cite_the_right_book_algorithm(self):
        first=audit_reasoning(self.observed,self.prop(setup="WM3",side="LONG",rule_ids=[]))
        self.assertIn("MISSING_SOURCE_RULE_FOR_SUBSTANTIVE_CLAIM",first["hard_failures"])
        second=audit_reasoning(self.observed,self.prop(
            setup="WM3",side="LONG",rule_ids=["TC2.WM1.ANG"]))
        self.assertIn("CITED_RULE_DOES_NOT_SUPPORT_THIS_SETUP",second["hard_failures"])

    def test_four_book_registry_distinguishes_read_source_from_executable_method(self):
        from williams_ai.book_rules import registry,coverage
        package=registry()
        self.assertEqual(len(package["books"]),4)
        self.assertEqual(len(package["rule_cards"]),36)
        checked=sum(x["status"]=="REVIEWED_PAGES" for x in package["rule_cards"])
        self.assertEqual(checked,36)
        by_id={x["id"]:x for x in package["rule_cards"]}
        for source in ("TC2.WM2","TC2.WM3","TC2.CAMPAIGN","NTD.EXITS"):
            self.assertEqual(by_id[source]["status"],"REVIEWED_PAGES")
        self.assertIn("наименее",by_id["NTD.EXITS"]["summary"])
        self.assertIn("обратная пирамида",by_id["TC2.CAMPAIGN"]["summary"])
        self.assertIsNone(coverage()["TC2"]["book_coverage_percent"])

    def test_opposite_signal_not_automatic_close(self):
        x=audit_reasoning(self.observed,self.prop(exit_basis="OPPOSITE_SIGNAL_ONLY"))
        self.assertIn("UNEXECUTED_OPPOSITE_WISE_MAN_NOT_A_TRADE_EXIT",x["hard_failures"])

    def test_confirmed_pending_wm3_does_not_require_any_past_price_touch(self):
        pending={**self.observed,"candidates":[{
            "type":"WM3_FRACTAL","side":"SHORT",
            "state":"AWAITING_PRICE_TRIGGER","entry_trigger":3000.0,
        }]}
        draft=self.prop(setup="WM3",side="SHORT",
                        entry_state="PENDING_TRIGGER",rule_ids=["TC2.WM3"])
        result=audit_reasoning(pending,draft)
        self.assertEqual(result["hard_failures"],[])
        self.assertIn("WM3_AWAITING_BREAKOUT_NOT_AN_EXCHANGE_FILL",
                      result["uncertainties"])
        self.assertIsNone(result["trade_action"])

    def test_wm3_touch_at_old_bar_not_new_fake_entry(self):
        old=self.rows[350]["close_time"]
        fake={**self.observed,"candidates":[{
            "type":"WM3_FRACTAL","side":"LONG",
            "state":"CLOSED_BAR_TOUCH_PASSES_TEETH_NOT_EXCHANGE_FILL",
            "price_touch_observed_at":old,
        }]}
        claim=self.prop(setup="WM3",side="LONG",
                        entry_state="CLOSED_BAR_TOUCH_ONLY",
                        rule_ids=["TC2.WM3"])
        out=audit_reasoning(fake,claim)
        self.assertIn("WM3_NO_PRICE_TOUCH_ON_THIS_CLOSED_BAR",out["hard_failures"])
        current=fake["candidates"][0]
        current["price_touch_observed_at"]=self.as_of
        ok=audit_reasoning(fake,claim)
        self.assertNotIn("WM3_NO_PRICE_TOUCH_ON_THIS_CLOSED_BAR",ok["hard_failures"])

    def test_ntd_observation_not_entry_before_first_external_fractal(self):
        fake={**self.observed,"edition_evidence":{"NTD":{
            "entry_admitted":False,"signals":[{
                "name":"AO_ZERO_CROSS","side":"LONG"
            }]}}}
        result=audit_reasoning(fake,self.prop(
            edition="NTD",setup="AO_ZERO_CROSS",side="LONG",
            entry_state="PENDING_TRIGGER",rule_ids=["NTD.AO"]))
        self.assertIn("NTD_ENTRY_CLAIM_WITHOUT_FIRST_OUTSIDE_FRACTAL",
                      result["hard_failures"])
        observed_only=audit_reasoning(fake,self.prop(
            edition="NTD",setup="AO_ZERO_CROSS",side="LONG",
            entry_state="CLOSED_CANDIDATE",rule_ids=["NTD.AO"]))
        self.assertEqual(observed_only["verdict"],"PROVISIONAL_REASONING_ONLY")

    def test_wm1_heuristic_failure_cannot_claim_price_execution(self):
        fake={**self.observed,"candidates":[{
            "type":"WM1","side":"LONG","angle_heuristic":False
        }]}
        x=audit_reasoning(fake,self.prop(
            setup="WM1",side="LONG",entry_state="CLOSED_BAR_TOUCH_ONLY",
            rule_ids=["TC2.WM1.ANG"]))
        self.assertIn("WM1_ENTRY_CLAIM_WITHOUT_EVEN_ANGLE_HEURISTIC",
                      x["hard_failures"])

    def test_source_alias_and_hallucinated_rule_rejected(self):
        x=audit_reasoning(self.observed,self.prop(
            edition="NTD",setup="WM1",side="LONG",
            rule_ids=["TC2.WM1.ANG","BOGUS.PDF.99"]))
        self.assertIn("TC2_WISE_MAN_IN_OTHER_EDITION",x["hard_failures"])
        self.assertIn("CROSS_EDITION_RULE_MIXING",x["hard_failures"])
        self.assertIn("HALLUCINATED_SOURCE_RULE",x["hard_failures"])

    def test_wm1_visual_uncertainty_stays_visible(self):
        obs={**self.observed,"candidates":[
            {"type":"WM1","side":"LONG","angle_heuristic":False,
             "state":"REVERSAL_BAR_VISIBLE_ANGULATION_REVIEW_REQUIRED"}]}
        x=audit_reasoning(obs,self.prop(
            setup="WM1",side="LONG",entry_state="PENDING_TRIGGER",
            rule_ids=["TC2.WM1.ANG"]))
        self.assertEqual(x["verdict"],"PROVISIONAL_REASONING_ONLY")
        self.assertIn("WM1_ANGLULATION_HUMAN_IMAGE_REVIEW_REQUIRED",x["uncertainties"])
        self.assertIn("WM1_NUMERICAL_ANGULATION_NOT_EVEN_POSITIVE",x["uncertainties"])

    def test_wm3_requires_teeth_at_breakout_not_formation(self):
        proposal=self.prop(setup="WM3",side="LONG",rule_ids=["TC2.WM3"])
        obs={**self.observed,"candidates":[{
            "type":"WM3_FRACTAL","side":"LONG",
            "state":"AWAITING_PRICE_TRIGGER",
            "teeth_filter_at_confirmation":True,
        }]}
        x=audit_reasoning(obs,proposal)
        self.assertIn("WM3_NO_VERIFIED_TEETH_PASS_AT_PRICE_TOUCH",x["hard_failures"])
        obs["candidates"][0]["state"]="CLOSED_BAR_TOUCH_PASSES_TEETH_NOT_EXCHANGE_FILL"
        x=audit_reasoning(obs,proposal)
        self.assertEqual(x["verdict"],"PROVISIONAL_REASONING_ONLY")
        self.assertIn("WM3_CLOSED_BAR_TOUCH_NOT_AN_EXCHANGE_FILL",x["uncertainties"])
        self.assertIsNone(x["trade_action"])

    def test_ntd_first_fractal_gate_not_same_as_tc2_wise_man(self):
        obs={**self.observed,"edition_evidence":{"NTD":{
            "entry_admitted":False,"signals":[{
                "name":"AO_ZERO_CROSS","side":"LONG",
            }]}}}
        x=audit_reasoning(obs,self.prop(edition="NTD",
            setup="AO_ZERO_CROSS",side="LONG",rule_ids=["NTD.AO"]))
        self.assertEqual(x["verdict"],"PROVISIONAL_REASONING_ONLY")
        self.assertIn("NTD_FIRST_OUTSIDE_FRACTAL_BREAKOUT_NOT_ESTABLISHED",
                      x["uncertainties"])
        y=audit_reasoning(obs,self.prop(edition="TC2",setup="AO_ZERO_CROSS",
                                       side="LONG",rule_ids=[]))
        self.assertIn("NTD_PATTERN_IS_NOT_TC2_WISE_MAN",y["hard_failures"])

    def test_unknown_wave_degree_cannot_claim_truth(self):
        x=audit_reasoning(self.observed,self.prop(
            wave_degree_verified=True,future_candle_used=True))
        self.assertIn("WAVE_DEGREE_UNVERIFIABLE_FROM_CURRENT_EVIDENCE",
                      x["hard_failures"])
        self.assertIn("FUTURE_CANDLE_CLAIM",x["hard_failures"])

    def test_real_closed_prefix_trial_no_future(self):
        res=evaluate_historical_thought(
            "ETHUSDT",self.rows,tick_size=self.x["tick_size"],
            bar_index=self.cutoff_index,edition="TC2",proposal=self.prop())
        self.assertEqual(res["last_visible_at"],self.as_of)
        self.assertFalse(res["model_saw_future_bars"])
        self.assertEqual(res["audit"]["hard_failures"],[])
        self.assertIsNone(res["book_mastery_percent"])
        self.assertFalse(res["execution_allowed"])
        later=evaluate_historical_thought("ETHUSDT",self.rows+[
            # Fault-injected fake future candle: the cutoff MUST be the same.
            {**self.rows[-1],"close_time":"2030-01-01T01:59:59.999Z"}],
            tick_size=self.x["tick_size"],bar_index=self.cutoff_index,
            proposal=self.prop())
        self.assertEqual(res,later)

    def test_local_gguf_mock_cannot_self_grant_permission(self):
        original=self.prop()
        original.update({"setup":"WM1","side":"LONG",
                         "exchange_fill_confirmed":True,
                         "trade_action":{"side":"BUY"}})
        # A fake engine is NOT a test of actual GGUF weights.
        class FakeEngine:
            def __init__(self,**kwargs):self.kwargs=kwargs;self.last=None
            def create_chat_completion(self,**kwargs):
                self.last=kwargs
                return {"choices":[{"message":{"content":json.dumps(original)}}]}
        with tempfile.TemporaryDirectory() as temp:
            model_path=Path(temp)/"private.gguf"
            model_path.write_bytes(b"GGUF-dummy")
            model=LocalWilliamsModel(model_path,loader=FakeEngine)
            res=evaluate_historical_thought(
                "ETHUSDT",self.rows,tick_size=self.x["tick_size"],
                bar_index=self.cutoff_index,edition="TC2",model=model)
            self.assertEqual(res["audit"]["verdict"],"UNSUPPORTED_CLAIM")
            self.assertIn("EXECUTION_REQUEST_FORBIDDEN",res["audit"]["hard_failures"])
            self.assertIsNone(res["trade_action"])
            prompt=model._model.last["messages"][1]["content"]
            self.assertIn("TC2.WM1.ANG",prompt)
            self.assertNotIn("secret_key",prompt)
            self.assertNotIn(self.rows[-1]["close_time"],prompt)


    def test_batch_exam_never_calls_no_setup_100pct_book_mastery(self):
        from williams_ai.reasoning_batch import examine_many
        class Conservative:
            def __init__(self):self.called=0
            def propose_chart_reasoning(self,chart,*,edition):
                self.called+=1
                return {
                    "symbol":chart["symbol"],"as_of":chart["as_of"],
                    "edition":edition,"setup":"NO_SETUP","side":"NONE",
                    "trade_action":None,"execution_allowed":False,
                    "reason":"No order or definitive wave proof",
                }
        model=Conservative()
        two={
            k:frozen_markets()[k] for k in ("BTCUSDT","ETHUSDT")
        }
        result=examine_many(two,model,editions=("TC2","NTD"),checkpoints=(299,599))
        self.assertEqual(result["tested_cases"],8)
        self.assertEqual(model.called,8)
        self.assertEqual(result["source_consistency_percent"],100)
        self.assertEqual(result["abstained_or_unknown_cases"],8)
        self.assertEqual(result["substantive_claim_cases"],0)
        self.assertIsNone(result["substantive_source_consistency_percent"])
        self.assertIsNone(result["full_book_understanding_percent"])
        self.assertFalse(result["execution_allowed"])

    def test_batch_exam_exposes_fabricated_execution_claims(self):
        from williams_ai.reasoning_batch import examine_many
        class Fabricator:
            def propose_chart_reasoning(self,chart,*,edition):
                return {
                    "symbol":chart["symbol"],"as_of":chart["as_of"],
                    "edition":edition,"setup":"WM1","side":"LONG",
                    "trade_action":{"buy":True},"exchange_fill_confirmed":True,
                    "wave_degree_verified":True,
                }
        result=examine_many({"ETHUSDT":self.x},Fabricator(),
                            editions=("TC2",),checkpoints=(299,))
        self.assertEqual(result["tested_cases"],1)
        self.assertEqual(result["unsupported_claim_cases"],1)
        self.assertIn("CANNOT_CONFIRM_EXCHANGE_FILL_FROM_OHLC",result["failure_counts"])
        self.assertIn("WAVE_DEGREE_UNVERIFIABLE_FROM_CURRENT_EVIDENCE",result["failure_counts"])
        self.assertIsNone(result["trade_action"])

if __name__=="__main__":
    unittest.main()
