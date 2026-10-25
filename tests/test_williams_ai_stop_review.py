"""Numerical risk boundaries/inference wiring; no claim of real model skill."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from williams_ai.stop_review import (
    audit_stop_review, diagnostic_cases, diagnose_model, normalize_context,
    review_with_model, INPUT_KEYS,
)
from williams_ai.local_model import LocalWilliamsModel
from williams_ai.causal_paper_lab import frozen_markets
from williams_ai.reasoning_trial import evaluate_historical_thought


EXPECTED = {
    "LONG_WIDER":("LOWER","WIDER",True,"KEEP_PROTECTION"),
    "SHORT_WIDER":("HIGHER","WIDER",True,"KEEP_PROTECTION"),
    "LONG_TIGHTER":("HIGHER","TIGHTER",True,"REVIEW_TIGHTENING"),
    "SHORT_TIGHTER":("LOWER","TIGHTER",True,"REVIEW_TIGHTENING"),
    "LONG_EQUAL":("EQUAL","UNCHANGED",True,"KEEP_PROTECTION"),
    "SHORT_EQUAL":("EQUAL","UNCHANGED",True,"KEEP_PROTECTION"),
    "LONG_WRONG_SIDE":("HIGHER","TIGHTER",False,"KEEP_PROTECTION"),
    "SHORT_WRONG_SIDE":("LOWER","TIGHTER",False,"KEEP_PROTECTION"),
    "LONG_EXISTING_CROSSED":("HIGHER","TIGHTER",False,"HALT_FOR_RECONCILIATION"),
    "SHORT_EXISTING_CROSSED":("LOWER","TIGHTER",False,"HALT_FOR_RECONCILIATION"),
}


def response(values):
    return dict(zip(("level_relation","risk_direction","candidate_protects_side","assessment"),values),
                reason="Synthetic protocol assessment, no order authority.")


class StopReviewTests(unittest.TestCase):
    def test_hand_specified_long_short_boundaries(self):
        for case in diagnostic_cases():
            with self.subTest(case=case["id"]):
                report=audit_stop_review(case["context"],response(EXPECTED[case["id"]]))
                self.assertEqual(report["hard_failures"],[])
                self.assertFalse(report["execution_allowed"])
                self.assertFalse(report["broker_authentication_performed"])

    def test_widening_wrong_side_and_already_crossed_stop_cannot_pass(self):
        cases={x["id"]:x["context"] for x in diagnostic_cases()}
        for name,code in (("LONG_WIDER","MODEL_ATTEMPTED_TO_WIDEN_STOP"),
                          ("SHORT_WIDER","MODEL_ATTEMPTED_TO_WIDEN_STOP"),
                          ("LONG_WRONG_SIDE","MODEL_CANDIDATE_ON_WRONG_PRICE_SIDE"),
                          ("SHORT_EXISTING_CROSSED","MODEL_IGNORED_POSSIBLY_TRIGGERED_STOP")):
            bad=response(EXPECTED[name]);bad["assessment"]="REVIEW_TIGHTENING"
            self.assertIn(code,audit_stop_review(cases[name],bad)["hard_failures"])

    def test_equal_or_wrong_side_abstention_is_not_passed_as_tightening(self):
        case=diagnostic_cases()[2]
        bad=response(EXPECTED[case["id"]]);bad["assessment"]="KEEP_PROTECTION"
        self.assertIn("STOP_REVIEW_MISMATCH_ASSESSMENT",audit_stop_review(case["context"],bad)["hard_failures"])
        bad["assessment"]="UNKNOWN"
        self.assertTrue(audit_stop_review(case["context"],bad)["hard_failures"])

    def test_decimal_comparison_does_not_collapse_close_levels_to_float(self):
        context=deepcopy(diagnostic_cases()[2]["context"])
        context.update(current_stop="100.000000000000000000000001",
                       candidate_stop="100.000000000000000000000002")
        self.assertEqual(audit_stop_review(context,response(EXPECTED["LONG_TIGHTER"]))["hard_failures"],[])

    def test_nonfinite_unbounded_or_non_decimal_inputs_fail_before_inference(self):
        for value in ("NaN","Infinity","0","-1","1e999999","1e-999999",True,100.0):
            context=deepcopy(diagnostic_cases()[0]["context"])
            context["current_stop"]=value
            with self.subTest(value=value),self.assertRaises(ValueError):
                normalize_context(context)
        for patch in ({"source":"SIGNED_BINANCE_VERIFIED"},{"execution_allowed":True},
                      {"trade_action":"CANCEL_STOP"},{"as_of":"not-a-time"},{"side":[]}):
            with self.assertRaises(ValueError):
                normalize_context({**diagnostic_cases()[0]["context"],**patch})

    def test_order_fields_and_arbitrary_output_do_not_get_exported_or_repaired(self):
        class Bad:
            def review_protective_stop(self,context):
                return {**response(EXPECTED["LONG_WIDER"]),"api_key":"PRIVATE_TEST_VALUE"}
        report=review_with_model(Bad(),diagnostic_cases()[0]["context"])
        self.assertIsNone(report["model_response"])
        self.assertNotIn("PRIVATE_TEST_VALUE",json.dumps(report))
        self.assertEqual(report["audit"]["verdict"],"UNSUPPORTED_STOP_REASONING")

    def test_runtime_failure_and_missing_model_interface_are_explicit(self):
        class Broken:
            def review_protective_stop(self,context):
                raise RuntimeError("PRIVATE_BOOK_OR_PATH_TEST_VALUE")
        for model in (Broken(),object()):
            report=review_with_model(model,diagnostic_cases()[0]["context"])
            self.assertIsNone(report["model_response"])
            self.assertEqual(report["audit"]["verdict"],"MODEL_STOP_REVIEW_ERROR")
            self.assertNotIn("PRIVATE_BOOK_OR_PATH_TEST_VALUE",json.dumps(report))

    def test_model_metadata_is_allowlisted_without_private_extension_fields(self):
        class Student:
            latest_stop_review_call={"mode":"SCHEMA_CONSTRAINED_STOP_REVIEW",
                "finish_reason":"stop","api_key":"SECRET_METADATA_MARKER",
                "usage":{"prompt_tokens":7,"private_path":"SECRET_METADATA_MARKER"}}
            def review_protective_stop(self,context):return response(EXPECTED["LONG_WIDER"])
        result=review_with_model(Student(),diagnostic_cases()[0]["context"])
        self.assertNotIn("SECRET_METADATA_MARKER",json.dumps(result))
        self.assertEqual(result["model_call_metadata"]["usage"],{"prompt_tokens":7})

    def test_model_receives_raw_context_not_reference_answer(self):
        class Runtime:
            def __init__(self,**kwargs):self.calls=[]
            def create_chat_completion(self,**kwargs):
                self.calls.append(kwargs)
                return {"choices":[{"message":{"content":json.dumps(response(EXPECTED["LONG_WIDER"]))}}]}
        with TemporaryDirectory() as temp:
            weights=Path(temp)/"mock.gguf";weights.write_bytes(b"FAKE_RUNTIME_INJECTED")
            model=LocalWilliamsModel(weights,loader=Runtime)
            model.review_protective_stop(diagnostic_cases()[0]["context"])
            payload=json.loads(model._model.calls[0]["messages"][1]["content"])
            self.assertEqual(set(payload["paper_observations"]),INPUT_KEYS)
            self.assertNotIn("assessment",payload)
            self.assertNotIn("reference",payload)
            self.assertIn("independent_engineering_risk_policy",payload)
            self.assertEqual(payload["book_source"]["kind"],"PUBLIC_REVIEWED_PARAPHRASE_NOT_ORIGINAL_PDF")
            schema=model._model.calls[0]["response_format"]["schema"]
            self.assertEqual(set(schema["properties"]),
                {"level_relation","risk_direction","candidate_protects_side","assessment","reason"})
            model.review_protective_stop(diagnostic_cases()[2]["context"])
            self.assertEqual(schema,model._model.calls[1]["response_format"]["schema"])
            self.assertIn("REVIEW_TIGHTENING",schema["properties"]["assessment"]["enum"])
            self.assertIn("KEEP_PROTECTION",schema["properties"]["assessment"]["enum"])

    def test_diagnostic_does_not_call_safe_constant_a_skilled_trader(self):
        class Constant:
            def review_protective_stop(self,context):
                return response(EXPECTED["LONG_WIDER"])
        report=diagnose_model(Constant())
        self.assertLess(report["consistent"],report["cases"])
        self.assertTrue(report["synthetic_arithmetic_not_market_charts"])
        self.assertTrue(report["different_protocol_not_comparable_to_previous_mcq_score"])
        self.assertFalse(report["model_weights_updated"])


class CampaignStopIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.market=frozen_markets()["ETHUSDT"]
        cls.cutoff=cls.market["bars"][299]["close_time"]

    def journal(self,protected=True):
        events=[{"at":self.market["bars"][250]["close_time"],
            "type":"OPEN_FILL","side":"LONG","price":1000,"exchange_confirmed":True}]
        if protected:
            events.append({"at":events[0]["at"],"type":"STOP_ACK",
                "stop_price":100,"exchange_confirmed":True})
        return {"symbol":"ETHUSDT","edition":"TC2","as_of":self.cutoff,
                "events":events,"api_key":"NEVER_FORWARD_JOURNAL_TEST_SECRET"}

    def run_case(self,model,journal):
        return evaluate_historical_thought("ETHUSDT",self.market["bars"],
            bar_index=299,tick_size=self.market["tick_size"],model=model,
            campaign_journal=journal,include_topdown=False)

    def student(self):
        class Student:
            def __init__(self):self.chart=None;self.stop_calls=[]
            def propose_chart_reasoning(self,chart,**kwargs):
                self.chart=chart
                return {"symbol":chart["symbol"],"as_of":chart["as_of"],"edition":"TC2",
                    "setup":"NO_SETUP","side":"NONE","entry_state":"NO_SETUP",
                    "exit_basis":"NONE","rule_ids":[],"execution_allowed":False,"trade_action":None}
            def review_protective_stop(self,context):
                self.stop_calls.append(context)
                return response(EXPECTED["LONG_TIGHTER"])
        return Student()

    def test_real_closed_prefix_and_journal_levels_reach_management_review(self):
        student=self.student();report=self.run_case(student,self.journal())
        self.assertEqual(len(student.stop_calls),1)
        context=student.stop_calls[0]
        self.assertEqual(context["as_of"],self.cutoff)
        self.assertEqual(context["current_stop"],"100.0")
        self.assertFalse(context["execution_allowed"])
        self.assertTrue(report["research_reasoning_consistent"])
        self.assertNotIn("NEVER_FORWARD_JOURNAL_TEST_SECRET",json.dumps(student.chart))
        self.assertNotIn("NEVER_FORWARD_JOURNAL_TEST_SECRET",json.dumps(report))

    def test_missing_protection_skips_neural_stop_change_and_blocks_acceptance(self):
        student=self.student();report=self.run_case(student,self.journal(False))
        self.assertEqual(student.stop_calls,[])
        self.assertFalse(report["research_reasoning_consistent"])
        self.assertEqual(report["campaign_stop_review"]["audit"]["verdict"],"HALT_FOR_RECONCILIATION")

    def test_future_journal_event_rejected_before_any_inference(self):
        journal=self.journal();journal["events"][-1]["at"]=self.market["bars"][300]["close_time"]
        student=self.student()
        with self.assertRaisesRegex(ValueError,"NO_LOOKAHEAD"):
            self.run_case(student,journal)
        self.assertIsNone(student.chart)

    def test_actual_runtime_payload_keeps_stop_numbers_and_excludes_teacher_answer(self):
        class Runtime:
            def __init__(self,**kwargs):self.calls=[]
            def create_chat_completion(self,**kwargs):
                self.calls.append(kwargs)
                payload=json.loads(kwargs["messages"][1]["content"])
                if "paper_observations" in payload:
                    result=response(EXPECTED["LONG_TIGHTER"])
                else:
                    chart=payload["chart"]
                    result={"symbol":chart["symbol"],"as_of":chart["as_of"],"edition":"TC2",
                        "setup":"NO_SETUP","side":"NONE","entry_state":"NO_SETUP",
                        "exit_basis":"NONE","rule_ids":[],"execution_allowed":False,"trade_action":None}
                return {"choices":[{"message":{"content":json.dumps(result)}}]}
        with TemporaryDirectory() as temp:
            weights=Path(temp)/"mock.gguf";weights.write_bytes(b"FAKE_RUNTIME_INJECTED")
            model=LocalWilliamsModel(weights,loader=Runtime)
            result=self.run_case(model,self.journal())
            self.assertTrue(result["research_reasoning_consistent"])
            self.assertEqual(len(model._model.calls),2)
            payload=json.loads(model._model.calls[0]["messages"][1]["content"])
            context=payload["chart"]["claimed_campaign_stop_observations"]
            self.assertEqual(context["current_stop"],"100.0")
            self.assertEqual(context["as_of"],self.cutoff)
            self.assertNotIn("assessment",context)
            self.assertNotIn("REVIEW_MONOTONIC_TRAILING_STOP",json.dumps(payload))
            self.assertNotIn("NEVER_FORWARD_JOURNAL_TEST_SECRET",json.dumps(model._model.calls))

    def test_future_management_context_rejected_by_actual_prompt_builder(self):
        from williams_ai.chart_reader import analyze_chart
        from williams_ai.tc2_trade_lifecycle_coach import teach_tc2_trade_lifecycle
        rows=self.market["bars"][:300]
        chart=analyze_chart("ETHUSDT","H1",rows,as_of=self.cutoff,tick_size=self.market["tick_size"])
        lesson=teach_tc2_trade_lifecycle(chart,rows,tick_size=self.market["tick_size"],journal=self.journal())
        lesson["management"]["stop_review_context"]["as_of"]=self.market["bars"][300]["close_time"]
        chart["tc2_trade_lifecycle_lesson"]=lesson
        class NeverCalled:
            def __init__(self,**kwargs):pass
            def create_chat_completion(self,**kwargs):raise AssertionError("Future context reached model")
        with TemporaryDirectory() as temp:
            weights=Path(temp)/"mock.gguf";weights.write_bytes(b"FAKE_RUNTIME_INJECTED")
            model=LocalWilliamsModel(weights,loader=NeverCalled)
            with self.assertRaisesRegex(ValueError,"STOP_REVIEW_CONTEXT_CHART_MISMATCH"):
                model.propose_chart_reasoning(chart)


if __name__=="__main__":
    unittest.main()
