"""Offline regression tests: no GGUF, GPU, credentials, private books or broker."""
import unittest
from williams_ai.research_source_router import source_route
from williams_ai.research_quality_audit import repeated_answers,resolved_forecasts


class NavigationTests(unittest.TestCase):
    def fixture(self):
        return {"books":[{"id":x,"pages":100} for x in ("TC1","NTD","TC2","COURSE")],
          "rule_cards":[
            {"id":"TC2.WM1.ANG","source":"TC2","pdf_pages":[5],"status":"REVIEWED_PAGES"},
            {"id":"TC2.WM2","source":"TC2","pdf_pages":[15],"status":"REVIEWED_PAGES"},
            {"id":"NTD.AO","source":"NTD","pdf_pages":[3],"status":"PRIOR_AUDIT_NOT_YET_RECHECKED"}]}

    def test_topics_route_to_distinct_confirmed_pages(self):
        r=self.fixture()
        w1=source_route("первый мудрец ангуляция",registry_data=r)
        w2=source_route("второй мудрец super ao",registry_data=r)
        self.assertIn("TC2.WM1.ANG",[v["rule_id"] for v in w1["references"]])
        self.assertNotIn("TC2.WM1.ANG",[v["rule_id"] for v in w2["references"]])
        self.assertIn("TC2.WM2",[v["rule_id"] for v in w2["references"]])
        self.assertNotIn("NTD.AO",[v["rule_id"] for v in w2["references"]])
        self.assertFalse(w1["execution_allowed"])

    def test_unreviewed_reference_never_exposed(self):
        response=source_route("осциллятор ao",registry_data=self.fixture())
        self.assertEqual(response["references"],[])

    def test_malformed_registry_or_question_rejected(self):
        with self.assertRaises(ValueError): source_route("")
        f=self.fixture();f["rule_cards"][0]["pdf_pages"]=[101]
        with self.assertRaises(ValueError): source_route("wm1",registry_data=f)


class QualityTests(unittest.TestCase):
    def test_repeated_answer_detected_without_running_model(self):
        rows=[{"case_id":str(i),"topic":t,"answer":"  Нет сигнала  "} for i,t
              in enumerate(("wm1","waves","risk","psychology"))]
        report=repeated_answers(rows)
        self.assertTrue(report["suspect_repetition"])
        self.assertFalse(report["semantic_correctness_proven"])
        rows[0]["answer"]="другое"
        self.assertFalse(repeated_answers(rows)["suspect_repetition"])

    def test_duplicate_response_rejected(self):
        q={"case_id":"A","topic":"WM1","answer":"x"}
        with self.assertRaises(ValueError): repeated_answers([q,q])

    def test_perfect_and_wrong_calibration(self):
        x={"case_id":"1","series":"ETHUSDT:1h","as_of":"2026-10-20T10:00:00Z",
           "resolved_at":"2026-10-21T10:00:00Z","outcome":"UP",
           "probabilities":{"UP":1.0,"DOWN":0.0,"RANGE":0.0}}
        good=resolved_forecasts([x])
        self.assertEqual(good["multiclass_brier"],0.0)
        self.assertEqual(good["log_loss"],0.0)
        self.assertFalse(good["forecast_precommit_externally_verified"])
        bad={**x,"probabilities":{"UP":0.0,"DOWN":1.0,"RANGE":0.0}}
        self.assertGreater(resolved_forecasts([bad])["multiclass_brier"],0)

    def test_future_time_and_nonfinite_guard(self):
        x={"case_id":"1","series":"ETHUSDT:1h","as_of":"2026-10-20T10:00:00Z",
           "resolved_at":"2026-10-20T09:00:00Z","outcome":"UP",
           "probabilities":{"UP":0.7,"DOWN":0.2,"RANGE":0.1}}
        with self.assertRaises(ValueError): resolved_forecasts([x])
        x["resolved_at"]="2026-10-21T10:00:00Z"
        x["probabilities"]["UP"]=float("nan")
        with self.assertRaises(ValueError): resolved_forecasts([x])


if __name__=="__main__": unittest.main()
