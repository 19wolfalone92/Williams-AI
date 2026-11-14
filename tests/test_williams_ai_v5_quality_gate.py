"""Quality is not equal to CI success; no synthetic exam permits trading."""
import copy
import unittest
from williams_ai.v5_postfit_exam import research_quality_gate, public_four_book_case_rows


def fixture(src=9,abstentions=5,hold=16):
    def per(src_ntd,src_tc2,hold_ntd,hold_tc2):
        return {"prior_16_case_with_source":{
            "correct":src,"abstentions":abstentions,
            "by_edition":{"TC1":{"correct":2},"NTD":{"correct":src_ntd},
                          "TC2":{"correct":src_tc2},"COURSE":{"correct":2}}},
            "new_rule_disjoint_28_case_exam":{
                "correct":hold,"by_edition":{"TC1":{"correct":2},"NTD":{"correct":hold_ntd},
                                             "TC2":{"correct":hold_tc2},"COURSE":{"correct":2}}}}
    return {
        "candidate_v5":per(3,2,5,4),
        "public_base_control":{
            "prior_16_case_with_source":{
                "correct":2,"abstentions":14,
                "by_edition":{"TC1":{"correct":1},"NTD":{"correct":0},
                              "TC2":{"correct":1},"COURSE":{"correct":1}}},
            "new_rule_disjoint_28_case_exam":{
                "correct":8,"by_edition":{"TC1":{"correct":1},"NTD":{"correct":1},
                                         "TC2":{"correct":3},"COURSE":{"correct":1}}}},
    }


class QualityGateAudit(unittest.TestCase):
    def test_bad_rag_does_not_pass_even_when_training_loss_improves(self):
        report=fixture(src=1,abstentions=15,hold=11)
        report["candidate_v5"]["prior_16_case_with_source"]["by_edition"]["NTD"]["correct"]=0
        report["candidate_v5"]["prior_16_case_with_source"]["by_edition"]["TC2"]["correct"]=0
        decision=research_quality_gate(report)
        self.assertFalse(decision["accepted_for_further_research"])
        self.assertIn("SOURCE_ASSISTED_BELOW_8_OF_16",decision["reasons"])
        self.assertIn("SOURCE_ASSISTED_EXCESSIVE_ABSTENTIONS",decision["reasons"])
        self.assertIn("SOURCE_ASSISTED_TC2_REGRESSION_VS_BASE",decision["reasons"])
        self.assertFalse(decision["execution_allowed"])
        self.assertFalse(decision["trading_qualification"])

    def test_separately_bounded_four_book_performance_only_allows_research(self):
        decision=research_quality_gate(fixture())
        self.assertTrue(decision["accepted_for_further_research"])
        self.assertFalse(decision["execution_allowed"])
        self.assertTrue(decision["not_a_live_trading_gate"])
        self.assertTrue(decision["checkpoint500_different_decoder_not_directly_comparable"])

    def test_ntd_and_tc2_new_test_regressions_do_not_pass(self):
        report=fixture()
        report["candidate_v5"]["new_rule_disjoint_28_case_exam"]["by_edition"]["TC2"]["correct"]=2
        report["candidate_v5"]["new_rule_disjoint_28_case_exam"]["by_edition"]["NTD"]["correct"]=0
        decision=research_quality_gate(report)
        self.assertIn("HELDOUT_TC2_REGRESSION_VS_BASE",decision["reasons"])
        self.assertIn("HELDOUT_NTD_REGRESSION_VS_BASE",decision["reasons"])
        self.assertFalse(decision["accepted_for_further_research"])

    def test_tampering_with_baseline_cannot_improve_self_report(self):
        report=fixture()
        outcome=research_quality_gate(report)
        bad=copy.deepcopy(report)
        bad["public_base_control"]["new_rule_disjoint_28_case_exam"]["correct"]=20
        self.assertFalse(research_quality_gate(bad)["accepted_for_further_research"])
        self.assertTrue(outcome["accepted_for_further_research"])


    def test_future_16_case_receipt_contains_redacted_verifiable_case_rows(self):
        rows=[]
        for ed in ("TC1","NTD","TC2","COURSE"):
            for n in range(4):
                rows.append({"id":f"{ed}-{n}","edition":ed,"stage":"ENTRY",
                    "source_rule_id":f"{ed}.TEST","chosen":"A","correct":True,
                    "abstained":False,"failure":None,
                    "retrieved_source_card_count":2,
                    "model_decoding_mode":"QWEN3_REASON_THEN_FINAL_JSON",
                    "initial_finish_reason":"stop",
                    "json_fallback_used":False,
                    "raw_completion":"PRIVATE_UNPUBLISHED_THOUGHT",
                    "question":"UNEXPORTED_QUESTION",
                    "expected":"A","gold":"SENSITIVE_EXAM_KEY"})
        sanitized=public_four_book_case_rows({"results":rows})
        self.assertEqual(len(sanitized),16)
        self.assertEqual({x["edition"] for x in sanitized},
                         {"TC1","NTD","TC2","COURSE"})
        for case in sanitized:
            self.assertNotIn("raw_completion",case)
            self.assertNotIn("question",case)
            self.assertNotIn("expected",case)
            self.assertNotIn("gold",case)

    def test_future_16_case_receipt_rejects_duplicate_or_malformed_grades(self):
        rows=[{"id":f"{ed}-{n}","edition":ed,"stage":"ENTRY",
               "source_rule_id":f"{ed}.TEST","chosen":"A",
               "correct":True,"abstained":False}
              for ed in ("TC1","NTD","TC2","COURSE") for n in range(4)]
        rows[1]["id"]=rows[0]["id"]
        with self.assertRaisesRegex(ValueError,"UNBALANCED_OR_DUPLICATE"):
            public_four_book_case_rows({"results":rows})
        rows[1]["id"]="TC1-1"
        rows[1]["chosen"]=""
        with self.assertRaisesRegex(ValueError,"INCONSISTENT_FOUR_BOOK"):
            public_four_book_case_rows({"results":rows})


if __name__=="__main__":
    unittest.main()
