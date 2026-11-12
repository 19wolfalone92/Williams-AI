"""Quality is not equal to CI success; no synthetic exam permits trading."""
import copy
import unittest
from williams_ai.v5_postfit_exam import research_quality_gate


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


if __name__=="__main__":
    unittest.main()
