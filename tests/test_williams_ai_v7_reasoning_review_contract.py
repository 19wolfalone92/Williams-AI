import copy,json,unittest
from pathlib import Path
from williams_ai.v7_reasoning_review_contract import validate
DATA=Path(__file__).resolve().parents[1]/"knowledge"/"WILLIAMS_V7_REASONING_REVIEW_CANDIDATES_20261117.json"

class ReasoningReviewContractTests(unittest.TestCase):
    def setUp(self):
        self.d=json.loads(DATA.read_text(encoding="utf-8"))
    def fail(self,mut,code):
        d=copy.deepcopy(self.d)
        mut(d)
        with self.assertRaisesRegex(ValueError,code):
            validate(d)
    def test_four_books_five_families_and_no_gold(self):
        v=validate(self.d)
        self.assertEqual(v["pair_count"],15)
        self.assertEqual(v["unlabelled_probe_variants"],30)
        self.assertEqual(v["verified_human_chart_labels"],0)
        self.assertIsNone(v["book_accuracy"])
        self.assertFalse(v["approved_for_sft"])
    def test_source_sha_is_pinned(self):
        self.fail(lambda d:d["pairs"][0].update(source_pdf_sha256="0"*64),"SOURCE_SHA")
    def test_cross_edition_rule_is_rejected(self):
        self.fail(lambda d:d["pairs"][0].update(rule_ids=["NTD.AO"]),"CROSS_EDITION")
    def test_pages_are_original_and_valid(self):
        self.fail(lambda d:d["pairs"][0].update(source_pages=[0]),"SOURCE_PAGES")
        self.fail(lambda d:d["pairs"][0].update(source_pages=[121,121]),"SOURCE_PAGES")
    def test_duplicate_pair_is_rejected(self):
        self.fail(lambda d:d["pairs"][1].update(pair_id=d["pairs"][0]["pair_id"]),"PAIR_ID")
    def test_gold_fields_rejected_even_when_nested(self):
        self.fail(lambda d:d["pairs"][0]["variant_a"].update(expected_decision="BUY"),
                  "VARIANT_FIELDS")
        self.fail(lambda d:d["pairs"][0]["variant_a"].update(answer="BUY"),
                  "GOLD_OR_PSEUDO_LABEL")
        self.fail(lambda d:d["pairs"][0].update(model_output="BUY"),"GOLD_OR_PSEUDO_LABEL")
    def test_training_cannot_be_approved(self):
        self.fail(lambda d:d["review_gate"].update(training_approved=True),"NOT_APPROVED")
        self.fail(lambda d:d["review_gate"].update(independent_expert_reviews=2),"NOT_APPROVED")
    def test_book_reference_must_exist(self):
        self.fail(lambda d:d["pairs"][0].update(source_review_ref=
                  "research/WILLIAMS_V7_RULE_CHECK_999_FAKE_20261115_RU.md"),
                  "REVIEW_REFERENCE")
    def test_both_variants_must_change_evidence(self):
        self.fail(lambda d:d["pairs"][0]["variant_b"].update(
                  scenario=d["pairs"][0]["variant_a"]["scenario"]),
                  "NO_COUNTERFACTUAL_CHANGE")
    def test_no_chart_file_can_be_inserted(self):
        self.fail(lambda d:d["pairs"][0].update(chart_file="future.json"),"GOLD_OR_PSEUDO_LABEL")
    def test_specific_adversarial_topics_present(self):
        txt=" ".join(p["comparison_axis"]+" "+p["variant_a"]["scenario"] for p in self.d["pairs"])
        for term in ("Jaw","UOPX","нул","100","AO","Teeth","фрактал","Lips","Squat"):
            self.assertIn(term,txt)
if __name__=="__main__":
    unittest.main()
