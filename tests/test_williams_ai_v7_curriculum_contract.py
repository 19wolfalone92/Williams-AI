import copy,json,unittest
from pathlib import Path
from williams_ai.v7_curriculum_contract import validate
PATH=Path(__file__).resolve().parents[1]/"knowledge"/"WILLIAMS_V7_SOURCE_CURRICULUM_CANDIDATES_20261115.json"
class V7CurriculumContractTests(unittest.TestCase):
 def setUp(self):self.data=json.loads(PATH.read_text(encoding="utf-8"))
 def test_ten_source_pinned_lessons_cover_four_books_no_gold(self):
  x=validate(self.data);self.assertEqual(x["lesson_candidate_count"],10)
  self.assertEqual(x["books_covered"],4);self.assertFalse(x["sft_ready"])
 def test_label_or_model_prediction_field_rejected(self):
  x=copy.deepcopy(self.data);x["lesson_cards"][0]["decision"]="LONG_STOP_ENTRY"
  with self.assertRaisesRegex(ValueError,"LABEL_LEAK"):validate(x)
  x=copy.deepcopy(self.data);x["lesson_cards"][0]["model_output"]="BUY"
  with self.assertRaisesRegex(ValueError,"LABEL_LEAK"):validate(x)
 def test_source_hash_and_rule_edition_must_match(self):
  x=copy.deepcopy(self.data);x["lesson_cards"][0]["source_pdf_sha256"]="0"*64
  with self.assertRaisesRegex(ValueError,"UNPINNED_SOURCE"):validate(x)
  x=copy.deepcopy(self.data);x["lesson_cards"][0]["rule_ids"]=["TC2.WM1.ANG"]
  with self.assertRaisesRegex(ValueError,"RULE_BOOK_MISMATCH"):validate(x)
 def test_cannot_mark_as_expert_approved(self):
  x=copy.deepcopy(self.data);x["review_gate"]["expert_approved"]=True
  with self.assertRaisesRegex(ValueError,"PREMATURE_APPROVAL"):validate(x)
if __name__=="__main__":unittest.main()
