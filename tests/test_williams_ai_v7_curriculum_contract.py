import copy,json,unittest
from pathlib import Path
from williams_ai.v7_curriculum_contract import validate
PATH=Path(__file__).resolve().parents[1]/"knowledge"/"WILLIAMS_V7_SOURCE_CURRICULUM_CANDIDATES_20261115.json"
class V7CurriculumContractTests(unittest.TestCase):
 def setUp(self):self.data=json.loads(PATH.read_text(encoding="utf-8"))
 def test_twenty_six_source_pinned_lessons_cover_four_books_no_gold(self):
  x=validate(self.data);self.assertEqual(x["lesson_candidate_count"],26)
  self.assertEqual(x["books_covered"],4);self.assertFalse(x["sft_ready"])
 def test_course_hump_thumb_mfi_and_trading_partner_are_separate(self):
  rows={x["id"]:x for x in self.data["lesson_cards"]}
  for key,rule in (("V7.COURSE.MFI.PROFITUNITY_WINDOWS","COURSE.MFI"),
                   ("V7.COURSE.THUMB.EARLY_FRACTAL_ENTRY","COURSE.THUMB"),
                   ("V7.COURSE.HUMP.OPPOSITE_NONOVERLAPPING_FRACTALS","COURSE.HUMP"),
                   ("V7.COURSE.TRADING_PARTNER.MULTITIMEFRAME_TABLE","COURSE.TRADING_PARTNER")):
   self.assertIn(rule,rows[key]["rule_ids"]);self.assertEqual(rows[key]["edition"],"COURSE")
 def test_ntd_dimensions_remain_separate(self):
  rows={x["id"]:x for x in self.data["lesson_cards"]}
  for key,rule in (("V7.NTD.ALLIGATOR.THREE_BALANCE_LINES","NTD.ALLIGATOR"),
                   ("V7.NTD.AO.SAUCER_CROSS_TWIN_PEAKS","NTD.AO"),
                   ("V7.NTD.AC.ACCELERATION_SIGNAL_COUNTS","NTD.AC"),
                   ("V7.NTD.ZONES.AO_AC_PRICE_ADDITION","NTD.ZONE")):
   self.assertIn(rule,rows[key]["rule_ids"]);self.assertEqual(rows[key]["edition"],"NTD")
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
