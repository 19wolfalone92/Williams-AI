import copy,json,unittest
from pathlib import Path
from williams_ai.v7_edition_family_retrieval import (
 REQUIRED,PROFILE_SCHEMA,build_evidence_bundle,required_rule_ids
)
ROOT=Path(__file__).resolve().parents[1]
def question(book,family):
 return {
  ("TC1","ENTRY"):"Проверка первого подтверждённого фрактального сигнала и рыночного ритма по Торговому Хаосу 1.",
  ("TC1","WAVE"):"Определите степень волны, завершение W4 и целевую область W5 по первоисточнику.",
  ("NTD","ENTRY"):"Проверьте фрактал, Аллигатор, AO, AC, зоны и линию баланса для входа в рынок по NTD.",
  ("TC2","EXIT"):"Проверьте противоположный сигнал Мудреца, условия стопа и допустимость разворота позиции по TC2.",
  ("COURSE","WAVE"):"Оцените степень волны, минимум W4, crossing 5/34/5 и вложенные A-B-C по Profitunity Course."
 }.get((book,family),"Проверка торговой задачи из указанного первоисточника.")

class V7EditionFamilyRetrievalTests(unittest.TestCase):
 def test_profiles_exist_for_all_four_books_and_three_families(self):
  self.assertEqual(len(REQUIRED),12)
  self.assertEqual({x[0] for x in REQUIRED},{ "TC1","NTD","TC2","COURSE"})
  self.assertEqual({x[1] for x in REQUIRED},{"ENTRY","EXIT","WAVE"})
 def test_historical_tc1_query_with_alligator_is_rejected_as_cross_edition(self):
  with self.assertRaisesRegex(ValueError,"CROSS_EDITION_TERM"):
   build_evidence_bundle("TC1: фрактал, AO и Аллигатор",edition="TC1",family="ENTRY")
 def test_cross_edition_russian_inflections_and_english(self):
  for phrase in ("Аллигатор","аллигатора","ALLIGATOR"):
   with self.subTest(phrase=phrase):
    with self.assertRaisesRegex(ValueError,"CROSS_EDITION_TERM"):
     build_evidence_bundle(f"TC1: фрактал и {phrase}",edition="TC1",family="ENTRY")
 def test_empty_registry_override_is_not_ignored(self):
  with self.assertRaisesRegex(ValueError,"INVALID_RULE_REGISTRY"):
   build_evidence_bundle(question("NTD","ENTRY"),edition="NTD",family="ENTRY",registry_data={})
 def test_optional_cards_must_be_from_the_same_reviewed_edition(self):
  from unittest.mock import patch
  with patch("williams_ai.v7_edition_family_retrieval.rank_reviewed_rules",return_value=[{"rule_id":"NTD.AO"}]):
   with self.assertRaisesRegex(ValueError,"OPTIONAL_RULE_WRONG_EDITION"):
    build_evidence_bundle(question("TC2","EXIT"),edition="TC2",family="EXIT")
 def test_ntd_entry_contains_all_mandatory_prerequisites_even_if_lexical_ranker_omits_them(self):
  x=build_evidence_bundle(question("NTD","ENTRY"),edition="NTD",family="ENTRY")
  self.assertEqual(x["schema"],PROFILE_SCHEMA)
  for rid in ("NTD.FRACTAL","NTD.ALLIGATOR","NTD.AO","NTD.AC","NTD.ZONE","NTD.BALANCE"):
   self.assertIn(rid,x["retrieved_rule_ids"])
  self.assertTrue(x["mandatory_coverage_complete"])
  self.assertTrue(all(z["edition"]=="NTD" for z in x["required_cards"]+x["optional_cards"]))
 def test_tc2_exit_forces_reverse_and_wm1_not_only_wm2_wm3(self):
  x=build_evidence_bundle(question("TC2","EXIT"),edition="TC2",family="EXIT")
  for rid in ("TC2.REVERSE","TC2.WM1.BAR","TC2.WM1.ANG","TC2.WM1.TRIGGER","TC2.WM1.STOP"):
   self.assertIn(rid,x["retrieved_rule_ids"])
 def test_course_wave_requires_degree_and_nested_visual_context(self):
  x=build_evidence_bundle(question("COURSE","WAVE"),edition="COURSE",family="WAVE")
  for rid in ("COURSE.W4.ZERO","COURSE.AO5345","COURSE.RHYTHM","COURSE.VISUAL.NESTED_ABC"):
   self.assertIn(rid,x["retrieved_rule_ids"])
 def test_no_model_labels_no_execution_and_v6_unchanged(self):
  x=build_evidence_bundle(question("TC1","WAVE"),edition="TC1",family="WAVE")
  self.assertFalse(x["model_predictions_used"])
  self.assertFalse(x["expert_gold_included"])
  self.assertFalse(x["execution_allowed"])
  self.assertTrue(x["frozen_v6_unchanged"])
 def test_required_unreviewed_or_wrong_edition_card_fails_closed(self):
  from williams_ai.book_rules import registry
  fake=registry();row=next(x for x in fake["rule_cards"] if x["id"]=="NTD.AO");row["status"]="UNREVIEWED"
  with self.assertRaisesRegex(ValueError,"NOT_PAGE_REVIEWED"):
   build_evidence_bundle(question("NTD","ENTRY"),edition="NTD",family="ENTRY",registry_data=fake)
 def test_unsupported_family_fails_closed(self):
  with self.assertRaisesRegex(ValueError,"UNKNOWN_EDITION_OR_TASK_FAMILY"):
   required_rule_ids("OTHER","ENTRY")

if __name__=="__main__":unittest.main()
