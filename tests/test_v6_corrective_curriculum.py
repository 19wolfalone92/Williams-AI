import json,re,unittest
from williams_ai.v6_corrective_curriculum import corrective_curriculum,training_rows,validate_curriculum

class CorrectiveCurriculumTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.rows,cls.manifest=corrective_curriculum()
 def test_split_sizes_and_source_coverage(self):
  self.assertEqual(self.manifest["split_rows"],{"train":128,"validation":32,"development":32,"locked_holdout":32})
  self.assertEqual(set(self.manifest["source_editions"]),{"TC1","NTD","TC2","COURSE"})
  self.assertEqual(len(self.rows),224)
 def test_independent_etalon_recomputation(self):
  result=validate_curriculum(self.rows)
  self.assertTrue(result["etalon_recomputed"]);self.assertEqual(result["rows"],224)
  self.assertEqual(result["unique_ids"],224);self.assertEqual(result["unique_prompts"],224)
 def test_training_export_excludes_development_and_locked_holdout(self):
  rows,manifest=training_rows()
  self.assertEqual(len(rows),160);self.assertEqual({r["split"] for r in rows},{"train","validation"})
  self.assertFalse(any(r["split"] in ("development","locked_holdout","final_exam") for r in rows))
  self.assertFalse(manifest["locked_holdout_in_training_export"])
 def test_template_only_shortcut_is_no_better_than_majority(self):
  for family,audit in self.manifest["training_template_shortcut_audit"].items():
   self.assertLessEqual(audit["template_only_accuracy"],audit["global_majority_accuracy"]+1e-9,family)
 def test_no_answer_numeric_leakage_and_explanations(self):
  for row in self.rows:
   target=json.loads(row["answer"])
   if row["family"] in ("ao_difference","wm1_stop"):
    self.assertIsNone(re.search(r"(?<![0-9.])"+re.escape(str(target["answer"]))+r"(?![0-9.])",row["prompt"]))
   for field in ("calculation","reason","counterexample","uncertainty","cancel_if"):self.assertTrue(target[field].strip())
   self.assertIn(row["rule_id"],self.manifest["rule_cards"])
 def test_no_candidate_labels_or_historical_chart_leakage(self):
  self.assertEqual(self.manifest["candidate_labels_used"],0);self.assertEqual(self.manifest["historical_chart_rows"],0)
  self.assertFalse(self.manifest["final_exam_included"])
  self.assertFalse(any(any(x in r["case_id"].lower() for x in ("base","v5","v6")) for r in self.rows))

if __name__=="__main__":unittest.main()
