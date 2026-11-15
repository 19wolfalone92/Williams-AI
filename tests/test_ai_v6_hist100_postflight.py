"""Gold-free regression validation of historical100 audit and technical abstentions."""
import unittest
from scripts.ai_v6_hist100_postflight import analyze,check,report,status

class FailClosedPostflightTests(unittest.TestCase):
    def test_error_is_not_valid_abstention(self):
        self.assertEqual(status({"decision":"ABSTAIN","finish_reason":"error"}),"TECHNICAL_ERROR")
        self.assertEqual(status({"decision":"ABSTAIN","finish_reason":"length"}),"TECHNICAL_LENGTH")
        self.assertEqual(status({"decision":"ABSTAIN","finish_reason":"stop"}),"MODEL_ABSTAIN_NOT_EXPERT_REVIEWED")
        self.assertEqual(status({"decision":"INSUFFICIENT_DATA","finish_reason":"stop"}),"MODEL_CLAIMS_INSUFFICIENT_DATA")
    def test_fake_200_answers_never_get_accepted(self):
        with self.assertRaisesRegex(ValueError,"NOT_COMPLETE_OR_HAS_GOLD"):
            check("cpu",{"complete":False,"accuracy":None,"gold_labels":None})
        with self.assertRaisesRegex(ValueError,"NOT_COMPLETE_OR_HAS_GOLD"):
            check("gpu",{"complete":True,"accuracy":0.99,"gold_labels":{}})
    def test_does_not_score_accuracy_or_leak_gold(self):
        with self.assertRaises(ValueError):
            analyze({}, {})
    def test_complete_report_is_explicitly_not_gold(self):
        text=report({
          "summaries":{v:{role:{"finish":{}, "technical_abstain_count":0,
          "model_abstain_count_not_expert_verified":0}
          for role in ("matched_base","candidate_v5")} for v in ("cpu","gpu")},
          "expert_review_priority_queue_NOT_TRAINING_DATA":[
              {"edition":b, "cpu_errors_lora":0,"gpu_errors_lora":0,
               "cpu_disagreements":0,"gpu_disagreements":0}
              for b in ("TC1","NTD","TC2","COURSE")],
          "model_disagreement_count_NOT_ACCURACY":{"cpu":0,"gpu":0},
          "same_prediction_across_backends_NOT_CORRECTNESS":{"matched_base":0,"candidate_v5":0}})
        self.assertIn("не gold",text.lower())

if __name__=="__main__":unittest.main()
