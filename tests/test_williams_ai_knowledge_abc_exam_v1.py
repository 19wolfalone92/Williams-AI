"""Exam protocol tests: question-only vs retrieved rule cards, never keys."""
import unittest

from williams_ai.knowledge_abc_exam_v1 import assess_model


class SyntheticModel:
    def __init__(self):
        self.calls=[]
    def answer_exam_case(self,question,*,prior_lessons=None):
        self.calls.append({"id":question["id"],"edition":question["edition"],
                           "saw_prior_lessons":prior_lessons is not None,
                           "safe_keys":set(question)=={"id","edition","question","choices"}})
        return {"choice":"A","reason":"Unit test fixed-choice deliberately non-expert."}


class KnowledgeProtocolTests(unittest.TestCase):
    def test_same_frozen_protocol_no_hidden_keys_and_no_fake_real_model_flag(self):
        fake=SyntheticModel()
        report=assess_model(fake,model_sha256="a"*64,
                            per_edition=4,seed=20261024)
        self.assertFalse(report["real_model_inference_performed"])
        self.assertFalse(report["user_trained_model_benchmarked"])
        self.assertEqual(report["scores"]["case_count"],16)
        self.assertEqual(len(fake.calls),32)
        self.assertTrue(all(c["safe_keys"] for c in fake.calls))
        self.assertFalse(any(c["saw_prior_lessons"] for c in fake.calls[:16]))
        self.assertFalse(report["execution_allowed"])
        self.assertIsNone(report["trade_action"])
        self.assertEqual(
            report["conditions"]["closed_book"]["protocol_sha256"],
            report["conditions"]["source_assisted"]["protocol_sha256"])
        self.assertEqual(set(report["conditions"]["closed_book"]["by_edition"]),
                         {"TC1","NTD","TC2","COURSE"})
        self.assertEqual(
            sum(item["total"] for item in
                report["conditions"]["closed_book"]["by_edition"].values()),16)
        self.assertEqual(
            report["scores"]["without_sources"],
            report["conditions"]["closed_book"]["correct"])
        self.assertFalse(report["book_visual_comprehension_tested"])

    def test_reject_forged_base_weight_hash(self):
        with self.assertRaisesRegex(ValueError,"PINNED_GGUF"):
            assess_model(SyntheticModel(),model_sha256="test",per_edition=2)

    def test_invalid_size_denied_by_underlying_exam(self):
        with self.assertRaisesRegex(ValueError,"INVALID_PER_EDITION"):
            assess_model(SyntheticModel(),model_sha256="a"*64,per_edition=9)


if __name__=="__main__":
    unittest.main()
