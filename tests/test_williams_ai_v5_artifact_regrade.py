"""Frozen v5 exact-key examiner tests: no GGUF, GPU, PDFs or trading keys."""
import json
import unittest

from scripts.ai_v5_regrade_frozen_exam_artifact import regrade_28
from williams_ai.knowledge_curriculum_v5 import build


def _all_correct_fixture():
    rows,_=build()
    grades=[]
    for row in rows:
        if row["split"]!="test" or row["task_type"]!="MCQ":
            continue
        grades.append({
            "id":row["case_id"],"edition":row["edition"],
            "chosen":json.loads(row["answer"])["choice"],
            "correct":True,"abstained":False,"json_fallback_used":False})
    by={ed:{"correct":sum(x["edition"]==ed for x in grades),
            "total":sum(x["edition"]==ed for x in grades),"abstained":0}
        for ed in ("TC1","NTD","TC2","COURSE")}
    return rows,{"case_pass_fail":grades,"by_edition":by,"correct":28,"total":28}


class FrozenV5RegradeTests(unittest.TestCase):
    def test_reconstructs_all_28_examiners_answers(self):
        rows,graded=_all_correct_fixture()
        recomputed=regrade_28(graded,rows)
        self.assertEqual(recomputed["correct"],28)
        self.assertEqual(len(recomputed["wrong"]),0)

    def test_reported_pass_is_rejected_if_answer_is_wrong(self):
        rows,graded=_all_correct_fixture()
        original=graded["case_pass_fail"][0]["chosen"]
        graded["case_pass_fail"][0]["chosen"]="A" if original!="A" else "B"
        with self.assertRaisesRegex(ValueError,"SELF_REPORTED_GRADE_MISMATCH"):
            regrade_28(graded,rows)

    def test_duplicate_case_id_fails_closed(self):
        rows,graded=_all_correct_fixture()
        graded["case_pass_fail"][1]["id"]=graded["case_pass_fail"][0]["id"]
        with self.assertRaisesRegex(ValueError,"DUPLICATE_CASE_ID"):
            regrade_28(graded,rows)

    def test_abstention_must_be_counted_wrong(self):
        rows,graded=_all_correct_fixture()
        graded["case_pass_fail"][0]["chosen"]=""
        graded["case_pass_fail"][0]["abstained"]=True
        with self.assertRaisesRegex(ValueError,"SELF_REPORTED_GRADE_MISMATCH"):
            regrade_28(graded,rows)

    def test_missing_one_book_fails_closed(self):
        rows,graded=_all_correct_fixture()
        graded["case_pass_fail"]=[r for r in graded["case_pass_fail"]
                                  if r["edition"]!="COURSE"]
        with self.assertRaisesRegex(ValueError,"UNEXPECTED_CASE_COUNT"):
            regrade_28(graded,rows)


if __name__=="__main__":
    unittest.main()
