"""Source-scoped and private adaptive curriculum regression tests.

Fake engines are NOT trained GGUF. A retry can memorize a demonstrated
answer, so any improved PRACTICE score is not book mastery.
"""
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest

from williams_ai.adaptive_coach import learning_cases, coach_practice, notebook, prior_lesson
from williams_ai.local_model import LocalWilliamsModel


class RehearsingFakeStudent:
    def __init__(self):
        self.seen=[]
    def answer_exam_case(self,question,*,prior_lessons=None):
        self.seen.append({"question":question,"notes":prior_lessons})
        if prior_lessons:
            known=next(x for x in learning_cases() if x["id"]==question["id"])
            return {"choice":known["correct"],"reason":"Lesson paraphrased"}
        return {"choice":"D","reason":"Not sure"}


class AdaptiveBookTutorTests(unittest.TestCase):
    def test_curated_cases_reach_all_four_verified_books(self):
        cases=learning_cases()
        self.assertGreaterEqual(len(cases),26)
        self.assertEqual({x["edition"] for x in cases},{"TC1","NTD","TC2","COURSE"})
        self.assertEqual(len(cases),len({x["id"] for x in cases}))
        self.assertGreaterEqual(len({x["rule_id"] for x in cases}),25)
        for case in cases:
            self.assertIn(case["correct"],case["choices"])
            self.assertEqual(len(case["choices"]),3)

    def test_practice_error_adds_private_lesson_and_retry_not_mastery(self):
        with TemporaryDirectory() as tmp:
            destination=Path(tmp)/"my_notes.sqlite3"
            m=RehearsingFakeStudent()
            output=coach_practice(m,destination,model_id="gguf-test-model",
                                  editions=("TC1",),max_cases=2)
            self.assertEqual(output["cases"],2)
            self.assertEqual(output["initial_pass"],0)
            self.assertEqual(output["corrected_same_case_retries"],2)
            self.assertEqual(output["private_misconception_topics"],2)
            self.assertIsNone(output["actual_mastery_percent"])
            self.assertIsNone(output["blind_new_case_score"])
            self.assertFalse(output["language_model_weights_modified"])
            self.assertFalse(output["execution_allowed"])
            self.assertIsNone(output["trade_action"])
            self.assertEqual(len(m.seen),4)
            for submitted in m.seen:
                self.assertEqual(set(submitted["question"]),{"id","edition","question","choices"})
                self.assertNotIn("correct",submitted["question"])
                self.assertNotIn("feedback",submitted["question"])
            self.assertIsNone(m.seen[0]["notes"])
            self.assertIn("TC1:RULE:",m.seen[1]["notes"][0]["source"])
            with notebook(destination) as db:
                self.assertEqual(db.execute("select count(*) from practice_attempts").fetchone()[0],4)
                self.assertEqual(db.execute("select count(*) from private_remediation").fetchone()[0],2)

    def test_saved_learning_influences_later_training_without_changed_weights(self):
        with TemporaryDirectory() as tmp:
            path=Path(tmp)/"notes.db"
            model=RehearsingFakeStudent()
            first=coach_practice(model,path,model_id="gguf-v1",
                                 editions=("TC1",),max_cases=1,repeat_after_feedback=False)
            second=coach_practice(model,path,model_id="gguf-v1",
                                  editions=("TC1",),max_cases=1,repeat_after_feedback=False)
            third=coach_practice(model,path,model_id="gguf-v2",
                                 editions=("TC1",),max_cases=1,repeat_after_feedback=False)
            self.assertEqual((first["initial_pass"],second["initial_pass"],
                              third["initial_pass"]),(0,1,0))
            self.assertFalse(first["language_model_weights_modified"])
            self.assertEqual(first["private_misconception_topics"],1)
            self.assertEqual(second["private_misconception_topics"],1)
            self.assertEqual(third["private_misconception_topics"],1)

    def test_mixed_edition_lesson_rejected_before_inference(self):
        with TemporaryDirectory() as tmp:
            mfile=Path(tmp)/"fake.gguf"
            mfile.write_bytes(b"test only")
            class Engine:
                def __init__(self,**kwargs):self.calls=[]
                def create_chat_completion(self,**kwargs):
                    self.calls.append(kwargs)
                    return {"choices":[{"message":{"content":'{"choice":"A","reason":"ok"}'}}]}
            m=LocalWilliamsModel(mfile,loader=Engine)
            case=next(x for x in learning_cases() if x["edition"]=="TC1")
            safe={k:case[k] for k in ("id","edition","question","choices")}
            with self.assertRaisesRegex(ValueError,"cross-edition"):
                m.answer_exam_case(safe,prior_lessons=[
                    {"source":"NTD:RULE:NTD.AO","excerpt":"Different model explanation"}
                ])
            self.assertEqual(m._model.calls,[])
            m.answer_exam_case(safe,prior_lessons=[
                {"source":"TC1:RULE:TC1.W3","excerpt":"Example of prior correction"}
            ])
            sent=json.loads(m._model.calls[0]["messages"][1]["content"])
            self.assertEqual(set(sent),{"case","prior_edition_scoped_lessons"})
            self.assertNotIn("correct",sent["case"])
            self.assertNotIn("feedback",sent["case"])
            self.assertEqual(sent["prior_edition_scoped_lessons"][0]["source"],
                             "TC1:RULE:TC1.W3")
            self.assertNotIn("tools",m._model.calls[0])

    def test_input_rejection_no_written_case_rollback(self):
        with TemporaryDirectory() as tmp:
            dest=Path(tmp)/"notes.sqlite3"
            for kw in ({"model_id":"fake"},{"editions":("NTD","NTD")},
                       {"max_cases":0}):
                with self.assertRaises(ValueError):
                    options={"model_id":"gguf-safe",**kw}
                    coach_practice(RehearsingFakeStudent(),dest,**options)
            self.assertFalse(dest.exists())

    def test_notebook_is_private_and_results_are_non_executable(self):
        with TemporaryDirectory() as tmp:
            from os import stat
            path=Path(tmp)/"notebook.sqlite3"
            coach_practice(RehearsingFakeStudent(),path,
                           model_id="gguf-example",max_cases=1)
            self.assertEqual(stat(path).st_mode & 0o777,0o600)
            with notebook(path) as db:
                lesson=prior_lesson(db,"gguf-example","TC1","TC1.W3")
                self.assertEqual(len(lesson),1)
                self.assertFalse(prior_lesson(db,"gguf-example","NTD","TC1.W3"))


if __name__=="__main__":
    unittest.main()
