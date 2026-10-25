"""Integration/adversarial tests of consultation, not a claim of AI skill."""
from copy import deepcopy
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest

from williams_ai.algorithm_guard import audit_reasoning
from williams_ai.book_index import index_pages
from williams_ai.book_rules import registry
from williams_ai.causal_paper_lab import frozen_markets
from williams_ai.chart_reader import analyze_chart
from williams_ai.deliberate_learning import DeliberateBookModel
from williams_ai.grounded_inference import verified_private_book_pages
from williams_ai.reasoning_trial import evaluate_historical_thought


def proposal(obs, *, setup="NO_SETUP", need="NONE"):
    return {"symbol":obs["symbol"], "as_of":obs["as_of"], "edition":"TC2",
        "setup":setup, "side":"LONG" if setup=="WM2" else "NONE",
        "entry_state":"CLOSED_CANDIDATE" if setup=="WM2" else "NO_SETUP",
        "exit_basis":"NONE", "rule_ids":["TC2.WM2"] if setup=="WM2" else [],
        "learning_need":need, "reason":"Research interpretation.",
        "execution_allowed":False, "trade_action":None}


class Student:
    def __init__(self, first=None, second=None, invented_quote=False):
        self.first=first or {"setup":"UNKNOWN", "need":"MANAGEMENT"}
        self.second=second or {}
        self.invented_quote=invented_quote
        self.calls=[]
        self.passages=[]

    def propose_chart_reasoning(self, obs, **kwargs):
        self.calls.append((deepcopy(obs),deepcopy(kwargs)))
        return proposal(obs, **(self.first if len(self.calls)==1 else self.second))

    def study_passage(self, text, **kwargs):
        self.passages.append((text,kwargs))
        return json.dumps({"summary":"Сопровождение стоп exit stop требует закрытых баров.",
            "claims":[{"statement":"Текст источника нужно проверить на рынке.",
                "support":"THIS SUPPORT DOES NOT EXIST IN THE SOURCE" if self.invented_quote else text[:60]}],
            "questions":["Какие факты подтвердят перенос стопа?"]})


class DeliberateLearningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.market=frozen_markets()["DOGEUSDT"]
        cls.rows=cls.market["bars"][:924]
        cls.obs=analyze_chart("DOGEUSDT","H1",cls.rows,as_of=cls.rows[-1]["close_time"],
            tick_size=cls.market["tick_size"])

    def setUp(self):
        self.temp=TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.books=self.root/"books.db"
        self.notes=self.root/"notes.db"
        ed=next(x for x in registry()["books"] if x["id"]=="TC2")
        pages=[""]*ed["pages"]
        # Artificial source fixtures, never real original-book validation.
        pages[130]="Защитный стоп проверяется по противоположному экстремуму закрытого бара. "*2
        pages[134]="Второй мудрец Super AO требует последовательности столбцов и ценового пробоя. "*2
        pages[160]="Сопровождение: стоп переносится за экстремумы завершённых баров; не расширять риск. "*2
        index_pages(self.books,"TC2",ed["sha256"],pages,declared_pages=ed["pages"])

    def wrap(self, student, model_id="gguf-student"):
        return DeliberateBookModel(student,self.books,self.notes,model_id=model_id)

    def test_rule_gap_reads_exact_pages_studies_and_reconsiders_same_cutoff(self):
        student=Student()
        model=self.wrap(student)
        result=model.propose_chart_reasoning(self.obs)
        self.assertEqual(result["setup"],"NO_SETUP")
        self.assertEqual(len(student.calls),2)
        self.assertEqual(student.calls[0][0],student.calls[1][0])
        self.assertEqual(student.calls[1][0]["as_of"],self.rows[-1]["close_time"])
        self.assertEqual(len(student.passages),2)
        refs=[x["source"] for x in student.calls[1][1]["primary_pages"]]
        self.assertEqual(set(refs),{"TC2: PDF 161","TC2: PDF 131"})
        self.assertTrue(student.calls[1][1]["study_lessons"])
        self.assertEqual(model.latest["status"],"RECONSIDERED_NOT_INDEPENDENTLY_VALIDATED")
        self.assertFalse(model.latest["model_weights_updated"])
        self.assertFalse(model.latest["independent_review_completed"])
        with sqlite3.connect(self.notes) as db:
            self.assertEqual(db.execute("SELECT count(*) FROM study_notes").fetchone()[0],2)
            event=json.loads(db.execute("SELECT diagnostics_json FROM decision_study_events").fetchone()[0])
            self.assertEqual(event["source_pages"],refs)
            self.assertNotIn("excerpt",json.dumps(event))

    def test_competent_wait_is_not_retried_to_manufacture_entry(self):
        student=Student(first={})
        student.first={}
        model=self.wrap(student)
        model.propose_chart_reasoning(self.obs)
        self.assertEqual(len(student.calls),1)
        self.assertEqual(student.passages,[])
        self.assertFalse(model.latest["reconsideration_performed"])

    def test_critic_detects_unsupported_claim_even_without_declared_gap(self):
        student=Student(first={"setup":"WM2"})
        model=self.wrap(student)
        model.propose_chart_reasoning(self.obs)
        self.assertIn("WM2_SUPER_AO_NOT_OBSERVED",model.latest["first_failures"])
        self.assertIn("TC2.WM2",model.latest["source_rules"])
        self.assertTrue(all(x["source"]=="TC2: PDF 135"
            for x in student.calls[-1][1]["primary_pages"]))

    def test_two_bad_attempts_remain_failed_after_consultation(self):
        student=Student(first={"setup":"WM2"},second={"setup":"WM2"})
        model=self.wrap(student)
        result=model.propose_chart_reasoning(self.obs)
        self.assertEqual(len(student.calls),2)
        self.assertEqual(model.latest["status"],"UNSUPPORTED_AFTER_STUDY")
        self.assertIn("WM2_SUPER_AO_NOT_OBSERVED",audit_reasoning(self.obs,result)["hard_failures"])

    def test_fabricated_quote_never_enters_memory_but_source_can_still_be_read(self):
        student=Student(invented_quote=True)
        model=self.wrap(student)
        model.propose_chart_reasoning(self.obs)
        self.assertEqual(model.latest["failed_study_chunks"],2)
        self.assertEqual(model.latest["studied_chunks"],0)
        self.assertEqual(student.calls[1][1]["study_lessons"],[])
        self.assertTrue(student.calls[1][1]["primary_pages"])

    def test_restart_reuses_only_same_model_notes_without_claiming_new_training(self):
        self.wrap(Student()).propose_chart_reasoning(self.obs)
        resumed=Student()
        wrapper=self.wrap(resumed)
        wrapper.propose_chart_reasoning(self.obs)
        self.assertEqual(resumed.passages,[])
        self.assertEqual(wrapper.latest["studied_chunks"],0)
        self.assertTrue(wrapper.latest["recalled_notes"])
        other=Student()
        self.wrap(other,"gguf-other").propose_chart_reasoning(self.obs)
        self.assertEqual(len(other.passages),2)
        self.assertTrue(all(n["model_id"]=="gguf-other"
            for n in other.calls[-1][1]["study_lessons"]))

    def test_missing_source_does_not_invent_a_repaired_answer(self):
        with sqlite3.connect(self.books) as db:
            db.execute("DELETE FROM excerpts")
        student=Student()
        wrapper=self.wrap(student)
        result=wrapper.propose_chart_reasoning(self.obs)
        self.assertEqual(result["setup"],"UNKNOWN")
        self.assertEqual(len(student.calls),1)
        self.assertEqual(wrapper.latest["status"],"SOURCE_TEXT_MISSING_VISUAL_REVIEW_REQUIRED")

    def test_declared_gap_cannot_also_authorize_an_entry(self):
        result=audit_reasoning(self.obs,proposal(self.obs,setup="WM2",need="MANAGEMENT"))
        self.assertIn("ENTRY_CLAIM_WHILE_BOOK_RULE_NOT_UNDERSTOOD",result["hard_failures"])
        self.assertIn("INVALID_MODEL_LEARNING_NEED",audit_reasoning(
            self.obs,{**proposal(self.obs),"learning_need":"execute Binance now"})["hard_failures"])

    def test_focus_rejects_cross_edition_rules_and_false_book_identity(self):
        with self.assertRaisesRegex(ValueError,"same-edition"):
            verified_private_book_pages(self.books,"TC2",rule_ids=("NTD.EXITS",))
        with sqlite3.connect(self.books) as db:
            db.execute("UPDATE volumes SET sha256=?",("0"*64,))
        student=Student()
        wrapper=self.wrap(student)
        wrapper.propose_chart_reasoning(self.obs)
        self.assertEqual(wrapper.latest["status"],"SOURCE_UNAVAILABLE_OR_IDENTITY_REJECTED")
        self.assertEqual(student.passages,[])

    def test_historical_entry_point_invokes_learning_before_final_audit(self):
        student=Student()
        result=evaluate_historical_thought("DOGEUSDT",self.market["bars"],
            tick_size=self.market["tick_size"],bar_index=923,model=student,
            include_topdown=False,book_index=self.books,study_memory=self.notes,
            model_id="gguf-student")
        self.assertTrue(result["deliberate_book_learning"]["reconsideration_performed"])
        self.assertFalse(result["execution_allowed"])
        self.assertEqual(student.calls[-1][0]["as_of"],self.rows[-1]["close_time"])
        self.assertEqual(result["audit"]["hard_failures"],[])


if __name__=="__main__":
    unittest.main()
