"""Drive FOUR verified local book sources into a resumable private lesson DB.

These tests use a fake model; they do not assert GGUF learned the books.
"""
import json
import io
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from williams_ai.book_index import index_pages
from williams_ai.book_rules import registry
from williams_ai.study_cli import train_private_curriculum,_model_identity
from williams_ai.study import study_report


class FakeBookStudent:
    def __init__(self,wrong=False):
        self.calls=[]
        self.wrong=wrong
    def study_passage(self,text,*,edition,page,rule_cards):
        self.calls.append((edition,page))
        if self.wrong:
            return json.dumps({"summary":"No source","claims":[{
                "statement":"Fake claim","support":"fabricated quote beyond source"
            }],"questions":[]})
        return json.dumps({
            "summary":f"{edition} — изучен отрывок {page}, требуется независимая проверка.",
            "claims":[{"statement":"Наблюдается текст исходной страницы.",
                       "support":text[:min(80,len(text))]}],
            "questions":["Подтвердить соответствие с книжным рисунком?"],
        },ensure_ascii=False)


class FourBookStudyCLITests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.books=self.root/"private_books.sqlite3"
        self.study=self.root/"private_lessons.sqlite3"
        self.model=self.root/"fake.gguf"
        self.model.write_bytes(b"TEST_MODEL_CONTENT")
        for book in registry()["books"]:
            pages=[""]*book["pages"]
            pages[3]=(f'{book["id"]} исследует движение рынка и особый сигнальный алгоритм. '
                      'Подтверждение не означает живой ордер.')
            pages[4]=(f'{book["id"]} другую структуру проверяет по завершенным барам. '
                      'Следует сверить временной масштаб.')
            index_pages(self.books,book["id"],book["sha256"],pages,
                        declared_pages=book["pages"])

    def test_four_book_study_is_resumable_private_and_source_anchored(self):
        model=FakeBookStudent()
        id=_model_identity(self.model)
        first=train_private_curriculum(
            self.books,self.study,model,model_id=id,
            budget_per_book=1,rounds=3)
        self.assertEqual(first["new_study_chunks"],8)
        self.assertEqual(len(model.calls),8)
        self.assertFalse(first["language_model_weights_modified"])
        self.assertIsNone(first["full_books_understood_percent"])
        self.assertFalse(first["execution_allowed"])
        for book in first["progress"]["books"]:
            self.assertEqual(book["studied_chunks"],2)
            self.assertEqual(book["pages_started"],2)
            self.assertGreater(book["total_pages"],100)
        second=train_private_curriculum(
            self.books,self.study,model,model_id=id,
            budget_per_book=1,rounds=1)
        self.assertEqual(second["new_study_chunks"],0)
        self.assertEqual(len(model.calls),8)
        self.assertFalse(second["progress"]["weights_retrained"])

    def test_two_different_models_cannot_share_claimed_book_coverage(self):
        first=train_private_curriculum(
            self.books,self.study,FakeBookStudent(),model_id="model-alpha",
            editions=("TC2",),budget_per_book=1,rounds=1)
        second=train_private_curriculum(
            self.books,self.study,FakeBookStudent(),model_id="model-beta",
            editions=("TC2",),budget_per_book=1,rounds=1)
        self.assertEqual(first["new_study_chunks"],1)
        self.assertEqual(second["new_study_chunks"],1)
        alpha=study_report(self.books,self.study,model_id="model-alpha")
        beta=study_report(self.books,self.study,model_id="model-beta")
        a=next(x for x in alpha["books"] if x["edition"]=="TC2")
        b=next(x for x in beta["books"] if x["edition"]=="TC2")
        self.assertEqual(a["studied_chunks"],1)
        self.assertEqual(b["studied_chunks"],1)
        self.assertFalse(a["model_has_reviewed_all_pdf_page_text"])
        unscoped=study_report(self.books,self.study)
        t=next(x for x in unscoped["books"] if x["edition"]=="TC2")
        self.assertEqual(t["studied_chunks"],1)  # best single student, not 2 combined
        self.assertEqual(t["per_model_progress"]["model-alpha"]["studied_chunks"],1)
        self.assertEqual(t["per_model_progress"]["model-beta"]["studied_chunks"],1)
        self.assertTrue(unscoped["unscoped_progress_not_summed_across_models"])
        self.assertFalse(alpha["weights_retrained"])

    def test_wrong_quote_cannot_be_laundered_into_accepted_memory(self):
        student=FakeBookStudent(wrong=True)
        run=train_private_curriculum(self.books,self.study,student,
                                     model_id="broken-v1",editions=("TC2",),
                                     budget_per_book=2,rounds=4)
        self.assertEqual(run["new_study_chunks"],0)
        self.assertGreater(run["failed_or_retry_chunks"],0)
        self.assertEqual(next(x for x in run["progress"]["books"] if x["edition"]=="TC2")["studied_chunks"],0)
        self.assertLessEqual(len(student.calls),2)

    def test_model_identity_uses_actual_bytes_not_name(self):
        first=_model_identity(self.model)
        self.model.write_bytes(b"CHANGED_WEIGHTS")
        self.assertNotEqual(first,_model_identity(self.model))
        self.assertTrue(first.startswith("gguf-"))
        with self.assertRaises(FileNotFoundError):
            _model_identity(self.root/"missing.gguf")

    def test_research_cli_uses_the_same_real_weight_identity_as_study_cli(self):
        from williams_ai.research_cli import main
        from williams_ai.study import recall
        args=["research_cli","study","--books",str(self.books),
              "--memory",str(self.study),"--edition","TC2",
              "--model",str(self.model),"--budget","1"]
        with patch("sys.argv",args),patch(
                "williams_ai.local_model.LocalWilliamsModel",return_value=FakeBookStudent()),redirect_stdout(io.StringIO()):
            main()
        report=study_report(self.books,self.study,model_id=_model_identity(self.model))
        tc2=next(x for x in report["books"] if x["edition"]=="TC2")
        self.assertEqual(tc2["studied_chunks"],1)

    def test_cross_edition_and_budget_rejected(self):
        model=FakeBookStudent()
        for kwargs in (
            {"editions":("TC2","TC2")},
            {"editions":("FAKE",)},
            {"budget_per_book":0},
            {"rounds":0},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    train_private_curriculum(self.books,self.study,model,
                                             model_id="fake-v1",**kwargs)

    def test_focused_source_pages_resume_without_counting_unread_pages(self):
        model=FakeBookStudent()
        first=train_private_curriculum(
            self.books,self.study,model,model_id="focused-model",
            editions=("NTD",),pdf_pages=(5,),budget_per_book=1,rounds=1)
        self.assertEqual(model.calls,[("NTD",5)])
        self.assertEqual(first["new_study_chunks"],1)
        self.assertEqual(first["focused_pdf_pages"],[5])
        again=train_private_curriculum(
            self.books,self.study,model,model_id="focused-model",
            editions=("NTD",),pdf_pages=(5,),budget_per_book=1,rounds=1)
        self.assertEqual(again["new_study_chunks"],0)
        ordinary=train_private_curriculum(
            self.books,self.study,model,model_id="focused-model",
            editions=("NTD",),budget_per_book=1,rounds=1)
        self.assertEqual(ordinary["new_study_chunks"],1)
        self.assertEqual(model.calls,[("NTD",5),("NTD",4)])

    def test_focused_study_rejects_mixed_edition_and_non_book_pages(self):
        for args in (
            {"pdf_pages":(4,)},
            {"editions":("NTD",),"pdf_pages":()},
            {"editions":("NTD",),"pdf_pages":(4,4)},
            {"editions":("NTD",),"pdf_pages":(True,)},
            {"editions":("NTD",),"pdf_pages":(99999,)},
        ):
            model=FakeBookStudent()
            with self.subTest(args=args),self.assertRaises(ValueError):
                train_private_curriculum(self.books,self.study,model,
                                         model_id="focused-model",**args)
            self.assertEqual(model.calls,[])


if __name__=="__main__":
    unittest.main()
