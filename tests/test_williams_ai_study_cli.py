"""Drive FOUR verified local book sources into a resumable private lesson DB.

These tests use a fake model; they do not assert GGUF learned the books.
"""
import json
from pathlib import Path
import tempfile
import unittest

from williams_ai.book_index import index_pages
from williams_ai.book_rules import registry
from williams_ai.study_cli import train_private_curriculum,_model_identity


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

    def test_wrong_quote_cannot_be_laundered_into_accepted_memory(self):
        student=FakeBookStudent(wrong=True)
        run=train_private_curriculum(self.books,self.study,student,
                                     model_id="broken-v1",editions=("TC2",),
                                     budget_per_book=2,rounds=4)
        self.assertEqual(run["new_study_chunks"],0)
        self.assertGreater(run["failed_or_retry_chunks"],0)
        self.assertEqual(run["progress"]["books"][2]["studied_chunks"],0)
        self.assertLessEqual(len(student.calls),2)

    def test_model_identity_uses_actual_bytes_not_name(self):
        first=_model_identity(self.model)
        self.model.write_bytes(b"CHANGED_WEIGHTS")
        self.assertNotEqual(first,_model_identity(self.model))
        self.assertTrue(first.startswith("gguf-"))
        with self.assertRaises(FileNotFoundError):
            _model_identity(self.root/"missing.gguf")

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


if __name__=="__main__":
    unittest.main()
