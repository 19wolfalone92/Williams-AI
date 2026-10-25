"""Offline RAG-to-GGUF bridge: the actual indexed book, not only a summary.

Unit tests prove source isolation and causality. They use a mock GGUF engine;
real cognitive ability, training and live trading are NOT certified by CI.
"""
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from williams_ai.book_index import index_pages
from williams_ai.book_rules import registry
from williams_ai.grounded_inference import verified_private_book_pages
from williams_ai.local_model import LocalWilliamsModel
from williams_ai.causal_paper_lab import frozen_markets
from williams_ai.reasoning_trial import evaluate_historical_thought


class LocalSourceGroundingTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.books=Path(self.tmp.name)/"licensed_books.sqlite3"
        texts={
            "TC1":"волна Эллиотта фрактал AO дивергенция импульс коррекция",
            "NTD":"Аллигатор фрактал Awesome Oscillator Acceleration Balance Line вход выход",
            "TC2":"Первый Второй Третий Мудрец ангуляция Super AO фрактал зубы стоп",
            "COURSE":"Profitunity Course Elliott fractal Hump Thumb Trading Partner",
        }
        for b in registry()["books"]:
            pages=[""]*b["pages"]
            pages[10]=texts[b["id"]]+" точное учебное объяснение по тексту книги без приказов на торги."
            index_pages(self.books,b["id"],b["sha256"],pages,declared_pages=b["pages"])
        self.x=frozen_markets()["ETHUSDT"]
        self.weights=Path(self.tmp.name)/"research.gguf"
        self.weights.write_bytes(b"MOCK-NOT-REAL")

    def test_each_edition_retrieves_only_its_own_book(self):
        for edition in ("TC1","NTD","TC2","COURSE"):
            with self.subTest(edition=edition):
                pages=verified_private_book_pages(self.books,edition,limit=2)
                self.assertTrue(pages)
                self.assertTrue(all(x["source"].startswith(edition+": PDF ") for x in pages))
                self.assertTrue(all(x["status"].startswith("PRIVATE_INDEXED") for x in pages))
                self.assertEqual(pages[0]["pdf_page"],11)
                self.assertTrue(all(len(x["excerpt"])<=650 for x in pages))

    def test_wrong_fingerprint_fails_before_model_read(self):
        con=sqlite3.connect(self.books)
        con.execute("UPDATE volumes SET sha256=? WHERE book='TC2'",("0"*64,))
        con.commit()
        con.close()
        with self.assertRaisesRegex(ValueError,"fingerprint"):
            verified_private_book_pages(self.books,"TC2")

    def test_precise_question_overrides_generic_topics_and_keeps_original_chunk_hash(self):
        import hashlib
        content="Nested Elliott degree correction вложенность степени коррекции. "*2
        with sqlite3.connect(self.books) as db:
            db.execute("INSERT INTO excerpts VALUES('TC2',170,?,'UNVERIFIED_EXTRACTION')",(content,))
        result=verified_private_book_pages(self.books,"TC2",question="Nested Elliott degree",limit=1)
        self.assertEqual(result[0]["pdf_page"],170)
        self.assertEqual(result[0]["chunk_sha256"],hashlib.sha256(content.encode()).hexdigest())
        self.assertEqual(verified_private_book_pages(self.books,"TC2",
            question="zzzzmissingchapter",selected_setup="WM1"),[])

    def test_private_pages_in_model_prompt_but_never_public_report(self):
        class Engine:
            def __init__(self,**kwargs):
                self.kwargs=kwargs
                self.input=None
            def create_chat_completion(self,**kwargs):
                self.input=kwargs
                prompt=json.loads(kwargs["messages"][1]["content"])
                result={
                    "symbol":prompt["chart"]["symbol"],
                    "as_of":prompt["chart"]["as_of"],
                    "edition":prompt["edition"],
                    "setup":"NO_SETUP","side":"NONE",
                    "entry_state":"NO_SETUP","exit_basis":"NONE",
                    "rule_ids":[],"trade_action":None,"execution_allowed":False,
                    "reason":"Нет подтверждённой действительной цены входа"
                }
                return {"choices":[{"message":{"content":json.dumps(result,ensure_ascii=False)}}]}
        model=LocalWilliamsModel(self.weights,loader=Engine)
        output=evaluate_historical_thought(
            "ETHUSDT",self.x["bars"],tick_size=self.x["tick_size"],
            bar_index=300,edition="TC2",model=model,book_index=self.books)
        self.assertFalse(output["model_saw_future_bars"])
        self.assertEqual(output["audit"]["verdict"],"PROVISIONAL_REASONING_ONLY")
        self.assertEqual(output["private_book_pages_used"],["TC2: PDF 11"])
        self.assertFalse(output["private_book_passages_exported"])
        self.assertNotIn("точное учебное объяснение",json.dumps(output,ensure_ascii=False))
        prompt=json.loads(model._model.input["messages"][1]["content"])
        self.assertEqual(prompt["edition"],"TC2")
        self.assertEqual(prompt["primary_book_pages"][0]["source"],"TC2: PDF 11")
        self.assertIn("точное учебное объяснение",prompt["primary_book_pages"][0]["excerpt"])
        self.assertNotIn("secret",prompt)
        self.assertNotIn("tools",model._model.input)
        self.assertEqual(model._model.input["temperature"],0.0)
        self.assertIsNone(output["book_mastery_percent"])

    def test_cannot_leak_other_book_as_matched_source(self):
        class NoCall:
            def __init__(self,**kwargs):pass
            def create_chat_completion(self,**kwargs):
                raise AssertionError("Must fail validation before inference")
        model=LocalWilliamsModel(self.weights,loader=NoCall)
        market=self.x
        from williams_ai.chart_reader import analyze_chart
        chart=analyze_chart("ETHUSDT","H1",market["bars"][:301],
                            as_of=market["bars"][300]["close_time"],
                            tick_size=market["tick_size"])
        with self.assertRaises(ValueError):
            model.propose_chart_reasoning(chart,edition="TC2",primary_pages=[{
                "source":"NTD: PDF 11","excerpt":"read me",
                "status":"PRIVATE_INDEXED_PDF_TEXT_NOT_INDEPENDENTLY_INTERPRETED"
            }])

    def test_book_index_is_optional_and_nonexistent_is_explicit_failure(self):
        class Fake:
            def propose_chart_reasoning(self,chart,*,edition):
                return {"symbol":chart["symbol"],"as_of":chart["as_of"],
                        "edition":edition,"setup":"NO_SETUP","side":"NONE",
                        "trade_action":None}
        evaluate_historical_thought(
            "ETHUSDT",self.x["bars"],tick_size=self.x["tick_size"],
            bar_index=300,model=Fake())
        with self.assertRaises(FileNotFoundError):
            evaluate_historical_thought(
                "ETHUSDT",self.x["bars"],tick_size=self.x["tick_size"],
                bar_index=300,model=Fake(),
                book_index=Path(self.tmp.name)/"no_source.sqlite3")


if __name__=="__main__":
    unittest.main()
