"""The four original books' studied lessons must reach *later* market thought.

Only the SAME model fingerprint, SAME edition, indexed PDF SHA, and unverified
source status can be reused. Tests never run a real local neural network and
never claim full reading, profitable trades or exact Elliott annotation.
"""
import json
import io
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from williams_ai.book_index import index_pages
from williams_ai.book_rules import registry
from williams_ai.study import study_batch,recall
from williams_ai.reasoning_trial import evaluate_historical_thought
from williams_ai.causal_paper_lab import frozen_markets
from williams_ai.local_model import LocalWilliamsModel
from williams_ai.chart_reader import analyze_chart


class ReadingStudent:
    def study_passage(self,text,*,edition,page,rule_cards):
        return json.dumps({
            "summary":"Мудрец TC2 изучает фрактал и ангуляцию на закрытых свечах.",
            "claims":[{"statement":"Есть текст страницы","support":text[:70]}],
            "questions":["Как проверяется фрактал?"],},ensure_ascii=False)


class InspectingStudent:
    def __init__(self):
        self.kwargs=None
    def propose_chart_reasoning(self,chart,**kwargs):
        self.kwargs=kwargs
        return {
            "symbol":chart["symbol"],"as_of":chart["as_of"],
            "edition":kwargs["edition"],"setup":"NO_SETUP","side":"NONE",
            "entry_state":"NO_SETUP","exit_basis":"NONE","rule_ids":[],
            "trade_action":None,"execution_allowed":False,
            "reason":"No confirmed signal.",
        }


class BookToChartLearningTests(unittest.TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.book=self.root/"private_books.db"
        self.memory=self.root/"study.db"
        ed=next(x for x in registry()["books"] if x["id"]=="TC2")
        pages=[""]*ed["pages"]
        pages[122]=(
            "Мудрец TC2 изучает фрактал и ангуляцию на закрытых свечах. "
            "Для подтверждения триггера необходимы следующие уже закрытые бары."
        )
        index_pages(self.book,"TC2",ed["sha256"],pages,declared_pages=ed["pages"])
        for model_id in ("gguf-alpha","gguf-beta"):
            study_batch(self.book,self.memory,"TC2",ReadingStudent(),
                        model_id=model_id,budget=1)

    def test_recall_is_isolated_by_student_model_and_book(self):
        alpha=recall(self.memory,"Мудрец фрактал ангуляция",
                     model_id="gguf-alpha",edition="TC2")
        self.assertEqual(len(alpha),1)
        self.assertEqual(alpha[0]["model_id"],"gguf-alpha")
        self.assertEqual(alpha[0]["source"],"STUDY:TC2: PDF 123")
        self.assertEqual(alpha[0]["status"],
                         "MODEL_SPECIFIC_SOURCE_DERIVED_UNVERIFIED")
        self.assertEqual(recall(self.memory,"Мудрец",model_id="gguf-other",
                                edition="TC2"),[])
        self.assertEqual(recall(self.memory,"Мудрец",model_id="gguf-alpha",
                                edition="NTD"),[])

    def test_chart_reasoning_consumes_studied_notes_only_from_exact_student(self):
        market=frozen_markets()["BTCUSDT"]
        spy=InspectingStudent()
        result=evaluate_historical_thought(
            "BTCUSDT",market["bars"],tick_size=market["tick_size"],
            bar_index=299,edition="TC2",model=spy,
            model_id="gguf-alpha",study_memory=self.memory)
        learned=spy.kwargs["study_lessons"]
        self.assertEqual(len(learned),1)
        self.assertTrue(learned[0]["excerpt"].startswith(
            "Непроверенный учебный вывод:"))
        self.assertEqual(spy.kwargs["model_id"],"gguf-alpha")
        self.assertEqual(result["same_model_unverified_book_lessons_used"],
                         ["STUDY:TC2: PDF 123"])
        self.assertFalse(result["execution_allowed"])
        spy2=InspectingStudent()
        evaluate_historical_thought(
            "BTCUSDT",market["bars"],tick_size=market["tick_size"],
            bar_index=299,edition="TC2",model=spy2,
            model_id="gguf-never-studied",study_memory=self.memory)
        self.assertEqual(spy2.kwargs["study_lessons"],[])
        with self.assertRaisesRegex(ValueError,"actual model"):
            evaluate_historical_thought(
                "BTCUSDT",market["bars"],tick_size=market["tick_size"],
                bar_index=299,edition="TC2",proposal={
                    "symbol":"BTCUSDT","as_of":"2025-01-01T00:00:00+00:00"},
                model_id="gguf-alpha",study_memory=self.memory)

    def test_ocr_correction_invalidates_old_notes_in_actual_reasoning_prompt(self):
        ed=next(x for x in registry()["books"] if x["id"]=="TC2")
        pages=[""]*ed["pages"]
        pages[122]="Исправленный текст про Мудрец фрактал ангуляция и порядок закрытых баров."
        index_pages(self.book,"TC2",ed["sha256"],pages,
                    declared_pages=ed["pages"])
        # Same source PDF bytes and same student, different OCR passage.
        self.assertEqual(recall(self.memory,"Мудрец фрактал ангуляция",
                                model_id="gguf-alpha",edition="TC2",
                                book_index=self.book),[])
        market=frozen_markets()["BTCUSDT"]
        spy=InspectingStudent()
        result=evaluate_historical_thought(
            "BTCUSDT",market["bars"],tick_size=market["tick_size"],
            bar_index=299,edition="TC2",model=spy,book_index=self.book,
            model_id="gguf-alpha",study_memory=self.memory)
        self.assertEqual(spy.kwargs["study_lessons"],[])
        self.assertTrue(result["book_study_notes_checked_against_current_source"])
        # A fresh real study operation can restore only the new passage.
        study_batch(self.book,self.memory,"TC2",ReadingStudent(),
                    model_id="gguf-alpha",budget=1)
        fresh=recall(self.memory,"Мудрец фрактал ангуляция",
                     model_id="gguf-alpha",edition="TC2",book_index=self.book)
        self.assertEqual(len(fresh),1)

    def test_chart_cli_does_not_borrow_another_models_book_memory(self):
        from williams_ai.chart_cli import main
        from williams_ai.study_cli import _model_identity
        weights=self.root/"selected.gguf";weights.write_bytes(b"SELECTED_TEST_MODEL")
        market=frozen_markets()["BTCUSDT"]
        bars=market["bars"][:350]
        path=self.root/"chart.json"
        path.write_text(json.dumps({"symbol":"BTCUSDT","tick_size":market["tick_size"],
                                    "as_of":bars[-1]["close_time"],
                                    "timeframes":{"H1":{"bars":bars}}}))
        calls=[]
        class Explainer:
            def answer(self,question,report,*,notes):
                calls.append(notes)
                return "Read-only mock explanation"
        argv=["chart_cli",str(path),"--question","Мудрец фрактал ангуляция",
              "--edition","TC2","--local-model",str(weights),
              "--study-memory",str(self.memory)]
        with patch("sys.argv",argv),patch(
                "williams_ai.local_model.LocalWilliamsModel",return_value=Explainer()),redirect_stdout(io.StringIO()):
            main()
        self.assertFalse(any(x["source"].startswith("STUDY:") for x in calls[-1]))
        study_batch(self.book,self.memory,"TC2",ReadingStudent(),
                    model_id=_model_identity(weights),budget=1)
        with patch("sys.argv",argv),patch(
                "williams_ai.local_model.LocalWilliamsModel",return_value=Explainer()),redirect_stdout(io.StringIO()):
            main()
        self.assertTrue(any(x["source"].startswith("STUDY:") for x in calls[-1]))

    def test_gguf_prompt_has_model_specific_notes_but_cannot_upgrade_book_rule(self):
        class Runtime:
            def __init__(self,**kwargs):self.calls=[]
            def create_chat_completion(self,**kwargs):
                self.calls.append(kwargs)
                chart=json.loads(kwargs["messages"][1]["content"])["chart"]
                answer={"symbol":chart["symbol"],"as_of":chart["as_of"],
                        "edition":"TC2","setup":"NO_SETUP","side":"NONE",
                        "trade_action":None,"execution_allowed":False,
                        "reason":"No confirmed signal."}
                return {"choices":[{"message":{"content":json.dumps(answer)}}]}
        path=self.root/"mock.gguf"
        path.write_bytes(b"MOCK_MODEL_NO_REAL_INFERENCE")
        model=LocalWilliamsModel(path,loader=Runtime)
        market=frozen_markets()["BTCUSDT"]
        rows=market["bars"][:350]
        chart=analyze_chart("BTCUSDT","H1",rows,
                            tick_size=market["tick_size"],
                            as_of=rows[-1]["close_time"])
        notes=recall(self.memory,"Мудрец",edition="TC2",
                     model_id="gguf-alpha")
        model.propose_chart_reasoning(
            chart,edition="TC2",study_lessons=notes,model_id="gguf-alpha")
        context=json.loads(model._model.calls[-1]["messages"][1]["content"])
        self.assertEqual(len(context["same_model_unverified_book_study_notes"]),1)
        self.assertIn("hypothesis_not_a_book_rule",
                      context["same_model_unverified_book_study_notes"][0])
        with self.assertRaisesRegex(ValueError,"Mixed-model"):
            model.propose_chart_reasoning(
                chart,edition="TC2",study_lessons=notes,model_id="gguf-beta")
        with self.assertRaisesRegex(ValueError,"Mixed-model"):
            model.propose_chart_reasoning(
                chart,edition="NTD",study_lessons=notes,model_id="gguf-alpha")


if __name__=="__main__":
    unittest.main()
