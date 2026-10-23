"""No real model in CI: validate the orchestrated *contracts*, not book mastery."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from williams_ai.bootcamp import validate_book_readiness,bootcamp
from williams_ai.book_index import index_pages
from williams_ai.book_rules import registry
from williams_ai.causal_paper_lab import frozen_markets


class MockStudent:
    def __init__(self):
        self.sources=[]
        self.scenes=[]
    def study_passage(self,text,*,edition,page,rule_cards):
        self.sources.append((edition,page))
        return json.dumps({
            "summary":"Источник создаёт учебный пример без торговых полномочий",
            "claims":[{"statement":"Источник виден","support":text[:60]}],
            "questions":["Что нельзя объявлять исполнением на основании текста?"]
        },ensure_ascii=False)
    def answer_exam_case(self,case,*,prior_lessons=None):
        # A fake B response tests orchestration only; not true GGUF ability.
        return {"choice":"B","reason":"Requires a closed confirmation"}
    def propose_chart_reasoning(self,chart,*,edition,primary_pages,prior_lessons):
        self.scenes.append((edition,chart["as_of"],primary_pages,prior_lessons))
        return {"symbol":chart["symbol"],"as_of":chart["as_of"],
                "edition":edition,"setup":"NO_SETUP","side":"NONE",
                "entry_state":"NO_SETUP","exit_basis":"NONE","rule_ids":[],
                "trade_action":None,"execution_allowed":False}


class BootcampContracts(unittest.TestCase):
    def setUp(self):
        self.tmp=TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.source=self.root/"private_books.db"
        self.study=self.root/"private_study.db"
        self.coach=self.root/"private_feedback.db"
        for item in registry()["books"]:
            pages=[""]*item["pages"]
            pages[5]=(
                item["id"]+" Аллигатор AO AC фрактал мудрец и волновой цикл. "
                "История должна быть закрыта до подтверждения сигнала. "
                "Данные биржи требуются для доказательства исполнения."
            )
            index_pages(self.source,item["id"],item["sha256"],pages,
                        declared_pages=item["pages"])

    def test_all_four_originals_indexed_but_no_book_mastery_claim(self):
        state=validate_book_readiness(self.source)
        self.assertEqual(len(state["indexed_editions"]),4)
        self.assertEqual(state["total_expected_pdf_pages"],642)
        self.assertEqual(state["pages_with_extracted_or_ocr_text"],4)
        self.assertTrue(state["readiness_for_partial_local_RAG"])
        self.assertTrue(state["issues"])
        self.assertIsNone(state["verified_model_book_mastery_percent"])
        self.assertFalse(state["execution_allowed"])

    def test_mismatched_or_absent_source_blocks_full_cycle(self):
        from williams_ai.bootcamp import validate_book_readiness
        with TemporaryDirectory() as tmp:
            import sqlite3
            short=Path(tmp)/"partial_books.db"
            item=registry()["books"][0]
            index_pages(short,item["id"],item["sha256"],[""]*item["pages"],
                        declared_pages=item["pages"])
            status=validate_book_readiness(short)
            self.assertFalse(status["readiness_for_partial_local_RAG"])
            with self.assertRaises(ValueError):
                bootcamp(MockStudent(),model_id="gguf-mock",
                         book_index=short,study_memory=self.study,
                         coaching_memory=self.coach,
                         historical_markets={"ETHUSDT":frozen_markets()["ETHUSDT"]},
                         historical_editions=("TC2",),checkpoints=(299,))

    def test_real_pipeline_paths_stay_private_and_no_trade(self):
        mock=MockStudent()
        market={"ETHUSDT":frozen_markets()["ETHUSDT"]}
        output=bootcamp(
            mock,model_id="gguf-mock-sha",
            book_index=self.source,
            study_memory=self.study,
            coaching_memory=self.coach,
            study_budget_per_book=1,study_rounds=1,
            practice_cases=5,
            historical_markets=market,
            historical_editions=("TC2",),
            checkpoints=(299,))
        self.assertEqual(output["source_study"]["new_chunks"],4)
        self.assertEqual(len(mock.sources),4)
        self.assertEqual(output["adaptive_practice"]["cases"],5)
        self.assertEqual(output["public_curated_diagnostic_exam"]["total_cases"],10)
        self.assertTrue(output["public_curated_diagnostic_exam"]["not_independent_blind_exam"])
        self.assertEqual(output["historical_chart_reasoning"]["tested_cases"],1)
        self.assertEqual(output["historical_chart_reasoning"]["unsupported_claim_cases"],0)
        self.assertEqual(output["historical_chart_reasoning"]["abstained_or_unknown_cases"],1)
        self.assertIsNone(output["full_book_mastery_percent"])
        self.assertFalse(output["actual_language_model_weights_changed"])
        self.assertFalse(output["execution_allowed"])
        self.assertIsNone(output["trade_action"])
        self.assertTrue(self.study.is_file())
        self.assertTrue(self.coach.is_file())
        self.assertEqual(mock.scenes[0][0],"TC2")
        self.assertTrue(mock.scenes[0][2])
        self.assertTrue(all(p["source"].startswith("TC2: PDF ") for p in mock.scenes[0][2]))
        self.assertNotIn("Аллигатор AO AC",json.dumps(output,ensure_ascii=False))


if __name__=="__main__":
    unittest.main()
