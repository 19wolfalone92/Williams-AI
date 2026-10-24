"""Private original books are studied by ONE model, never certified by a mock."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from williams_ai.book_rules import registry
from williams_ai.book_index import index_pages
from williams_ai.private_curriculum import complete_private_readthrough


class Reader:
    def __init__(self):
        self.calls=[]
    def study_passage(self,text,*,edition,page,rule_cards):
        self.calls.append((edition,page))
        return json.dumps({
            "summary":"Исходный текст книги о волнах и фракталах, который требует проверки.",
            "claims":[{"statement":"Цитата из текста","support":text[:65]}],
            "questions":["Сможет ли модель интерпретировать реальные свечи?"],
        },ensure_ascii=False)


class ReadThroughTests(unittest.TestCase):
    def setUp(self):
        self.tmp=TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root=Path(self.tmp.name)
        self.bookdb=root/"source.db"
        self.notes=root/"notes.db"
        for b in registry()["books"]:
            pages=[""]*b["pages"]
            pages[8]="Динамика волн и фракталов на разных таймфреймах. "*3
            pages[9]="Фракталы и движение цены должны быть подтверждены. "*3
            index_pages(self.bookdb,b["id"],b["sha256"],pages,
                        declared_pages=b["pages"])

    def test_resume_processes_unique_chunks_and_flags_unreadable_pdf_pages(self):
        reader=Reader()
        first=complete_private_readthrough(
            self.bookdb,self.notes,reader,model_id="gguf-pinned-test",
            batch_budget=1,rounds=1)
        self.assertEqual(first["newly_studied_chunks"],4)
        self.assertEqual(first["indexed_chunks_remaining_for_model"],4)
        self.assertFalse(first["execution_allowed"])
        second=complete_private_readthrough(
            self.bookdb,self.notes,reader,model_id="gguf-pinned-test",
            batch_budget=1,rounds=3)
        self.assertEqual(second["newly_studied_chunks"],4)
        self.assertEqual(second["indexed_chunks_remaining_for_model"],0)
        self.assertEqual(second["readthrough_status"],
                         "BLOCKED_SOURCE_SCAN_OR_IMAGE_GAPS")
        self.assertTrue(all(not b["text_pass_completed_for_this_specific_model"]
                            for b in second["books"]))
        again=complete_private_readthrough(
            self.bookdb,self.notes,reader,model_id="gguf-pinned-test",
            batch_budget=1,rounds=1)
        self.assertEqual(again["newly_studied_chunks"],0)
        self.assertEqual(len(reader.calls),8)
        self.assertIsNone(second["book_mastery_percent"])
        self.assertFalse(second["real_weight_updates_occurred"])

    def test_second_model_does_not_reuse_prior_ones_readthrough(self):
        reader=Reader()
        a=complete_private_readthrough(
            self.bookdb,self.notes,reader,model_id="one",batch_budget=1,rounds=1)
        b=complete_private_readthrough(
            self.bookdb,self.notes,reader,model_id="two",batch_budget=1,rounds=1)
        self.assertEqual(a["newly_studied_chunks"],4)
        self.assertEqual(b["newly_studied_chunks"],4)
        self.assertNotEqual(a["readthrough_protocol_sha256"],
                            b["readthrough_protocol_sha256"])

    def test_missing_edition_wrong_db_and_bad_budget_fail_closed(self):
        with TemporaryDirectory() as d:
            p=Path(d)/"missing.db"
            with self.assertRaisesRegex(ValueError,"ALL_FOUR"):
                complete_private_readthrough(
                    self.bookdb,self.notes,Reader(),model_id="model-x",
                    rounds=1,batch_budget=0)
            with self.assertRaisesRegex(ValueError,"INVALID_LOCAL"):
                complete_private_readthrough(
                    self.bookdb,self.bookdb,Reader(),model_id="model-x")
            with self.assertRaisesRegex(ValueError,"ALL_FOUR"):
                complete_private_readthrough(
                    p,self.notes,Reader(),model_id="model-x")

if __name__=="__main__":
    unittest.main()
