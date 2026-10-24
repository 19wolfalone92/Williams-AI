"""Draft assistance never becomes invented human supervision."""
import copy
import hashlib
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest

from test_williams_ai_chart_supervision import chart_case
from test_williams_ai_private_lora import candidates
from williams_ai.book_rules import registry
from williams_ai.private_lora import checked_private_training_data, train_private_lora
from williams_ai.review_drafts import draft_review


class AnalystDraftTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.index = Path(self.temp.name) / "books.sqlite3"
        self.rows = []
        for row in candidates()[:4]:
            item = chart_case(row)
            item.update(answer="", review_status="PENDING_INDEPENDENT_REVIEW",
                        reviewer_id="review-required", independent_reviews=[])
            self.rows.append(item)
        books = {x["id"]: x for x in registry()["books"]}
        with sqlite3.connect(self.index) as c:
            c.execute("CREATE TABLE volumes(book TEXT,sha256 TEXT,total_pages INTEGER)")
            c.execute("CREATE TABLE excerpts(book TEXT,pdf_page INTEGER,content TEXT,evidence_status TEXT)")
            c.execute("CREATE TABLE page_provenance(book TEXT,pdf_page INTEGER,source_text_method TEXT,ocr_text_unreviewed INTEGER,text_chars INTEGER,normalized_text_sha256 TEXT)")
            for row in self.rows:
                ed, page = row["edition"], row["pdf_page"]
                c.execute("INSERT INTO volumes VALUES(?,?,?)", (ed, books[ed]["sha256"], books[ed]["pages"]))
                c.execute("INSERT INTO excerpts VALUES(?,?,?,?)",
                          (ed, page, "PRIVATE_ORIGINAL_PARAGRAPH_NEVER_EXPORT", "UNVERIFIED_EXTRACTION"))
                c.execute("INSERT INTO page_provenance VALUES(?,?,?,?,?,?)",
                          (ed, page, "TEST_EXTRACTION", int(ed == "COURSE"), 38, "a" * 64))

    def test_draft_preserves_original_pending_lesson_and_private_index_bytes(self):
        original = copy.deepcopy(self.rows[0])
        before = hashlib.sha256(self.index.read_bytes()).hexdigest()
        draft = draft_review(self.rows[0], index=self.index)
        self.assertEqual(self.rows[0], original)
        self.assertEqual(hashlib.sha256(self.index.read_bytes()).hexdigest(), before)
        self.assertFalse(draft["ready_for_weight_training"])
        self.assertEqual(draft["independent_reviews_completed"], 0)
        self.assertEqual(len(draft["alternative_scenarios_not_gold"]), 5)
        self.assertFalse(draft["execution_allowed"])

    def test_original_source_text_is_not_exported_or_declared_independently_verified(self):
        draft = draft_review(self.rows[0], index=self.index)
        self.assertNotIn("PRIVATE_ORIGINAL_PARAGRAPH_NEVER_EXPORT", json.dumps(draft))
        self.assertTrue(draft["source_receipt"]["requires_independent_semantic_and_figure_review"])
        self.assertFalse(draft["source_receipt"]["original_pdf_bytes_or_figures_rechecked_here"])

    def test_ocr_limitation_and_each_source_remain_explicit(self):
        for row in self.rows:
            result = draft_review(row, index=self.index)
            self.assertEqual(result["source_receipt"]["edition"], row["edition"])
            self.assertEqual(result["edition_gate_not_execution"]["book"], row["edition"])
            self.assertEqual(result["source_receipt"]["ocr_unreviewed"], row["edition"] == "COURSE")
            self.assertTrue(result["must_not_be_shown_to_blind_reviewers_before_their_answers"])

    def test_cross_edition_source_or_wrong_index_sha_cannot_generate_draft(self):
        row = copy.deepcopy(self.rows[0])
        row["edition"] = "NTD"
        with self.assertRaisesRegex(ValueError, "EDITION_OR_PAGE_MISMATCH"):
            draft_review(row, index=self.index)
        with sqlite3.connect(self.index) as c:
            c.execute("UPDATE volumes SET sha256=?", ("a" * 64,))
        with self.assertRaisesRegex(ValueError, "IDENTITY_MISMATCH"):
            draft_review(self.rows[0], index=self.index)

    def test_missing_page_or_changed_candle_fails_before_draft(self):
        row = copy.deepcopy(self.rows[0])
        row["market_context"]["bars"][10]["volume"] += 1
        with self.assertRaisesRegex(ValueError, "CHANGED_AFTER_REVIEW"):
            draft_review(row, index=self.index)
        with sqlite3.connect(self.index) as c:
            c.execute("DELETE FROM excerpts")
        with self.assertRaisesRegex(ValueError, "NO_USABLE_TEXT"):
            draft_review(self.rows[0], index=self.index)

    def test_draft_format_is_not_a_reviewed_dataset_and_never_reaches_gpu(self):
        draft = draft_review(self.rows[0], index=self.index)
        path = Path(self.temp.name) / "draft.jsonl"
        path.write_text(json.dumps(draft, ensure_ascii=False) + "\n")
        with self.assertRaisesRegex(ValueError, "UNRECOGNIZED_SUPERVISION_FIELDS"):
            checked_private_training_data(path)
        with self.assertRaisesRegex(ValueError, "UNRECOGNIZED_SUPERVISION_FIELDS"):
            train_private_lora(data=path, model_dir=Path(self.temp.name) / "model",
                               output_dir=Path(self.temp.name) / "adapter")

    def test_already_answered_or_reviewed_examples_are_not_modified(self):
        row = copy.deepcopy(self.rows[0])
        row["answer"] = "Someone's independent answer must remain untouched."
        with self.assertRaisesRegex(ValueError, "ONLY_UNLABELLED"):
            draft_review(row, index=self.index)

    def test_known_empty_scanned_page_remains_visually_blocked_without_substituting_rule_text(self):
        row = self.rows[-1]
        with sqlite3.connect(self.index) as c:
            c.execute("DELETE FROM excerpts WHERE book=?", (row["edition"],))
            c.execute("UPDATE page_provenance SET text_chars=0 WHERE book=?", (row["edition"],))
        draft = draft_review(row, index=self.index)
        self.assertTrue(draft["source_page_requires_visual_inspection"])
        self.assertFalse(draft["source_receipt"]["source_page_text_available"])
        self.assertIn("Текст страницы не извлечён", draft["draft_answer"])
        self.assertNotIn("Рабочая интерпретация существующей карточки", draft["draft_answer"])
        self.assertFalse(draft["ready_for_weight_training"])


if __name__ == "__main__":
    unittest.main()
