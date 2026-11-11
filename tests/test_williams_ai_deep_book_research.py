import base64
from hashlib import sha256
import json
import unittest

from scripts.ai_run_private_deep_kaggle import worker_source, GPU_RUNTIME_LIMIT_SECONDS, POLL_INTERVAL_SECONDS, POLL_ATTEMPTS
from williams_ai.book_rules import registry
from williams_ai.deep_book_research import checked_payload, practice_curriculum, corrective_course_practice
from scripts.ai_continue_private_deep_kaggle import checked_own_previous_result


def example_payload():
    books = {r["id"]: r for r in registry()["books"]}
    manifest = {"status": "SOURCE_ONLY_NOT_EXPERT_APPROVED", "page_boundary_embargo": 2,
        "all_figures_understood": False, "market_training_or_profitability_verified": False,
        "book_registry": {e: {"sha256": r["sha256"], "pages": r["pages"]} for e, r in books.items()}, "splits": {}}
    files = {}
    for split, page in (("train", 10), ("validation", 20)):
        rows = [{"edition": e, "pdf_page": page, "source_sha256": b["sha256"],
                 "text": "This is a synthetic source fragment for testing only.",
                 "extraction_status": "UNVERIFIED_OCR" if e == "COURSE" else "PDF_TEXT_NOT_EXPERT_REVIEWED"}
                for e, b in books.items()]
        raw = "\n".join(json.dumps(r) for r in rows).encode()
        files[split] = base64.b64encode(raw).decode()
        manifest["splits"][split] = {"sha256": sha256(raw).hexdigest(), "rows": 4, "books": {e: 1 for e in books}}
    manifest["splits"]["test"] = {"books": {e: 1 for e in books}}
    return {"format": "PRIVATE_SOURCE_TRAIN_VALIDATION_ONLY_V1", "manifest": manifest, "files": files,
        "withheld_test_page_ids": {e: [30] for e in books}, "audit": {},
        "test_texts_in_cloud_payload": False, "execution_allowed": False}


class DeepResearchTests(unittest.TestCase):
    def test_controller_waits_beyond_the_actual_gpu_budget(self):
        # Regression: 220*20 seconds formerly timed out before the 5400-second
        # GPU allocation and falsely reported an archive that was not due yet.
        self.assertGreater(POLL_ATTEMPTS * POLL_INTERVAL_SECONDS, GPU_RUNTIME_LIMIT_SECONDS)

    def test_checked_originals_keep_test_texts_absent_and_preserve_ocr_limitations(self):
        result = checked_payload(example_payload())
        self.assertEqual(len(result["train"]), 4)
        self.assertNotIn("test", result)
        self.assertFalse(result["audit"]["original_test_texts_sent_to_gpu"])

    def test_changed_original_source_bytes_are_rejected(self):
        data = example_payload()
        data["files"]["train"] = base64.b64encode(b"changed").decode()
        with self.assertRaisesRegex(ValueError, "SHA_MISMATCH"):
            checked_payload(data)

    def test_test_text_or_nearby_page_leakage_rejected(self):
        data = example_payload()
        data["files"]["test"] = "not allowed"
        with self.assertRaises(ValueError):
            checked_payload(data)
        data = example_payload()
        data["withheld_test_page_ids"]["TC1"] = [22]
        with self.assertRaisesRegex(ValueError, "EMBARGO"):
            checked_payload(data)

    def test_private_worker_has_no_credential_or_market_outcome_input(self):
        text = worker_source("a" * 40, b"private training only", b"as-of questions")
        self.assertNotIn("KAGGLE_API_TOKEN", text)
        self.assertNotIn("WITHHELD_OUTCOME_KEYS", text)
        self.assertIn('shutil.rmtree(repo/"examples")', text)
        compile(text, "private_worker", "exec")

    def test_practice_is_diverse_and_keeps_numeric_values_disjoint(self):
        train, validation, report = practice_curriculum()
        self.assertGreater(len(train), 250)
        self.assertEqual(set(validation), {"TC1", "NTD", "TC2", "COURSE"})
        self.assertTrue(report["synthetic_arithmetic_not_human_chart_labels"])
        self.assertFalse(set(r["question"] for r in train) & set(r["question"] for rr in validation.values() for r in rr))
        self.assertGreater(len({r["answer"] for r in train}), 200)

    def test_corrective_course_uses_counterexamples_and_separate_parameter_values(self):
        cases = corrective_course_practice()
        self.assertEqual(len(cases["train"]), 128)
        self.assertEqual(len(cases["validation"]), 32)
        self.assertFalse({r["question"] for r in cases["train"]} & {r["question"] for r in cases["validation"]})
        humps = [r for r in cases["train"] if r["rule_id"] == "COURSE.HUMP"]
        self.assertTrue(any(r["answer"].startswith("Нет.") for r in humps))
        self.assertTrue(any("неперекрытия соблюдён" in r["answer"] for r in humps))

    def test_warmstart_worker_is_compilable_and_never_contains_kaggle_secret(self):
        config = {"url": "https://example.invalid/private-expiring-output", "archive_sha256": "a"*64,
                  "adapter_sha256": "b"*64}
        source = worker_source("a"*40, b"private training only", b"as-of questions", initialization=config)
        self.assertNotIn("KAGGLE_API_TOKEN", source)
        self.assertIn("OWN_RANK16_INITIAL_ADAPTER_SHA_MISMATCH", source)
        compile(source, "warmstart", "exec")
