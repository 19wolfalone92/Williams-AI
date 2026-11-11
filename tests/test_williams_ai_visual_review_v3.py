"""Vision-v3 tests use synthetic image bytes only, never licensed book data."""
from pathlib import Path
import json
import tempfile
import unittest

from tests.test_williams_ai_course_private_ocr_visual import PrivateCourseOCRVisualTests
from williams_ai.visual_vision_review_v3 import (
    LocalQwenVLBackend, prepare_visual_jobs, review_visual_job,
    save_private_visual_proposals,
)


class VisionPipelineTests(unittest.TestCase):
    def setUp(self):
        self.fixture=PrivateCourseOCRVisualTests(
            methodName="test_122_page_footprint_and_visual_checksum")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.db=self.fixture.db

    @staticmethod
    def fake_visual_backend(image_path, job):
        data=Path(image_path).read_bytes()
        if not data or job["pdf_page"] not in range(1,123):
            raise AssertionError("ACTUAL_LOCAL_IMAGE_INPUT_REQUIRED")
        return {
            "page_summary":"Synthetic graph page, no expert interpretation.",
            "figure_observations":["Two arrows visible in mock illustration."],
            "visible_labels":["1","2","A"],
            "uncertainties":["Which line belongs to which degree is unknown."],
            "ocr_conflicts":["Mock OCR cannot establish vector orientation."],
            "grounding_notes":["Local bytes were opened for this exact page."],
        }

    def test_all_122_pages_with_page56_visual_only(self):
        jobs=prepare_visual_jobs(self.db)
        self.assertEqual(len(jobs),122)
        self.assertEqual([x["pdf_page"] for x in jobs],list(range(1,123)))
        target=jobs[55]
        self.assertEqual(target["pdf_page"],56)
        self.assertTrue(target["existing_unverified_diagrams"])
        self.assertFalse(target["usable_as_qlora_gold"])

    def test_image_backend_receives_image_and_outputs_unverified_claims(self):
        job=prepare_visual_jobs(self.db,pages=[56])[0]
        proposal=review_visual_job(self.db,job,self.fake_visual_backend)
        self.assertEqual(proposal["pdf_page"],56)
        self.assertFalse(proposal["usable_as_qlora_gold"])
        self.assertTrue(proposal["source_visual_ocr_conflicts_found"])
        self.assertEqual(proposal["review_status"],
            "VISION_MODEL_PROPOSAL_NOT_EXPERT_VERIFIED")
        forged={**job,"page_image_sha256":"0"*64}
        with self.assertRaisesRegex(ValueError,"STALE_OR_FORGED"):
            review_visual_job(self.db,forged,self.fake_visual_backend)
        def bad_backend(_image,_job):
            return {"page_summary":"This makes money"}
        with self.assertRaisesRegex(ValueError,"EXACT_VISION"):
            review_visual_job(self.db,job,bad_backend)

    def test_private_proposals_are_0600_and_never_auto_approved(self):
        path=self.fixture.root/"visual_proposals.jsonl"
        summary=save_private_visual_proposals(
            self.db,self.fake_visual_backend,path,pages=[47,56,71])
        self.assertEqual(summary["pages_processed"],3)
        self.assertEqual(summary["independent_expert_approval_count"],0)
        self.assertEqual(path.stat().st_mode & 0o077,0)
        rows=[json.loads(x) for x in path.read_text().splitlines()]
        self.assertEqual([x["pdf_page"] for x in rows],[47,56,71])
        self.assertFalse(any(x["usable_as_qlora_gold"] for x in rows))


if __name__=="__main__":
    unittest.main()
