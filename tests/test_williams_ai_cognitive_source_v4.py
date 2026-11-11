"""Root source-provenance regression after integrating v4 into current main."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from williams_ai.book_index import index_pages
from williams_ai.book_rules import registry
from williams_ai.grounded_inference import verified_private_book_pages
from williams_ai.cognitive_engine_v4 import CognitiveResearchEngine
from williams_ai.cognitive_memory_v4 import CognitiveMemory
from tests.test_williams_ai_cognitive_v4 import facts, now


class CourseBackend:
    def __init__(self):
        self.seen_sources=[]
    def __call__(self, stage, packet):
        p=packet.get("context",packet)
        if stage=="hypothesis":
            self.seen_sources=p["original_private_pages_unverified_extraction"]
            return {"claim":"The source page has unverified OCR and needs interpretation.",
                    "alternative":"The scanned figure may have another valid meaning.",
                    "evidence_ids":["H1_AO"],"counterevidence_ids":["H4_RANGE"],
                    "rule_ids":["COURSE.HUMP"],
                    "invalidation":"An authoritative human diagram review may disprove the hypothesis.",
                    "confidence":0.3}
        if stage=="skeptic":
            return {"counterhypothesis":"The OCR does not determine the actual figure semantics.",
                    "challenged_fact_ids":["H1_AO"],
                    "missing_information":["Independent figure interpretation is unavailable."],
                    "rule_conflicts":[],
                    "suggestion":"WAIT"}
        return {"stance":"WAIT","direction":"UNDETERMINED",
                "supported_fact_ids":[],"reason":"Only a book search hypothesis; no market trade.",
                "confidence":0.2}


class SourceProvenanceV4Tests(unittest.TestCase):
    def test_private_course_ocr_is_marked_and_pinned_in_full_cognitive_flow(self):
        with TemporaryDirectory() as directory:
            root=Path(directory)
            source=root/"private_original_books.sqlite3"
            course=next(x for x in registry()["books"] if x["id"]=="COURSE")
            pages=[""]*course["pages"]
            pages[52]="Hump correction structure and its uncertain wave count. "*4
            index_pages(source,"COURSE",course["sha256"],pages,
                        declared_pages=course["pages"],
                        source_type="PDF_TEXT_WITH_UNREVIEWED_OCR")
            pulled=verified_private_book_pages(source,"COURSE",
                        question="Hump correction structure",limit=1)
            self.assertTrue(pulled)
            self.assertEqual(pulled[0]["status"],
                "PRIVATE_OCR_UNVERIFIED_NEEDS_PAGE_IMAGE_REVIEW")
            backend=CourseBackend()
            engine=CognitiveResearchEngine(
                memory=CognitiveMemory(root/"cognitive_memory.sqlite3"),
                backend=backend,model_id="gguf-course-test-model",
                run_id="course-research-lab",book_index=source)
            report=engine.reason("Hump",edition="COURSE",
                as_of=now(),closed_facts=facts())
            self.assertEqual(report["effective_stance"],"WAIT")
            self.assertTrue(backend.seen_sources)
            self.assertEqual(backend.seen_sources[0]["status"],
                "PRIVATE_OCR_UNVERIFIED_NEEDS_PAGE_IMAGE_REVIEW")
            self.assertFalse(report["execution_allowed"])

    def test_source_book_digest_mismatch_cannot_enter_legacy_or_new_brain(self):
        with TemporaryDirectory() as directory:
            root=Path(directory)
            source=root/"wrong_books.sqlite3"
            course=next(x for x in registry()["books"] if x["id"]=="COURSE")
            pages=[""]*course["pages"]
            pages[22]="Hump technical note for scanned original."*3
            index_pages(source,"COURSE",course["sha256"],pages,
                        declared_pages=course["pages"],
                        source_type="PDF_TEXT_WITH_UNREVIEWED_OCR")
            import sqlite3
            with sqlite3.connect(source) as db:
                db.execute("UPDATE volumes SET sha256=? WHERE book='COURSE'",
                           ("0"*64,))
            with self.assertRaisesRegex(ValueError,"fingerprint"):
                verified_private_book_pages(source,"COURSE",
                      question="Hump",limit=1)


if __name__=="__main__":
    unittest.main()
