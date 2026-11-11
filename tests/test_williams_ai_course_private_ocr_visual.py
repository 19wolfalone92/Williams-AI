"""Synthetic fixtures: safe retrieval of personal OCR and page-picture links.

NO original Course PDF pages, text, diagrams or OCR included in Git.
"""
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from williams_ai.book_index import index_pages
from williams_ai.book_rules import registry
from williams_ai.course_visual_private import course_visual_refs, course_visual_audit, combine_with_existing_private_books
from williams_ai.grounded_inference import verified_private_book_pages


class PrivateCourseOCRVisualTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root=Path(self.tmp.name)
        self.root=root
        self.db=root/"private_book_index.sqlite3"
        source=next(x for x in registry()["books"] if x["id"]=="COURSE")
        self.sha=source["sha256"]
        content="Fractal market geometry, Elliott candidate waves and hypothetical corrections. "*3
        pages=[""]*122
        pages[17]=content
        index_pages(self.db,"COURSE",self.sha,pages,
                    declared_pages=122,source_type="PDF_TEXT_WITH_UNREVIEWED_OCR")
        (root/"page_images").mkdir()
        for n in range(1,123):
            path=root/"page_images"/f"page_{n:03d}.jpg"
            path.write_bytes(f"SYNTHETIC-PAGE-{n}".encode())
        with sqlite3.connect(self.db) as db:
            db.execute("""CREATE TABLE course_pages(
                pdf_page INTEGER PRIMARY KEY, section TEXT,
                page_image_path TEXT,page_image_sha256 TEXT,source_sha256 TEXT,
                ocr_text TEXT,ocr_confidence REAL,ocr_word_count INTEGER,
                ocr_rotation INTEGER,source_status TEXT,semantic_review_status TEXT)""")
            db.execute("""CREATE TABLE course_visuals(
                figure_id TEXT PRIMARY KEY,pdf_page INTEGER,description TEXT,
                page_image_path TEXT,image_sha256 TEXT,
                associated_paragraph_indices TEXT,association_method TEXT,
                semantic_status TEXT)""")
            db.execute("""CREATE TABLE course_paragraphs(
                pdf_page INTEGER, paragraph_index INTEGER, body TEXT,
                bbox_json TEXT, ocr_confidence REAL)""")
            db.execute("INSERT INTO course_paragraphs VALUES(?,?,?,?,?)",
                       (18,0,"Synthetic illustration description","[1,2,3,4]",96.0))
            for n in range(1,123):
                rel=f"page_images/page_{n:03d}.jpg"
                image_hash=hashlib.sha256((root/rel).read_bytes()).hexdigest()
                db.execute("INSERT INTO course_pages VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                           (n,"TEST ONLY",rel,image_hash,self.sha,"",95.0,50,0,
                            "OCR_IMAGE_LAYOUT_UNVERIFIED","NOT_INDEPENDENTLY_VERIFIED"))
            rel="page_images/page_018.jpg"
            image_hash=hashlib.sha256((root/rel).read_bytes()).hexdigest()
            db.execute("INSERT INTO course_visuals VALUES(?,?,?,?,?,?,?,?)",
                       ("COURSE:P018:PAGE_VISUAL",18,
                        "Two illustrative price bars with different open and close positions.",
                        rel,image_hash,"[0]",
                        "PAGE_PROVENANCE_AND_OCR_LEXICAL_OVERLAP",
                        "UNVERIFIED_VISUAL_DESCRIPTION"))
            rel="page_images/page_056.jpg"
            image_hash=hashlib.sha256((root/rel).read_bytes()).hexdigest()
            db.execute("INSERT INTO course_visuals VALUES(?,?,?,?,?,?,?,?)",
                       ("COURSE:P056:PAGE_VISUAL",56,
                        "Composite Elliott wave-count diagram with nested degrees and impulse labels.",
                        rel,image_hash,"[]",
                        "PAGE_PROVENANCE_AND_OCR_LEXICAL_OVERLAP",
                        "UNVERIFIED_VISUAL_DESCRIPTION"))

        (root/"VISUAL_DRAFTS_7.json").write_text(json.dumps([{
            "page":18,"source_pdf_sha256":self.sha,
            "page_image_sha256":hashlib.sha256(
                (root/"page_images/page_018.jpg").read_bytes()).hexdigest(),
            "geometry_observations":["A possible directional line must not be treated as a verified fill."],
            "independently_verified":False,"use_as_expert_gold":False
        }]),encoding="utf-8")

    def test_122_page_footprint_and_visual_checksum(self):
        report=course_visual_audit(self.db)
        self.assertEqual(report["pages_with_images"],122)
        self.assertEqual(report["illustrated_page_annotations"],2)
        self.assertFalse(report["use_as_verified_training_gold"])
        ref=course_visual_refs(self.db,18)
        self.assertEqual(len(ref),1)
        self.assertFalse(ref[0]["expert_figure_interpretation_verified"])
        self.assertTrue(ref[0]["image_relative_path"].startswith("page_images/"))
        self.assertEqual(course_visual_refs(self.db,19),[])

    def test_ocr_retrieval_keeps_image_link_and_ocr_warning(self):
        result=verified_private_book_pages(self.db,"COURSE",
                         question="Fractal Elliott candidate waves",limit=2)
        self.assertTrue(result)
        self.assertEqual(result[0]["pdf_page"],18)
        self.assertEqual(result[0]["status"],
                         "PRIVATE_OCR_UNVERIFIED_NEEDS_PAGE_IMAGE_REVIEW")
        self.assertEqual(len(result[0]["visual_refs"]),1)
        self.assertTrue(result[0]["visual_meaning_not_authoritatively_checked"])
        self.assertEqual(len(result[0]["private_visual_drafts"]),1)
        self.assertFalse(result[0]["private_visual_drafts"][0]["use_as_expert_gold"])

    def test_diagram_only_page_remains_searchable_without_fabricating_ocr(self):
        from williams_ai.course_visual_private import search_course_visual_descriptions
        visuals=search_course_visual_descriptions(
            self.db,"composite Elliott diagram",limit=2)
        self.assertTrue(any(x["pdf_page"]==56 for x in visuals))
        self.assertEqual(visuals[0]["status"],
              "PRIVATE_VISUAL_DESCRIPTION_UNVERIFIED_NEEDS_PAGE_IMAGE_REVIEW")
        found=verified_private_book_pages(self.db,"COURSE",
                     question="composite Elliott diagram",limit=2)
        self.assertTrue(any(x["pdf_page"]==56 and
               x["status"]=="PRIVATE_VISUAL_DESCRIPTION_UNVERIFIED_NEEDS_PAGE_IMAGE_REVIEW"
               for x in found))

    def test_ocr_course_and_figure_description_reach_local_text_only_model(self):
        from williams_ai.local_model import LocalWilliamsModel
        from williams_ai.causal_paper_lab import frozen_markets
        from williams_ai.reasoning_trial import evaluate_historical_thought
        weights=self.root/"synthetic.gguf"
        weights.write_bytes(b"TEST-MODEL-NEVER-TRAINED")
        class Engine:
            def __init__(self,**kwargs):
                self.input=None
            def create_chat_completion(self,**kwargs):
                self.input=kwargs
                prompt=json.loads(kwargs["messages"][1]["content"])
                output={"symbol":prompt["chart"]["symbol"],
                        "as_of":prompt["chart"]["as_of"],
                        "edition":prompt["edition"],
                        "setup":"NO_SETUP","side":"NONE",
                        "entry_state":"NO_SETUP","exit_basis":"NONE",
                        "rule_ids":[],"trade_action":None,
                        "execution_allowed":False,
                        "reason":"No corroborated market setup or actual orders."}
                return {"choices":[{"message":{"content":json.dumps(output)}}]}
        model=LocalWilliamsModel(weights,loader=Engine)
        history=frozen_markets()["ETHUSDT"]
        result=evaluate_historical_thought("ETHUSDT",history["bars"],
                   tick_size=history["tick_size"],bar_index=300,
                   edition="COURSE",model=model,book_index=self.db)
        payload=json.loads(model._model.input["messages"][1]["content"])
        self.assertEqual(payload["edition"],"COURSE")
        self.assertTrue(payload["primary_book_pages"])
        source=payload["primary_book_pages"][0]
        self.assertEqual(source["source_status"],
                         "PRIVATE_OCR_UNVERIFIED_NEEDS_PAGE_IMAGE_REVIEW")
        self.assertTrue(source["not_expert_verified"])
        self.assertTrue(source["visual_refs"])
        self.assertEqual(source["visual_refs"][0]["visual_status"],
                         "UNVERIFIED_DESCRIPTIVE_ONLY")
        self.assertEqual(source["visual_draft_observations"][0]["status"],
                         "UNVERIFIED_VISUAL_OBSERVATION")
        self.assertFalse(source["visual_draft_observations"][0]["gold_label"])
        self.assertFalse(result["private_book_passages_exported"])

    def test_merge_into_copy_does_not_erase_other_books(self):
        other=self.root/"original_three_books.sqlite3"
        tc2=next(x for x in registry()["books"] if x["id"]=="TC2")
        text=[""]*tc2["pages"]
        text[40]="A locally indexed, synthetic TC2 research reference (not original text)."
        index_pages(other,"TC2",tc2["sha256"],text,declared_pages=tc2["pages"])
        output=self.root/"all_four_private.sqlite3"
        report=combine_with_existing_private_books(
            other,self.db,destination=output)
        self.assertEqual(report["course_page_images_verified"],122)
        self.assertIn("TC2",report["editions"])
        self.assertIn("COURSE",report["editions"])
        self.assertTrue(output.is_file())
        self.assertEqual(course_visual_audit(output)["pages_with_images"],122)
        with sqlite3.connect(other) as old:
            editions={x[0] for x in old.execute("SELECT book FROM volumes")}
        self.assertEqual(editions,{"TC2"})
        with self.assertRaisesRegex(ValueError,"NEW_COMBINED"):
            combine_with_existing_private_books(other,self.db,destination=output)

    def test_worksheets_are_never_treated_as_tactical_rules(self):
        from williams_ai.course_page_semantics_v3 import (
            private_course_page_role, require_not_an_empty_rule_template)
        for page in (90,120,121):
            role=private_course_page_role(page)
            self.assertTrue(role["blank_form_not_a_trading_rule"])
            self.assertFalse(role["automatic_market_gold_allowed"])
            with self.assertRaisesRegex(ValueError,"BLANK_WORKSHEET"):
                require_not_an_empty_rule_template(page)
        self.assertEqual(private_course_page_role(56)["role"],
                         "NESTED_ELLiOTT_DEGREES_DIAGRAM")
        self.assertEqual(private_course_page_role(89)["role"],
                         "UNKNOWN_UNREVIEWED")

    def test_global_audit_checks_all_122_images_not_only_annotated(self):
        unannotated=self.root/"page_images/page_119.jpg"
        unannotated.write_bytes(b"CORRUPTED UNANNOTATED PDF PAGE")
        with self.assertRaisesRegex(ValueError,"COURSE_PAGE_IMAGE_HASH"):
            course_visual_audit(self.db)

    def test_corrupt_image_is_rejected_without_inventing_figures(self):
        picture=self.root/"page_images/page_018.jpg"
        picture.write_bytes(b"NOT THE SAME IMAGE")
        with self.assertRaisesRegex(ValueError,"SHA256_MISMATCH"):
            course_visual_refs(self.db,18)

    def test_reject_forged_book_volume(self):
        with sqlite3.connect(self.db) as db:
            db.execute("UPDATE volumes SET sha256=? WHERE book='COURSE'",("0"*64,))
        with self.assertRaisesRegex(ValueError,"NOT_THE_PINNED"):
            course_visual_audit(self.db)


if __name__=="__main__":
    unittest.main()
