"""Offline cryptographic negative/positive source-signature tests, no secret uploads."""
from datetime import datetime,timezone,timedelta
import sqlite3
import unittest
import importlib.util

from williams_ai.signed_visual_reviews_v3 import (
    approval_message,verify_two_ed25519_course_approvals,
)


@unittest.skipUnless(importlib.util.find_spec("cryptography") is not None,
                     "Optional Ed25519 dependency unavailable; source-gold signing fails closed")
class SignedCourseReviewTests(unittest.TestCase):
    def setUp(self):
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        from cryptography.hazmat.primitives import serialization
        self.db=sqlite3.connect(":memory:")
        self.addCleanup(self.db.close)
        self.db.execute("""CREATE TABLE trusted_reviewer_keys_v3 (
            reviewer_id TEXT PRIMARY KEY,public_key_hex TEXT NOT NULL)""")
        self.db.execute("""CREATE TABLE course_expert_case_reviews_v3 (
            case_sha256 TEXT,pdf_page INTEGER,image_sha256 TEXT,source_sha256 TEXT,
            reviewer_id TEXT,reviewed_at TEXT,verdict TEXT,signature_hex TEXT)""")
        self.key1=Ed25519PrivateKey.generate()
        self.key2=Ed25519PrivateKey.generate()
        self.case="a"*64;self.image="b"*64;self.source="c"*64
        self.at=(datetime.now(timezone.utc)-timedelta(minutes=1)).isoformat()
        self.reviewers={"expert-alpha","expert-bravo"}
        for label,key in (("expert-alpha",self.key1),("expert-bravo",self.key2)):
            public=key.public_key().public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw).hex()
            self.db.execute("INSERT INTO trusted_reviewer_keys_v3 VALUES (?,?)",
                            (label,public))
            signed=approval_message(
                case_sha256=self.case,pdf_page=56,
                image_sha256=self.image,source_sha256=self.source,
                reviewer_id=label,reviewed_at=self.at,verdict="APPROVED")
            signature=key.sign(signed).hex()
            self.db.execute("INSERT INTO course_expert_case_reviews_v3 VALUES(?,?,?,?,?,?,?,?)",
                (self.case,56,self.image,self.source,label,self.at,"APPROVED",signature))

    def check(self,**changes):
        data=dict(case_sha256=self.case,pdf_page=56,
                  image_sha256=self.image,source_sha256=self.source,
                  expected_reviewers=self.reviewers)
        data.update(changes)
        return verify_two_ed25519_course_approvals(self.db,**data)

    def test_two_distinct_signers_approve_exact_page_and_case(self):
        result=self.check()
        self.assertTrue(result["source_attested_by_two_distinct_keys"])
        self.assertTrue(result["human_expertise_not_proven_by_cryptography"])
        self.assertFalse(result["trade_execution_allowed"])

    def test_missing_or_tampered_or_reused_key_fails(self):
        with self.assertRaisesRegex(ValueError,"TWO_VALID"):
            self.check(image_sha256="d"*64)
        self.db.execute("UPDATE course_expert_case_reviews_v3 SET signature_hex=? "
                        "WHERE reviewer_id='expert-alpha'",("0"*128,))
        with self.assertRaisesRegex(ValueError,"TWO_VALID"):
            self.check()
        self.db.execute("DELETE FROM course_expert_case_reviews_v3 "
                        "WHERE reviewer_id='expert-alpha'")
        self.db.execute("UPDATE trusted_reviewer_keys_v3 SET public_key_hex="
                        "(SELECT public_key_hex FROM trusted_reviewer_keys_v3 "
                        "WHERE reviewer_id='expert-alpha') "
                        "WHERE reviewer_id='expert-bravo'")
        with self.assertRaises(ValueError):
            self.check()

    def test_signed_visual_only_page_56_can_be_referenced_without_fake_ocr(self):
        from tempfile import TemporaryDirectory
        from pathlib import Path
        from copy import deepcopy
        import hashlib
        from cryptography.hazmat.primitives import serialization
        from williams_ai.book_rules import registry
        from williams_ai.book_index import index_pages
        from williams_ai.source_provenance_gate_v3 import check_private_expert_source_cases
        from tests.test_williams_ai_reasoning_v2_blocker_fixes import full_case,reseal
        case=deepcopy(full_case())
        course=next(b for b in registry()["books"] if b["id"]=="COURSE")
        case["edition"]="COURSE"
        case["pdf_page"]=56
        case["rule_id"]="COURSE.VISUAL.NESTED_ABC"
        case["source_sha256"]=course["sha256"]
        case["book_evidence"].update(
            {"edition":"COURSE","pdf_page":56,"source_sha256":course["sha256"],
             "excerpt":"Expert-reviewed description of nested degree wave labels on the diagram."})
        case["reviewers"][0]["reviewer_id"]="expert-alpha"
        case["reviewers"][1]["reviewer_id"]="expert-bravo"
        reseal(case)
        case_sha=__import__("williams_ai.reasoning_research_v2",fromlist=["digest"]).digest(
            {k:v for k,v in case.items() if k!="reviewers"})
        with TemporaryDirectory() as folder:
            root=Path(folder)
            db_path=root/"private_course.sqlite3"
            index_pages(db_path,"COURSE",course["sha256"],[""]*122,
                        declared_pages=122,source_type="PDF_TEXT_WITH_UNREVIEWED_OCR")
            (root/"page_images").mkdir()
            rel="page_images/page_056.jpg"
            raw=b"SYNTHETIC-WAVE-PAGE-56-NO-ORIGINAL-COPY"
            (root/rel).write_bytes(raw)
            image_sha=hashlib.sha256(raw).hexdigest()
            with sqlite3.connect(db_path) as db:
                db.execute("""CREATE TABLE course_pages(
                    pdf_page INTEGER PRIMARY KEY,page_image_path TEXT,
                    page_image_sha256 TEXT,source_sha256 TEXT,
                    semantic_review_status TEXT)""")
                db.execute("INSERT INTO course_pages VALUES(?,?,?,?,?)",
                    (56,rel,image_sha,course["sha256"],
                     "HUMAN_CHECKED_TEXT_AND_DIAGRAMS"))
                db.execute("""CREATE TABLE trusted_reviewer_keys_v3(
                    reviewer_id TEXT PRIMARY KEY,public_key_hex TEXT)""")
                db.execute("""CREATE TABLE course_expert_case_reviews_v3(
                    case_sha256 TEXT,pdf_page INTEGER,image_sha256 TEXT,
                    source_sha256 TEXT,reviewer_id TEXT,reviewed_at TEXT,
                    verdict TEXT,signature_hex TEXT)""")
                for reviewer,key in (("expert-alpha",self.key1),("expert-bravo",self.key2)):
                    public=key.public_key().public_bytes(
                        serialization.Encoding.Raw,
                        serialization.PublicFormat.Raw).hex()
                    db.execute("INSERT INTO trusted_reviewer_keys_v3 VALUES (?,?)",
                               (reviewer,public))
                    msg=approval_message(
                        case_sha256=case_sha,pdf_page=56,image_sha256=image_sha,
                        source_sha256=course["sha256"],reviewer_id=reviewer,
                        reviewed_at=self.at,verdict="APPROVED")
                    db.execute("INSERT INTO course_expert_case_reviews_v3 VALUES(?,?,?,?,?,?,?,?)",
                        (case_sha,56,image_sha,course["sha256"],
                         reviewer,self.at,"APPROVED",key.sign(msg).hex()))
            verified=check_private_expert_source_cases([case],db_path)
            self.assertEqual(verified["verified_case_count"],1)
            self.assertFalse(verified["case_receipts"][0]["page_text_found"])
            self.assertTrue(verified["case_receipts"][0]["signed_visual_interpretation"])
            (root/rel).write_bytes(b"TAMPERED")
            with self.assertRaisesRegex(ValueError,"IMAGE_BYTES_NOT_PINNED"):
                check_private_expert_source_cases([case],db_path)

    def test_future_fake_approval_is_rejected(self):
        with self.assertRaisesRegex(ValueError,"FUTURE_DATED"):
            approval_message(case_sha256=self.case,pdf_page=56,
                             image_sha256=self.image,source_sha256=self.source,
                             reviewer_id="expert-alpha",
                             reviewed_at=(datetime.now(timezone.utc)+timedelta(days=1)).isoformat(),
                             verdict="APPROVED")


if __name__=="__main__":
    unittest.main()
