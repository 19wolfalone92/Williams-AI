"""Offline cryptographic negative/positive source-signature tests, no secret uploads."""
from datetime import datetime,timezone,timedelta
import sqlite3
import unittest

from williams_ai.signed_visual_reviews_v3 import (
    approval_message,verify_two_ed25519_course_approvals,
)


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

    def test_future_fake_approval_is_rejected(self):
        with self.assertRaisesRegex(ValueError,"FUTURE_DATED"):
            approval_message(case_sha256=self.case,pdf_page=56,
                             image_sha256=self.image,source_sha256=self.source,
                             reviewer_id="expert-alpha",
                             reviewed_at=(datetime.now(timezone.utc)+timedelta(days=1)).isoformat(),
                             verdict="APPROVED")


if __name__=="__main__":
    unittest.main()
