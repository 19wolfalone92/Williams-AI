"""No real experts or labels are created by these tests."""
import copy
import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from williams_ai.v7_blind_review_pack import (
    build_packet,validate_packet,validate_issues,create_private
)

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/"knowledge/WILLIAMS_V7_REASONING_REVIEW_CANDIDATES_20261117.json"
ISSUES=ROOT/"research/WILLIAMS_V7_SOURCE_CONFLICTS_20261117.json"

class BlindReviewPacketTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source=json.loads(SOURCE.read_text(encoding="utf-8"))
        cls.issues=json.loads(ISSUES.read_text(encoding="utf-8"))
    def test_two_deterministic_blind_slots_thirty_unlabelled(self):
        a=build_packet(self.source,"A")
        b=build_packet(self.source,"B")
        self.assertEqual(validate_packet(a,self.source)["review_items"],30)
        self.assertEqual(validate_packet(b,self.source)["review_items"],30)
        self.assertEqual(len(a["items"]),len(b["items"]))
        self.assertEqual({x["review_item_id"] for x in a["items"]},
                         {x["review_item_id"] for x in b["items"]})
        self.assertEqual(a,build_packet(self.source,"A"))
        self.assertNotEqual(a["reviewer_slot_not_identity"],
                            b["reviewer_slot_not_identity"])
        for obj in (a,b):
            self.assertEqual(obj["independent_human_reviews_received"],0)
            self.assertFalse(obj["expert_identity_vetted"])
            self.assertFalse(obj["sft_approved"])
            for item in obj["items"]:
                self.assertNotIn("decision",item)
                self.assertNotIn("answer",item)
                self.assertNotIn("gold",item)
                self.assertNotIn("model_output",item)
    def test_model_output_injection_is_rejected(self):
        x=build_packet(self.source,"A")
        x["items"][0]["model_output"]="BUY"
        with self.assertRaisesRegex(ValueError,"TAMPER"):
            validate_packet(x,self.source)
    def test_gold_field_added_is_rejected(self):
        x=build_packet(self.source,"A")
        x["expert_approved"]=True
        with self.assertRaisesRegex(ValueError,"TAMPER"):
            validate_packet(x,self.source)
    def test_source_tampering_is_detected(self):
        x=build_packet(self.source,"A")
        x["items"][0]["source_pdf_sha256"]="0"*64
        with self.assertRaisesRegex(ValueError,"TAMPER"):
            validate_packet(x,self.source)
    def test_issue_registry_open_and_sha_bound(self):
        x=validate_issues(self.issues,self.source)
        self.assertEqual(x["open_source_issues"],4)
        self.assertFalse(x["sft_ready"])
        names={s["id"] for s in self.issues["issues"]}
        self.assertIn("SOURCE_CONFLICT_AO_MEDIAN_PRICE_ENGLISH_TC2",names)
        self.assertIn("SOURCE_CONFLICT_NTD_LIPS_SHIFT",names)
    def test_cannot_approve_conflict_by_editing_flag(self):
        x=copy.deepcopy(self.issues)
        x["issues"][0]["training_approved"]=True
        with self.assertRaisesRegex(ValueError,"PREMATURE"):
            validate_issues(x,self.source)
        x=copy.deepcopy(self.issues)
        x["independent_experts"]=2
        with self.assertRaisesRegex(ValueError,"FALSE_SOURCE_APPROVAL"):
            validate_issues(x,self.source)
    def test_source_issue_must_refer_to_existing_probe(self):
        x=copy.deepcopy(self.issues)
        x["issues"][1]["affected_probe_ids"]=["W7R-999"]
        with self.assertRaisesRegex(ValueError,"PROBE_ID"):
            validate_issues(x,self.source)
    def test_private_packet_create_only_0600(self):
        with TemporaryDirectory() as t:
            path=Path(t)/"review.json"
            create_private(path,build_packet(self.source,"A"))
            self.assertEqual(path.stat().st_mode & 0o777,0o600)
            self.assertEqual(len(json.loads(path.read_text())["items"]),30)
            with self.assertRaises(FileExistsError):
                create_private(path,build_packet(self.source,"B"))
    def test_unsupported_slot_rejected(self):
        with self.assertRaisesRegex(ValueError,"INVALID_REVIEWER_SLOT"):
            build_packet(self.source,"AUTO_EXPERT")
if __name__=="__main__":
    unittest.main()
