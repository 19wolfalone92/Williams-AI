"""Synthetic signatures and synthetic OHLCV are software fixtures, never experts."""
import copy,unittest
from datetime import datetime,timedelta,timezone
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives import serialization
from tests.test_williams_ai_independent_exam_v6 import V6IndependentIntakeTests
from williams_ai.independent_exam_v6 import canonical,validate_cases
from williams_ai.book_rules import registry
from williams_ai.v7_training_data_contract import (
 build,SPLIT_SCHEMA,RULEMAP_SCHEMA,FORBIDDEN
)

class V7TrainingDataContractTests(unittest.TestCase):
 def setUp(self):
  self.fixture=V6IndependentIntakeTests("test_frozen_chart_signatures_all_four_book_counts")
  self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
  self.cases={"schema":self.fixture.dataset["schema"],
   "dataset_id":self.fixture.dataset["dataset_id"],
   "cases":self.fixture.cases,
   "provenance_note":self.fixture.dataset["provenance_note"]}
  cards=registry()["rule_cards"]
  book_rule={b:next(x for x in cards if x["source"]==b) for b in ("TC1","NTD","TC2","COURSE")}
  self.rule_map={"schema":RULEMAP_SCHEMA,"dataset_sha256":None,"items":[]}
  self.split={"schema":SPLIT_SCHEMA,"dataset_sha256":None,"frozen_before_reviews":True,"assignments":[]}
  for i,c in enumerate(self.cases["cases"]):
   card=book_rule[c["edition"]]
   c["pdf_pages"]=list(card["pdf_pages"])
   self.rule_map["items"].append({"case_id":c["id"],"rule_ids":[card["id"]]})
   book_index=i//8;in_book=i%8
   split="train" if in_book<4 else "dev" if in_book<6 else "holdout"
   self.split["assignments"].append({"case_id":c["id"],"split":split,"group_id":"G_"+c["id"]})
  self.manifest=validate_cases(self.cases,self.fixture.root)
  self.rule_map["dataset_sha256"]=self.manifest["dataset_sha256"]
  self.split["dataset_sha256"]=self.manifest["dataset_sha256"]
  self.keys={k:Ed25519PrivateKey.generate() for k in ("review_A","review_B")}
  self.trust={}
  for name,key in self.keys.items():
   pub=key.public_key().public_bytes(encoding=serialization.Encoding.Raw,
      format=serialization.PublicFormat.Raw).hex()
   self.trust[name]={"ed25519_public_key_hex":pub,
     "externally_vetted_expert":True,"independent_of_training":True}
  labels=self.fixture.labels()
  for row in labels:
   c=next(x for x in self.cases["cases"] if x["id"]==row["id"])
   row["pages"]=list(c["pdf_pages"])
  self.review_a=self._signed("review_A",labels)
  self.review_b=self._signed("review_B",labels)
 def _signed(self,rid,labels):
  payload={"schema":"WILLIAMS_V6_EXPERT_REVIEW_V1",
   "dataset_sha256":self.manifest["dataset_sha256"],
   "reviewer_id":rid,"labels":copy.deepcopy(labels)}
  return {"payload":payload,"signature_hex":self.keys[rid].sign(canonical(payload)).hex()}
 def build(self,**changes):
  args={"cases":self.cases,"manifest":self.manifest,"review_a":self.review_a,
   "review_b":self.review_b,"trust":self.trust,"split_plan":self.split,
   "rule_map":self.rule_map,"chart_root":self.fixture.root}
  args.update(changes);return build(**args)
 def test_requires_two_signed_reviewers_agreement_and_sealed_holdout(self):
  x=self.build()
  self.assertEqual(x["independent_human_agreed_examples"],32)
  self.assertEqual(len(x["candidate_train_dev"]["examples"]),24)
  self.assertEqual(len(x["sealed_holdout"]["labels"]),8)
  self.assertFalse(x["training_authorized"]);self.assertFalse(x["gpu_training_authorized"])
  self.assertFalse(x["model_predictions_used_as_labels"])
 def test_forged_or_unsigned_review_rejected(self):
  bad=copy.deepcopy(self.review_a);bad["payload"]["labels"][0]["decision"]="NO_SETUP"
  with self.assertRaisesRegex(ValueError,"REVIEW_SIGNATURE_INVALID"):self.build(review_a=bad)
  bad=copy.deepcopy(self.review_a);bad["signature_hex"]="0"*128
  with self.assertRaisesRegex(ValueError,"REVIEW_SIGNATURE_INVALID"):self.build(review_a=bad)
 def test_same_reviewer_cannot_sign_twice(self):
  with self.assertRaisesRegex(ValueError,"REVIEWERS_NOT_INDEPENDENT"):self.build(review_b=self.review_a)
 def test_model_prediction_fields_rejected_anywhere(self):
  bad=copy.deepcopy(self.rule_map);bad["model_prediction"]="BUY"
  with self.assertRaisesRegex(ValueError,"PSEUDO_LABEL"):self.build(rule_map=bad)
 def test_split_plan_cannot_leak_overlapping_windows(self):
  split=copy.deepcopy(self.split)
  # All fixture charts use different symbols. Force two overlapping charts
  # to the same instrument and assign them across two partitions.
  charts=self.fixture.root
  import json
  a=self.cases["cases"][0];b=self.cases["cases"][1]
  aa=json.loads((charts/a["chart_file"]).read_bytes());bb=json.loads((charts/b["chart_file"]).read_bytes())
  bb["symbol"]=aa["symbol"]
  bp=charts/b["chart_file"];bp.write_bytes(canonical(bb))
  b["chart_sha256"]=sha256(bp.read_bytes()).hexdigest()
  self.manifest=validate_cases(self.cases,charts)
  self.rule_map["dataset_sha256"]=self.manifest["dataset_sha256"]
  self.split["dataset_sha256"]=self.manifest["dataset_sha256"]
  self.split["assignments"][0]["split"]="train";self.split["assignments"][1]["split"]="dev"
  with self.assertRaisesRegex(ValueError,"OVERLAPPING_SAME_SYMBOL"):self.build()
 def test_private_outputs_are_create_only_and_not_promoted(self):
  import os,json
  x=self.build()
  self.assertNotIn("models",x)
  self.assertFalse(x["trading_allowed"])
if __name__=="__main__":unittest.main()
