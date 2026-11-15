"""Validate source-grounded v7 curriculum candidates; never treat as labels."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from .independent_exam_v6 import BOOKS,PDF_SHA,PAGES
from .book_rules import registry

SCHEMA="WILLIAMS_V7_SOURCE_GROUNDED_CURRICULUM_CANDIDATES_NO_GOLD_V1"
FORBIDDEN={"gold","answer","decision","prediction","model_output","label","expected","correct","target_class"}
def validate(data):
 if type(data) is not dict or data.get("schema")!=SCHEMA:raise ValueError("V7_CURRICULUM_SCHEMA")
 gate=data.get("review_gate",{})
 if gate.get("expert_approved") is not False or gate.get("independent_expert_review_count")!=0 or gate.get("sft_ready") is not False:
  raise ValueError("V7_CURRICULUM_PREMATURE_APPROVAL")
 cards=registry()["rule_cards"];allowed={x["id"]:x["source"] for x in cards}
 if type(data.get("lesson_cards")) is not list or len(data["lesson_cards"])!=10:
  raise ValueError("V7_CURRICULUM_EXPECTED_FOURTEEN_SOURCE_SPOTCHECK_LESSONS")
 seen=set()
 def reject(obj):
  if isinstance(obj,dict):
   for k,v in obj.items():
    if k.lower() in FORBIDDEN:raise ValueError("V7_CURRICULUM_LABEL_LEAK")
    reject(v)
  elif isinstance(obj,list):
   for v in obj:reject(v)
 reject(data)
 for c in data["lesson_cards"]:
  ed=c.get("edition")
  if ed not in BOOKS or c.get("source_pdf_sha256")!=PDF_SHA[ed]:raise ValueError("V7_CURRICULUM_UNPINNED_SOURCE")
  if c.get("id") in seen:raise ValueError("V7_CURRICULUM_DUPLICATE_ID")
  seen.add(c["id"])
  pages=c.get("pages")
  if type(pages) is not list or not pages or any(type(p) is not int or p<1 or p>PAGES[ed] for p in pages):
   raise ValueError("V7_CURRICULUM_INVALID_PAGE")
  rules=c.get("rule_ids")
  if type(rules) is not list or not rules or any(r not in allowed or allowed[r]!=ed for r in rules):
   raise ValueError("V7_CURRICULUM_RULE_BOOK_MISMATCH")
  if not c.get("principles") or not c.get("required_sequence") or not c.get("invalid_inferences"):
   raise ValueError("V7_CURRICULUM_INCOMPLETE_LESSON")
 if {c["edition"] for c in data["lesson_cards"]}!=set(BOOKS):
  raise ValueError("V7_CURRICULUM_BOOK_COVERAGE")
 if data.get("review_gate",{}).get("chart_answer_labels_included") is not False or data.get("review_gate",{}).get("model_predictions_used") is not False:
  raise ValueError("V7_CURRICULUM_NOT_SOURCE_ONLY")
 return {"lesson_candidate_count":len(seen),"books_covered":len(BOOKS),
  "independent_expert_reviews":0,"chart_labels":0,"sft_ready":False,
  "training_authorized":False,"trading_authorized":False}
def main():
 p=argparse.ArgumentParser();p.add_argument("--input",required=True);a=p.parse_args()
 print(json.dumps(validate(json.loads(Path(a.input).read_text(encoding="utf-8"))),ensure_ascii=False,sort_keys=True))
if __name__=="__main__":main()
