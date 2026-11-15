"""Validate six NTD primary-source rechecks; no labels/training authority."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from .independent_exam_v6 import PDF_SHA,PAGES
from .book_rules import registry
SCHEMA="WILLIAMS_V7_NTD_FOUR_DIMENSIONS_SPOTCHECK_NO_GOLD_V1"
def validate(x):
 if not isinstance(x,dict) or x.get("schema")!=SCHEMA:raise ValueError("NTD_RECHECK_SCHEMA")
 if x.get("book")!="NTD" or x.get("source_pdf_sha256")!=PDF_SHA["NTD"]:raise ValueError("NTD_RECHECK_SOURCE_HASH")
 if x.get("expert_reviews")!=0 or x.get("model_labels")!=0 or x.get("training_approved") is not False:raise ValueError("NTD_RECHECK_GOLD_LEAK")
 cards={c["id"]:c["source"] for c in registry()["rule_cards"]}
 if len(x.get("reviews",[]))!=6:raise ValueError("NTD_RECHECK_EXPECTED_SIX")
 seen=set()
 for r in x["reviews"]:
  if r.get("id") in seen or not r.get("rule_ids"):raise ValueError("NTD_RECHECK_DUPLICATE")
  seen.add(r["id"])
  if any(i not in cards or cards[i]!="NTD" for i in r["rule_ids"]):raise ValueError("NTD_RECHECK_WRONG_BOOK_RULE")
  if any(type(p) is not int or p<1 or p>PAGES["NTD"] for p in r["pages"]):raise ValueError("NTD_RECHECK_PAGE")
  if not r.get("scope"):raise ValueError("NTD_RECHECK_SCOPE")
 return {"reviews":len(seen),"source_book":"NTD","expert_gold":0,"training_authorized":False}
def main():
 p=argparse.ArgumentParser();p.add_argument("--input",required=True);a=p.parse_args()
 print(json.dumps(validate(json.loads(Path(a.input).read_text())),sort_keys=True))
if __name__=="__main__":main()
