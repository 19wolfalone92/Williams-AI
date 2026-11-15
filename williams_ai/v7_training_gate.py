"""Fail closed for unreviewed Williams v7 book/training review backlog.

This does not produce an SFT dataset or an expert answer key.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
from .independent_exam_v6 import PDF_SHA,PAGES

SCHEMA="WILLIAMS_V7_SOURCE_BOOK_REVIEW_BACKLOG_NO_TRAINING_OR_GOLD_V1"
FORBIDDEN={"gold","expected","answer","decision","prediction","label","model_output",
           "trading_signal","ground_truth","score"}

def validate(data):
    if not isinstance(data,dict) or data.get("schema")!=SCHEMA:
        raise ValueError("TRAINING_GUARD_INVALID_SOURCE_REVIEW_MANIFEST")
    if data.get("training_approved") is not False or data.get("valid_gold_labels") is not None or data.get("expert_reviews_received")!=0:
        raise ValueError("TRAINING_GUARD_UNSUPPORTED_EXPERT_APPROVAL")
    cases=data.get("cases")
    if not isinstance(cases,list) or len(cases)<8:
        raise ValueError("TRAINING_GUARD_INSUFFICIENT_CANDIDATES")
    seen=set()
    for row in cases:
        if not isinstance(row,dict) or set(row)&FORBIDDEN:
            raise ValueError("TRAINING_GUARD_FORBIDDEN_LABEL_IN_SOURCE_REVIEW")
        rid=row.get("review_id");ed=row.get("edition");pages=row.get("source_pdf_pages")
        if rid in seen or ed not in PDF_SHA or row.get("source_pdf_sha256")!=PDF_SHA[ed]:
            raise ValueError("TRAINING_GUARD_UNPINNED_BOOK_OR_DUPLICATE")
        seen.add(rid)
        if not isinstance(pages,list) or not pages or any(
           type(i) is not int or i<1 or i>PAGES[ed] for i in pages):
            raise ValueError("TRAINING_GUARD_INVALID_SOURCE_PAGES")
        if (row.get("proposed_trade_decision") is not None or
            row.get("verified_chart_gold") is not None or
            row.get("approved_for_SFT") is not False or
            row.get("reviewer_signatures_received")!=0 or
            row.get("reviewers_minimum")<2 or
            row.get("raw_model_prediction_forbidden_in_review") is not True):
            raise ValueError("TRAINING_GUARD_FAKE_GOLD_OR_TRAINING")
    return {"source_review_targets":len(cases),
            "independent_human_labels":0,"training_authorized":False,
            "model_weights_modified":False}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",type=Path,required=True)
    args=p.parse_args()
    print(json.dumps(validate(json.loads(args.input.read_bytes())),
        sort_keys=True,ensure_ascii=False))
if __name__=="__main__":main()
