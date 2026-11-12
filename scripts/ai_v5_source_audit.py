"""Read-only source-card retrieval coverage audit.

Existing reviewed example questions are DIAGNOSTIC, not new hidden tests.
Never use the expected answer to alter retrieval or to train Qwen.
"""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import argparse
import json

from williams_ai.four_book_blind_exam import four_book_blind_cases
from williams_ai.unprompted_source_application import self_retrieve_source


def audit() -> dict:
    groups=defaultdict(list)
    cases=four_book_blind_cases(seed=20261024)
    for case in cases:
        question=case["public"]["question"]
        edition=case["public"]["edition"]
        found=self_retrieve_source(question,edition,max_cards=2)
        ids=[x["rule_id"] for x in found["retrieved"]]
        if any(not x.startswith(edition+".") for x in ids):
            raise ValueError("CROSS_EDITION_SOURCE_RETRIEVAL")
        groups[edition].append({
            "case_id":case["public"]["id"],"retrieved":ids,
            "expected_source_present":case["source_rule_id"] in ids,
            "no_card_found":not ids,
            "gold_answer_not_used_in_search":True,
        })
    per_book={}
    for ed,rows in groups.items():
        per_book[ed]={
            "total":len(rows),
            "correct_rule_in_top2":sum(x["expected_source_present"] for x in rows),
            "retrieval_missing":sum(x["no_card_found"] for x in rows),
            "details":rows,
        }
    return {"schema":"WILLIAMS_V5_RETRIEVAL_DIAGNOSTIC_NOT_BLIND",
            "case_count":len(cases),
            "target_rule_top2_hit":sum(x["expected_source_present"]
                                      for rows in groups.values() for x in rows),
            "per_book":per_book,
            "no_model_inference":True,
            "no_test_label_to_training":True,
            "not_expert_chart_gold":True,
            "model_weight_changes":False,
            "execution_allowed":False}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():
        raise FileExistsError("DO_NOT_OVERWRITE_RETRIEVAL_AUDIT")
    report=audit()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2,sort_keys=True))
    print("WILLIAMS_V5_SOURCE_RETRIEVAL_AUDIT",
          json.dumps({"total":report["case_count"],
                      "source_top2_hits":report["target_rule_top2_hit"],
                      "by_book":{k:{"hits":v["correct_rule_in_top2"],
                                    "missing":v["retrieval_missing"]}
                                 for k,v in report["per_book"].items()}},
                     sort_keys=True),flush=True)


if __name__=="__main__":
    main()
