"""Strict source-bound, UNLABELLED counterfactual probes; not SFT or gold."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from .book_rules import registry
from .independent_exam_v6 import BOOKS, PDF_SHA, PAGES

SCHEMA = "WILLIAMS_V7_REASONING_COUNTERFACTUAL_REVIEW_NO_GOLD_V1"
FOCUSES = {"ANGULATION","WAVE_STRUCTURE","WISE_MEN","CAMPAIGN_CONTEXT","AMBIGUITY"}
RUBRIC = {"PRIMARY_BOOK_EDITION_AND_PAGE","OBSERVABLE_CLOSED_BAR_EVIDENCE",
          "CAUSE_AND_EVENT_SEQUENCE","ALTERNATIVE_WAVE_OR_SIGNAL_INTERPRETATION",
          "MISSING_DATA_AND_ABSTENTION","NO_BROKER_FILL_INFERENCE"}
FORBIDDEN = {"gold","expected","answer","decision","prediction","model_output",
             "label","score","target_class","correct","verified_chart_gold",
             "expert_vote","approved_for_sft","chart_file","model_response",
             "desired_output","test_answer","grade"}
ROOT = Path(__file__).resolve().parents[1]

def require(ok, why):
    if not ok:
        raise ValueError("V7_REASONING_REVIEW_" + why)

def ban_labels(obj):
    if type(obj) is dict:
        for k,v in obj.items():
            require(type(k) is str and k.lower() not in FORBIDDEN,
                    "GOLD_OR_PSEUDO_LABEL_FIELD")
            ban_labels(v)
    elif type(obj) is list:
        for x in obj:
            ban_labels(x)

def valid_text(x):
    return type(x) is str and bool(x.strip()) and len(x) <= 2400

def validate(data):
    require(type(data) is dict and
            set(data)=={"schema","review_gate","rubric_axes","pairs"} and
            data.get("schema")==SCHEMA,"SCHEMA")
    ban_labels(data)
    gate=data["review_gate"]
    require(type(gate) is dict and set(gate)=={
        "training_approved","sft_ready","independent_expert_reviews",
        "chart_gold_labels","model_predictions_used","real_market_chart_cases",
        "source_review_status","gpu_training_authorized","trading_authorized"
    },"GATE_FIELDS")
    require(gate=={
        "training_approved":False,"sft_ready":False,
        "independent_expert_reviews":0,"chart_gold_labels":0,
        "model_predictions_used":False,"real_market_chart_cases":0,
        "source_review_status":"SOURCE_SPOTCHECK_INDEPENDENT_ARBITRATION_REQUIRED",
        "gpu_training_authorized":False,"trading_authorized":False
    },"NOT_APPROVED")
    require(type(data["rubric_axes"]) is list and
            len(data["rubric_axes"])==len(RUBRIC) and
            set(data["rubric_axes"])==RUBRIC,"RUBRIC")
    pairs=data["pairs"]
    require(type(pairs) is list and len(pairs)==15,"PAIR_COUNT")
    cards=registry()["rule_cards"]
    rules={r["id"]:r["source"] for r in cards}
    seen=set();book_counts={ed:0 for ed in BOOKS};focus_counts={f:0 for f in FOCUSES}
    for p in pairs:
        require(type(p) is dict and set(p)=={
            "pair_id","focus","edition","source_pdf_sha256","source_pages",
            "rule_ids","source_review_ref","comparison_axis","question",
            "variant_a","variant_b"},"PAIR_FIELDS")
        pid=p["pair_id"];ed=p["edition"];focus=p["focus"]
        require(type(pid) is str and pid.startswith("W7R-") and
                len(pid)==7 and pid not in seen,"PAIR_ID")
        seen.add(pid)
        require(ed in BOOKS and focus in FOCUSES,"EDITION_OR_FOCUS")
        book_counts[ed]+=1;focus_counts[focus]+=1
        require(p["source_pdf_sha256"]==PDF_SHA[ed],"SOURCE_SHA")
        pages=p["source_pages"]
        require(type(pages) is list and bool(pages) and
                len(pages)==len(set(pages)) and
                all(type(i) is int and 1<=i<=PAGES[ed] for i in pages),
                "SOURCE_PAGES")
        ids=p["rule_ids"]
        require(type(ids) is list and bool(ids) and
                len(ids)==len(set(ids)) and
                all(type(x) is str and rules.get(x)==ed for x in ids),
                "CROSS_EDITION_RULE_ID")
        ref=p["source_review_ref"]
        require(type(ref) is str and
                ref.startswith("research/WILLIAMS_V7_RULE_CHECK_") and
                ref.endswith("_20261115_RU.md") and
                (ROOT / ref).is_file(),"REVIEW_REFERENCE")
        require(valid_text(p["comparison_axis"]) and
                valid_text(p["question"]),"MISSING_REVIEW_QUESTION")
        for variant in ("variant_a","variant_b"):
            v=p[variant]
            require(type(v) is dict and set(v)=={"scenario","unavailable_facts"},
                    "VARIANT_FIELDS")
            require(valid_text(v["scenario"]) and
                    type(v["unavailable_facts"]) is list and
                    1<=len(v["unavailable_facts"])<=6 and
                    all(valid_text(q) for q in v["unavailable_facts"]),
                    "VARIANT_CONTENT")
        require(p["variant_a"]["scenario"]!=p["variant_b"]["scenario"],
                "NO_COUNTERFACTUAL_CHANGE")
    require(all(book_counts.values()) and all(focus_counts.values()),
            "FOUR_BOOK_FIVE_FOCUS_COVERAGE")
    return {"pair_count":len(pairs),"unlabelled_probe_variants":2*len(pairs),
            "source_books":book_counts,"focus_counts":focus_counts,
            "verified_human_chart_labels":0,"approved_for_sft":False,
            "model_training_started":False,"book_accuracy":None}

def main():
    a=argparse.ArgumentParser()
    a.add_argument("--input",type=Path,required=True)
    p=a.parse_args()
    print(json.dumps(validate(json.loads(p.input.read_text(encoding="utf-8"))),
                     sort_keys=True,ensure_ascii=False))
if __name__=="__main__":
    main()
