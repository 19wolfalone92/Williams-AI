"""v7 candidate edition+family retrieval profiles with mandatory book-rule coverage.

Does NOT replace the frozen v6 retriever or alter any historic output.
Every mandatory card is sourced from the pinned edition registry; optional
lexical retrieval supplements but can never displace required evidence.
"""
from __future__ import annotations
from typing import Any
from .book_rules import registry
from .source_ranker_v2 import rank_reviewed_rules

PROFILE_SCHEMA="WILLIAMS_V7_EDITION_FAMILY_EVIDENCE_PROFILE_V1"
REQUIRED={
 ("TC1","ENTRY"):("TC1.FRACTAL","TC1.RHYTHM"),
 ("TC1","EXIT"):("TC1.FRACTAL","TC1.W5","TC1.MAGIC.TARGET","TC1.PSY.PLAN"),
 ("TC1","WAVE"):("TC1.W3","TC1.W4","TC1.W5","TC1.RHYTHM",
                 "TC1.MAGIC.DIVERGENCE","TC1.MAGIC.TARGET",
                 "TC1.MAGIC.FRACTAL","TC1.MAGIC.SQUAT","TC1.MAGIC.MOMENTUM"),
 ("NTD","ENTRY"):("NTD.FRACTAL","NTD.ALLIGATOR","NTD.AO","NTD.AC",
                  "NTD.ZONE","NTD.BALANCE","NTD.ZONE.ADDED"),
 ("NTD","EXIT"):("NTD.EXITS","NTD.ALLIGATOR","NTD.AO","NTD.AC","NTD.ZONE"),
 ("NTD","WAVE"):("NTD.ALLIGATOR","NTD.AO","NTD.AC","NTD.BALANCE"),
 ("TC2","ENTRY"):("TC2.WM1.BAR","TC2.WM1.ANG","TC2.WM1.TRIGGER",
                  "TC2.WM1.STOP","TC2.WM2","TC2.WM3","TC2.CAMPAIGN"),
 ("TC2","EXIT"):("TC2.REVERSE","TC2.WM1.BAR","TC2.WM1.ANG",
                 "TC2.WM1.TRIGGER","TC2.WM1.STOP","TC2.WM2",
                 "TC2.WM3","TC2.CAMPAIGN"),
 ("TC2","WAVE"):("TC2.WM1.ANG","TC2.WM2","TC2.WM3","TC2.CAMPAIGN"),
 ("COURSE","ENTRY"):("COURSE.THUMB","COURSE.HUMP","COURSE.AO5345",
                     "COURSE.RHYTHM","COURSE.TRADING_PARTNER"),
 ("COURSE","EXIT"):("COURSE.TRADING_PARTNER","COURSE.HUMP",
                    "COURSE.THUMB","COURSE.AO5345","COURSE.RHYTHM"),
 ("COURSE","WAVE"):("COURSE.W4.ZERO","COURSE.AO5345",
                    "COURSE.RHYTHM","COURSE.VISUAL.NESTED_ABC"),
}
FORBIDDEN_TERMS={
 "TC1":("alligator","аллигатор"),
}

def need(ok,why):
    if not ok:
        raise ValueError("V7_RETRIEVAL_PROFILE_"+why)

def _registry_cards(data:dict|None=None)->dict[str,dict[str,Any]]:
    source=registry() if data is None else data
    rows=source.get("rule_cards")
    need(isinstance(rows,list),"INVALID_RULE_REGISTRY")
    result={}
    for row in rows:
        rid=row.get("id")
        need(type(rid) is str and rid not in result,"DUPLICATE_RULE_ID")
        result[rid]=row
    return result

def required_rule_ids(edition:str,family:str)->tuple[str,...]:
    key=(edition,family)
    need(key in REQUIRED,"UNKNOWN_EDITION_OR_TASK_FAMILY")
    return REQUIRED[key]

def build_evidence_bundle(question:str,*,edition:str,family:str,
                          optional_max:int=3,registry_data:dict|None=None)->dict:
    need(type(question) is str and 1<=len(question)<=1500,"INVALID_QUESTION")
    need(type(optional_max) is int and 0<=optional_max<=3,"INVALID_OPTIONAL_CARD_COUNT")
    key=(edition,family)
    required=required_rule_ids(edition,family)
    lower=question.casefold().replace("ё","е")
    for term in FORBIDDEN_TERMS.get(edition,()):
        need(term not in lower,"CROSS_EDITION_TERM_REQUIRES_REWRITE")
    cards=_registry_cards(registry_data)
    selected=[]
    for rid in required:
        row=cards.get(rid)
        need(row is not None,"MANDATORY_RULE_MISSING_"+rid)
        need(row.get("source")==edition,"MANDATORY_RULE_WRONG_EDITION_"+rid)
        need(row.get("status")=="REVIEWED_PAGES","MANDATORY_RULE_NOT_PAGE_REVIEWED_"+rid)
        pages=row.get("pdf_pages")
        need(isinstance(pages,list) and bool(pages),"MANDATORY_RULE_NO_PAGE_CITATION_"+rid)
        selected.append({
            "rule_id":rid,"edition":edition,"status":row["status"],
            "pdf_pages":list(pages),"title":row["title"],
            "source_paraphrase":row["summary"],
            "selection_reason":"REQUIRED_BY_FROZEN_V7_EDITION_FAMILY_PROFILE",
        })
    have=set(required)
    optional=[]
    if optional_max:
        for match in rank_reviewed_rules(question,edition=edition,max_cards=12):
            rid=match["rule_id"]
            if rid in have:
                continue
            row=cards.get(rid)
            need(row is not None,"OPTIONAL_RULE_MISSING_"+rid)
            need(row.get("source")==edition,"OPTIONAL_RULE_WRONG_EDITION_"+rid)
            need(row.get("status")=="REVIEWED_PAGES","OPTIONAL_RULE_NOT_PAGE_REVIEWED_"+rid)
            need(isinstance(row.get("pdf_pages"),list) and bool(row["pdf_pages"]),
                 "OPTIONAL_RULE_NO_PAGE_CITATION_"+rid)
            optional.append({
                "rule_id":rid,"edition":edition,"status":row["status"],
                "pdf_pages":list(row["pdf_pages"]),"title":row["title"],
                "source_paraphrase":row["summary"],
                "selection_reason":"OPTIONAL_LEXICAL_RETRIEVAL_SUPPLEMENT",
            })
            have.add(rid)
            if len(optional)>=optional_max:
                break
    all_rows=selected+optional
    need(len(all_rows)==len({x["rule_id"] for x in all_rows}),
         "DUPLICATE_EVIDENCE_CARD")
    need(all(x["edition"]==edition for x in all_rows),
         "CROSS_EDITION_EVIDENCE_LEAK")
    return {
        "schema":PROFILE_SCHEMA,"edition":edition,"family":family,
        "required_rule_ids":list(required),
        "retrieved_rule_ids":[x["rule_id"] for x in all_rows],
        "required_cards":selected,"optional_cards":optional,
        "mandatory_coverage_complete":True,
        "frozen_v6_unchanged":True,
        "model_predictions_used":False,"expert_gold_included":False,
        "execution_allowed":False,
    }
