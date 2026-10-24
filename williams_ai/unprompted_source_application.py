"""Unprompted, source-retrieving Williams knowledge application.

A real local model receives an unseen task with NO hand-selected rule,
prior mistake ID, answer key, or human hint. The retrieval policy searches
ONLY the question text (never multiple choice alternatives or case ID) and
only reviewed cards of that edition. Source use is automatic like reference
lookup, not LoRA fine tuning or proof of general intelligence.

Unlike targeted source transfer (same questions twice), this mechanism
can be independently evaluated once per unseen task, frozen in advance.
"""
from __future__ import annotations

from typing import Any

from .book_rules import lookup_rules,registry


def self_retrieve_source(question: str, edition: str, *,
                         max_cards: int=2) -> dict[str,Any]:
    if (not isinstance(question,str) or not 10<=len(question)<=1200
            or edition not in {"TC1","NTD","TC2","COURSE"}
            or isinstance(max_cards,bool) or not isinstance(max_cards,int)
            or not 1<=max_cards<=3):
        raise ValueError("INVALID_UNPROMPTED_SOURCE_QUERY")
    # Keyword search is intentionally *question-only*: wrong answers
    # frequently contain seductive keywords from a different book.
    found=lookup_rules(question,edition=edition,max_cards=12)
    registered={x["id"]:x for x in registry()["rule_cards"]
                if x["source"]==edition and x["status"]=="REVIEWED_PAGES"}
    used=[]
    for result in found:
        card=registered.get(result["rule_id"])
        if card is None or card["summary"]!=result["excerpt"]:
            continue
        if len(card["summary"])>450 or not card["pdf_pages"]:
            continue
        used.append({"source":edition+":RULE:"+card["id"],
                     "excerpt":card["summary"],
                     "pdf_pages":list(card["pdf_pages"]),
                     "rule_id":card["id"]})
        if len(used)==max_cards:break
    return {
        "mode":"AUTONOMOUS_BOOK_QUESTION_ONLY_SOURCE_DISCOVERY",
        "edition":edition,"retrieved":used,
        "no_human_supplied_rule_id":True,
        "no_answer_options_used_in_retrieval":True,
        "no_answer_key_used":True,
        "all_source_cards_are_page_reviewed":True,
        "retrieval_unavailable_requires_abstention_or_unassisted_answer":not used,
        "model_weights_updated":False,
        "trade_action":None,"execution_allowed":False,
    }


def answer_with_self_retrieval(model:Any,case:dict[str,Any],
                               *,max_cards:int=2)->dict[str,Any]:
    """One-pass model answer, with automatic book lookup if relevant.

    No oracle-like fallback: we never provide the model's own wrong answer,
    expected answer, or failing rule. Tests keep human answer keys private.
    """
    if (not isinstance(case,dict) or set(case)!={"id","edition","question","choices"}
            or not isinstance(case["id"],str)
            or not isinstance(case["choices"],dict)
            or set(case["choices"])!={"A","B","C"}):
        raise ValueError("INVALID_BLIND_EXAM_CASE")
    search=self_retrieve_source(case["question"],case["edition"],
                                max_cards=max_cards)
    usable=[{"source":x["source"],"excerpt":x["excerpt"]}
            for x in search["retrieved"]]
    # The model will still need to interpret the question + evidence.
    # Retrieval never supplies the answer letter or selected response.
    result=model.answer_exam_case(
        dict(case),prior_lessons=usable if usable else None)
    choice=result.get("choice") if isinstance(result,dict) else None
    return {
        "mode":"LOCAL_GGUF_AUTONOMOUS_SOURCE_ASSISTED_BLIND_RESPONSE",
        "edition":case["edition"],"id":case["id"],
        "selected":choice if choice in {"A","B","C"} else "",
        "source_ids_used":[x["rule_id"] for x in search["retrieved"]],
        "source_pages_used":[x["pdf_pages"] for x in search["retrieved"]],
        "retrieval_invoked_automatically":True,
        "independent_success_unknown_without_hidden_key":True,
        "model_weights_updated":False,"trade_action":None,
        "execution_allowed":False,
    }
