"""Real GGUF targeted SOURCE retrieval for verified failures, not LoRA.

Errors were observed in the public 2026-10-24 16-question Qwen3-1.7B test:
NTD.AC, TC2.WM1.STOP and TC2.CAMPAIGN/no-widening.
We show the verified paraphrased book rule *after* an unguided answer,
then ask the SAME cases with a source-scoped reminder. This diagnoses
use of retrieved knowledge and is NOT independent improvement, unseen
future performance, full book mastery, or trainable weight updates.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import random
from typing import Any
import re

from .book_rules import registry


_CASES = (
    ("AC-NEGATIVE","NTD","NTD.AC",
     "На закрытом баре AC для BUY ниже 0. Какое число ПОСЛЕДОВАТЕЛЬНЫХ растущих зелёных изменений требуется в NTD?",
     "Три растущих зелёных изменения AC.","Два независимо от знака AC.",
     "Достаточно положительного AO, AC не проверяем."),
    ("AC-POSITIVE","NTD","NTD.AC",
     "На закрытом баре AC выше нуля, два последовательных зелёных подъёма AC уже наблюдались. Что требуется для BUY по правилу NTD?",
     "Два растущих зелёных изменения AC при AC выше нуля.",
     "Три, потому что AO может быть ниже нуля.",
     "Обязательное пересечение AC через ноль."),
    ("WM1-LONG-STOP","TC2","TC2.WM1.STOP",
     "Для LONG фактически исполнен buy-пробой бара бычьего разворота WM1. Где авторский первоначальный защитный SL?",
     "Чуть ниже противоположного LOW сигнального бычьего бара.",
     "Чуть выше HIGH того же бара, где был buy-пробой.",
     "Всегда фиксированный 10-процентный stop без барной структуры."),
    ("WM1-SHORT-STOP","TC2","TC2.WM1.STOP",
     "SHORT по сработавшему WM1 медвежьего разворота. Где должен быть первый книжный stop?",
     "Чуть выше противоположного HIGH сигнального медвежьего бара.",
     "Чуть ниже LOW, где был sell-пробой.",
     "Не нужен, потому что AO зелёный."),
    ("TRAIL-LONG","TC2","TC2.CAMPAIGN",
     "На подтверждённой LONG-кампании SL 100. Новый скользящий структурный ориентир по последним пяти закрытым барам 97. Что делать с защитой?",
     "Не переносить SL вниз на 97 — это расширение риска; ждать допустимого повышения.",
     "Перенести SL вниз 97, чтобы стоп стал дальше от цены.",
     "Отменить SL и безусловно добавить."),
    ("TRAIL-SHORT","TC2","TC2.CAMPAIGN",
     "На подтверждённом SHORT уже есть SL 110. Новый фрактально-барный ориентир 113. Как трактовать перенос стопа?",
     "Нельзя повышать SHORT SL с 110 до 113: защитный риск увеличится.",
     "Обязательно поднять SL до 113, потому что ориентир новый.",
     "Автоматически закрыть по одному встречному AO."),
)


def remediation_cases(seed: int=20261024) -> list[dict[str,Any]]:
    if isinstance(seed,bool) or not isinstance(seed,int) or not 0<=seed<2**32:
        raise ValueError("INVALID_SOURCE_TRANSFER_SEED")
    reviewed={x["id"]:x for x in registry()["rule_cards"]
              if x["status"]=="REVIEWED_PAGES"}
    rng=random.Random(seed)
    out=[]
    for case_id,ed,rule,q,correct,wrong1,wrong2 in _CASES:
        card=reviewed.get(rule)
        if card is None or card["source"]!=ed:
            raise ValueError("MISSING_INDEPENDENTLY_REVIEWED_SOURCE_RULE")
        answers=[correct,wrong1,wrong2]
        rng.shuffle(answers)
        # Only original-source-verified paraphrase; no case answer key.
        reminder={"source":ed+":RULE:"+rule,
                  "excerpt":"Проверенное книжное правило: "+card["summary"]}
        if not 10<=len(reminder["excerpt"])<=450:
            raise ValueError("UNSAFE_SOURCE_REMINDER_LENGTH")
        out.append({
            "public":{"id":"REMEDIATION-"+case_id,"edition":ed,
                      "question":q,"choices":dict(zip("ABC",answers))},
            "expected":"ABC"[answers.index(correct)],"rule":rule,
            "reminder":reminder,"pages":card["pdf_pages"],
        })
    return out


def paired_source_remediation(model:Any, *,seed:int=20261024) -> dict[str,Any]:
    """BEFORE and WITH citation on same cases, never authorize anything."""
    items=remediation_cases(seed)
    results=[]
    for row in items:
        scores=[]
        for guided in (False,True):
            try:
                kw={"prior_lessons":[row["reminder"]]} if guided else {}
                response=model.answer_exam_case(row["public"],**kw)
                got=response.get("choice") if isinstance(response,dict) else None
                got=got if got in ("A","B","C") else ""
                failure=None if got else "MALFORMED_OR_ABSTENTION"
            except (ValueError,RuntimeError,TypeError,KeyError):
                got="";failure="REAL_GGUF_CALL_ERROR"
            scores.append({
                "phase":"with_checked_source" if guided else "unaided",
                "selected":got,"correct":got==row["expected"],
                "failure":failure,
            })
        results.append({"id":row["public"]["id"],
                        "edition":row["public"]["edition"],"rule":row["rule"],
                        "unaided":scores[0],"source_assisted":scores[1],
                        "pdf_pages":row["pages"]})
    before=sum(x["unaided"]["correct"] for x in results)
    after=sum(x["source_assisted"]["correct"] for x in results)
    return {
        "mode":"ACTUAL_MODEL_BOOK_SOURCE_TRANSFER_DIAGNOSTIC",
        "original_failures_found_in_public_16_CASE_EXAM":(
            ["NTD.AC","TC2.WM1.STOP","TC2.CAMPAIGN"]),
        "cases":len(items),"source_scoped":True,
        "unguided_correct":before,"retrieval_assisted_correct":after,
        "delta_source_assistance_not_weight_training":after-before,
        "answer_keys_not_in_model_input":True,
        "same_questions_in_both_phases_not_independent_blind_holdout":True,
        "source_text_exported":False,"model_weights_updated":False,
        "market_profitability_proven":False,
        "results":results,
        "protocol_sha256":sha256(json.dumps(items,sort_keys=True,
            ensure_ascii=False,separators=(",",":")).encode()).hexdigest(),
        "execution_allowed":False,"trade_action":None,
    }


def self_study_stop_remediation(model:Any, memory_path:Path, *,
                               model_id:str) -> dict[str,Any]:
    """Actual inference/read/checkpoint/recall using PUBLIC paraphrase cards.

    No private PDF is available in public CI. These notes have their OWN
    table and cannot masquerade as lessons from the original-page index.
    The four repeated questions are diagnostic practice, not a holdout.
    Generated summaries stay quarantined; the second answer gets source
    cards only. Matching a quoted span cannot validate an interpretation.
    """
    from .study import memory, validate_lesson, _lesson_error_code
    if not isinstance(model_id,str) or not re.fullmatch(r"gguf-[a-zA-Z0-9_.:-]{1,115}",model_id):
        raise ValueError("Exact local model identity required")
    cases=[x for x in remediation_cases() if x["public"]["edition"]=="TC2"]
    cards={x["id"]:x for x in registry()["rule_cards"]}
    def answer(case,notes=None):
        try:
            got=model.answer_exam_case(case["public"],**({"prior_lessons":notes} if notes else {}))
            choice=got.get("choice") if isinstance(got,dict) else None
            return {"selected":choice if choice in {"A","B","C"} else "",
                    "correct":choice==case["expected"],
                    "call_error":None}
        except (ValueError,RuntimeError,TypeError,KeyError):
            return {"selected":"","correct":False,"call_error":"MODEL_CALL_FAILED"}
    before=[answer(c) for c in cases]
    study_results=[]
    notes=set()
    with memory(memory_path) as db:
        db.execute("""CREATE TABLE IF NOT EXISTS public_rule_study_notes(
            model_id TEXT NOT NULL,rule_id TEXT NOT NULL,card_sha256 TEXT NOT NULL,
            raw_lesson_json TEXT NOT NULL,
            PRIMARY KEY(model_id,rule_id,card_sha256))""")
        for rule in dict.fromkeys(c["rule"] for c in cases):
            card=cards[rule]
            passage=card["summary"]
            digest=sha256(passage.encode()).hexdigest()
            row=db.execute("SELECT raw_lesson_json FROM public_rule_study_notes "
                "WHERE model_id=? AND rule_id=? AND card_sha256=?",
                (model_id,rule,digest)).fetchone()
            reused=False
            failure=None
            try:
                raw=row[0] if row else model.study_passage(passage,edition=card["source"],
                    page=card["pdf_pages"][0],rule_cards=[])
                lesson=validate_lesson(raw,passage)
                if not lesson["claims"]:
                    raise ValueError("Invalid support span")
                if row:
                    reused=True
                else:
                    db.execute("INSERT INTO public_rule_study_notes VALUES(?,?,?,?)",
                        (model_id,rule,digest,raw))
                    db.commit()
                notes.add(rule)
            except (ValueError,RuntimeError,TypeError,KeyError) as exc:
                failure=_lesson_error_code(exc)
            study_results.append({"rule":rule,"saved_or_recalled":rule in notes,
                "reused_same_model_note":reused,"failure":failure,
                "original_pdf_read":False})
    # Close/reopen the durable notebook before reusing anything: this is a
    # real recall, not an in-memory answer smuggled into the next call.
    with memory(memory_path) as db:
        for rule in sorted(notes):
            card=cards[rule]
            digest=sha256(card["summary"].encode()).hexdigest()
            raw=db.execute("SELECT raw_lesson_json FROM public_rule_study_notes "
                "WHERE model_id=? AND rule_id=? AND card_sha256=?",
                (model_id,rule,digest)).fetchone()[0]
            validate_lesson(raw,card["summary"])
    results=[]
    for case,initial in zip(cases,before):
        context=[case["reminder"]]
        after=answer(case,context)
        results.append({"id":case["public"]["id"],"rule":case["rule"],
            "unaided":initial,"source_after_study":after,
            "self_note_saved_or_recalled":case["rule"] in notes,
            "self_note_used":False})
    return {"mode":"ACTUAL_GGUF_SELF_STUDY_PUBLIC_PARAPHRASE_DIAGNOSTIC",
        "model_id":model_id,"cases":len(results),
        "unaided_correct":sum(x["unaided"]["correct"] for x in results),
        "source_only_after_study_correct":sum(x["source_after_study"]["correct"] for x in results),
        "unverified_self_notes_quarantined":True,
        "source_kind":"PUBLIC_REVIEWED_RULE_PARAPHRASE_NOT_ORIGINAL_PDF",
        "study_results":study_results,"results":results,
        "same_questions_repeated_not_independent_holdout":True,
        "answer_keys_not_in_model_input":True,
        "notes_remain_unverified":True,"model_weights_updated":False,
        "market_profitability_proven":False,"execution_allowed":False,"trade_action":None}


def main()->None:
    p=argparse.ArgumentParser(description="Real Qwen TC2/NTD weakness remediation with rule retrieval; no trading")
    p.add_argument("--model",type=Path,required=True)
    p.add_argument("--expected-sha256",required=True)
    p.add_argument("--ctx",type=int,default=8192)
    p.add_argument("--reasoning-budget",type=int,default=1024)
    p.add_argument("--self-study-memory",type=Path,
                   help="Run checkpointed public-paraphrase study diagnostic, not original PDF study")
    args=p.parse_args()
    from .real_gguf_acceptance import actual_gguf_sha256
    from .local_model import LocalWilliamsModel
    actual_gguf_sha256(args.model,args.expected_sha256)
    model=LocalWilliamsModel(args.model,n_ctx=args.ctx,
        learning_mode="auto",reasoning_budget=args.reasoning_budget)
    if args.self_study_memory:
        from .study_cli import _model_identity
        result=self_study_stop_remediation(model,args.self_study_memory,
            model_id=_model_identity(args.model))
    else:
        result=paired_source_remediation(model)
    print(json.dumps(result,
                     ensure_ascii=False,indent=2,sort_keys=True))


if __name__=="__main__":
    main()
