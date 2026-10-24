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


def main()->None:
    p=argparse.ArgumentParser(description="Real Qwen TC2/NTD weakness remediation with rule retrieval; no trading")
    p.add_argument("--model",type=Path,required=True)
    p.add_argument("--expected-sha256",required=True)
    p.add_argument("--ctx",type=int,default=8192)
    p.add_argument("--reasoning-budget",type=int,default=1024)
    args=p.parse_args()
    from .real_gguf_acceptance import actual_gguf_sha256
    from .local_model import LocalWilliamsModel
    actual_gguf_sha256(args.model,args.expected_sha256)
    model=LocalWilliamsModel(args.model,n_ctx=args.ctx,
        learning_mode="auto",reasoning_budget=args.reasoning_budget)
    print(json.dumps(paired_source_remediation(model),
                     ensure_ascii=False,indent=2,sort_keys=True))


if __name__=="__main__":
    main()
