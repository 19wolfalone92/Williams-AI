"""Adaptive *private* GGUF practice, grounded in the four Williams sources.

This DOES NOT fine-tune GGUF weights. Wrong answers create a short
source-scoped correction in a PRIVATE SQLite notebook. The correction can
be recalled on later questions; passing the same question after seeing the
answer is NOT a blind proof of understanding or profitable trading.

All cases are practice cases, NOT the independent 10-question exam or a
new untouched market holdout. No model or book text leaves the local host.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator
import json
import sqlite3

from .book_index import _private_database
from .book_rules import registry
from .competency_exam import public_question

CURRICULUM_PATH = Path(__file__).resolve().parents[1] / "knowledge/ADAPTIVE_TUTOR_PRACTICE_RU.json"


def learning_cases() -> tuple[dict[str, Any], ...]:
    data=json.loads(CURRICULUM_PATH.read_text(encoding="utf-8"))
    if data.get("schema")!="WILLIAMS_AI_PRIVATE_TUTOR_CURRICULUM_V1":
        raise ValueError("Unexpected teaching curriculum")
    cards={x["id"]:x for x in registry()["rule_cards"]}
    seen=set()
    cases=data.get("cases",[])
    if not isinstance(cases,list) or not cases:
        raise ValueError("Teaching curriculum empty")
    for case in cases:
        if not isinstance(case,dict) or set(case)!={"id","edition","rule_id","question","choices","correct","feedback"}:
            raise ValueError("Incorrect practice case")
        ident=case["id"]
        if not isinstance(ident,str) or not ident.isascii() or not ident.isalnum() or ident in seen:
            raise ValueError("Nonunique case identifier")
        seen.add(ident)
        card=cards.get(case["rule_id"])
        if card is None or card["source"]!=case["edition"] or card["status"]!="REVIEWED_PAGES":
            raise ValueError("Unverified or cross-edition educational claim")
        choices=case["choices"]
        if not isinstance(choices,dict) or set(choices)!={"A","B","C"}:
            raise ValueError("Incorrect answer shape")
        if case["correct"] not in choices:
            raise ValueError("Unknown practice key")
        if (not isinstance(case["question"],str) or len(case["question"])>1200
                or not isinstance(case["feedback"],str) or not 12<=len(case["feedback"])<=400
                or any(not isinstance(v,str) or not 1<=len(v)<=400 for v in choices.values())):
            raise ValueError("Incorrect educational text")
    return tuple(cases)


@contextmanager
def notebook(path:str|Path)->Iterator[sqlite3.Connection]:
    """Private memory; model-specific; never a path inside the public repo."""
    dest=_private_database(path)
    dest.parent.mkdir(parents=True,exist_ok=True)
    if not dest.exists():
        import os
        fd=os.open(dest,os.O_CREAT|os.O_EXCL|os.O_RDWR,0o600)
        os.close(fd)
    db=sqlite3.connect(dest,timeout=30)
    try:
        db.execute("""CREATE TABLE IF NOT EXISTS practice_attempts(
            seq INTEGER PRIMARY KEY AUTOINCREMENT,
            model_id TEXT NOT NULL,
            edition TEXT NOT NULL, rule_id TEXT NOT NULL,
            case_id TEXT NOT NULL, choice TEXT,
            success INTEGER NOT NULL CHECK(success IN (0,1)),
            coaching_visible INTEGER NOT NULL CHECK(coaching_visible IN (0,1)),
            attempted_at TEXT NOT NULL
        )""")
        db.execute("""CREATE INDEX IF NOT EXISTS practice_model_rule
                      ON practice_attempts(model_id,edition,rule_id)""")
        db.execute("""CREATE TABLE IF NOT EXISTS private_remediation(
            model_id TEXT NOT NULL, edition TEXT NOT NULL, rule_id TEXT NOT NULL,
            source TEXT NOT NULL, explanation TEXT NOT NULL,
            failures INTEGER NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY(model_id,edition,rule_id)
        )""")
        yield db
        db.commit()
    finally:
        db.close()


def prior_lesson(
    db:sqlite3.Connection,model_id:str,edition:str,rule_id:str
)->list[dict[str,str]]:
    row=db.execute(
        "SELECT source,explanation FROM private_remediation "
        "WHERE model_id=? AND edition=? AND rule_id=?",
        (model_id,edition,rule_id),
    ).fetchone()
    if not row:
        return []
    return [{"source":row[0],"excerpt":row[1]}]



def recall_model_corrections(
    notebook_db:Path,model_id:str,edition:str, *,
    focus_rule_ids:tuple[str,...]=(),limit:int=2,
)->list[dict[str,str]]:
    """Retrieve REAL previous private mistakes for this exact model fingerprint.

    No past examiner answers, no holdings, no data from other GGUF models,
    and no lessons across editions may enter a new decision.
    """
    if not isinstance(model_id,str) or not model_id.startswith("gguf-"):
        raise ValueError("Model fingerprint required for research memory")
    if edition not in {"TC1","NTD","TC2","COURSE"}:
        raise ValueError("Edition required for research memory")
    if not isinstance(limit,int) or not 1<=limit<=3:
        raise ValueError("Max 3 source-specific reminders")
    verified={x["id"] for x in registry()["rule_cards"] if x["source"]==edition}
    if not isinstance(focus_rule_ids,tuple) or not set(focus_rule_ids)<=verified:
        raise ValueError("Only known same-edition rule IDs")
    if not _private_database(notebook_db).is_file():
        raise FileNotFoundError("Model has not yet completed a private coaching session")
    with notebook(notebook_db) as db:
        rows=db.execute(
            "SELECT rule_id,source,explanation FROM private_remediation "
            "WHERE model_id=? AND edition=? "
            "ORDER BY failures DESC,updated_at DESC",
            (model_id,edition),
        ).fetchall()
    selected=[r for r in rows if not focus_rule_ids or r[0] in focus_rule_ids]
    return [{"source":source,"excerpt":explanation}
            for rule,source,explanation in selected[:limit]]


def _model_answer(model:Any, question:dict[str,Any],
                  reminders:list[dict[str,str]])->dict[str,Any]:
    # Without reminders, preserve the original answer_exam_case interface.
    if reminders:
        answer=model.answer_exam_case(question,prior_lessons=reminders)
    else:
        answer=model.answer_exam_case(question)
    return answer if isinstance(answer,dict) else {}


def coach_practice(
    model:Any, notebook_db:Path, *,model_id:str,
    editions:tuple[str,...]=("TC1","NTD","TC2","COURSE"),
    max_cases:int=29,repeat_after_feedback:bool=True,
)->dict[str,Any]:
    """Correct -> cited remediation -> same-case guided retry -> private memory.

    1st-pass score is separate from coached repeats. A student that uses a
    lesson to answer the SAME question must not be called book proficient.
    """
    if not isinstance(model_id,str) or not model_id.startswith("gguf-") or len(model_id)>120:
        raise ValueError("Model fingerprint required")
    if not isinstance(editions,tuple) or len(set(editions))!=len(editions) or not editions:
        raise ValueError("At least one unique edition")
    if not set(editions)<={"TC1","NTD","TC2","COURSE"}:
        raise ValueError("Unknown edition")
    if not isinstance(max_cases,int) or not 1<=max_cases<=150:
        raise ValueError("max_cases must be 1..150")
    available=[c for c in learning_cases() if c["edition"] in editions][:max_cases]
    cards={c["id"]:c for c in registry()["rule_cards"]}
    results=[]
    initial_ok=coached_ok=0
    with notebook(notebook_db) as db:
        for case in available:
            notes=prior_lesson(db,model_id,case["edition"],case["rule_id"])
            question=public_question(case)
            # Answer key and remediation for THIS case are absent from question.
            response=_model_answer(model,question,notes)
            selected=response.get("choice")
            if selected not in {"A","B","C"}:
                selected=None
            correct=(selected==case["correct"])
            initial_ok+=int(correct)
            now=datetime.now(timezone.utc).isoformat()
            db.execute(
                "INSERT INTO practice_attempts(model_id,edition,rule_id,case_id,choice,"
                "success,coaching_visible,attempted_at) VALUES (?,?,?,?,?,?,?,?)",
                (model_id,case["edition"],case["rule_id"],case["id"],selected,
                 int(correct),int(bool(notes)),now),
            )
            result={
                "id":case["id"],"edition":case["edition"],
                "rule_id":case["rule_id"],
                "initial_correct":correct,"prior_lesson_used":bool(notes),
                "same_case_retry_correct":None,
                "trade_action":None,
            }
            if not correct:
                source=f'{case["edition"]}:RULE:{case["rule_id"]}'
                # It is a paraphrased verified lesson, NOT a dumped book page.
                explanation=case["feedback"]
                db.execute("""
                    INSERT INTO private_remediation
                    (model_id,edition,rule_id,source,explanation,failures,updated_at)
                    VALUES (?,?,?,?,?,1,?)
                    ON CONFLICT(model_id,edition,rule_id)
                    DO UPDATE SET failures=failures+1,
                                  source=excluded.source,
                                  explanation=excluded.explanation,
                                  updated_at=excluded.updated_at
                """,(model_id,case["edition"],case["rule_id"],source,explanation,now))
                if repeat_after_feedback:
                    learned=prior_lesson(db,model_id,case["edition"],case["rule_id"])
                    retry=_model_answer(model,question,learned)
                    retry_choice=retry.get("choice")
                    if retry_choice not in {"A","B","C"}:
                        retry_choice=None
                    retry_ok=retry_choice==case["correct"]
                    coached_ok+=int(retry_ok)
                    result["same_case_retry_correct"]=retry_ok
                    db.execute(
                        "INSERT INTO practice_attempts(model_id,edition,rule_id,case_id,choice,"
                        "success,coaching_visible,attempted_at) VALUES (?,?,?,?,?,?,?,?)",
                        (model_id,case["edition"],case["rule_id"],case["id"],retry_choice,
                         int(retry_ok),1,datetime.now(timezone.utc).isoformat()),
                    )
            results.append(result)
        db.commit()
        stored=db.execute(
            "SELECT count(*) FROM private_remediation WHERE model_id=?",
            (model_id,),
        ).fetchone()[0]
    total=len(results)
    return {
        "mode":"LOCAL_GGUF_PRIVATE_ADAPTIVE_PRACTICE",
        "model_id":model_id,
        "cases":total,
        "initial_pass":initial_ok,
        "initial_pass_percent":round(initial_ok/total*100,2) if total else None,
        "corrected_same_case_retries":coached_ok,
        "private_misconception_topics":stored,
        "lessons_persist_across_sessions":True,
        "practice_attempts":results,
        "blind_new_case_score":None,
        "language_model_weights_modified":False,
        "actual_mastery_percent":None,
        "trade_action":None,
        "execution_allowed":False,
        "limitations":[
            "Supervised example correction updates PRIVATE MEMORY, not GGUF weights.",
            "Answering the SAME question after feedback does not prove generalization.",
            "The practice curriculum is public in the repo, never an unbiased blind holdout.",
            "A separate untouched exam and unseen temporal market period are necessary.",
            "No actual exchange fills, stop protection or trade authority verified.",
        ],
    }


def main()->None:
    import argparse
    p=argparse.ArgumentParser(description="Run a local GGUF sourced misconception coach")
    p.add_argument("--local-model",type=Path,required=True)
    p.add_argument("--notebook",type=Path,required=True)
    p.add_argument("--edition",choices=["TC1","NTD","TC2","COURSE","ALL"],default="ALL")
    p.add_argument("--max-cases",type=int,default=29)
    p.add_argument("--no-retry",action="store_true")
    p.add_argument("--gpu-layers",type=int,default=0)
    args=p.parse_args()
    from .local_model import LocalWilliamsModel
    from .study_cli import _model_identity
    answer=coach_practice(
        LocalWilliamsModel(args.local_model,gpu_layers=args.gpu_layers),
        args.notebook,model_id=_model_identity(args.local_model),
        editions=("TC1","NTD","TC2","COURSE") if args.edition=="ALL" else (args.edition,),
        max_cases=args.max_cases,repeat_after_feedback=not args.no_retry)
    print(json.dumps(answer,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
