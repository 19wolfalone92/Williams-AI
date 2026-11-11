"""Williams AI 5.0 — reproducible source-grounded SFT data, NOT chart gold.

Only publicly reviewed *paraphrase cards* and pinned original PDF metadata
are used. Original copyrighted books, OCR, images and private client data
are NOT included. Exclude visually unverified diagrams, unknown labels,
future market information, synthetic PnL or invented expert approvals.

Split on the WHOLE (edition, rule_id), not on paraphrased row IDs.
Validation and test labels never enter training. Revisioned SQLite contains
independently queryable provenance and full rule-disjoint split receipts.
"""
from __future__ import annotations

from collections import Counter
from hashlib import sha256
import argparse
import json
from pathlib import Path
import random
import sqlite3

from .book_rules import registry

EDITIONS=("TC1","NTD","TC2","COURSE")
SPLITS=("train","validation","test")
EXCLUDE={"COURSE.VISUAL.NESTED_ABC"}  # requires actual figure + independent expert visual review
SEED=20261111
SCHEMA="WILLIAMS_V5_SOURCE_CARDS_ONLY_RULE_DISJOINT_SFT"
_VARIANTS=12
SAFETY=("Ни одна книжная карточка сама по себе не подтверждает текущую рыночную "
        "сделку, цену исполнения, волну Эллиотта, биржевой SL или доходность.")


def canonical(obj)->bytes:
    return json.dumps(obj,sort_keys=True,ensure_ascii=False,
                      allow_nan=False,separators=(",",":")).encode("utf-8")


def digest(obj)->str:
    return sha256(canonical(obj)).hexdigest()


def _fold(card,position):
    if card["source"]=="TC2" and position==0:
        return "validation"
    if position == 0:
        return "validation"
    if position in (1,2) and card["source"]!="COURSE":
        return "test"
    if position == 1 and card["source"]=="COURSE":
        return "test"
    return "train"


def _target(card,style,correct_letter=None):
    if style=="MCQ":
        if correct_letter is None:
            raise ValueError("CORRECT_MCQ_LETTER_REQUIRED")
        return json.dumps({
            "choice":correct_letter,
            "reason":("В "+card["source"]+" это правило имеет ограничения; "
                      "см. PDF "+str(card["pdf_pages"][0])+"."),
        },ensure_ascii=False,separators=(",",":"))
    if style=="KNOWLEDGE":
        return json.dumps({
            "status":"SOURCE_SCOPED_RULE_NOT_MARKET_CONFIRMATION",
            "edition":card["source"],"rule_id":card["id"],
            "pdf_pages":card["pdf_pages"],
            "rule":card["summary"],"limits":SAFETY,
            "execution_allowed":False,
        },ensure_ascii=False,separators=(",",":"))
    if style=="BOUNDARY":
        return json.dumps({
            "status":"INSUFFICIENT_MARKET_CONFIRMATION",
            "edition":card["source"],"rule_id":card["id"],
            "evidence":card["summary"],
            "missing":"Закрытые согласованные H1/H4/D1, подтверждения структуры и биржевые статусы.",
            "research_only":True,"execution_allowed":False,
        },ensure_ascii=False,separators=(",",":"))
    raise ValueError("UNRECOGNIZED_TRAINING_TASK")


def _sample(card,n):
    ed,rule,title=card["source"],card["id"],card["title"]
    p=card["pdf_pages"][0]
    rng=random.Random(int(digest({"rule":rule,"n":n,"seed":SEED})[:16],16))
    if n%3==0:
        style="MCQ"
        correct=card["summary"]
        # Two intentionally false claims about automatic execution and
        # edition conflation; never invent a second plausible expert rule.
        wrongs=[
            ("Достаточно заметить термин «"+title+
             "»: он уже доказывает открытие позиции и фактический fill."),
            ("Любой сигнал из "+ed+
             " без дополнительной проверки всегда совпадает с правилами других книг."),
        ]
        choices=[correct]+wrongs
        rng.shuffle(choices)
        prompt={"edition":ed,
                "question":(
                    f"Для книги {ed}: какое утверждение о «{title}» "
                    "соответствует проверенной карточке и не обещает исполнение сделки?"
                    if n<6 else
                    f"Проверь применение определения «{title}» в профиле {ed}. "
                    "Выбери одно корректное утверждение, не смешивая книги."),
                "choices":dict(zip("ABC",choices))}
        answer=_target(card,style,"ABC"[choices.index(correct)])
    elif n%3==1:
        style="KNOWLEDGE"
        prompt={"edition":ed,
                "question":(
                    f"Дай проверяемое объяснение правила «{title}» по {ed} "
                    "с PDF-страницами и ограничениями применимости."
                    if n<7 else
                    f"Что действительно известно из первоисточника {ed} "
                    f"о «{title}» и что нельзя считать доказанным?"
                )}
        answer=_target(card,style)
    else:
        style="BOUNDARY"
        prompt={"edition":ed,
                "question":(
                    f"Есть только название «{title}» и его книжная карточка {ed}. "
                    "Можно ли на её основании подтвердить открытие сделки прямо сейчас?"
                    if n<8 else
                    f"Построй исследовательское решение для «{title}» ({ed}) "
                    "при отсутствии закрытых рыночных свечей и подтверждённых ордеров."
                ),"reviewed_source_context":{
                    "rule_id":rule,"pages":card["pdf_pages"],
                    "paraphrased_rule":card["summary"],
                    "not_expert_market_gold":True}}
        answer=_target(card,style)
    return {"schema":SCHEMA,
        "case_id":"V5-"+digest({"r":rule,"v":n,"seed":SEED})[:16],
        "rule_id":rule,"edition":ed,"pdf_pages":list(card["pdf_pages"]),
        "source_sha256":card["source_sha256"],
        "review_status":"PAGE_CHECKED_RULE_PARAPHRASE_NOT_INDEPENDENT_MARKET_GOLD",
        "split":card["split"],"variant":n,"task_type":style,
        "prompt":prompt,"answer":answer,
        "source_excerpt_in_training":False,
        "visual_semantics_independently_verified":False,
        "chart_expert_gold":False,
        "eligible_for_sft":card["split"]=="train",
        "execution_allowed":False}


def build():
    source=registry()
    books={b["id"]:b for b in source["books"]}
    rules=[r for r in source["rule_cards"]
           if r["status"]=="REVIEWED_PAGES" and r["id"] not in EXCLUDE]
    omitted=[r["id"] for r in source["rule_cards"] if r not in rules]
    mapped=[]
    for ed in EDITIONS:
        cards=sorted((c for c in rules if c["source"]==ed),key=lambda c:c["id"])
        if len(cards)<4:
            raise ValueError("INSUFFICIENT_PER_BOOK_CHECKED_RULE_CARDS")
        # Seeded selection BEFORE generating a single paraphrased question.
        order=cards[:]
        random.Random(SEED+EDITIONS.index(ed)*1009).shuffle(order)
        for i,c in enumerate(order):
            if any(not isinstance(p,int) or not 1<=p<=books[ed]["pages"]
                   for p in c["pdf_pages"]):
                raise ValueError("UNPINNED_RULE_CARD_PAGE")
            mapped.append({**c,"source_sha256":books[ed]["sha256"],
                           "split":_fold(c,i)})
    counts=Counter((c["source"],c["split"]) for c in mapped)
    if any(counts[(ed,split)]<=0 for ed in EDITIONS for split in SPLITS):
        raise ValueError("ALL_FOUR_BOOKS_IN_EACH_FROZEN_SPLIT_REQUIRED")
    out=[_sample(card,v) for card in mapped for v in range(_VARIANTS)]
    checks={part:[x for x in out if x["split"]==part] for part in SPLITS}
    ids=[x["case_id"] for x in out]
    prompts=[digest({"prompt":x["prompt"],"edition":x["edition"]}) for x in out]
    owners={}
    for x in out:
        old=owners.setdefault((x["edition"],x["rule_id"]),x["split"])
        if old!=x["split"]:
            raise ValueError("TRAIN_VALIDATION_TEST_RULE_LEAKAGE")
    if len(ids)!=len(set(ids)) or len(prompts)!=len(set(prompts)):
        raise ValueError("DUPLICATED_QUESTION_OR_CASE_ACROSS_FROZEN_SPLITS")
    if min(len(checks[x]) for x in SPLITS)<10:
        raise ValueError("INSUFFICIENT_DEV_OR_HOLDOUT")
    manifest={
        "schema":SCHEMA,"seed":SEED,"variants_per_rule":_VARIANTS,
        "source_registry_sha256":digest(source),
        "sources":[{"book":b["id"],"pdf_sha256":b["sha256"],
                    "pages":b["pages"]} for b in source["books"]],
        "excluded_rules":omitted,
        "approved_market_labels":0,
        "independently_reviewed_diagrams":0,
        "split_rule_ids":{part:sorted(x["rule_id"] for x in mapped
                    if x["split"]==part) for part in SPLITS},
        "split_rows":{part:len(checks[part]) for part in SPLITS},
        "split_prompt_sha256":{part:digest([x["prompt"] for x in checks[part]])
                               for part in SPLITS},
        "split_gold_sha256":{part:digest([x["answer"] for x in checks[part]])
                             for part in SPLITS},
        "data_quality_gate":"RULE_CARDS_ONLY_CONDITIONAL_NOT_EXPERT_CHART_GOLD",
        "future_data_used":False,"test_answers_allowed_into_gpu":False,
        "source_pdf_text_or_images_in_dataset":False,
        "test_locked_before_training":True,
        "live_trading_allowed":False}
    return out,manifest


def export_database(path:Path):
    rows,manifest=build()
    if path.exists() or path.is_symlink():
        raise FileExistsError("NEW_VERSIONED_TRAINING_DATABASE_REQUIRED")
    path.parent.mkdir(parents=True,exist_ok=True)
    import os
    fd=os.open(path,os.O_CREAT|os.O_EXCL|os.O_RDWR,0o600)
    os.close(fd)
    try:
        with sqlite3.connect(path) as db:
            db.execute("""CREATE TABLE rules(
                edition TEXT NOT NULL,rule_id TEXT NOT NULL,
                split TEXT NOT NULL,source_sha256 TEXT NOT NULL,
                pdf_pages TEXT NOT NULL,
                review_status TEXT NOT NULL,
                PRIMARY KEY(edition,rule_id))""")
            db.execute("""CREATE TABLE curriculum(
                case_id TEXT PRIMARY KEY,edition TEXT NOT NULL,rule_id TEXT NOT NULL,
                split TEXT NOT NULL,task_type TEXT NOT NULL,
                variant INTEGER NOT NULL,prompt_json TEXT NOT NULL,
                gold_answer TEXT NOT NULL,content_sha256 TEXT NOT NULL,
                eligible_for_sft INTEGER NOT NULL,
                expert_market_gold INTEGER NOT NULL DEFAULT 0,
                FOREIGN KEY(edition,rule_id) REFERENCES rules(edition,rule_id))""")
            db.execute("""CREATE TABLE manifest(key TEXT PRIMARY KEY,value_json TEXT NOT NULL)""")
            db.execute("PRAGMA foreign_keys=ON")
            rule_cards={x["id"]:x for x in registry()["rule_cards"]}
            for split,rules in manifest["split_rule_ids"].items():
                for rule_id in rules:
                    c=rule_cards[rule_id]
                    row=next(x for x in rows if x["rule_id"]==rule_id)
                    db.execute("INSERT INTO rules VALUES(?,?,?,?,?,?)",
                        (c["source"],rule_id,split,row["source_sha256"],
                        json.dumps(c["pdf_pages"]),"PAGE_CHECKED_NOT_MARKET_GOLD"))
            for x in rows:
                db.execute("INSERT INTO curriculum VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                    (x["case_id"],x["edition"],x["rule_id"],x["split"],
                     x["task_type"],x["variant"],
                     json.dumps(x["prompt"],ensure_ascii=False),x["answer"],
                     digest(x),int(x["eligible_for_sft"]),0))
            for key,value in manifest.items():
                db.execute("INSERT INTO manifest VALUES(?,?)",
                    (key,json.dumps(value,ensure_ascii=False,sort_keys=True)))
            db.commit()
        # Guard private DB access even if user umask is lax.
        path.chmod(0o600)
        return manifest
    except Exception:
        path.unlink(missing_ok=True)
        raise


def main():
    parser=argparse.ArgumentParser(description="Build read-only Williams source-card training DB")
    parser.add_argument("--database",type=Path,required=True)
    parser.add_argument("--manifest",type=Path)
    a=parser.parse_args()
    manifest=export_database(a.database)
    if a.manifest:
        if a.manifest.exists():
            raise FileExistsError("NEW_MANIFEST_DESTINATION_REQUIRED")
        a.manifest.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),
                              encoding="utf-8")
    print("WILLIAMS_V5_SFT_DATABASE",
          json.dumps({"path":str(a.database),"split_rows":manifest["split_rows"],
                      "market_gold":manifest["approved_market_labels"],
                      "excluded_visual":manifest["excluded_rules"],
                      "live_trading":False},ensure_ascii=False,sort_keys=True),
          flush=True)

if __name__=="__main__":
    main()
