"""Create a LOCAL 4-book expert review queue for Qwen training proposals.

Raw rule summaries are source-derived, NOT automatically correct chart labels.
This never marks generated items as independent expert reviewed. A real human
reviewer must check each original page, figure, competing hypothesis and
proposed answer before the gated private_lora loader accepts it.

No original copyrighted text is bundled; queue saved OUTSIDE git.
"""
from __future__ import annotations
import argparse
from collections import defaultdict,Counter
import hashlib
import json
from pathlib import Path
from typing import Any
from .book_rules import registry
from .private_lora import _private_file

_TASKS=(
    ("Что именно утверждает указанное правило? Объясни условия, подтверждение и его границы.",
     "Источник содержит следующую проверяемую интерпретацию: "),
    ("Как применить правило в историческом тесте без заглядывания в будущие свечи?",
     "Первоначальная проверяемая книжная трактовка: "),
    ("Какие конкурирующие волновые и рыночные сценарии ещё нельзя исключить?",
     "Книжная трактовка не является гарантией будущего результата: "),
    ("Что здесь нельзя считать подтверждением исполнения ордера или стопа?",
     "Первоначальный разбор книжного условия, а не биржевой квитанции: "),
)


def source_rules_by_split()->dict[str,dict[str,list[dict[str,Any]]]]:
    """One partition assignment shared by text and market lesson queues."""
    grouped=defaultdict(list)
    for card in registry()["rule_cards"]:
        if card["status"]=="REVIEWED_PAGES":
            grouped[card["source"]].append(card)
    result={k:{} for k in ("train","validation","test")}
    for edition,cards in grouped.items():
        rules=sorted(cards,key=lambda r:r["id"])
        train=max(1,len(rules)*3//5)
        validation=max(1,(len(rules)-train)//2)
        for split,items in (("train",rules[:train]),
                            ("validation",rules[train:train+validation]),
                            ("test",rules[train+validation:])):
            if not items:raise ValueError("FOUR_BOOKS_NEED_WHOLE_RULE_HOLDOUTS")
            result[split][edition]=items
    if any(set(editions)!={"TC1","NTD","TC2","COURSE"}
           for editions in result.values()):
        raise ValueError("FOUR_BOOK_RULE_REFERENCE_REQUIRED")
    return result


def propose_private_book_review_queue(output_path:str|Path)->dict[str,Any]:
    """Deterministic per-rule split; no single rule shared train/test."""
    dest=_private_file(output_path,must_exist=False)
    if dest.suffix.lower()!=".jsonl" or dest.exists() or not dest.parent.is_dir():
        raise ValueError("NEW_PRIVATE_JSONL_DESTINATION_REQUIRED")
    books={b["id"]:b for b in registry()["books"]}
    partitioned=source_rules_by_split()
    output=[]
    for edition in sorted(books):
        i=0
        for split in ("train","validation","test"):
          for rule in partitioned[split][edition]:
            for j,(q,a) in enumerate(_TASKS,1):
                item={
                    "case_id":f"pending-{edition.lower()}-{i:03d}-{j:02d}",
                    "edition":edition,"rule_id":rule["id"],
                    "pdf_page":rule["pdf_pages"][0],
                    "source_sha256":books[edition]["sha256"],
                    "question":f"{q} Укажи страницу и правило {rule['id']}.",
                    "answer":a+rule["summary"],
                    "review_status":"PENDING_INDEPENDENT_REVIEW",
                    "reviewer_id":"review-required",
                    "split":split,
                }
                output.append(item)
            i+=1
    # All items deliberately fail checked_private_training_data until
    # independent review and actual improvements to answers are completed.
    payload="".join(json.dumps(x,ensure_ascii=False,sort_keys=True)+"\n"
                    for x in output)
    with dest.open("x",encoding="utf-8") as stream:
        dest.chmod(0o600)
        stream.write(payload)
    return {
        "mode":"PENDING_4_BOOK_INDEPENDENT_EXPERT_REVIEW_QUEUE",
        "proposals":len(output),
        "splits":dict(sorted(Counter(x["split"] for x in output).items())),
        "source_editions":dict(sorted(Counter(x["edition"] for x in output).items())),
        "dataset_sha256":hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "all_examples_human_reviewed":False,
        "do_not_use_as_qwen_training_labels_yet":True,
        "location":str(dest),
        "trade_action":None,"execution_allowed":False,
    }


def main()->None:
    ap=argparse.ArgumentParser(description="Prepare private unapproved four-book expert label queue")
    ap.add_argument("--output-private-jsonl",type=Path,required=True)
    args=ap.parse_args()
    print(json.dumps(propose_private_book_review_queue(args.output_private_jsonl),
                     ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
