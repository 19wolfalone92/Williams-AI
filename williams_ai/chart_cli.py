"""Offline chart reading with edition-cited local GGUF explanations.

Usage:
python -m williams_ai.chart_cli examples/ai_chart_example.json
python -m williams_ai.chart_cli examples/ai_chart_example.json --question "Где рынок?" --local-model /private/qwen.gguf
python -m williams_ai.chart_cli examples/ai_chart_example.json --question "Где рынок?" --local-model /private/qwen.gguf --book-index /private/williams_books.db

The model is strictly an explainer of verified candle evidence. No orders.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from .analyst import build_report
from .chart_reader import analyze_timeframes
from .book_rules import lookup_rules


def _to_snapshot(result: dict) -> dict:
    tfs = {}
    for name, data in result["timeframes"].items():
        if not data["last_closed_at"]:
            continue
        obs = data["evidence"]
        tfs[name] = {
            "last_closed_at":data["last_closed_at"],
            "ao":obs.get("ao"),
            "ac":obs.get("ac"),
            "phase":("TREND" if obs.get("market_context") in ("UP_CONTEXT","DOWN_CONTEXT") else "UNKNOWN"),
            "wave_hypothesis":(
                "Unconfirmed W4 possibility; need validated wave degree"
                if data["hypotheses"] else "degree remains unknown"
            ),
        }
    return {
        "symbol":result["symbol"],"as_of":result["as_of"],
        "timeframes":tfs,"signals":[],
    }


def _chart_notes(result: dict) -> list[dict[str,str]]:
    notes=[]
    for tf,data in result["timeframes"].items():
        if not data["evidence"]:
            continue
        evidence = data["evidence"]
        notes.append({
            "source": f'OHLCV:{tf}:{data["last_closed_at"]}',
            "excerpt": (
                f'Контекст {evidence["market_context"]}; AO={evidence["ao"]}; '
                f'AC={evidence["ac"]}; zone={evidence["zone"]}; '
                f'jaw={evidence["jaw_visible"]}; teeth={evidence["teeth_visible"]}; '
                f'lips={evidence["lips_visible"]}. '
                f'Последние сигнальные гипотезы: ' +
                ", ".join(
                    f'{candidate["type"]}:{candidate["side"]}:{candidate["state"]}'
                    for candidate in data["candidates"][-4:]
                ) +
                '. Ни одна не является утверждённой сделкой.'
            )
        })
    return notes


def main():
    p=argparse.ArgumentParser(description="Williams AI causal chart reader")
    p.add_argument("chart",type=Path)
    p.add_argument("--question")
    p.add_argument("--local-model",type=Path)
    p.add_argument("--book-index",type=Path,help="Private index OUTSIDE git, optional")
    p.add_argument("--study-memory",type=Path,help="Private unverified study notes, optional")
    p.add_argument("--edition",choices=["TC1","NTD","TC2","COURSE"])
    a=p.parse_args()
    if a.question and not a.local_model:
        p.error("--question requires --local-model")
    if a.book_index and not a.question:
        p.error("--book-index requires --question")
    if a.study_memory and not a.question:
        p.error("--study-memory requires --question")

    chart=json.loads(a.chart.read_text(encoding="utf-8"))
    observations=analyze_timeframes(chart)
    print(json.dumps(observations,ensure_ascii=False,indent=2))
    if not a.question:
        return
    from .local_model import LocalWilliamsModel
    # Reserve the scarce inference-context slots for primary book pages.
    # The earlier order dropped all book citations on >=4 timeframes.
    book_pages=[]
    if a.book_index:
        from .book_index import search_pages
        book_pages=search_pages(a.book_index,a.question,edition=a.edition,limit=2)
    book_cards=lookup_rules(a.question,edition=a.edition,max_cards=2)
    chart_notes=_chart_notes(observations)
    learned=[]
    if a.study_memory:
        from .study import recall
        learned=recall(a.study_memory,a.question,edition=a.edition,limit=1)
    # Primary pages always win. Wave handout is NOT another Williams book;
    # when requested, it replaces an optional paraphrase card, not a primary
    # page or the closed-candle chart evidence.
    from .wave_reference import recall_wave
    wave_query = any(tag in a.question.casefold() for tag in (
        "волн", "импульс", "эллиот", "зигзаг", "треуголь",
        "диагонал", "клин", "коррекц", "wave", "elliott",
        "zigzag", "triangle", "flat", "w-x-y",
    ))
    wave_notes = recall_wave(a.question, limit=1) if wave_query else []
    notes=book_pages[:2]
    if not notes:
        notes+=book_cards[:1]
    if chart_notes:
        notes+=chart_notes[:1]
    notes+=wave_notes
    if len(notes)<4 and book_pages:
        notes+=book_cards[:1]
    if len(notes)<4:
        notes+=learned[:4-len(notes)]
    notes=notes[:4]
    model=LocalWilliamsModel(a.local_model)
    report=build_report(_to_snapshot(observations))
    print("\n--- Локальное объяснение (не торговый приказ) ---")
    print(model.answer(a.question,report,notes=notes))
    print("\nКонтрольные источники (только реально переданные выдержки):")
    for item in notes[:4]:
        print("- "+item["source"])


if __name__=="__main__":
    main()
