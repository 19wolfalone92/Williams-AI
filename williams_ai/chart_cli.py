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
    p.add_argument("--coach-notebook",type=Path,
                   help="Private GGUF-specific learning corrections, same edition only")
    p.add_argument("--edition",choices=["TC1","NTD","TC2","COURSE"])
    p.add_argument("--causal-top-down",action="store_true",
                   help="Reconstruct only COMPLETE UTC H4/D1 candles from closed H1")
    a=p.parse_args()
    if a.question and not a.local_model:
        p.error("--question requires --local-model")
    if a.book_index and not a.question:
        p.error("--book-index requires --question")
    if a.study_memory and not a.question:
        p.error("--study-memory requires --question")
    if a.coach_notebook and (not a.question or not a.edition):
        p.error("--coach-notebook requires --question and --edition")

    chart=json.loads(a.chart.read_text(encoding="utf-8"))
    observations=analyze_timeframes(chart)
    print(json.dumps(observations,ensure_ascii=False,indent=2))
    topdown=None
    if a.causal_top_down:
        from .causal_timeframes import top_down_market_evidence
        h1=chart.get("timeframes",{}).get("H1",{})
        if not isinstance(h1,dict) or not isinstance(h1.get("bars"),list):
            p.error("--causal-top-down requires chart.timeframes.H1.bars")
        topdown=top_down_market_evidence(
            chart["symbol"],h1["bars"],tick_size=chart.get("tick_size"))
        print("\\n--- Разделённый подтверждённый H1/H4/D1-контекст ---")
        print(json.dumps(topdown,ensure_ascii=False,indent=2))
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
    if topdown is not None:
        evidence=topdown["timeframes"]
        parts=[]
        for tf in ("D1","H4","H1"):
            detail=evidence[tf]
            parts.append(
                f'{tf}: {detail["status"]}, {detail["market_context"]}, '
                f'ao={detail.get("ao","NO_DATA")}, '
                f'ac={detail.get("ac","NO_DATA")}, '
                f'wave_degree={detail["wave_degree"]}; '
                f'last={detail["closed_through"]}'
            )
        chart_notes=[{
            "source":"CLOSED_H1_AGGREGATED_H4_D1:"+topdown["as_of"],
            "excerpt":" | ".join(parts)+
                      ". Нет подтверждённой вложенной разметки Elliott; "
                      "конфликт старших ТФ — наблюдение, а не запрет начального TC2 WM."
        }]+chart_notes
    learned=[]
    if a.study_memory:
        from .study import recall
        learned=recall(a.study_memory,a.question,edition=a.edition,limit=1)
    coaching=[]
    if a.coach_notebook:
        from .adaptive_coach import recall_model_corrections
        from .study_cli import _model_identity
        coaching=recall_model_corrections(
            a.coach_notebook,_model_identity(a.local_model),a.edition,limit=1)
    # Primary pages always win. Wave handout is NOT another Williams book;
    # when requested, it replaces an optional paraphrase card, not a primary
    # page or the closed-candle chart evidence.
    from .wave_reference import recall_wave, recall_annotated_charts
    wave_query = any(tag in a.question.casefold() for tag in (
        "волн", "импульс", "эллиот", "зигзаг", "треуголь",
        "диагонал", "клин", "коррекц", "wave", "elliott",
        "zigzag", "triangle", "flat", "w-x-y",
    ))
    wave_notes = recall_wave(a.question, limit=1) if wave_query else []
    image_query = any(tag in a.question.casefold() for tag in (
        "схем", "картин", "рисунк", "вложенн", "син", "диверг", "скрин",
        "график", "annotation", "chart screenshot",
    ))
    annotated_notes = recall_annotated_charts(a.question,limit=1) if image_query else []
    # Prefer specifically requested hand-marked examples to a generic
    # archetype. Neither source displaces original verified book pages.
    context_note = annotated_notes if annotated_notes else wave_notes
    notes=book_pages[:2]
    if not notes:
        notes+=book_cards[:1]
    if chart_notes:
        notes+=chart_notes[:1]
    # A previous error in THIS very GGUF deserves one slot, after
    # primary source evidence and observed candles, ahead of generic
    # textbook diagrams. It cannot replace facts or authorize orders.
    notes+=coaching[:1]
    if len(notes)<4:
        notes+=context_note[:4-len(notes)]
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
