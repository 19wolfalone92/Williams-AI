"""Local study/recall, opt-in web research and trainable paper experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .book_index import search_pages
from .book_rules import lookup_rules
from .study import recall, recall_web, remember_web, study_batch, study_report
from .web_research import research


def main():
    parser = argparse.ArgumentParser(description="Williams AI learning laboratory (no orders)")
    sub = parser.add_subparsers(dest="action", required=True)
    study = sub.add_parser("study", help="Resume reading private book chunks using local GGUF")
    study.add_argument("--books", required=True)
    study.add_argument("--memory", required=True)
    study.add_argument("--model", type=Path, required=True)
    study.add_argument("--edition", choices=["TC1", "NTD", "TC2", "COURSE"], required=True)
    study.add_argument("--budget", type=int, default=4, help="Maximum source chunks per invocation, 1..100")
    study.add_argument("--until-complete", action="store_true", help="Continue bounded batches automatically; each completed chunk is checkpointed")
    report = sub.add_parser("report")
    report.add_argument("--books", required=True)
    report.add_argument("--memory", required=True)
    ask = sub.add_parser("ask", help="Books first, study memory second, optional public lookup")
    ask.add_argument("question")
    ask.add_argument("--books")
    ask.add_argument("--memory")
    ask.add_argument("--edition", choices=["TC1", "NTD", "TC2", "COURSE"])
    ask.add_argument("--web", action="store_true", help="Send only this public question to research providers")
    train = sub.add_parser("learn-paper", help="Train a local statistical scorer, evaluate purged holdouts")
    train.add_argument("chart", type=Path)
    train.add_argument("--timeframe", choices=["M5", "M15", "H1", "H4", "D1"], required=True)
    train.add_argument("--horizon", type=int, default=12)
    opts = parser.parse_args()
    if opts.action == "study":
        from .local_model import LocalWilliamsModel
        from .study_cli import _model_identity
        model_id = _model_identity(opts.model)
        model = LocalWilliamsModel(opts.model)
        result = study_batch(opts.books, opts.memory, opts.edition, model,
                             model_id=model_id, budget=opts.budget)
        if opts.until_complete:
            totals = {"studied_chunks": result["studied_chunks"], "failed_chunks": result["failed_chunks"]}
            for _ in range(1000):
                if result["studied_chunks"] + result["failed_chunks"] == 0:
                    break
                result = study_batch(opts.books, opts.memory, opts.edition, model,
                                     model_id=model_id, budget=opts.budget)
                for key in totals:
                    totals[key] += result[key]
            result = {**result, **totals, "report": study_report(opts.books, opts.memory)}
    elif opts.action == "report":
        result = study_report(opts.books, opts.memory)
    elif opts.action == "ask":
        pages = search_pages(opts.books, opts.question, edition=opts.edition, limit=3) if opts.books else []
        learned = recall(opts.memory, opts.question, edition=opts.edition,
                         book_index=opts.books) if opts.memory else []
        web = research(opts.question, enabled=opts.web)
        if opts.memory and opts.web:
            remember_web(opts.memory, web)
        result = {"question": opts.question, "primary_pages": pages,
                  "reviewed_rule_cards": lookup_rules(opts.question, edition=opts.edition, max_cards=3),
                  "unverified_study_notes": learned,
                  "web": web,
                  "cached_web_evidence": recall_web(opts.memory, opts.question) if opts.memory else [],
                  "trade_action": None, "execution_allowed": False}
    else:
        from .paper_learning import learn_from_history
        chart = json.loads(opts.chart.read_text(encoding="utf-8"))
        # Respect chart's declared as_of, not only the timestamp of its last row.
        from .chart_reader import validate_closed_bars
        rows = chart["timeframes"][opts.timeframe]["bars"]
        validate_closed_bars(rows, opts.timeframe, chart["as_of"])
        result = learn_from_history(rows, opts.timeframe, horizon=opts.horizon)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
