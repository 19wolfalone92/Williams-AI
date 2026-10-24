"""Prepare chart supervision without inventing independent expert answers.

Unlike a live market capture, this historical corpus makes no live-clock or
order-execution claim. Whole symbols and whole source rules have fixed
partitions; heavily overlapping daily windows stay in the same partition.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

from .book_rules import registry
from .chart_supervision import (
    SCHEMA, checked_chart_context, check_market_partitions, digest, market_digest,
)
from .private_lora import _private_file
from .private_review_queue import source_rules_by_split

_EDS = ("TC1", "NTD", "TC2", "COURSE")
_PARTS = ("train", "validation", "test")
_QUESTION = (
    "Проанализируй {symbol} {timeframe} только до {cutoff} по {edition}, правило {rule}. "
    "Отдели наблюдаемые факты от гипотез. Объясни применимость правила, "
    "подтверждение и ценовой триггер; альтернативы волновой структуры и "
    "условие отказа от сделки. Опиши структурную защиту и условия сопровождения/выхода, "
    "если это правило их определяет. Исполнение заявки и наличие позиции неизвестны: "
    "не выдумывай fill, стоп или текущую кампанию. Численный риск без данных счёта неизвестен."
)


def build_market_review_queue(contexts: list[dict[str, Any]], *,
                              symbol_partitions: dict[str, str],
                              utc_days: int = 8) -> dict[str, Any]:
    if (not isinstance(contexts, list) or not 3 <= len(contexts) <= 30
            or isinstance(utc_days, bool) or not isinstance(utc_days, int)
            or not 8 <= utc_days <= 30
            or not isinstance(symbol_partitions, dict)
            or set(symbol_partitions.values()) != set(_PARTS)):
        raise ValueError("PREDECLARED_THREE_PARTITION_HISTORICAL_CORPUS_REQUIRED")
    checked = []
    seen = set()
    common_days = None
    venues = set()
    for ctx in contexts:
        view = checked_chart_context(ctx)
        symbol = view["symbol"]
        if symbol in seen or ctx["timeframe"] != "H1":
            raise ValueError("UNIQUE_H1_SYMBOLS_REQUIRED_FOR_DAILY_REVIEW_QUEUE")
        seen.add(symbol)
        venues.add(view["venue"])
        ends = {}
        for i, candle in enumerate(view["candles"]):
            if i >= 219 and candle.at.date() < view["end"].date():
                ends[candle.at.date().isoformat()] = i
        common_days = set(ends) if common_days is None else common_days & set(ends)
        checked.append((ctx, ends))
    if seen != set(symbol_partitions) or len(venues) != 1:
        raise ValueError("PARTITION_SYMBOLS_OR_MARKET_VENUE_MISMATCH")
    if not common_days or len(common_days) < utc_days:
        raise ValueError("NOT_ENOUGH_COMMON_COMPLETED_REVIEW_DATES")
    days = sorted(common_days)[-utc_days:]
    rules = source_rules_by_split()
    books = {b["id"]: b for b in registry()["books"]}
    rows = []
    partition_windows = []
    for symbol_index, (ctx, ends) in enumerate(sorted(checked, key=lambda x: x[0]["symbol"])):
        split = symbol_partitions[ctx["symbol"]]
        for index, day in enumerate(days):
            edition = _EDS[index % len(_EDS)]
            pool = rules[split][edition]
            rule = pool[(symbol_index + index // len(_EDS)) % len(pool)]
            bars = [{k: b[k] for k in ("close_time", "open", "high", "low", "close", "volume", "closed")}
                    for b in ctx["bars"][:ends[day] + 1]]
            market = {"schema": SCHEMA, "symbol": ctx["symbol"], "venue": ctx["venue"],
                      "timeframe": "H1", "tick_size": ctx["tick_size"],
                      "cutoff": bars[-1]["close_time"], "bars": bars}
            market["market_sha256"] = market_digest(market)
            partition_windows.append({**checked_chart_context(market),
                                      "split": split, "edition": edition})
            rows.append({
                "case_id": f"market-{ctx['symbol']}-{market['market_sha256'][:20]}",
                "edition": edition, "rule_id": rule["id"],
                "pdf_page": rule["pdf_pages"][0], "source_sha256": books[edition]["sha256"],
                "question": _QUESTION.format(symbol=ctx["symbol"], timeframe="H1",
                    cutoff=market["cutoff"], edition=edition, rule=rule["id"]),
                "answer": "", "review_status": "PENDING_INDEPENDENT_REVIEW",
                "reviewer_id": "review-required", "split": split,
                "market_context": market, "independent_reviews": [],
            })
    audit = check_market_partitions(partition_windows)
    manifest = {
        "schema": "WILLIAMS_AI_PENDING_MARKET_SUPERVISION_V1",
        "cases": len(rows), "distinct_market_symbols": len(seen), "utc_dates": days,
        "source_context_sha256": sorted(ctx["market_sha256"] for ctx in contexts),
        "symbol_partitions": dict(sorted(symbol_partitions.items())),
        "source_rule_partitions_shared_with_text_queue": True,
        "editions": dict(Counter(r["edition"] for r in rows)),
        **audit, "queue_sha256": digest(rows), "labels_completed": 0,
        "historical_not_prospective": True, "live_exchange_clock_verified": False,
        "overlapping_same_partition_cases_are_not_independent_observations": True,
        "do_not_train_until_two_humans_approve_each_completed_answer": True,
        "trade_action": None, "execution_allowed": False,
    }
    manifest["manifest_sha256"] = digest(manifest)
    return {"manifest": manifest, "lessons": rows}


def write_pending_market_queue(bundle: dict[str, Any], output: Path) -> None:
    dest = _private_file(output, must_exist=False)
    if dest.suffix != ".jsonl" or not dest.parent.is_dir():
        raise ValueError("EXISTING_PRIVATE_DIRECTORY_AND_NEW_JSONL_REQUIRED")
    # The exclusive open keeps a frozen queue intact on competing writers.
    with dest.open("x", encoding="utf-8") as stream:
        dest.chmod(0o600)
        for row in bundle["lessons"]:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Private unlabelled 4-book chart lessons, no orders")
    parser.add_argument("--contexts", type=Path, required=True,
                        help="JSON array of exact closed historical chart contexts")
    parser.add_argument("--partitions", type=Path, required=True,
                        help="Frozen JSON object: symbol -> train/validation/test")
    parser.add_argument("--output-private-jsonl", type=Path, required=True)
    args = parser.parse_args()
    contexts = json.loads(_private_file(args.contexts).read_text(encoding="utf-8"))
    partitions = json.loads(_private_file(args.partitions).read_text(encoding="utf-8"))
    bundle = build_market_review_queue(contexts, symbol_partitions=partitions)
    write_pending_market_queue(bundle, args.output_private_jsonl)
    print(json.dumps(bundle["manifest"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
