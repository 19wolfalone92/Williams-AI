"""Private analyst drafts, explicitly separate from independent expert gold.

Reuses the causal book-scoped teacher. Checks the private source index and
records extraction/OCR limitations without exporting its text. No reviews
are invented and pending lessons are never modified or approved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any

from .book_rules import registry
from .chart_supervision import checked_chart_context, digest
from .practical_market_teacher import rehearse_trade_reasoning
from .private_lora import _private_file

_SCENARIOS = {
    "CONTINUATION_UP": "продолжение роста",
    "CONTINUATION_DOWN": "продолжение снижения",
    "CORRECTION_OR_COMPLEX_RANGE": "коррекция или сложный диапазон",
    "POSSIBLE_REVERSAL_UP": "возможный разворот вверх",
    "POSSIBLE_REVERSAL_DOWN": "возможный разворот вниз",
}
_NOTES = {
    "TC1": "Волновая степень и полный набор волшебных пуль не подтверждены. "
           "Биржевой volume не является доказанным tick volume; Squat не додумывать.",
    "NTD": "Первый вход требует собственного фрактального допуска NTD; "
           "AO/AC/Zone не заменяют его. Книжное условие выхода и биржевой SL различаются.",
    "TC2": "WM1/WM2/WM3 — кандидаты до ценового триггера. "
           "SL структурный; доливка только при прибыли и подтверждённой защите. "
           "Противоположный кандидат без исполненного выхода не закрывает кампанию.",
    "COURSE": "Вложенные степени и графические Hump/Thumb не подтверждены визуально. "
              "Непроверенный OCR не считать надёжным прочтением рисунков курса.",
}


def page_receipt(index: Path, lesson: dict[str, Any]) -> dict[str, Any]:
    """Read-only provenance checks; matching index metadata is not PDF QA."""
    path = _private_file(index)
    edition, page = lesson["edition"], lesson["pdf_page"]
    cards = {r["id"]: r for r in registry()["rule_cards"]}
    books = {b["id"]: b for b in registry()["books"]}
    rule = cards.get(lesson["rule_id"])
    if (edition not in books or not rule or rule["source"] != edition
            or page not in rule["pdf_pages"]
            or lesson["source_sha256"] != books[edition]["sha256"]):
        raise ValueError("DRAFT_SOURCE_EDITION_OR_PAGE_MISMATCH")
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as conn:
        volume = conn.execute("SELECT sha256,total_pages FROM volumes WHERE book=?", (edition,)).fetchone()
        if (not volume or volume[0] != lesson["source_sha256"]
                or not 1 <= page <= volume[1]):
            raise ValueError("PRIVATE_INDEX_SOURCE_IDENTITY_MISMATCH")
        fragments = conn.execute(
            "SELECT content,evidence_status FROM excerpts WHERE book=? AND pdf_page=? ORDER BY rowid",
            (edition, page)).fetchall()
        provenance = conn.execute(
            "SELECT source_text_method,ocr_text_unreviewed,text_chars,normalized_text_sha256 "
            "FROM page_provenance WHERE book=? AND pdf_page=?", (edition, page)).fetchone()
    if not provenance or (not fragments and provenance[2] > 0):
        raise ValueError("PRIVATE_SOURCE_PAGE_HAS_NO_USABLE_TEXT")
    text = "".join(x[0] for x in fragments)
    return {"edition": edition, "pdf_page": page, "source_sha256": volume[0],
            "indexed_page_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "indexed_page_characters": len(text), "text_method": provenance[0],
            "source_page_text_available": bool(text.strip()),
            "ocr_unreviewed": bool(provenance[1]),
            "index_evidence_statuses": sorted({x[1] for x in fragments}),
            "requires_independent_semantic_and_figure_review": True,
            "original_pdf_bytes_or_figures_rechecked_here": False,
            "raw_book_text_exported": False}


def draft_review(lesson: dict[str, Any], *, index: Path) -> dict[str, Any]:
    if (lesson.get("review_status") != "PENDING_INDEPENDENT_REVIEW"
            or lesson.get("answer") != "" or lesson.get("independent_reviews") != []):
        raise ValueError("ONLY_UNLABELLED_PENDING_MARKET_LESSONS_ACCEPTED")
    context = lesson["market_context"]
    checked_chart_context(context)
    source = page_receipt(index, lesson)
    teacher = rehearse_trade_reasoning(
        context["symbol"], context["timeframe"], context["bars"],
        edition=lesson["edition"], tick_size=context["tick_size"])
    if teacher["as_of"] != context["cutoff"] or teacher["future_bars_read"] is not False:
        raise ValueError("DRAFT_CAUSAL_CUTOFF_MISMATCH")
    rule = next(r for r in registry()["rule_cards"] if r["id"] == lesson["rule_id"])
    levels = teacher["confirmed_swing_levels_not_orders"]
    alternatives = teacher["five_competing_falsifiable_scenarios"]
    facts = {
        "as_of": teacher["as_of"], "closed_bars": len(context["bars"]),
        "market_context": teacher["market_context"],
        "ao_sign": teacher["ao_sign"], "ao_direction": teacher["ao_direction"],
        "ac_direction": teacher["ac_direction"], "swing_references_not_orders": levels,
    }
    basis = (f"Рабочая интерпретация существующей карточки: {rule['summary']}"
             if source["source_page_text_available"] else
             "Текст страницы не извлечён. Источниковую интерпретацию требуется "
             "проверить по оригинальному изображению; ответ карточки не подставляется.")
    answer = (
        f"Черновик для проверки. {context['symbol']} {context['timeframe']}, "
        f"только до {context['cutoff']}. Источник: {lesson['edition']}, "
        f"PDF {lesson['pdf_page']}, правило {lesson['rule_id']}. "
        f"{basis}\n"
        f"Наблюдения: закрытых баров {len(context['bars'])}; close={levels['last_close']}; "
        f"AO знак={teacher['ao_sign']}, изменение={teacher['ao_direction']}; "
        f"AC изменение={teacher['ac_direction']}. Опорные подтверждённые экстремумы: "
        f"high={levels['high']}, low={levels['low']}; это не цены ордеров.\n"
        f"Сценарии для сопоставления: {', '.join(_SCENARIOS[x['hypothesis']] for x in alternatives)}. "
        "Их факты, противоречия и условия опровержения приведены отдельно; "
        "вероятности и точная степень Эллиотта не установлены.\n"
        f"{_NOTES[lesson['edition']]} "
        "Исполнение, позиция и подтверждённый SL неизвестны. Численный риск и объём "
        "по этим свечам не вычисляются без счёта и структурной проверки защиты. "
        "Торговое распоряжение не выдаётся. Для окончательного ответа нужны "
        "проверка исходной страницы/рисунка и две независимые рецензии."
    )
    return {
        "schema": "WILLIAMS_AI_UNAPPROVED_ANALYST_DRAFT_V1",
        "case_id": lesson["case_id"], "edition": lesson["edition"],
        "rule_id": lesson["rule_id"], "split": lesson["split"],
        "market_sha256": context["market_sha256"],
        "pending_lesson_sha256": digest(lesson), "source_receipt": source,
        "observations": facts, "alternative_scenarios_not_gold": alternatives,
        "edition_gate_not_execution": teacher["edition_specific_entry_gate"],
        "draft_answer": answer, "review_status": "MACHINE_DRAFT_NOT_EXPERT_REVIEW",
        "generated_by": "DETERMINISTIC_CAUSAL_TEACHER_NOT_QWEN_WEIGHTS",
        "source_page_requires_visual_inspection": not source["source_page_text_available"],
        "independent_reviews_completed": 0, "ready_for_weight_training": False,
        "must_not_be_shown_to_blind_reviewers_before_their_answers": True,
        "model_weights_updated": False, "trade_action": None, "execution_allowed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Private unapproved chart drafts; no training or orders")
    parser.add_argument("--pending-queue", type=Path, required=True)
    parser.add_argument("--private-book-index", type=Path, required=True)
    parser.add_argument("--private-output", type=Path, required=True)
    args = parser.parse_args()
    source = _private_file(args.pending_queue)
    output = _private_file(args.private_output, must_exist=False)
    if source.stat().st_size > 40_000_000 or output.exists():
        raise ValueError("BOUNDED_QUEUE_AND_NEW_PRIVATE_OUTPUT_REQUIRED")
    pending = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not 1 <= len(pending) <= 450:
        raise ValueError("BOUNDED_PENDING_REVIEW_QUEUE_REQUIRED")
    drafts = [draft_review(row, index=args.private_book_index) for row in pending]
    with output.open("x", encoding="utf-8") as stream:
        output.chmod(0o600)
        for row in drafts:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"drafts": len(drafts), "human_reviews": 0,
                      "trained_weights": False, "execution_allowed": False}))


if __name__ == "__main__":
    main()
