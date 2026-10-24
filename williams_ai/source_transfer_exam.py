"""Counterexamples for the actual Qwen NTD zone-color error.

These are synthetic indicator arithmetic exercises, NOT historical trades or
independent expert wave labels. Compare one frozen model without a passage
and with the original privately indexed NTD definition. No weights change;
a better source-assisted score is retrieval assistance, not trained skill.
Original book text never enters the result or a public artifact.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
from typing import Any

from .study import _source


def zone_cases(seed: int = 20261024) -> list[dict[str, Any]]:
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed <= 2**32-1:
        raise ValueError("INVALID_EXAM_SEED")
    rng = random.Random(seed)
    cases = []
    # Deliberately vary signs, candle body, choice order and both gray-zone
    # orientations. A candle rising or AO being positive is not this rule.
    for i, (up_ao, up_ac) in enumerate([(True, True), (False, False),
                                       (True, False), (False, True)] * 3):
        before_ao = rng.randint(-200, 200)
        before_ac = rng.randint(-200, 200)
        after_ao = before_ao + (1 if up_ao else -1) * rng.randint(1, 100)
        after_ac = before_ac + (1 if up_ac else -1) * rng.randint(1, 100)
        colors = ["GREEN", "RED", "GRAY"]
        rng.shuffle(colors)
        text = {"GREEN": "Зелёная зона", "RED": "Красная зона", "GRAY": "Серая зона"}
        choices = {letter: text[color] for letter, color in zip("ABC", colors)}
        truth = "GREEN" if up_ao and up_ac else "RED" if not up_ao and not up_ac else "GRAY"
        cases.append({
            "public": {
                "id": f"NTD-ZONE-TRANSFER-{i+1:02d}", "edition": "NTD",
                "question": (
                    "Все значения даны на двух последовательных закрытых барах. "
                    f"AO: {before_ao/100:.2f} → {after_ao/100:.2f}; "
                    f"AC: {before_ac/100:.2f} → {after_ac/100:.2f}. "
                    f"Тело последней ценовой свечи {'растущее' if i % 2 else 'падающее'}. "
                    "Какой цвет зоны NTD на последнем баре? Объясни, что сравнил."
                ),
                "choices": choices,
            },
            "expected": "ABC"[colors.index(truth)], "truth": truth,
            "inputs": {"ao": [before_ao, after_ao], "ac": [before_ac, after_ac]},
        })
    return cases


def _zone_source(book_index: Path) -> tuple[list[dict[str, str]], str]:
    source = _source(book_index, "NTD")
    try:
        rows = source.execute("SELECT content FROM excerpts WHERE book=? AND pdf_page=?",
                              ("NTD", 91)).fetchall()
    finally:
        source.close()
    passages = [text for (text,) in rows
                if "Зона формируется" in text and "Серой Зоне" in text]
    if len(passages) != 1 or not 100 <= len(passages[0]) <= 1350:
        raise ValueError("ORIGINAL_NTD_ZONE_DEFINITION_REQUIRES_SOURCE_REVIEW")
    passage = passages[0]
    notes = [{"source": "NTD:RULE:NTD.ZONE", "excerpt": passage[i:i+450]}
             for i in range(0, len(passage), 450)]
    if any(len(x["excerpt"]) < 10 for x in notes):
        raise ValueError("SOURCE_FRAGMENT_BOUNDARY_REQUIRES_REVIEW")
    return notes, hashlib.sha256(passage.encode()).hexdigest()


def run_zone_transfer_exam(model: Any, book_index: Path, *,
                           seed: int = 20261024) -> dict[str, Any]:
    cases = zone_cases(seed)
    protocol = hashlib.sha256(json.dumps(cases, sort_keys=True, ensure_ascii=False,
                                         separators=(",", ":")).encode()).hexdigest()
    notes, source_hash = _zone_source(book_index)
    results = {}
    for phase in ("without_original_passage", "with_original_passage"):
        graded = []
        for case in cases:
            try:
                kwargs = {"prior_lessons": notes} if phase == "with_original_passage" else {}
                # The expected choice and truth never enter the model input.
                answer = model.answer_exam_case(case["public"], **kwargs)
                if not isinstance(answer, dict) or answer.get("choice") not in {"A", "B", "C"}:
                    raise ValueError("MALFORMED_EXAM_ANSWER")
                chosen = answer["choice"]
                reason = answer.get("reason", "")
                reason = reason[:500] if isinstance(reason, str) else ""
                failure = None
            except (ValueError, TypeError, KeyError, RuntimeError) as exc:
                # Never retain arbitrary model exceptions containing private text.
                chosen, reason, failure = "", "", type(exc).__name__
            item = {"id": case["public"]["id"], "truth": case["truth"],
                    "expected": case["expected"], "chosen": chosen,
                    "correct": chosen == case["expected"],
                    # A free-form explanation can repeat the licensed
                    # passage verbatim. Export a digest, never its text.
                    "explanation_sha256": hashlib.sha256(reason.encode()).hexdigest(),
                    "explanation_characters": len(reason), "failure": failure}
            graded.append(item)
            print("NTD_SOURCE_APPLICATION_CASE", phase, item["id"],
                  item["correct"], failure, flush=True)
        results[phase] = {
            "correct": sum(x["correct"] for x in graded), "total": len(graded),
            "malformed_or_failed": sum(x["failure"] is not None for x in graded),
            "chosen_letter_counts":dict(sorted(Counter(x["chosen"] for x in graded
                                                        if x["chosen"]).items())),
            "constant_choice_on_varied_answer_key":(
                len({x["expected"] for x in graded})>1
                and all(x["chosen"] for x in graded)
                and len({x["chosen"] for x in graded})==1),
            "results": graded,
        }
    return {
        "mode": "PAIRED_NTD_ORIGINAL_SOURCE_APPLICATION_DIAGNOSTIC",
        "protocol_sha256": protocol, "seed": seed,
        "source_edition": "NTD", "original_pdf_page": 91,
        "private_source_passage_sha256": source_hash,
        "private_source_passage_exported": False,
        "synthetic_indicator_cases_not_historical_charts": len(cases),
        "class_distribution": dict(Counter(x["truth"] for x in cases)),
        "model_calls_requested": 2*len(cases),
        "phases": results,
        "source_assisted_correct_count_change": (
            results["with_original_passage"]["correct"] -
            results["without_original_passage"]["correct"]),
        "same_questions_reused_to_measure_source_assistance_not_blind_generalization": True,
        "numeric_correctness_does_not_verify_explanation_or_visual_understanding": True,
        "original_passage_assistance_not_weight_training": True,
        "model_weights_modified": False,
        "independent_wave_labels_or_market_profit_verified": False,
        "trade_action": None, "execution_allowed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Private original NTD rule application, no trading")
    parser.add_argument("--book-index", type=Path, required=True)
    parser.add_argument("--local-model", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--seed", type=int, default=20261024)
    parser.add_argument("--learning-mode",choices=["auto","legacy_json","reasoning"],default="legacy_json")
    parser.add_argument("--reasoning-budget",type=int,default=2048)
    args = parser.parse_args()
    from .real_gguf_acceptance import actual_gguf_sha256
    from .local_model import LocalWilliamsModel
    weights = actual_gguf_sha256(args.local_model, args.expected_sha256)
    result = run_zone_transfer_exam(LocalWilliamsModel(args.local_model, n_ctx=8192,
                                    learning_mode=args.learning_mode,
                                    reasoning_budget=args.reasoning_budget),
                                    args.book_index, seed=args.seed)
    result["verified_gguf_weights"] = weights
    result["requested_learning_mode"] = args.learning_mode
    result["reasoning_token_budget"] = args.reasoning_budget
    print("NTD_SOURCE_APPLICATION_SUMMARY", json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
