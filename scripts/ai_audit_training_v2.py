"""Read-only audit of the frozen Qwen3-4B serial-50 curriculum, not a model exam.

Rebuilds the actual train/dev/test cases with the project generator.
Does not send test answers to an accelerator or start training.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def fingerprint(value):
    return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                             separators=(",", ":"), default=str).encode()).hexdigest()


def audit(groups):
    splits = ("train", "validation", "test")
    if set(groups) != set(splits):
        raise ValueError("EXACT_THREE_SPLITS_REQUIRED")
    seen_ids, seen_questions, seen_windows = {}, {}, {}
    report = {"schema": "WILLIAMS_AI_REAL_CURRICULUM_AUDIT_V1", "cohorts": {},
              "cross_split_duplicate_questions": [], "cross_split_duplicate_windows": [],
              "cross_split_duplicate_ids": [], "warnings": [], "execution_allowed": False}
    for split in splits:
        report["cohorts"][split] = {}
        for cohort, rows in groups[split].items():
            if not isinstance(rows, list):
                raise ValueError("EXPECTED_LIST_OF_CASES")
            labels, rules, origins, numeric_errors = Counter(), Counter(), Counter(), []
            responses = defaultdict(Counter)
            templates, complete_answers = Counter(), Counter()
            symbols, windows = set(), set()
            for row in rows:
                ident = row["case_id"]
                question = " ".join(row["question"].split())
                qhash = fingerprint(question)
                templates[row["question"].splitlines()[0].strip()] += 1
                complete_answers[row["answer"]] += 1
                if ident in seen_ids and seen_ids[ident] != split:
                    report["cross_split_duplicate_ids"].append(ident)
                seen_ids.setdefault(ident, split)
                if qhash in seen_questions and seen_questions[qhash] != split:
                    report["cross_split_duplicate_questions"].append(ident)
                seen_questions.setdefault(qhash, split)
                window = row.get("window")
                if isinstance(window, dict) and window:
                    symbol = window.get("symbol")
                    if symbol:
                        symbols.add(symbol)
                    whash = fingerprint(window)
                    if whash in seen_windows and seen_windows[whash] != split:
                        report["cross_split_duplicate_windows"].append(ident)
                    seen_windows.setdefault(whash, split)
                    windows.add(whash)
                rules[row.get("rule_id", "UNKNOWN")] += 1
                origins[row.get("label_origin", "UNSPECIFIED")] += 1
                try:
                    answer = json.loads(row["answer"])
                    if not isinstance(answer, dict):
                        raise ValueError("NOT_DICT")
                except (ValueError, TypeError) as error:
                    numeric_errors.append(ident + ":" + str(error))
                    continue
                for name, value in answer.items():
                    responses[name][json.dumps(value, sort_keys=True)] += 1
                labels["positive_wave_confirmed"] += int(answer.get("wave_confirmed") is True)
                labels["positive_squat_verified"] += int(answer.get("squat_verified") is True)
                labels["execution_allowed_true"] += int(answer.get("execution_allowed") is True)
            if numeric_errors:
                raise ValueError("INVALID_REFERENCE_ANSWERS: " + ",".join(numeric_errors[:5]))
            report["cohorts"][split][cohort] = {
                "cases": len(rows), "rules": dict(sorted(rules.items())),
                "label_origins": dict(sorted(origins.items())),
                "symbols": sorted(symbols), "distinct_windows": len(windows),
                "distinct_question_leads": len(templates),
                "largest_shared_question_lead": max(templates.values(), default=0),
                "distinct_complete_answers": len(complete_answers),
                "top_question_leads": templates.most_common(3),
                "positive_wave_confirmed": labels["positive_wave_confirmed"],
                "positive_squat_verified": labels["positive_squat_verified"],
                "execution_allowed_true": labels["execution_allowed_true"],
                "field_values": {k: dict(sorted(v.items())) for k, v in sorted(responses.items())},
                "constant_fields": sorted(k for k, v in responses.items() if len(v) == 1),
            }
    for key in ("cross_split_duplicate_questions", "cross_split_duplicate_windows",
                "cross_split_duplicate_ids"):
        report[key] = sorted(set(report[key]))
    if any(report[k] for k in ("cross_split_duplicate_questions", "cross_split_duplicate_windows",
                              "cross_split_duplicate_ids")):
        report["warnings"].append("CROSS_SPLIT_EXACT_DUPLICATES_REQUIRE_INVESTIGATION")
    waves = {part: report["cohorts"][part].get("waves", {}) for part in splits}
    if all(w.get("positive_wave_confirmed", 0) == 0 for w in waves.values()):
        report["warnings"].append("NO_POSITIVE_WAVE_CONFIRMED_LABELS")
    if all(w.get("positive_squat_verified", 0) == 0 for w in waves.values()):
        report["warnings"].append("NO_POSITIVE_SQUAT_VERIFIED_LABELS")
    if any("CAUSAL_OBSERVATIONS_AND_RESEARCH_POLICY_NOT_HUMAN_GOLD" in
           x["label_origins"] for part in report["cohorts"].values() for x in part.values()):
        report["warnings"].append("SYNTHETIC_OR_PROGRAMMATIC_MARKET_LABELS_NOT_EXPERT_GOLD")
    report["audit_passed_no_exact_cross_split_duplicates"] = not any(
        report[k] for k in ("cross_split_duplicate_questions", "cross_split_duplicate_windows",
                          "cross_split_duplicate_ids"))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--strict-duplicates", action="store_true")
    args = parser.parse_args()
    from scripts.ai_run_serial_50_kaggle import frozen
    groups, _ = frozen()
    report = audit(groups)
    data = json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(data + "\n", encoding="utf-8")
    print(json.dumps({"schema": report["schema"],
                      "counts": {s: {g: x["cases"] for g, x in rows.items()}
                                 for s, rows in report["cohorts"].items()},
                      "warnings": report["warnings"],
                      "duplicate_check": report["audit_passed_no_exact_cross_split_duplicates"]},
                     ensure_ascii=False, sort_keys=True))
    if args.strict_duplicates and not report["audit_passed_no_exact_cross_split_duplicates"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
