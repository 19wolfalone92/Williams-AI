"""Local-only Williams AI 2.0 research pipeline.

Prepare requires two approval records for every market label. Outputs training
rows only from train, and prompts ONLY for validation/test. Private files
must remain outside the public repository. Does not call Kaggle or Binance.

Examples:
 python scripts/ai_reasoning_v2_lab.py prepare --cases /private/reviewed.jsonl --output /private/reasoning-v2
 python scripts/ai_reasoning_v2_lab.py schedule --cases /private/reviewed.jsonl --failures /private/validation-errors.jsonl --output /private/reasoning-v2/plan.json
 python scripts/ai_reasoning_v2_lab.py compare --cases /private/reviewed.jsonl --parent /private/parent-pred.jsonl --candidate /private/candidate-pred.jsonl --output /private/reasoning-v2/compare.json
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from williams_ai.reasoning_research_v2 import adaptive_schedule, check_corpus, export_research
from williams_ai.reasoning_walkforward_v2 import compare_frozen_predictions


def private_path(value):
    path = Path(value).expanduser()
    resolved = path.resolve()
    if resolved == ROOT or ROOT in resolved.parents:
        raise ValueError("PRIVATE_DATA_MUST_NOT_LIVE_IN_GIT_CHECKOUT")
    if path.is_symlink():
        raise ValueError("NO_SYMLINKS_IN_PRIVATE_DATA")
    return path


def load_rows(path):
    src = private_path(path)
    if not src.is_file() or src.stat().st_size > 100_000_000:
        raise ValueError("BOUNDED_PRIVATE_JSONL_REQUIRED")
    rows = []
    with src.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
            if len(rows) > 20000:
                raise ValueError("TOO_MANY_PRIVATE_ROWS")
    return rows


def write_new(path, value, *, jsonl=False):
    path = private_path(path)
    if not path.parent.is_dir() or path.exists():
        raise ValueError("NEW_OUTPUT_FILE_AND_EXISTING_PRIVATE_DIR_REQUIRED")
    payload = ("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
                       for row in value) if jsonl else
               json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(payload)


def main():
    p = argparse.ArgumentParser()
    commands = p.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--cases", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    schedule = commands.add_parser("schedule")
    schedule.add_argument("--cases", type=Path, required=True)
    schedule.add_argument("--failures", type=Path, required=True)
    schedule.add_argument("--output", type=Path, required=True)
    compare = commands.add_parser("compare")
    compare.add_argument("--cases", type=Path, required=True)
    compare.add_argument("--parent", type=Path, required=True)
    compare.add_argument("--candidate", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    cases = load_rows(args.cases)
    manifest = check_corpus(cases)
    if args.command == "prepare":
        output = private_path(args.output)
        if not output.is_dir() or output.stat().st_mode & 0o077:
            raise ValueError("EXISTING_PRIVATE_0700_OUTPUT_DIRECTORY_REQUIRED")
        paths = [output / (name + ".jsonl") for name in
                 ("train_sft", "validation_prompts", "test_prompts")]
        paths.append(output / "manifest.json")
        if any(path.exists() for path in paths):
            raise ValueError("OUTPUT_ALREADY_EXISTS_NO_OVERWRITE")
        payload, manifest = export_research(cases)
        for name, path in zip(("train_sft", "validation_prompts", "test_prompts"), paths[:3]):
            write_new(path, payload[name], jsonl=True)
        write_new(paths[-1], manifest)
        print(json.dumps({"prepared": manifest["counts"],
                          "test_gold_exported": False}, sort_keys=True))
    elif args.command == "schedule":
        rows = [c for c in cases if c["split"] == "train"]
        failures = load_rows(args.failures)
        result = adaptive_schedule(rows, failures)
        write_new(args.output, result)
        print(json.dumps({"rounds": len(result["schedule"]), "validation_answers_used": False}))
    elif args.command == "compare":
        parents = load_rows(args.parent)
        candidates = load_rows(args.candidate)
        heldout = [c for c in cases if c["split"] == "test"]
        result = compare_frozen_predictions(heldout, parents, candidates)
        write_new(args.output, result)
        print(json.dumps({"parent": result["parent"]["decision_accuracy"],
                          "candidate": result["candidate"]["decision_accuracy"],
                          "risk_regression": result["risk_regression"]}))


if __name__ == "__main__":
    main()
