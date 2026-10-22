"""Offline-by-default CLI: python -m williams_ai.cli examples/ai_market_snapshot.json"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .analyst import build_report, render_russian_report
from .llm import explain_with_openai


def main() -> None:
    parser = argparse.ArgumentParser(description="Williams AI: read-only analysis")
    parser.add_argument("snapshot", type=Path, help="JSON file with a read-only snapshot")
    parser.add_argument("--question", help="Opt-in to the OpenAI API and ask a question")
    args = parser.parse_args()
    snapshot = json.loads(args.snapshot.read_text(encoding="utf-8"))
    report = build_report(snapshot)
    print(render_russian_report(report))
    if args.question is not None:
        print("\n--- Ответ ИИ ---")
        print(explain_with_openai(
            args.question, report,
            api_key=os.environ.get("OPENAI_API_KEY", ""),
            model=os.environ.get("OPENAI_MODEL", ""),
        ))


if __name__ == "__main__":
    main()
