"""Williams AI CLI: local GGUF inference, no remote AI API required.

Examples:
  python -m williams_ai.cli examples/ai_market_snapshot.json
  python -m williams_ai.cli examples/ai_market_snapshot.json \
     --local-model models/my-model.gguf --question "Что видит бот?"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analyst import build_report, render_russian_report
from .knowledge import retrieve_notes


def main() -> None:
    parser = argparse.ArgumentParser(description="Williams AI: offline read-only analysis")
    parser.add_argument("snapshot", type=Path, help="Read-only local JSON snapshot")
    parser.add_argument("--local-model", type=Path, help="Existing GGUF model path; never downloaded automatically")
    parser.add_argument("--question", help="Question to ask the local language model")
    parser.add_argument("--knowledge-dir", type=Path, help="Local Markdown/text research notes, no cloud")
    parser.add_argument("--gpu-layers", type=int, default=0, help="llama.cpp GPU offload (default CPU)")
    args = parser.parse_args()
    if args.question and not args.local_model:
        parser.error("--question requires --local-model; no cloud AI is used")
    if args.knowledge_dir and not args.question:
        parser.error("--knowledge-dir requires --question")
    if args.local_model and not args.question:
        parser.error("--local-model requires --question")

    data = json.loads(args.snapshot.read_text(encoding="utf-8"))
    report = build_report(data)
    print(render_russian_report(report))
    if args.question:
        from .local_model import LocalWilliamsModel
        notes = retrieve_notes(args.question, args.knowledge_dir) if args.knowledge_dir else []
        model = LocalWilliamsModel(args.local_model, gpu_layers=args.gpu_layers)
        print("\n--- Ответ локальной модели (не торговая команда) ---")
        print(model.answer(args.question, report, notes=notes))


if __name__ == "__main__":
    main()
