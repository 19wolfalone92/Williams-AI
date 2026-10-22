"""Run an optional, honest four-edition GGUF comprehension smoke-exam.

python -m williams_ai.exam_cli --local-model /private/williams.gguf
A score measures only ten fixed cases; it is NOT a book mastery percentage.
"""
from __future__ import annotations

import argparse
import json

from .competency_exam import grade_exam
from .local_model import LocalWilliamsModel


def main() -> None:
    p = argparse.ArgumentParser(description="Williams AI offline scenario exam")
    p.add_argument("--local-model", required=True, help="Existing user-provided GGUF")
    p.add_argument("--gpu-layers", type=int, default=0)
    a = p.parse_args()
    model = LocalWilliamsModel(a.local_model, gpu_layers=a.gpu_layers)
    result = grade_exam(model.answer_exam_case)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
