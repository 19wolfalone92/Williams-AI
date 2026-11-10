"""50 sequential 20-step training mini-cycles with a shared dual-T4 LoRA model.

Cohorts are authored / machine-checked research, NOT expert-labeled markets.
Continue from verified Nov-10 V2 research adapter; never promote without blind audit.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import traceback

from .serial_50_engine import JointRun, train as train_joint
from .market_decision_curriculum import COHORTS

MODE = "SERIAL_50_FOCUSED_RISK_WAVES_CAMPAIGN_BOOKS_20261110_V1"
SLUG = "williams-ai-fifty-serial-micro-20261110-r1"
INITIAL_SHA = "2a18d0b3788c10bc851a155ee89c2997e08bf63b00fa6e6d345e676e5960ec9a"
CONTROL_URL = (
    "https://raw.githubusercontent.com/19wolfalone92/Williams-AI/"
    "research/fifty-serial-mini-cycles-20261110/experiments/serial-50-control.json"
)
LIMITS = {
    "optimizer_attempts": 1000,
    "mini_rounds": 50,
    "updates_per_mini_round": 20,
    "evaluation_steps": 250,
    "learning_rate": 2e-6,
    "training_seconds": 10800,
    "maximum_kernel_seconds": 18000,
    "minimum_free_gpu_seconds": 22000,
}
PROTOCOL = {"mode": "direct", "max_new_tokens": 192, "seconds": 25, "do_sample": False}
RUN = JointRun(
    MODE, SLUG, CONTROL_URL, INITIAL_SHA, COHORTS,
    ("base", "parent", "candidate"), LIMITS, PROTOCOL, 20261112,
    source_repetitions=3,
    rank_cohorts=(("context", "precision"), ("waves", "campaign")),
)


def train(source, curriculum, questions, parent, reference_copy, reference, output):
    try:
        return train_joint(source, curriculum, questions, parent, reference_copy, reference, output, run=RUN)
    except Exception as exc:
        detail = str(exc)
        code = detail if re.fullmatch(r"[A-Z][A-Z0-9_]{5,99}", detail) else None
        if isinstance(exc, KeyError) and exc.args and exc.args[0] in ("required_bars", "required_green"):
            code = "MIXED_AC_ANSWER_FIELDS"
        rank = int(os.environ["LOCAL_RANK"])
        trace = [{"file": Path(frame.filename).name, "line": frame.lineno}
                 for frame in traceback.extract_tb(exc.__traceback__)[-10:]]
        result = {"error_type": type(exc).__name__, "safe_code": code,
                  "rank": rank, "frames": trace, "execution_allowed": False}
        (output.parent / f"serial_50_rank{rank}_failure.json").write_text(json.dumps(result))
        raise
