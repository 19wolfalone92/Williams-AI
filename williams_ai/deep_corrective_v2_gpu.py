"""Bounded dual-T4 Qwen3-4B corrective QLoRA, NEVER trader activation.

Cohorts are authored / machine-checked research, NOT expert-labeled markets.
Continue from the byte-verified 2026-11-10 v1 research candidate. No promotion.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import traceback

from .joint_rule_gpu import JointRun, train as train_joint
from .market_decision_curriculum import COHORTS

MODE = "DEEP_CORRECTIVE_BOOK_RISK_STOP_AC_20261110_V2"
SLUG = "williams-ai-deep-corrective-20261110-r2"
INITIAL_SHA = "976349c696879efb180c41e9d9090cf5ff928f93a77bf23a5e17dbedd6c8bd41"
CONTROL_URL = (
    "https://raw.githubusercontent.com/19wolfalone92/Williams-AI/"
    "research/deep-corrective-v2-20261110/experiments/deep-corrective-v2-control.json"
)
LIMITS = {
    "optimizer_attempts": 1024,
    "evaluation_steps": 256,
    "learning_rate": 3e-6,
    "training_seconds": 7200,
    "maximum_kernel_seconds": 14400,
    "minimum_free_gpu_seconds": 18000,
}
PROTOCOL = {"mode": "direct", "max_new_tokens": 192, "seconds": 25, "do_sample": False}
RUN = JointRun(
    MODE, SLUG, CONTROL_URL, INITIAL_SHA, COHORTS,
    ("base", "parent", "candidate"), LIMITS, PROTOCOL, 20261111,
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
        (output.parent / f"deep_corrective_v2_rank{rank}_failure.json").write_text(json.dumps(result))
        raise
