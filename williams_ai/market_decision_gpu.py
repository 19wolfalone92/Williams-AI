"""Pinned dual-T4 continuation for the closed-chart research exam."""
import json
import os
from pathlib import Path
import re
import traceback

from .joint_rule_gpu import JointRun, train as train_joint
from .market_decision_curriculum import COHORTS

MODE = "CLOSED_CHART_DECISION_TRANSFER_20261107_V1"
SLUG = "williams-ai-closed-chart-transfer-20261107-r2"
INITIAL_SHA = "d961d7e7406d255e699da3959a3c67606330fee191807627b15de35ac4508ca7"
CONTROL_URL = (
    "https://raw.githubusercontent.com/19wolfalone92/Williams-AI/"
    "research/private-four-book-deep-training-20261027/experiments/market-decision-control.json"
)
LIMITS = {"optimizer_attempts": 1024, "evaluation_steps": 512, "learning_rate": 7e-6,
          "training_seconds": 5400, "maximum_kernel_seconds": 13200,
          "minimum_free_gpu_seconds": 16200}
PROTOCOL = {"mode": "direct", "max_new_tokens": 192, "seconds": 25, "do_sample": False}
RUN = JointRun(MODE, SLUG, CONTROL_URL, INITIAL_SHA, COHORTS,
               ("base", "parent", "candidate"), LIMITS, PROTOCOL, 20261107,
               source_repetitions=1, rank_cohorts=(("context", "precision"), ("waves", "campaign")))


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
        (output.parent / f"market_decision_rank{rank}_failure.json").write_text(json.dumps(result))
        raise
