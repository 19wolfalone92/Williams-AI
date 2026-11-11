"""Read-only diagnostic of Williams AI curriculum. No GPU or trading access.

Usage: python scripts/audit_learning_labels.py
"""
from __future__ import annotations
import ast
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = {
    "market": ROOT / "williams_ai/market_decision_curriculum.py",
    "training": ROOT / "williams_ai/serial_50_engine.py",
    "tokenization": ROOT / "williams_ai/contrastive_book_practice.py",
    "selection": ROOT / "williams_ai/joint_rule_curriculum.py",
}

def inspect():
    trees = {key: ast.parse(path.read_text(encoding="utf-8")) for key, path in FILES.items()}
    market = trees["market"]
    funcs = {n.name: n for n in ast.walk(market) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    waves = funcs["_waves"]
    constant_fields = {}
    for node in ast.walk(waves):
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if isinstance(key, ast.Constant) and key.value in {"wave_confirmed", "squat_verified"} and isinstance(value, ast.Constant):
                    constant_fields[key.value] = value.value
    campaign = funcs["_campaign_at"]
    synthetic_journal = any(isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "fills" for t in n.targets) for n in ast.walk(campaign))
    training = trees["training"]
    focus = None
    for n in ast.walk(training):
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "FOCUS" for t in n.targets):
            try: focus = ast.literal_eval(n.value)
            except (ValueError, TypeError): pass
    return {
        "source_files": {k: str(p.relative_to(ROOT)) for k,p in FILES.items()},
        "wave_label_constants": constant_fields,
        "synthetic_campaign_fills": synthetic_journal,
        "phase_focus": focus,
        "warnings": [
            "A constant wave label cannot teach positive recognition in this generator.",
            "Synthetic campaign journals are not independently observed executions.",
            "AST checks are structural only; they do not establish model behavior or generalization.",
            "Do not train on final test answers or use this report to authorize trading."
        ],
        "execution_allowed": False,
    }

if __name__ == "__main__":
    print(json.dumps(inspect(), ensure_ascii=False, indent=2))
