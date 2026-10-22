"""Safe offline experiential learning across fifteen frozen market histories.

The small logistic model's coefficients are fitted ONLY on the chronological
training set; validation/test labels cannot update them. This does NOT
fine-tune the GGUF language model or authorize futures trading.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import argparse
import os
import tempfile
from datetime import datetime, timezone

from .causal_paper_lab import FROZEN_DATA,frozen_markets
from .paper_learning import learn_from_history


def study_15(path:Path=FROZEN_DATA)->dict[str,Any]:
    markets=frozen_markets(path)
    reports=[]
    for symbol in sorted(markets):
        x=markets[symbol]
        fit=learn_from_history(x["bars"],"H1")
        test=fit["test"];valid=fit["validation"];m=fit["model"]
        delta=test["brier_score"]-test["constant_prior_brier"]
        reports.append({
            "symbol":symbol,"algorithm":m["algorithm"],
            "feature_names":m["features"],
            "weights_trained_on_past_only":[round(z,7) for z in m["weights"]],
            "train_count":m["train_count"],"trained_through":m["trained_through"],
            "validation_brier":round(valid["brier_score"],6),
            "validation_baseline_brier":round(valid["constant_prior_brier"],6),
            "test_brier":round(test["brier_score"],6),
            "test_constant_prior_brier":round(test["constant_prior_brier"],6),
            "test_delta_lower_is_better":round(delta,6),
            "outperformed_baseline_test":delta<0,
            "test_samples":test["samples"],
            "test_selected_at_0_6":test["selected_at_fixed_0_6"],
            "learning_status":"RESEARCH_ONLY_NOT_CANONICAL_WILLIAMS",
            "execution_allowed":False,
        })
    return {
        "version":"WILLIAMS_15PAIR_PAST_ONLY_LEARNING_V1",
        "trained_models":len(reports),
        "outperformed_constant_baseline_test_count":sum(x["outperformed_baseline_test"] for x in reports),
        "reports":reports,
        "trade_action":None,"execution_allowed":False,
        "language_model_weights_modified":False,
        "limitations":[
            "Each tiny logistic scorer learned from chronological historical price/indicator features, not copyrighted book pages.",
            "Training examples are LONG and SHORT at fixed intervals, NOT book-confirmed WM1/2/3 entries.",
            "The 12-bar paper horizon is an experimental score label, NEVER a campaign exit permission.",
            "No trade recommendation, demo authorization, GGUF weight fine-tuning or market edge is implied.",
            "Backtest set has been inspected; new truly unseen future periods are required for independent proof.",
        ],
    }


def save_private_experience(output:Path, result:dict[str,Any])->Path:
    """Atomic private 0600 checkpoint, NEVER inside public GitHub checkout."""
    root=Path(__file__).resolve().parents[1]
    dest=Path(output).expanduser().resolve()
    if root==dest or root in dest.parents or dest.suffix!=".json":
        raise ValueError("Private research memory must be .json OUTSIDE the repository")
    if dest.exists() and dest.is_symlink():
        raise ValueError("Refuse symlink")
    dest.parent.mkdir(parents=True,exist_ok=True)
    payload=json.dumps(result,ensure_ascii=False,indent=2)+"\n"
    fd,tmp=tempfile.mkstemp(prefix=".williams-experience-",suffix=".tmp",dir=str(dest.parent))
    try:
        os.fchmod(fd,0o600)
        with os.fdopen(fd,"w",encoding="utf-8") as out:
            out.write(payload)
            out.flush()
            os.fsync(out.fileno())
        os.replace(tmp,dest)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return dest


def recall_past_experience(
    journal_path:Path, symbol:str, *, as_of:str
)->list[dict[str,Any]]:
    """Do not leak a model fitted AFTER the replay date into that date's analysis."""
    if not symbol.endswith("USDT"):
        raise ValueError("Invalid symbol")
    when=datetime.fromisoformat(as_of.replace("Z","+00:00"))
    if when.tzinfo is None:raise ValueError("Timezone required")
    content=json.loads(Path(journal_path).read_text(encoding="utf-8"))
    if content.get("version")!="WILLIAMS_15PAIR_PAST_ONLY_LEARNING_V1":
        raise ValueError("Unsupported experience journal")
    result=[]
    for row in content["reports"]:
        if row.get("symbol")!=symbol:
            continue
        trained=datetime.fromisoformat(row["trained_through"].replace("Z","+00:00"))
        if trained>when:
            continue
        # Deliberately omit future test/validation labels and returns:
        # a scoring report is not authorized trade evidence.
        result.append({
            "source":f'LOCAL_PAPER_EXPERIENCE:{symbol}:{row["trained_through"]}',
            "excerpt":(
                "Имеется локальная прошлая исследовательская модель на 6 признаках "
                f'(train={row["train_count"]}, cutoff={row["trained_through"]}). '
                "Это НЕ проверенная стратегия Вильямса, не сигнал и не основание для сделки."
            ),
            "trade_action":None,
        })
    return result


def main()->None:
    p=argparse.ArgumentParser(description="15-market private chronological learning")
    p.add_argument("--save-private",type=Path,help="Private absolute path OUTSIDE public git checkout")
    a=p.parse_args()
    r=study_15()
    if a.save_private:
        target=save_private_experience(a.save_private,r)
        print("Saved private non-executable experience:",str(target))
        print("Trained:",r["trained_models"],
              "test better than constant:",r["outperformed_constant_baseline_test_count"])
    else:
        print(json.dumps(r,ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()
