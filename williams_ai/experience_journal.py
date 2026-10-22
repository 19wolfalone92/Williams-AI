"""Safe offline experiential learning across fifteen frozen market histories.

The small logistic model's coefficients are fitted ONLY on the chronological
training set; validation/test labels cannot update them. This does NOT
fine-tune the GGUF language model or authorize futures trading.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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


def main()->None:
    r=study_15()
    print(json.dumps(r,ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()
