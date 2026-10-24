"""Temporal replay of a genuinely UPDATING trading-research skill.

At historical decision time t this learner has ONLY training signals with
resolution time strictly < t. A global chronological market-wide queue prevents
ETH/BTC correlated future labels and stopped trades from entering early.
Evaluation/selection periods freeze the trained weights; their outcomes never
update the model (no backdoor test-set tuning).

This is an SGD learned probability of a PAPER WM candidate surviving fees,
not Qwen GGUF weight training, not full Williams campaign logic, not a
real/demo exchange trade. Candidate execution always remains disabled.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import heapq
import json
import math
from typing import Any

from .continual_trader import Outcome, PrivateExperience, _finite_vector, _time
from .paper_learning import FEATURES

VERSION = "WILLIAMS_CAUSAL_ONLINE_SKILL_FROZEN_HOLDOUT_V1"


def _predict(weights: list[float], vector: tuple[float, ...]) -> float:
    score = max(-25.0, min(25.0, sum(w * x for w, x in zip(weights, vector))))
    return 1.0 / (1.0 + math.exp(-score))


def _one_pass_update(
    weights: list[float], ex: Outcome, *, learning_rate: float, l2: float
) -> None:
    y = float(ex.net_return > 0)
    features = _finite_vector(ex.features)
    error = _predict(weights, features) - y
    for k, value in enumerate(features):
        penalty = l2 * weights[k] if k else 0.0
        weights[k] -= learning_rate * (error * value + penalty)


def _metrics(rows: list[tuple[Outcome, float]], baseline_prior: float) -> dict[str, Any]:
    if not rows:
        return {"examples": 0, "status": "INSUFFICIENT_INDEPENDENT_HOLDOUT"}
    n = len(rows)
    truth = [float(x.net_return > 0) for x, _ in rows]
    predictions = [p for _, p in rows]
    threshold = [x for x, p in rows if p >= 0.60]
    gains = sum(x.net_return for x in threshold)
    # Even if favorable, these are overlapping hypothetical trades with no
    # capital allocation, funding fees, partial fill or SL order confirmation.
    raw_paper_mtm = sum(x.net_return for x, _ in rows)
    brier = sum((p - y)**2 for p, y in zip(predictions, truth)) / n
    constant = sum((baseline_prior - y)**2 for y in truth) / n
    timeline = {(x.symbol,x.interval,x.decision_at) for x, _ in rows}
    return {
        "examples": n,
        "unique_symbol_timeframe_decisions": len(timeline),
        "distinct_assets": len({x.symbol for x,_ in rows}),
        "brier": round(brier, 9),
        "constant_past_train_prior_brier": round(constant, 9),
        "positive_outcome_frequency": round(sum(truth)/n, 6),
        "fixed_p_threshold": 0.60,
        "paper_selected_examples": len(threshold),
        "paper_selected_mean_net": (
            round(gains/len(threshold), 9) if threshold else None),
        "unselected_paper_mean_net": round(raw_paper_mtm/n, 9),
        "paper_selection_is_not_a_portfolio_strategy": True,
        "future_labels_never_tune_this_frozen_evaluation": True,
    }


def causal_incremental_study(
    ledger: PrivateExperience, *,train_until: str,
    validation_until: str, observe_until: str,
    learning_rate: float = 0.02, l2: float = 0.002,
    min_training_decisions: int = 40,
    min_validation_decisions: int = 12,
) -> dict[str, Any]:
    """Updates weights only as prior paper outcomes resolve.

    The 'train_until' and subsequent evaluation ranges are selected in
    advance. The current function never chooses/revises those timestamps to
    maximize the frozen validation or holdout scores.
    """
    a,b,c = (_time(x) for x in (train_until,validation_until,observe_until))
    if not a<b<c:
        raise ValueError("CHRONOLOGICAL_GLOBAL_EMBARGO_ORDER")
    if (not isinstance(learning_rate, (int,float)) or isinstance(learning_rate,bool)
            or not math.isfinite(learning_rate) or not 0<learning_rate<=0.1
            or not isinstance(l2, (int,float)) or isinstance(l2,bool)
            or not math.isfinite(l2) or not 0<=l2<=0.1
            or not isinstance(min_training_decisions,int)
            or not isinstance(min_validation_decisions,int)
            or min_training_decisions<1 or min_validation_decisions<1):
        raise ValueError("INVALID_PREGREGISTERED_LEARNING_HYPERPARAMETERS")
    all_cases=[
        ex for ex in ledger.samples(resolved_before=observe_until)
        if ex.setup in ("WM1","WM2","WM3")
        and _time(ex.decision_at)<c and _time(ex.resolved_at)<=c
    ]
    grouped: dict[datetime,list[Outcome]] = defaultdict(list)
    for ex in all_cases:
        grouped[_time(ex.decision_at)].append(ex)
    if not grouped:
        raise ValueError("NO_CAUSALLY_RESOLVED_WILLIAMS_EPISODES")
    weights = [0.0] * len(FEATURES)
    pending: list[tuple[datetime,int,Outcome]] = []
    seq = 0
    observed_training = 0
    trained_times:set[datetime] = set()
    last_train_resolved:datetime|None = None
    predictions:dict[str,list[tuple[Outcome,float]]] = {
        "train_prequential":[], "validation_frozen":[], "test_frozen":[]}
    training_examples = 0
    prior_positive = 0
    for decision in sorted(grouped):
        # Only OBSERVED, settled, strictly earlier training results may
        # update the weights prior to the current decision group. Every
        # market at the same instant gets the SAME history-informed model.
        while (pending and pending[0][0] < decision
               and pending[0][0] < a):
            resolved, _, completed = heapq.heappop(pending)
            _one_pass_update(weights, completed,
                             learning_rate=learning_rate,l2=l2)
            last_train_resolved=resolved
            observed_training += 1
            prior_positive += int(completed.net_return > 0)
        group = grouped[decision]
        if decision < a:
            partition = "train_prequential"
            trained_times.add(decision)
        elif decision < b:
            partition = "validation_frozen"
        else:
            partition = "test_frozen"
        for ex in group:
            p = _predict(weights, _finite_vector(ex.features))
            predictions[partition].append((ex,p))
            if partition == "train_prequential" and _time(ex.resolved_at)<a:
                heapq.heappush(pending,(_time(ex.resolved_at),seq,ex))
                seq += 1
            if partition == "train_prequential":
                training_examples += 1
    # No use of unresolved training labels at the freeze cutoff.
    # Those labels must not leak into validation or final holdout.
    while pending and pending[0][0]<a:
        t,_,ex=heapq.heappop(pending)
        _one_pass_update(weights,ex,learning_rate=learning_rate,l2=l2)
        last_train_resolved=t
        observed_training+=1
        prior_positive+=int(ex.net_return>0)
    if len(trained_times)<min_training_decisions or observed_training<min_training_decisions:
        raise ValueError("INSUFFICIENT_INDEPENDENT_SETTLED_TRAINING_DECISIONS")
    if len({_time(x.decision_at) for x,_ in predictions["validation_frozen"]})<min_validation_decisions:
        raise ValueError("INSUFFICIENT_INDEPENDENT_FROZEN_VALIDATION_TIMES")
    if not predictions["test_frozen"]:
        raise ValueError("EMPTY_FROZEN_TEST")
    baseline = prior_positive/observed_training
    mval = _metrics(predictions["validation_frozen"],baseline)
    mtest = _metrics(predictions["test_frozen"],baseline)
    improvement = mval["constant_past_train_prior_brier"] - mval["brier"]
    # Predeclared eligibility on validation; NEVER select a model by the test.
    meets_validation=(
        improvement>=0.005
        and mval["paper_selected_examples"]>=10
        and mval["paper_selected_mean_net"] is not None
        and mval["paper_selected_mean_net"]>0
        and mval["distinct_assets"]>=3
    )
    # Even a qualifying paper model is still NOT an exchange strategy.
    status=("PROMISING_PAPER_ONLY_REQUIRES_PROSPECTIVE_FORWARD"
            if meets_validation else "ABSTAIN_NO_VALIDATED_PAPER_ADVANTAGE")
    state={
        "algorithm": "CAUSAL_ONLINE_LOGISTIC_SGD_AFTER_OUTCOME_RESOLUTION",
        "trained_until_cutoff":a.isoformat(),
        "last_training_outcome_resolved_at":last_train_resolved.isoformat()
             if last_train_resolved else None,
        "training_updates_after_settlement": observed_training,
        "training_decision_timestamps":len(trained_times),
        "learned_weights":[round(x,12) for x in weights],
        "training_prior_positive_rate":round(baseline,9),
        "learning_rate":learning_rate,"l2":l2,
        "validation_until":b.isoformat(),
        "research_only":True,
    }
    digest=hashlib.sha256(json.dumps(
        state,sort_keys=True,separators=(",", ":")).encode()).hexdigest()
    return {
        "mode": VERSION,
        "fingerprint":digest,
        **state,
        "training_prequential":_metrics(predictions["train_prequential"],baseline),
        "validation_frozen":mval,
        "test_frozen":mtest,
        "validation_improvement_vs_train_prior_brier":round(improvement,9),
        "qualification":status,
        "frozen_test_never_selected_or_trained":True,
        "simultaneous_crypto_decisions_share_previous_past_only_weights":True,
        "same_symbol_trade_fill_or_portfolio_risk_not_simulated":True,
        "source_edition":"TC2_PROVISIONAL_WM1_WM2_WM3",
        "actual_gguf_weights_changed":False,
        "actual_binance_orders":0,"actual_exchange_stop_ack":0,
        "trade_action":None,"execution_allowed":False,
    }
