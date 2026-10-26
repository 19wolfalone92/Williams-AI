"""Causal empirical scenario frequencies for nonlinear signal families.

Works on coarse causal FEATURE REGIMES, not duplicate shapes or screenshots.
No forecast exists until sufficient independent, already-resolved samples.
"""
from __future__ import annotations
from collections import Counter
from datetime import datetime,timezone
import math
from typing import Any

_LABELS=("UP","DOWN","RANGE")


def _instant(value:str):
    if not isinstance(value,str):raise ValueError("AWARE_TIMESTAMP_REQUIRED")
    try:t=datetime.fromisoformat(value.replace("Z","+00:00"))
    except ValueError as exc:raise ValueError("AWARE_TIMESTAMP_REQUIRED") from exc
    if t.tzinfo is None or t.utcoffset() is None:
        raise ValueError("AWARE_TIMESTAMP_REQUIRED")
    return t.astimezone(timezone.utc)


def _wilson(success:int,n:int,z:float=1.96)->list[float]:
    if n==0:return [0.0,1.0]
    p=success/n
    d=1+z*z/n
    middle=(p+z*z/(2*n))/d
    width=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [round(max(0,middle-width),5),round(min(1,middle+width),5)]


def empirical_regime_scenarios(
    resolved:list[dict[str,Any]],*,as_of:str,
    signal_family:str,market_regime:str,horizon_bars:int,
    min_cases:int=100
)->dict[str,Any]:
    """Use only outcomes that were verifiably KNOWN before the target as-of.

    Each historical record must itself contain source observation, future
    resolution and time of availability. Independent external provenance is
    still required; locally supplied timestamps alone are not cryptographic proof.
    """
    if (not isinstance(resolved,list) or len(resolved)>100000
        or type(min_cases) is not int or not 30<=min_cases<=10000
        or type(horizon_bars) is not int or not 1<=horizon_bars<=2000
        or not all(isinstance(x,str) and 1<=len(x)<=100 and x.strip()
                   for x in (signal_family,market_regime))):
        raise ValueError("BOUNDED_EMPIRICAL_REGIME_REQUEST_REQUIRED")
    cutoff=_instant(as_of)
    sample=Counter()
    seen=set()
    ignored_future=0
    for row in resolved:
        if not isinstance(row,dict) or set(row)!={
            "case_id","signal_family","market_regime","horizon_bars",
            "observed_at","resolved_at","available_at","outcome"}:
            raise ValueError("STRICT_RESOLVED_SIGNAL_FIELDS_REQUIRED")
        cid=row["case_id"]
        if not isinstance(cid,str) or not cid or cid in seen:
            raise ValueError("UNIQUE_SIGNAL_CASE_ID_REQUIRED")
        seen.add(cid)
        if (not isinstance(row["signal_family"],str)
            or not isinstance(row["market_regime"],str)
            or type(row["horizon_bars"]) is not int
            or not 1<=row["horizon_bars"]<=2000
            or row["outcome"] not in _LABELS):
            raise ValueError("INVALID_SIGNAL_OR_RESOLVED_OUTCOME")
        observed=_instant(row["observed_at"])
        resolved_at=_instant(row["resolved_at"])
        available=_instant(row["available_at"])
        if not observed<resolved_at<=available:
            raise ValueError("INVALID_CAUSAL_OUTCOME_CHRONOLOGY")
        # Data appearing at or after decision time cannot inform the prediction.
        if available>=cutoff:
            ignored_future+=1
            continue
        if (row["signal_family"]!=signal_family or
            row["market_regime"]!=market_regime or
            row["horizon_bars"]!=horizon_bars):
            continue
        sample[row["outcome"]]+=1
    n=sum(sample.values())
    # Data volume alone does not guarantee stability or independent provenance.
    probs=({lab:round((sample[lab]+1)/(n+3),6) for lab in _LABELS}
           if n>=min_cases else None)
    intervals=({lab:_wilson(sample[lab],n) for lab in _LABELS}
               if n>=min_cases else None)
    return {"mode":"CAUSAL_EMPIRICAL_REGIME_FREQUENCIES_NOT_TRADING_SIGNAL",
        "as_of":as_of,"signal_family":signal_family,"market_regime":market_regime,
        "horizon_bars":horizon_bars,"resolved_past_cases":n,
        "excluded_future_unavailable_cases":ignored_future,
        "minimum_cases_required":min_cases,"probabilities":probs,
        "approximate_95pct_outcome_intervals":intervals,
        "sample_sufficient_for_experimental_estimate":probs is not None,
        "independent_external_outcome_seal_checked":False,
        "out_of_sample_calibration_proven":False,
        "nonstationary_regimes_may_invalidate_rates":True,
        "execution_allowed":False}
