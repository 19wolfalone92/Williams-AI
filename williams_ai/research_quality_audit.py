"""Offline research-only audits for repeated answers and resolved forecasts."""
from __future__ import annotations
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import re
from typing import Any

LABELS = ("UP","DOWN","RANGE")


def repeated_answers(rows: list[dict], *, topic_threshold: int = 4) -> dict:
    if type(topic_threshold) is not int or not 2 <= topic_threshold <= 100:
        raise ValueError("INVALID_TOPIC_THRESHOLD")
    seen=set()
    by_hash=defaultdict(lambda: {"topics":set(),"cases":[]})
    if not isinstance(rows,list) or len(rows)>10000:
        raise ValueError("BOUNDED_ANSWER_SET_REQUIRED")
    for row in rows:
        if not isinstance(row,dict) or set(row)!={"case_id","topic","answer"}:
            raise ValueError("EXACT_RESPONSE_FIELDS_REQUIRED")
        cid,topic,answer=(row[k] for k in ("case_id","topic","answer"))
        if (not all(isinstance(v,str) and v.strip() for v in (cid,topic,answer))
            or cid in seen or len(answer)>20000):
            raise ValueError("DUPLICATE_OR_INVALID_RESPONSE")
        seen.add(cid)
        normalized=re.sub(r"\s+"," ",answer).strip().casefold()
        digest=hashlib.sha256(normalized.encode()).hexdigest()
        by_hash[digest]["topics"].add(topic.casefold().strip())
        by_hash[digest]["cases"].append(cid)
    flags=[{"answer_sha256":digest,"topics":sorted(v["topics"]),"cases":sorted(v["cases"])}
           for digest,v in by_hash.items() if len(v["topics"])>=topic_threshold]
    return {"mode":"CROSS_TOPIC_IDENTICAL_ANSWER_HEURISTIC","samples":len(rows),
        "unique_answers":len(by_hash),"suspect_repetition":bool(flags),
        "flags":flags,"model_called":False,"semantic_correctness_proven":False}


def _moment(value: str):
    if not isinstance(value,str): raise ValueError("OFFSET_AWARE_DATETIME_REQUIRED")
    try: out=datetime.fromisoformat(value.replace("Z","+00:00"))
    except ValueError as exc: raise ValueError("OFFSET_AWARE_DATETIME_REQUIRED") from exc
    if out.tzinfo is None or out.utcoffset() is None:
        raise ValueError("OFFSET_AWARE_DATETIME_REQUIRED")
    return out.astimezone(timezone.utc)


def resolved_forecasts(rows: list[dict[str, Any]], *, min_cases: int = 100) -> dict[str,Any]:
    """Evaluate fixed forecasts AFTER outcomes; does not verify external precommit."""
    if type(min_cases) is not int or min_cases<10 or not isinstance(rows,list) or len(rows)>100000:
        raise ValueError("BOUNDED_FORECAST_AUDIT_REQUIRED")
    seen=set()
    brier=loss=0.0
    bins=[{"n":0,"confidence":0.0,"correct":0} for _ in range(10)]
    for row in rows:
        if not isinstance(row,dict) or set(row)!={"case_id","series","as_of","resolved_at","outcome","probabilities"}:
            raise ValueError("STRICT_FORECAST_FIELDS_REQUIRED")
        case_id=row["case_id"]
        if not isinstance(case_id,str) or not case_id or case_id in seen:
            raise ValueError("INVALID_OR_DUPLICATE_CASE_ID")
        seen.add(case_id)
        if not isinstance(row["series"],str) or not row["series"]:
            raise ValueError("SERIES_REQUIRED")
        if _moment(row["resolved_at"])<=_moment(row["as_of"]):
            raise ValueError("OUTCOME_MUST_BE_IN_FUTURE")
        probs=row["probabilities"]
        label=row["outcome"]
        if label not in LABELS or not isinstance(probs,dict) or set(probs)!=set(LABELS):
            raise ValueError("THREE_SCENARIO_PROBABILITIES_REQUIRED")
        if any(type(p) not in (int,float) or not math.isfinite(p) or not 0<=p<=1
               for p in probs.values()):
            raise ValueError("FINITE_PROBABILITIES_REQUIRED")
        if not math.isclose(sum(probs.values()),1,rel_tol=0,abs_tol=1e-6):
            raise ValueError("NORMALIZED_PROBABILITIES_REQUIRED")
        brier+=sum((probs[k]-float(k==label))**2 for k in LABELS)
        loss-=math.log(max(float(probs[label]),1e-15))
        prediction=max(LABELS,key=probs.__getitem__)
        confidence=probs[prediction]
        b=bins[min(9,int(confidence*10))]
        b["n"]+=1;b["confidence"]+=confidence;b["correct"]+=int(prediction==label)
    count=len(rows)
    ece=0.0
    reliability=[]
    for i,bin in enumerate(bins):
        if not bin["n"]: continue
        confidence=bin["confidence"]/bin["n"]
        accuracy=bin["correct"]/bin["n"]
        ece+=abs(confidence-accuracy)*bin["n"]/count
        reliability.append({"bin":i,"cases":bin["n"],
                            "mean_confidence":round(confidence,6),
                            "observed_accuracy":round(accuracy,6)})
    return {"mode":"RESOLVED_PROSPECTIVE_THREE_SCENARIO_AUDIT",
        "cases":count, "multiclass_brier":round(brier/count,8) if count else None,
        "log_loss":round(loss/count,8) if count else None,
        "expected_calibration_error":round(ece,8) if count else None,
        "reliability":reliability,"enough_cases":count>=min_cases,
        "forecast_precommit_externally_verified":False,
        "profitable_live_trading_proven":False,"execution_allowed":False}
