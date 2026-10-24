"""Blinded expert wave benchmark. Nothing is called gold without two reviewers.

The evaluator sees ONLY annotated CLOSED market prefixes; every annotation
has an explicit decision timestamp and source checksum. Multiple experts
can disagree legitimately on Elliott degrees. Disputed/unknown examples are
not fabricated into right/wrong training labels. Archived labels alone do
not prove that reviewers are independent professional traders.

Use only private datasets; original charts/books are not committed. This
module neither trains model weights nor calls Binance, nor routes orders.
"""
from __future__ import annotations

from collections import Counter,defaultdict
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any

LABELS=frozenset({
    "IMPULSE_UP","IMPULSE_DOWN","CORRECTION",
    "TRANSITION","UNDETERMINED"
})
_TFS=frozenset({"M5","M15","H1","H4","H6","D1"})
_EDS=frozenset({"TC1","NTD","TC2","COURSE"})
_HEX=re.compile(r"^[0-9a-f]{64}$")
_SYMBOL=re.compile(r"^[A-Z0-9]{3,22}$")
_RATER=re.compile(r"^[A-Za-z0-9_.:-]{5,100}$")
_ALLOWED=frozenset({
    "case_id","symbol","timeframe","venue","cutoff",
    "visible_prefix_sha256","edition","reviewer_id","reviewed_at",
    "market_structure","wave_degree","source_page","justification",
})
_REPORT_KEYS=frozenset({
    "case_id","symbol","timeframe","cutoff","visible_prefix_sha256",
    "model_id","predicted_structure","prediction_created_at",
    "precommitted_prediction_sha256",
})


def _utc(value:str)->datetime:
    if not isinstance(value,str):raise ValueError("NEED_AWARE_TIME")
    try:d=datetime.fromisoformat(value.replace("Z","+00:00"))
    except ValueError as e:raise ValueError("NEED_AWARE_TIME") from e
    if d.tzinfo is None:raise ValueError("NEED_AWARE_TIME")
    return d.astimezone(timezone.utc)


def prediction_commitment(record:dict[str,Any])->str:
    """Canonical seal excluding ONLY the digest; never accept loose fields."""
    obj={k:v for k,v in record.items() if k!="precommitted_prediction_sha256"}
    if set(obj)!=_REPORT_KEYS-{"precommitted_prediction_sha256"}:
        raise ValueError("INVALID_PREDICTION_SCHEMA")
    return hashlib.sha256(json.dumps(
        obj,sort_keys=True,separators=(",",":"),ensure_ascii=False
    ).encode("utf-8")).hexdigest()


def _read_rows(file:Path,max_rows:int)->list[dict[str,Any]]:
    path=Path(file).expanduser()
    if not path.is_file() or path.is_symlink() or path.suffix.lower()!=".jsonl":
        raise ValueError("PRIVATE_JSONL_REQUIRED")
    if path.stat().st_size>16_000_000:raise ValueError("REVIEW_FILE_TOO_LARGE")
    result=[]
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():continue
        result.append(json.loads(line))
        if len(result)>max_rows:raise ValueError("TOO_MANY_EXPERT_CASES")
    return result


def evaluate_blind_wave_gold(
    independent_reviews:Path,
    model_predictions:Path,
    *,min_agreed_cases:int=30,min_utc_days:int=8,
)->dict[str,Any]:
    """Two independent annotations, strictly before predictions were scored.

    This is an observational understanding exam. After training on any case,
    it can NEVER again serve as an untouched blind holdout.
    """
    if any(isinstance(v,bool) or not isinstance(v,int) or v<1
           for v in (min_agreed_cases,min_utc_days)):
        raise ValueError("INVALID_BENCHMARK_MINIMUM")
    from .book_rules import registry
    book_pages={x["id"]:x["pages"] for x in registry()["books"]}
    reviews=_read_rows(independent_reviews,10000)
    predictions=_read_rows(model_predictions,5000)
    groups=defaultdict(list)
    unique_rater=defaultdict(set)
    snapshot_owners={}
    for row in reviews:
        if not isinstance(row,dict) or set(row)!=_ALLOWED:
            raise ValueError("INVALID_EXPERT_REVIEW_FIELDS")
        case=row["case_id"]
        if (not isinstance(case,str) or not _RATER.fullmatch(case)
                or not _SYMBOL.fullmatch(row["symbol"])
                or row["timeframe"] not in _TFS
                or row["venue"] not in {"SPOT","FUTURES"}
                or row["edition"] not in _EDS
                or row["market_structure"] not in LABELS
                or not isinstance(row["source_page"],int)
                or isinstance(row["source_page"],bool)
                or row["source_page"]<1
                or row["source_page"]>book_pages[row["edition"]]
                or not isinstance(row["wave_degree"],str)
                or not 1<=len(row["wave_degree"])<=80
                or not isinstance(row["justification"],str)
                or not 30<=len(row["justification"])<=1800
                or not _HEX.fullmatch(str(row["visible_prefix_sha256"]))
                or not isinstance(row["reviewer_id"],str)
                or not _RATER.fullmatch(row["reviewer_id"])):
            raise ValueError("MALFORMED_SOURCE_VERIFIED_EXPERT_CASE")
        cutoff=_utc(row["cutoff"])
        reviewed=_utc(row["reviewed_at"])
        if reviewed<cutoff:
            raise ValueError("REVIEW_CANNOT_PREDATE_MARKET_OBSERVATION")
        # Renaming a case (or teaching it under another book edition) must
        # not multiply the independent market sample. The prefix digest
        # already binds the visible candle timestamps, including cutoff.
        snapshot=(row["venue"],row["symbol"],row["timeframe"],
                  row["visible_prefix_sha256"])
        owner=snapshot_owners.setdefault(snapshot,case)
        if owner!=case:
            raise ValueError("DUPLICATE_MARKET_SNAPSHOT_UNDER_DIFFERENT_CASE_ID")
        if row["reviewer_id"] in unique_rater[case]:
            raise ValueError("DUPLICATE_REVIEWER_FOR_SAME_SNAPSHOT")
        unique_rater[case].add(row["reviewer_id"])
        groups[case].append(row)
    if not groups:raise ValueError("EMPTY_EXPERT_GOLD_SET")
    gold={}
    disputed=[]
    underreview=[]
    for case,rs in groups.items():
        if len(rs)!=2:underreview.append(case);continue
        one,two=rs
        immutable=("symbol","timeframe","venue","cutoff",
                   "visible_prefix_sha256","edition","source_page")
        if any(one[k]!=two[k] for k in immutable):
            raise ValueError("MISMATCHED_EXPERT_CASE_PROVENANCE")
        if (one["market_structure"]!=two["market_structure"]
                or one["wave_degree"]!=two["wave_degree"]):
            disputed.append(case)
            continue
        gold[case]=one
    checked=[]
    ids=set()
    model_ids=set()
    for row in predictions:
        if not isinstance(row,dict) or set(row)!=_REPORT_KEYS:
            raise ValueError("INVALID_SEALED_MODEL_REPORT")
        case=row["case_id"]
        if case in ids:raise ValueError("DUPLICATE_MODEL_PREDICTION")
        ids.add(case)
        if row["predicted_structure"] not in LABELS:
            raise ValueError("UNKNOWN_MODEL_WAVE_HYPOTHESIS")
        if row["precommitted_prediction_sha256"]!=prediction_commitment(row):
            raise ValueError("MODEL_REPORT_CHANGED_AFTER_SEAL")
        if case not in groups:
            raise ValueError("PREDICTION_NOT_IN_FROZEN_EXPERT_COHORT")
        reference=groups[case][0]
        for k in ("symbol","timeframe","cutoff","visible_prefix_sha256"):
            if row[k]!=reference[k]:
                raise ValueError("PREDICTION_INPUT_DOES_NOT_MATCH_EXPERT_MARKET")
        pred_at=_utc(row["prediction_created_at"])
        if pred_at<_utc(row["cutoff"]):
            raise ValueError("PREDICTION_CANNOT_PREDATE_VISIBLE_CANDLE")
        # The reviewer may prepare gold after prediction; the model may
        # not inspect the results or future bars by this function contract.
        if not isinstance(row["model_id"],str) or not _RATER.fullmatch(row["model_id"]):
            raise ValueError("MISSING_PINNED_MODEL_IDENTITY")
        model_ids.add(row["model_id"])
        if len(model_ids)>1:
            raise ValueError("MIXED_MODELS_CANNOT_CLAIM_ONE_BLIND_EXAM")
        if case in gold:
            truth=gold[case]["market_structure"]
            correct=(row["predicted_structure"]==truth
                     and truth!="UNDETERMINED")
            checked.append({
                "case_id":case,"symbol":row["symbol"],
                "day":_utc(row["cutoff"]).date().isoformat(),
                "truth":truth,"prediction":row["predicted_structure"],
                "correct_nonabstention":correct,
                "abstained":row["predicted_structure"]=="UNDETERMINED",
            })
    decidable=[x for x in gold.values() if x["market_structure"]!="UNDETERMINED"]
    decidable_days={_utc(x["cutoff"]).date() for x in decidable}
    qualified=(len(decidable)>=min_agreed_cases
               and len(decidable_days)>=min_utc_days
               and len(checked)==len(gold))
    scored=[x for x in checked if x["truth"]!="UNDETERMINED"]
    voiced=[x for x in scored if not x["abstained"]]
    label_counts=Counter(x["truth"] for x in scored)
    # The selective (non-abstained) accuracy can be trivially inflated by
    # answering only one easy case. A full-cohort score treats every
    # abstention / missing answer as NOT correct, and compares against
    # predicting the actual majority class for EVERY case.
    all_gold_classes=Counter(x["market_structure"] for x in decidable)
    correct_all=sum(x["correct_nonabstention"] for x in scored)
    full_score=(correct_all/len(decidable) if decidable else None)
    majority_full=(max(all_gold_classes.values())/len(decidable)
                   if decidable else None)
    class_recall={
        cls:sum(x["correct_nonabstention"] for x in scored
                if x["truth"]==cls)/size
        for cls,size in sorted(all_gold_classes.items())
    }
    # A 30-case, one-class benchmark is NOT proof of discriminating
    # impulses, corrections and transitions, even at apparent 100% accuracy.
    varied=bool(qualified and len(all_gold_classes)>=3
                and min(all_gold_classes.values())>=5)
    baseline_beaten=bool(
        varied and full_score is not None and majority_full is not None
        and full_score>majority_full
        and all(value>=0.5 for value in class_recall.values())
        and len(voiced)/len(decidable)>=0.7
    )
    return {
        "mode":"DOUBLE_EXPERT_REVIEW_CLOSED_MARKET_PREFIX_WAVE_EXAM",
        "submitted_review_rows":len(reviews),
        "reviewed_unique_cases":len(groups),
        "independently_agreed_cases":len(gold),
        "decidable_agreed_cases":len(decidable),
        "distinct_utc_days_of_decidable_cases":len(decidable_days),
        "model_id":next(iter(model_ids)) if model_ids else None,
        "expert_disputed_cases":len(disputed),
        "not_yet_double_reviewed_cases":len(underreview),
        "matched_precommitted_model_predictions":len(checked),
        "distinct_utc_days_of_agreed_cases":len({
            _utc(x["cutoff"]).date() for x in gold.values()}),
        "model_substantive_predictions":len(voiced),
        "model_explicit_abstentions":sum(x["abstained"] for x in checked),
        "agreed_gold_unresolved_wave_degrees":sum(
            x["wave_degree"]=="UNKNOWN" for x in gold.values()),
        "nonabstained_structure_accuracy":(
            round(sum(x["correct_nonabstention"] for x in voiced)/len(voiced),5)
            if voiced else None),
        "coverage_on_decidable_cases":(
            round(len(voiced)/len(scored),5) if scored else None),
        "scored_expert_structure_counts":dict(sorted(label_counts.items())),
        "descriptive_majority_class_accuracy_not_a_trained_baseline":(
            round(max(label_counts.values())/len(scored),5) if scored else None),
        "bench_sample_size_and_multi_day_sufficiency":qualified,
        "full_cohort_gold_class_counts":dict(sorted(all_gold_classes.items())),
        "full_cohort_accuracy_abstentions_and_missing_as_wrong":(
            round(full_score,5) if full_score is not None else None),
        "full_cohort_majority_class_baseline":(
            round(majority_full,5) if majority_full is not None else None),
        "per_gold_class_recall_abstentions_as_wrong":{
            key:round(value,5) for key,value in class_recall.items()},
        "diverse_three_class_min_five_each_benchmark":varied,
        "research_only_discriminative_baseline_beaten":baseline_beaten,
        "discriminative_score_not_elliott_degree_or_profit_proof":True,
        "prediction_seal_is_not_independent_timestamp_or_blindness_proof":True,
        "human_reviewer_independence_technically_unverifiable":True,
        "disputed_cases_do_not_become_false_gold_labels":True,
        "correct_market_structure_not_proof_of_canonical_elliott_degree":True,
        "blindness_against_model_weight_pretraining_unverifiable":True,
        "do_not_claim_profitable_executions":True,
        "expert_truth_incomplete":True,
        "trade_action":None,
        "execution_allowed":False,
    }


def main()->None:
    import argparse
    parser=argparse.ArgumentParser(
        description="Score double-expert reviewed closed-prefix wave classifications; NEVER certify trading")
    parser.add_argument("--independent-reviews",type=Path,required=True)
    parser.add_argument("--sealed-model-predictions",type=Path,required=True)
    parser.add_argument("--min-agreed-cases",type=int,default=30)
    parser.add_argument("--min-utc-days",type=int,default=8)
    args=parser.parse_args()
    result=evaluate_blind_wave_gold(
        args.independent_reviews,args.sealed_model_predictions,
        min_agreed_cases=args.min_agreed_cases,
        min_utc_days=args.min_utc_days)
    print(json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2))


if __name__=="__main__":
    main()
