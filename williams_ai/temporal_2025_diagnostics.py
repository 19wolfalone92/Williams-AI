"""Independent read-only causal state transitions on all 15 NEW 2025 markets.

No GGUF in this diagnostic. It checks deterministic structural observation:
older hypotheses lose support when later CLOSED price contradicts them, AO
changes, H1 changes within H4/D1, and incomplete five-bullet conclusions
remain UNKNOWN when there is no measured true tick count.
"""
from __future__ import annotations

import argparse,json
from collections import Counter
from pathlib import Path

from .temporal_wave_revision import temporal_revision
from .unseen_2025_wave_exam import _book_observation,load_archives

CUTS=(847,1039)


def diagnose_all_fifteen(manifest:dict)->dict:
    rows=[]
    counts=Counter()
    for symbol in sorted(manifest["markets"]):
        market=manifest["markets"][symbol]
        a=_book_observation(market,CUTS[0],symbol)
        b=_book_observation(market,CUTS[1],symbol)
        delta=temporal_revision(
            a,b,
            earlier_higher_context=a["higher_timeframe_context"],
            later_higher_context=b["higher_timeframe_context"],
        )
        if a["as_of"]>=b["as_of"]:
            raise ValueError("Noncausal chronological observation")
        if delta["trade_action"] is not None or delta["execution_allowed"]:
            raise ValueError("AI attempted to trade in diagnostic")
        for change in delta["changed_confirmed_facts"]:
            counts["changed_"+change]+=1
        counts["pairs_with_structural_price_invalidation"]+=bool(
            delta["was_prior_hypothesis_invalidated_by_price"])
        counts["pairs_with_any_new_confirmed_facts"]+=bool(
            delta["changed_confirmed_facts"])
        counts["pairs_unknown_TC1_tick_count"]+=bool(
            delta["tick_volume_missing_prevents_TC1_squat_confirmation"])
        rows.append({
            "symbol":symbol,
            "first_as_of":delta["earlier_as_of"],
            "later_as_of":delta["later_as_of"],
            "changed_facts":delta["changed_confirmed_facts"],
            "structural_thesis_invalidations":delta["prior_scenarios_lose_support"],
            "h4_d1_transitions":delta["higher_tf_market_context_revision"],
            "wave_degree":"UNKNOWN",
        })
    return {
        "mode":"FIRST_LOOK_2025_REAL_PUBLIC_H1_H4_D1_TEMPORAL_DIAGNOSTIC",
        "manifest_sha256":manifest["source_manifest_sha256"],
        "pairs":len(rows),
        "old_new_H1_snapshots":len(rows)*2,
        "independent_timeframe_passports_included":len(rows)*2*3,
        "counts":dict(counts),
        "observations":rows,
        "GGUF_called":False,
        "human_expert_wave_truth_present":False,
        "good_nonentry_decision_is_not_validated_profitability":True,
        "no_subsequent_test_can_reuse_this_as_pristine_holdout":True,
        "trade_action":None,"execution_allowed":False,
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--manifest",type=Path,required=True)
    ap.add_argument("--summary",action="store_true")
    args=ap.parse_args()
    report=diagnose_all_fifteen(load_archives(args.manifest))
    if args.summary:report.pop("observations")
    print("NEW_2025_MARKET_EVOLUTION",
          json.dumps(report,ensure_ascii=False,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
