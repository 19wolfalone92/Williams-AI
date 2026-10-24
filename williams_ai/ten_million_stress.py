"""10 million DETERMINISTIC PAPER cost-sensitivity trials on REAL WM histories.

A stress trial is one (previously recorded Williams-candidate trade, fee,
slippage, extra adverse gap) combination. These are NOT 10 million distinct
charts, 10 million GGUF thoughts or 10 million independent human trades.

A fixed, preregistered grid prevents selecting optimistic friction assumptions
from the holdout. The only model learning occurs in continual_trader.py with
absolute chronological train/validation/test splits; never from this grid.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .continual_trader import PrivateExperience, Outcome, _time

EXTRA_FEE_BPS = tuple(range(0, 22))       # 22 values, cost each side
EXTRA_SLIPPAGE_BPS = tuple(range(0, 20))  # 20 values, cost each side
ADVERSE_STOP_GAP_BPS = tuple(range(0, 10))  # 10, stop-only
GRID_SIZE = len(EXTRA_FEE_BPS)*len(EXTRA_SLIPPAGE_BPS)*len(ADVERSE_STOP_GAP_BPS)
VERSION = "WM_TC2_FIXED_4400_EXECUTION_FRICTION_GRID_V1"


def cost_stress_from_settled_experience(
    ledger: PrivateExperience, *,
    known_at: str, target: int = 10_000_000,
    min_symbols: int = 5, min_decision_times: int = 300,
) -> dict[str, Any]:
    if (not isinstance(target,int) or isinstance(target,bool)
            or not 1<=target<=100_000_000
            or not isinstance(min_symbols,int) or min_symbols<1
            or not isinstance(min_decision_times,int) or min_decision_times<1):
        raise ValueError("Invalid historical experiment budget")
    as_of = _time(known_at).isoformat()
    historical=[s for s in ledger.samples(resolved_before=as_of)
                if s.setup in ("WM1","WM2","WM3")
                and s.resolved_at<=as_of]
    assets={s.symbol for s in historical}
    distinct={ (s.symbol,s.interval,s.decision_at) for s in historical }
    if len(assets)<min_symbols or len(distinct)<min_decision_times:
        raise ValueError("INSUFFICIENT_REAL_DISTINCT_MARKET_DECISIONS")
    max_distinct_scenarios=len(historical)*GRID_SIZE
    if max_distinct_scenarios<target:
        raise ValueError("INSUFFICIENT_DISTINCT_FRICTION_CONFIGURATIONS_FOR_TARGET")
    # Every scenario is unique by a full immutable trade-source identity plus
    # its 3 predefined cost axes; no random resampling or repeated grid cells.
    checked=0
    positive=0
    stopped=0
    raw_positive=sum(s.net_return>0 for s in historical)
    worst=1e9
    best=-1e9
    checksum=hashlib.sha256()
    checksum.update(VERSION.encode())
    for ex in historical:
        if checked>=target:
            break
        # Immutable source snapshot linkage. This checksum also verifies
        # exact case order and all previously observed trade directions.
        checksum.update(json.dumps(
            (ex.symbol,ex.interval,ex.decision_at,ex.resolved_at,ex.side,
             ex.setup,ex.rule_id,ex.source_hash,ex.net_return,ex.stopped),
            separators=(",", ":"),sort_keys=False).encode())
        for extra_fee in EXTRA_FEE_BPS:
            if checked>=target:
                break
            fee_cost=2.0*extra_fee/10000.0
            for extra_slippage in EXTRA_SLIPPAGE_BPS:
                if checked>=target:
                    break
                fill_cost=fee_cost+2.0*extra_slippage/10000.0
                for stop_gap in ADVERSE_STOP_GAP_BPS:
                    if checked>=target:
                        break
                    # Penalize ONLY an actually stopped paper observation.
                    penalty=fill_cost+(stop_gap/10000.0 if ex.stopped else 0.0)
                    adjusted=ex.net_return-penalty
                    positive+=int(adjusted>0.0)
                    stopped+=int(ex.stopped)
                    worst=min(worst,adjusted)
                    best=max(best,adjusted)
                    checked+=1
    return {
        "mode":"REAL_BINANCE_HISTORICAL_TC2_PROXY_EXECUTION_COST_STRESS",
        "verified_execution_cost_scenario_checks":checked,
        "requested_exact_scenario_goal":target,
        "goal_met":checked==target,
        "cost_grid_unique_combinations_per_paper_trade":GRID_SIZE,
        "historical_eligible_wm_paper_outcomes":len(historical),
        "distinct_historical_asset_timeframe_decisions":len(distinct),
        "historical_asset_symbols":sorted(assets),
        "source_order_sha256":checksum.hexdigest(),
        "friction_scenarios_profitable_after_extra_costs":positive,
        "friction_scenario_positive_fraction":round(positive/checked,7),
        "stopped_position_scenarios":stopped,
        "worst_scenario_net_fraction":round(worst,8),
        "best_scenario_net_fraction":round(best,8),
        "original_positive_paper_outcomes":raw_positive,
        "ten_million_separate_market_decisions":False,
        "ten_million_distinct_price_graphs":False,
        "actual_gguf_inference_calls":0,
        "source_signals_canonical_four_books_confirmed":False,
        "actual_fills_and_protective_stops_confirmed":False,
        "optimization_on_test_prices_performed":False,
        "model_weights_updated_in_this_stress":False,
        "performance_not_predictive_out_of_sample":True,
        "execution_allowed":False,"trade_action":None,
    }


def main() -> None:
    p=argparse.ArgumentParser(description="Exact 10m paper fill-cost scenario stress; not 10m trades")
    p.add_argument("--private-journal",type=Path,required=True)
    p.add_argument("--known-at",required=True,
                   help="Historical UTC cutoff; only previously settled outcomes counted")
    p.add_argument("--target",type=int,default=10_000_000)
    p.add_argument("--min-symbols",type=int,default=5)
    p.add_argument("--min-decisions",type=int,default=300)
    a=p.parse_args()
    x=cost_stress_from_settled_experience(
        PrivateExperience(a.private_journal),known_at=a.known_at,
        target=a.target,min_symbols=a.min_symbols,
        min_decision_times=a.min_decisions)
    print(json.dumps(x,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
