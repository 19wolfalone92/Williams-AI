"""Reproducible, resumable market-experience marathon for Williams AI.

Discovers all currently listed USDT perpetuals if requested; learns over any
strictly verified, closed Binance-native candle caches. No fabricated candles
or extrapolation of missing tokens. The only affordable million-scale step
is batched numerical replay; million GGUF forward passes remain a separately
measured compute budget and are NOT claimed here.

Usage:
 python -m williams_ai.trader_marathon --frozen-fixture \
     --private-journal /private/williams_trader_memory.sqlite3
 python -m williams_ai.trader_marathon --history-db /private/market.db \
     --venue PUBLIC_USDM --symbols BTCUSDT ETHUSDT SOLUSDT \
     --intervals 1h 4h 1d --private-journal /private/williams_memory.sqlite3
 python -m williams_ai.trader_marathon --discover-universe \
     --private-universe-snapshot /private/binance_universe_20261024.json

All evaluations are TC2 approximate signal-conditioned PAPER, not authoritative
Bill Williams trades. All holds are global chronological across crypto assets.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from .binance_wave_history import DurableWaveHistory, INTERVAL_MS, VENUES
from .binance_universe import discover_universe, write_universe_snapshot, read_universe_snapshot
from .causal_paper_lab import frozen_markets
from .continual_trader import PrivateExperience, learn_from_private_experience
from .williams_signal_experience import signal_conditioned_outcomes

SUPPORTED = {"15m":"M15","1h":"H1","4h":"H4","1d":"D1"}


def _split_points(items: list[Any]) -> tuple[str,str,str] | None:
    times = sorted({x.decision_at for x in items})
    if len(times) < 15:
        return None
    a = times[int(len(times) * .6)]
    b = times[int(len(times) * .8)]
    last = max(x.resolved_at for x in items)
    if not a < b < last:
        return None
    return a,b,last


def marathon(
    ledger: PrivateExperience,
    markets: dict[str,dict[str,Any]], *,
    horizon: int = 12,goal: int = 10_000_000,
    min_train: int = 40,min_validation: int = 10,
) -> dict[str,Any]:
    if not isinstance(goal,int) or isinstance(goal,bool) or goal<1:
        raise ValueError("Invalid 10m-budget counter")
    per_market=[]
    all_cases=[]
    checked=0
    no_data=0
    for symbol in sorted(markets):
        state=markets[symbol]
        for tf,entry in sorted(state.items()):
            try:
                history=signal_conditioned_outcomes(
                    symbol,entry["rows"],tf,tick_size=entry["tick_size"],
                    horizon=horizon)
            except (ValueError,KeyError,TypeError) as exc:
                per_market.append({"symbol":symbol,"tf":tf,
                                   "status":"INVALID_OR_INSUFFICIENT_CLOSED_HISTORY",
                                   "error_type":type(exc).__name__})
                no_data+=1
                continue
            checked += history["research_approximated_wm_candidates"]
            all_cases.extend(history["outcomes"])
            per_market.append({
                "symbol":symbol,"tf":tf,
                "source_sha256":history["source_sha256"],
                "wm_candidates_checked":history["research_approximated_wm_candidates"],
                "counterfactual_next_bar_entries":len(history["outcomes"]),
                "missed_triggers":history["candidates_no_next_bar_trigger"],
                "rejected_invalid":history["other_rejections"],
                "setup_types":history["setup_counts"],
                "trade_action":None,
            })
    if not all_cases:
        return {
            "mode":"WILLIAMS_AI_SIGNAL_CONDITIONED_EXPERIENCE",
            "reason":"NO_VERIFIED_HISTORICAL_NEXT_BAR_SIGNAL_SAMPLES",
            "candidate_evaluations_performed":checked,
            "goal_candidate_evaluations":goal,
            "goal_verified_as_completed":False,
            "market_failures":no_data,
            "markets":per_market,"execution_allowed":False,"trade_action":None,
        }
    observed=max(x.resolved_at for x in all_cases)
    written=ledger.append(all_cases,knowledge_as_of=observed)
    points=_split_points(all_cases)
    model=None
    if points is not None:
        try:
            model=learn_from_private_experience(
                ledger,train_until=points[0],validate_until=points[1],
                observe_until=points[2],min_train=min_train,
                min_validation=min_validation,
                require_williams_setup=True)
            model["persisted_new_candidate"]=ledger.save_policy(model)
        except ValueError as exc:
            model={"state":"INSUFFICIENT_TRAIN_VALID_TEST_EVIDENCE",
                   "reason":str(exc),"execution_allowed":False}
    return {
        "mode":"WILLIAMS_AI_CLOSED_MARKET_CONTINUAL_LEARNING",
        "symbol_timeframe_streams":len(per_market),
        "distinct_crypto_symbols":len({x["symbol"] for x in per_market}),
        "book_signal_proxy_checks_completed":checked,
        "resolved_book_signal_proxy_entry_examples":len(all_cases),
        "goal_candidate_evaluations":goal,
        "additional_candidate_checks_needed_to_goal":max(0,goal-checked),
        "goal_verified_as_completed":checked>=goal,
        "ten_million_independent_trades_claimed":False,
        "actual_gguf_book_trading_decisions_tested_in_this_run":0,
        "verified_historical_binance_exchange_fills":0,
        "private_persistent_learning":written,
        "learnt_numeric_policy":model,
        "invalid_or_incomplete_market_streams":no_data,
        "per_market":per_market,
        "source_rule_disclaimer":"WM1 angulation and WM2/WM3 exits are TC2 proxies, not full approved Williams campaigns.",
        "model_weight_training":"SMALL_SCORER_ONLY_NO_GGUF_FINETUNE",
        "live_or_testnet_trading_authorized":False,
        "execution_allowed":False,"trade_action":None,
    }


def main() -> None:
    p=argparse.ArgumentParser(description="Self-improving Williams market research; no trading authorization")
    p.add_argument("--private-journal",type=Path)
    p.add_argument("--frozen-fixture",action="store_true",
                   help="Use 15-pair previously reviewed real public Binance H1 archive (NOT blind)")
    p.add_argument("--history-db",type=Path)
    p.add_argument("--venue",choices=tuple(VENUES),default="PUBLIC_USDM")
    p.add_argument("--symbols",nargs="+")
    p.add_argument("--from-universe-snapshot",type=Path,
                   help="Process all exchange-verified USDT perpetual pairs in bounded batches")
    p.add_argument("--universe-batch-offset",type=int,default=0)
    p.add_argument("--universe-batch-size",type=int,default=25)
    p.add_argument("--intervals",nargs="+",default=["1h"])
    p.add_argument("--window",type=int,default=30000)
    p.add_argument("--horizon",type=int,default=12)
    p.add_argument("--goal",type=int,default=10_000_000)
    p.add_argument("--discover-universe",action="store_true")
    p.add_argument("--private-universe-snapshot",type=Path)
    args=p.parse_args()
    if args.discover_universe:
        u=discover_universe(args.venue)
        if not args.private_universe_snapshot:
            p.error("An immutable private universe snapshot path is required")
        write_universe_snapshot(u,args.private_universe_snapshot)
        print(json.dumps(u.public_report(),ensure_ascii=False,indent=2))
        return
    if not args.private_journal:
        p.error("Learning requires a local private journal")
    if args.frozen_fixture == bool(args.history_db):
        p.error("Provide either --frozen-fixture or --history-db, not both/neither")
    if (args.from_universe_snapshot and (args.frozen_fixture or args.symbols)):
        p.error("--from-universe-snapshot requires a historical DB and no explicit symbols")
    if (args.universe_batch_offset<0 or
            not 1<=args.universe_batch_size<=100):
        p.error("Universe batches require a nonnegative offset and 1..100 symbols")
    if not 250<=args.window<=50000 or not 1<=args.horizon<=48:
        p.error("Invalid research history window or fixed valuation horizon")
    if args.frozen_fixture:
        recorded=frozen_markets()
        subset=args.symbols or sorted(recorded)
        if any(s not in recorded for s in subset):
            p.error("Unknown symbol in actual frozen public futures archive")
        markets={s:{"H1":{"rows":recorded[s]["bars"],
                            "tick_size":recorded[s]["tick_size"]}}
                 for s in subset}
        dataset="PREVIOUSLY_REVIEWED_15PAIR_REAL_USDM_H1_NOT_BLIND"
    else:
        assert args.history_db is not None
        if args.from_universe_snapshot:
            snapshot=read_universe_snapshot(
                args.from_universe_snapshot,venue=args.venue)
            chosen=list(snapshot.instruments)[
                args.universe_batch_offset:
                args.universe_batch_offset+args.universe_batch_size]
            if not chosen:
                p.error("Selected all-token universe batch is empty")
            args.symbols=chosen
        if not args.symbols:
            p.error("--symbols OR --from-universe-snapshot must specify source")
        if not args.intervals or any(t not in SUPPORTED for t in args.intervals):
            p.error("Only currently book-signal-tested native 15m/1h/4h/1d")
        store=DurableWaveHistory(args.history_db)
        markets={}
        for symbol in args.symbols:
            markets[symbol]={}
            for native in args.intervals:
                stats=store.status(args.venue,symbol,native)
                if stats["tick_size"] is None:
                    continue
                rows=store.recent(args.venue,symbol,native,limit=args.window)
                markets[symbol][SUPPORTED[native]]={
                    "rows":rows,"tick_size":float(stats["tick_size"])}
        dataset="PRIVATE_NATIVE_CLOSED_EXCHANGE_CANDLE_ARCHIVE_NOT_PROVEN_BLIND"
    result=marathon(PrivateExperience(args.private_journal),markets,
                    horizon=args.horizon,goal=args.goal)
    result["dataset_type"]=dataset
    result["historical_futures_venue"]=args.venue
    result["universe_is_live_listing_snapshot_not_delisted_history"]=(
        bool(args.from_universe_snapshot))
    result["universe_batch_offset"]=args.universe_batch_offset
    result["universe_batch_size_requested"]=args.universe_batch_size
    result["all_available_binance_tokens_were_tested"]=False
    if args.from_universe_snapshot:
        result["universe_total_current_perpetual_symbols"]=len(snapshot.instruments)
        result["universe_batch_symbols"]=args.symbols
        result["universe_remaining_after_batch"]=max(
            0,len(snapshot.instruments)-
            (args.universe_batch_offset+len(args.symbols)))
    print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))


if __name__=="__main__":
    main()
