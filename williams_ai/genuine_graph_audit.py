"""Genuine DIFFERENT historical OHLC graphs across many Binance instruments.

Unlike 10m artificial transformations, these are 4 separate real historical
closed timepoints x H1/H4/D1 x each verified January-February 2025 symbol.
No synthetic price candles, future bars, orders, GGUF inference, or wave truth.
"""
from __future__ import annotations
from collections import Counter
from typing import Any

from .chart_reader import analyze_chart
from .causal_timeframes import aggregate_closed_h1

CUTS=(1079,1175,1271,1367)
TF=("H1","H4","D1")


def genuine_market_graph_audit(markets:dict[str,list[dict[str,Any]]])->dict:
    totals=Counter()
    per_token={}
    for sym,history in sorted(markets.items()):
        if len(history)<1370 or any(x.get("closed") is not True for x in history):
            raise ValueError("Cannot audit incomplete real 2025 symbol history")
        facts=Counter()
        seen=set()
        for end in CUTS:
            prefix=history[:end+1]
            for tf in TF:
                bars=prefix if tf=="H1" else aggregate_closed_h1(prefix,tf)
                if len(bars)<45:
                    raise ValueError("Not enough CLOSED parent TF candles")
                chart=analyze_chart(sym,tf,bars,as_of=bars[-1]["close_time"])
                if chart["as_of"]!=bars[-1]["close_time"] or chart["gaps"]:
                    raise ValueError("Unreliable timeframe/continuous graph result")
                if chart["execution_allowed"] is not False or not chart["no_lookahead"]:
                    raise ValueError("Executed order or future candle contamination")
                if chart.get("wave_structure",{}).get("wave_degree")!="UNKNOWN":
                    raise ValueError("Unverified Elliott wave degree asserted as certainty")
                magic=chart.get("tc1_five_magic_bullets",{})
                for item in magic.get("last_n_wave_candidates",[]):
                    if item["all_five_observed"]:
                        raise ValueError("Binance base volume falsely certified TC1 Squat")
                facts["candlestick_graphs"]+=1
                facts["signals_observed"]+=len(chart["candidates"])
                facts["fractal_signals"]+=sum(
                    x.get("type")=="WM3_FRACTAL" for x in chart["candidates"])
                facts["wise_man_two_candidates"]+=sum(
                    x.get("type")=="WM2_SUPER_AO" for x in chart["candidates"])
                facts["five_bullet_wave_hypotheses"]+=magic.get("wave_candidate_count",0)
                facts["highest_tf_D1_passports"]+=tf=="D1"
                facts["H1_passports"]+=tf=="H1"
                facts["H4_passports"]+=tf=="H4"
                seen.add((end,tf))
            # Future timestamp cannot be sent with a past chart cutoff.
            if end+1<len(history):
                try:
                    analyze_chart(sym,"H1",history[:end+2],
                                  as_of=prefix[-1]["close_time"])
                except ValueError as e:
                    if "NO_LOOKAHEAD" not in str(e):
                        raise
                    facts["actual_chart_reader_future_bar_blocked"]+=1
                else:
                    raise ValueError("FUTURE_BAR_WAS_NOT_REJECTED")
        if len(seen)!=12:
            raise ValueError("Did not inspect 12 different historic charts per token")
        totals.update(facts)
        per_token[sym]=dict(facts)
    return {
        "mode":"GENUINE_2025_BINANCE_GRAPH_SNAPSHOTS_4_CUTS_X_3_TF_PER_TOKEN",
        "genuine_graphs_analyzed":totals["candlestick_graphs"],
        "unique_real_markets":len(markets),
        "real_causal_graphs_per_market":12,
        "each_graph_has_different_timeframe_or_closed_timestamp":True,
        "closed_future_bar_rejection_tests":totals["actual_chart_reader_future_bar_blocked"],
        "deterministic_WM3_confirmations_seen":totals["fractal_signals"],
        "WM2_momentum_candidates_seen":totals["wise_man_two_candidates"],
        "unverified_five_bullet_wave_hypotheses":totals["five_bullet_wave_hypotheses"],
        "counting_multiple_graph_contexts_not_independent_price_outcomes":True,
        "GGUF_visual_understanding_verified":False,
        "trading_profitability_verified":False,
        "per_token":per_token,
        "trade_action":None,"execution_allowed":False,
    }
