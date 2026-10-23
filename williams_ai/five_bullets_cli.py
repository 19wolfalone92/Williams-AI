"""Cross-scale TC1 five-magic-bullet RESEARCH scan on 15 frozen Binance pairs.

M5/M15 require their own observed candles; this dataset has only H1,
from which complete H4/D1 candles can be formed without lookahead.
We never fabricate M5/M15 data from an H1 bar.

This is a diagnostic of observable features, NOT a historical trading
performance test (no signals sent to Binance, no fills, no slippage model).
"""
from __future__ import annotations
import argparse,json
from collections import Counter

from .causal_paper_lab import frozen_markets
from .causal_timeframes import aggregate_closed_h1
from .chart_reader import analyze_chart


def scan_fifteen() -> dict:
    markets=frozen_markets()
    results=[]
    for symbol in sorted(markets):
        x=markets[symbol]
        h1=x["bars"]
        for tf in ("H1","H4","D1"):
            rows=h1 if tf=="H1" else aggregate_closed_h1(h1,tf)
            o=analyze_chart(symbol,tf,rows,
                            as_of=rows[-1]["close_time"],tick_size=x["tick_size"])
            magic=o["tc1_five_magic_bullets"]
            possibilities=magic["last_n_wave_candidates"]
            findings=[
                {"at":v["terminal_extreme_at"],"observed":v["observed_bullets"],
                 "unknown":v["unknown_bullets"],
                 "all_five":v["all_five_observed"],
                 "bullet_status":{k:q["status"] for k,q in v["bullets"].items()}}
                for v in possibilities
            ]
            results.append({
                "symbol":symbol,"timeframe":tf,"closed_bars":len(rows),
                "wave_candidates":magic["wave_candidate_count"],
                "full_five_observed":sum(v["all_five"] for v in findings),
                "unknown_tick_volume_at_some_candidates":sum(
                    v["bullet_status"]["SQUAT_MFI_VOLUME"]=="UNKNOWN"
                    for v in findings),
                "recent_findings":findings[-2:],
                "trade_action":None,"execution_allowed":False,
            })
    summary=Counter()
    for row in results:
        summary["confirmed_wave_candidates"]+=row["wave_candidates"]
        summary["complete_bullet_sets"]+=row["full_five_observed"]
        summary["candidate_sets_missing_tick_volume"]+=row["unknown_tick_volume_at_some_candidates"]
    return {
        "mode":"READ_ONLY_TC1_BULLET_EVIDENCE_15_BINANCE_FUTURES_H1_H4_D1",
        "symbols":len(markets),
        "timeframe_symbol_scans":len(results),
        "without_observed_M5_M15_data":True,
        "volume_semantics":"Binance H1 base asset quantity is NOT TC1 tick count",
        "summary":dict(summary),
        "by_pair_timeframe":results,
        "profitability_checked":False,
        "model_intelligence_checked":False,
        "trade_action":None,"execution_allowed":False,
    }


def main()->None:
    p=argparse.ArgumentParser()
    p.add_argument("--summary",action="store_true")
    a=p.parse_args()
    result=scan_fifteen()
    if a.summary:
        result={k:v for k,v in result.items() if k!="by_pair_timeframe"}
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
