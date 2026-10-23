"""Read-only chronological Williams signal laboratory for frozen Binance H1 candles.

No API keys, no exchange calls, no auto-trade. Only closed candles are used
to *form* signals at bar close; an order simulated on the NEXT H1 candle can
be hit intrabar. Stop-first when same H1 OHLC makes path ambiguous.
This is intentionally a simplified research proxy, NOT canonical full TC2.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
import json
import math
from typing import Any

from .chart_reader import validate_closed_bars, indicators

START_EQUITY = 1000.0
FEE_BPS = 4.0
SLIPPAGE_BPS = 2.0
RISK_PER_ENTRY = 0.01
MAX_NOTIONAL_MULTIPLE = 2.0
WARMUP = 220
MODES = {"WM3", "WM2", "WM1"}
SIGNAL_TTL = {"WM3": 16, "WM2": 3, "WM1": 3}
FROZEN_DATA = Path(__file__).resolve().parents[1] / "examples/ai_binance_15pair_h1_frozen_20261022.json"


def frozen_markets(path: Path = FROZEN_DATA) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data["schema"] != "WILLIAMS_AI_PUBLIC_BINANCE_15_H1_V1":
        raise ValueError("Unexpected dataset")
    result = {}
    for symbol, val in data["pairs"].items():
        rows = [{
            "close_time": datetime.fromtimestamp(a[0] / 1000, timezone.utc).isoformat(),
            "open": a[1], "high": a[2], "low": a[3], "close": a[4],
            "volume": a[5], "closed": True,
        } for a in val["klines"]]
        result[symbol] = {"bars": rows, "tick_size": float(val["tick_size"])}
    return result


def _reversal(bars: list[Any], ind: dict[str, list[Any]], i: int, side: str) -> bool:
    """Explicit WM1 slope *approximation*, not proof of canonical angulation."""
    if i < 45:
        return False
    x,prev = bars[i], bars[i-1]
    jaw, teeth, lips = (ind[k][i] for k in ("jaw", "teeth", "lips"))
    if any(v is None for v in (jaw, teeth, lips, ind["jaw"][i-5], ind["jaw"][i-4])):
        return False
    mid = (x.low+x.high)/2
    if side=="LONG":
        valid = x.low<prev.low and x.close>mid and x.high<min(jaw,teeth,lips)
        slope = (x.low-bars[i-5].low)/5
        prior = bars[i-5].close>bars[i-1].close
        growing = abs(x.low-jaw)>abs(bars[i-4].low-ind["jaw"][i-4])
    else:
        valid = x.high>prev.high and x.close<mid and x.low>max(jaw,teeth,lips)
        slope = (x.high-bars[i-5].high)/5
        prior = bars[i-5].close<bars[i-1].close
        growing = abs(x.high-jaw)>abs(bars[i-4].high-ind["jaw"][i-4])
    mouth_slope=(jaw-ind["jaw"][i-5])/5
    return bool(valid and prior and growing and abs(slope)>1.05*abs(mouth_slope))


def _candidate(bars: list[Any], ind: dict[str, list[Any]], mode: str,
               i: int, tick: float) -> list[dict[str, Any]]:
    """Form at bar i CLOSE, eligible only on bar i+1 or later."""
    if i < 45:
        return []
    results=[]
    for side in ("LONG","SHORT"):
        sign=1 if side=="LONG" else -1
        if mode=="WM3":
            p=i-2
            span=bars[p-2:p+3]
            extreme_high=bars[p].high>max(c.high for j,c in enumerate(span) if j!=2)
            extreme_low=bars[p].low<min(c.low for j,c in enumerate(span) if j!=2)
            if extreme_high and extreme_low:  # outside bar ambiguous
                continue
            if not ((side=="LONG" and extreme_high) or (side=="SHORT" and extreme_low)):
                continue
            pivot=bars[p]
            level=pivot.high if sign==1 else pivot.low
            trigger=round(level+sign*tick,10)
        elif mode=="WM2":
            ao=ind["ao"]
            if any(v is None for v in ao[i-3:i+1]):
                continue
            seq=ao[i-3:i+1]
            if not all(sign*(seq[k]-seq[k-1])>0 for k in (1,2,3)):
                continue
            # First observed 3-bar run; do not produce duplicate signals
            if i>45 and ao[i-4] is not None and sign*(ao[i-3]-ao[i-4])>0:
                continue
            trigger=round((bars[i].high if sign==1 else bars[i].low)+sign*tick,10)
        else:
            if not _reversal(bars,ind,i,side):
                continue
            trigger=round((bars[i].high if sign==1 else bars[i].low)+sign*tick,10)
        stop=min(x.low for x in bars[i-4:i+1]) if sign==1 else max(x.high for x in bars[i-4:i+1])
        if sign*(trigger-stop)<=2*tick:
            continue
        results.append({"formed":i, "side":side, "trigger":trigger,
                        "stop":stop, "type":mode})
    return results


def _entry_price(c: Any, trigger: float, side: str) -> float:
    # Opening gap through the level gets the worse price.
    raw=max(c.open,trigger) if side=="LONG" else min(c.open,trigger)
    sign=1 if side=="LONG" else -1
    return raw*(1+sign*SLIPPAGE_BPS/10000)


def _exit_price(c: Any, stop:float, side:str) -> float:
    # Conservative gap/worst-case assumption, including the entry bar.
    raw=min(stop,c.open) if side=="LONG" else max(stop,c.open)
    sign=1 if side=="LONG" else -1
    return raw*(1-sign*SLIPPAGE_BPS/10000)


def _pnl(qty:float, side:str, entry:float, exit_price:float) -> float:
    sign=1 if side=="LONG" else -1
    return qty*sign*(exit_price-entry)


def simulate_one(
    rows: list[dict[str,Any]], symbol:str, tick_size:float, *,
    mode:str="WM3", start_index:int=WARMUP, final_index:int|None=None,
    max_events:int=300,
) -> dict[str,Any]:
    """Single strategy, one campaign per pair, no leverage/funding realism claim.

    Initial stop is derived from the signal's 5 known CLOSED bars. Existing
    position is never closed merely because an opposite Wise Man appears.
    Trailing takes place AFTER bar close and only tightens SL.
    """
    if mode not in MODES:
        raise ValueError("Research mode must be WM1, WM2 or WM3")
    if not isinstance(tick_size,(int,float)) or tick_size<=0 or not math.isfinite(tick_size):
        raise ValueError("Invalid Binance tick")
    end=len(rows) if final_index is None else final_index
    if not isinstance(end,int) or not 220 < end <= len(rows):
        raise ValueError("Need > 220 chronological bars")
    if not 220<=start_index<end:
        raise ValueError("Invalid evaluation start")
    candles,gaps=validate_closed_bars(rows[:end],"H1",rows[end-1]["close_time"])
    if gaps:
        raise ValueError("History contains gaps")
    ind=indicators(candles)
    pending:dict[str,dict[str,Any]]={}
    position=None
    eq=START_EQUITY
    peak=eq
    max_dd=0.0
    mtm_peak=START_EQUITY
    mtm_max_dd=0.0
    exposure_bars=0
    closed=0
    wins=0
    losses=0
    gross_wins=0.0
    gross_losses=0.0
    fees=0.0
    signals=0
    rejected=0
    forced_flat=0
    stops=0
    traces=[]

    def append(i:int,event:str,**fields:Any)->None:
        if len(traces)<max_events:
            traces.append({"index":i,"time":candles[i].at.isoformat(),
                           "event":event,**fields})

    for i in range(start_index,end):
        c=candles[i]
        # A trigger created on the previous bar may fill on THIS bar.
        if position is None:
            hit=[]
            for side,candidate in list(pending.items()):
                if i<=candidate["formed"]:
                    continue
                if i-candidate["formed"]>SIGNAL_TTL[mode]:
                    del pending[side]
                    continue
                level=candidate["trigger"]
                crossed=(c.high>=level if side=="LONG" else c.low<=level)
                if not crossed:
                    continue
                teeth=ind["teeth"][i] if mode=="WM3" else None
                if mode=="WM3" and (teeth is None or
                        (level<=teeth if side=="LONG" else level>=teeth)):
                    rejected+=1
                    append(i,"REJECTED_TEETH_AT_TRIGGER",side=side,trigger=level)
                    del pending[side]
                    continue
                hit.append(candidate)
            if len(hit)>1:
                forced_flat+=1
                append(i,"AMBIGUOUS_BOTH_SIDES_TOUCHED_NO_ENTRY")
                pending.clear()
            elif hit:
                signal=hit[0]
                pending.clear()
                side=signal["side"]
                entry=_entry_price(c,signal["trigger"],side)
                stop=signal["stop"]
                sign=1 if side=="LONG" else -1
                risk_unit=sign*(entry-stop)
                if risk_unit<=tick_size:
                    rejected+=1
                    append(i,"INVALID_STOP_AFTER_GAP",side=side)
                else:
                    budget=eq*RISK_PER_ENTRY
                    qty=min(budget/risk_unit,eq*MAX_NOTIONAL_MULTIPLE/entry)
                    if qty<=0:
                        rejected+=1
                    else:
                        entry_fee=qty*entry*FEE_BPS/10000
                        eq-=entry_fee
                        fees+=entry_fee
                        position={"side":side,"entry":entry,"qty":qty,
                                  "entry_fee":entry_fee,"stop":stop,"entered":i,"source":mode}
                        append(i,"SIM_ENTRY",side=side,entry=round(entry,8),
                               stop=round(stop,8),qty=round(qty,8),source=mode)
        # Stop-first within any OHLC bar, even ambiguous entry and stop touch.
        if position is not None:
            side=position["side"]
            stop=position["stop"]
            if (c.low<=stop if side=="LONG" else c.high>=stop):
                price=_exit_price(c,stop,side)
                outcome=_pnl(position["qty"],side,position["entry"],price)
                close_fee=position["qty"]*price*FEE_BPS/10000
                fees+=close_fee
                eq+=outcome-close_fee
                closed+=1
                stops+=1
                # Count trade outcomes AFTER entry and exit commission.
                # The original win-rate classified tiny gross wins as profitable
                # even when both exchange-side fees made the campaign lose.
                net_trade=outcome-position["entry_fee"]-close_fee
                if net_trade>0:wins+=1;gross_wins+=net_trade
                else:losses+=1;gross_losses+=abs(net_trade)
                append(i,"SIM_STOP",side=side,exit=round(price,8),
                       pnl_before_exit_fee=round(outcome,6),
                       pnl_net_all_fees=round(net_trade,6),
                       entry_fee=round(position["entry_fee"],6),
                       exit_fee=round(close_fee,6))
                position=None
                peak=max(peak,eq)
                max_dd=max(max_dd,(peak-eq)/peak)
            else:
                # New protective level can only become effective next candle.
                recent=candles[max(0,i-4):i+1]
                trail=(min(b.low for b in recent) if side=="LONG"
                       else max(b.high for b in recent))
                old=position["stop"]
                safe=(min(c.close-tick_size,trail) if side=="LONG"
                      else max(c.close+tick_size,trail))
                tightened=(safe>old+tick_size/2 if side=="LONG" else safe<old-tick_size/2)
                if tightened:
                    position["stop"]=safe
                    append(i,"SIM_STOP_TIGHTEN_NEXT_BAR",side=side,
                           old=round(old,8),new=round(safe,8))
        # Closed-bar signal formation. Never enter until i+1.
        if position is None:
            for sig in _candidate(candles,ind,mode,i,tick_size):
                pending[sig["side"]]=sig
                signals+=1
                append(i,"SETUP_CONFIRMED_WAIT_NEXT_BAR",side=sig["side"],
                       trigger=sig["trigger"],signal=mode)
        # Opposite signals while holding are observations, NEVER forced exits.
        else:
            candidates=_candidate(candles,ind,mode,i,tick_size)
            if any(s["side"]!=position["side"] for s in candidates):
                append(i,"OPPOSITE_SIGNAL_IGNORED_NOT_EXIT")
        # Evaluate observable mark-to-market risk at EACH completed candle,
        # including unrealized losses on the still-open position. An assumed
        # close here is for risk reporting ONLY; it is NOT a trade exit.
        mtm=eq
        if position is not None:
            exposure_bars+=1
            side=position["side"]
            sign=1 if side=="LONG" else -1
            mark_exit=c.close*(1-sign*SLIPPAGE_BPS/10000)
            mtm+=_pnl(position["qty"],side,position["entry"],mark_exit)
            mtm-=position["qty"]*mark_exit*FEE_BPS/10000
        mtm_peak=max(mtm_peak,mtm)
        mtm_max_dd=max(mtm_max_dd,(mtm_peak-mtm)/mtm_peak)
    unrealized=None
    mark_to_market_equity=eq
    if position is not None:
        sign=1 if position["side"]=="LONG" else -1
        mark=candles[end-1].close
        unrealized=_pnl(position["qty"],position["side"],position["entry"],mark)
        sign=1 if position["side"]=="LONG" else -1
        assumed_exit=mark*(1-sign*SLIPPAGE_BPS/10000)
        mark_to_market_equity=eq+_pnl(position["qty"],position["side"],position["entry"],assumed_exit)
        mark_to_market_equity-=position["qty"]*assumed_exit*FEE_BPS/10000
    return {
        "symbol":symbol,"mode":mode,"timeframe":"H1",
        "from":candles[start_index].at.isoformat(),"to":candles[end-1].at.isoformat(),
        "closed":closed,"wins":wins,"losses":losses,"signals_confirmed":signals,
        "rejected_setups":rejected,"both_sides_ambiguous":forced_flat,
        "stop_exits":stops,"non_stop_exits":0,
        "realized_return_pct":round((eq/START_EQUITY-1)*100,4),
        "cash_equity":round(eq,5),
        "mark_to_market_equity_after_estimated_exit_costs":round(mark_to_market_equity,5),
        "mark_to_market_total_return_pct":round((mark_to_market_equity/START_EQUITY-1)*100,4),
        "fees_in_equity":round(fees,5),
        "max_realized_drawdown_pct":round(max_dd*100,4),
        "max_mtm_closed_bar_drawdown_pct":round(mtm_max_dd*100,4),
        "bars_with_open_risk":exposure_bars,
        "profit_factor_net_all_fees":round(gross_wins/gross_losses,4) if gross_losses else None,
        "open_position":(
            {"side":position["side"],"stop":position["stop"],
             "unrealized_mark_pnl":round(unrealized,6),
             "NOT_a_closed_trade":True} if position else None
        ),
        "benchmark_buy_and_hold_pct":round(
            (candles[end-1].close/candles[start_index].open-1)*100,4),
        "trace":traces,"trace_truncated":len(traces)>=max_events,
        "read_only":True,"trade_action":None,"execution_allowed":False,
        "assumptions":{
            "max_notional_multiple":MAX_NOTIONAL_MULTIPLE,
            "risk_per_trade_fraction":RISK_PER_ENTRY,
            "fee_bps_each_side":FEE_BPS,"slippage_bps_each_side":SLIPPAGE_BPS,
            "protective_stop":"assumed instantaneous in simulation, NOT exchange verified",
            "fills":"OHLC trigger conservatively filled; ambiguous stop prioritized",
            "last_position":"unrealized only; not forced closed",
            "funding":"excluded",
            "market_microstructure":"no order book, queue, latency, partial fills, liquidation or bid/ask",
            "strategies":"simplified WM candidates; not canonical full Williams campaign/add-ons",
        }
    }


def run_15(path:Path=FROZEN_DATA,*,test_fraction:float=0.25)->dict[str,Any]:
    if not 0.15<=test_fraction<=0.5:
        raise ValueError("Invalid test fraction")
    markets=frozen_markets(path)
    reports=[]
    for symbol in sorted(markets):
        x=markets[symbol]
        n=len(x["bars"])
        beginning=max(WARMUP,int(n*(1-test_fraction)))
        for mode in ("WM3","WM2","WM1"):
            out=simulate_one(x["bars"],symbol,x["tick_size"],
                             mode=mode,start_index=beginning,max_events=80)
            out.pop("trace")  # count summary only; replay CLI supports complete journal.
            reports.append(out)
    aggregate={}
    for mode in ("WM3","WM2","WM1"):
        subset=[x for x in reports if x["mode"]==mode]
        aggregate[mode]={
            "pairs":len(subset),
            "closed_trades":sum(x["closed"] for x in subset),
            "wins":sum(x["wins"] for x in subset),
            "cash_realized_return_pct_equal_initial_capital":round(
                sum(x["realized_return_pct"] for x in subset)/len(subset),4),
            "mtm_total_return_pct_equal_initial_capital":round(
                sum(x["mark_to_market_total_return_pct"] for x in subset)/len(subset),4),
            "worst_pair_mtm_drawdown_pct":max(x["max_mtm_closed_bar_drawdown_pct"] for x in subset),
            "pairs_with_open_position":sum(x["open_position"] is not None for x in subset),
            "rejected":sum(x["rejected_setups"] for x in subset),
            "non_stop_exits":sum(x["non_stop_exits"] for x in subset),
        }
    return {
        "status":"FROZEN_HISTORICAL_WALK_FORWARD_RESEARCH_ONLY",
        "pairs":len(markets),"modes":["WM3","WM2","WM1"],"results":reports,
        "aggregate":aggregate,"trade_action":None,"execution_allowed":False,
        "model_weights_updated":False,
        "cautions":[
            "This exercise is deterministic rule-based AI training scaffolding, NOT decisions from an actual GGUF model.",
            "WM1 angulation is a proxy; WM2/WM3 trade management and add-ons are not fully book-verified.",
            "No hidden future candle enters a prefix decision, but the test set is not a pristine unseen market.",
            "An OHLC-only simulator may significantly misestimate fills, drawdowns, funding and PnL.",
            "Capital is isolated per pair and per research mode, with no portfolio exposure constraints.",
        ],
    }
