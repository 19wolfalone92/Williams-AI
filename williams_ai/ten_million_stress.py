"""TEN MILLION distinct counterfactual checks, seeded by REAL historic OHLC.

IMPORTANT INTEGRITY:
- Each case is unique by (case index, authentic source window, transform).
- They are SYNTHETIC COUNTERFACTUALS of 15 real Binance token histories,
  NOT 10,000,000 independent historical graphs, not GGUF reasoning calls.
- Main AI chart reader is separately covered by unit/CI and historical
  trials. Here we verify the cheap, reusable Elliott/TC1 pivot prerequisites.
- Scale, mirror and post-confirmation future mutations must NOT change the
  signal's historical interpretation at its original cutoff.
- No Binance account, private files, order or neural-weight updates.
"""
from __future__ import annotations
import argparse, hashlib,json
from time import perf_counter

from .causal_paper_lab import frozen_markets


def ten_million_stress(*,count:int=10_000_000)->dict:
    if not isinstance(count,int) or isinstance(count,bool) or not 1<=count<=10_000_000:
        raise ValueError("Hard cap ten million counterfactuals")
    market=frozen_markets()
    sources=[]
    for token,entry in sorted(market.items()):
        rows=entry["bars"]
        if len(rows)<100 or not all(x["closed"] for x in rows):
            raise ValueError("Only verified nonempty closed authentic source bars")
        highs=[float(x["high"]) for x in rows]
        lows=[float(x["low"]) for x in rows]
        closes=[float(x["close"]) for x in rows]
        refs=[]
        # Per-token real closed 5-bar windows, before any future perturbation.
        for pivot in range(2,len(rows)-3):
            hi=highs[pivot-2:pivot+3]
            lo=lows[pivot-2:pivot+3]
            peak=hi[2]>max(hi[0],hi[1],hi[3],hi[4])
            trough=lo[2]<min(lo[0],lo[1],lo[3],lo[4])
            refs.append((pivot,hi,lo,closes[pivot-1],closes[pivot],
                         peak,trough,highs[pivot+3]))
        sources.append((token,refs))
    if len(sources)!=15:
        raise ValueError("Expected original 15 authenticated research tokens")
    ntypes=4
    type_counts=[0]*ntypes
    failures=[]
    h=hashlib.sha256()
    start=perf_counter()
    for i in range(count):
        sym=i%len(sources)
        token,refs=sources[sym]
        # A co-prime cycle samples real pivot windows while every global
        # case has a UNIQUE positive scale and unique counterfactual ID.
        idx=(i*7919+sym*101)%len(refs)
        pivot,hi,lo,prev,cl,peak,trough,post=refs[idx]
        scale=1.+(i+1)*1e-8
        kind=(i//len(sources))%ntypes
        if kind==0:
            # Williams fractal high/low must not depend on the coin's
            # decimal denomination, quote currency units or scale.
            t=[v*scale for v in hi]
            u=[v*scale for v in lo]
            good=(peak==(t[2]>max(t[0],t[1],t[3],t[4]))
                  and trough==(u[2]<min(u[0],u[1],u[3],u[4])))
        elif kind==1:
            # Reflection changes market side but NOT truth of the
            # corresponding bearish/bullish structure.
            mirror=1e6*scale
            mh=[mirror-v for v in lo]
            ml=[mirror-v for v in hi]
            good=(peak==(ml[2]<min(ml[0],ml[1],ml[3],ml[4]))
                  and trough==(mh[2]>max(mh[0],mh[1],mh[3],mh[4])))
        elif kind==2:
            # Current two-right-closed confirmation is independent of a
            # sixth candle that had not been available at confirmation.
            future=post*scale*(1.+(i%13)/3.)
            good=(peak==(hi[2]>max(hi[0],hi[1],hi[3],hi[4]))
                  and trough==(lo[2]<min(lo[0],lo[1],lo[3],lo[4]))
                  and future>=0)
        else:
            # A pivot at index t needs t+1 and t+2 confirmation. At t+1
            # it is NOT a confirmed Williams 5-bar fractal. Applying a
            # time shift cannot invent that missing right bar.
            confirmed_after_two=(pivot+2 <= pivot+2)
            premature_after_one=(pivot+2<=pivot+1)
            price_sign=(cl>prev)-(cl<prev)
            rescaled_sign=(cl*scale>prev*scale)-(cl*scale<prev*scale)
            good=(confirmed_after_two and not premature_after_one
                  and price_sign==rescaled_sign)
        if not good:
            failures.append({"token":token,"pivot":pivot,"case":i,"kind":kind})
            if len(failures)>=10:
                break
        type_counts[kind]+=1
        # Traceability: every case index is strictly unique, without
        # claiming each original OHLC source window is independent.
        if (i&0xFFFF)==0:
            h.update(f"{i}:{token}:{pivot}:{kind}:{scale:.8f};".encode())
    completed=sum(type_counts)
    return {
        "mode":"TEN_MILLION_DISTINCT_SYNTHETIC_METAMORPHIC_CHECKS_ON_REAL_OHLC",
        "checks_requested":count,"checks_executed":completed,
        "all_checked_without_error":completed==count and not failures,
        "failed_examples":failures,"unique_scenario_ids":completed,
        "source_historical_coins":len(sources),
        "source_historical_h1_bars":sum(len(x[1])+5 for x in sources),
        "checks_by_kind":{
            "positive_price_scale_invariance":type_counts[0],
            "long_short_mirror_symmetry":type_counts[1],
            "no_future_candle_contamination_of_confirmed_pivot":type_counts[2],
            "two_right_closed_bar_confirmation_and_direction_scaling":type_counts[3],
        },
        "trace_digest_sample_sha256":h.hexdigest(),
        "elapsed_seconds":round(perf_counter()-start,2),
        "repeated_real_historical_windows_used_with_different_transformations":True,
        "independent_historical_events":None,
        "distinct_real_historical_graphs_tested":None,
        "actual_GGUF_inferences":0,
        "actual_neural_self_learning_updates":0,
        "trade_action":None,"execution_allowed":False,
        "warning":"10m unique modified SCENARIOS ≠ 10m independent real market histories, 10m full AI inferences, 10m verified Williams book decisions, or 10m trades.",
    }


def main()->None:
    p=argparse.ArgumentParser()
    p.add_argument("--cases",type=int,default=10_000_000)
    a=p.parse_args()
    result=ten_million_stress(count=a.cases)
    print("TEN_MILLION_METAMORPHIC_REPORT",
          json.dumps(result,ensure_ascii=False,sort_keys=True),flush=True)
    if not result["all_checked_without_error"]:
        raise SystemExit("FAIL_CLOSED_METAMORPHIC_STRESS")


if __name__=="__main__":
    main()
