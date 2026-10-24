"""Repeatable offline research on 15 frozen USDT perpetual H1 histories.

python -m williams_ai.paper_lab_cli --summary
python -m williams_ai.paper_lab_cli --symbol ETHUSDT --mode WM3 --tail 35

Only the candle prefix is visible to each decision. No Binance calls.
"""
from __future__ import annotations
import argparse
import json
from .causal_paper_lab import frozen_markets,run_15,simulate_one

def main()->None:
    p=argparse.ArgumentParser()
    p.add_argument("--summary",action="store_true")
    p.add_argument("--symbol",help="USDT pair to inspect")
    p.add_argument("--mode",choices=["WM1","WM2","WM3"],default="WM3")
    p.add_argument("--tail",type=int,default=25)
    args=p.parse_args()
    if args.symbol:
        markets=frozen_markets()
        if args.symbol not in markets:
            p.error("Unknown symbol in frozen 15-pair dataset")
        x=markets[args.symbol]
        start=max(220,int(len(x["bars"])*0.75))
        result=simulate_one(x["bars"],args.symbol,x["tick_size"],mode=args.mode,start_index=start)
        result["trace"]=result["trace"][-min(max(args.tail,0),150):]
    else:
        result=run_15()
        if args.summary:
            result={"status":result["status"],"aggregate":result["aggregate"],
                    "pairs":result["pairs"],"cautions":result["cautions"]}
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()
