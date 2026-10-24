"""Strict supervised single-ETHUSDT Binance Futures DEMO pilot.

Examples on the USER'S computer, not GitHub runners:
 python -m williams_ai.demo_pilot_cli check --side BUY --quantity 0.05 --stop 1900
 python -m williams_ai.demo_pilot_cli status --journal /private/demo_once.sqlite3
 python -m williams_ai.demo_pilot_cli supervised-send --side BUY --quantity 0.05 \
   --stop 1900 --journal /private/demo_once.sqlite3

The illustrative stop/quantity must be recalculated for current DEMO mark.
This client NEVER accepts a model forecast, JSON-generated trading command,
external endpoint or production keys. Check/preview has ZERO order capability.
The one-shot database remains LOCKED even after crash / network timeout.
Do not manually remove it while any exchange order could still exist.
"""
from __future__ import annotations
import argparse
from getpass import getpass
import json
from pathlib import Path
import sys

from .demo_futures import DemoFutures,DemoHttp,DemoSafetyError
from .demo_pilot_readiness import (
    PAIR,OneCampaignReservation,check_public_demo,validate_small_pilot,
)


def main()->None:
    p=argparse.ArgumentParser(description="Single protected ETHUSDT DEMO pilot, never real trading")
    sub=p.add_subparsers(dest="mode",required=True)
    check=sub.add_parser("check",help="PUBLIC-DEMO read-only + strict limits, no API keys")
    send=sub.add_parser("supervised-send",help="Only with DEMO keys, TTY, two approvals and permanent SQLite lock")
    status=sub.add_parser("status",help="Show private one-campaign lock state, no trading")
    for cmd in (check,send):
        cmd.add_argument("--side",required=True,choices=("BUY","SELL"))
        cmd.add_argument("--quantity",required=True)
        cmd.add_argument("--stop",required=True)
        cmd.add_argument("--symbol",default=PAIR,choices=(PAIR,))
    for cmd in (send,status):
        cmd.add_argument("--journal",type=Path,required=True)
    args=p.parse_args()
    if args.mode=="status":
        print(json.dumps(OneCampaignReservation(args.journal).status(),
                         ensure_ascii=False,indent=2))
        return
    if args.mode=="supervised-send" and not sys.stdin.isatty():
        p.error("TTY required for operator DEMO-only credentials and approvals")
    public=check_public_demo(symbol=args.symbol)
    preview=validate_small_pilot(
        public,side=args.side,quantity=args.quantity,stop=args.stop,
    )
    print(json.dumps(preview,ensure_ascii=False,indent=2))
    if args.mode=="check":
        print("DEMO_PUBLIC_PREFLIGHT_ONLY. No keys read. No orders created.")
        return

    journal=OneCampaignReservation(args.journal)
    if journal.status()["reservations"]:
        raise DemoSafetyError("SINGLE_PILOT_ALREADY_RESERVED: reconcile exchange manually")
    print("MANUAL DEMO PILOT ONLY; no autonomous Williams AI trading.")
    print("The journal is a permanent single-use lock until MANUAL reconciliation.")
    print("Do NOT use Binance production API credentials.")
    key=getpass("Your separate Binance DEMO API key (hidden): ")
    secret=getpass("Your separate Binance DEMO API secret (hidden): ")
    first=input("Confirm this is a dedicated Binance DEMO account: DEMO_ACCOUNT_ONLY: ").strip()
    if first!="DEMO_ACCOUNT_ONLY":
        raise DemoSafetyError("DEMO account was not explicitly approved")
    client=DemoHttp(key,secret)
    engine=DemoFutures(client,enabled=True,
                       acknowledgement=first,max_notional="50",max_risk="1")
    # The exchange-side preconditions and reference mark are read again.
    # No order can arise from propose(). Model output has no authority.
    proposed=engine.propose(args.symbol,args.side,args.quantity,args.stop)
    if (proposed.quantity!=preview["quantity"] or
        proposed.stop_price!=preview["stop"] or
        proposed.notional_usdt!=preview["estimated_notional_demo_usdt"]
            and float(proposed.notional_usdt)>50):
        raise DemoSafetyError("MARKET_MOVED_AFTER_PREFLIGHT")
    print(json.dumps({
        "ticket_expires_at":proposed.expires_at,
        "risk_demo_usdt":proposed.risk_usdt,
        "notional_demo_usdt":proposed.notional_usdt,
        "protective_sl_acknowledged":False,
        "orders_sent":False,"market":"BINANCE_FUTURES_DEMO",
    },ensure_ascii=False,indent=2))
    second=input("Submit ONE manual DEMO entry, with pre-entry SL attempt? PLACE_DEMO_ORDER: ").strip()
    if second!="PLACE_DEMO_ORDER":
        print("Cancelled. No exchange orders. Reservation not consumed.")
        return
    # PERMANENT SQLite BEGIN IMMEDIATE reservation is committed BEFORE ANY
    # order submission. Another process/restart cannot reset unknown effects.
    journal.reserve_once(preview)
    try:
        confirmation=engine.execute(proposed.ticket,confirmation=second)
        print(json.dumps(confirmation,ensure_ascii=False,indent=2))
        print("PILOT_LOCK_REMAINS: check actual Binance DEMO positions and stop manually.")
    except Exception:
        # Don't leak signed URLs, private keys, exchange responses or verbose
        # model text. Never automatically unlock or retry market orders.
        print("PILOT_LOCKED_UNKNOWN: inspect authenticated Binance DEMO position/algo orders; NO RETRY",file=sys.stderr)
        raise


if __name__=="__main__":
    main()
