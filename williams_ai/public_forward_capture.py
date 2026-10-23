"""Capture only *new, finalized* public Binance H1 bars for prospective AI tests.

No private user data, credentials, orders, API keys, authenticated account
endpoints, or automatic venue fallback. USDT-M futures and SPOT have distinct
market semantics and cannot be mixed in a performance report.

Uses official REST market-data endpoints, but fails CLOSED if freshness,
clock, venue, history continuity, candle finality or exchange price filters
cannot be established. Do not claim a real-money trading backtest from this.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .chart_reader import validate_closed_bars

VENUES = {
    "FUTURES": ("https://fapi.binance.com", "/fapi/v1"),
    "SPOT": ("https://data-api.binance.vision", "/api/v3"),
}
SYMBOLS = {"BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT"}
MAX_CLOCK_DRIFT_MS = 5 * 60_000
HOUR_MS = 3_600_000


def _number(x: Any, field: str, *, strictly_positive: bool = True) -> float:
    if isinstance(x, bool):
        raise ValueError("Invalid " + field)
    try:
        val=float(x)
    except (ValueError, TypeError) as exc:
        raise ValueError("Invalid " + field) from exc
    if not math.isfinite(val) or (val<=0 if strictly_positive else val<0):
        raise ValueError("Invalid " + field)
    return val


def _public_get(url: str) -> Any:
    """Fixed known URLs only; never accept arbitrary user-supplied hosts."""
    if not any(url.startswith(base + path + "/") for base,path in VENUES.values()):
        raise ValueError("Non-public or unapproved Binance host forbidden")
    req=Request(url,headers={"User-Agent":"Williams-AI-read-only-public-H1/1.0",
                             "Accept":"application/json"})
    with urlopen(req,timeout=15) as response:
        data=response.read(3_000_001)
    if len(data)>3_000_000:
        raise ValueError("Market data response exceeds bound")
    return json.loads(data)


def capture_latest_closed_h1(
    symbol: str, *, venue: str, fetch_json: Callable[[str], Any] = _public_get,
    clock_ms: int | None = None, limit: int = 500,
) -> dict[str, Any]:
    """A self-contained H1 observation at the *latest completed hour*."""
    if symbol not in SYMBOLS or venue not in VENUES:
        raise ValueError("Explicit approved USDT symbol and market venue required")
    if not callable(fetch_json) or not isinstance(limit,int) or isinstance(limit,bool) or (
        not 260 <= limit <= 750
    ):
        raise ValueError("Invalid public fetcher or bar count")
    host,api=VENUES[venue]
    clock=int(datetime.now(timezone.utc).timestamp()*1000) if clock_ms is None else clock_ms
    if not isinstance(clock,int) or isinstance(clock,bool) or clock<=0:
        raise ValueError("UTC wall-clock milliseconds required")
    server=fetch_json(host+api+"/time")
    if not isinstance(server,dict):
        raise ValueError("Missing Binance public time response")
    remote_ms=server.get("serverTime")
    if not isinstance(remote_ms,int) or isinstance(remote_ms,bool) or (
        abs(remote_ms-clock)>MAX_CLOCK_DRIFT_MS
    ):
        raise ValueError("Binance time deviates from observed UTC clock")
    rules=fetch_json(host+api+"/exchangeInfo?"+urlencode({"symbol":symbol}))
    if (not isinstance(rules,dict) or not isinstance(rules.get("symbols"),list) or
            len(rules["symbols"])!=1):
        raise ValueError("Exchange contract filters not verified")
    pair=rules["symbols"][0]
    if (not isinstance(pair,dict) or pair.get("symbol")!=symbol or
            pair.get("status")!="TRADING" or not isinstance(pair.get("filters"),list)):
        raise ValueError("Unexpected symbol, market status or rules")
    price_filters=[f for f in pair["filters"]
                   if isinstance(f,dict) and f.get("filterType")=="PRICE_FILTER"]
    if len(price_filters)!=1:
        raise ValueError("Missing verified exchange PRICE_FILTER")
    tick=_number(price_filters[0].get("tickSize"),"tick size")
    response=fetch_json(host+api+"/klines?"+urlencode({
        "symbol":symbol,"interval":"1h","limit":limit,
    }))
    if not isinstance(response,list) or len(response)<260 or len(response)>limit:
        raise ValueError("Insufficient recent H1 public market data")
    bars=[]
    last_open=None
    last_close=None
    for entry in response:
        if not isinstance(entry,list) or len(entry)<7:
            raise ValueError("Unexpected Binance H1 row")
        open_ms,close_ms=entry[0],entry[6]
        if (not isinstance(open_ms,int) or isinstance(open_ms,bool) or
                not isinstance(close_ms,int) or isinstance(close_ms,bool) or
                open_ms%HOUR_MS!=0 or close_ms-open_ms!=HOUR_MS-1):
            raise ValueError("Misaligned or malformed hourly candle")
        if last_open is not None and open_ms-last_open!=HOUR_MS:
            raise ValueError("Missing or unordered public market candles")
        if last_close is not None and close_ms-last_close!=HOUR_MS:
            raise ValueError("Non-hourly timestamp history")
        last_open,last_close=open_ms,close_ms
        # Binance also serves the current unfinished hour. DO NOT export it
        # to the model, even if price/volume fields are already populated.
        if close_ms >= remote_ms-5_000:
            continue
        op,hi,lo,cl=(_number(entry[k],k) for k in (1,2,3,4))
        vol=_number(entry[5],"volume",strictly_positive=False)
        if not lo<=min(op,cl)<=max(op,cl)<=hi:
            raise ValueError("Invalid Binance OHLC geometry")
        at=datetime.fromtimestamp(close_ms/1000,timezone.utc).isoformat()
        bars.append({"close_time":at,"open":op,"high":hi,"low":lo,
                     "close":cl,"volume":vol,"closed":True})
    if len(bars)<220:
        raise ValueError("Not enough fully CLOSED H1 bars")
    stamp=datetime.fromisoformat(bars[-1]["close_time"]).astimezone(timezone.utc)
    age=(remote_ms/1000)-stamp.timestamp()
    if not (5<=age<7200):
        raise ValueError("LATEST_H1_NOT_FRESH_ENOUGH_FOR_PROSPECTIVE_TEST")
    if any(datetime.fromisoformat(b["close_time"]).astimezone(timezone.utc)-
           datetime.fromisoformat(a["close_time"]).astimezone(timezone.utc)
           !=timedelta(hours=1) for a,b in zip(bars,bars[1:])):
        raise ValueError("H1 dataset is not one continuous hourly series")
    candles,gaps=validate_closed_bars(bars,"H1",bars[-1]["close_time"])
    if gaps or len(candles)!=len(bars):
        raise ValueError("Candle history failed strict causal verification")
    market={"symbol":symbol,"tick_size":tick,"bars":bars}
    digest=hashlib.sha256(json.dumps(market,sort_keys=True,separators=(",",":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()
    return {
        "market":market,
        "source":{
            "venue":venue,
            "source_base":host+api,
            "as_of":bars[-1]["close_time"],
            "observed_server_time":datetime.fromtimestamp(remote_ms/1000,timezone.utc).isoformat(),
            "fully_closed_h1":len(bars),
            "market_sha256":digest,
            "clock_checked":True,
            "no_future_bar_in_market":True,
            "execution_allowed":False,
            "trade_action":None,
            "limitations":(
                "Spot and futures are different; venue is an immutable part of study identity."
                if venue=="SPOT" else
                "Futures OHLC is not an exchange fill or protective stop ACK."
            ),
        },
    }


def main() -> None:
    ap=argparse.ArgumentParser(description="Capture fresh PUBLIC H1 candles without credentials")
    ap.add_argument("--symbol",choices=sorted(SYMBOLS),required=True)
    ap.add_argument("--venue",choices=sorted(VENUES),required=True)
    ap.add_argument("--market-file",type=Path,required=True)
    ap.add_argument("--stamp-file",type=Path,required=True)
    args=ap.parse_args()
    from .prospective_cli import _write_once_private
    out=capture_latest_closed_h1(args.symbol,venue=args.venue)
    _write_once_private(args.market_file,out["market"])
    _write_once_private(args.stamp_file,out["source"])
    print("PUBLIC_H1_CAPTURE",out["source"]["venue"],args.symbol,
          out["source"]["as_of"],out["source"]["market_sha256"],"NO_TRADE",flush=True)


if __name__=="__main__":
    main()
