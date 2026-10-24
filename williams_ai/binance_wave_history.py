"""Durable full-history, all-native-interval Binance USD-M FUTURES chart ingestion.

No account, orders, API keys or model-supplied URLs. MARKET DATA ONLY.
The caller must choose either PUBLIC_USDM or DEMO_USDM; never silently swap
venues, never stitch datasets. A partial current candle is never stored.
Binance futures max 1500 klines/request; intentionally use <=1000 to reduce
request weight. Paginate until up-to-date or explicit budget exhaustion,
resumable without destructive overwrite (with 1-bar overlap verification).

15 native Binance intervals: 1m/3m/5m/15m/30m/1h/2h/4h/6h/8h/12h/
1d/3d/1w/1M. Includes base/quote volume, number trades, taker buy volumes.
No infinite userless background loop, per invocation bounded API page budget.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime,timezone,timedelta
from decimal import Decimal,InvalidOperation
import calendar
import hashlib
import http.client
import json
from pathlib import Path
import os
import sqlite3
import stat
import time
from typing import Any, Callable
from urllib.parse import urlencode

VENUES = {
    "PUBLIC_USDM":"fapi.binance.com",
    "DEMO_USDM":"demo-fapi.binance.com",
}
INTERVAL_MS = {
    "1m":60_000,"3m":180_000,"5m":300_000,"15m":900_000,
    "30m":1_800_000,"1h":3_600_000,"2h":7_200_000,
    "4h":14_400_000,"6h":21_600_000,"8h":28_800_000,
    "12h":43_200_000,"1d":86_400_000,"3d":259_200_000,
    "1w":604_800_000,"1M":None,
}
ANALYSIS_TFS = {
    "1m":"M1","3m":"M3","5m":"M5","15m":"M15","30m":"M30",
    "1h":"H1","2h":"H2","4h":"H4","6h":"H6","8h":"H8",
    "12h":"H12","1d":"D1","3d":"D3","1w":"W1","1M":"MN1",
}
PAGE_MAX=1000
FINISH_SAFETY_MS=2000
MAX_CLOCK_SKEW_MS=300_000


class HistoryUnavailable(RuntimeError):
    """No fabricated data or implicit venue fallback when Binance is blocked."""


def _millis(dt:datetime)->int:
    return int(dt.timestamp()*1000)


def _month_open_ms(dt:datetime)->int:
    return _millis(datetime(dt.year,dt.month,1,tzinfo=timezone.utc))


def floor_open_ms(ms:int, interval:str)->int:
    if interval not in INTERVAL_MS or not isinstance(ms,int) or isinstance(ms,bool) or ms<0:
        raise ValueError("INVALID_BINANCE_INTERVAL_OR_UTC_MILLIS")
    if interval=="1M":
        return _month_open_ms(datetime.fromtimestamp(ms/1000,timezone.utc))
    if interval=="3d":
        # Verified against actual Binance USD-M Futures ETHUSDT 3d klines:
        # 2026-10-23 00:00 UTC is a legal open, despite its Unix-epoch
        # modulo(3 days) being ONE DAY, not zero. Anchor is epoch + 1d.
        offset=86_400_000
        step=INTERVAL_MS["3d"]
        return ((ms-offset)//step)*step+offset
    if interval=="1w":
        # Binance weekly UTC candles open on Monday.
        dt=datetime.fromtimestamp(ms/1000,timezone.utc)
        monday=datetime(dt.year,dt.month,dt.day,tzinfo=timezone.utc)-timedelta(days=dt.weekday())
        return _millis(monday)
    return ms//INTERVAL_MS[interval]*INTERVAL_MS[interval]


def next_open_ms(open_ms:int,interval:str)->int:
    if interval=="1M":
        dt=datetime.fromtimestamp(open_ms/1000,timezone.utc)
        y,m=dt.year,dt.month
        return _month_open_ms(datetime(y+1,1,1,tzinfo=timezone.utc) if m==12
                              else datetime(y,m+1,1,tzinfo=timezone.utc))
    if interval not in INTERVAL_MS:
        raise ValueError("INVALID_BINANCE_INTERVAL")
    return open_ms+INTERVAL_MS[interval]


def _decimal_string(value:Any, *,allow_zero:bool=False)->str:
    if isinstance(value,bool):
        raise HistoryUnavailable("INVALID_NUMERIC_KLINE")
    try:
        n=Decimal(str(value))
    except (TypeError,ValueError,InvalidOperation) as exc:
        raise HistoryUnavailable("INVALID_NUMERIC_KLINE") from exc
    if not n.is_finite() or n<0 or (n==0 and not allow_zero):
        raise HistoryUnavailable("INVALID_NUMERIC_KLINE")
    return format(n,"f")


def _int(value:Any)->int:
    if not isinstance(value,int) or isinstance(value,bool) or value<0:
        raise HistoryUnavailable("INVALID_KLINE_INTEGER")
    return value


def parse_closed_candle(raw:Any,interval:str, *,server_ms:int)->dict[str,Any]|None:
    if not isinstance(raw,list) or len(raw)<11:
        raise HistoryUnavailable("INVALID_FUTURES_KLINE_SHAPE")
    op,cl=_int(raw[0]),_int(raw[6])
    if op!=floor_open_ms(op,interval) or cl!=next_open_ms(op,interval)-1:
        raise HistoryUnavailable("INVALID_FUTURES_KLINE_BOUNDARIES")
    if cl>=server_ms-FINISH_SAFETY_MS:
        return None
    o,h,l,c=[_decimal_string(raw[i]) for i in (1,2,3,4)]
    if Decimal(l)>min(Decimal(o),Decimal(c)) or Decimal(h)<max(Decimal(o),Decimal(c)):
        raise HistoryUnavailable("OHLC_OUTSIDE_HIGH_LOW")
    if Decimal(h)<Decimal(l):
        raise HistoryUnavailable("CANDLE_HIGH_BELOW_LOW")
    base=_decimal_string(raw[5],allow_zero=True)
    quote=_decimal_string(raw[7],allow_zero=True)
    trades=_int(raw[8])
    taker_base=_decimal_string(raw[9],allow_zero=True)
    taker_quote=_decimal_string(raw[10],allow_zero=True)
    if Decimal(taker_base)>Decimal(base) or Decimal(taker_quote)>Decimal(quote):
        raise HistoryUnavailable("TAKER_VOLUME_EXCEEDS_TOTAL_VOLUME")
    return {
        "open_ms":op,"close_ms":cl,"open":o,"high":h,"low":l,
        "close":c,"volume":base,"quote_volume":quote,"trades":trades,
        "taker_buy_base_volume":taker_base,"taker_buy_quote_volume":taker_quote,
    }


def _read_market(venue:str,path:str,params:dict[str,str]|None=None)->Any:
    """Only pinned public GET routes, no redirects or caller-supplied hostname."""
    if venue not in VENUES or path not in {
        "/fapi/v1/time","/fapi/v1/exchangeInfo","/fapi/v1/klines",
    }:
        raise HistoryUnavailable("MARKET_ENDPOINT_NOT_ALLOWLISTED")
    connection=http.client.HTTPSConnection(VENUES[venue],timeout=13)
    try:
        q=urlencode(params or {})
        connection.request("GET",path+("?" + q if q else ""),
                           headers={"Accept":"application/json",
                                    "User-Agent":"WilliamsAI-public-closed-market/1"})
        r=connection.getresponse()
        budget=4_000_000 if path=="/fapi/v1/exchangeInfo" else 2_500_000
        raw=r.read(budget+1)
        if r.status!=200:
            # Explicitly expose HTTP451/429 rather than silently switching
            # to Spot, an unsafe live-production trading venue, or mock bars.
            raise HistoryUnavailable(f"{venue}_PUBLIC_HTTP_{r.status}")
        if len(raw)>budget:
            raise HistoryUnavailable("PUBLIC_RESPONSE_EXCEEDS_BOUNDS")
        return json.loads(raw.decode("utf8"))
    except (OSError,TimeoutError,ValueError) as exc:
        raise HistoryUnavailable("MARKET_NETWORK_UNAVAILABLE_OR_INVALID_JSON") from exc
    finally:
        connection.close()


def _market_snapshot(venue:str,symbol:str,fetch:Callable[...,Any],
                     *,now_ms:int|None=None)->tuple[int,str]:
    if venue not in VENUES or not isinstance(symbol,str) or (
            len(symbol)>24 or not symbol.endswith("USDT") or
            not all(c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789" for c in symbol)):
        raise HistoryUnavailable("UNKNOWN_FUTURES_SOURCE_OR_SYMBOL")
    local_ms=int(time.time()*1000) if now_ms is None else now_ms
    if not isinstance(local_ms,int) or isinstance(local_ms,bool) or local_ms<=0:
        raise ValueError("INVALID_UTC_MILLIS")
    t=fetch(venue,"/fapi/v1/time",None)
    if not isinstance(t,dict) or not isinstance(t.get("serverTime"),int) or (
            isinstance(t["serverTime"],bool) or
            abs(t["serverTime"]-local_ms)>MAX_CLOCK_SKEW_MS):
        raise HistoryUnavailable("FUTURES_PUBLIC_SERVER_CLOCK_UNTRUSTED")
    info=fetch(venue,"/fapi/v1/exchangeInfo",None)
    pairs=info.get("symbols") if isinstance(info,dict) else None
    if not isinstance(pairs,list):
        raise HistoryUnavailable("FUTURES_CONTRACT_NOT_VERIFIED")
    matching=[x for x in pairs if isinstance(x,dict) and x.get("symbol")==symbol]
    if len(matching)!=1:
        raise HistoryUnavailable("FUTURES_CONTRACT_NOT_VERIFIED")
    pair=matching[0]
    if not isinstance(pair,dict) or pair.get("symbol")!=symbol or (
            pair.get("status")!="TRADING" or
            not isinstance(pair.get("filters"),list)):
        raise HistoryUnavailable("FUTURES_CONTRACT_NOT_TRADING")
    ticks=[x for x in pair["filters"] if isinstance(x,dict) and
           x.get("filterType")=="PRICE_FILTER"]
    if len(ticks)!=1:
        raise HistoryUnavailable("FUTURES_PRICE_FILTER_MISSING")
    tick=_decimal_string(ticks[0].get("tickSize"))
    return t["serverTime"],tick


class DurableWaveHistory:
    """Local private DB. Each stream is segregated by venue, symbol and TF."""

    def __init__(self,path:Path|str):
        self.path=Path(path).expanduser()
        if self.path.is_symlink() or not self.path.parent.is_dir():
            raise HistoryUnavailable("HISTORY_REQUIRES_EXISTING_PRIVATE_DIRECTORY")
        if self.path.exists():
            if not self.path.is_file() or self.path.stat().st_mode&0o077:
                raise HistoryUnavailable("HISTORY_DATABASE_IS_NOT_PRIVATE")
        else:
            fd=os.open(str(self.path),os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
            os.close(fd)
        with self._connect() as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS candles(
                venue TEXT NOT NULL, symbol TEXT NOT NULL, interval TEXT NOT NULL,
                open_ms INTEGER NOT NULL, close_ms INTEGER NOT NULL,
                open TEXT NOT NULL, high TEXT NOT NULL, low TEXT NOT NULL,
                close TEXT NOT NULL, volume TEXT NOT NULL,
                quote_volume TEXT NOT NULL, trades INTEGER NOT NULL,
                taker_buy_base_volume TEXT NOT NULL,
                taker_buy_quote_volume TEXT NOT NULL,
                PRIMARY KEY(venue,symbol,interval,open_ms))""")
            conn.execute("""CREATE TABLE IF NOT EXISTS streams(
                venue TEXT NOT NULL, symbol TEXT NOT NULL, interval TEXT NOT NULL,
                requested_since_ms INTEGER NOT NULL, tick_size TEXT NOT NULL,
                last_checked_server_ms INTEGER NOT NULL, PRIMARY KEY(venue,symbol,interval))""")

    def _connect(self):
        conn=sqlite3.connect(str(self.path),timeout=6,isolation_level=None)
        conn.execute("PRAGMA synchronous=FULL")
        conn.execute("PRAGMA busy_timeout=6000")
        return conn

    def _last(self,venue:str,symbol:str,interval:str)->int|None:
        with self._connect() as conn:
            row=conn.execute(
                "SELECT MAX(open_ms) FROM candles WHERE venue=? AND symbol=? AND interval=?",
                (venue,symbol,interval)).fetchone()
            return row[0]

    def _write_page(self,venue:str,symbol:str,interval:str,
                    rows:list[dict[str,Any]],since:int,tick:str,server_ms:int)->int:
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                row=conn.execute(
                    "SELECT requested_since_ms FROM streams WHERE venue=? AND symbol=? AND interval=?",
                    (venue,symbol,interval)).fetchone()
                if row is not None and since<row[0]:
                    raise HistoryUnavailable("EARLIER_HISTORY_REQUIRES_NEW_DATABASE")
                if row is None:
                    conn.execute(
                        "INSERT INTO streams VALUES(?,?,?,?,?,?)",
                        (venue,symbol,interval,since,tick,server_ms))
                else:
                    conn.execute("UPDATE streams SET last_checked_server_ms=?,tick_size=? "
                                 "WHERE venue=? AND symbol=? AND interval=?",
                                 (server_ms,tick,venue,symbol,interval))
                inserted=0
                for bar in rows:
                    previous=conn.execute(
                        "SELECT open,high,low,close,volume,quote_volume,trades,"
                        "taker_buy_base_volume,taker_buy_quote_volume,close_ms "
                        "FROM candles WHERE venue=? AND symbol=? AND interval=? AND open_ms=?",
                        (venue,symbol,interval,bar["open_ms"])).fetchone()
                    content=tuple(bar[k] for k in (
                        "open","high","low","close","volume","quote_volume",
                        "trades","taker_buy_base_volume","taker_buy_quote_volume","close_ms"))
                    if previous is not None:
                        if previous!=content:
                            raise HistoryUnavailable("BINANCE_FINAL_CANDLE_CHANGED_QUARANTINE")
                        continue
                    prior=conn.execute(
                        "SELECT MAX(open_ms) FROM candles WHERE venue=? AND symbol=? AND interval=?",
                        (venue,symbol,interval)).fetchone()[0]
                    if prior is not None and bar["open_ms"]!=next_open_ms(prior,interval):
                        raise HistoryUnavailable("HISTORY_GAP_PREVENTS_FALSE_WAVE_CONTEXT")
                    conn.execute(
                        "INSERT INTO candles VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (venue,symbol,interval,bar["open_ms"],bar["close_ms"],bar["open"],
                         bar["high"],bar["low"],bar["close"],bar["volume"],
                         bar["quote_volume"],bar["trades"],
                         bar["taker_buy_base_volume"],bar["taker_buy_quote_volume"]))
                    inserted+=1
                conn.execute("COMMIT")
                return inserted
            except Exception:
                conn.execute("ROLLBACK")
                raise

    def sync(self,*,venue:str,symbol:str,interval:str,
             since_ms:int=0,max_pages:int=20,page_size:int=1000,
             fetch:Callable[...,Any]=_read_market,now_ms:int|None=None,
             sleep_seconds:float=0.15)->dict[str,Any]:
        """One bounded invocation. On repeat, resumes from last saved bar.

        Initial since=0 requests the earliest obtainable public futures data.
        If the DEMO source has short history it remains explicitly DEMO-only.
        """
        if (interval not in INTERVAL_MS or
                not isinstance(since_ms,int) or isinstance(since_ms,bool) or since_ms<0
                or not isinstance(max_pages,int) or isinstance(max_pages,bool)
                or not 1<=max_pages<=5000 or not isinstance(page_size,int)
                or isinstance(page_size,bool) or not 20<=page_size<=PAGE_MAX
                or not isinstance(sleep_seconds,(int,float))
                or not 0<=sleep_seconds<=3):
            raise ValueError("INVALID_PUBLIC_KLINE_HISTORY_BUDGET")
        remote_ms,tick=_market_snapshot(venue,symbol,fetch,now_ms=now_ms)
        cutoff=remote_ms-FINISH_SAFETY_MS
        # startTime=0 asks from the earliest publicly available candle.
        # UTC weekly floors Jan 1 1970 to a NEGATIVE Monday, which Binance
        # must never receive in a public query.
        since=0 if since_ms==0 else floor_open_ms(since_ms,interval)
        with self._connect() as conn:
            existing=conn.execute(
                "SELECT requested_since_ms FROM streams WHERE venue=? AND symbol=? AND interval=?",
                (venue,symbol,interval)).fetchone()
        if existing is not None and since<existing[0]:
            raise HistoryUnavailable("EARLIER_HISTORY_REQUIRES_NEW_DATABASE")
        tail=self._last(venue,symbol,interval)
        cursor=tail if tail is not None else since
        pages=0
        total_inserted=0
        state="PAGE_BUDGET_EXHAUSTED"
        while pages<max_pages:
            if cursor>=cutoff:
                state="AT_LATEST_CLOSED_BAR"
                break
            raw=fetch(venue,"/fapi/v1/klines",{
                "symbol":symbol,"interval":interval,
                "startTime":str(cursor),"endTime":str(cutoff),
                "limit":str(page_size),
            })
            pages+=1
            if not isinstance(raw,list) or len(raw)>page_size:
                raise HistoryUnavailable("INVALID_PAGINATED_BINANCE_KLINES")
            if not raw:
                if tail is None and total_inserted==0:
                    raise HistoryUnavailable("NO_HISTORY_FROM_REQUESTED_OR_LISTING_START")
                state="AT_LATEST_CLOSED_BAR"
                break
            batch=[]
            prev=None
            unfinished=False
            for item in raw:
                candle=parse_closed_candle(item,interval,server_ms=remote_ms)
                if candle is None:
                    unfinished=True
                    break
                t=candle["open_ms"]
                if t<cursor or t<since:
                    raise HistoryUnavailable("HISTORY_PAGE_REWINDS_CURSOR")
                if prev is not None and t!=next_open_ms(prev,interval):
                    raise HistoryUnavailable("HISTORY_PAGE_GAPPED")
                prev=t
                batch.append(candle)
            if not batch:
                if unfinished and tail is not None:
                    state="AT_LATEST_CLOSED_BAR"
                    break
                raise HistoryUnavailable("NO_CLOSED_CANDLES_IN_FUTURES_PAGE")
            # A legitimate overlap of exactly one last cached candle is
            # rechecked byte-for-byte. No blind revision of closed history.
            inserted=self._write_page(venue,symbol,interval,batch,since,tick,remote_ms)
            total_inserted+=inserted
            tail=self._last(venue,symbol,interval)
            if tail is None or tail<cursor:
                raise HistoryUnavailable("BINANCE_HISTORY_FAILED_TO_ADVANCE")
            next_cursor=next_open_ms(tail,interval)
            if unfinished or tail>=floor_open_ms(cutoff,interval):
                state="AT_LATEST_CLOSED_BAR"
                break
            if next_cursor<=cursor or (len(batch)==1 and inserted==0):
                state="AT_LATEST_CLOSED_BAR"
                break
            cursor=next_cursor
            if pages<max_pages and sleep_seconds:
                time.sleep(sleep_seconds)
        stats=self.status(venue,symbol,interval)
        return {
            **stats,
            "checked_server_ms":remote_ms,"price_tick_size":tick,
            "pages_requested":pages,"new_closed_bars_added":total_inserted,
            "sync_state":state,
            # An end cursor reaching latest available bars is not
            # independent evidence that DEMO/public APIs preserve ALL
            # historical candles back to the pair's listing date.
            "up_to_snapshot_caught_up":state=="AT_LATEST_CLOSED_BAR",
            "complete_download_claimed":False,
            "exchange_listing_date_completeness_unverified":True,
            "venue_switched_automatically":False,
            "trade_action":None,"execution_allowed":False,
        }

    def status(self,venue:str,symbol:str,interval:str)->dict[str,Any]:
        with self._connect() as conn:
            row=conn.execute(
                "SELECT COUNT(*),MIN(open_ms),MAX(open_ms),MAX(close_ms) "
                "FROM candles WHERE venue=? AND symbol=? AND interval=?",
                (venue,symbol,interval)).fetchone()
            meta=conn.execute(
                "SELECT requested_since_ms,tick_size,last_checked_server_ms FROM streams "
                "WHERE venue=? AND symbol=? AND interval=?",
                (venue,symbol,interval)).fetchone()
        return {
            "venue":venue,"symbol":symbol,"interval":interval,
            "stored_closed_bars":row[0],"first_open_ms":row[1],
            "last_open_ms":row[2],"last_closed_ms":row[3],
            "initial_requested_since_ms":meta[0] if meta else None,
            "tick_size":meta[1] if meta else None,
            "last_checked_server_ms":meta[2] if meta else None,
            "trade_action":None,"execution_allowed":False,
        }

    def recent(self,venue:str,symbol:str,interval:str,
               limit:int=1200,*,as_of_ms:int|None=None)->list[dict[str,Any]]:
        """Return an immutable closed-market prefix. Never hand a later M1/M5
        candle to an H1 decision that occurred before that candle closed.
        """
        if (interval not in INTERVAL_MS or not isinstance(limit,int)
                or isinstance(limit,bool) or not 1<=limit<=50000
                or (as_of_ms is not None and (
                    not isinstance(as_of_ms,int) or isinstance(as_of_ms,bool)
                    or as_of_ms<=0))):
            raise ValueError("INVALID_ANALYSIS_WINDOW_OR_CUTOFF")
        with self._connect() as conn:
            query=(
                "SELECT open_ms,close_ms,open,high,low,close,volume,quote_volume,"
                "trades,taker_buy_base_volume,taker_buy_quote_volume "
                "FROM candles WHERE venue=? AND symbol=? AND interval=? "
            )
            if as_of_ms is None:
                rows=conn.execute(query+"ORDER BY open_ms DESC LIMIT ?",
                                  (venue,symbol,interval,limit)).fetchall()
            else:
                rows=conn.execute(
                    query+"AND close_ms<=? ORDER BY open_ms DESC LIMIT ?",
                    (venue,symbol,interval,as_of_ms,limit)).fetchall()
        rows.reverse()
        out=[]
        prev=None
        for r in rows:
            if prev is not None and r[0]!=next_open_ms(prev,interval):
                raise HistoryUnavailable("CACHED_HISTORY_HAS_GAPS")
            prev=r[0]
            out.append({
                "open_ms":r[0],"close_ms":r[1],
                "close_time":datetime.fromtimestamp(r[1]/1000,timezone.utc).isoformat(),
                "open":float(r[2]),"high":float(r[3]),"low":float(r[4]),
                "close":float(r[5]),"volume":float(r[6]),
                "quote_volume":float(r[7]),"number_of_trades":r[8],
                "taker_buy_base_volume":float(r[9]),
                "taker_buy_quote_volume":float(r[10]),"closed":True,
            })
        return out
