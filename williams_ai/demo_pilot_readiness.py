"""One-pair, read-only Binance Futures Demo pilot gate, with durable lock.

This does NOT route an LLM decision to an order and does not grant trading
permission. A single explicitly supervised DEMO manual order may be admitted
by a separate interactive CLI only after a verified health check, operator
prompts and an atomic one-time reservation. Reservation is NEVER silently
cleared after restart, timeout, an entry rejection or ambiguous fills.

No live Binance endpoint, no secret storage, no production API keys.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
import http.client
import json
import os
import re
import sqlite3
import stat
import time
from typing import Any, Callable

from .demo_futures import DEMO_HOST, DemoSafetyError, DemoHttp

PAIR = "ETHUSDT"  # One pair, one campaign per pilot. All others blocked.
PUBLIC_ENDPOINTS = frozenset({"/fapi/v1/time", "/fapi/v1/exchangeInfo", "/fapi/v1/premiumIndex",
                             "/fapi/v1/klines"})
MAX_TIME_DRIFT_MS = 10_000
MAX_PREVIEW_NOTIONAL = Decimal("50")
MAX_PREVIEW_STOP_DISTANCE_RISK = Decimal("1")
MAX_PILOT_AGE_SECONDS = 30


def _dec(value: Any)->Decimal:
    if isinstance(value,bool):
        raise DemoSafetyError("Invalid numeric source value")
    try:
        d=Decimal(str(value))
    except (ValueError,InvalidOperation,TypeError) as e:
        raise DemoSafetyError("Invalid numeric source value") from e
    if not d.is_finite() or d<=0:
        raise DemoSafetyError("Nonpositive or nonfinite exchange value")
    return d


def _read_only_demo_json(path: str, args: dict[str,str]|None=None)->Any:
    """No caller-controlled URLs, auth or request methods. Only demo hostname."""
    if path not in PUBLIC_ENDPOINTS:
        raise DemoSafetyError("UNAPPROVED_DEMO_PUBLIC_ENDPOINT")
    from urllib.parse import urlencode
    query=urlencode(args or {})
    conn=http.client.HTTPSConnection(DEMO_HOST,timeout=8)
    try:
        conn.request("GET",path+("?" + query if query else ""),
                     headers={"Accept":"application/json",
                              "User-Agent":"Williams-AI-controlled-DEMO-health/1"})
        response=conn.getresponse()
        budget=4_000_000 if path=="/fapi/v1/exchangeInfo" else 131072
        raw=response.read(budget+1)
        if response.status!=200:
            # Status code alone is public, carries no signed URL or secrets.
            # Distinguish geographic HTTP 451 from malformed market data.
            raise DemoSafetyError("DEMO_PUBLIC_HTTP_"+str(int(response.status)))
        if len(raw)>budget:
            raise DemoSafetyError("DEMO_PUBLIC_MARKET_RESPONSE_TOO_LARGE")
        return json.loads(raw.decode("utf-8"))
    except (TimeoutError,OSError,ValueError) as exc:
        raise DemoSafetyError("DEMO_MARKET_FEED_UNTRUSTED_OR_UNAVAILABLE") from exc
    finally:
        conn.close()


@dataclass(frozen=True)
class PublicDemoPreflight:
    symbol: str
    mark_price: str
    tick_size: str
    lot_step: str
    min_qty: str
    max_qty: str
    checked_server_ms: int
    checked_local_ms: int
    venue: str = "BINANCE_FUTURES_DEMO"
    trade_action: None = None
    execution_allowed: bool = False


def check_public_demo(
    *, fetch:Callable[[str,dict[str,str]|None],Any]=_read_only_demo_json,
    now_ms:int|None=None,
    symbol:str=PAIR,
)->PublicDemoPreflight:
    """Time/mark/contract filters must all come from EXACT DEMO endpoint."""
    if symbol!=PAIR:
        raise DemoSafetyError("PILOT_SINGLE_PAIR_ONLY")
    local=int(time.time()*1000) if now_ms is None else now_ms
    if not isinstance(local,int) or isinstance(local,bool) or local<=0:
        raise DemoSafetyError("INVALID_UTC_CLOCK")
    t=fetch("/fapi/v1/time",None)
    if not isinstance(t,dict) or not isinstance(t.get("serverTime"),int) or isinstance(t["serverTime"],bool):
        raise DemoSafetyError("DEMO_SERVER_TIME_INVALID")
    observed=t["serverTime"]
    if abs(observed-local)>MAX_TIME_DRIFT_MS:
        raise DemoSafetyError("DEMO_SERVER_TIME_DRIFT")
    info=fetch("/fapi/v1/exchangeInfo",None)
    if (not isinstance(info,dict) or not isinstance(info.get("symbols"),list)
            or not info["symbols"]):
        raise DemoSafetyError("DEMO_SYMBOL_METADATA_UNTRUSTED")
    matching=[x for x in info["symbols"] if isinstance(x,dict) and x.get("symbol")==PAIR]
    if len(matching)!=1:
        raise DemoSafetyError("DEMO_SYMBOL_METADATA_UNTRUSTED")
    pair=matching[0]
    if (not isinstance(pair,dict) or pair.get("symbol")!=PAIR
            or pair.get("status")!="TRADING"
            or not isinstance(pair.get("filters"),list)):
        raise DemoSafetyError("DEMO_SYMBOL_NOT_TRADABLE")
    filters={q.get("filterType"):q for q in pair["filters"] if isinstance(q,dict)}
    if "PRICE_FILTER" not in filters or "MARKET_LOT_SIZE" not in filters:
        raise DemoSafetyError("DEMO_PRICE_AND_LOT_FILTERS_REQUIRED")
    tick=_dec(filters["PRICE_FILTER"].get("tickSize"))
    lot=filters["MARKET_LOT_SIZE"]
    step=_dec(lot.get("stepSize"))
    minq=_dec(lot.get("minQty"))
    maxq=_dec(lot.get("maxQty"))
    if maxq<minq or ((maxq-minq)/step)<0:
        raise DemoSafetyError("DEMO_LOT_LIMITS_INVALID")
    mark=fetch("/fapi/v1/premiumIndex",{"symbol":symbol})
    if (not isinstance(mark,dict) or mark.get("symbol")!=PAIR or
            "markPrice" not in mark or "time" not in mark or
            not isinstance(mark["time"],int) or isinstance(mark["time"],bool)
            or abs(mark["time"]-observed)>MAX_TIME_DRIFT_MS):
        raise DemoSafetyError("DEMO_MARK_FRESHNESS_UNVERIFIED")
    px=_dec(mark["markPrice"])
    return PublicDemoPreflight(
        symbol=PAIR,mark_price=str(px),tick_size=str(tick),
        lot_step=str(step),min_qty=str(minq),max_qty=str(maxq),
        checked_server_ms=observed,checked_local_ms=local,
    )


def validate_small_pilot(
    preflight:PublicDemoPreflight, *,
    side:str, quantity:str, stop:str,
    now_ms:int|None=None,
)->dict[str,Any]:
    """Previews cash risk, never auto-size or auto-enter from GGUF/H1."""
    now=int(time.time()*1000) if now_ms is None else now_ms
    if (not isinstance(preflight,PublicDemoPreflight)
            or preflight.execution_allowed is not False
            or preflight.venue!="BINANCE_FUTURES_DEMO"
            or preflight.symbol!=PAIR or side not in ("BUY","SELL")
            or isinstance(now,bool) or not isinstance(now,int)
            or not 0<=now-preflight.checked_server_ms<=MAX_PILOT_AGE_SECONDS*1000):
        raise DemoSafetyError("PILOT_PREFLIGHT_EXPIRED_OR_UNTRUSTED")
    price=_dec(preflight.mark_price)
    q=_dec(quantity)
    sl=_dec(stop)
    tick=_dec(preflight.tick_size)
    step=_dec(preflight.lot_step)
    if (q<_dec(preflight.min_qty) or q>_dec(preflight.max_qty)
            or (q/step)%1!=0 or (sl/tick)%1!=0):
        raise DemoSafetyError("PILOT_QUANTITY_OR_STOP_FILTER_REJECTED")
    if (side=="BUY" and sl>=price) or (side=="SELL" and sl<=price):
        raise DemoSafetyError("PILOT_PROTECTIVE_STOP_WRONG_SIDE")
    risk=abs(price-sl)*q
    notional=price*q
    if risk>MAX_PREVIEW_STOP_DISTANCE_RISK or notional>MAX_PREVIEW_NOTIONAL:
        raise DemoSafetyError("PILOT_RISK_LIMIT_REJECTED")
    if risk/notional<Decimal("0.001"):
        raise DemoSafetyError("PILOT_STOP_DISTANCE_TOO_SMALL")
    return {
        "mode":"SINGLE_PAIR_DEMO_PREVIEW_ONLY",
        "venue":"BINANCE_FUTURES_DEMO",
        "symbol":PAIR,"side":side,"quantity":str(q),"stop":str(sl),
        "marked_at_ms":preflight.checked_server_ms,
        "estimated_risk_demo_usdt":str(risk),
        "estimated_notional_demo_usdt":str(notional),
        "exchange_position_verified_flat":False,
        "stop_acknowledged":False,
        "manual_operator_approval_required":True,
        "automated_model_orders_supported":False,
        "execution_allowed":False,"trade_action":None,
    }


def verify_requoted_manual_pilot(preview:dict[str,Any], proposal:Any)->None:
    """A new DEMO quote can drift without silently increasing pilot exposure.

    The signed executor rechecks the market once more before submitting.
    This function only checks the supervised bridge's second, fresh quote:
    it is NOT an LLM execution authorization or a trade recommendation.
    """
    try:
        if (not isinstance(preview,dict)
                or preview.get("mode")!="SINGLE_PAIR_DEMO_PREVIEW_ONLY"
                or preview.get("venue")!="BINANCE_FUTURES_DEMO"
                or preview.get("execution_allowed") is not False
                or preview.get("trade_action","INVALID") is not None
                or preview.get("symbol")!=PAIR
                or proposal.symbol!=PAIR or proposal.side!=preview["side"]):
            raise DemoSafetyError("PILOT_PREVIEW_OR_PROPOSAL_UNTRUSTED")
        original_qty=_dec(preview["quantity"])
        original_stop=_dec(preview["stop"])
        original_notional=_dec(preview["estimated_notional_demo_usdt"])
        current_qty=_dec(proposal.quantity)
        current_stop=_dec(proposal.stop_price)
        current_notional=_dec(proposal.notional_usdt)
        current_risk=_dec(proposal.risk_usdt)
    except (AttributeError,KeyError,TypeError) as exc:
        raise DemoSafetyError("PILOT_PREVIEW_OR_PROPOSAL_UNTRUSTED") from exc
    if (current_qty!=original_qty or current_stop!=original_stop
            or current_notional>MAX_PREVIEW_NOTIONAL
            or current_risk>MAX_PREVIEW_STOP_DISTANCE_RISK):
        raise DemoSafetyError("PILOT_REQUOTE_CHANGED_INTENT_OR_EXCEEDED_LIMIT")
    old_mark=original_notional/original_qty
    new_mark=current_notional/current_qty
    if abs(new_mark/old_mark-1)>Decimal("0.0025"):
        raise DemoSafetyError("PILOT_MARK_MOVED_MORE_THAN_25_BPS")


class OneCampaignReservation:
    """An intentionally permanent per-PAIR one-shot lock until manual audit.

    Atomic SQLite BEGIN IMMEDIATE and UNIQUE symbol prevent duplicate pilots
    from two processes, including after crash. SQLite or disk failure is fatal.
    No API key, secret or account identifier is persisted. Non-secret
    deterministic client order IDs ARE persisted for exchange reconciliation.
    """
    def __init__(self,path:str|Path):
        self.path=Path(path).expanduser()
        if self.path.is_symlink() or not self.path.parent.is_dir():
            raise DemoSafetyError("PRIVATE_EXISTING_JOURNAL_DIRECTORY_REQUIRED")
        if self.path.exists():
            mode=self.path.stat().st_mode
            if (not stat.S_ISREG(mode) or mode & 0o077):
                raise DemoSafetyError("PILOT_JOURNAL_NOT_PRIVATE_REGULAR_FILE")
        else:
            descriptor=os.open(str(self.path),os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
            os.close(descriptor)
        self._setup()

    def _connection(self):
        conn=sqlite3.connect(str(self.path),timeout=3,isolation_level=None)
        conn.execute("PRAGMA busy_timeout = 3000")
        conn.execute("PRAGMA synchronous=FULL")
        return conn

    def _setup(self):
        with self._connection() as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS pilot_once(
                symbol TEXT PRIMARY KEY,
                direction TEXT NOT NULL,
                quantity TEXT NOT NULL,
                reserved_at_ms INTEGER NOT NULL,
                state TEXT NOT NULL CHECK(state = 'UNKNOWN_UNTIL_EXCHANGE_RECONCILED')
            )""")
            conn.execute("""CREATE TABLE IF NOT EXISTS pilot_client_ids(
                symbol TEXT PRIMARY KEY,
                expected_stop_client_algo_id TEXT NOT NULL,
                expected_entry_client_order_id TEXT NOT NULL,
                expected_stop_price TEXT NOT NULL,
                FOREIGN KEY (symbol) REFERENCES pilot_once(symbol)
            )""")

    def reserve_once(self,proposal:dict[str,Any],*,now_ms:int|None=None,
                     ticket:str|None=None)->None:
        if (not isinstance(proposal,dict) or proposal.get("symbol")!=PAIR
            or proposal.get("mode")!="SINGLE_PAIR_DEMO_PREVIEW_ONLY"
            or proposal.get("execution_allowed") is not False
            or proposal.get("trade_action","INVALID") is not None):
            raise DemoSafetyError("UNTRUSTED_MANUAL_PILOT_PROPOSAL")
        if ticket is not None and (not isinstance(ticket,str) or
                re.fullmatch(r"[0-9a-f]{32}",ticket) is None):
            raise DemoSafetyError("INVALID_DEMO_INTENT_TICKET")
        now=int(time.time()*1000) if now_ms is None else now_ms
        if isinstance(now,bool) or not isinstance(now,int) or now<=0:
            raise DemoSafetyError("INVALID_PILOT_RESERVATION_TIME")
        with self._connection() as conn:
            try:
                conn.execute("BEGIN IMMEDIATE")
                conn.execute(
                    "INSERT INTO pilot_once(symbol,direction,quantity,reserved_at_ms,state) VALUES(?,?,?,?,?)",
                    (PAIR,proposal["side"],proposal["quantity"],now,
                     "UNKNOWN_UNTIL_EXCHANGE_RECONCILED"),
                )
                if ticket is not None:
                    sealed_stop=str(_dec(proposal.get("stop")))
                    conn.execute(
                        "INSERT INTO pilot_client_ids VALUES(?,?,?,?)",
                        (PAIR,"WAISL"+ticket[:23],"WAIEN"+ticket[:23],
                         sealed_stop),
                    )
                conn.execute("COMMIT")
            except sqlite3.IntegrityError as exc:
                conn.execute("ROLLBACK")
                raise DemoSafetyError("PILOT_ALREADY_RESERVED_RECONCILE_BINANCE_MANUALLY") from exc
            except Exception:
                conn.execute("ROLLBACK")
                raise

    def bound_client_ids(self)->dict[str,str]|None:
        """Non-secret exchange IDs atomically sealed before any DEMO order."""
        with self._connection() as conn:
            rows=conn.execute(
                "SELECT expected_stop_client_algo_id,expected_entry_client_order_id,expected_stop_price "
                "FROM pilot_client_ids WHERE symbol=?", (PAIR,)
            ).fetchall()
        if len(rows)>1:
            raise DemoSafetyError("AMBIGUOUS_BOUND_DEMO_ORDER_IDS")
        if not rows:
            return None  # Old/legacy reservation: fail closed for attribution.
        stop,entry,trigger=rows[0]
        if (not isinstance(stop,str) or not isinstance(entry,str) or
                not re.fullmatch(r"WAISL[0-9a-f]{23}",stop) or
                not re.fullmatch(r"WAIEN[0-9a-f]{23}",entry) or
                stop[5:]!=entry[5:]):
            raise DemoSafetyError("CORRUPTED_BOUND_DEMO_ORDER_IDS")
        return {"stop_client_algo_id":stop,"entry_client_order_id":entry,
                "expected_stop_price":str(_dec(trigger))}

    def status(self)->dict[str,Any]:
        with self._connection() as conn:
            rows=conn.execute("SELECT symbol,direction,quantity,reserved_at_ms,state FROM pilot_once").fetchall()
        return {
            "mode":"DEMO_ONE_CAMPAIGN_DURABLE_RESERVATION",
            "reservations":[{"symbol":r[0],"side":r[1],"quantity":r[2],
                             "reserved_at_ms":r[3],"state":r[4]} for r in rows],
            "requires_manual_authenticated_exchange_reconciliation_to_clear":True,
            "execution_allowed":False,"trade_action":None,
        }
