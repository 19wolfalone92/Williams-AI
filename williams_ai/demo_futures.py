"""Explicitly gated Binance USD-M Futures DEMO trading, not production.

The LLM NEVER calls this module. Only an approved structured user intent
may enter execute(); no arbitrary URLs, methods, order types or symbols.

Safe workflow: verify one-way, flat position and demo mark; pre-place and
verify close-all STOP_MARKET algo; submit single MARKET entry; re-check stop.
If protection fails after entry, attempt REDUCE_ONLY flatten and lock trading.
Production Binance hosts cannot be supplied via constructor or environment.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import hmac
import http.client
import json
import re
import threading
import time
import uuid
from urllib.parse import urlencode
from typing import Any, Callable

DEMO_HOST = "demo-fapi.binance.com"
ALLOWED_SYMBOLS = frozenset({"BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"})
_ALLOWED = {
    ("GET", "/fapi/v1/premiumIndex"),
    ("GET", "/fapi/v1/exchangeInfo"),
    ("GET", "/fapi/v1/positionSide/dual"),
    ("GET", "/fapi/v3/positionRisk"),
    ("GET", "/fapi/v1/algoOrder"),
    ("GET", "/fapi/v1/openAlgoOrders"),
    ("GET", "/fapi/v1/order"),
    ("GET", "/fapi/v1/openOrders"),
    ("POST", "/fapi/v1/algoOrder"),
    ("POST", "/fapi/v1/order"),
    ("DELETE", "/fapi/v1/algoOrder"),
}
FINAL_BAD_ALGO = {"CANCELED", "EXPIRED", "FAILED", "REJECTED"}
READONLY_STATUSES = {"NEW", "TRIGGERED", "FINISHED"}


class DemoSafetyError(RuntimeError):
    """Safety invariant violated. Do not try a second execution blindly."""


@dataclass(frozen=True)
class TradeProposal:
    ticket: str
    symbol: str
    side: str
    quantity: str
    stop_price: str
    expires_at: int
    risk_usdt: str
    notional_usdt: str


def _decimal(v: Any) -> Decimal:
    try:
        d = Decimal(str(v))
    except (InvalidOperation, TypeError) as exc:
        raise DemoSafetyError("Некорректное числовое значение") from exc
    if not d.is_finite() or d <= 0:
        raise DemoSafetyError("Значение должно быть конечным и положительным")
    return d


def _symbol(symbol: str) -> str:
    if not isinstance(symbol, str) or symbol not in ALLOWED_SYMBOLS:
        raise DemoSafetyError("Разрешены только BTC/ETH/SOL/XRP USDT-M DEMO")
    return symbol


def _qty_text(v: Decimal) -> str:
    return format(v, "f")


class DemoHttp:
    """Minimal, fixed-host HTTPS connection; no redirects, cookies or URL overrides."""

    def __init__(
        self, api_key: str, secret: str,
        *, transport: Callable[..., Any] | None = None,
        clock: Callable[[], float] = time.time
    ):
        if not api_key or not secret:
            raise DemoSafetyError("Требуется отдельный демо API-ключ")
        self._key = api_key
        self._secret = secret.encode("utf-8")
        self._transport = transport or self._raw
        self._clock = clock

    @staticmethod
    def _raw(method: str, path: str, data: str, headers: dict[str, str]) -> Any:
        # HTTPSConnection uses certificate verification by default.
        connection = http.client.HTTPSConnection(DEMO_HOST, timeout=6)
        try:
            path_with_query = path + ("?" + data if method == "GET" else "")
            body = data if method != "GET" else None
            connection.request(method, path_with_query, body=body, headers=headers)
            response = connection.getresponse()
            budget=4_000_000 if path=="/fapi/v1/exchangeInfo" else 128*1024
            payload = response.read(budget+1)
            if len(payload) > budget:
                raise DemoSafetyError("Слишком большой ответ DEMO API")
            if response.status >= 400:
                # Never log API credentials or full signed URL.
                raise DemoSafetyError(
                    "DEMO HTTP status " + str(response.status) +
                    " (статус исполнения при 5xx неизвестен: проверьте ордера вручную)"
                )
            return json.loads(payload.decode("utf-8"))
        finally:
            connection.close()

    def request(self, method: str, path: str, values: dict[str, str] | None = None, *,
                signed: bool = False) -> Any:
        if (method, path) not in _ALLOWED:
            raise DemoSafetyError("Попытка запрещённой операции/API-пути")
        values = dict(values or {})
        if "signature" in values or "timestamp" in values or "recvWindow" in values:
            raise DemoSafetyError("Нельзя передавать собственную подпись")
        if signed:
            values["timestamp"] = str(int(self._clock() * 1000))
            values["recvWindow"] = "5000"
        data = urlencode(values)
        if signed:
            data += "&signature=" + hmac.new(
                self._secret, data.encode("utf-8"), hashlib.sha256
            ).hexdigest()
        headers = {"X-MBX-APIKEY": self._key}
        if method != "GET":
            headers["Content-Type"] = "application/x-www-form-urlencoded"
        return self._transport(method, path, data, headers)


class DemoFutures:
    """One-at-a-time manual demo orders. Deliberately no autonomous LLM entry."""

    def __init__(
        self, client: DemoHttp, *, enabled: bool = False, acknowledgement: str = "",
        max_notional: str = "50", max_risk: str = "1",
        clock: Callable[[], float] = time.time,
    ):
        self._client = client
        self._enabled = enabled and acknowledgement == "DEMO_ACCOUNT_ONLY"
        self._max_notional = _decimal(max_notional)
        self._max_risk = _decimal(max_risk)
        self._clock = clock
        self._lock = threading.Lock()
        self._issued: dict[str, TradeProposal] = {}
        self._used: set[str] = set()
        self._fatal = False
        self._attempts = 0

    def propose(self, symbol: str, side: str, qty: str, stop: str) -> TradeProposal:
        _symbol(symbol)
        if side not in {"BUY", "SELL"}:
            raise DemoSafetyError("Направление должно быть BUY или SELL")
        quantity, stop_price = _decimal(qty), _decimal(stop)
        market = self._client.request(
            "GET", "/fapi/v1/premiumIndex", {"symbol": symbol}
        )
        mark = _decimal(market["markPrice"])
        if side == "BUY" and not stop_price < mark:
            raise DemoSafetyError("Стоп LONG должен быть ниже рыночной цены")
        if side == "SELL" and not stop_price > mark:
            raise DemoSafetyError("Стоп SHORT должен быть выше рыночной цены")
        notional = mark * quantity
        risk = abs(mark - stop_price) * quantity
        if notional > self._max_notional or risk > self._max_risk:
            raise DemoSafetyError("Превышен лимит номинала/структурного риска")
        if risk / notional < Decimal("0.001"):
            raise DemoSafetyError("Стоп слишком близкий для этого демо-теста")
        proposal = TradeProposal(
            ticket=uuid.uuid4().hex, symbol=symbol, side=side,
            quantity=_qty_text(quantity), stop_price=_qty_text(stop_price),
            expires_at=int(self._clock() + 45), risk_usdt=_qty_text(risk),
            notional_usdt=_qty_text(notional)
        )
        with self._lock:
            self._issued[proposal.ticket] = proposal
        return proposal

    def _check_contract_filters(self, proposal: TradeProposal) -> None:
        info = self._client.request(
            "GET", "/fapi/v1/exchangeInfo"
        )
        listing = info.get("symbols") if isinstance(info,dict) else None
        if not isinstance(listing, list):
            raise DemoSafetyError("Нет точных биржевых ограничений символа")
        matching=[x for x in listing if isinstance(x,dict) and x.get("symbol")==proposal.symbol]
        if len(matching)!=1:
            raise DemoSafetyError("Нет точных биржевых ограничений символа")
        item = matching[0]
        if item.get("symbol") != proposal.symbol or item.get("status") != "TRADING":
            raise DemoSafetyError("Контракт недоступен на DEMO")
        qty = _decimal(proposal.quantity)
        stop = _decimal(proposal.stop_price)
        found_market_lot = False
        found_price_filter = False
        for rule in item.get("filters", []):
            typ = rule.get("filterType")
            if typ == "MARKET_LOT_SIZE":
                step = _decimal(rule["stepSize"])
                minimum = _decimal(rule["minQty"])
                maximum = _decimal(rule["maxQty"])
                if qty < minimum or qty > maximum or qty % step:
                    raise DemoSafetyError("Объём не соответствует MARKET_LOT_SIZE")
                found_market_lot = True
            if typ == "PRICE_FILTER":
                tick = _decimal(rule["tickSize"])
                if stop % tick:
                    raise DemoSafetyError("Уровень SL не соответствует PRICE_FILTER")
                found_price_filter = True
        if not found_market_lot or not found_price_filter:
            raise DemoSafetyError("Неполные ограничения Binance exchangeInfo")

    def _flat_one_way(self, symbol: str) -> None:
        # Do NOT trust a separate signed-check done seconds earlier. This
        # final independent gate runs inside the actual executor immediately
        # before attempting any mutation. Other symbols' exposure can make
        # even a tiny DEMO order undesirable while account state is unknown.
        dual = self._client.request("GET", "/fapi/v1/positionSide/dual", signed=True)
        if not isinstance(dual,dict) or dual.get("dualSidePosition") is not False:
            raise DemoSafetyError("Нужен режим One-way (BOTH), Hedge запрещён")
        positions = self._client.request(
            "GET", "/fapi/v3/positionRisk", signed=True
        )
        # Binance positionRisk V3 returns ONLY active symbols; an authenticated
        # empty array is how a completely flat, order-free account appears.
        if not isinstance(positions,list) or len(positions)>3000:
            raise DemoSafetyError("Не удалось подтвердить позиции всего DEMO-счёта")
        seen=set()
        for pos in positions:
            if not isinstance(pos,dict):
                raise DemoSafetyError("Недостоверный список позиций DEMO")
            sym=pos.get("symbol")
            if (not isinstance(sym,str) or len(sym)>30 or sym in seen
                    or pos.get("positionSide")!="BOTH"):
                raise DemoSafetyError("Неизвестная позиция или режим счёта")
            seen.add(sym)
            try:
                amount=Decimal(str(pos.get("positionAmt")))
            except (ValueError,TypeError,InvalidOperation) as exc:
                raise DemoSafetyError("Неверный объём позиции DEMO") from exc
            if not amount.is_finite() or amount != 0:
                raise DemoSafetyError("Позиция уже существует на DEMO-счёте")
        orders=self._client.request(
            "GET","/fapi/v1/openOrders",{"symbol":symbol},signed=True)
        algos=self._client.request(
            "GET","/fapi/v1/openAlgoOrders",{"symbol":symbol},signed=True)
        if not isinstance(orders,list) or not isinstance(algos,list):
            raise DemoSafetyError("Недостоверный список открытых DEMO-ордеров")
        if orders or algos:
            raise DemoSafetyError("На паре уже есть активные DEMO-ордера")

    def _stop_confirmed(self, algo_id: str, proposal: TradeProposal) -> bool:
        state = self._client.request(
            "GET", "/fapi/v1/algoOrder", {"algoId": algo_id}, signed=True
        )
        if not isinstance(state, dict):
            return False
        # A similar stop on the same symbol is NOT proof of *our* SL.
        # Reject wrong client ID, algo ID, mark/contract side and trigger.
        try:
            valid_price = _decimal(state.get("triggerPrice")) == _decimal(
                proposal.stop_price)
            identity = str(int(state.get("algoId"))) == str(int(algo_id))
        except (TypeError, ValueError, DemoSafetyError):
            return False
        return (
            identity
            and state.get("clientAlgoId") == "WAISL" + proposal.ticket[:23]
            and state.get("symbol") == proposal.symbol
            and state.get("side") == ("SELL" if proposal.side == "BUY" else "BUY")
            and state.get("orderType") == "STOP_MARKET"
            and state.get("positionSide") == "BOTH"
            and state.get("workingType") == "MARK_PRICE"
            and state.get("closePosition") is True
            and valid_price
            and state.get("algoStatus") == "NEW"
        )

    def execute(self, ticket: str, *, confirmation: str) -> dict[str, Any]:
        if not self._enabled or confirmation != "PLACE_DEMO_ORDER":
            raise DemoSafetyError("DEMO торговля выключена; требуется явное подтверждение")
        with self._lock:
            if self._fatal or self._attempts >= 2:
                raise DemoSafetyError("DEMO заблокирована после ошибки/лимита попыток")
            proposal = self._issued.get(ticket)
            if not proposal or ticket in self._used:
                raise DemoSafetyError("Заявка неизвестна или уже использована")
            self._used.add(ticket)  # Never automatically retry an uncertain order.
            self._attempts += 1
        if self._clock() > proposal.expires_at:
            raise DemoSafetyError("Подтверждение устарело — создайте новую заявку")
        self._flat_one_way(proposal.symbol)
        self._check_contract_filters(proposal)
        # Re-evaluate market for adverse move since the quote was prepared.
        now = self._client.request(
            "GET", "/fapi/v1/premiumIndex", {"symbol": proposal.symbol}
        )
        mark = _decimal(now["markPrice"])
        if mark * _decimal(proposal.quantity) > self._max_notional:
            raise DemoSafetyError("Слишком большой номинал после изменения цены")
        if abs(mark - _decimal(proposal.stop_price)) * _decimal(proposal.quantity) > self._max_risk:
            raise DemoSafetyError("Превышен риск после изменения цены")
        if proposal.side == "BUY" and not _decimal(proposal.stop_price) < mark:
            raise DemoSafetyError("Стоп потерял валидность")
        if proposal.side == "SELL" and not _decimal(proposal.stop_price) > mark:
            raise DemoSafetyError("Стоп потерял валидность")

        algo_id: str | None = None
        entered = False
        entry_attempted = False
        try:
            protective = self._client.request(
                "POST", "/fapi/v1/algoOrder", {
                    "algoType": "CONDITIONAL",
                    "symbol": proposal.symbol,
                    "side": "SELL" if proposal.side == "BUY" else "BUY",
                    "type": "STOP_MARKET",
                    "positionSide": "BOTH",
                    "triggerPrice": proposal.stop_price,
                    "workingType": "MARK_PRICE",
                    "closePosition": "true",
                    "clientAlgoId": "WAISL" + ticket[:23],
                }, signed=True
            )
            algo_id = str(protective["algoId"])
            if not self._stop_confirmed(algo_id, proposal):
                raise DemoSafetyError("DEMO STOP не подтверждён, вход запрещён")
            entry_attempted = True  # After this point a timeout may mean FILLED.
            order_response = self._client.request(
                "POST", "/fapi/v1/order", {
                    "symbol": proposal.symbol,
                    "side": proposal.side,
                    "type": "MARKET",
                    "positionSide": "BOTH",
                    "quantity": proposal.quantity,
                    "newOrderRespType": "RESULT",
                    "newClientOrderId": "WAIEN" + ticket[:23],
                }, signed=True
            )
            entered = True
            # Do not mistake an ACK/FILLED label for an authenticated
            # *entire* position. Result must include exact filled quantity
            # and durable exchange order ID, then fresh signed REST GET must
            # independently confirm status/quantity and account position.
            if (not isinstance(order_response, dict)
                    or order_response.get("symbol") != proposal.symbol
                    or order_response.get("status") != "FILLED"):
                raise DemoSafetyError("DEMO_ENTRY_NOT_VERIFIED_FILLED")
            try:
                order_id = str(int(order_response["orderId"]))
                actual_fill = _decimal(order_response["executedQty"])
            except (KeyError,TypeError,ValueError,DemoSafetyError) as exc:
                raise DemoSafetyError("DEMO_ENTRY_EXECUTION_DETAILS_UNVERIFIED") from exc
            if int(order_id) <= 0 or actual_fill != _decimal(proposal.quantity):
                raise DemoSafetyError("DEMO_ENTRY_FILL_QUANTITY_MISMATCH")
            exchange_order = self._client.request(
                "GET", "/fapi/v1/order",
                {"symbol":proposal.symbol,"orderId":order_id}, signed=True)
            if (not isinstance(exchange_order,dict)
                    or exchange_order.get("symbol")!=proposal.symbol
                    or str(exchange_order.get("orderId"))!=order_id
                    or exchange_order.get("clientOrderId")!="WAIEN"+ticket[:23]
                    or exchange_order.get("side")!=proposal.side
                    or exchange_order.get("positionSide")!="BOTH"
                    or exchange_order.get("status")!="FILLED"
                    or _decimal(exchange_order.get("executedQty"))
                    !=_decimal(proposal.quantity)):
                raise DemoSafetyError("DEMO_SIGNED_ORDER_RECONCILIATION_FAILED")
            account_positions = self._client.request(
                "GET", "/fapi/v3/positionRisk",
                {"symbol":proposal.symbol}, signed=True)
            expected_position = (_decimal(proposal.quantity)
                                 if proposal.side=="BUY"
                                 else -_decimal(proposal.quantity))
            if (not isinstance(account_positions,list)
                    or len(account_positions)!=1
                    or account_positions[0].get("symbol")!=proposal.symbol
                    or account_positions[0].get("positionSide")!="BOTH"
                    or Decimal(str(account_positions[0].get("positionAmt")))
                    !=expected_position):
                raise DemoSafetyError("DEMO_ACCOUNT_POSITION_RECONCILIATION_FAILED")
            if not self._stop_confirmed(algo_id, proposal):
                raise DemoSafetyError("После входа подтверждение защитного SL потеряно")
            return {
                "venue": "BINANCE_FUTURES_DEMO",
                "ticket": ticket,
                "symbol": proposal.symbol,
                "stop_algo_id": algo_id,
                "stop_confirmed": True,
                "result": "DEMO_ENTRY_SENT_VERIFY_POSITION",
            }
        except Exception as exc:
            with self._lock:
                self._fatal = True
            # Never blindly send a duplicate entry. If POST timed out,
            # the result is unknown: retain the stop and require reconciliation.
            if entered:
                try:
                    self._client.request(
                        "POST", "/fapi/v1/order", {
                            "symbol": proposal.symbol,
                            "side": "SELL" if proposal.side == "BUY" else "BUY",
                            "type": "MARKET",
                            "positionSide": "BOTH",
                            "quantity": proposal.quantity,
                            "reduceOnly": "true",
                            "newClientOrderId": "WAIFL" + ticket[:23],
                        }, signed=True
                    )
                except Exception:
                    pass
            # If entry may have happened on timeout, DO NOT remove SL.
            if algo_id and not entry_attempted and isinstance(exc, DemoSafetyError):
                try:
                    self._client.request(
                        "DELETE", "/fapi/v1/algoOrder",
                        {"algoId": algo_id}, signed=True
                    )
                except Exception:
                    pass
            raise DemoSafetyError(
                "ДЕМО заблокирована: проверьте позицию и биржевой SL вручную. Причина: " +
                str(exc)[:150]
            ) from exc
