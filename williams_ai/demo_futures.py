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
    ("GET", "/fapi/v1/positionSide/dual"),
    ("GET", "/fapi/v3/positionRisk"),
    ("GET", "/fapi/v1/algoOrder"),
    ("GET", "/fapi/v1/order"),
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
            payload = response.read(128 * 1024 + 1)
            if len(payload) > 128 * 1024:
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
        max_notional: str = "150", max_risk: str = "3",
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

    def _flat_one_way(self, symbol: str) -> None:
        dual = self._client.request("GET", "/fapi/v1/positionSide/dual", signed=True)
        if dual.get("dualSidePosition") is not False:
            raise DemoSafetyError("Нужен режим One-way (BOTH), Hedge запрещён")
        positions = self._client.request(
            "GET", "/fapi/v3/positionRisk", {"symbol": symbol}, signed=True
        )
        if not isinstance(positions, list) or not positions:
            raise DemoSafetyError("Не удалось подтвердить позиции на демо-бирже")
        for pos in positions:
            if pos.get("symbol") != symbol or pos.get("positionSide") != "BOTH":
                raise DemoSafetyError("Неизвестная позиция или режим счёта")
            if Decimal(str(pos["positionAmt"])) != 0:
                raise DemoSafetyError("Позиция уже существует, новый вход запрещён")

    def _stop_confirmed(self, algo_id: str, proposal: TradeProposal) -> bool:
        state = self._client.request(
            "GET", "/fapi/v1/algoOrder", {"algoId": algo_id}, signed=True
        )
        return (
            state.get("symbol") == proposal.symbol
            and state.get("side") == ("SELL" if proposal.side == "BUY" else "BUY")
            and state.get("orderType") == "STOP_MARKET"
            and state.get("closePosition") is True
            and _decimal(state.get("triggerPrice")) == _decimal(proposal.stop_price)
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
            self._client.request(
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
            entered = True  # Also conservatively assumed after timeout below.
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
            # Never blindly send duplicate entry. If POST timed out, result
            # is unknown and must be verified by operator.
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
            # If entry may have happened on timeout, do NOT remove SL.
            if algo_id and not entered and isinstance(exc, DemoSafetyError):
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
