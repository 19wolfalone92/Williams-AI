"""Demo-only finance safety contracts. Mocks NEVER access the exchange."""
import unittest
from urllib.parse import parse_qs
from williams_ai.demo_futures import (
    DemoFutures, DemoHttp, DemoSafetyError, DEMO_HOST, ALLOWED_SYMBOLS
)


class FakeVenue:
    def __init__(self, stop_ok=True):
        self.calls = []
        self.stop_ok = stop_ok
        self.market_submissions = 0
        self.uncertain_entry = False
        self.position_qty = "0"
        self.last_order_qty = None
        self.fake_partial_fill = False
        self.signed_order_reconciliation_fails = False
        self.position_reconciliation_fails = False

    def __call__(self, method, path, body, headers):
        self.calls.append((method, path, parse_qs(body), headers))
        if method == "GET" and path == "/fapi/v1/exchangeInfo":
            return {"symbols": [{"symbol": "ETHUSDT", "status": "TRADING",
                "filters": [{"filterType": "MARKET_LOT_SIZE", "stepSize": "0.001",
                             "minQty": "0.001", "maxQty": "100"},
                            {"filterType": "PRICE_FILTER", "tickSize": "0.01"}]}]}
        if method == "GET" and path == "/fapi/v1/premiumIndex":
            return {"markPrice": "100"}
        if method == "GET" and path == "/fapi/v1/positionSide/dual":
            return {"dualSidePosition": False}
        if method == "GET" and path == "/fapi/v3/positionRisk":
            return [{"symbol": "ETHUSDT", "positionSide": "BOTH", "positionAmt":
                "0" if self.position_reconciliation_fails else self.position_qty}]
        if method == "POST" and path == "/fapi/v1/algoOrder":
            return {"algoId": 99999, "algoStatus": "NEW"}
        if method == "GET" and path == "/fapi/v1/algoOrder":
            return {
                "symbol": "ETHUSDT", "side": "SELL",
                "orderType": "STOP_MARKET", "closePosition": True,
                "triggerPrice": "95", "algoStatus": "NEW" if self.stop_ok else "CANCELED"
            }
        if method == "POST" and path == "/fapi/v1/order":
            self.market_submissions += 1
            if self.uncertain_entry:
                raise TimeoutError("fake exchange timeout")
            data=parse_qs(body)
            qty=data.get("quantity",["0"])[0]
            self.last_order_qty=qty
            if data.get("reduceOnly")!=["true"]:
                self.position_qty=(
                    qty if data.get("side")==["BUY"] else "-"+qty)
            if self.fake_partial_fill:
                return {"symbol":"ETHUSDT","orderId":42,
                        "status":"FILLED","executedQty":"0.001"}
            return {"symbol": "ETHUSDT", "orderId": 42, "status": "FILLED",
                    "executedQty":qty}
        if method=="GET" and path=="/fapi/v1/order":
            if self.signed_order_reconciliation_fails:
                return {"symbol":"ETHUSDT","orderId":42,
                        "status":"NEW","executedQty":"0"}
            return {"symbol":"ETHUSDT","orderId":42,
                    "status":"FILLED","executedQty":self.last_order_qty}
        if method == "DELETE" and path == "/fapi/v1/algoOrder":
            return {"algoId": 99999, "algoStatus": "CANCELED"}
        raise AssertionError(f"Unexpected request {method} {path}")


class DemoGatewayTests(unittest.TestCase):
    def setUp(self):
        self.venue = FakeVenue()
        self.client = DemoHttp("DEMO_KEY", "TEST_SECRET", transport=self.venue)
        self.engine = DemoFutures(self.client, enabled=True, acknowledgement="DEMO_ACCOUNT_ONLY")

    def test_domain_is_immutable_demo_and_methods_restricted(self):
        self.assertEqual(DEMO_HOST, "demo-fapi.binance.com")
        with self.assertRaises(DemoSafetyError):
            self.client.request("POST", "/fapi/v1/withdraw", signed=True)
        with self.assertRaises(DemoSafetyError):
            self.client.request("GET", "https://fapi.binance.com/fapi/v1/order")

    def test_default_disabled_and_symbol_limited(self):
        disabled = DemoFutures(self.client)
        p = disabled.propose("ETHUSDT", "BUY", "0.2", "95")
        with self.assertRaisesRegex(DemoSafetyError, "выключена"):
            disabled.execute(p.ticket, confirmation="PLACE_DEMO_ORDER")
        self.assertEqual(self.venue.market_submissions, 0)
        for symbol in ("BTCUSD", "DOGEUSDT", "btcusdt"):
            self.assertNotIn(symbol, ALLOWED_SYMBOLS)
            with self.assertRaises(DemoSafetyError):
                self.engine.propose(symbol, "BUY", "1", "95")

    def test_prearm_stop_before_single_market_order(self):
        p = self.engine.propose("ETHUSDT", "BUY", "0.2", "95")
        with self.assertRaisesRegex(DemoSafetyError, "подтверждение"):
            self.engine.execute(p.ticket, confirmation="WRONG")
        result = self.engine.execute(p.ticket, confirmation="PLACE_DEMO_ORDER")
        self.assertTrue(result["stop_confirmed"])
        call_names = [(method, path) for method, path, *_ in self.venue.calls]
        stop_post = call_names.index(("POST", "/fapi/v1/algoOrder"))
        market_post = call_names.index(("POST", "/fapi/v1/order"))
        self.assertLess(stop_post, market_post)
        self.assertIn("signature", self.venue.calls[market_post][2])
        self.assertNotIn("DEMO_KEY", str(self.venue.calls[market_post][2]))
        with self.assertRaisesRegex(DemoSafetyError, "использована"):
            self.engine.execute(p.ticket, confirmation="PLACE_DEMO_ORDER")
        self.assertEqual(self.venue.market_submissions, 1)

    def test_exchange_fill_size_mismatch_locks_single_attempt(self):
        self.venue.fake_partial_fill = True
        p = self.engine.propose("ETHUSDT", "BUY", "0.2", "95")
        with self.assertRaisesRegex(DemoSafetyError,"заблокирована"):
            self.engine.execute(p.ticket,confirmation="PLACE_DEMO_ORDER")
        with self.assertRaisesRegex(DemoSafetyError,"заблокирована"):
            self.engine.execute(p.ticket,confirmation="PLACE_DEMO_ORDER")
        self.assertGreaterEqual(self.venue.market_submissions,1)

    def test_missing_signed_fill_recheck_is_not_success(self):
        self.venue.signed_order_reconciliation_fails = True
        p = self.engine.propose("ETHUSDT", "BUY", "0.2", "95")
        with self.assertRaisesRegex(DemoSafetyError,"заблокирована"):
            self.engine.execute(p.ticket,confirmation="PLACE_DEMO_ORDER")

    def test_disagreement_with_account_position_is_not_success(self):
        self.venue.position_reconciliation_fails = True
        p = self.engine.propose("ETHUSDT", "BUY", "0.2", "95")
        with self.assertRaisesRegex(DemoSafetyError,"заблокирована"):
            self.engine.execute(p.ticket,confirmation="PLACE_DEMO_ORDER")

    def test_stop_rejected_means_no_entry(self):
        self.venue.stop_ok = False
        p = self.engine.propose("ETHUSDT", "BUY", "0.2", "95")
        with self.assertRaisesRegex(DemoSafetyError, "заблокирована"):
            self.engine.execute(p.ticket, confirmation="PLACE_DEMO_ORDER")
        self.assertEqual(self.venue.market_submissions, 0)
        with self.assertRaisesRegex(DemoSafetyError, "заблокирована"):
            self.engine.execute(p.ticket, confirmation="PLACE_DEMO_ORDER")

    def test_no_order_when_risk_exceeded_or_stop_wrong(self):
        with self.assertRaisesRegex(DemoSafetyError, "Стоп LONG"):
            self.engine.propose("ETHUSDT", "BUY", "0.2", "120")
        with self.assertRaisesRegex(DemoSafetyError, "лимит"):
            self.engine.propose("ETHUSDT", "BUY", "5", "95")
        self.assertEqual(self.venue.market_submissions, 0)

    def test_uncertain_market_response_leaves_protection_in_place(self):
        self.venue.uncertain_entry = True
        p = self.engine.propose("ETHUSDT", "BUY", "0.2", "95")
        with self.assertRaises(DemoSafetyError):
            self.engine.execute(p.ticket, confirmation="PLACE_DEMO_ORDER")
        self.assertFalse(any(method == "DELETE" for method, *_ in self.venue.calls))
        self.assertEqual(self.venue.market_submissions, 1)
        with self.assertRaises(DemoSafetyError):
            self.engine.execute(p.ticket, confirmation="PLACE_DEMO_ORDER")

    def test_order_qty_fails_exchange_precision_without_post(self):
        p = self.engine.propose("ETHUSDT", "BUY", "0.12345", "95")
        with self.assertRaises(DemoSafetyError):
            self.engine.execute(p.ticket, confirmation="PLACE_DEMO_ORDER")
        self.assertEqual(self.venue.market_submissions, 0)

    def test_no_external_timestamp_or_signature(self):
        with self.assertRaises(DemoSafetyError):
            self.client.request("GET", "/fapi/v1/premiumIndex", {"timestamp": "0"})


if __name__ == "__main__":
    unittest.main()
