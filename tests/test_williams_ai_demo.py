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
        self.stop_client_id = None
        self.entry_client_id = None
        self.forge_stop_id = False
        self.forge_entry_id = False
        self.other_symbol_position = "0"
        self.open_orders = []
        self.open_algo_orders = []

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
            query=parse_qs(body)
            position={"symbol":"ETHUSDT","positionSide":"BOTH",
                      "positionAmt":("0" if self.position_reconciliation_fails
                                     else self.position_qty)}
            return ([position] if query.get("symbol")==["ETHUSDT"]
                    else [position,{"symbol":"BTCUSDT","positionSide":"BOTH",
                                    "positionAmt":self.other_symbol_position}])
        if method == "GET" and path == "/fapi/v1/openOrders":
            return self.open_orders
        if method == "GET" and path == "/fapi/v1/openAlgoOrders":
            return self.open_algo_orders
        if method == "POST" and path == "/fapi/v1/algoOrder":
            self.stop_client_id=parse_qs(body).get("clientAlgoId",[None])[0]
            return {"algoId": 99999, "algoStatus": "NEW"}
        if method == "GET" and path == "/fapi/v1/algoOrder":
            return {
                "algoId":99999,
                "clientAlgoId":("A_FOREIGN_STOP" if self.forge_stop_id else self.stop_client_id),
                "symbol": "ETHUSDT", "side": "SELL",
                "positionSide":"BOTH","workingType":"MARK_PRICE",
                "orderType": "STOP_MARKET", "closePosition": True,
                "triggerPrice": "95", "algoStatus": "NEW" if self.stop_ok else "CANCELED"
            }
        if method == "POST" and path == "/fapi/v1/order":
            self.market_submissions += 1
            if self.uncertain_entry:
                raise TimeoutError("fake exchange timeout")
            data=parse_qs(body)
            qty=data.get("quantity",["0"])[0]
            self.entry_client_id=data.get("newClientOrderId",[None])[0]
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
                    "clientOrderId":("A_FOREIGN_ENTRY" if self.forge_entry_id else self.entry_client_id),
                    "side":"BUY","positionSide":"BOTH",
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

    def test_last_moment_other_symbol_exposure_blocks_all_mutations(self):
        self.venue.other_symbol_position="0.07"
        p=self.engine.propose("ETHUSDT","BUY","0.2","95")
        with self.assertRaisesRegex(DemoSafetyError,"позиция|Позиция|заблокирована"):
            self.engine.execute(p.ticket,confirmation="PLACE_DEMO_ORDER")
        self.assertFalse(any(method in ("POST","DELETE")
                             for method,path,*_ in self.venue.calls))

    def test_last_moment_pending_regular_or_conditional_order_blocks(self):
        for flag in ("open_orders","open_algo_orders"):
            with self.subTest(flag=flag):
                self.setUp()
                setattr(self.venue,flag,[{"orderId":123,"symbol":"ETHUSDT"}])
                p=self.engine.propose("ETHUSDT","BUY","0.2","95")
                with self.assertRaisesRegex(DemoSafetyError,"ордера|ордера|заблокирована"):
                    self.engine.execute(p.ticket,confirmation="PLACE_DEMO_ORDER")
                self.assertFalse(any(method in ("POST","DELETE")
                                     for method,path,*_ in self.venue.calls))

    def test_stop_rejected_means_no_entry(self):
        self.venue.stop_ok = False
        p = self.engine.propose("ETHUSDT", "BUY", "0.2", "95")
        with self.assertRaisesRegex(DemoSafetyError, "заблокирована"):
            self.engine.execute(p.ticket, confirmation="PLACE_DEMO_ORDER")
        self.assertEqual(self.venue.market_submissions, 0)
        with self.assertRaisesRegex(DemoSafetyError, "заблокирована"):
            self.engine.execute(p.ticket, confirmation="PLACE_DEMO_ORDER")

    def test_foreign_stop_on_same_symbol_cannot_authorize_new_entry(self):
        self.venue.forge_stop_id=True
        proposal=self.engine.propose("ETHUSDT","BUY","0.2","95")
        with self.assertRaisesRegex(DemoSafetyError,"заблокирована"):
            self.engine.execute(proposal.ticket,confirmation="PLACE_DEMO_ORDER")
        self.assertEqual(self.venue.market_submissions,0)

    def test_signed_get_foreign_filled_entry_is_not_authenticated(self):
        self.venue.forge_entry_id=True
        proposal=self.engine.propose("ETHUSDT","BUY","0.2","95")
        with self.assertRaisesRegex(DemoSafetyError,"заблокирована"):
            self.engine.execute(proposal.ticket,confirmation="PLACE_DEMO_ORDER")
        self.assertGreaterEqual(self.venue.market_submissions,1)
        with self.assertRaises(DemoSafetyError):
            self.engine.execute(proposal.ticket,confirmation="PLACE_DEMO_ORDER")

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
