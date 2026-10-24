"""Actual DE-MO-only safety preflight mocked offline; no real orders."""
import os
import stat
import tempfile
import unittest
from pathlib import Path

from williams_ai.demo_pilot_readiness import (
    PAIR,DEMO_HOST,OneCampaignReservation,check_public_demo,
    validate_small_pilot,DemoSafetyError,
)

CLOCK=2_000_000_000_000


def public_fixture(*,status="TRADING",server=CLOCK,mark_time=CLOCK,
                   price="100",tick="0.01",step="0.001",
                   min_qty="0.001",max_qty="20"):
    calls=[]
    def fetch(path,params):
        calls.append((path,params))
        if path=="/fapi/v1/time":
            return {"serverTime":server}
        if path=="/fapi/v1/exchangeInfo":
            return {"symbols":[{"symbol":"ETHUSDT","status":status,
                "filters":[{"filterType":"PRICE_FILTER","tickSize":tick},
                           {"filterType":"MARKET_LOT_SIZE","stepSize":step,
                            "minQty":min_qty,"maxQty":max_qty}]}]}
        if path=="/fapi/v1/premiumIndex":
            return {"symbol":"ETHUSDT","markPrice":price,"time":mark_time}
        raise AssertionError("unexpected public endpoint")
    return fetch,calls


def small_preview():
    fetch,_=public_fixture()
    view=check_public_demo(fetch=fetch,now_ms=CLOCK)
    return validate_small_pilot(
        view,side="BUY",quantity="0.1",stop="95",now_ms=CLOCK),view


class PilotGateTests(unittest.TestCase):
    def test_real_binance_demo_endpoint_pinned_and_public_checks_only(self):
        self.assertEqual(DEMO_HOST,"demo-fapi.binance.com")
        self.assertEqual(PAIR,"ETHUSDT")
        fetch,calls=public_fixture()
        health=check_public_demo(fetch=fetch,now_ms=CLOCK)
        self.assertEqual(calls,[
            ("/fapi/v1/time",None),
            ("/fapi/v1/exchangeInfo",None),
            ("/fapi/v1/premiumIndex",{"symbol":"ETHUSDT"}),
        ])
        self.assertFalse(health.execution_allowed)
        self.assertIsNone(health.trade_action)
        self.assertEqual(health.mark_price,"100")
        self.assertEqual(health.venue,"BINANCE_FUTURES_DEMO")
        with self.assertRaisesRegex(DemoSafetyError,"SINGLE_PAIR"):
            check_public_demo(fetch=fetch,now_ms=CLOCK,symbol="BTCUSDT")

    def test_safe_tiny_manual_preview_only_no_auth_or_orders(self):
        preview,_=small_preview()
        self.assertEqual(preview["mode"],"SINGLE_PAIR_DEMO_PREVIEW_ONLY")
        self.assertEqual(preview["symbol"],"ETHUSDT")
        self.assertEqual(preview["estimated_risk_demo_usdt"],"0.5")
        self.assertEqual(preview["estimated_notional_demo_usdt"],"10.0")
        self.assertFalse(preview["exchange_position_verified_flat"])
        self.assertFalse(preview["stop_acknowledged"])
        self.assertFalse(preview["execution_allowed"])
        self.assertIsNone(preview["trade_action"])

    def test_clock_and_price_age_fail_closed(self):
        old,_=public_fixture(server=CLOCK-11_000)
        with self.assertRaisesRegex(DemoSafetyError,"TIME_DRIFT"):
            check_public_demo(fetch=old,now_ms=CLOCK)
        old,_=public_fixture(mark_time=CLOCK-11_000)
        with self.assertRaisesRegex(DemoSafetyError,"MARK_FRESHNESS"):
            check_public_demo(fetch=old,now_ms=CLOCK)
        fail,_=public_fixture(server=True)
        with self.assertRaisesRegex(DemoSafetyError,"SERVER_TIME"):
            check_public_demo(fetch=fail,now_ms=CLOCK)
        _,state=small_preview()
        with self.assertRaisesRegex(DemoSafetyError,"PREFLIGHT_EXPIRED"):
            validate_small_pilot(state,side="BUY",quantity="0.1",
                                 stop="95",now_ms=CLOCK+31_000)

    def test_unknown_contract_or_missing_lot_filter_blocks(self):
        halted,_=public_fixture(status="BREAK")
        with self.assertRaisesRegex(DemoSafetyError,"NOT_TRADABLE"):
            check_public_demo(fetch=halted,now_ms=CLOCK)
        missing,_=public_fixture(step="0")
        with self.assertRaises(DemoSafetyError):
            check_public_demo(fetch=missing,now_ms=CLOCK)
        weird,_=public_fixture(price="-1")
        with self.assertRaises(DemoSafetyError):
            check_public_demo(fetch=weird,now_ms=CLOCK)

    def test_wrong_stop_risk_size_and_precision_rejected(self):
        _,state=small_preview()
        for side,qty,stop in (
            ("BUY","0.1","110"),("SELL","0.1","95"),
            ("BUY","0.1","95.005"),("BUY","0.1001","95"),
            ("BUY","1","95"),("BUY","0.25","95"),
            ("BUY","0.1","99.99"),("BUY","true","95")
        ):
            with self.subTest(side=side,qty=qty,stop=stop):
                with self.assertRaises(DemoSafetyError):
                    validate_small_pilot(state,side=side,
                                         quantity=qty,stop=stop,now_ms=CLOCK)

    def test_durable_restart_prevents_second_order_and_secret_storage(self):
        proposal,_=small_preview()
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"pilot.sqlite3"
            journal=OneCampaignReservation(path)
            self.assertEqual(stat.S_IMODE(path.stat().st_mode),0o600)
            journal.reserve_once(proposal,now_ms=CLOCK)
            clone=OneCampaignReservation(path)
            self.assertEqual(len(clone.status()["reservations"]),1)
            with self.assertRaisesRegex(DemoSafetyError,"ALREADY_RESERVED"):
                clone.reserve_once(proposal,now_ms=CLOCK+1)
            self.assertEqual(clone.status()["reservations"][0]["state"],
                             "UNKNOWN_UNTIL_EXCHANGE_RECONCILED")
            self.assertFalse(clone.status()["execution_allowed"])
            self.assertNotIn("api_key",str(clone.status()).lower())
            self.assertNotIn("secret",str(clone.status()).lower())

    def test_stale_or_privately_incorrect_journal_fails(self):
        proposal,_=small_preview()
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"pilot.sqlite3"
            path.write_text("not a sqlite journal")
            os.chmod(path,0o644)
            with self.assertRaisesRegex(DemoSafetyError,"NOT_PRIVATE"):
                OneCampaignReservation(path)
            path.unlink()
            link=Path(d)/"somewhere"
            link.write_text("target")
            path.symlink_to(link)
            with self.assertRaisesRegex(DemoSafetyError,"PRIVATE_EXISTING"):
                OneCampaignReservation(path)

    def test_inference_cannot_upgrade_paper_to_order_intent(self):
        proposal,_=small_preview()
        with tempfile.TemporaryDirectory() as d:
            ledger=OneCampaignReservation(Path(d)/"pilot.sqlite3")
            variants=[
                {"symbol":"BTCUSDT"}, {"trade_action":{"BUY":"ETHUSDT"}},
                {"execution_allowed":True},{"mode":"AUTO_PILOT_BUY"},
            ]
            for variant in variants:
                with self.subTest(variant=variant):
                    fake=dict(proposal)
                    fake.update(variant)
                    with self.assertRaises(DemoSafetyError):
                        ledger.reserve_once(fake,now_ms=CLOCK)
            self.assertEqual(ledger.status()["reservations"],[])


if __name__=="__main__":
    unittest.main()
