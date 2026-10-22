import unittest
from copy import deepcopy

from williams_ai.analyst import build_report, render_russian_report


BASE = {
    "symbol": "ETHUSDT", "as_of": "2026-10-22T18:00:00Z",
    "timeframes": {"H1": {
        "last_closed_at": "2026-10-22T18:00:00Z",
        "phase": "CORRECTION", "ao": -1.0,
        "wave_hypothesis": "возможная четвёртая волна"
    }},
    "signals": [{
        "name": "WM3", "timeframe": "H1", "direction": "LONG",
        "confirmed_at": "2026-10-22T18:00:00Z", "confirmed": False,
        "reason": "фрактал не подтверждён"
    }]
}


class WilliamsAIIsolationTest(unittest.TestCase):
    def test_read_only_and_no_trade_action(self):
        report = build_report(deepcopy(BASE))
        self.assertEqual(report["mode"], "READ_ONLY")
        self.assertIsNone(report["trade_action"])
        self.assertIn("НЕ подтверждён", render_russian_report(report))

    def test_no_future_candle(self):
        data = deepcopy(BASE)
        data["timeframes"]["H1"]["last_closed_at"] = "2026-10-22T19:00:00Z"
        with self.assertRaisesRegex(ValueError, "no look-ahead"):
            build_report(data)

    def test_no_future_signal(self):
        data = deepcopy(BASE)
        data["signals"][0]["confirmed_at"] = "2026-10-22T19:00:00Z"
        with self.assertRaisesRegex(ValueError, "no look-ahead"):
            build_report(data)

    def test_warn_when_real_exchange_stop_not_confirmed(self):
        data = deepcopy(BASE)
        data["position"] = {"status": "OPEN", "side": "LONG",
                            "stop_confirmed_on_exchange": False}
        report = build_report(data)
        self.assertIn("КРИТИЧНО", report["warnings"][0])

    def test_never_forward_secrets_or_account(self):
        data = deepcopy(BASE)
        data["api_key"] = "SENSITIVE_CANARY_VALUE"
        data["account_id"] = "PRIVATE_ACCOUNT_CANARY"
        data["timeframes"]["H1"]["secret"] = "ANOTHER_CANARY"
        text = str(build_report(data))
        for value in ("SENSITIVE_CANARY_VALUE", "PRIVATE_ACCOUNT_CANARY", "ANOTHER_CANARY"):
            self.assertNotIn(value, text)

    def test_stale_data_warns(self):
        data = deepcopy(BASE)
        data["timeframes"]["H1"]["last_closed_at"] = "2026-10-22T10:00:00Z"
        self.assertTrue(any("устарели" in x for x in build_report(data)["warnings"]))

    def test_naive_datetime_rejected(self):
        data = deepcopy(BASE)
        data["as_of"] = "2026-10-22T18:00:00"
        with self.assertRaisesRegex(ValueError, "timezone"):
            build_report(data)


if __name__ == "__main__":
    unittest.main()
