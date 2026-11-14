"""Offline unit verification for historic100 official-bar model separation (no market fetch)."""
import unittest
from scripts.ai_v6_historical100_real_binance import (
    HOUR_MS, HISTORY, FUTURE, aggregate, derived_ao_ac, months_for,
    normalize_ms, chart_for,
)


class Historical100Contract(unittest.TestCase):
    def test_month_span_two_months(self):
        from datetime import datetime, timezone
        cutoff = int(datetime(2025, 2, 15, tzinfo=timezone.utc).timestamp() * 1000)
        self.assertEqual(months_for(cutoff - HISTORY * HOUR_MS, cutoff + FUTURE * HOUR_MS - 1),
                         ["2025-01", "2025-02"])

    def test_microsecond_timestamp_normalization(self):
        self.assertEqual(normalize_ms("1736895600000000"), 1736895600000)
        self.assertEqual(normalize_ms("1736895600000"), 1736895600000)

    def test_causal_ao_ac_no_future(self):
        bars = [{"high": i + 2, "low": i + 1} for i in range(80)]
        before = derived_ao_ac(bars)
        after = derived_ao_ac(bars + [{"high": 9999, "low": 9998}])
        self.assertNotEqual(before, after)

    def test_incomplete_daily_prefix_not_exposed(self):
        from datetime import datetime, timedelta, timezone
        base = datetime(2025, 1, 3, 8, tzinfo=timezone.utc)
        bars = [{
            "open_utc": (base + timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "high": i + 2, "low": i + 1, "open": i + 1,
            "close": i + 1.5, "volume": 1
        } for i in range(1024)]
        self.assertEqual(len(aggregate(bars, 24)), 42)
        self.assertEqual(len(aggregate(bars, 4)), 256)

    def test_dataset_fails_closed_without_exact_1024_plus_future(self):
        from datetime import datetime, timezone
        cutoff = int(datetime(2025, 2, 15, tzinfo=timezone.utc).timestamp() * 1000)
        begin = cutoff - HISTORY * HOUR_MS
        rows = [(begin + i * HOUR_MS, 100, 101, 99, 100, 1)
                for i in range(HISTORY + FUTURE - 1)]
        case = {"id": "x", "symbol": "BTCUSDT",
                "cutoff_utc": "2025-02-15T00:00:00Z",
                "edition": "TC1", "family": "ENTRY"}
        with self.assertRaisesRegex(ValueError, "HIST100_MISSING_WINDOW_H1"):
            chart_for(case, {
                ("BTCUSDT", "2025-01"): (rows, {"official_checksum_verified": True}),
                ("BTCUSDT", "2025-02"): ([], {"official_checksum_verified": True})
            })


if __name__ == "__main__":
    unittest.main()
