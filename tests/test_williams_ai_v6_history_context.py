"""V6 causal H1/H4/D1 feature construction tests (mock, not market gold)."""
from datetime import datetime, timedelta, timezone
import copy
import unittest

from williams_ai.v6_history_context import causal_context
from williams_ai.v6_public_chart_collector import BARS, MS, normalized_chart


CUTOFF = datetime(2026, 11, 15, tzinfo=timezone.utc)


def candle_chart():
    start = int(CUTOFF.timestamp()*1000) - BARS*MS
    klines = []
    for i in range(BARS):
        op = 100 + 0.1*i
        klines.append([start+i*MS, str(op), str(op+2),
                       str(op-1), str(op+1), "12",
                       start+(i+1)*MS-1, "0", 9, "0", "0", "0"])
    return normalized_chart("BTCUSDT", CUTOFF, klines)


class ForwardMultiScaleContext(unittest.TestCase):
    def test_1024_bars_yield_past_only_H1_H4_D1_ao_ac(self):
        chart = candle_chart()
        x = causal_context(chart, cutoff_utc="2026-11-15T00:00:00Z")
        h1, h4, d1 = (x["timeframes"][t] for t in ("H1", "H4", "D1"))
        self.assertEqual(h1["closed_bars"], BARS)
        self.assertGreaterEqual(h4["closed_bars"], 255)
        self.assertGreaterEqual(d1["closed_bars"], 40)
        self.assertIsNotNone(d1["ao_ac_median_price"]["ao"])
        self.assertIsNotNone(d1["ao_ac_median_price"]["ac"])
        self.assertTrue(all(frame["latest_close_utc"] <=
                            "2026-11-15T00:00:00Z" for frame in (h1, h4, d1)))
        self.assertTrue(x["alligator_angulation_not_established"])
        self.assertFalse(x["wave_degree_expert_verified"])
        self.assertTrue(x["not_executable_signals"])

    def test_future_bar_rejected_before_any_derived_signal(self):
        chart = candle_chart()
        chart["bars"][-1]["close_utc"] = "2026-11-15T01:00:00Z"
        with self.assertRaisesRegex(ValueError, "FUTURE_BAR"):
            causal_context(chart, cutoff_utc="2026-11-15T00:00:00Z")

    def test_only_complete_larger_buckets_get_indicators(self):
        chart = candle_chart()
        # Artificially lower the cutoff by 3 H1 bars, keeping only true
        # earlier candles. Incomplete later H4/D1 bucket must be excluded.
        subset = {"symbol": chart["symbol"], "bars": chart["bars"][:-3]}
        cutoff = subset["bars"][-1]["close_utc"]
        data = causal_context(subset, cutoff_utc=cutoff)
        self.assertLessEqual(data["timeframes"]["H4"]["latest_close_utc"], cutoff)
        self.assertLessEqual(data["timeframes"]["D1"]["latest_close_utc"], cutoff)
        self.assertFalse(data["wave_degree_expert_verified"])


if __name__ == "__main__":
    unittest.main()
