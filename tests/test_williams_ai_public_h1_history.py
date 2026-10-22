"""Real *public* Binance USD-M H1 candles, frozen before the test date.

An audit of causal indicators and signal observability, NOT a simulated
Binance execution or proof of trading profitability. No network in CI.
"""
import json
import unittest
from datetime import datetime, timezone
from pathlib import Path

from williams_ai.chart_reader import (
    analyze_chart, indicators, validate_closed_bars,
)
from williams_ai.replay import replay_closed_chart


FIXTURE = Path(__file__).resolve().parents[1] / "examples" / "ai_public_h1_20261022_3pairs.json"


def public_bars(symbol):
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert data["schema"] == "BINANCE_USDM_CLOSED_KLINES_V1"
    return [{
        "close_time": datetime.fromtimestamp(r[0]/1000, timezone.utc).isoformat(),
        "open": r[1], "high": r[2], "low": r[3], "close": r[4],
        "volume": r[5], "closed": True,
    } for r in data["markets"][symbol]]


class PublicHistoricalEvidenceTests(unittest.TestCase):
    SYMBOLS = ("ETHUSDT", "SOLUSDT", "XRPUSDT")

    def test_all_three_real_pairs_causal_math(self):
        for symbol in self.SYMBOLS:
            with self.subTest(symbol=symbol):
                rows = public_bars(symbol)
                self.assertEqual(len(rows), 240)
                bars, gaps = validate_closed_bars(rows,"H1",rows[-1]["close_time"])
                self.assertEqual(gaps, 0)
                lines = indicators(bars)
                # Independently recompute latest AO from median-price moving means.
                med = [(b.high + b.low)/2 for b in bars]
                ao_last = sum(med[-5:])/5 - sum(med[-34:])/34
                self.assertAlmostEqual(lines["ao"][-1], ao_last, places=7)
                self.assertAlmostEqual(
                    lines["ac"][-1],
                    ao_last - sum(lines["ao"][-5:])/5,
                    places=7,
                )
                chart = analyze_chart(symbol,"H1",rows,
                                      as_of=rows[-1]["close_time"],tick_size=0.01)
                self.assertFalse(chart["execution_allowed"])
                self.assertEqual(chart["bars_used"], 240)
                self.assertEqual(chart["gaps"], 0)
                for f in chart["candidates"]:
                    if f["type"] != "WM3_FRACTAL":
                        continue
                    self.assertLessEqual(f["confirmed_at"], rows[-1]["close_time"])
                    self.assertLess(f["pivot_time"], f["confirmed_at"])
                    self.assertFalse(f["can_prove_exchange_fill"])
                    if f["price_touch_observed_at"] is not None:
                        self.assertGreater(f["price_touch_observed_at"], f["confirmed_at"])

    def test_historical_future_cannot_rewrite_known_events(self):
        # Force a window including 19 October 2026, but limit cost on CI.
        for symbol in self.SYMBOLS:
            with self.subTest(symbol=symbol):
                rows=public_bars(symbol)[-150:]
                first=replay_closed_chart(symbol,"H1",rows[:92],tick_size=0.01)
                later=replay_closed_chart(symbol,"H1",rows[:112],tick_size=0.01)
                until=rows[91]["close_time"]
                prior=[e for e in later["events"] if e["observed_at"]<=until]
                self.assertEqual(first["events"],prior)
                self.assertTrue(first["no_lookahead"])
                self.assertIsNone(later["trade_action"])
                self.assertTrue(all(e["trade_action"] is None for e in later["events"]))

    def test_historical_signals_do_not_prove_fills(self):
        results = {}
        for symbol in self.SYMBOLS:
            rows=public_bars(symbol)
            view=analyze_chart(symbol,"H1",rows,
                               as_of=rows[-1]["close_time"],tick_size=0.01)
            results[symbol]={
                "fractal_candidates_recent":sum(f["type"]=="WM3_FRACTAL"
                                                for f in view["candidates"]),
                "touches_not_fills":sum(f.get("price_touch_observed_at") is not None
                                       for f in view["candidates"]
                                       if f["type"]=="WM3_FRACTAL"),
            }
        print("Frozen public history audit counts (NOT trades):",results)
        self.assertEqual(len(results), 3)


if __name__ == "__main__":
    unittest.main()
