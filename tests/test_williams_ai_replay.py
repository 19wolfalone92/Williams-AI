import unittest
from datetime import datetime, timezone, timedelta
from williams_ai.replay import replay_closed_chart


def candles(n=54):
    origin = datetime(2026, 10, 1, tzinfo=timezone.utc)
    rows = []
    for i in range(n):
        price = 200 + i * 0.01
        rows.append({
            "close_time": (origin + timedelta(hours=i)).isoformat(),
            "open": price, "high": price + 0.5, "low": price - 0.5,
            "close": price + 0.1, "volume": 4.0, "closed": True,
        })
    return rows


class NoLookAheadReplayTests(unittest.TestCase):
    def test_no_event_predates_its_confirmation_candle(self):
        bars = candles()
        bars[48]["high"] += 5  # pivot cannot be seen before index 50
        out = replay_closed_chart("ETHUSDT", "H1", bars)
        self.assertFalse(out["trade_action"])
        self.assertTrue(out["no_lookahead"])
        for event in out["events"]:
            self.assertLessEqual(event["observable_earliest_at"], event["observed_at"])
            self.assertIsNone(event["trade_action"])
        specific = [x for x in out["events"] if x["type"] == "WM3_FRACTAL"
                    and bars[48]["close_time"] in x["signal_key"]]
        self.assertEqual(specific[0]["observed_at"], bars[50]["close_time"])

    def test_appending_future_bars_cannot_change_past_events(self):
        bars = candles(58)
        bars[48]["high"] += 5
        first = replay_closed_chart("ETHUSDT", "H1", bars[:54])
        all_ticks = replay_closed_chart("ETHUSDT", "H1", bars)
        prior = [x for x in all_ticks["events"] if x["observed_at"] <= bars[53]["close_time"]]
        self.assertEqual(first["events"], prior)

    def test_invalid_history_rejected(self):
        bars = candles(52)
        bars[30]["closed"] = False
        with self.assertRaisesRegex(ValueError, "finalized"):
            replay_closed_chart("ETHUSDT", "H1", bars)
        bars = candles(52)
        bars.pop(25)
        with self.assertRaisesRegex(ValueError, "gaps"):
            replay_closed_chart("ETHUSDT", "H1", bars)

    def test_at_least_45_bars(self):
        with self.assertRaisesRegex(ValueError, "45"):
            replay_closed_chart("ETHUSDT", "H1", candles(44))


if __name__ == "__main__":
    unittest.main()
