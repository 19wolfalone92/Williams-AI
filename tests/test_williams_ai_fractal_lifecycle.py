"""Causal, read-only acceptance tests for TC2 confirmed-fractal lifecycle."""
import unittest
from datetime import datetime, timedelta, timezone

from williams_ai.chart_reader import _confirmed_fractals, analyze_chart, validate_closed_bars
from williams_ai.replay import replay_closed_chart


def make_rows(n=90):
    start = datetime(2026, 9, 1, tzinfo=timezone.utc)
    return [{
        "close_time": (start + timedelta(hours=i)).isoformat(),
        "open": 100.0, "high": 101.0, "low": 99.0,
        "close": 100.0, "volume": 200, "closed": True
    } for i in range(n)]


def lifecycle(rows, teeth):
    bars, gaps = validate_closed_bars(rows, "H1", rows[-1]["close_time"])
    assert gaps == 0
    items = _confirmed_fractals(
        bars, {"teeth": teeth}, tick_size=0.1)
    return items


def select(items, index, side):
    return next(i for i in items if i["pivot_time"] == make_rows()[index]["close_time"]
                and i["side"] == side)


class FractalLifecycleProofTests(unittest.TestCase):
    def test_must_wait_for_two_closed_confirmation_bars(self):
        rows = make_rows(69)
        rows[66]["high"] = 110
        # Only ONE right closed bar at i=67.
        self.assertFalse(any(x["pivot_time"] == rows[66]["close_time"]
                             for x in lifecycle(rows[:68], [90]*68)))
        proof = select(lifecycle(rows[:69], [90]*69), 66, "LONG")
        self.assertEqual(proof["confirmed_at"], rows[68]["close_time"])
        self.assertEqual(proof["state"], "AWAITING_PRICE_TRIGGER")
        self.assertFalse(proof["can_prove_exchange_fill"])

    def test_teeth_at_price_touch_not_at_formation(self):
        rows = make_rows()
        rows[66]["high"] = 110
        rows[70]["high"] = 112  # touch 110.1 after confirmation at 68
        teeth = [90.0] * len(rows)
        teeth[68] = 115  # invalid at confirmation; valid at trigger
        got = select(lifecycle(rows, teeth), 66, "LONG")
        self.assertFalse(got["teeth_filter_at_confirmation"])
        self.assertTrue(got["teeth_filter_at_trigger"])
        self.assertEqual(got["price_touch_observed_at"], rows[70]["close_time"])
        self.assertEqual(got["state"], "CLOSED_BAR_TOUCH_PASSES_TEETH_NOT_EXCHANGE_FILL")
        self.assertFalse(got["can_prove_exchange_fill"])

    def test_opposite_trigger_filter_blocks_invalid_long(self):
        rows = make_rows()
        rows[66]["high"] = 110
        rows[70]["high"] = 112
        teeth = [90.0] * len(rows)
        teeth[70] = 115  # trigger below Teeth => not a valid TC2 breakout
        got = select(lifecycle(rows, teeth), 66, "LONG")
        self.assertTrue(got["teeth_filter_at_confirmation"])
        self.assertFalse(got["teeth_filter_at_trigger"])
        self.assertEqual(got["state"], "CLOSED_BAR_TOUCH_REJECTED_BY_TEETH")

    def test_short_side_uses_mirrored_trigger_and_teeth(self):
        rows = make_rows()
        rows[65]["low"] = 90
        rows[69]["low"] = 88
        teeth = [120.0] * len(rows)
        teeth[69] = 89.5  # at trigger 89.9 > 89.5, invalid short
        got = select(lifecycle(rows, teeth), 65, "SHORT")
        self.assertEqual(got["entry_trigger"], 89.9)
        self.assertEqual(got["state"], "CLOSED_BAR_TOUCH_REJECTED_BY_TEETH")

    def test_new_confirmed_fractal_supersedes_old_pending_level(self):
        rows = make_rows()
        rows[60]["high"] = 110
        rows[68]["high"] = 108
        items = lifecycle(rows, [90.0] * len(rows))
        old = select(items, 60, "LONG")
        new = select(items, 68, "LONG")
        self.assertEqual(old["state"], "SUPERSEDED_BY_NEWER_CONFIRMED_FRACTAL")
        self.assertEqual(old["newer_fractal_confirmed_at"], rows[70]["close_time"])
        self.assertEqual(new["state"], "AWAITING_PRICE_TRIGGER")

    def test_outside_bar_never_silently_assumes_long(self):
        rows = make_rows()
        rows[65]["high"] = 111
        rows[65]["low"] = 89
        items = lifecycle(rows, [100.0]*len(rows))
        for side in ("LONG","SHORT"):
            candidate = select(items, 65, side)
            self.assertEqual(candidate["state"], "AMBIGUOUS_OUTSIDE_BAR")
            self.assertIsNone(candidate["price_touch_observed_at"])

    def test_read_only_chart_and_causal_replay(self):
        rows = make_rows()
        rows[65]["high"] = 110
        rows[72]["high"] = 112
        early = replay_closed_chart("ETHUSDT","H1",rows[:70],tick_size=0.1)
        late = replay_closed_chart("ETHUSDT","H1",rows[:77],tick_size=0.1)
        prior = [e for e in late["events"] if e["observed_at"]<=rows[69]["close_time"]]
        self.assertEqual(early["events"],prior)
        touching = [e for e in late["events"] if e["type"]=="WM3_FRACTAL"
                    and e["side"]=="LONG" and e["state"]=="CLOSED_BAR_TOUCH_PASSES_TEETH_NOT_EXCHANGE_FILL"]
        self.assertTrue(touching)
        self.assertEqual(touching[0]["observed_at"], rows[72]["close_time"])
        self.assertIsNone(touching[0]["trade_action"])
        result=analyze_chart("ETHUSDT","H1",rows,as_of=rows[-1]["close_time"],tick_size=0.1)
        self.assertFalse(result["execution_allowed"])


if __name__ == "__main__":
    unittest.main()
