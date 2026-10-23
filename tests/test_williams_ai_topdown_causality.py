"""Proof tests for finished H4/D1 candles and source-scoped wave context."""
import unittest
from copy import deepcopy
from datetime import datetime, timedelta, timezone

from williams_ai.causal_timeframes import aggregate_closed_h1, top_down_market_evidence
from williams_ai.causal_paper_lab import frozen_markets


def fixture(n=1200):
    start=datetime(2026,1,1,0,59,59,999000,tzinfo=timezone.utc)
    result=[]
    for i in range(n):
        price=100+i/100
        result.append({
            "close_time":(start+timedelta(hours=i)).isoformat(),
            "open":price,"high":price+1,"low":price-1,
            "close":price+0.5,"volume":10,"closed":True
        })
    return result


class UTCHigherTimeframeTests(unittest.TestCase):
    def test_exact_completed_h4_and_d1_only(self):
        rows=fixture(25)
        four=aggregate_closed_h1(rows,"H4")
        day=aggregate_closed_h1(rows,"D1")
        self.assertEqual(len(four),6)
        self.assertEqual(len(day),1)
        self.assertEqual(four[0]["close_time"],rows[3]["close_time"])
        self.assertEqual(four[-1]["close_time"],rows[23]["close_time"])
        self.assertEqual(day[-1]["close_time"],rows[23]["close_time"])
        self.assertEqual(day[0]["open"],rows[0]["open"])
        self.assertEqual(day[0]["close"],rows[23]["close"])
        self.assertEqual(day[0]["volume"],240)
        self.assertEqual(day[0]["high"],max(x["high"] for x in rows[:24]))
        self.assertEqual(day[0]["low"],min(x["low"] for x in rows[:24]))

    def test_partial_first_and_last_window_never_promoted(self):
        rows=fixture(55)[2:52]
        four=aggregate_closed_h1(rows,"H4")
        day=aggregate_closed_h1(rows,"D1")
        self.assertEqual(len(day),1)
        self.assertEqual(day[0]["close_time"],fixture(55)[47]["close_time"])
        self.assertTrue(all(x["close_time"]<=rows[-1]["close_time"] for x in four+day))

    def test_future_cannot_rewrite_matured_higher_candles(self):
        rows=fixture(250)
        for tf in ("H4","D1"):
            earlier=aggregate_closed_h1(rows[:175],tf)
            later=aggregate_closed_h1(rows,tf)
            self.assertEqual(later[:len(earlier)],earlier)

    def test_reject_gap_open_hour_and_misalignment(self):
        rows=fixture(100)
        with self.assertRaisesRegex(ValueError,"Incomplete"):
            aggregate_closed_h1(rows[:50]+rows[51:],"H4")
        rows[20]["close_time"]="2026-01-01T20:59:00+00:00"
        with self.assertRaisesRegex(ValueError,"aligned"):
            aggregate_closed_h1(rows,"D1")

    def test_topdown_awaits_completed_d1_not_partial(self):
        rows=fixture(1200)
        at_1000=top_down_market_evidence("BTCUSDT",rows[:1000],tick_size=0.1)
        self.assertEqual(at_1000["timeframes"]["D1"]["status"],"INSUFFICIENT_COMPLETE_BARS")
        self.assertEqual(at_1000["wave_hierarchy_status"],"UNCONFIRMED_NESTED_DEGREES")
        later=top_down_market_evidence("BTCUSDT",rows,tick_size=0.1)
        self.assertEqual(later["timeframes"]["D1"]["status"],"OBSERVED_CLOSED_BARS")
        self.assertFalse(later["execution_allowed"])
        self.assertIsNone(later["trade_action"])

    def test_disjoint_second_market_window_no_overlap_and_no_orders(self):
        from williams_ai.causal_paper_lab import FROZEN_DATA,run_15
        from pathlib import Path
        older_path=Path(__file__).resolve().parents[1]/"examples/ai_binance_15pair_h1_prior_holdout_20260902.json"
        recent=frozen_markets(FROZEN_DATA)
        older=frozen_markets(older_path)
        self.assertEqual(set(recent),set(older))
        for symbol in recent:
            self.assertEqual(len(older[symbol]["bars"]),1200)
            self.assertLess(older[symbol]["bars"][-1]["close_time"],recent[symbol]["bars"][0]["close_time"])
        report=run_15(older_path)
        self.assertEqual(report["pairs"],15)
        self.assertEqual(len(report["results"]),45)
        self.assertFalse(report["execution_allowed"])
        for group in report["aggregate"].values():
            self.assertEqual(group["non_stop_exits"],0)
            self.assertGreater(group["closed_trades"],0)

    def test_all_fifteen_actual_markets_have_caused_context_not_orders(self):
        markets=frozen_markets()
        self.assertEqual(len(markets),15)
        for symbol,raw in markets.items():
            with self.subTest(symbol=symbol):
                result=top_down_market_evidence(symbol,raw["bars"],tick_size=raw["tick_size"])
                self.assertIn(result["higher_context"],{
                    "HIGHER_TIMEFRAMES_DIRECTIONALLY_ALIGNED",
                    "HIGHER_TIMEFRAMES_CONFLICT",
                    "HIGHER_DIRECTION_UNRESOLVED"})
                self.assertIsNone(result["trade_action"])
                self.assertEqual(result["timeframes"]["H1"]["closed_bars"],1200)
                self.assertLess(result["timeframes"]["H4"]["closed_bars"],1200)
                for tf in ("H1","H4","D1"):
                    row=result["timeframes"][tf]
                    self.assertLessEqual(row["closed_through"],result["as_of"])
                    self.assertEqual(row["wave_degree"],"UNKNOWN")
                    self.assertIsNone(row["trade_action"])


if __name__=="__main__":
    unittest.main()
