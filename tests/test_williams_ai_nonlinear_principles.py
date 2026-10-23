"""Principles, NOT frozen textbook-chart images, govern market interpretation.

This verifies same underlying market relationships under radically different
absolute coin prices, different nonlinear wave shapes, multiple pivot scales,
late confirmations, and a real 15-market chronological candle dataset.

A deterministic scenario critic is NOT a proof of GGUF intelligence or
profitable price forecasting. Every output is read-only.
"""
import math
import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from williams_ai.chart_reader import analyze_chart
from williams_ai.causal_paper_lab import frozen_markets
from williams_ai.nonlinear_market_principles import (
    _swings, nonlinear_market_principles, SCALES,
)


def nonlinear_fixture(n=240, *,period_fast=11.0,period_slow=45.0,trend=.035):
    start=datetime(2026,1,1,0,59,59,999000,tzinfo=timezone.utc)
    rows=[]
    for i in range(n):
        # Two nested, NONLINEAR oscillation lengths. There is no Elliott
        # chart picture, no fixed wave count, and different periods produce
        # entirely different visual patterns.
        base=(110+trend*i+2.5*math.sin(i*2*math.pi/period_slow)
              +.65*math.sin(i*2*math.pi/period_fast))
        op=base-0.05
        close=base+0.05
        rows.append({
            "close_time":(start+timedelta(hours=i)).isoformat(),
            "open":op,"high":base+.5,"low":base-.5,
            "close":close,"volume":10+(i%7),"closed":True,
        })
    return rows


class NonlinearPrinciplesTests(unittest.TestCase):
    def test_multi_scale_pivots_wait_for_right_closed_bars(self):
        start=datetime(2026,1,1,tzinfo=timezone.utc)
        bars=[]
        for i in range(23):
            bars.append(SimpleNamespace(
                at=start+timedelta(hours=i),
                high=20.0 if i==10 else 10.0,
                low=1.0+(i%3)*.2,
                close=8.0,
            ))
        for name,radius in SCALES:
            with self.subTest(name=name):
                # With fewer than radius bars after pivot, no observation.
                early=_swings(bars[:10+radius],radius)
                confirmed=_swings(bars[:11+radius],radius)
                self.assertFalse(any(p["index"]==10 for p in early))
                self.assertTrue(any(
                    p["index"]==10
                    and p["type"]=="H"
                    and p["confirmed_at_index"]==10+radius
                    for p in confirmed
                ))

    def test_market_geometry_is_relative_not_exact_printed_picture(self):
        for fast,slow in ((7.,27.),(13.,53.),(19.,87.)):
            with self.subTest(fast=fast,slow=slow):
                rows=nonlinear_fixture(250,period_fast=fast,period_slow=slow)
                o=analyze_chart("BTCUSDT","H1",rows,as_of=rows[-1]["close_time"],
                                tick_size=.01)
                r=o["principle_evidence"]
                self.assertFalse(r["hardcoded_wave_picture_matching"])
                self.assertTrue(r["no_fixed_100_140_bar_requirement"])
                self.assertEqual(r["mode"],"PRINCIPLES_NOT_PICTURE_TEMPLATES")
                self.assertEqual(r["wave_degree"],"UNKNOWN")
                self.assertEqual(len(r["competing_market_scenarios"]),5)
                self.assertFalse(r["execution_allowed"])
                self.assertIsNone(r["trade_action"])
                self.assertTrue(all(
                    s["probability"] is None and
                    s["trade_action"] is None and
                    s["verified_elliott_degree"] is False
                    for s in r["competing_market_scenarios"]
                ))
                self.assertEqual(set(r["observable_facts"]["multi_resolution_price_structure"]),
                                 {"LOCAL","INTERMEDIATE","BROAD"})

    def test_positive_price_scaling_cannot_change_market_principles(self):
        original=nonlinear_fixture(240,period_fast=9.,period_slow=47.)
        base=analyze_chart("ETHUSDT","H1",original,as_of=original[-1]["close_time"],
                           tick_size=.01)
        multiple=17.5
        changed=[{
            **row,
            **{key:row[key]*multiple for key in ("open","high","low","close")}
        } for row in original]
        scaled=analyze_chart("ETHUSDT","H1",changed,
                             as_of=changed[-1]["close_time"],
                             tick_size=.01*multiple)
        a=base["principle_evidence"]
        b=scaled["principle_evidence"]
        self.assertEqual(a["observable_facts"]["multi_resolution_price_structure"],
                         b["observable_facts"]["multi_resolution_price_structure"])
        self.assertEqual(a["observable_facts"]["momentum"],
                         b["observable_facts"]["momentum"])
        self.assertEqual(a["observable_facts"]["volatility"]["state"],
                         b["observable_facts"]["volatility"]["state"])
        self.assertEqual(
            [(x["hypothesis"],x["status"],x["supporting_observations"])
             for x in a["competing_market_scenarios"]],
            [(x["hypothesis"],x["status"],x["supporting_observations"])
             for x in b["competing_market_scenarios"]],
        )
        self.assertEqual(a["as_of"],b["as_of"])

    def test_closed_prefix_will_not_consult_future_bar(self):
        rows=nonlinear_fixture(270,trend=-.01)
        prefix=rows[:220]
        baseline=analyze_chart("SOLUSDT","H1",prefix,
                               as_of=prefix[-1]["close_time"],tick_size=.01)
        manipulated=rows[:]
        # Deliberately dramatic FUTURE decline after the decision timestamp
        for idx in range(220,270):
            manipulated[idx]={**manipulated[idx],
                              "open":1.0,"high":1.2,"low":.7,"close":.9}
        again=analyze_chart("SOLUSDT","H1",manipulated[:220],
                            as_of=prefix[-1]["close_time"],tick_size=.01)
        self.assertEqual(baseline["principle_evidence"],again["principle_evidence"])
        with self.assertRaisesRegex(ValueError,"NO_LOOKAHEAD"):
            analyze_chart("SOLUSDT","H1",manipulated,
                          as_of=prefix[-1]["close_time"],tick_size=.01)

    def test_near_zero_ao_not_proof_of_wave_four(self):
        rows=nonlinear_fixture(180)
        from williams_ai.chart_reader import validate_closed_bars
        candles,gaps=validate_closed_bars(rows,"H1",rows[-1]["close_time"])
        self.assertEqual(gaps,0)
        ao=[1.0]*180
        ao[-1]=0.001
        ac=[0.0]*180
        r=nonlinear_market_principles(
            candles,ao,ac,alligator_context="MIXED_OR_TRANSITION")
        momentum=r["observable_facts"]["momentum"]
        self.assertTrue(momentum["ao_near_zero_relative_to_recent_amplitude"])
        self.assertEqual(momentum["wave_four_claim"],"NOT_PROVEN_BY_AO_ZERO")
        self.assertTrue(any(
            "AO_NEAR_ZERO_RELATIVE_TO_RECENT_SWING" in s["supporting_observations"]
            for s in r["competing_market_scenarios"]
        ))
        self.assertEqual(r["wave_degree"],"UNKNOWN")
        with self.assertRaises(ValueError):
            nonlinear_market_principles(candles,ao[:-1],ac)

    def test_opposing_divergences_remain_alternatives_not_orders(self):
        rows=nonlinear_fixture(160)
        from williams_ai.chart_reader import validate_closed_bars
        candles,_=validate_closed_bars(rows,"H1",rows[-1]["close_time"])
        src={"divergences":[
            {"kind":"BEARISH_AO_DIVERGENCE_CANDIDATE"},
            {"kind":"BULLISH_AO_DIVERGENCE_CANDIDATE"},
        ]}
        r=nonlinear_market_principles(candles,[2.0]*160,[.1]*160,
                                      diverging_evidence=src)
        pair={s["hypothesis"]:s for s in r["competing_market_scenarios"]}
        self.assertIn("AO_BULLISH_DIVERGENCE",pair["POSSIBLE_REVERSAL_UP"]["supporting_observations"])
        self.assertIn("AO_BEARISH_DIVERGENCE",pair["POSSIBLE_REVERSAL_DOWN"]["supporting_observations"])
        self.assertFalse(r["automatic_entry_or_exit_allowed"])

    def test_all_15_real_markets_support_nonlinear_explanation_without_orders(self):
        markets=frozen_markets()
        self.assertEqual(len(markets),15)
        seen=set()
        for name,market in markets.items():
            with self.subTest(symbol=name):
                bars=market["bars"]
                chart=analyze_chart(name,"H1",bars,
                                    as_of=bars[-1]["close_time"],
                                    tick_size=market["tick_size"])
                principles=chart["principle_evidence"]
                seen.add(tuple(principles["observable_facts"]
                               ["multi_resolution_price_structure"].values()))
                self.assertEqual(principles["as_of"],chart["as_of"])
                self.assertEqual(principles["wave_degree"],"UNKNOWN")
                self.assertFalse(principles["execution_allowed"])
                self.assertIsNone(principles["trade_action"])
                for scale in principles["observable_facts"]["scale_details"].values():
                    self.assertLessEqual(len(scale["confirmed_swings"]),8)
                    self.assertTrue(all(
                        p["confirmed_at"]<=chart["as_of"]
                        for p in scale["confirmed_swings"]
                    ))
        # Real market morphologies need not produce one same template.
        self.assertGreaterEqual(len(seen),2)


if __name__=="__main__":
    unittest.main()
