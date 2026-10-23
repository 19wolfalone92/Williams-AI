"""Regression tests for REAL-event extraction semantics (fixtures are synthetic).

Synthetic bars are ONLY for no-lookahead and stability contracts; the
separate GitHub Actions job evaluates actual checksum-verified Binance OHLC.
"""
from __future__ import annotations

import unittest
from datetime import datetime,timedelta,timezone
from math import sin,pi

from williams_ai.real_wiseman_replay import replay_closed_history,score_january_to_february


def candles(*,hours:int=260,month:int=1,period:float=21.,
            offset:float=0.)->list[dict]:
    start=datetime(2025,month,1,tzinfo=timezone.utc)
    out=[]
    prev=100.+offset
    for t in range(hours):
        current=100.+offset+.015*t+2.*sin(2*pi*t/period)+.3*sin(2*pi*t/7.)
        lo=min(prev,current)-.18
        hi=max(prev,current)+.18
        out.append({
            "close_time":(start+timedelta(hours=t+1)
                          -timedelta(milliseconds=1)).isoformat(),
            "open":prev,"high":hi,"low":lo,"close":current,
            "volume":1000.,"closed":True,
        })
        prev=current
    return out


class RealClosedWiseManEventTests(unittest.TestCase):
    def test_h1_event_proofs_are_closed_and_have_no_self_fulfilling_forward_features(self):
        sample=candles()
        events=replay_closed_history(sample,symbol="BTCUSDT")
        self.assertGreater(len(events),0)
        keys=set()
        for e in events:
            self.assertLess(e["decision_index"],e["resolved_index"])
            self.assertLess(e["decision_at"],e["resolved_at"])
            self.assertEqual(e["resolved_index"]-e["decision_index"],12)
            self.assertIn(e["kind"],("WM1","WM2","WM3"))
            self.assertEqual(len(e["context_features"]),6)
            self.assertFalse(e["future_was_available_to_decision"])
            self.assertFalse(e["true_williams_entry_validated"])
            self.assertIsNone(e["trade_action"])
            self.assertFalse(e["execution_allowed"])
            self.assertNotIn(e["event_id"],keys)
            keys.add(e["event_id"])

    def test_changing_future_candles_cannot_rewrite_earlier_event_detection(self):
        old=candles()
        a=replay_closed_history(old,symbol="ETHUSDT")
        altered=candles()
        for i in range(210,260):
            # Dramatic but VALID alternative future path; all candles <=209
            # are byte-identical and were already finalized.
            previous_close=altered[i-1]["close"] if i>210 else old[209]["close"]
            cl=max(.3, previous_close*.97)
            altered[i]["open"]=previous_close
            altered[i]["high"]=max(cl,previous_close)*1.005
            altered[i]["low"]=min(cl,previous_close)*.995
            altered[i]["close"]=cl
        b=replay_closed_history(altered,symbol="ETHUSDT")
        fingerprint=lambda records:[
            (v["event_id"],v["decision_at"],v["kind"],v["side"],
             v["context_features"])
            for v in records if v["decision_index"]<200
        ]
        self.assertEqual(fingerprint(a),fingerprint(b))

    def test_m5_not_synthesized_from_h1(self):
        with self.assertRaisesRegex(ValueError,"REAL H1"):
            replay_closed_history(candles(),symbol="BTCUSDT",tf="M5")

    def test_invalid_horizon_or_future_candle_rejected(self):
        source=candles()
        with self.assertRaises(ValueError):
            replay_closed_history(source,symbol="BTCUSDT",horizon=0)
        source[-1]["closed"]=False
        with self.assertRaises(ValueError):
            replay_closed_history(source,symbol="BTCUSDT")

    def test_training_gate_requires_multiple_authentic_market_sources(self):
        with self.assertRaisesRegex(ValueError,"Not enough genuine"):
            score_january_to_february({"BTCUSDT":candles(hours=400)})


if __name__=="__main__":
    unittest.main()
