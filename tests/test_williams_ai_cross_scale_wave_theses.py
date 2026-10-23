"""Causal, never-certain competing Elliott hypotheses across H1/H4/D1."""
import unittest

from williams_ai.cross_scale_wave_theses import cross_scale_wave_theses
from williams_ai.reasoning_trial import evaluate_historical_thought
from williams_ai.causal_paper_lab import frozen_markets

NOW="2026-10-10T12:00:00+00:00"
PAST="2026-10-10T08:00:00+00:00"


def frame(trend,at,price=100.0,ao=1.0):
    return {
        "status":"OBSERVED_CLOSED_BARS",
        "closed_through":at,"market_context":trend,
        "ao":ao,"ac":.1,"recent_levels":{
            "latest_confirmed_high":price*1.1,
            "latest_confirmed_low":price*.9,
            "close":price,
        },
    }


class WaveThesisTests(unittest.TestCase):
    def test_child_countertrend_is_multiple_hypotheses_not_wave_four(self):
        frames={
            "D1":frame("DOWN_CONTEXT",PAST,ao=-3),
            "H4":frame("DOWN_CONTEXT",PAST,ao=-1),
            "H1":frame("UP_CONTEXT",NOW,ao=1),
        }
        result=cross_scale_wave_theses(frames,as_of=NOW)
        relation=result["parents_and_children"][1]
        self.assertEqual(relation["parent"],"H4")
        self.assertEqual(relation["child"],"H1")
        self.assertEqual(len(relation["alternatives"]),3)
        self.assertIn("CHILD_CORRECTION_WITHIN_PARENT_MOVEMENT",
                      [x["hypothesis"] for x in relation["alternatives"]])
        self.assertIn("EARLY_REVERSAL_OF_PARENT_MOVEMENT",
                      [x["hypothesis"] for x in relation["alternatives"]])
        self.assertFalse(relation["verified_wave_degree"])
        self.assertIsNone(relation["alternatives"][0]["probability"])
        self.assertFalse(result["execution_allowed"])
        self.assertIsNone(result["trade_action"])
        self.assertFalse(result["fourth_wave_ao_near_zero_certified"])

    def test_aligned_parent_is_not_proven_final_wave(self):
        aligned={tf:frame("UP_CONTEXT",PAST if tf!="H1" else NOW,ao=2)
                 for tf in ("D1","H4","H1")}
        result=cross_scale_wave_theses(aligned,as_of=NOW)
        for v in result["parents_and_children"]:
            self.assertIn("PARENT_CHILD_TREND_CONTINUATION",
                          [x["hypothesis"] for x in v["alternatives"]])
            self.assertFalse(v["verified_wave_degree"])
        self.assertFalse(result["fifth_wave_certified"])

    def test_future_parent_is_hard_rejected_not_classified(self):
        frames={
            "D1":frame("DOWN_CONTEXT","2026-10-11T00:00:00+00:00"),
            "H4":frame("UP_CONTEXT",PAST),
            "H1":frame("UP_CONTEXT",NOW),
        }
        with self.assertRaisesRegex(ValueError,"FUTURE_TIMEFRAME_CANDLE"):
            cross_scale_wave_theses(frames,as_of=NOW)

    def test_missing_higher_tf_yields_unknown_not_guess(self):
        frames={"D1":{"status":"INSUFFICIENT_CLOSED_HISTORY"},
                "H4":frame("DOWN_CONTEXT",PAST),
                "H1":frame("UP_CONTEXT",NOW)}
        result=cross_scale_wave_theses(frames,as_of=NOW)
        self.assertEqual(result["parents_and_children"][0]["alternatives"],[])
        self.assertEqual(result["parents_and_children"][1]["status"],
                         "CONDITIONAL_CAUSAL_WAVE_ALTERNATIVES")

    def test_realtime_causal_historical_topdown_passes_unverified_theses(self):
        market=frozen_markets()["BTCUSDT"]
        rows=market["bars"]
        i=1090
        cutoff=rows[i]["close_time"]
        proposal={
            "symbol":"BTCUSDT","as_of":cutoff,"edition":"TC2",
            "setup":"NO_SETUP","side":"NONE","entry_state":"NO_SETUP",
            "exit_basis":"NONE","rule_ids":[],
            "trade_action":None,"execution_allowed":False,
        }
        result=evaluate_historical_thought(
            "BTCUSDT",rows,tick_size=market["tick_size"],
            bar_index=i,edition="TC2",proposal=proposal)
        top=result["top_down_context"]
        relation=top["cross_scale_wave_theses"]
        self.assertFalse(relation["execution_allowed"])
        self.assertEqual(relation["as_of"],cutoff)
        self.assertFalse(relation["fifth_wave_certified"])
        self.assertEqual(len(relation["parents_and_children"]),2)
        for pair in relation["parents_and_children"]:
            self.assertFalse(pair["execution_allowed"])
            if pair["status"]=="CONDITIONAL_CAUSAL_WAVE_ALTERNATIVES":
                self.assertLessEqual(pair["parent_closed_through"],cutoff)
                for x in pair["alternatives"]:
                    self.assertIsNone(x["trade_action"])
        self.assertFalse(result["model_saw_future_bars"])


if __name__=="__main__":
    unittest.main()
