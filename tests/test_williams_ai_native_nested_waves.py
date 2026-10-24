"""Multiple Binance TF wave hypotheses must be confirmed and future-blind."""
import unittest
from datetime import datetime,timezone
from williams_ai.native_nested_waves import (
    cross_scale_pivot_evidence,nested_evidence_for_views,
)

ASOF=int(datetime(2026,10,23,tzinfo=timezone.utc).timestamp()*1000)
def pivot(stamp,typ,index):
    return {"time":stamp,"type":typ,"price":100+index,
            "confirmed_at_index":index}

def view(stamp,swings):
    return {"last_closed_at":stamp,"bars_used":80,
            "wave_structure":{"recent_confirmed_swings":swings}}

class NestedWavesProofOnly(unittest.TestCase):
    def setUp(self):
        self.parent=view("2026-10-22T23:59:59+00:00",[
            pivot("2026-10-20T00:00:00+00:00","L",7),
            pivot("2026-10-22T00:00:00+00:00","H",40)])
        self.child=view("2026-10-22T23:59:59+00:00",[
            pivot("2026-10-20T04:00:00+00:00","H",8),
            pivot("2026-10-20T10:00:00+00:00","L",14),
            pivot("2026-10-20T17:00:00+00:00","H",20),
            pivot("2026-10-21T21:00:00+00:00","L",26)])

    def test_confirmed_subpivots_are_not_declared_as_wave_five(self):
        q=cross_scale_pivot_evidence(
            self.parent,self.child,parent_tf="4h",child_tf="1h",
            common_as_of_ms=ASOF)
        self.assertEqual(q["status"],"POTENTIAL_CHILD_SUBDIVISION_HYPOTHESIS")
        self.assertEqual(q["confirmed_child_pivots_inside"],4)
        self.assertEqual(q["candidate_parent_leg"]["direction"],"UP")
        self.assertFalse(q["parent_leg_is_a_confirmed_elliott_degree"])
        self.assertFalse(q["child_waves_are_confirmed_elliott_subwaves"])
        self.assertIsNone(q["verified_nested_wave_label"])
        self.assertIsNone(q["trade_action"])
        self.assertFalse(q["execution_allowed"])

    def test_shared_h1_cutoff_rejects_future_tf_or_pivot(self):
        x=dict(self.child)
        x["last_closed_at"]="2026-10-24T02:00:00+00:00"
        with self.assertRaisesRegex(ValueError,"LOOKAHEAD"):
            cross_scale_pivot_evidence(
                self.parent,x,parent_tf="4h",child_tf="1h",
                common_as_of_ms=ASOF)
        x=view("2026-10-22T23:59:59+00:00",[
            pivot("2026-10-24T00:00:00+00:00","H",70)])
        with self.assertRaisesRegex(ValueError,"FUTURE_EXTREMUM"):
            cross_scale_pivot_evidence(
                self.parent,x,parent_tf="4h",child_tf="1h",
                common_as_of_ms=ASOF)
        x=view("2026-10-22T23:59:59+00:00",[
            pivot("2026-10-21T12:00:00+00:00","H",81)])
        with self.assertRaisesRegex(ValueError,"UNCONFIRMED_EXTREMUM"):
            cross_scale_pivot_evidence(
                self.parent,x,parent_tf="4h",child_tf="1h",
                common_as_of_ms=ASOF)

    def test_no_pivots_is_not_an_invented_elliott_count(self):
        q=cross_scale_pivot_evidence(
            self.parent,None,parent_tf="4h",child_tf="1h",
            common_as_of_ms=ASOF)
        self.assertEqual(q["status"],"INSUFFICIENT_MATCHED_CONFIRMED_SWINGS")
        self.assertIsNone(q["candidate_parent_leg"])
        self.assertFalse(q["execution_allowed"])
        result=nested_evidence_for_views(
            {"4h":self.parent,"1h":self.child},
            intervals=["1h","4h"],as_of_ms=ASOF)
        self.assertEqual(len(result),1)
        self.assertEqual(result[0]["parent"],"4h")
        self.assertEqual(result[0]["child"],"1h")
        self.assertEqual(result[0]["status"],
                         "POTENTIAL_CHILD_SUBDIVISION_HYPOTHESIS")

if __name__=="__main__":
    unittest.main()
