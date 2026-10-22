import unittest
from datetime import datetime,timezone,timedelta
from types import SimpleNamespace
from williams_ai.wave_evidence import alternating_impulse_skeleton,confirmed_price_swings,wave_context


def pivot(kind,price,i):
    return {"index":i,"confirmed_at_index":i+2,"type":kind,
            "price":price,"time":(datetime(2026,1,1,tzinfo=timezone.utc)+timedelta(hours=i)).isoformat()}


class WaveEvidenceTests(unittest.TestCase):
    def test_impulse_is_hypothesis_not_absolute(self):
        series=[pivot(k,v,i) for (k,v,i) in [
            ("L",100,2),("H",120,25),("L",110,40),
            ("H",145,69),("L",130,92),("H",165,125)
        ]]
        found=alternating_impulse_skeleton(series)
        self.assertEqual(len(found),1)
        hypothesis=found[0]
        self.assertEqual(hypothesis["status"],"HYPOTHESIS_NOT_BOOK_VERIFIED")
        self.assertEqual(hypothesis["pattern"],"IMPULSE_SKELETON_POSSIBLE")
        self.assertTrue(hypothesis["w3_not_shortest"])
        self.assertTrue(hypothesis["w4_nonoverlap"])
        self.assertTrue(hypothesis["rhythm_100_140_context"])
        self.assertFalse(wave_context([],[])["can_execute"])

    def test_overlapping_fourth_is_not_classic_impulse(self):
        series=[pivot(k,v,i) for (k,v,i) in [
            ("L",100,1),("H",120,20),("L",110,40),
            ("H",150,60),("L",118,80),("H",165,102)
        ]]
        found=alternating_impulse_skeleton(series)
        self.assertEqual(found[0]["pattern"],"DIAGONAL_OR_ALTERNATE_POSSIBLE")
        self.assertFalse(found[0]["w4_nonoverlap"])
        self.assertIn("W4 overlaps W1",found[0]["issues"][0])

    def test_no_out_of_order_or_incomplete_swings(self):
        start=datetime(2026,1,1,tzinfo=timezone.utc)
        rows=[SimpleNamespace(at=start+timedelta(hours=i),
              high=110-i*.02,low=100-i*.02) for i in range(40)]
        rows[-2].high=1000
        swings=confirmed_price_swings(rows)
        self.assertFalse(any(x["index"]==len(rows)-2 for x in swings))
        rows[-3].high=1100
        swings=confirmed_price_swings(rows)
        self.assertTrue(any(x["index"]==len(rows)-3 and
                            x["confirmed_at_index"]==len(rows)-1
                            for x in swings))

    def test_w4_ao_near_zero_is_only_observation(self):
        points=[pivot(k,v,i) for (k,v,i) in [
            ("L",100,2),("H",120,22),("L",110,42),
            ("H",145,62),("L",130,82),("H",160,122)
        ]]
        ao=[1.0]*140
        ao[50]=10
        ao[70]=0.05
        hypothesis=alternating_impulse_skeleton(points,ao)[0]
        self.assertTrue(hypothesis["w4_ao_near_zero_observed"])
        self.assertEqual(hypothesis["status"],"HYPOTHESIS_NOT_BOOK_VERIFIED")

    def test_reject_third_shortest_and_w2_origin(self):
        points=[pivot(k,v,i) for (k,v,i) in [
            ("L",100,2),("H",120,22),("L",98,42),
            ("H",121,62),("L",118,82),("H",160,122)
        ]]
        self.assertEqual(alternating_impulse_skeleton(points),[])


if __name__=="__main__":
    unittest.main()
