import unittest
from datetime import datetime,timezone,timedelta
from types import SimpleNamespace
from williams_ai.market_position import market_position


def bars(highs, lows):
    start = datetime(2026, 10, 1, tzinfo=timezone.utc)
    return [SimpleNamespace(
        high=h,low=l,close=(h+l)/2,
        at=start+timedelta(hours=i)
    ) for i,(h,l) in enumerate(zip(highs,lows))]


class MarketPositionTests(unittest.TestCase):
    def test_only_confirmed_pivots_can_be_used(self):
        hi=[20.0]*30
        lo=[10.0]*30
        for index,value in [(5,28),(13,30),(21,33),(28,50)]:
            hi[index]=value
        for index,value in [(9,1),(17,3),(25,5)]:
            lo[index]=value
        sequence=bars(hi,lo)
        result=market_position(sequence,[0.0]*30)
        # 28 high has only one bar on its right and is NOT confirmed.
        self.assertEqual(result["recent_levels"]["latest_confirmed_high"],33)
        self.assertEqual(result["structure"],"HIGHER_HIGHS_HIGHER_LOWS")

    def test_divergence_does_not_allow_orders(self):
        hi=[20.0]*30
        lo=[10.0]*30
        hi[7]=28
        hi[16]=35
        lo[12]=5
        lo[23]=6
        ao=[1.0]*30
        ao[7]=8.0
        ao[16]=3.0
        view=market_position(bars(hi,lo),ao)
        self.assertTrue(any(x["kind"]=="BEARISH_AO_DIVERGENCE_CANDIDATE"
                            for x in view["divergences"]))
        self.assertIsNone(view["trade_action"])

    def test_unknown_without_swings(self):
        x=market_position(bars([20.0]*10,[10.0]*10),[0.1]*10)
        self.assertEqual(x["structure"],"UNRESOLVED")
        self.assertEqual(x["recent_levels"]["relative_to_known_swing"],"UNKNOWN")

    def test_misalignment_rejected(self):
        with self.assertRaises(ValueError):
            market_position(bars([20.0]*10,[10.0]*10),[1.0])

if __name__=="__main__":
    unittest.main()
