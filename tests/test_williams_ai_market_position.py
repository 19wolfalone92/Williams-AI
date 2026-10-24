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
        lo[3]=4
        lo[12]=5
        lo[23]=6
        ao=[1.0]*30
        ao[7]=8.0
        ao[16]=3.0
        view=market_position(bars(hi,lo),ao)
        self.assertTrue(any(x["kind"]=="BEARISH_AO_DIVERGENCE_CANDIDATE"
                            for x in view["divergences"]))
        self.assertIsNone(view["trade_action"])

    def test_ao_peak_can_precede_high_pivot(self):
        hi=[20.0]*36
        lo=[10.0]*36
        for i,v in ((3,4),(12,5),(22,6),(31,7)):
            lo[i]=v
        hi[7]=30
        hi[17]=35
        hi[27]=33  # late HIGH ensures latest pair 17 and 27; focus use prefix
        ao=[0.5]*36
        # The high-pivot candles have misleading AO values.
        ao[7]=1
        ao[17]=9
        # Leg A rises to a stronger AO hump BEFORE its high candle.
        ao[5]=12
        # Leg B has smaller max although AO at the HIGH itself grew.
        ao[15]=9
        first=market_position(bars(hi[:20],lo[:20]),ao[:20])
        self.assertTrue(any(x["kind"]=="BEARISH_AO_DIVERGENCE_CANDIDATE"
                            for x in first["divergences"]))
        div=first["divergences"][0]
        self.assertEqual(div["ao_leg_extrema"],[12,9])
        self.assertTrue(div["pivot_ao_only_is_not_used"])
        self.assertEqual(div["ao_leg_extremum_times"][0],bars(hi[:20],lo[:20])[5].at.isoformat())
        self.assertIsNone(first["trade_action"])

    def test_missing_prior_opposite_swing_blocks_fake_divergence(self):
        hi=[20.0]*26
        lo=[10.0]*26
        hi[5]=30
        hi[15]=35
        lo[10]=3
        ao=[0.0]*26
        ao[5]=12
        ao[15]=1
        x=market_position(bars(hi,lo),ao)
        self.assertEqual(x["divergences"],[])

    def test_divergence_confirmation_requires_pivot_plus_two(self):
        hi=[20.0]*30
        lo=[10.0]*30
        lo[3]=4
        lo[12]=5
        hi[7]=30
        hi[27]=35  # need indices 28 and 29 to confirm
        ao=[1.0]*30
        ao[5]=11.0
        ao[24]=4.0
        early=market_position(bars(hi[:29],lo[:29]),ao[:29])
        late=market_position(bars(hi,lo),ao)
        self.assertEqual(early["divergences"],[])
        self.assertTrue(late["divergences"])
        self.assertEqual(late["divergences"][0]["newer_swing_confirmed_at"],
                         bars(hi,lo)[29].at.isoformat())

    def test_unknown_without_swings(self):
        x=market_position(bars([20.0]*10,[10.0]*10),[0.1]*10)
        self.assertEqual(x["structure"],"UNRESOLVED")
        self.assertEqual(x["recent_levels"]["relative_to_known_swing"],"UNKNOWN")

    def test_misalignment_rejected(self):
        with self.assertRaises(ValueError):
            market_position(bars([20.0]*10,[10.0]*10),[1.0])

if __name__=="__main__":
    unittest.main()
