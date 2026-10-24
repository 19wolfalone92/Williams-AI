import unittest
from williams_ai.ntd_evidence import ntd_closed_bar_evidence as read


def types(report):
    return {(s["name"], s["side"]) for s in report["signals"]}


class NTDSourceRulesTests(unittest.TestCase):
    def test_three_bar_saucer_requires_positive_region_for_buy(self):
        x=read([1, .6, .9],[0.1,0.2,0.3])
        self.assertIn(("AO_SAUCER_THREE_BAR","LONG"), types(x))
        self.assertFalse(x["entry_admitted"])
        self.assertTrue(all(s["admission"].startswith("WAIT") for s in x["signals"]))
        self.assertIsNone(x["trade_action"])
        self.assertNotIn(("AO_SAUCER_THREE_BAR","LONG"),types(read([-1,-.6,-.9],[0,0,0])))

    def test_zero_line_cross_is_separate_from_tc2(self):
        x=read([-1,1],[-.5,-.3],initial_fractal_triggered=True)
        self.assertIn(("AO_ZERO_CROSS","LONG"),types(x))
        self.assertTrue(all(s["profile"]=="NTD_FIVE_DIMENSIONS" for s in x["signals"]))
        self.assertNotIn(("AO_ZERO_CROSS","LONG"),types(read([-1,-.2],[-.5,-.3])))

    def test_ac_crossing_zero_alone_is_not_signal(self):
        x=read([-0.4,-0.4,-0.4,-0.4],[-.02,-.01,.02,.01])
        self.assertFalse(any(s["name"]=="AC_ZERO_CROSS" for s in x["signals"]))

    def test_ac_requires_three_bars_against_momentum(self):
        a=read([-2]*5,[1,2,3,4,5])
        self.assertIn(("AC_ACCELERATION_CANDIDATE","LONG"),types(a))
        b=read([-2]*5,[1,2,3,3,5])
        self.assertNotIn(("AC_ACCELERATION_CANDIDATE","LONG"),types(b))

    def test_ac_histogram_sign_not_ao_sign_controls_signal_length(self):
        # NTD PDF 73-76: AC above zero buys after TWO rising bars,
        # even if AO is negative. Old implementation incorrectly required
        # THREE simply because AO was below zero.
        x=read([-2,-2,-2,-2],[1,1,2,3])
        self.assertIn(("AC_ACCELERATION_CANDIDATE","LONG"),types(x))
        self.assertIsNone(x["trade_action"])
        self.assertTrue(all(s["trade_action"] is None for s in x["signals"]))

    def test_ac_below_zero_requires_three_rising_columns_even_if_ao_positive(self):
        # The converse: AO positive cannot substitute for AC sign.
        x=read([2,2,2,2],[-4,-4,-3,-2])
        self.assertNotIn(("AC_ACCELERATION_CANDIDATE","LONG"),types(x))
        y=read([2,2,2,2],[-5,-4,-3,-2])
        self.assertIn(("AC_ACCELERATION_CANDIDATE","LONG"),types(y))

    def test_ac_short_mirror_uses_negative_ac_sign(self):
        x=read([2,2,2,2],[-1,-1,-2,-3])
        self.assertIn(("AC_ACCELERATION_CANDIDATE","SHORT"),types(x))
        y=read([-2,-2,-2,-2],[4,4,3,2])
        self.assertNotIn(("AC_ACCELERATION_CANDIDATE","SHORT"),types(y))
        z=read([-2,-2,-2,-2],[5,4,3,2])
        self.assertIn(("AC_ACCELERATION_CANDIDATE","SHORT"),types(z))

    def test_all_price_execution_remains_disallowed(self):
        x=read([1,2,3,4,5],[.1,.2,.3,.4,.5],initial_fractal_triggered=True)
        self.assertIsNone(x["trade_action"])
        self.assertTrue(all(s["trade_action"] is None for s in x["signals"]))

    def test_indicators_need_equal_length(self):
        with self.assertRaises(ValueError):
            read([1,2],[1])


if __name__ == "__main__":
    unittest.main()
