"""Run separately with pandas/numpy installed. No neural weights, exchange or trading."""
import unittest
import numpy as np
import pandas as pd

from strategy import calculate_indicators, config_from_env, smma

class SourceArithmeticRegressions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rows=[]
        for i in range(96):
            high=100.0+0.82*i+0.17*(i%3)
            low=high-(1.0+0.09*(i%5))
            mid=(high+low)/2
            rows.append({"open":mid,"high":high,"low":low,"close":mid,
                         "volume":1000.0+(i%7),
                         "open_time_ms":1800000000000+i*3600000})
        cls.frame=pd.DataFrame(rows)
        cls.cfg=config_from_env({})
        cls.ind=calculate_indicators(cls.frame,cls.cfg)

    def test_real_ao_uses_median_not_misprinted_half_range(self):
        df=self.frame
        mid=(df["high"]+df["low"])/2
        expected=mid.rolling(5).mean()-mid.rolling(34).mean()
        pd.testing.assert_series_equal(self.ind["ao"],expected,check_names=False)
        half_range=(df["high"]-df["low"])/2
        mistaken=half_range.rolling(5).mean()-half_range.rolling(34).mean()
        self.assertGreater(float((expected-mistaken).abs().iloc[50:].max()),3.0)
        self.assertFalse(self.ind["ao"].iloc[45:].isna().any())

    def test_alligator_correct_lips_shift_3_vs_2(self):
        mid=(self.frame["high"]+self.frame["low"])/2
        expected=smma(mid,5).shift(3)
        mistaken=smma(mid,5).shift(2)
        pd.testing.assert_series_equal(self.ind["lips_current"],expected,
                                       check_names=False)
        self.assertGreater(float((expected-mistaken).abs().iloc[35:].max()),0.2)
        self.assertEqual(self.cfg["lips_shift"],3)

    def test_ac_is_ao_minus_five_bar_mean_not_a_second_ao(self):
        ref=self.ind["ao"]-self.ind["ao"].rolling(5).mean()
        pd.testing.assert_series_equal(self.ind["ac"],ref,check_names=False)
        pd.testing.assert_series_equal(self.ind["profitunity_macd_histogram"],
                                       ref,check_names=False)

    def test_indicators_at_a_cutoff_do_not_change_with_future_candles(self):
        prefix=calculate_indicators(self.frame.iloc[:68].copy(),self.cfg)
        for field in ("ao","ac","jaw_current","teeth_current","lips_current"):
            pd.testing.assert_series_equal(prefix[field],self.ind[field].iloc[:68],
                                           check_names=False)

    def test_wm2_green_color_requires_ao_rising_not_ao_positive(self):
        ao=self.ind["ao"]
        self.assertTrue((self.ind["ao_green"]==(ao>ao.shift(1))).all())
        self.assertTrue((self.ind["ao_red"]==(ao<ao.shift(1))).all())
        self.assertTrue((self.ind["ao_green_streak"].iloc[50:]>0).any() or
                        (self.ind["ao_red_streak"].iloc[50:]>0).any())

if __name__=="__main__":
    unittest.main()
