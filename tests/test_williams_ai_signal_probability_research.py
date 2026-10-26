"""Prequential frequencies may NOT see any resolved future bar."""
from datetime import datetime,timedelta,timezone
import unittest
from williams_ai.signal_probability_research import empirical_regime_scenarios


def case(i,outcome="UP",future=False):
    base=datetime(2025,1,1,tzinfo=timezone.utc)+timedelta(hours=i)
    if future:base+=timedelta(days=1000)
    stamp=lambda d:d.isoformat()
    return {"case_id":f"case-{i}","signal_family":"WM1",
       "market_regime":"H4_DOWN_H1_CORRECTION","horizon_bars":12,
       "observed_at":stamp(base),"resolved_at":stamp(base+timedelta(hours=12)),
       "available_at":stamp(base+timedelta(hours=13)),"outcome":outcome}


class ConditionalProbabilities(unittest.TestCase):
    def kwargs(self):
        return dict(as_of="2026-10-26T10:00:00+00:00",
                    signal_family="WM1",
                    market_regime="H4_DOWN_H1_CORRECTION",horizon_bars=12)

    def test_silent_abstain_when_not_enough_cases(self):
        v=empirical_regime_scenarios([case(i) for i in range(20)],**self.kwargs())
        self.assertIsNone(v["probabilities"])
        self.assertFalse(v["execution_allowed"])

    def test_closed_only_and_no_future_leakage(self):
        old=[case(i,"UP" if i<45 else "DOWN") for i in range(60)]
        a=empirical_regime_scenarios(old,**self.kwargs(),min_cases=30)
        b=empirical_regime_scenarios(old+[case(61,"DOWN",True)],
                                     **self.kwargs(),min_cases=30)
        self.assertEqual(a["probabilities"],b["probabilities"])
        self.assertEqual(b["excluded_future_unavailable_cases"],1)
        self.assertGreater(a["probabilities"]["UP"],a["probabilities"]["DOWN"])
        self.assertFalse(a["out_of_sample_calibration_proven"])

    def test_wrong_chronology_rejected(self):
        bad=case(1)
        bad["available_at"]=bad["observed_at"]
        with self.assertRaisesRegex(ValueError,"CHRONOLOGY"):
            empirical_regime_scenarios([bad],**self.kwargs())

    def test_different_horizon_does_not_create_fake_sample(self):
        rows=[case(i) for i in range(60)]
        v=empirical_regime_scenarios(rows,**{**self.kwargs(),"horizon_bars":24})
        self.assertIsNone(v["probabilities"])


if __name__=="__main__":unittest.main()
