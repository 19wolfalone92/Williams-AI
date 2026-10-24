"""Profit-driven research must not chase an unprofitable high win rate."""
import dataclasses
from datetime import datetime,timedelta,timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from williams_ai.continual_trader import Outcome,PrivateExperience
from williams_ai.causal_trade_quality import _past_edge,causal_profit_quality_study

T=datetime(2025,1,1,tzinfo=timezone.utc)
def at(day):return (T+timedelta(days=day)).isoformat()
def case(day,symbol="ETHUSDT",*,return_value=.012,
         setup="WM3",side="LONG",delay=1):
    return Outcome(symbol,"H1",side,at(day),at(day+delay),
                   (1.,0.2,0.4,0.6,0.1,.5),
                   return_value,return_value<0,"d"*64,setup,"TC2."+setup)

class ProfitEdgeTests(unittest.TestCase):
    def test_high_win_rate_with_negative_expectancy_is_rejected(self):
        # 80% small winners and 20% very large losers is NET NEGATIVE,
        # despite a seductive hit rate. Three symbols on 35 different days.
        rows=[case(d,s,return_value=(.001 if d%5 else -.017))
              for d in range(1,36)
              for s in ("ETHUSDT","BTCUSDT","SOLUSDT")]
        gate=_past_edge(rows)
        self.assertGreater(gate["observed_win_fraction"],.75)
        self.assertLess(gate["daily_cluster_mean_net"],0)
        self.assertFalse(gate["qualified"])

    def test_same_day_many_crypto_signals_is_not_much_evidence(self):
        rows=[case(1,s,return_value=.05) for s in (
            "ETHUSDT","BTCUSDT","SOLUSDT","BNBUSDT","XRPUSDT","DOGEUSDT")]
        result=_past_edge(rows)
        self.assertEqual(result["distinct_utc_dates"],1)
        self.assertFalse(result["qualified"])

    def test_profitable_training_cannot_see_unresolved_training_labels(self):
        with TemporaryDirectory() as d:
            original=PrivateExperience(Path(d)/"original.db")
            poisoned=PrivateExperience(Path(d)/"poisoned.db")
            markets=("ETHUSDT","BTCUSDT","SOLUSDT")
            train=[case(day,s,return_value=(.017 if day%5 else -.003))
                   for day in range(1,40) for s in markets]
            # This apparently profitable episode is DECIDED in training,
            # but SETTLES AFTER its cutoff (day 40). It cannot train gate.
            late=[case(38,s,return_value=.3,delay=15,setup="WM2")
                  for s in markets]
            val=[case(day,s,return_value=.006 if day%2 else -.003)
                 for day in range(41,60) for s in markets]
            test=[case(day,s,return_value=.008 if day%2 else -.01)
                  for day in range(61,80) for s in markets]
            old=train+late+val+test
            new=train+[
                dataclasses.replace(x,net_return=-.4,stopped=True,
                                    source_hash="e"*64) for x in late]+val+[
                dataclasses.replace(x,net_return=-x.net_return,
                                    stopped=not x.stopped,
                                    source_hash="e"*64) for x in test]
            original.append(old,knowledge_as_of=at(100))
            poisoned.append(new,knowledge_as_of=at(100))
            args=dict(train_until=at(40),
                      validation_until=at(60),observe_until=at(80))
            a=causal_profit_quality_study(original,**args)
            b=causal_profit_quality_study(poisoned,**args)
            self.assertEqual(a["training_only_cohort_gates"],
                             b["training_only_cohort_gates"])
            self.assertEqual(a["validation_frozen"],b["validation_frozen"])
            self.assertEqual(a["test_frozen"]["paper_positions"],
                             b["test_frozen"]["paper_positions"])
            self.assertEqual(
                a["cutoff_purged_not_counted_in_each_partition"]["train"],3)
            self.assertFalse(a["execution_allowed"])
            self.assertIsNone(a["trade_action"])
            self.assertTrue(a["positive_win_rate_alone_never_authorizes_trading"])

    def test_book_profile_setup_is_not_cross_trained_with_other_wise_man(self):
        with TemporaryDirectory() as d:
            db=PrivateExperience(Path(d)/"journal.db")
            xs=[case(day,s,return_value=.014 if day%6 else -.004)
                for day in range(1,40)
                for s in ("ETHUSDT","BTCUSDT","SOLUSDT")]
            # WM1 has only 3 training dates and losses: it may not borrow
            # profitability from WM3 or magically become a verified WM1.
            xs += [case(d,"ETHUSDT",return_value=-.02,setup="WM1")
                   for d in range(1,4)]
            xs += [case(d,s) for d in range(41,60)
                   for s in ("ETHUSDT","BTCUSDT","SOLUSDT")]
            xs += [case(d,s) for d in range(61,79)
                   for s in ("ETHUSDT","BTCUSDT","SOLUSDT")]
            db.append(xs,knowledge_as_of=at(90))
            r=causal_profit_quality_study(
                db,train_until=at(40),validation_until=at(60),
                observe_until=at(80))
            self.assertFalse(r["training_only_cohort_gates"]["WM1:LONG"]["qualified"])
            self.assertIn("WM3:LONG",r["training_only_cohort_gates"])
            self.assertTrue(r["no_test_selection_or_weight_training"])

    def test_missing_future_refuses_fake_test(self):
        with TemporaryDirectory() as d:
            db=PrivateExperience(Path(d)/"journal.db")
            db.append([case(1)],knowledge_as_of=at(3))
            with self.assertRaisesRegex(ValueError,"NO_COMPLETE_PURGED"):
                causal_profit_quality_study(
                    db,train_until=at(10),validation_until=at(15),
                    observe_until=at(20))
            with self.assertRaisesRegex(ValueError,"INVALID_FIXED"):
                causal_profit_quality_study(
                    db,train_until=at(20),validation_until=at(15),
                    observe_until=at(30))

if __name__=="__main__":
    unittest.main()
