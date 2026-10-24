"""Never train on an unresolved outcome or a held-out future market."""
from datetime import datetime,timezone,timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
import dataclasses
import unittest

from williams_ai.continual_trader import Outcome,PrivateExperience
from williams_ai.prequential_skill import causal_incremental_study, _metrics

ORIGIN=datetime(2025,2,1,tzinfo=timezone.utc)
HASH="7"*64
def stamp(i):return (ORIGIN+timedelta(hours=i)).isoformat()
def ex(i, symbol="ETHUSDT",*,future_resolution=3,win=None):
    win=i%6<4 if win is None else win
    return Outcome(
        symbol,"H1", "LONG" if i%2 else "SHORT",
        stamp(i),stamp(i+future_resolution),
        (1.0,(i%8-4)*0.22,0.5*(-1)**i,0.7,0.1,0.9),
        0.015 if win else -0.025,
        not win,HASH,"WM3","TC2.WM3")

def run(db):
    return causal_incremental_study(
        db,train_until=stamp(100),validation_until=stamp(145),
        observe_until=stamp(190),min_training_decisions=30,
        min_validation_decisions=15)

class RealOnlineFeedbackTests(unittest.TestCase):
    def test_global_chronology_and_frozen_holdout(self):
        with TemporaryDirectory() as d:
            db=PrivateExperience(Path(d)/"memory.db")
            markets=("ETHUSDT","BTCUSDT","SOLUSDT")
            xs=[ex(i,s) for s in markets for i in range(1,186)]
            db.append(xs,knowledge_as_of=stamp(190))
            out=run(db)
            self.assertEqual(out["training_decision_timestamps"],99)
            self.assertEqual(out["training_updates_after_settlement"],3*96)
            self.assertEqual(out["test_frozen"]["distinct_assets"],3)
            self.assertEqual(out["validation_frozen"]["distinct_assets"],3)
            self.assertTrue(out["frozen_test_never_selected_or_trained"])
            self.assertTrue(out["simultaneous_crypto_decisions_share_previous_past_only_weights"])
            self.assertFalse(out["actual_gguf_weights_changed"])
            self.assertFalse(out["execution_allowed"])
            self.assertEqual(out["actual_binance_orders"],0)
            self.assertIsNone(out["trade_action"])

    def test_poisoned_future_test_results_do_not_change_previous_learning(self):
        markets=("ETHUSDT","BTCUSDT","SOLUSDT")
        original=[ex(i,s) for s in markets for i in range(1,186)]
        poisoned=[dataclasses.replace(
             x,net_return=(-.03 if x.net_return>0 else .03),
             stopped=not x.stopped,source_hash="9"*64)
             if x.decision_at>=stamp(145) else x for x in original]
        with TemporaryDirectory() as d:
            p=PrivateExperience(Path(d)/"previous.db")
            q=PrivateExperience(Path(d)/"poisoned.db")
            p.append(original,knowledge_as_of=stamp(190))
            q.append(poisoned,knowledge_as_of=stamp(190))
            a,b=run(p),run(q)
            self.assertEqual(a["fingerprint"],b["fingerprint"])
            self.assertEqual(a["learned_weights"],b["learned_weights"])
            self.assertEqual(a["validation_frozen"],b["validation_frozen"])
            self.assertNotEqual(a["test_frozen"],b["test_frozen"])
            self.assertFalse(a["execution_allowed"])

    def test_poisoning_future_training_labels_which_havent_resolved(self):
        with TemporaryDirectory() as d:
            markets=("ETHUSDT","BTCUSDT","SOLUSDT")
            xs=[ex(i,s,future_resolution=(25 if i>=90 else 3))
                for s in markets for i in range(1,186)
                if i<90 or i>=100 or i+25<190]
            # All training cases decided from t90..t99 settle AFTER
            # train cut; they MUST be excluded from SGD updates, though
            # decision evidence itself existed at the decision time.
            a=PrivateExperience(Path(d)/"original.db")
            b=PrivateExperience(Path(d)/"changed.db")
            a.append(xs,knowledge_as_of=stamp(220))
            changed=[
                dataclasses.replace(x,net_return=-x.net_return,
                                   stopped=not x.stopped,source_hash="2"*64)
                if stamp(90)<=x.decision_at<stamp(100) else x for x in xs]
            b.append(changed,knowledge_as_of=stamp(220))
            r1=causal_incremental_study(
                a,train_until=stamp(100),validation_until=stamp(145),
                observe_until=stamp(190))
            r2=causal_incremental_study(
                b,train_until=stamp(100),validation_until=stamp(145),
                observe_until=stamp(190))
            self.assertEqual(r1["learned_weights"],r2["learned_weights"])
            self.assertLess(
                datetime.fromisoformat(r1["last_training_outcome_resolved_at"]),
                datetime.fromisoformat(stamp(100)))


    def test_delayed_validation_labels_are_not_used_to_qualify_model(self):
        # An event decided before the validation cutoff may resolve during
        # the final TEST period. Such hindsight must not select a model at b.
        with TemporaryDirectory() as d:
            markets=("ETHUSDT","BTCUSDT","SOLUSDT")
            xs=[ex(i,s,future_resolution=(25 if 130<=i<145 else 3))
                for s in markets for i in range(1,186)]
            contaminated=[
                dataclasses.replace(x,net_return=-x.net_return,
                                    stopped=not x.stopped,source_hash="a"*64)
                if stamp(130)<=x.decision_at<stamp(145) else x
                for x in xs]
            clean_db=PrivateExperience(Path(d)/"clean.db")
            late_poison_db=PrivateExperience(Path(d)/"late_poison.db")
            clean_db.append(xs,knowledge_as_of=stamp(190))
            late_poison_db.append(contaminated,knowledge_as_of=stamp(190))
            clean=run(clean_db)
            poisoned=run(late_poison_db)
            self.assertEqual(clean["fingerprint"],poisoned["fingerprint"])
            self.assertEqual(clean["validation_frozen"],poisoned["validation_frozen"])
            self.assertEqual(clean["qualification"],poisoned["qualification"])
            self.assertEqual(clean["test_frozen"],poisoned["test_frozen"])
            self.assertEqual(clean["validation_frozen"]["examples"],90)
            self.assertEqual(
                clean["cutoff_purged_late_resolved_outcomes"]["validation_frozen"],45)
            self.assertTrue(
                clean["validation_labels_strictly_settled_before_selection_cutoff"])
            self.assertFalse(clean["execution_allowed"])

    def test_validation_only_with_future_labels_cannot_qualify(self):
        with TemporaryDirectory() as d:
            db=PrivateExperience(Path(d)/"late.db")
            xs=[ex(i,s,future_resolution=(80 if 100<=i<145 else 3))
                for s in ("ETHUSDT","BTCUSDT","SOLUSDT")
                for i in range(1,186)]
            db.append(xs,knowledge_as_of=stamp(250))
            with self.assertRaisesRegex(
                    ValueError,"INSUFFICIENT_INDEPENDENT_FROZEN_VALIDATION_TIMES"):
                run(db)

    def test_multimarket_one_day_is_not_many_independent_dates(self):
        # A market-wide synchronized move should not fake independent
        # validation evidence just because many coins signaled together.
        rows=[(ex(i,s),0.99) for i in range(1,12)
              for s in ("BTCUSDT","ETHUSDT","SOLUSDT")]
        result=_metrics(rows,0.50)
        self.assertEqual(result["paper_selected_examples"],33)
        self.assertEqual(result["paper_selected_distinct_utc_days"],1)
        self.assertEqual(result["paper_selected_distinct_assets"],3)

    def test_invalid_learning_rate_and_missing_holdout_block(self):
        with TemporaryDirectory() as d:
            db=PrivateExperience(Path(d)/"empty.db")
            with self.assertRaises(ValueError):
                run(db)
            with self.assertRaisesRegex(ValueError,"CHRONOLOGICAL"):
                causal_incremental_study(
                    db,train_until=stamp(50),validation_until=stamp(45),
                    observe_until=stamp(100))
            with self.assertRaisesRegex(ValueError,"HYPERPARAMETERS"):
                causal_incremental_study(
                    db,train_until=stamp(50),validation_until=stamp(70),
                    observe_until=stamp(100),learning_rate=2.0)

if __name__=="__main__":
    unittest.main()
