"""Chronological expected-NET model is a research hypothesis, never an order."""
import dataclasses
from datetime import datetime,timezone,timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from williams_ai.continual_trader import Outcome,PrivateExperience
from williams_ai.prequential_net_skill import causal_expected_net_study

ORIGIN=datetime(2025,1,1,tzinfo=timezone.utc)
STAMP=lambda day:(ORIGIN+timedelta(days=day)).isoformat()
ASSETS=("BTCUSDT","ETHUSDT","SOLUSDT")
def item(day,symbol,*,net=None,resolve_days=1,setup="WM3"):
    if net is None: net=0.001 if day%5 else -0.012
    return Outcome(symbol,"H1","LONG",STAMP(day),STAMP(day+resolve_days),
                   (1.,0.25,0.1,0.7,0.3,0.2),
                   net,net<0,"b"*64,setup,"TC2."+setup)

def run(db):
    return causal_expected_net_study(
        db,train_until=STAMP(60),validation_until=STAMP(85),
        observe_until=STAMP(110),min_training_decisions=40,
        min_validation_decisions=12)

class AfterCostResearchTests(unittest.TestCase):
    def test_learn_only_from_settled_train_and_abstain_on_loss_expectancy(self):
        with TemporaryDirectory() as temp:
            db=PrivateExperience(Path(temp)/"experience.db")
            samples=[item(i,s) for i in range(1,106) for s in ASSETS]
            db.append(samples,knowledge_as_of=STAMP(110))
            r=run(db)
            self.assertEqual(r["trained_after_prior_settlement"],3*58)
            self.assertEqual(r["test_frozen"]["assets"],3)
            self.assertLess(r["train_cohort_average_net"]["WM3:LONG"],0)
            self.assertFalse(r["execution_allowed"])
            self.assertIsNone(r["trade_action"])
            self.assertFalse(r["model_is_small_numeric_not_GGUF_finetuning"] is False)
            self.assertEqual(r["qualification"],"ABSTAIN_EXPECTED_RETURN_NOT_VALIDATED")

    def test_poisoning_holdout_does_not_change_learning_or_selection(self):
        with TemporaryDirectory() as temp:
            old=[item(i,s,net=(.015 if i%3 else -.006))
                 for s in ASSETS for i in range(1,106)]
            poisoned=[
                dataclasses.replace(x,net_return=(-.3 if x.net_return>0 else .3),
                                    stopped=not x.stopped,source_hash="c"*64)
                if x.decision_at>=STAMP(85) else x for x in old]
            x=PrivateExperience(Path(temp)/"x.db")
            y=PrivateExperience(Path(temp)/"y.db")
            x.append(old,knowledge_as_of=STAMP(110))
            y.append(poisoned,knowledge_as_of=STAMP(110))
            a,b=run(x),run(y)
            self.assertEqual(a["learned_weights"],b["learned_weights"])
            self.assertEqual(a["validation_frozen"],b["validation_frozen"])
            self.assertEqual(a["qualification"],b["qualification"])
            self.assertEqual(a["test_frozen"]["paper_selected_ideas"],
                             b["test_frozen"]["paper_selected_ideas"])
            self.assertNotEqual(a["test_frozen"]["paper_net_mse"],
                                b["test_frozen"]["paper_net_mse"])

    def test_poison_unresolved_train_label_never_enters_model(self):
        with TemporaryDirectory() as temp:
            old=[item(i,s,net=.02 if i%6 else -.02)
                 for i in range(1,106) for s in ASSETS]
            late=[item(55,s,resolve_days=30,net=.4,setup="WM2") for s in ASSETS]
            changed=[dataclasses.replace(x,net_return=-.4,stopped=True,
                                        source_hash="9"*64) for x in late]
            a=PrivateExperience(Path(temp)/"a.db")
            b=PrivateExperience(Path(temp)/"b.db")
            a.append(old+late,knowledge_as_of=STAMP(115))
            b.append(old+changed,knowledge_as_of=STAMP(115))
            left,right=run(a),run(b)
            self.assertEqual(left["learned_weights"],right["learned_weights"])
            self.assertEqual(left["validation_frozen"],right["validation_frozen"])
            self.assertEqual(left["train_cutoff_purged_late_outcomes"]["train"],6)

    def test_malformed_bounds_and_missing_validation_fail_closed(self):
        with TemporaryDirectory() as temp:
            db=PrivateExperience(Path(temp)/"study.db")
            db.append([item(1,"ETHUSDT")],knowledge_as_of=STAMP(3))
            with self.assertRaisesRegex(ValueError,"INSUFFICIENT_SETTLED"):
                run(db)
            with self.assertRaisesRegex(ValueError,"CHRONOLOGICAL"):
                causal_expected_net_study(
                    db,train_until=STAMP(85),validation_until=STAMP(60),
                    observe_until=STAMP(110))

if __name__=="__main__":
    unittest.main()
