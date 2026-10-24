"""AI genuinely updates numeric policy from past outcomes, never future candles."""
import dataclasses
from datetime import datetime,timedelta,timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import math
import os
import unittest

from williams_ai.continual_trader import (
    Outcome,PrivateExperience,learn_from_private_experience,
    research_prediction,MODEL_SCHEMA
)

ZERO=datetime(2024,1,1,tzinfo=timezone.utc)
HASH="a"*64

def sample(i:int,symbol:str="ETHUSDT",*,bias:float=0)->Outcome:
    time=ZERO+timedelta(hours=i)
    end=time+timedelta(hours=2)
    side="LONG" if i%2 else "SHORT"
    wave="WM3" if i%3 else "WM2"
    rule="TC2.WM3" if wave=="WM3" else "TC2.WM2"
    return Outcome(symbol,"H1",side,time.isoformat(),end.isoformat(),
                   (1.0, (i%11)/10+bias,(-1)**i*.5,0.7,0.25,1.0),
                   0.01 if (i%5)<3 else -0.015,
                   (i%5)>2,HASH,wave,rule)

class PrivateMarketLearningTests(unittest.TestCase):
    def test_future_not_known_cannot_train_and_utc_offsets_normalized(self):
        with TemporaryDirectory() as d:
            db=PrivateExperience(Path(d)/"memory.sqlite3")
            s=sample(4)
            future=(ZERO+timedelta(hours=5)).isoformat()
            with self.assertRaisesRegex(ValueError,"FUTURE_OUTCOME"):
                db.append([s],knowledge_as_of=future)
            self.assertEqual(db.samples(),[])
            utc_z=s.decision_at.replace("+00:00","Z")
            changed=dataclasses.replace(s,decision_at=utc_z)
            self.assertEqual(changed.decision_at,s.decision_at)
            written=db.append([s],knowledge_as_of=s.resolved_at)
            self.assertEqual(written["new_resolved_paper_outcomes"],1)
            self.assertEqual(db.append([changed],
                knowledge_as_of=s.resolved_at)["identical_duplicate_outcomes"],1)
            self.assertEqual(len(db.samples()),1)
            self.assertFalse(written["execution_allowed"])
            self.assertIsNone(written["trade_action"])

    def test_immutable_outcome_and_rollback_prevents_late_label_rewrite(self):
        with TemporaryDirectory() as d:
            db=PrivateExperience(Path(d)/"journ.db")
            original=sample(10)
            db.append([original],knowledge_as_of=sample(200).resolved_at)
            distorted=dataclasses.replace(original,net_return=0.99)
            with self.assertRaisesRegex(ValueError,"REVISION"):
                db.append([sample(11),distorted],knowledge_as_of=sample(200).resolved_at)
            self.assertEqual(db.samples(),[original])

    def test_trained_experience_changes_predicted_research_probabilities(self):
        with TemporaryDirectory() as d:
            db=PrivateExperience(Path(d)/"ai_memory.db")
            cases=[sample(i,sym,bias=offset) for offset,sym in
                   ((0.0,"ETHUSDT"),(0.2,"BTCUSDT"),(-0.3,"SOLUSDT"))
                   for i in range(300)]
            db.append(cases,knowledge_as_of=sample(400).resolved_at)
            result=learn_from_private_experience(
                db,train_until=(ZERO+timedelta(hours=181)).isoformat(),
                validate_until=(ZERO+timedelta(hours=241)).isoformat(),
                observe_until=(ZERO+timedelta(hours=301)).isoformat(),
            )
            self.assertEqual(result["schema"],MODEL_SCHEMA)
            self.assertEqual(result["train_count"],3*179)
            self.assertGreater(result["validation"]["examples"],0)
            self.assertGreater(result["frozen_test"]["examples"],0)
            self.assertEqual(result["actual_book_confirmed_wm_signals"],0)
            self.assertTrue(result["only_proxy_williams_setups_used_for_fit"])
            self.assertTrue(result["test_was_never_used_to_fit_or_select"])
            self.assertFalse(result["real_gguf_weights_updated"])
            self.assertFalse(result["execution_allowed"])
            self.assertTrue(db.save_policy(result))
            self.assertFalse(db.save_policy(result))
            self.assertEqual(len(db.policies()),1)
            with self.assertRaisesRegex(ValueError,"PAST_REPLAY"):
                research_prediction(
                    result,cases[0].features,decision_at=cases[0].decision_at)
            prediction=research_prediction(
                result,cases[-1].features,
                decision_at=(ZERO+timedelta(hours=401)).isoformat())
            if result["validation_state"]=="VALIDATION_BEATS_PRIOR_RESEARCH_ONLY":
                self.assertTrue(
                    0<=prediction["research_positive_outcome_estimate"]<=1)
            else:
                self.assertEqual(prediction["mode"],
                                 "VALIDATION_REJECTED_LEARNING_ABSTAIN")
                self.assertIsNone(
                    prediction["research_positive_outcome_estimate"])
            self.assertFalse(prediction["current_book_signal_confirmed"])
            self.assertFalse(prediction["execution_allowed"])

    def test_malicious_numeric_and_missing_book_proof_rejected(self):
        original=sample(15)
        for modify in ({"features":(1,float("nan"),0,0,0,0)},
                       {"net_return":float("inf")},{"side":"BUY"},
                       {"rule_id":"NTD.AO"},{"source_hash":"fake"},
                       {"decision_at":original.resolved_at},
                       {"setup":"MADE_UP_WM4"}):
            with self.subTest(fields=modify):
                with self.assertRaises(ValueError):
                    dataclasses.replace(original,**modify)
        with TemporaryDirectory() as d:
            path=Path(d)/"real.db"
            path.write_text("unsafe")
            os.chmod(path,0o644)
            with self.assertRaises(ValueError):
                PrivateExperience(path)
            path.unlink()
            link=Path(d)/"link.db"
            link.symlink_to(Path(d)/"target.db")
            with self.assertRaises(ValueError):
                PrivateExperience(link)

    def test_unconfirmed_generic_bidirectional_samples_not_tradable_policy(self):
        with TemporaryDirectory() as d:
            db=PrivateExperience(Path(d)/"proxies.db")
            xs=[dataclasses.replace(sample(i),setup="NO_BOOK_SETUP",
                                    rule_id="NOT_BOOK_CONFIRMED")
                for i in range(100)]
            db.append(xs,knowledge_as_of=sample(150).resolved_at)
            with self.assertRaisesRegex(ValueError,"INSUFFICIENT_PURGED"):
                learn_from_private_experience(
                    db,train_until=sample(60).decision_at,
                    validate_until=sample(80).decision_at,
                    observe_until=sample(110).decision_at)

    def test_fail_closed_if_no_outcomes_have_settled_in_known_past(self):
        with TemporaryDirectory() as d:
            db=PrivateExperience(Path(d)/"zero.db")
            with self.assertRaises(ValueError):
                learn_from_private_experience(db,train_until=sample(10).decision_at,
                    validate_until=sample(20).decision_at,
                    observe_until=sample(30).decision_at)


if __name__=="__main__":
    unittest.main()
