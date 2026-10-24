"""Qwen's real forward market snapshot yields unlabelled, non-leaking review tasks."""
import copy
import unittest

from williams_ai.causal_paper_lab import frozen_markets
from williams_ai.prospective_qualification import _fingerprint
from williams_ai.wave_gold_review_queue import prepare_wave_review_tasks


def sealed_public_cohort():
    records=[]
    for name in ("BTCUSDT","ETHUSDT","SOLUSDT"):
        m=frozen_markets()[name]
        bars=m["bars"][:350]
        asof=bars[-1]["close_time"]
        rows={
            "symbol":name,
            "source":{"venue":"SPOT","as_of":asof},
            "market_prefix":{"symbol":name,"tick_size":m["tick_size"],
                             "bars":bars},
            "sealed_forecast":{"symbol":name,"as_of":asof,
                               "history_sha256":_fingerprint(bars),
                               "direction":"UP","confidence":0.99},
        }
        records.append(rows)
    body={
        "schema":"WILLIAMS_AI_PUBLIC_FORWARD_COHORT_V1",
        "venue":"SPOT","model_id":"pinned-model-001",
        "code_id":"model-source-v1",
        "trials":records,
        "trade_action":None,"execution_allowed":False,
    }
    return {**body,"manifest_sha256":_fingerprint(body)}


class WaveReviewTaskContracts(unittest.TestCase):
    def test_prepare_three_precommitted_charts_without_leaking_predictions(self):
        cohort=sealed_public_cohort()
        result=prepare_wave_review_tasks(cohort)
        self.assertEqual(len(result["tasks"]),3)
        self.assertTrue(result["all_wave_labels_missing_on_purpose"])
        self.assertEqual(result["source_precommit_sha256"],
                         cohort["manifest_sha256"])
        for task in result["tasks"]:
            self.assertTrue(task["model_prediction_hidden_from_reviewers"])
            self.assertEqual(task["review_status"],
                             "UNLABELLED_REQUIRES_TWO_INDEPENDENT_EXPERTS")
            self.assertFalse(task["execution_allowed"])
            self.assertNotIn("direction",task)
            self.assertNotIn("confidence",task)
            self.assertNotIn("observed_future",task)
            self.assertNotIn("market_structure",task)

    def test_source_changed_after_commitment_fails(self):
        cohort=sealed_public_cohort()
        c=copy.deepcopy(cohort)
        c["trials"][0]["market_prefix"]["bars"][17]["close"]+=1
        # Even when the outer seal was recomputed, inner history seal fails.
        c["manifest_sha256"]=_fingerprint({
            k:v for k,v in c.items() if k!="manifest_sha256"})
        with self.assertRaisesRegex(ValueError,"UNSEALED_FUTURE_OR_INCONSISTENT"):
            prepare_wave_review_tasks(c)

    def test_cannot_fake_future_bars_or_wrong_venue(self):
        c=sealed_public_cohort()
        c["trials"][0]["source"]["venue"]="FUTURES"
        c["manifest_sha256"]=_fingerprint({
            k:v for k,v in c.items() if k!="manifest_sha256"})
        with self.assertRaisesRegex(ValueError,"UNSEALED_FUTURE_OR_INCONSISTENT"):
            prepare_wave_review_tasks(c)

    def test_single_modified_model_response_invalidates_precommit(self):
        c=sealed_public_cohort()
        c["trials"][0]["sealed_forecast"]["direction"]="DOWN"
        with self.assertRaisesRegex(ValueError,"UNTRUSTED_OR_TAMPERED"):
            prepare_wave_review_tasks(c)

if __name__=="__main__":
    unittest.main()
