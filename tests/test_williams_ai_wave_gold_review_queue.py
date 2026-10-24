"""Qwen's real forward market snapshot yields unlabelled, non-leaking review tasks."""
import copy
from datetime import datetime, timedelta, timezone
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
            view=task["review_visible_market_prefix"]
            self.assertEqual(len(view["bars"]),350)
            self.assertEqual(view["timeframe"],"H1")
            self.assertEqual(view["bars"][-1]["close_time"],task["cutoff"])
            self.assertEqual(view["review_view_sha256"],_fingerprint(view["bars"]))
            for bar in view["bars"]:
                self.assertEqual(set(bar),{
                    "close_time","open","high","low","close","volume","closed"})
            self.assertTrue(task["model_prediction_hidden_from_reviewers"])
            self.assertEqual(task["review_status"],
                             "UNLABELLED_REQUIRES_TWO_INDEPENDENT_EXPERTS")
            self.assertFalse(task["execution_allowed"])
            self.assertNotIn("direction",task)
            self.assertNotIn("confidence",task)
            self.assertNotIn("observed_future",task)
            self.assertNotIn("market_structure",task)

    def _reseal_market_and_cohort(self, cohort):
        # An attacker can recompute both local digests. This must not turn
        # invalid or future bars into a valid expert-review market prefix.
        for trial in cohort["trials"]:
            trial["sealed_forecast"]["history_sha256"] = _fingerprint(
                trial["market_prefix"]["bars"])
        cohort["manifest_sha256"] = _fingerprint({
            k: v for k, v in cohort.items() if k != "manifest_sha256"})
        return cohort

    def test_rehashed_duplicate_hour_is_not_a_closed_h1_history(self):
        cohort = sealed_public_cohort()
        bars = cohort["trials"][0]["market_prefix"]["bars"]
        bars[30]["close_time"] = bars[29]["close_time"]
        self._reseal_market_and_cohort(cohort)
        with self.assertRaisesRegex(ValueError, "UNSEALED_FUTURE_OR_INCONSISTENT"):
            prepare_wave_review_tasks(cohort)

    def test_rehashed_invalid_ohlc_is_not_expert_chart(self):
        cohort = sealed_public_cohort()
        bar = cohort["trials"][0]["market_prefix"]["bars"][29]
        bar["low"] = bar["high"] + 1
        self._reseal_market_and_cohort(cohort)
        with self.assertRaisesRegex(ValueError, "UNSEALED_FUTURE_OR_INCONSISTENT"):
            prepare_wave_review_tasks(cohort)

    def test_future_candle_disguised_as_earlier_timezone_fails(self):
        cohort = sealed_public_cohort()
        trial = cohort["trials"][0]
        cutoff = datetime.fromisoformat(
            trial["source"]["as_of"].replace("Z", "+00:00"))
        apparent_past = (cutoff + timedelta(hours=1)).astimezone(
            timezone(timedelta(hours=-12))).isoformat()
        # The old lexical comparison considered this an earlier string,
        # although its actual UTC timestamp is AFTER the sealed cutoff.
        self.assertLess(apparent_past, trial["source"]["as_of"])
        trial["market_prefix"]["bars"][-2]["close_time"] = apparent_past
        self._reseal_market_and_cohort(cohort)
        with self.assertRaisesRegex(ValueError, "UNSEALED_FUTURE_OR_INCONSISTENT"):
            prepare_wave_review_tasks(cohort)

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
