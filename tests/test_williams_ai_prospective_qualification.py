"""Prospective experiment: first seal as-of result, only then score future."""
import copy
import unittest
from datetime import datetime, timedelta, timezone

from williams_ai.prospective_qualification import (
    freeze_prospective_call, settle_prospective_call,
    summarize_sealed_results,
)
from williams_ai.causal_paper_lab import frozen_markets

START=datetime(2026, 1, 1, tzinfo=timezone.utc)

def rows(n=260):
    out=[]
    for i in range(n):
        px=100.0+i*.025
        out.append({
            "close_time":(START+timedelta(hours=i)).isoformat(),
            "open":px-.01,"high":px+.2,
            "low":px-.2,"close":px,"volume":100.0,
            "closed":True,
        })
    return out


def decision(obs,direction="UP"):
    return {
        "symbol":obs["symbol"],"as_of":obs["as_of"],
        "direction":direction,"trade_action":None,
        "execution_allowed":False,
    }


def freeze(prefix,side="UP",predictor=None):
    if predictor is None:
        predictor=lambda obs:decision(obs,side)
    return freeze_prospective_call(
        "ETHUSDT",prefix,tick_size=.01,
        model_id="fixed-model-v1",code_id="fixed-code-commit-v1",
        predictor=predictor)


class ProspectiveCausalResearch(unittest.TestCase):
    def test_two_stage_seal_and_evaluate_realized_future(self):
        full=rows(260)
        calls=[]
        def predictor(obs):
            calls.append(obs)
            return decision(obs)
        seal=freeze(full[:230],predictor=predictor)
        self.assertEqual(len(calls),1)
        self.assertEqual(calls[0]["last_closed_at"],full[229]["close_time"])
        self.assertEqual(seal["closed_history_count"],230)
        self.assertEqual(seal["horizon_hours"],24)
        self.assertFalse(seal["future_seen_by_this_function"])
        self.assertFalse(seal["execution_allowed"])
        self.assertIsNone(seal["trade_action"])
        scored=settle_prospective_call(seal,full)
        self.assertEqual(scored["observed_24h_direction"],"UP")
        self.assertTrue(scored["correct"])
        self.assertFalse(scored["execution_allowed"])
        self.assertIsNone(scored["actual_exchange_fills"])
        self.assertEqual(scored["future_confirmed_at"],full[253]["close_time"])

    def test_no_future_not_yet_known_must_fail(self):
        full=rows(260);seal=freeze(full[:230])
        for n in (230,240,253):
            with self.subTest(n=n),self.assertRaises(ValueError):
                settle_prospective_call(seal,full[:n])

    def test_mislabeled_half_hour_history_cannot_be_used_for_h1(self):
        full=rows(260)
        broken=copy.deepcopy(full)
        broken[150]["close_time"]=(START+timedelta(hours=149,minutes=30)).isoformat()
        with self.assertRaisesRegex(ValueError,"EXACT_H1_CADENCE_REQUIRED"):
            freeze(broken[:230])

    def test_future_bar_reordering_gap_or_unfinished_fails(self):
        full=rows(260);seal=freeze(full[:230])
        for idx in (230,253):
            corrupt=copy.deepcopy(full)
            corrupt[idx]["closed"]=False
            with self.subTest(idx=idx),self.assertRaises(ValueError):
                settle_prospective_call(seal,corrupt)
        full=rows(260);full[230]["close_time"]=full[232]["close_time"]
        with self.assertRaises(ValueError):
            settle_prospective_call(seal,full)

    def test_retroactive_change_forecast_or_chart_is_rejected(self):
        full=rows(260)
        seal=freeze(full[:230])
        revised={**seal,"direction":"DOWN"}
        with self.assertRaises(ValueError):
            settle_prospective_call(revised,full)
        revised={**seal,"model_id":"totally-different-model"}
        with self.assertRaises(ValueError):
            settle_prospective_call(revised,full)
        altered=copy.deepcopy(full)
        altered[210]["close"]+=.1
        with self.assertRaises(ValueError):
            settle_prospective_call(seal,altered)

    def test_invalid_or_executable_llm_output_rejected(self):
        full=rows(260)
        cases=[
            lambda o:{**decision(o),"trade_action":{"buy":True}},
            lambda o:{**decision(o),"execution_allowed":True},
            lambda o:{**decision(o),"direction":"BUY"},
            lambda o:{**decision(o),"as_of":"2026-02-01T00:00:00+00:00"},
            lambda o:{**decision(o),"future_price":999999},
        ]
        for callback in cases:
            with self.subTest(callback=callback), self.assertRaises(ValueError):
                freeze(full[:230],predictor=callback)

    def test_abstention_is_not_counted_as_high_accuracy(self):
        full=rows(260)
        s1=freeze(full[:230],"ABSTAIN")
        out=settle_prospective_call(s1,full)
        self.assertIsNone(out["correct"])
        report=summarize_sealed_results([out])
        self.assertEqual(report["directional_coverage_fraction"],0.0)
        self.assertIsNone(report["called_accuracy"])
        self.assertEqual(report["live_trade_qualification"],"NOT_QUALIFIED")

    def test_accuracy_matched_to_same_non_neutral_baseline(self):
        full=rows(260)
        good=settle_prospective_call(freeze(full[:230],"UP"),full)
        bad=settle_prospective_call(freeze(full[:231],"DOWN"),full)
        abstain=settle_prospective_call(freeze(full[:232],"ABSTAIN"),full)
        report=summarize_sealed_results([good,bad,abstain])
        self.assertEqual(report["directional_calls"],2)
        self.assertEqual(report["called_non_neutral_cases"],2)
        self.assertEqual(report["called_accuracy"],.5)
        self.assertEqual(report["directional_coverage_fraction"],round(2/3,6))
        self.assertEqual(report["matched_baseline_accuracy"],1.0)
        self.assertFalse(report["beats_lagged_momentum_on_selected_cases"])
        self.assertFalse(report["beats_all_declared_baselines_on_selected"])
        self.assertEqual(report["best_matched_baseline_accuracy"],1.0)
        self.assertEqual(report["model_id"],"fixed-model-v1")

    def test_mixed_revisions_cannot_launder_accuracy_into_one_cohort(self):
        full=rows(260)
        a=settle_prospective_call(freeze(full[:230]),full)
        b=settle_prospective_call(freeze(full[:231]),full)
        altered=dict(b)
        altered["model_id"]="other-trained-model-revision"
        with self.assertRaisesRegex(ValueError,"Cannot mix model"):
            summarize_sealed_results([a,altered])
        altered=dict(b)
        altered["code_id"]="new-source-version-id"
        with self.assertRaisesRegex(ValueError,"Cannot mix model"):
            summarize_sealed_results([a,altered])

    def test_duplicate_call_cannot_boost_qualification(self):
        full=rows(260);out=settle_prospective_call(freeze(full[:230]),full)
        with self.assertRaises(ValueError):
            summarize_sealed_results([out,out])

    def test_frozen_h1_real_market_prefix_accepts_only_closed_data(self):
        # This public historical sample is ALREADY REVIEWED, not true blind.
        markets=frozen_markets()
        self.assertEqual(len(markets),15)
        for symbol in sorted(markets)[:3]:
            market=markets[symbol];all_bars=market["bars"]
            index=900
            cutoff=all_bars[:index]
            seen=[]
            sealed=freeze_prospective_call(
                symbol,cutoff,tick_size=market["tick_size"],
                model_id="static-mock-check",code_id="source-commit-identity",
                predictor=lambda obs:(seen.append(obs["as_of"]) or decision(obs,"ABSTAIN")),
            )
            self.assertEqual(seen,[cutoff[-1]["close_time"]])
            out=settle_prospective_call(sealed,all_bars)
            self.assertIsNone(out["correct"])
            self.assertFalse(out["execution_allowed"])


if __name__=="__main__":
    unittest.main()
