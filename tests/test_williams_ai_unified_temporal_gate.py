"""Regression: actual time ordering, not ISO lexicographic ordering.

Proofs, model hints and auditor must reject an unconfirmed WM3, a stale
breakout or a forged cached manifest. This never initiates trading.
"""
import copy
import unittest

from williams_ai.algorithm_guard import audit_reasoning
from williams_ai.decision_contract import legal_wm3_discussion_states
from williams_ai.signal_proofs import current_signal_proofs


CUTOFF = "2026-10-10T12:00:00+00:00"


def observation(candidate):
    return {
        "symbol": "BTCUSDT", "timeframe": "H1", "as_of": CUTOFF,
        "gaps": 0, "candidates": [candidate],
        "execution_allowed": False,
    }


def candidate(*, confirmed, state="AWAITING_PRICE_TRIGGER", touch=None):
    return {
        "type": "WM3_FRACTAL", "side": "LONG", "confirmed_at": confirmed,
        "price_touch_observed_at": touch,
        "state": state, "entry_trigger": 120.5,
    }


def proposal(entry="PENDING_TRIGGER"):
    return {
        "symbol": "BTCUSDT", "as_of": CUTOFF,
        "edition": "TC2", "setup": "WM3", "side": "LONG",
        "entry_state": entry, "rule_ids": ["TC2.WM3"],
        "exit_basis": "NONE", "trade_action": None,
        "execution_allowed": False,
    }


class IndependentTemporalWM3Gate(unittest.TestCase):
    def verify(self, c, entry="PENDING_TRIGGER"):
        obs = observation(c)
        obs["signal_proofs"] = current_signal_proofs(obs)
        audit = audit_reasoning(obs, proposal(entry))
        self.assertFalse(audit["execution_allowed"])
        self.assertIsNone(audit["trade_action"])
        return obs, audit

    def test_future_absolute_confirmation_rejected_even_if_iso_string_looks_earlier(self):
        # 09:00-04 == 13:00Z, later than 12:00Z. String comparison says earlier.
        obs, audit = self.verify(candidate(confirmed="2026-10-10T09:00:00-04:00"))
        self.assertEqual(legal_wm3_discussion_states(obs), [])
        self.assertIn("WM3_NO_CURRENT_PENDING_VALID_FRACTAL", audit["hard_failures"])
        self.assertIn("NO_CAUSAL_CURRENT_SOURCE_EVIDENCE_FOR_CLAIM", audit["hard_failures"])

    def test_past_absolute_confirmation_is_allowed_pending_not_filled(self):
        # 16:00+05 == 11:00Z, before 12:00Z. Lexicographic ordering gets this wrong.
        obs, audit = self.verify(candidate(confirmed="2026-10-10T16:00:00+05:00"))
        self.assertEqual(
            legal_wm3_discussion_states(obs)[0]["entry_state"], "PENDING_TRIGGER")
        self.assertNotIn("WM3_NO_CURRENT_PENDING_VALID_FRACTAL", audit["hard_failures"])
        self.assertFalse(audit["execution_allowed"])

    def test_absent_and_invalid_confirmation_are_not_current_wm3(self):
        for confirmed in (None, "no timestamp", "2026-10-11T01:00:00+00:00"):
            with self.subTest(confirmed=confirmed):
                obs, audit = self.verify(candidate(confirmed=confirmed))
                self.assertIn("WM3_NO_CURRENT_PENDING_VALID_FRACTAL", audit["hard_failures"])
                self.assertEqual(obs["signal_proofs"]["current_proof_count"], 0)

    def test_no_valid_touch_when_current_breakout_is_unconfirmed(self):
        obs, audit = self.verify(candidate(
            confirmed=None,
            state="CLOSED_BAR_TOUCH_PASSES_TEETH_NOT_EXCHANGE_FILL",
            touch=CUTOFF), "CLOSED_BAR_TOUCH_ONLY")
        self.assertIn("WM3_NO_PRICE_TOUCH_ON_THIS_CLOSED_BAR", audit["hard_failures"])
        self.assertIn("NO_CAUSAL_CURRENT_SOURCE_EVIDENCE_FOR_CLAIM", audit["hard_failures"])

    def test_forged_cached_proof_is_not_trusted(self):
        stale = candidate(
            confirmed="2026-10-09T10:00:00+00:00",
            state="CLOSED_BAR_TOUCH_PASSES_TEETH_NOT_EXCHANGE_FILL",
            touch="2026-10-09T12:00:00+00:00")
        obs = observation(stale)
        obs["signal_proofs"] = current_signal_proofs(obs)
        counterfeit = copy.deepcopy(obs["signal_proofs"])
        counterfeit["observed_proofs"][0]["status"] = "CURRENT_CLOSED_BAR_TOUCH_TEETH_PASS_NOT_FILL"
        counterfeit["observed_proofs"][0]["is_current_eligible_observation"] = True
        obs["signal_proofs"] = counterfeit
        audited = audit_reasoning(obs, proposal("CLOSED_BAR_TOUCH_ONLY"))
        self.assertIn("NO_CAUSAL_CURRENT_SOURCE_EVIDENCE_FOR_CLAIM",
                      audited["hard_failures"])
        self.assertFalse(audited["execution_allowed"])


if __name__ == "__main__":
    unittest.main()
