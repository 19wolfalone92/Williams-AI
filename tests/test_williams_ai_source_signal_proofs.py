"""Expert-negative chart tests: source presence is NOT a new filled trade.

A language model can hallucinate a pretty fifth wave, stale fractal,
missing initial NTD fractal or fake Squat. The independent causal proof
ledger must still reject its unsupported current-entry claims.
"""
import unittest

from williams_ai.algorithm_guard import audit_reasoning
from williams_ai.chart_reader import analyze_chart
from williams_ai.causal_paper_lab import frozen_markets
from williams_ai.signal_proofs import current_signal_proofs,proof_supports_current_claim


NOW="2026-10-21T12:59:59+00:00"
YESTERDAY="2026-10-20T12:59:59+00:00"


def fake_chart(candidate=None):
    return {
        "symbol":"BTCUSDT","timeframe":"H1","as_of":NOW,
        "execution_allowed":False,"gaps":0,"candidates":[candidate] if candidate else [],
        "edition_evidence":{"NTD":{"entry_admitted":False,"signals":[]}},
        "tc1_five_magic_bullets":{"last_n_wave_candidates":[]},
    }


def decision(setup,side,entry="UNKNOWN",rules=None):
    return {
        "symbol":"BTCUSDT","as_of":NOW,"edition":"TC2","setup":setup,
        "side":side,"entry_state":entry,"exit_basis":"NONE",
        "rule_ids":rules or ["TC2.WM3"],
        "trade_action":None,"execution_allowed":False,
    }


class SourceBoundSignalProofTests(unittest.TestCase):
    def test_stale_wm3_touch_never_current_entry_even_with_unknown_entry_state(self):
        old={
            "type":"WM3_FRACTAL","side":"LONG",
            "state":"CLOSED_BAR_TOUCH_PASSES_TEETH_NOT_EXCHANGE_FILL",
            "entry_trigger":44000.,
            "price_touch_observed_at":YESTERDAY,"confirmed_at":YESTERDAY,
        }
        obs=fake_chart(old)
        proof=current_signal_proofs(obs)
        obs["signal_proofs"]=proof
        self.assertIn("OLD_OR_UNVERIFIED_FRACTAL_TOUCH_NOT_CURRENT_SIGNAL",
                      [x["status"] for x in proof["observed_proofs"]])
        self.assertFalse(proof_supports_current_claim(proof,"WM3","LONG","UNKNOWN"))
        audited=audit_reasoning(obs,decision("WM3","LONG"))
        self.assertIn("NO_CAUSAL_CURRENT_SOURCE_EVIDENCE_FOR_CLAIM",
                      audited["hard_failures"])
        self.assertFalse(audited["execution_allowed"])

    def test_pending_fractal_is_not_a_price_break_or_fill(self):
        obs=fake_chart({
            "type":"WM3_FRACTAL","side":"SHORT",
            "state":"AWAITING_PRICE_TRIGGER",
            "entry_trigger":42100.,"confirmed_at":YESTERDAY,
        })
        proof=current_signal_proofs(obs);obs["signal_proofs"]=proof
        self.assertTrue(proof_supports_current_claim(proof,"WM3","SHORT","PENDING_TRIGGER"))
        out=audit_reasoning(obs,decision("WM3","SHORT","PENDING_TRIGGER"))
        self.assertNotIn("NO_CAUSAL_CURRENT_SOURCE_EVIDENCE_FOR_CLAIM",
                         out["hard_failures"])
        self.assertFalse(out["execution_allowed"])
        self.assertIsNone(out["trade_action"])

    def test_current_touch_must_match_closed_bar_time(self):
        x={"type":"WM3_FRACTAL","side":"LONG",
           "state":"CLOSED_BAR_TOUCH_PASSES_TEETH_NOT_EXCHANGE_FILL",
           "entry_trigger":44000.,"price_touch_observed_at":NOW,"confirmed_at":YESTERDAY}
        obs=fake_chart(x)
        p=current_signal_proofs(obs)
        self.assertTrue(proof_supports_current_claim(p,"WM3","LONG","CLOSED_BAR_TOUCH_ONLY"))
        self.assertFalse(proof_supports_current_claim(p,"WM3","LONG","PENDING_TRIGGER"))
        self.assertFalse(proof_supports_current_claim(p,"WM3","SHORT","CLOSED_BAR_TOUCH_ONLY"))

    def test_exhaustion_is_not_proof_without_tick_volume(self):
        obs=fake_chart()
        obs["tc1_five_magic_bullets"]["last_n_wave_candidates"]=[{
            "observed_bullets":4,"unknown_bullets":1,
            "all_five_observed":False,
        }]
        book=current_signal_proofs(obs)
        x=next(x for x in book["observed_proofs"]
               if x["setup"]=="TC1_FIVE_MAGIC_BULLETS")
        self.assertEqual(x["status"],
            "INCOMPLETE_FIVE_SIGNS_NOT_ZERO_POINT_CONFIRMATION")
        self.assertFalse(proof_supports_current_claim(
            book,"TC1_FIVE_MAGIC_BULLETS","NONE","UNKNOWN"))

    def test_ntd_is_an_observation_not_a_first_entry_without_outside_fractal(self):
        obs=fake_chart()
        obs["edition_evidence"]["NTD"]["signals"]=[{
            "name":"AO_ZERO_CROSS","side":"LONG","formation":"CLOSED_INDICATOR_CANDIDATE",
        }]
        book=current_signal_proofs(obs)
        self.assertTrue(proof_supports_current_claim(book,"AO_ZERO_CROSS","LONG","UNKNOWN"))
        self.assertFalse(proof_supports_current_claim(book,"AO_ZERO_CROSS","LONG","PENDING_TRIGGER"))
        row=next(x for x in book["observed_proofs"] if x["setup"]=="AO_ZERO_CROSS")
        self.assertEqual(row["status"],
            "INDICATOR_SIGN_PRESENT_BUT_INITIAL_NTD_FRACTAL_GATE_MISSING")

    def test_real_closed_history_evidence_is_causal_15_markets(self):
        markets=frozen_markets()
        self.assertEqual(len(markets),15)
        seen=set()
        for symbol,market in markets.items():
            rows=market["bars"]
            # Distinct historical cutoffs, not the last market state only.
            for limit in (500,900,len(rows)):
                prefix=rows[:limit]
                obs=analyze_chart(symbol,"H1",prefix,
                                  as_of=prefix[-1]["close_time"],
                                  tick_size=market["tick_size"])
                book=obs["signal_proofs"]
                self.assertEqual(book["as_of"],obs["as_of"])
                self.assertFalse(book["execution_allowed"])
                self.assertIsNone(book["trade_action"])
                for p in book["observed_proofs"]:
                    self.assertEqual(p["as_of"],obs["as_of"])
                    self.assertFalse(p["execution_allowed"])
                    self.assertTrue(p["source_rule_id"].startswith(("TC1.","TC2.","NTD.")))
                    self.assertNotIn(p.get("setup"),("BUY","SELL","CLOSE"))
                    seen.add(p["setup"])
        self.assertIn("WM3",seen)
        self.assertIn("WM2",seen)


if __name__=="__main__":
    unittest.main()
