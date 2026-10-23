"""Paper-only shared-capital and correlated crypto risk-budget regressions."""
import copy
import math
import unittest

from williams_ai.algorithm_guard import audit_reasoning
from williams_ai.portfolio_risk_research import PaperRiskLimits, size_paper_hypothesis


NOW="2026-10-10T12:00:00+00:00"
BEFORE="2026-10-10T09:00:00+00:00"


def accepted_wm3(side="LONG"):
    obs={
        "symbol":"ETHUSDT","timeframe":"H1","as_of":NOW,
        "execution_allowed":False,"gaps":0,
        "candidates":[{
            "type":"WM3_FRACTAL","side":side,
            "state":"CLOSED_BAR_TOUCH_PASSES_TEETH_NOT_EXCHANGE_FILL",
            "confirmed_at":BEFORE,"price_touch_observed_at":NOW,
            "entry_trigger":101.0 if side=="LONG" else 99.0,
        }],
    }
    claim={
        "symbol":"ETHUSDT","as_of":NOW,"edition":"TC2",
        "setup":"WM3","side":side,
        "entry_state":"CLOSED_BAR_TOUCH_ONLY",
        "exit_basis":"NONE","rule_ids":["TC2.WM3"],
        "trade_action":None,"execution_allowed":False,
    }
    result=audit_reasoning(obs,claim)
    assert result["hard_failures"]==[],result
    assert result["execution_allowed"] is False
    return result


def signal(side="LONG"):
    return {
        "symbol":"ETHUSDT","as_of":NOW,"confirmed_at":BEFORE,
        "side":side,"entry":100.0,
        "stop":90.0 if side=="LONG" else 110.0,
        "tick_size":0.01,"lot_step":0.001,
        "entry_state":"CLOSED_BAR_TOUCH_ONLY",
        "source_audit":accepted_wm3(side),
    }


def portfolio(positions=None):
    return {
        "as_of":NOW,"equity":1000.0,
        "session_start_equity":1000.0,
        "daily_realized_pnl":0.0,
        "positions":[] if positions is None else positions,
    }


def existing(symbol="BTCUSDT",side="LONG",qty=0.5,stop=90.0):
    return {
        "symbol":symbol,"side":side,
        "entry":100.0,"stop":stop,"qty":qty,
        "stop_confirmed":True,
    }


class PortfolioPaperGateTests(unittest.TestCase):
    def assert_never_executes(self, out):
        self.assertIs(out["execution_allowed"],False)
        self.assertIsNone(out["trade_action"])
        self.assertIs(out["exchange_stop_acknowledged"],False)

    def assert_blocked(self, s, p, expected):
        output=size_paper_hypothesis(s,p)
        self.assert_never_executes(output)
        self.assertEqual(output["study_status"],"BLOCKED")
        self.assertEqual(output["hypothetical_quantity"],0.0)
        self.assertIn(expected,output["reason_codes"])

    def test_valid_wm3_paper_qty_uses_fee_and_slippage_and_never_orders(self):
        out=size_paper_hypothesis(signal(),portfolio())
        self.assert_never_executes(out)
        self.assertEqual(out["study_status"],"PAPER_SIZED_NOT_TRADABLE")
        self.assertGreater(out["hypothetical_quantity"],0)
        self.assertLessEqual(out["hypothetical_notional"],500.0)
        self.assertLessEqual(out["hypothetical_risk_including_friction"],5.0)
        self.assertLess(out["hypothetical_risk_including_friction"],5.000001)

    def test_short_mirror_same_budget(self):
        long=size_paper_hypothesis(signal("LONG"),portfolio())
        short=size_paper_hypothesis(signal("SHORT"),portfolio())
        self.assert_never_executes(short)
        self.assertEqual(long["hypothetical_quantity"],short["hypothetical_quantity"])

    def test_opposite_stop_or_tiny_stop_are_blocked(self):
        s=signal()
        s["stop"]=101.0
        self.assert_blocked(s,portfolio(),"NO_VALID_STRUCTURAL_STOP_DISTANCE")
        s["stop"]=99.99
        self.assert_blocked(s,portfolio(),"NO_VALID_STRUCTURAL_STOP_DISTANCE")

    def test_existing_position_on_same_pair_prevents_another_campaign(self):
        self.assert_blocked(signal(),portfolio([existing("ETHUSDT")]),
                            "CAMPAIGN_ALREADY_OWNS_SYMBOL")

    def test_unverified_open_position_stop_blocks_everything(self):
        x=existing();x["stop_confirmed"]=False
        self.assert_blocked(signal(),portfolio([x]),
                            "UNPROTECTED_OR_UNVERIFIED_PAPER_POSITION")

    def test_correlated_crypto_stops_share_risk_budget(self):
        # BTC stop distance ~ 10% times 1 unit ~= $10 > 1% portfolio cap.
        blocked=portfolio([existing(qty=1.0)])
        self.assert_blocked(signal(),blocked,"CRYPTO_CORRELATION_BUDGET_EXHAUSTED")
        # Existing risk below budget must reduce the available quantity.
        fresh=size_paper_hypothesis(signal(),portfolio())
        occupied=size_paper_hypothesis(signal(),portfolio([existing(qty=0.5)]))
        self.assertEqual(occupied["study_status"],"PAPER_SIZED_NOT_TRADABLE")
        self.assertLessEqual(occupied["hypothetical_quantity"],
                             fresh["hypothetical_quantity"])

    def test_realized_daily_loss_lock_fails_closed(self):
        p=portfolio()
        p["daily_realized_pnl"]=-20.0
        self.assert_blocked(signal(),p,"DAILY_LOSS_LOCKED")

    def test_stale_market_snapshot_and_future_confirmation_fail_closed(self):
        p=portfolio()
        p["as_of"]="2026-10-10T11:00:00+00:00"
        self.assert_blocked(signal(),p,"FUTURE_CONFIRMATION_OR_STALE_PORTFOLIO")
        s=signal()
        s["confirmed_at"]="2026-10-10T09:00:00-04:00" # 13:00Z is future
        self.assert_blocked(s,portfolio(),"FUTURE_CONFIRMATION_OR_STALE_PORTFOLIO")
        # Equivalent ISO timezone offsets of the same absolute instant work.
        p=portfolio()
        p["as_of"]="2026-10-10T17:00:00+05:00"
        out=size_paper_hypothesis(signal(),p)
        self.assertEqual(out["study_status"],"PAPER_SIZED_NOT_TRADABLE")

    def test_llm_momentum_claim_or_unverified_audit_cannot_be_sized(self):
        s=signal()
        s["source_audit"]=copy.deepcopy(s["source_audit"])
        s["source_audit"]["setup"]="WM2"
        self.assert_blocked(s,portfolio(),"SOURCE_PROOF_NOT_INDEPENDENTLY_ACCEPTED")
        s=signal()
        s["entry_state"]="PENDING_TRIGGER"
        self.assert_blocked(s,portfolio(),
                            "NO_CURRENT_VERIFIED_PRICE_TOUCH_FOR_PAPER_SIZING")
        s=signal()
        s["source_audit"]["verdict"]="UNSUPPORTED_CLAIM"
        self.assert_blocked(s,portfolio(),"SOURCE_PROOF_NOT_INDEPENDENTLY_ACCEPTED")

    def test_no_proof_can_become_exchange_fill_or_stop(self):
        x=size_paper_hypothesis(signal(),portfolio())
        self.assertNotIn("order",x)
        self.assertFalse(x["execution_allowed"])
        self.assertFalse(x["exchange_stop_acknowledged"])

    def test_invalid_position_and_inputs_rejected_not_guessed(self):
        s=signal()
        s["entry"]=float("nan")
        with self.assertRaises(ValueError):size_paper_hypothesis(s,portfolio())
        s=signal()
        s["lot_step"]=0
        with self.assertRaises(ValueError):size_paper_hypothesis(s,portfolio())
        with self.assertRaises(ValueError):
            PaperRiskLimits(max_single_idea_risk=float("inf"))
        x=existing()
        x["stop"]=0.0
        with self.assertRaises(ValueError):
            size_paper_hypothesis(signal(),portfolio([x]))

    def test_lot_rounding_never_expands_risk(self):
        s=signal()
        s["lot_step"]=0.07
        out=size_paper_hypothesis(s,portfolio())
        self.assert_never_executes(out)
        self.assertLessEqual(out["hypothetical_risk_including_friction"],5.0+1e-7)
        if out["hypothetical_quantity"]>0:
            self.assertAlmostEqual(out["hypothetical_quantity"]/0.07,
                                   round(out["hypothetical_quantity"]/0.07),places=6)


if __name__=="__main__":
    unittest.main()
