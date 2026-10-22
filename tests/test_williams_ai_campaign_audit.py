"""Historical event contract: inspect execution evidence without executing trades."""
import unittest
from copy import deepcopy

from williams_ai.campaign_audit import audit_campaign


def row(kind, minute, **extra):
    return {"type": kind, "at": f"2026-10-20T10:{minute:02d}:00Z", **extra}


def journal(events, *, edition="TC2"):
    return {
        "symbol": "ETHUSDT", "edition": edition,
        "as_of": "2026-10-20T11:00:00Z",
        "events": events,
        "api_key": "CANARY_NEVER_FORWARD",
        "account_id": "CANARY_PRIVATE_ACCOUNT",
    }


def open_long():
    return row("OPEN_FILL", 0, side="LONG", price=100, exchange_confirmed=True)


def stop_long(price=95, minute=0):
    return row("STOP_ACK", minute, stop_price=price, exchange_confirmed=True)


def codes(report):
    return [f["code"] for f in report["findings"]]


class CampaignExecutionEvidenceTests(unittest.TestCase):
    def test_immediate_stop_ack_and_opposite_signal_do_not_close(self):
        events = [open_long(), stop_long(), row("OPPOSITE_SIGNAL", 1, side="SHORT")]
        result = audit_campaign(journal(events))
        self.assertEqual(result["position"]["status"], "OPEN")
        self.assertTrue(result["position"]["stop_confirmed_in_journal"])
        self.assertEqual(result["campaigns_closed"], 0)
        self.assertNotIn("EXPOSURE_INTERVAL_BEFORE_FIRST_STOP_ACK", codes(result))
        self.assertFalse(result["execution_allowed"])
        self.assertIsNone(result["trade_action"])
        self.assertNotIn("CANARY_NEVER_FORWARD", str(result))
        self.assertNotIn("CANARY_PRIVATE_ACCOUNT", str(result))

    def test_delayed_stop_ack_highlights_exposure_window(self):
        r = audit_campaign(journal([open_long(), stop_long(minute=3)]))
        self.assertIn("EXPOSURE_INTERVAL_BEFORE_FIRST_STOP_ACK", codes(r))
        self.assertTrue(r["position"]["stop_confirmed_in_journal"])

    def test_open_without_ack_is_critical_at_cutoff(self):
        r = audit_campaign(journal([open_long()]))
        self.assertIn("OPEN_POSITION_WITHOUT_CONFIRMED_PROTECTIVE_SL", codes(r))

    def test_failed_stop_ack_does_not_create_protection(self):
        r = audit_campaign(journal([open_long(),
                    row("STOP_ACK",1,stop_price=95,exchange_confirmed=False)]))
        self.assertIn("STOP_UNCONFIRMED_BY_EXCHANGE", codes(r))
        self.assertFalse(r["position"]["stop_confirmed_in_journal"])

    def test_stop_widen_long_and_short(self):
        r = audit_campaign(journal([open_long(),stop_long(),
              row("STOP_REPLACE_ACK",2,stop_price=94,exchange_confirmed=True)]))
        self.assertIn("PROTECTIVE_STOP_WIDENED", codes(r))
        self.assertEqual(r["position"]["stop_price"],95)
        short = [row("OPEN_FILL",0,side="SHORT",price=100,exchange_confirmed=True),
                 row("STOP_ACK",0,stop_price=105,exchange_confirmed=True),
                 row("STOP_REPLACE_ACK",1,stop_price=106,exchange_confirmed=True)]
        self.assertIn("PROTECTIVE_STOP_WIDENED",codes(audit_campaign(journal(short))))

    def test_cancel_of_protective_stop_is_critical(self):
        r = audit_campaign(journal([open_long(),stop_long(),
                  row("STOP_CANCEL_ACK",1,exchange_confirmed=True)]))
        self.assertIn("PROTECTIVE_STOP_CANCELLED_WHILE_OPEN",codes(r))
        self.assertFalse(r["position"]["stop_confirmed_in_journal"])

    def test_opposite_wise_man_alone_cannot_justify_exit(self):
        seq=[open_long(),stop_long(),
             row("OPPOSITE_SIGNAL",1,side="SHORT"),
             row("CLOSE_FILL",2,reason="OPPOSITE_TRIGGER",
                 exchange_confirmed=True,opposite_trigger_fill_confirmed=False)]
        r=audit_campaign(journal(seq))
        self.assertIn("OPPOSITE_SIGNAL_MISTAKEN_FOR_EXECUTED_REVERSAL",codes(r))
        self.assertEqual(r["campaigns_closed"],1)

    def test_verified_reversal_vs_stop_vs_manual_exit(self):
        for reason,extra in (
            ("SL_TRIGGERED",{}),
            ("OPPOSITE_TRIGGER",{"opposite_trigger_fill_confirmed":True}),
            ("MANUAL",{}),
        ):
            with self.subTest(reason=reason):
                seq=[open_long(),stop_long(),
                     row("CLOSE_FILL",2,reason=reason,exchange_confirmed=True,**extra)]
                r=audit_campaign(journal(seq))
                self.assertEqual(r["campaigns_closed"],1)
                self.assertNotIn("OPPOSITE_SIGNAL_MISTAKEN_FOR_EXECUTED_REVERSAL",codes(r))
                self.assertNotIn("CLOSE_REASON_UNKNOWN",codes(r))

    def test_no_future_unsorted_or_unknown_profile(self):
        events=[open_long(),row("RESTART",1)]
        x=journal(events)
        with self.assertRaisesRegex(ValueError,"TC2"):
            audit_campaign(journal(events,edition="NTD"))
        with self.assertRaisesRegex(ValueError,"chronologically"):
            audit_campaign(journal(list(reversed(events))))
        x["events"].append({"type":"CLOSE_FILL","at":"2026-10-20T12:00:00Z",
                            "exchange_confirmed":True})
        with self.assertRaisesRegex(ValueError,"NO_LOOKAHEAD"):
            audit_campaign(x)

    def test_report_cannot_change_with_unobserved_future_events(self):
        old=journal([open_long(),stop_long()])
        first=audit_campaign(old)
        extended=deepcopy(old)
        extended["events"].append(row("OPPOSITE_SIGNAL",3,side="SHORT"))
        second=audit_campaign(extended)
        self.assertEqual(first["findings"],second["findings"])
        self.assertEqual(first["position"],second["position"])


if __name__ == "__main__":
    unittest.main()
