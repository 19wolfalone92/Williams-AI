"""No network calls: signed Demo recovery failure injection and second-quote gating."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from williams_ai.demo_account_recovery import (
    inspect_demo_after_restart, inspect_demo_account_before_first_order,
)
from williams_ai.demo_futures import DemoSafetyError
from williams_ai.demo_pilot_readiness import (
    OneCampaignReservation, verify_requoted_manual_pilot,
)
from test_williams_ai_demo_pilot import small_preview


class FakeSignedDemo:
    def __init__(self, *, position="0.1", stop=True):
        self.position=position
        self.stop=stop
        self.requests=[]

    def request(self, method, path, args=None, *, signed=False):
        self.requests.append((method,path,args,signed))
        if method!="GET" or not signed:
            raise AssertionError("Recovery attempted unsigned or mutating HTTP")
        if path=="/fapi/v1/positionSide/dual":
            return {"dualSidePosition": False}
        if path=="/fapi/v3/positionRisk":
            if args!={"symbol":"ETHUSDT"}: raise AssertionError("Unscoped positions")
            return [{"symbol":"ETHUSDT","positionSide":"BOTH",
                     "positionAmt":self.position}]
        if path=="/fapi/v1/openAlgoOrders":
            if args!={"symbol":"ETHUSDT"}: raise AssertionError("Unscoped algo orders")
            if not self.stop:
                return []
            return [{
                "symbol":"ETHUSDT","side":"SELL","positionSide":"BOTH",
                "orderType":"STOP_MARKET","algoStatus":"NEW",
                "workingType":"MARK_PRICE","closePosition":True,
                "triggerPrice":"95",
            }]
        raise AssertionError("Unapproved read-only API: "+path)


class DemoAccountRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.journal=OneCampaignReservation(Path(self.tmp.name)/"once.sqlite3")
        self.preview,_=small_preview()

    def reserve(self):
        self.journal.reserve_once(self.preview,now_ms=2_000_000_000_000)

    def test_no_reservation_never_accesses_private_account(self):
        out=inspect_demo_after_restart(None,self.journal)
        self.assertEqual(out["status"],"NO_RESERVED_PILOT_DO_NOT_ASSUME_FLAT")
        self.assertFalse(out["authenticated_checks_performed"])
        self.assertFalse(out["execution_allowed"])

    def test_matching_stop_is_never_mistaken_for_verified_ownership(self):
        self.reserve()
        client=FakeSignedDemo()
        before=self.journal.status()
        result=inspect_demo_after_restart(client,self.journal)
        self.assertEqual(result["status"],"POSITION_WITH_POSSIBLE_STOP_OWNERSHIP_NOT_PROVEN")
        self.assertEqual(result["possible_protective_stops"],1)
        self.assertFalse(result["stop_acknowledged"])
        self.assertTrue(result["stop_belongs_to_pilot_unverified"])
        self.assertFalse(result["execution_allowed"])
        self.assertEqual(before,self.journal.status())
        self.assertEqual([q[1] for q in client.requests],[
            "/fapi/v1/positionSide/dual",
            "/fapi/v3/positionRisk",
            "/fapi/v1/openAlgoOrders"])
        self.assertTrue(all(q[0]=="GET" and q[3] for q in client.requests))

    def test_unprotected_and_wrong_side_must_be_flagged(self):
        self.reserve()
        no_sl=inspect_demo_after_restart(FakeSignedDemo(stop=False),self.journal)
        self.assertEqual(no_sl["status"],"POSITION_WITH_UNVERIFIED_PROTECTION")
        wrong=inspect_demo_after_restart(FakeSignedDemo(position="-0.1"),self.journal)
        self.assertEqual(wrong["status"],"EXCHANGE_POSITION_DIRECTION_CONFLICT")
        partial=inspect_demo_after_restart(FakeSignedDemo(position="0.05"),self.journal)
        self.assertEqual(partial["status"],"EXCHANGE_POSITION_SIZE_MISMATCH")

    def test_bound_client_id_exact_signed_stop_and_price_are_verified(self):
        self.journal.reserve_once(self.preview,now_ms=2_000_000_000_000,
                                  ticket="a"*32)
        bound=self.journal.bound_client_ids()
        self.assertEqual(bound["stop_client_algo_id"],"WAISL"+"a"*23)
        self.assertEqual(bound["entry_client_order_id"],"WAIEN"+"a"*23)
        self.assertEqual(bound["expected_stop_price"],"95")
        class OwnedVenue(FakeSignedDemo):
            def request(self,method,path,args=None,*,signed=False):
                ret=super().request(method,path,args,signed=signed)
                if path=="/fapi/v1/openAlgoOrders":
                    ret[0]["clientAlgoId"]=bound["stop_client_algo_id"]
                return ret
        out=inspect_demo_after_restart(OwnedVenue(),self.journal)
        self.assertEqual(out["status"],
            "POSITION_WITH_SIGNED_MATCHING_BOUND_STOP_NEEDS_FILL_AUDIT")
        self.assertTrue(out["bound_exchange_stop_identity_matched"])
        self.assertTrue(out["stop_acknowledged"])
        self.assertTrue(out["entry_exchange_fill_not_independently_reconciled"])
        self.assertEqual(len(self.journal.status()["reservations"]),1)

    def test_matching_price_but_foreign_stop_does_not_prove_ownership(self):
        self.journal.reserve_once(self.preview,now_ms=2_000_000_000_000,
                                  ticket="b"*32)
        out=inspect_demo_after_restart(FakeSignedDemo(),self.journal)
        self.assertFalse(out["stop_acknowledged"])
        self.assertFalse(out["bound_exchange_stop_identity_matched"])
        self.assertEqual(out["status"],"POSITION_WITH_UNVERIFIED_PROTECTION")

    def test_extra_competing_stop_never_counts_as_safe_protection(self):
        self.journal.reserve_once(self.preview,now_ms=2_000_000_000_000,
                                  ticket="d"*32)
        class CompetingVenue(FakeSignedDemo):
            def request(self,method,path,args=None,*,signed=False):
                rows=super().request(method,path,args,signed=signed)
                if path=="/fapi/v1/openAlgoOrders":
                    rows[0]["clientAlgoId"]="WAISL"+"d"*23
                    rows.append(dict(rows[0],clientAlgoId="UNRELATED_DEMO_STOP"))
                return rows
        out=inspect_demo_after_restart(CompetingVenue(),self.journal)
        self.assertEqual(out["status"],
            "POSITION_WITH_EXTRA_OPEN_STOP_CANDIDATES_REVIEW")
        self.assertFalse(out["stop_acknowledged"])
        self.assertEqual(out["possible_protective_stops"],2)
        self.assertEqual(len(self.journal.status()["reservations"]),1)

    def test_bound_stop_with_modified_stop_price_raises_and_keeps_lock(self):
        self.journal.reserve_once(self.preview,now_ms=2_000_000_000_000,
                                  ticket="c"*32)
        class WrongStop(FakeSignedDemo):
            def request(self,method,path,args=None,*,signed=False):
                ret=super().request(method,path,args,signed=signed)
                if path=="/fapi/v1/openAlgoOrders":
                    ret[0]["clientAlgoId"]="WAISL"+"c"*23
                    ret[0]["triggerPrice"]="94"
                return ret
        with self.assertRaisesRegex(DemoSafetyError,
                                    "BOUND_DEMO_STOP_ID_HAS_WRONG_PARAMETERS"):
            inspect_demo_after_restart(WrongStop(),self.journal)
        self.assertEqual(len(self.journal.status()["reservations"]),1)

    def test_invalid_ticket_and_rollback_never_create_reserved_campaign(self):
        for ticket in ("", "z"*32, "a"*31, "A"*32, True):
            with self.subTest(ticket=ticket):
                with self.assertRaisesRegex(DemoSafetyError,"INVALID_DEMO_INTENT_TICKET"):
                    self.journal.reserve_once(self.preview,
                        now_ms=2_000_000_000_000,ticket=ticket)
        self.assertIsNone(self.journal.bound_client_ids())
        self.assertEqual(self.journal.status()["reservations"],[])

    def test_flat_with_orphan_stop_does_not_unlock(self):
        self.reserve()
        out=inspect_demo_after_restart(FakeSignedDemo(position="0"),self.journal)
        self.assertEqual(out["status"],"FLAT_WITH_OPEN_STOP_CANDIDATE_MANUAL_REVIEW")
        self.assertEqual(len(self.journal.status()["reservations"]),1)

    def test_bad_or_ambiguous_exchange_responses_fail_closed(self):
        self.reserve()
        for bad in ("NaN","Infinity","not-a-position"):
            with self.subTest(position=bad):
                with self.assertRaisesRegex(DemoSafetyError,"AMOUNT_INVALID"):
                    inspect_demo_after_restart(FakeSignedDemo(position=bad),self.journal)
        class UnknownResponse(FakeSignedDemo):
            def request(self,method,path,args=None,*,signed=False):
                if path=="/fapi/v1/openAlgoOrders":
                    return {"not":"a list"}
                return super().request(method,path,args,signed=signed)
        with self.assertRaisesRegex(DemoSafetyError,"LIST_UNTRUSTED"):
            inspect_demo_after_restart(UnknownResponse(),self.journal)

    def test_journal_malformed_state_blocks_all_signed_calls(self):
        class CorruptJournal:
            def status(self):
                return {"reservations":[{"symbol":"BTCUSDT","side":"BUY",
                                         "quantity":"0.1",
                                         "state":"UNKNOWN_UNTIL_EXCHANGE_RECONCILED"}]}
        client=FakeSignedDemo()
        with self.assertRaisesRegex(DemoSafetyError,"JOURNAL_INCONSISTENT"):
            inspect_demo_after_restart(client,CorruptJournal())
        self.assertFalse(client.requests)


class SignedDemoReadOnlyPreflightTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.journal=OneCampaignReservation(Path(self.tmp.name)/"once.sqlite3")

    class Account:
        def __init__(self, *,open_orders=None,open_algos=None,
                     amount="0",hedge=False,other_amount="0",bad_list=False):
            self.calls=[]
            self.open_orders=[] if open_orders is None else open_orders
            self.open_algos=[] if open_algos is None else open_algos
            self.amount=amount
            self.hedge=hedge
            self.other_amount=other_amount
            self.bad_list=bad_list

        def request(self,method,path,args=None,*,signed=False):
            if method!="GET" or not signed:
                raise AssertionError("Never perform mutating or unsigned checks")
            self.calls.append((method,path,args,signed))
            if path=="/fapi/v1/positionSide/dual":
                return {"dualSidePosition":self.hedge}
            if path=="/fapi/v3/positionRisk":
                if args is not None:raise AssertionError("Must inspect entire account")
                if self.bad_list:return []
                return [
                    {"symbol":"ETHUSDT","positionSide":"BOTH",
                     "positionAmt":self.amount},
                    {"symbol":"BTCUSDT","positionSide":"BOTH",
                     "positionAmt":self.other_amount},
                ]
            if path=="/fapi/v1/openOrders":
                if args!={"symbol":"ETHUSDT"}:raise AssertionError("Wrong scope")
                return self.open_orders
            if path=="/fapi/v1/openAlgoOrders":
                if args!={"symbol":"ETHUSDT"}:raise AssertionError("Wrong scope")
                return self.open_algos
            raise AssertionError("Unsupported DEMO signed call")

    def test_signed_preflight_never_posts_and_does_not_consume_pilot(self):
        client=self.Account()
        before=self.journal.status()
        report=inspect_demo_account_before_first_order(client,self.journal)
        self.assertEqual(report["status"],
            "ACCOUNT_FLAT_AND_ETHUSDT_NO_OPEN_ORDERS_AT_CHECK_TIME")
        self.assertTrue(report["signed_api_key_works_on_demo_host"])
        self.assertFalse(report["pre_entry_close_position_stop_acceptance_tested"])
        self.assertFalse(report["execution_allowed"])
        self.assertTrue(all(call[0]=="GET" and call[3] for call in client.calls))
        self.assertEqual(len(client.calls),4)
        self.assertEqual(before,self.journal.status())

    def test_positions_on_other_pair_and_any_pending_orders_block(self):
        for opts in (
            {"amount":"0.1"}, {"other_amount":"-0.02"}, {"hedge":True},
            {"open_orders":[{"orderId":123}]},
            {"open_algos":[{"algoId":456}]},
            {"amount":"NaN"},{"bad_list":True},
        ):
            with self.subTest(opts=opts):
                with self.assertRaises(DemoSafetyError):
                    inspect_demo_account_before_first_order(
                        self.Account(**opts),self.journal)
        self.assertEqual(self.journal.status()["reservations"],[])

    def test_existing_pilot_lock_blocks_without_any_signed_request(self):
        from test_williams_ai_demo_pilot import small_preview
        preview,_=small_preview()
        self.journal.reserve_once(preview,now_ms=2_000_000_000_000)
        venue=self.Account()
        with self.assertRaisesRegex(DemoSafetyError,"PILOT_LOCK_EXISTS"):
            inspect_demo_account_before_first_order(venue,self.journal)
        self.assertEqual(venue.calls,[])


class PilotSecondQuoteTests(unittest.TestCase):
    def setUp(self):
        self.preview,_=small_preview()

    def proposal(self, **kw):
        data={"symbol":"ETHUSDT","side":"BUY",
              "quantity":"0.1","stop_price":"95",
              "notional_usdt":"10.0","risk_usdt":"0.5"}
        data.update(kw)
        return SimpleNamespace(**data)

    def test_small_requote_allowed_but_mark_move_rejected(self):
        verify_requoted_manual_pilot(self.preview,self.proposal(
            notional_usdt="10.02",risk_usdt="0.52"))
        with self.assertRaisesRegex(DemoSafetyError,"MARK_MOVED"):
            verify_requoted_manual_pilot(self.preview,self.proposal(
                notional_usdt="10.1",risk_usdt="0.6"))

    def test_quantity_stop_and_risk_limit_never_change(self):
        for kw in (
            {"quantity":"0.2"},
            {"stop_price":"96"},
            {"risk_usdt":"1.01"},
            {"notional_usdt":"50.01"},
            {"symbol":"BTCUSDT"},
            {"side":"SELL"},
        ):
            with self.subTest(kw=kw):
                with self.assertRaises(DemoSafetyError):
                    verify_requoted_manual_pilot(self.preview,self.proposal(**kw))


if __name__=="__main__":
    unittest.main()
