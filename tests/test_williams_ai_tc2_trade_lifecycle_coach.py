"""Actual closed-prefix TC2 education, no broker and no Qwen authorization."""
import unittest
from copy import deepcopy
from datetime import datetime, timedelta, timezone

from williams_ai.chart_reader import analyze_chart
from williams_ai.tc2_trade_lifecycle_coach import teach_tc2_trade_lifecycle


def bars(count=180):
    start=datetime(2026,10,1,tzinfo=timezone.utc)
    rows=[]
    for i in range(count):
        mid=150.0+0.16*i + (i%13-6)*0.43
        rows.append({
            "close_time":(start+timedelta(hours=i+1,milliseconds=-1)).isoformat(),
            "open":mid-0.2,"high":mid+1.0,"low":mid-1.2,
            "close":mid+0.3,"volume":50+i,"closed":True,
        })
    return rows


def prepared(data):
    return analyze_chart("ETHUSDT","H1",data,as_of=data[-1]["close_time"],
                         tick_size=0.01)


def audit_journal(data,events):
    return {"edition":"TC2","symbol":"ETHUSDT",
            "as_of":prepared(data)["as_of"],"events":events}


class BookGroundedLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.data=bars()
        self.view=prepared(self.data)

    def lesson(self, journal=None):
        return teach_tc2_trade_lifecycle(
            self.view,self.data,tick_size=0.01,journal=journal)

    def test_no_broker_orders_and_first_entry_can_be_any_wise_man(self):
        result=self.lesson()
        self.assertFalse(result["execution_allowed"])
        self.assertIsNone(result["trade_action"])
        self.assertFalse(result["model_weights_modified"])
        self.assertEqual(result["inverse_pyramid_book_illustration"],[1,5,4,3,2])
        self.assertEqual(result["edition"],"TC2")
        self.assertEqual(result["campaign_state"],"FLAT_READ_ONLY_STUDY")
        self.assertIn("TC2.WM3",{c["rule_id"] for c in result["source_rule_cards"]})
        self.assertTrue(all(not c["exchange_fill_known"]
                            for c in result["entry_observations"]))

    def test_journal_stopped_without_protection_never_teaches_trailing(self):
        opened="2026-10-02T00:00:00Z"
        journal=audit_journal(self.data,[
            {"at":opened,"type":"OPEN_FILL","side":"LONG","price":145,
             "exchange_confirmed":True}])
        result=self.lesson(journal)
        self.assertEqual(result["management"]["status"],
                         "HALT_FOR_PROTECTION_OR_FILL_AUDIT")
        self.assertIsNone(result["management"]["trailing_stop_paper_reference"])

    def test_unchanged_or_wider_stop_never_suggests_widening(self):
        opened="2026-10-02T00:00:00Z"
        journal=audit_journal(self.data,[
            {"at":opened,"type":"OPEN_FILL","side":"LONG","price":145,
             "exchange_confirmed":True},
            {"at":opened,"type":"STOP_ACK","stop_price":100,
             "exchange_confirmed":True}])
        result=self.lesson(journal)
        self.assertEqual(result["management"]["status"],
                         "REVIEW_MONOTONIC_TRAILING_STOP")
        self.assertGreater(float(result["management"]["trailing_stop_paper_reference"]),100)
        self.assertLess(float(result["management"]["trailing_stop_paper_reference"]),
                        float(self.data[-1]["close"]))
        stronger=deepcopy(journal)
        stronger["events"][1]["stop_price"]=float(self.data[-1]["close"])-0.1
        new=self.lesson(stronger)
        self.assertEqual(new["management"]["status"],"HOLD_PROTECTIVE_STOP")
        self.assertIsNone(new["management"]["trailing_stop_paper_reference"])

    def test_opposite_signal_without_fill_cannot_close_or_reverse(self):
        t="2026-10-02T00:00:00Z"
        journal=audit_journal(self.data,[
            {"at":t,"type":"OPEN_FILL","side":"LONG","price":145,
             "exchange_confirmed":True},
            {"at":t,"type":"STOP_ACK","stop_price":100,
             "exchange_confirmed":True},
            {"at":"2026-10-02T01:00:00Z",
             "type":"OPPOSITE_SIGNAL","side":"SHORT"}])
        report=self.lesson(journal)
        self.assertEqual(report["campaign_state"],
                         "CLAIMED_OPEN_POSITION_REQUIRE_EXCHANGE_RECONCILIATION")
        self.assertTrue(report["management"]["opposite_signal_alone_must_not_flatten"])
        self.assertEqual(report["exit_requires_one_of"][0],
                         "AUTHENTICATED_PROTECTIVE_SL_FILL")
        self.assertIsNone(report["trade_action"])

    def test_old_history_alone_not_award_next_price_break(self):
        past=self.lesson()
        earlier=self.data[:-1]
        past_view=prepared(earlier)
        earlier_lesson=teach_tc2_trade_lifecycle(
            past_view,earlier,tick_size=0.01)
        self.assertEqual(earlier_lesson["as_of"],past_view["as_of"])
        self.assertNotEqual(earlier_lesson["as_of"],past["as_of"])
        self.assertTrue(all(x["price_breakout_requires_next_observation"]
                            for x in earlier_lesson["entry_observations"]))

    def test_fail_closed_gaps_future_and_wrong_profile(self):
        x=deepcopy(self.view);x["execution_allowed"]=True
        with self.assertRaisesRegex(ValueError,"READ_ONLY"):
            teach_tc2_trade_lifecycle(x,self.data,tick_size=0.01)
        with self.assertRaisesRegex(ValueError,"PRICE_OR_TICK"):
            self.lesson_with_tick(0)
        x=deepcopy(self.view);x["as_of"]=self.data[-2]["close_time"]
        with self.assertRaises(ValueError):
            teach_tc2_trade_lifecycle(x,self.data,tick_size=0.01)
        broken=deepcopy(self.data);broken[-1]["close_time"]=(datetime.fromisoformat(
            broken[-1]["close_time"])+timedelta(hours=2)).isoformat()
        with self.assertRaises(ValueError):
            teach_tc2_trade_lifecycle(self.view,broken,tick_size=0.01)
        with self.assertRaisesRegex(ValueError,"JOURNAL_CHART"):
            self.lesson({"symbol":"ETHUSDT","edition":"NTD",
                         "as_of":self.view["as_of"],"events":[]})

    def lesson_with_tick(self,tick):
        return teach_tc2_trade_lifecycle(self.view,self.data,tick_size=tick)


if __name__=="__main__":
    unittest.main()
