"""Historical H1 candles -> source proof -> portfolio sizing (paper only)."""
from datetime import datetime, timedelta, timezone
import copy
import unittest

from williams_ai.paper_from_chart import paper_wm3_from_closed_h1


def history(side="LONG", *, touch=True):
    begin=datetime(2026, 8, 1, tzinfo=timezone.utc)
    rows=[]
    for i in range(240):
        close=100.0 + 0.01*(i%3)
        op=close
        high=close+0.5
        low=close-0.5
        if i==235:
            if side=="LONG":
                high=110.0
            else:
                low=90.0
        if i==239 and touch:
            if side=="LONG":
                high=111.5
                close=108.0
            else:
                low=88.5
                close=92.0
        rows.append({
            "close_time":(begin+timedelta(hours=i)).isoformat(),
            "open":op,"high":high,"low":low,"close":close,
            "volume":100.0,"closed":True,
        })
    return rows


def portfolio(rows):
    return {"as_of":rows[-1]["close_time"],"equity":1000.0,
            "session_start_equity":1000.0,"daily_realized_pnl":0.0,
            "positions":[]}


def check(side="LONG",touch=True,positions=None):
    rows=history(side,touch=touch)
    p=portfolio(rows)
    if positions:
        p["positions"]=positions
    result=paper_wm3_from_closed_h1(
        "ETHUSDT",rows,p,side=side,tick_size=0.01,lot_step=0.001,
    )
    return rows,result


class RebuiltChartWM3PaperTests(unittest.TestCase):
    def test_current_long_five_bar_fractal_builds_paper_passport_not_fill(self):
        rows,o=check("LONG")
        self.assertEqual(o["study_status"],"PAPER_SIZED_NOT_TRADABLE")
        self.assertEqual(o["paper_method"],"CURRENT_CLOSED_H1_WM3_TOUCH_AUDITED_AS_OF_CLOSE")
        self.assertEqual(o["paper_structural_stop_basis"],"FIVE_PREVIOUS_CLOSED_H1_BAR_PROXY")
        self.assertEqual(o["observed_touch_time"],rows[-1]["close_time"])
        self.assertLess(o["source_confirmation_time"],o["observed_touch_time"])
        self.assertGreater(o["hypothetical_quantity"],0)
        self.assertLessEqual(o["hypothetical_risk_including_friction"],5.0)
        self.assertGreaterEqual(o["hypothetical_entry_reference"],110.01)
        self.assertFalse(o["execution_allowed"])
        self.assertIsNone(o["trade_action"])
        self.assertFalse(o["exchange_stop_acknowledged"])
        self.assertTrue(o["requires_new_future_signal_for_actual_execution"])

    def test_short_mirror_requires_price_breakout(self):
        _,o=check("SHORT")
        self.assertEqual(o["study_status"],"PAPER_SIZED_NOT_TRADABLE")
        self.assertGreater(o["hypothetical_quantity"],0)
        self.assertLessEqual(o["hypothetical_entry_reference"],89.99)
        self.assertFalse(o["execution_allowed"])

    def test_fractal_pending_without_current_touch_is_not_a_fill(self):
        _,o=check("LONG",touch=False)
        self.assertEqual(o["study_status"],"BLOCKED")
        self.assertIn("NO_CURRENT_CLOSED_WM3_TOUCH",o["reason_codes"])
        self.assertEqual(o["hypothetical_quantity"],0)

    def test_one_active_campaign_per_pair_even_after_book_signal(self):
        x={"symbol":"ETHUSDT","side":"LONG","entry":100.0,
           "stop":90.0,"qty":0.5,"stop_confirmed":True}
        _,o=check("LONG",positions=[x])
        self.assertEqual(o["study_status"],"BLOCKED")
        self.assertIn("CAMPAIGN_ALREADY_OWNS_SYMBOL",o["reason_codes"])

    def test_open_candle_and_gapped_history_are_not_repaired(self):
        rows=history()
        p=portfolio(rows)
        rows[-1]["closed"]=False
        with self.assertRaises(ValueError):
            paper_wm3_from_closed_h1("ETHUSDT",rows,p,side="LONG",
                                     tick_size=.01,lot_step=.001)
        rows=history()
        rows[-1]["close_time"]=(datetime.fromisoformat(rows[-1]["close_time"])
                                 +timedelta(hours=3)).isoformat()
        p=portfolio(rows)
        out=paper_wm3_from_closed_h1("ETHUSDT",rows,p,side="LONG",
                                    tick_size=.01,lot_step=.001)
        self.assertEqual(out["study_status"],"BLOCKED")
        self.assertIn("GAPS_OR_STALE_H1_CANDLE",out["reason_codes"])


if __name__=="__main__":
    unittest.main()
