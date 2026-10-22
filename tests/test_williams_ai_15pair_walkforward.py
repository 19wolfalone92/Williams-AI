"""Causal 15-pair lab and six-page wave-diagram reference regressions."""
import json
from datetime import datetime,timedelta,timezone
import unittest

from williams_ai.causal_paper_lab import (
    frozen_markets,simulate_one,run_15,_candidate,_entry_price,_exit_price
)
from williams_ai.chart_reader import validate_closed_bars,indicators
from williams_ai.wave_reference import registry,recall_wave,structural_exam


def synthetic(n=330):
    t=datetime(2026,5,1,tzinfo=timezone.utc)
    return [{
        "close_time":(t+timedelta(hours=i)).isoformat(),
        "open":100.0+i*.03,"high":100.8+i*.03,
        "low":99.4+i*.03,"close":100.2+i*.03,
        "volume":10+i,"closed":True
    } for i in range(n)]


class WaveMemoryTests(unittest.TestCase):
    def test_reference_is_not_mislabeled_as_bill_williams(self):
        pack=registry()
        self.assertEqual(len(pack["cards"]),12)
        self.assertFalse(pack["reference"]["is_bill_williams_book"])
        self.assertEqual(pack["reference"]["total_pages"],6)
        self.assertEqual(pack["reference"]["sha256"],
                         "295bc6616170b19bf03b8d0179365582956e246358a7758adba4d4452eb70aeb")
        self.assertTrue(all(x["execution_authority"] is False for x in pack["cards"]))

    def test_fast_diagram_lookup_has_explicit_source_and_uncertainty(self):
        out=recall_wave("Отличие двойного зигзага от двойной тройки",limit=3)
        self.assertTrue(out)
        self.assertTrue(all(x["source"].startswith("VoznyWaveMemo: PDF ") for x in out))
        self.assertTrue(all(not x["authoritative_for_trade"] for x in out))
        self.assertEqual(recall_wave("Международные авиалинии 438493"),[])

    def test_classic_impulse_and_leading_diagonal_are_not_same(self):
        overlap=[100,120,110,140,118,150]
        impulse=structural_exam({"pattern":"IMPULSE","pivots":overlap})
        diagonal=structural_exam({"pattern":"LEADING_DIAGONAL","pivots":overlap})
        self.assertIn("W4_OVERLAPS_W1_CLASSIC_IMPULSE",impulse["violations"])
        self.assertNotIn("W4_OVERLAPS_W1_CLASSIC_IMPULSE",diagonal["violations"])
        self.assertFalse(diagonal["internal_subwaves_verified"])
        self.assertEqual(diagonal["geometry_status"],"UNVERIFIED_CANDIDATE")
        self.assertIsNone(diagonal["trade_action"])

    def test_w2_origin_and_w3_shortest_rejected(self):
        x=structural_exam({"pattern":"IMPULSE","pivots":[100,120,99,125,120,170]})
        self.assertIn("W2_BREACHES_ORIGIN",x["violations"])
        self.assertIn("W3_SHORTEST",x["violations"])


class FrozenMarketTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.all=frozen_markets()

    def test_fifteen_actual_bars_are_contiguous_and_closed(self):
        self.assertEqual(len(self.all),15)
        for sym,record in self.all.items():
            with self.subTest(sym=sym):
                self.assertEqual(len(record["bars"]),1200)
                _,gaps=validate_closed_bars(record["bars"],"H1",record["bars"][-1]["close_time"])
                self.assertEqual(gaps,0)

    def test_causal_append_cannot_rewrite_old_trades(self):
        for sym in ("ETHUSDT","SOLUSDT","XRPUSDT"):
            x=self.all[sym]
            # 100-bar evaluation starts after indicators already warmed up.
            partial=simulate_one(x["bars"],sym,x["tick_size"],start_index=830,final_index=980,max_events=10000)
            more=simulate_one(x["bars"],sym,x["tick_size"],start_index=830,final_index=1040,max_events=10000)
            fixed=partial["to"]
            earlier=[e for e in more["trace"] if e["time"]<=fixed]
            self.assertEqual(partial["trace"],earlier)
            self.assertTrue(all(e["event"] not in ("SIM_CLOSE_ON_OPPOSITE","TIME_EXIT") for e in earlier))

    def test_all_models_never_trade_without_stop_or_authorize_real_orders(self):
        r=run_15()
        self.assertEqual(r["pairs"],15)
        self.assertEqual(len(r["results"]),45)
        self.assertFalse(r["execution_allowed"])
        self.assertIsNone(r["trade_action"])
        self.assertFalse(r["model_weights_updated"])
        for x in r["results"]:
            self.assertEqual(x["closed"],x["stop_exits"])
            self.assertEqual(x["non_stop_exits"],0)
            self.assertTrue(x["read_only"])
            self.assertIsNone(x["trade_action"])
            self.assertTrue(x["fees_in_equity"]>=0)
            self.assertEqual(x["wins"]+x["losses"],x["closed"])

    def test_both_sides_and_stop_priority_are_conservative(self):
        class Bar: pass
        c=Bar();c.open=105;c.high=111;c.low=95
        self.assertEqual(_entry_price(c,110,"LONG"),110*(1+0.0002))
        self.assertEqual(_exit_price(c,100,"LONG"),100*(1-0.0002))
        self.assertEqual(_exit_price(c,100,"SHORT"),105*(1+0.0002))

    def test_future_must_not_enter_past_candidate(self):
        rows=synthetic()
        bars,_=validate_closed_bars(rows,"H1",rows[-1]["close_time"])
        ind=indicators(bars)
        part=_candidate(bars,ind,"WM3",280,0.01)
        more=_candidate(bars[:281],{k:v[:281] for k,v in ind.items()},"WM3",280,0.01)
        self.assertEqual(part,more)


if __name__=="__main__":
    unittest.main()
