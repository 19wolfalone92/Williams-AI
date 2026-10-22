import json
import tempfile
import unittest
from datetime import datetime,timezone,timedelta
from pathlib import Path

from williams_ai.book_rules import registry,lookup_rules,coverage
from williams_ai.book_index import index_pages,search_pages,index_report,_private_database,SourceMismatch
from williams_ai.chart_reader import analyze_chart,analyze_timeframes,indicators,validate_closed_bars
from williams_ai.chart_cli import _to_snapshot
from williams_ai.analyst import build_report


def candles(n=60, tf_minutes=60):
    start=datetime(2026,10,1,tzinfo=timezone.utc)
    rows=[]
    for i in range(n):
        price=200-i*0.17
        rows.append({
            "close_time":(start+timedelta(minutes=i*tf_minutes)).isoformat(),
            "open":price+0.15,"high":price+0.55,"low":price-0.55,
            "close":price,"volume":100+i,"closed":True,
        })
    return rows


class BookProvenanceTest(unittest.TestCase):
    def test_four_sources_and_all_cards_scoped(self):
        pack=registry()
        self.assertEqual({x["id"] for x in pack["books"]},{"TC1","NTD","TC2","COURSE"})
        self.assertGreaterEqual(len(pack["rule_cards"]),24)
        self.assertTrue(all(x["pdf_pages"] for x in pack["rule_cards"]))
        self.assertIsNone(coverage()["TC2"]["accuracy_percent"])

    def test_wm1_edition_source_does_not_override_ntd(self):
        cards=lookup_rules("WM1 ангуляция",edition="TC2")
        self.assertTrue(cards)
        self.assertTrue(all(x["source"].startswith("TC2:") for x in cards))
        self.assertFalse(any("NTD:" in x["source"] for x in cards))

    def test_local_book_db_never_put_in_repository(self):
        with self.assertRaises(ValueError):
            _private_database(Path(__file__).resolve().parents[1] / "books.db")

    def test_index_and_search_keep_page_refs(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"private.db"
            source=next(x for x in registry()["books"] if x["id"]=="TC2")
            pages=[""] * source["pages"]
            pages[121]="Ангуляция должна проверять открывающийся угол цены относительно Аллигатора"
            report=index_pages(path,"TC2",source["sha256"],pages,declared_pages=source["pages"])
            self.assertEqual(report["nonempty_pages"],1)
            self.assertEqual(report["missing_text_pages"],source["pages"]-1)
            hits=search_pages(path,"ангуляция",edition="TC2")
            self.assertEqual(hits[0]["source"],"TC2: PDF 122")
            self.assertEqual(index_report(path)["volumes"][0]["pages_without_text"],source["pages"]-1)
            with self.assertRaises(SourceMismatch):
                index_pages(path,"TC2","bad-sha",pages,declared_pages=source["pages"])
            self.assertEqual(len(search_pages(path,"ангуляция")),1)

    def test_rulepack_not_verbatim_reproduction(self):
        pack=registry()
        # The public registry stores only short rule summaries (not book text).
        self.assertTrue(all(len(x["summary"])<600 for x in pack["rule_cards"]))


class ChartReaderTests(unittest.TestCase):
    def test_reject_future_unfinished_duplicate_and_bad_ohlc(self):
        sample=candles()
        at=sample[-2]["close_time"]
        with self.assertRaisesRegex(ValueError,"future"):
            validate_closed_bars(sample,"H1",at)
        sample[-1]["closed"]=False
        with self.assertRaisesRegex(ValueError,"finalized"):
            validate_closed_bars(sample,"H1",sample[-1]["close_time"])
        sample=candles()
        sample[-1]["close_time"]=sample[-2]["close_time"]
        with self.assertRaisesRegex(ValueError,"Duplicate"):
            validate_closed_bars(sample,"H1",candles()[-1]["close_time"])
        sample=candles()
        sample[-1]["high"]=sample[-1]["low"]-0.01
        with self.assertRaisesRegex(ValueError,"OHLC"):
            validate_closed_bars(sample,"H1",candles()[-1]["close_time"])

    def test_displaced_lines_reference_only_past_values(self):
        rows=candles(55)
        bars,_=validate_closed_bars(rows,"H1",rows[-1]["close_time"])
        a=indicators(bars)
        # Jaw is shifted ahead 8 bars: at index 20 it equals first
        # SMA(13)-seeded jaw from index 12, never a future computed value.
        med=[(x.high+x.low)/2 for x in bars]
        self.assertAlmostEqual(a["jaw"][20],sum(med[:13])/13)
        self.assertIsNone(a["jaw"][19])
        self.assertIsNone(a["ao"][32])
        self.assertIsNotNone(a["ao"][33])
        self.assertIsNone(a["ac"][36])
        self.assertIsNotNone(a["ac"][37])

    def test_no_fractal_before_two_right_closed_bars(self):
        rows=candles(60)
        # Place peak pivot at final candle minus 1: no complete right side.
        rows[-2]["high"]=rows[-2]["high"]+7
        view=analyze_chart("BTCUSDT","H1",rows,as_of=rows[-1]["close_time"])
        self.assertFalse(any(c.get("pivot_time")==rows[-2]["close_time"]
                             for c in view["candidates"]))
        self.assertFalse(view["execution_allowed"])

    def test_confirm_fractal_only_on_t_plus_two(self):
        rows=candles(60)
        rows[52]["high"]=rows[52]["high"]+10
        partial=analyze_chart("BTCUSDT","H1",rows[:54],as_of=rows[53]["close_time"])
        self.assertFalse(any(c.get("pivot_time")==rows[52]["close_time"] for c in partial["candidates"]))
        completed=analyze_chart("BTCUSDT","H1",rows[:55],as_of=rows[54]["close_time"])
        self.assertTrue(any(c.get("pivot_time")==rows[52]["close_time"]
                            and c.get("confirmed_at")==rows[54]["close_time"]
                            for c in completed["candidates"]))

    def test_signals_disallowed_on_holes(self):
        rows=candles(61)
        rows.pop(20)
        view=analyze_chart("BTCUSDT","H1",rows,as_of=rows[-1]["close_time"])
        self.assertEqual(view["gaps"],1)
        self.assertEqual(view["candidates"],[])
        self.assertIn("Дыры",view["warnings"][0])

    def test_zone_from_ao_ac_not_body_color(self):
        rows=candles()
        view=analyze_chart("BTCUSDT","H1",rows,as_of=rows[-1]["close_time"])
        self.assertIn(view["evidence"]["zone"],("GREEN","RED","GREY"))
        self.assertEqual(view["trade_action"] if "trade_action" in view else view["execution_allowed"],False)

    def test_multitimeframe_report_consistent(self):
        one=candles(60,60)
        four=candles(55,240)
        asof=four[-1]["close_time"]
        chart={"symbol":"ETHUSDT","as_of":asof,
               "timeframes":{"H1":{"bars":one},"H4":{"bars":four}}}
        output=analyze_timeframes(chart)
        self.assertEqual(output["trade_action"],None)
        report=build_report(_to_snapshot(output))
        self.assertEqual(report["mode"],"READ_ONLY")
        self.assertEqual(set(report["evidence"]["timeframes"]),{"H1","H4"})

    def test_wave_is_not_fabricated_as_confirmed(self):
        rows=candles(70)
        view=analyze_chart("BTCUSDT","H1",rows,as_of=rows[-1]["close_time"])
        self.assertTrue(all(x["status"]=="UNCONFIRMED_WAVE_DEGREE" for x in view["hypotheses"]))


if __name__=="__main__":
    unittest.main()
