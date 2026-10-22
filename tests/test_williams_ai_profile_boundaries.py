import unittest
from pathlib import Path
from datetime import datetime, timezone, timedelta
from williams_ai.chart_reader import analyze_chart

ROOT=Path(__file__).resolve().parents[1]


def make_bars(n=65):
    t=datetime(2026,10,1,tzinfo=timezone.utc)
    out=[]
    for i in range(n):
        price=100+i*.03
        out.append({
            "close_time":(t+timedelta(hours=i)).isoformat(),
            "open":price,"high":price+1,"low":price-1,
            "close":price+.1,"volume":100,
            "closed":True,
        })
    return out


class ProfileSeparationTests(unittest.TestCase):
    def test_profiles_have_separate_names_and_no_orders(self):
        b=make_bars()
        report=analyze_chart("ETHUSDT","H1",b,as_of=b[-1]["close_time"])
        profiles=report["edition_evidence"]
        self.assertEqual(profiles["NTD"]["profile"],"NTD_FIVE_DIMENSIONS")
        self.assertEqual(profiles["TC2"]["profile"],"TC2_THREE_WISE_MEN")
        self.assertFalse(profiles["NTD"]["entry_admitted"])
        self.assertIsNone(profiles["TC2"]["trade_action"])
        self.assertFalse(report["execution_allowed"])
        self.assertIsNone(report["market_position"]["trade_action"])
        self.assertTrue(all(s["trade_action"] is None for s in profiles["NTD"]["signals"]))

    def test_readonly_modules_do_not_import_exchange_gateway(self):
        for name in ("chart_reader.py", "market_position.py", "replay.py",
                     "wave_evidence.py", "ntd_evidence.py"):
            text=(ROOT/"williams_ai"/name).read_text(encoding="utf-8")
            self.assertNotIn("from .demo_futures import",text)
            self.assertNotIn("api.binance.com",text)
            self.assertNotIn("demo-fapi.binance.com",text)

    def test_missing_knowledge_not_reported_as_full_book_accuracy(self):
        text=(ROOT/"docs/AI_MARKET_VISION_ACCEPTANCE_RU.md").read_text(encoding="utf-8")
        self.assertIn("НЕ доказано",text)
        self.assertIn("Trading Chaos Course",text)


if __name__=="__main__": unittest.main()
