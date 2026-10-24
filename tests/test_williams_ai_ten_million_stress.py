"""Ten-million grid counts are NOT ten million independent market trades."""
import unittest
from datetime import datetime,timedelta,timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from williams_ai.continual_trader import Outcome, PrivateExperience
from williams_ai.ten_million_stress import (
    GRID_SIZE,cost_stress_from_settled_experience
)

START=datetime(2025,1,1,tzinfo=timezone.utc)

class CostGridHonesty(unittest.TestCase):
    def insert(self, db, *, assets=5, n=75):
        cases=[]
        for sym in range(assets):
            symbol=f"COIN{sym}USDT"
            for i in range(n):
                begin=START+timedelta(hours=i)
                resolved=begin+timedelta(hours=2)
                cases.append(Outcome(
                    symbol,"H1","LONG",begin.isoformat(),resolved.isoformat(),
                    (1.,float(i%5),0.2,0.2,0.5,1.),.03 if i%3 else -.01,
                    i%3==0,"a"*64,"WM3","TC2.WM3"))
        db.append(cases,knowledge_as_of=(START+timedelta(days=10)).isoformat())

    def test_grid_has_real_unique_cases_and_cannot_claim_10m_trades(self):
        with TemporaryDirectory() as d:
            db=PrivateExperience(Path(d)/"memory.db")
            self.insert(db)
            report=cost_stress_from_settled_experience(
                db,known_at=(START+timedelta(days=10)).isoformat(),
                target=10000,min_decision_times=300)
            self.assertEqual(report["verified_execution_cost_scenario_checks"],10000)
            self.assertEqual(report["historical_eligible_wm_paper_outcomes"],375)
            self.assertEqual(report["distinct_historical_asset_timeframe_decisions"],375)
            self.assertEqual(report["cost_grid_unique_combinations_per_paper_trade"],4400)
            self.assertTrue(report["goal_met"])
            self.assertEqual(report["actual_gguf_inference_calls"],0)
            self.assertFalse(report["ten_million_separate_market_decisions"])
            self.assertFalse(report["source_signals_canonical_four_books_confirmed"])
            self.assertFalse(report["execution_allowed"])
            self.assertLessEqual(report["friction_scenario_positive_fraction"],1.0)

    def test_rejects_unobserved_10m_from_tiny_dataset_or_future(self):
        with TemporaryDirectory() as d:
            db=PrivateExperience(Path(d)/"private.sqlite3")
            self.insert(db,assets=1,n=3)
            with self.assertRaisesRegex(ValueError,"INSUFFICIENT_REAL"):
                cost_stress_from_settled_experience(
                    db,known_at=(START+timedelta(days=10)).isoformat(),
                    target=10_000_000)
            with self.assertRaisesRegex(ValueError,"INSUFFICIENT_REAL"):
                cost_stress_from_settled_experience(
                    db,known_at=(START-timedelta(days=1)).isoformat(),
                    target=1,min_symbols=1,min_decision_times=1)

if __name__=="__main__":
    unittest.main()
