"""End-to-end causal three-timeframe source checks for cognitive v4."""
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from tests.test_williams_ai_cognitive_v4 import FakeQwen
from williams_ai.cognitive_market_bridge_v4 import market_facts_from_closed_history
from williams_ai.unified_ai import WilliamsAI


def history_fixture():
    cutoff=datetime(2026,11,11,12,0,0,tzinfo=timezone.utc)
    history={}
    for tf,step,hours_back in (("H1",timedelta(hours=1),0),
                               ("H4",timedelta(hours=4),0),
                               ("D1",timedelta(days=1),12)):
        end=cutoff-timedelta(hours=hours_back)
        bars=[]
        for i in range(220):
            t=end-step*(219-i)
            base=100+i/3
            bars.append({"close_time":t.isoformat(),"open":base,
                         "high":base+1.3,"low":base-0.6,
                         "close":base+0.2,"volume":90+i/10,
                         "closed":True})
        history[tf]=bars
    return cutoff.isoformat(),history


class CognitiveMarketBridgeV4Tests(unittest.TestCase):
    def test_algorithmic_facts_all_three_closed_timeframes(self):
        as_of,hist=history_fixture()
        package=market_facts_from_closed_history(hist,as_of=as_of)
        self.assertEqual(package["schema"],"WILLIAMS_CAUSAL_MARKET_FACTS_V4")
        self.assertEqual(set(package["frames"]),{"H1","H4","D1"})
        ids={x["fact_id"] for x in package["facts"]}
        self.assertIn("H1_AO",ids)
        self.assertIn("H4_AC",ids)
        self.assertIn("D1_ALLIGATOR",ids)
        self.assertTrue(all(x["closed_bar_verified"] is True for x in package["facts"]))
        self.assertFalse(package["input_exchange_authenticity_independently_verified"])
        self.assertFalse(package["wave_degree_expert_verified"])

    def test_future_stale_open_gap_and_missing_higher_frame_blocked(self):
        as_of,hist=history_fixture()
        from copy import deepcopy
        for mutation,code in (
            (lambda h:h["H1"][-1].update({"closed":False}),"finalized"),
            (lambda h:h["H4"][55].update({"close_time":(
                datetime.fromisoformat(h["H4"][55]["close_time"])+timedelta(hours=1)
                ).isoformat()}),"CONTIGUOUS"),
            (lambda h:[x.update({"close_time":(
                datetime.fromisoformat(x["close_time"])+timedelta(hours=4)).isoformat()})
                for x in h["H4"]],"NO_LOOKAHEAD"),
        ):
            damaged=deepcopy(hist)
            mutation(damaged)
            with self.subTest(code=code),self.assertRaisesRegex(ValueError,code):
                market_facts_from_closed_history(damaged,as_of=as_of)
        with self.assertRaisesRegex(ValueError,"EXACT_CLOSED"):
            market_facts_from_closed_history({"H1":hist["H1"]},as_of=as_of)

    def test_actual_unified_api_uses_same_private_model_identity_and_safe_veto(self):
        from tests.test_williams_ai_unified_runtime import fake_bundle
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            base,adapter,manifest=fake_bundle(root)
            brain=WilliamsAI(local_model=base,memory=root/"cognitive.sqlite3",
                             public_rules_only=True)
            as_of,hist=history_fixture()
            thought=brain.cognitive_from_closed_bars(
                "фрактал",edition="TC2",as_of=as_of,
                historical_bars=hist,run_id="cognitive-lab-0001",
                research_backend=FakeQwen(conflict=True),use_memory=False)
            self.assertFalse(thought["execution_allowed"])
            self.assertEqual(thought["effective_stance"],"WAIT")
            self.assertTrue(thought["integration"]["facts_computed_from_closed_bars"])
            self.assertEqual(len(thought["source_market_snapshot"]["frames"]),3)
            self.assertFalse(brain.status()["runtime_loaded"])
            self.assertIn("cognitive_memory",brain.status())


if __name__=="__main__":
    unittest.main()
