"""Evidence-derived TC2 WM3 combinations and model prompt are not fills."""
import json
import tempfile
import unittest
from pathlib import Path

from williams_ai.decision_contract import (
    contract_for_model, legal_wm3_discussion_states,
)
from williams_ai.local_model import LocalWilliamsModel
from williams_ai.causal_paper_lab import frozen_markets
from williams_ai.chart_reader import analyze_chart


NOW="2026-10-10T07:59:59.999000+00:00"
BEFORE="2026-10-10T05:59:59.999000+00:00"


def chart(candidates):
    return {
        "symbol":"BTCUSDT","as_of":NOW,"execution_allowed":False,
        "gaps":0,"candidates":candidates,
    }


class WM3DiscussionContract(unittest.TestCase):
    def test_pending_long_has_explicit_entry_state_not_exchange_fill(self):
        case=chart([{
            "type":"WM3_FRACTAL","side":"LONG",
            "confirmed_at":BEFORE,"state":"AWAITING_PRICE_TRIGGER",
            "price_touch_observed_at":None,
        }])
        v=legal_wm3_discussion_states(case)
        self.assertEqual(len(v),1)
        self.assertEqual(
            (v[0]["setup"],v[0]["side"],v[0]["entry_state"]),
            ("WM3","LONG","PENDING_TRIGGER")
        )
        self.assertFalse(contract_for_model(case,"TC2")["execution_allowed"])

    def test_new_current_touch_and_old_touch_are_not_confused(self):
        p={
            "type":"WM3_FRACTAL","side":"SHORT",
            "confirmed_at":BEFORE,
            "state":"CLOSED_BAR_TOUCH_PASSES_TEETH_NOT_EXCHANGE_FILL",
        }
        current=legal_wm3_discussion_states(chart([
            {**p,"price_touch_observed_at":NOW}
        ]))
        stale=legal_wm3_discussion_states(chart([
            {**p,"price_touch_observed_at":BEFORE}
        ]))
        self.assertEqual(current[0]["entry_state"],"CLOSED_BAR_TOUCH_ONLY")
        self.assertEqual(stale,[])

    def test_unconfirmed_future_or_nonpending_fills_never_allowed(self):
        inputs=[
            {"type":"WM3_FRACTAL","side":"LONG","confirmed_at":"2026-10-11T00:00:00+00:00",
             "state":"AWAITING_PRICE_TRIGGER"},
            {"type":"WM3_FRACTAL","side":"LONG","confirmed_at":BEFORE,
             "state":"ALREADY_BROKEN","price_touch_observed_at":BEFORE},
            {"type":"WM3_FRACTAL","side":"UNKNOWN","confirmed_at":BEFORE,
             "state":"AWAITING_PRICE_TRIGGER"},
        ]
        self.assertEqual(legal_wm3_discussion_states(chart(inputs)),[])
        with self.assertRaises(ValueError):
            legal_wm3_discussion_states({**chart([]),"execution_allowed":True})

    def test_all_15_real_binance_frozen_h1_passports_use_current_cutoff_only(self):
        markets=frozen_markets()
        self.assertEqual(len(markets),15)
        for symbol,m in markets.items():
            with self.subTest(pair=symbol):
                candles=m["bars"][:900]
                obs=analyze_chart(symbol,"H1",candles,
                                  as_of=candles[-1]["close_time"],
                                  tick_size=m["tick_size"])
                r=contract_for_model(obs,"TC2")
                self.assertFalse(r["execution_allowed"])
                self.assertEqual(r["market_as_of"],obs["as_of"])
                for option in r["wm3_valid_setup_side_entry_state_combinations"]:
                    self.assertEqual(option["setup"],"WM3")
                    self.assertIn(option["entry_state"],
                                  ("PENDING_TRIGGER","CLOSED_BAR_TOUCH_ONLY"))

    def test_real_model_context_contains_state_side_without_hallucinating_entry(self):
        from williams_ai.chart_reader import analyze_chart
        market=frozen_markets()["BTCUSDT"]
        rows=market["bars"][:900]
        observation=analyze_chart("BTCUSDT","H1",rows,
                as_of=rows[-1]["close_time"],tick_size=market["tick_size"])
        class Fake:
            def __init__(self,**kwargs):self.calls=[]
            def create_chat_completion(self,**kwargs):
                self.calls.append(kwargs)
                payload=json.loads(kwargs["messages"][1]["content"])
                c=payload["chart"]
                response={
                    "symbol":c["symbol"],"as_of":c["as_of"],
                    "edition":payload["edition"],"setup":"NO_SETUP",
                    "side":"NONE","entry_state":"NO_SETUP",
                    "rule_ids":[],"trade_action":None,
                    "execution_allowed":False,
                }
                return {"choices":[{"message":{"content":json.dumps(response)}}]}
        with tempfile.TemporaryDirectory() as tmp:
            weights=Path(tmp)/"fake.gguf"
            weights.write_bytes(b"FAKE_UNITTTEST_ONLY")
            m=LocalWilliamsModel(weights,loader=Fake,use_decision_schema=True)
            m.propose_chart_reasoning(observation,edition="TC2")
            ctx=json.loads(m._model.calls[0]["messages"][1]["content"])
            self.assertEqual(ctx["chart"]["observed_book_setup_state_contract"],
                             contract_for_model(observation,"TC2"))
            self.assertTrue(ctx["chart"]["observed_book_setup_state_contract"]
                            ["no_trade_is_always_permitted"])
            self.assertFalse(ctx["chart"]["observed_book_setup_state_contract"]
                             ["execution_allowed"])


if __name__=="__main__":
    unittest.main()
