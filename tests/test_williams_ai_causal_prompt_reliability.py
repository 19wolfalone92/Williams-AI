"""All eight exact historical GGUF cuts fit a bounded, source-preserving JSON prompt.

This is a CONTRACT test with mock inference; it does NOT establish that the
actual Qwen model understands Williams or has an edge. Actual CPU GGUF smoke
has an independent run that can remain intentionally RED.
"""
import json
import tempfile
import unittest
from pathlib import Path

from williams_ai.causal_paper_lab import frozen_markets
from williams_ai.local_model import LocalWilliamsModel
from williams_ai.reasoning_trial import evaluate_historical_thought
from williams_ai.real_gguf_history_exam import CASES
from williams_ai.book_rules import registry


class SourceRespectingFakeGGUF:
    def __init__(self,**kw):
        self.calls=[]
    def create_chat_completion(self,**kwargs):
        self.calls.append(kwargs)
        prompt=kwargs["messages"][1]["content"]
        decoded=json.loads(prompt)
        assert decoded["chart"]["as_of"] is not None
        claim={
            "symbol":decoded["chart"]["symbol"],
            "as_of":decoded["chart"]["as_of"],
            "edition":decoded["edition"],
            "setup":"NO_SETUP","side":"NONE",
            "rule_ids":[],"reason":"Insufficient independent evidence for a price trigger.",
            "entry_state":"NO_SETUP","exit_basis":"NONE",
            "wave_degree_verified":False,
            "exchange_fill_confirmed":False,
            "exchange_sl_confirmed":False,
            "trade_action":None,"execution_allowed":False,
        }
        return {"choices":[{"message":{"content":json.dumps(claim)}}]}


class GGUFPromptHistoricalReliability(unittest.TestCase):
    def test_all_predeclared_historical_model_contexts_reliably_fit(self):
        dataset=frozen_markets()
        cards=registry()["rule_cards"]
        by_ed={}
        for c in cards:by_ed.setdefault(c["source"],set()).add(c["id"])
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"contract-only.gguf"
            path.write_bytes(b"Mock engine no real weights")
            model=LocalWilliamsModel(path,loader=SourceRespectingFakeGGUF,
                                     use_decision_schema=True)
            lengths=[]
            for symbol,index,edition in CASES:
                with self.subTest(symbol=symbol,edition=edition,index=index):
                    market=dataset[symbol]
                    result=evaluate_historical_thought(
                        symbol,market["bars"],tick_size=market["tick_size"],
                        bar_index=index,edition=edition,model=model,
                        include_topdown=True)
                    self.assertEqual(result["audit"]["hard_failures"],[])
                    self.assertEqual(result["audit"]["setup"],"NO_SETUP")
                    self.assertFalse(result["model_saw_future_bars"])
                    self.assertFalse(result["execution_allowed"])
                    payload=model._model.calls[-1]["messages"][1]["content"]
                    lengths.append(len(payload))
                    self.assertLessEqual(len(payload),16000)
                    context=json.loads(payload)
                    self.assertEqual(context["edition"],edition)
                    self.assertEqual(context["chart"]["as_of"],
                                     market["bars"][index]["close_time"])
                    self.assertEqual(
                        {x["id"] for x in context["source_cards"]},
                        by_ed[edition],
                    )
                    self.assertIn("tc1_five_magic_bullets",context["chart"])
                    self.assertIn("principle_first_nonlinear_evidence",
                                  context["chart"])
                    self.assertFalse(
                        context["chart"]["tc1_five_magic_bullets"]
                               ["zero_point_is_hypothesis_not_executable_entry"]
                        is False
                    )
            self.assertEqual(len(lengths),len(CASES))
            self.assertGreater(min(lengths),0)

    def test_context_does_not_expand_to_unbounded_history_with_more_bars(self):
        markets=frozen_markets()
        x=markets["XRPUSDT"]
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/"fake.gguf"
            p.write_bytes(b"fake")
            model=LocalWilliamsModel(p,loader=SourceRespectingFakeGGUF)
            for index in (899,1099,1199):
                with self.subTest(index=index):
                    result=evaluate_historical_thought(
                        "XRPUSDT",x["bars"],tick_size=x["tick_size"],
                        bar_index=index,edition="TC1",model=model)
                    self.assertEqual(result["audit"]["hard_failures"],[])
                    self.assertLessEqual(
                        len(model._model.calls[-1]["messages"][1]["content"]),16000
                    )


if __name__=="__main__":
    unittest.main()
