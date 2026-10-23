"""Ensure ONLY causal facts reach an actual local GGUF inference adapter."""
import json
import tempfile
import unittest
from pathlib import Path

from williams_ai.local_model import LocalWilliamsModel
from williams_ai.chart_reader import analyze_chart
from williams_ai.prospective_qualification import freeze_prospective_call
from williams_ai.causal_paper_lab import frozen_markets


class LlamaMock:
    def __init__(self,**kwargs):
        self.calls=[]
        self.broken=None

    def create_chat_completion(self,**kwargs):
        self.calls.append(kwargs)
        payload=json.loads(kwargs["messages"][1]["content"])
        output={
            "symbol":payload["symbol"],"as_of":payload["as_of"],
            "direction":"ABSTAIN", "trade_action":None,
            "execution_allowed":False,
        }
        if self.broken is not None:
            output.update(self.broken)
        return {"choices":[{"message":{"content":json.dumps(output)}}]}


class ProspectiveModelBridgeTests(unittest.TestCase):
    def setUp(self):
        self.data=frozen_markets()["ETHUSDT"]
        self.rows=self.data["bars"][:900]
        self.evidence=analyze_chart(
            "ETHUSDT","H1",self.rows,
            as_of=self.rows[-1]["close_time"],
            tick_size=self.data["tick_size"],
        )

    def with_model(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        weights=Path(self.tmp.name)/"mock-not-real.gguf"
        weights.write_bytes(b"FAKE_UNITTTEST_GGUF_NOT_REAL_AI")
        return LocalWilliamsModel(weights,loader=LlamaMock)

    def test_model_receives_only_bounded_past_facts_and_produces_seal(self):
        model=self.with_model()
        sealed=freeze_prospective_call(
            "ETHUSDT",self.rows,tick_size=self.data["tick_size"],
            model_id="mock-model-identity",code_id="pinned-code-identity",
            predictor=model.propose_24h_research_forecast,
        )
        self.assertEqual(sealed["direction"],"ABSTAIN")
        self.assertFalse(sealed["execution_allowed"])
        call=model._model.calls[0]
        payload=json.loads(call["messages"][1]["content"])
        self.assertNotIn("future_price",payload)
        self.assertNotIn("raw_bars",payload)
        self.assertNotIn("api_key",payload)
        self.assertNotIn("private_book_text",payload)
        self.assertEqual(payload["as_of"],self.rows[-1]["close_time"])
        self.assertEqual(payload["study_kind"],
                         "24H_PROSPECTIVE_DIRECTIONAL_HYPOTHESIS_NOT_BOOK_ENTRY")
        self.assertEqual(call["temperature"],0)
        self.assertEqual(len(model._model.calls),1)
        self.assertFalse(payload["no_trading_execution"] is False)

    def test_model_cannot_output_order_or_claim_execution(self):
        model=self.with_model()
        model._model.broken={"trade_action":{"BUY":"ETHUSDT"}}
        with self.assertRaisesRegex(ValueError,"STRICT_READ_ONLY_SCHEMA"):
            model.propose_24h_research_forecast(self.evidence)
        model._model.broken={"execution_allowed":True}
        with self.assertRaisesRegex(ValueError,"STRICT_READ_ONLY_SCHEMA"):
            model.propose_24h_research_forecast(self.evidence)
        model._model.broken={"direction":"BUY"}
        with self.assertRaisesRegex(ValueError,"STRICT_READ_ONLY_SCHEMA"):
            model.propose_24h_research_forecast(self.evidence)
        model._model.broken={"future_price":200}
        with self.assertRaisesRegex(ValueError,"STRICT_READ_ONLY_SCHEMA"):
            model.propose_24h_research_forecast(self.evidence)


if __name__=="__main__":
    unittest.main()
