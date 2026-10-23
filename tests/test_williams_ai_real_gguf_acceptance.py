"""Real-model acceptance contract: mock weights must NOT satisfy real-bytes gate.

A fake model in these tests is a protocol test only; only the separate
public GGUF GitHub Actions job may report real inference.
"""
import tempfile
from pathlib import Path
import unittest

from williams_ai.real_gguf_acceptance import actual_gguf_sha256,public_model_trial


class RealModelContractTests(unittest.TestCase):
    def test_schema_never_allows_non_observed_wm1_even_with_real_llm(self):
        from williams_ai.causal_paper_lab import frozen_markets
        from williams_ai.chart_reader import analyze_chart
        from williams_ai.local_model import LocalWilliamsModel
        market=frozen_markets()["BTCUSDT"]
        bars=market["bars"][:900]
        chart=analyze_chart("BTCUSDT","H1",bars,
                            as_of=bars[-1]["close_time"],
                            tick_size=market["tick_size"])
        self.assertFalse(any(x.get("type")=="WM1" for x in chart["candidates"][-8:]))
        class Fake:
            def __init__(self,**kwargs):self.kwargs=kwargs;self.calls=[]
            def create_chat_completion(self,**kwargs):
                self.calls.append(kwargs)
                return {"choices":[{"message":{"content":"{}"}}]}
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"fake.gguf"
            path.write_bytes(b"fake-content-only")
            model=LocalWilliamsModel(path,loader=Fake,use_decision_schema=True)
            model.propose_chart_reasoning(chart,edition="TC2")
            schema=model._model.calls[0]["response_format"]["schema"]
            allowed=schema["properties"]["setup"]["enum"]
            self.assertNotIn("WM1",allowed)
            self.assertIn("NO_SETUP",allowed)
            self.assertEqual(schema["properties"]["as_of"]["enum"],
                             [bars[-1]["close_time"]])
            self.assertEqual(schema["properties"]["edition"]["enum"],["TC2"])
            self.assertEqual(schema["properties"]["execution_allowed"]["enum"],[False])
            # A machine-confirmed fractal candidate may be DISCUSSED, not
            # executed; independent Teeth/time auditor remains in force.
            synthetic={**chart,"candidates":chart["candidates"]+[{
                "type":"WM3_FRACTAL","side":"LONG",
                "state":"AWAITING_PRICE_TRIGGER"}]}
            model.propose_chart_reasoning(synthetic,edition="TC2")
            next_schema=model._model.calls[1]["response_format"]["schema"]
            self.assertIn("WM3",next_schema["properties"]["setup"]["enum"])
            self.assertNotIn("WM1",next_schema["properties"]["setup"]["enum"])

    def test_fake_gguf_magic_is_not_sufficient(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"fake.gguf"
            path.write_bytes(b"GGUFfake")
            with self.assertRaisesRegex(ValueError,"unexpectedly tiny"):
                actual_gguf_sha256(path,"0"*64)

    def test_fake_weights_without_magic_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"fake.gguf"
            path.write_bytes(b"not a model")
            with self.assertRaisesRegex(ValueError,"Not genuine GGUF"):
                actual_gguf_sha256(path,"0"*64)

    def test_missing_model_blocked_not_substituted_with_mock(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                actual_gguf_sha256(Path(tmp)/"absent.gguf","0"*64)

    def test_valid_structure_cannot_infer_claim_from_no_verified_weights(self):
        class Fake:
            def answer_exam_case(self,case):
                return {"choice":"B","reason":"fake"}
        with self.assertRaisesRegex(ValueError,"not passed"):
            public_model_trial(Fake(),weight_report={},max_questions=1,
                               history_cases=0)
        with self.assertRaisesRegex(ValueError,"Need 1..10"):
            public_model_trial(Fake(),weight_report={"verified_model_magic":"GGUF"},
                               max_questions=0,history_cases=0)


if __name__=="__main__":
    unittest.main()
