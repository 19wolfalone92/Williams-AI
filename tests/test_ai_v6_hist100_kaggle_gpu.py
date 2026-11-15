"""Safety- and isolation-focused tests for mixed CPU/GPU historical100."""
import unittest
from scripts.ai_v6_hist100_kaggle_gpu_submit import make_metadata, KERNEL_NAME, TRAINING_NAME
from scripts.ai_v6_hist100_kaggle_gpu_worker import MAX_CONTEXT_TOKENS, MAX_NEW_TOKENS


class KaggleGPUContract(unittest.TestCase):
    def test_private_gpu_and_genuine_source_only(self):
        m, worker = make_metadata("my-kaggle-account", "e" * 40)
        self.assertEqual(m["is_private"], "true")
        self.assertEqual(m["enable_gpu"], "true")
        self.assertEqual(m["machine_shape"], "NvidiaTeslaT4")
        self.assertEqual(m["kernel_sources"], ["my-kaggle-account/"+TRAINING_NAME])
        self.assertFalse(m["dataset_sources"])
        self.assertFalse(m["model_sources"])
        self.assertIn("scripts.ai_v6_hist100_kaggle_gpu_worker", worker)
        self.assertNotIn("KAGGLE_API_TOKEN", worker)
        self.assertEqual(len(m["kernel_sources"]), 1)
        self.assertEqual(MAX_CONTEXT_TOKENS, 8192)
        self.assertEqual(MAX_NEW_TOKENS, 256)


    def test_gpu_compacted_prompt_bounded_and_causal(self):
        import json
        from scripts.ai_v6_hist100_kaggle_gpu_worker import (
            compact_gpu_prompt, GPU_MAX_INPUT_TOKENS,
            GPU_VISIBLE_CLOSED_H1_MAX, GPU_VISIBLE_CLOSED_H1_MIN,
        )
        class DummyTokenizer:
            def apply_chat_template(self, messages, **kwargs):
                bars=json.loads(messages[1]["content"])["closed_bars"]
                return list(range(1600 + 75*len(bars)))
        candles=[["2025-01-01",i,1,1,1,1] for i in range(96)]
        messages=[
            {"role":"system","content":"book-only, no future"},
            {"role":"user","content":json.dumps({
                "closed_bars": candles,
                "causal_H1_H4_D1_indicators_not_book_signals":{"ao":12},
                "book_source_cards_not_expert_gold":[{"rule_id":"R1"}],
            })},
        ]
        compact,digest,ntokens,n=compact_gpu_prompt(DummyTokenizer(),messages)
        self.assertLessEqual(ntokens,GPU_MAX_INPUT_TOKENS)
        self.assertTrue(GPU_VISIBLE_CLOSED_H1_MIN<=n<=GPU_VISIBLE_CLOSED_H1_MAX)
        self.assertEqual(len(digest),64)
        self.assertEqual(json.loads(compact[1]["content"])["closed_bars"],
                         candles[-n:])
        self.assertEqual(len(json.loads(messages[1]["content"])["closed_bars"]),96)
        self.assertEqual(json.loads(compact[1]["content"])["causal_H1_H4_D1_indicators_not_book_signals"],{"ao":12})


    def test_fallback_only_for_extreme_final_case(self):
        import json
        from scripts.ai_v6_hist100_kaggle_gpu_worker import compact_gpu_prompt
        class Dummy:
            def apply_chat_template(self, messages, **kwargs):
                n=len(json.loads(messages[1]["content"])["closed_bars"])
                return list(range(3000+35*n))
        m=[{"role":"system","content":"causal"},
           {"role":"user","content":json.dumps({"closed_bars":[[i] for i in range(96)],
                "book_source_cards_not_expert_gold":[{"rule_id":"fixed"}]})}]
        with self.assertRaisesRegex(ValueError,"GPU_PROMPT_TOO_LARGE"):
            compact_gpu_prompt(Dummy(),m)
        compact,hashval,tokens,n=compact_gpu_prompt(Dummy(),m,max_input_tokens=4096,min_closed_h1_bars=16)
        self.assertLessEqual(tokens,4096)
        self.assertGreaterEqual(n,16)
        self.assertEqual(json.loads(compact[1]["content"])["closed_bars"],[[i] for i in range(96-n,96)])
        self.assertEqual(len(hashval),64)

    def test_invalid_input_rejected(self):
        with self.assertRaisesRegex(ValueError, "INVALID_KAGGLE_OWNER"):
            make_metadata("../escape", "e" * 40)
        with self.assertRaisesRegex(ValueError, "INVALID_SOURCE_REVISION"):
            make_metadata("goodname", "branch-name")


if __name__ == "__main__":
    unittest.main()
