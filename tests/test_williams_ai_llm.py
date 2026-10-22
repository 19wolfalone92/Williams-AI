import io
import json
import unittest

from williams_ai.analyst import build_report
from williams_ai.llm import explain_with_openai


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


class WilliamsAILLMTest(unittest.TestCase):
    def test_only_sanitized_report_sent_without_raw_keys(self):
        snapshot = {
            "symbol": "ETHUSDT", "as_of": "2026-10-22T18:00:00Z",
            "timeframes": {}, "api_key": "SENSITIVE_CANARY_KEY"
        }
        report = build_report(snapshot)
        sent = {}

        def fake_opener(req, timeout):
            sent["body"] = req.data.decode("utf-8")
            sent["url"] = req.full_url
            sent["auth"] = req.get_header("Authorization")
            return FakeResponse(json.dumps({"output": [{
                "type": "message", "content": [{
                    "type": "output_text", "text": "Недостаточно данных."
                }]
            }]}).encode("utf-8"))

        answer = explain_with_openai(
            "Что видит бот?", report, api_key="TEST_KEY",
            model="MODEL_FROM_ENV", opener=fake_opener
        )
        self.assertEqual(answer, "Недостаточно данных.")
        self.assertNotIn("SENSITIVE_CANARY_KEY", sent["body"])
        self.assertNotIn("TEST_KEY", sent["body"])
        self.assertEqual(sent["auth"], "Bearer TEST_KEY")
        self.assertIn('"store": false', sent["body"])
        self.assertTrue(sent["url"].startswith("https://api.openai.com/"))

    def test_reject_raw_snapshot(self):
        with self.assertRaises(ValueError):
            explain_with_openai("вопрос", {"symbol": "ETHUSDT"},
                                api_key="TEST_KEY", model="M",
                                opener=lambda *_a, **_kw: None)


if __name__ == "__main__":
    unittest.main()
