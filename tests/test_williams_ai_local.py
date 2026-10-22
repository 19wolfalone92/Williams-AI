import json
import tempfile
import unittest
from pathlib import Path

from williams_ai.analyst import build_report
from williams_ai.knowledge import retrieve_notes
from williams_ai.local_model import LocalWilliamsModel


class FakeLlama:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls = []

    def create_chat_completion(self, **kwargs):
        self.calls.append(kwargs)
        return {"choices": [{"message": {"content": "Волна пока не подтверждена."}}]}


class OfflineModelTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.model = self.folder / "local.gguf"
        self.model.write_bytes(b"fake-weights-for-contract-tests-only")
        self.snapshot = {
            "symbol": "BTCUSDT",
            "as_of": "2026-10-22T18:00:00Z",
            "timeframes": {"H1": {
                "last_closed_at": "2026-10-22T18:00:00Z",
                "ao": 1.4, "wave_hypothesis": "четвёртая волна"
            }},
            "api_key": "SECRET_AT_INPUT_NOT_TO_BE_FORWARDED",
        }

    def test_creates_local_model_without_external_inference(self):
        model = LocalWilliamsModel(self.model, loader=FakeLlama)
        result = model.answer("Где сейчас рынок?", build_report(self.snapshot))
        self.assertIn("Волна", result)
        self.assertEqual(model._model.kwargs["n_gpu_layers"], 0)
        call = model._model.calls[0]
        self.assertEqual(len(call["messages"]), 2)
        self.assertEqual(call["messages"][0]["role"], "system")
        self.assertNotIn("SECRET_AT_INPUT_NOT_TO_BE_FORWARDED", json.dumps(call))
        self.assertNotIn("tools", call)
        self.assertEqual(call["max_tokens"], 700)

    def test_must_be_existing_local_gguf(self):
        with self.assertRaises(FileNotFoundError):
            LocalWilliamsModel(self.folder / "missing.gguf", loader=FakeLlama)
        with self.assertRaises(FileNotFoundError):
            LocalWilliamsModel(self.folder / "wrong.txt", loader=FakeLlama)

    def test_rejects_raw_unsanitized_snapshot(self):
        model = LocalWilliamsModel(self.model, loader=FakeLlama)
        with self.assertRaisesRegex(ValueError, "READ_ONLY"):
            model.answer("Почему?", self.snapshot)

    def test_warning_is_never_hidden_by_ai(self):
        self.snapshot["position"] = {
            "status": "OPEN", "side": "LONG", "stop_confirmed_on_exchange": False
        }
        model = LocalWilliamsModel(self.model, loader=FakeLlama)
        answer = model.answer("Что со стопом?", build_report(self.snapshot))
        self.assertTrue(answer.startswith("КРИТИЧНО:"))

    def test_local_search_and_no_symlink(self):
        notes = self.folder / "knowledge"
        notes.mkdir()
        (notes / "rules.md").write_text("Проверка четвёртой волны по AO\nФрактал требует подтверждения", encoding="utf-8")
        (notes / "external.md").symlink_to(self.folder / "local.gguf")
        results = retrieve_notes("четвёртой волны AO", notes)
        self.assertTrue(results)
        self.assertEqual(results[0]["source"], "rules.md:1")
        self.assertFalse(any("external.md" in item["source"] for item in results))

    def test_question_constraints(self):
        model = LocalWilliamsModel(self.model, loader=FakeLlama)
        with self.assertRaises(ValueError):
            model.answer("", build_report(self.snapshot))
        with self.assertRaises(ValueError):
            model.answer("x" * 1201, build_report(self.snapshot))


if __name__ == "__main__":
    unittest.main()
