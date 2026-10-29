import ast
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from scripts.ai_run_historical_rule_kaggle import worker_source, numeric_cases
from scripts.ai_verify_restorative_result import independent_key
from williams_ai.contrastive_book_practice import question_rows
from williams_ai.historical_rule_curriculum import FAMILIES, digest
from williams_ai.historical_rule_gpu import selection, PROTOCOL


class HistoricalGpuProtocolTests(unittest.TestCase):
    def test_source_or_practice_regression_vetoes_a_perfect_historical_score(self):
        grade = {"accuracy": 1., "per_rule": {r: {"correct": 8, "total": 8} for r in FAMILIES}, "constant_fields_despite_varied_expected_values": {}}
        before = dict.fromkeys(("TC1", "NTD", "TC2", "COURSE", "ALL"), 1.)
        refs = {name: before for name in ("focused69_source", "previous40_source", "focused69_practice", "previous40_practice")}
        self.assertGreater(selection(grade, before, before, refs)["score"], 1)
        for source, practice in (({**before, "TC2": 1.021}, before), (before, {**before, "NTD": 1.021})):
            self.assertLess(selection(grade, source, practice, refs)["score"], -3)

    def test_generated_worker_is_private_training_only_with_real_dual_cuda_entry(self):
        files = {name: {"url": "https://example.invalid/own-file", "sha256": "a"*64} for name in ("adapter_config.json", "adapter_model.safetensors")}
        worker = worker_source("a"*40, b"PRIVATE_TRAIN_VALIDATION_ONLY", b"NO_TEST_ROWS", b'{"historical":[],"numeric":[]}', files, files, {})
        tree = ast.parse(worker)
        self.assertIn("--nproc_per_node=2", worker)
        self.assertIn("shutil.rmtree(repo/'examples')", worker)
        self.assertIn("shutil.rmtree(repo/'experiments')", worker)
        self.assertNotIn("KAGGLE_API_TOKEN", worker)
        with tempfile.TemporaryDirectory() as td:
            entry = Path(td)/"entry.py"
            node = next(n for n in tree.body if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                and isinstance(n.value.func, ast.Attribute) and isinstance(n.value.func.value, ast.Name)
                and n.value.func.value.id == "entry" and n.value.func.attr == "write_text")
            exec(compile(ast.Module(body=[node], type_ignores=[]), "entry_writer", "exec"),
                {"entry": entry, "repo": Path(td), "private": Path(td)/"private", "out": Path(td)/"output"})
            ast.parse(entry.read_text())
            self.assertIn("historical_rule_gpu import train", entry.read_text())

    def test_numeric_prompts_are_committed_before_training_and_keys_are_independent(self):
        cases = numeric_cases()
        manifest = json.loads(Path("experiments/historical-rule-curriculum-20261029-design.json").read_bytes())
        self.assertEqual(digest(question_rows(cases)), manifest["numeric_test_questions_sha256"])
        self.assertEqual(len(cases), 112)
        for row, question in zip(cases, question_rows(cases)):
            self.assertEqual(independent_key(question), json.loads(row["answer"]))
        self.assertEqual(PROTOCOL["temperature"], .7)
        self.assertTrue(PROTOCOL["do_sample"])


if __name__ == "__main__": unittest.main()
