"""Synthetic fixtures only: v6 expert exam guards (no real expert labels or trades)."""
import copy
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from williams_ai.independent_exam_v6 import (
    BOOKS, PDF_SHA, SCHEMA, canonical, validate_cases, evaluate,
)

HAS_CRYPTO = importlib.util.find_spec("cryptography") is not None


class V6IndependentIntakeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.cases = []
        for i in range(32):
            cutoff = datetime(2026, 8, 1, tzinfo=timezone.utc) + timedelta(days=i // 4)
            bars = []
            for j in range(64):
                start = cutoff + timedelta(hours=j - 64)
                end = start + timedelta(hours=1)
                bars.append({"open_utc": start.isoformat().replace("+00:00", "Z"),
                             "close_utc": end.isoformat().replace("+00:00", "Z"),
                             "open": 100.0 + i, "high": 102.0 + i,
                             "low": 99.0 + i, "close": 100.5 + i,
                             "volume": 0.3 + j})
            chart = {"symbol": f"CASE{i:03d}USDT", "bars": bars}
            name = f"chart_{i:02d}.json"
            raw = canonical(chart)
            (self.root / name).write_bytes(raw)
            ed = BOOKS[i // 8]
            family = ("ENTRY", "EXIT", "WAVE")[i % 3]
            self.cases.append({
                "id": f"E6-stress-{i:04d}", "edition": ed, "family": family,
                "chart_file": name, "chart_sha256": sha256(raw).hexdigest(),
                "cutoff_utc": cutoff.isoformat().replace("+00:00", "Z"),
                "source_pdf_sha256": PDF_SHA[ed], "pdf_pages": [2],
                "provenance_id": f"external-test-fixture-{i:04d}",
                "question": "Determine book-specific decision from closed bars; do not invent confirmation or an order."})
        self.dataset = {
            "schema": SCHEMA, "dataset_id": "test-fixture-not-experts",
            "cases": self.cases,
            "provenance_note": "FAKE synthetic data only, never independent expert market gold."}

    def labels(self):
        options = {"ENTRY": ("LONG_STOP_ENTRY", "SHORT_STOP_ENTRY", "NO_SETUP"),
                   "EXIT": ("EXIT_LONG", "HOLD"),
                   "WAVE": ("WAVE_HYPOTHESIS", "WAVE_CONFIRMED")}
        return [{
            "id": c["id"], "decision": options[c["family"]][(i // 3) % len(options[c["family"]])],
            "pages": c["pdf_pages"], "reviewable": True,
            "rationale": "Synthetic test-only explanation for the frozen source-page contract; not expert proof."
        } for i, c in enumerate(self.cases)]

    def signed(self, rid, key, manifest):
        payload = {
            "schema": "WILLIAMS_V6_EXPERT_REVIEW_V1",
            "dataset_sha256": manifest["dataset_sha256"],
            "reviewer_id": rid, "labels": self.labels()}
        return {"payload": payload, "signature_hex": key.sign(canonical(payload)).hex()}

    def test_frozen_chart_signatures_all_four_book_counts(self):
        manifest = validate_cases(self.dataset, self.root)
        self.assertEqual(manifest["case_count"], 32)
        self.assertEqual(manifest["by_book"], dict.fromkeys(BOOKS, 8))
        self.assertEqual(manifest["distinct_utc_dates"], 8)
        self.assertFalse(manifest["training_allowed"])
        self.assertFalse(manifest["expert_labels_attached"])

    def test_gold_exfiltration_is_rejected(self):
        bad = copy.deepcopy(self.dataset)
        bad["cases"][0]["answer"] = "LONG_STOP_ENTRY"
        with self.assertRaisesRegex(ValueError, "GOLD_LEAK"):
            validate_cases(bad, self.root)

    def test_chart_digest_tampering_is_rejected(self):
        bad = copy.deepcopy(self.dataset)
        bad["cases"][0]["chart_sha256"] = "a" * 64
        with self.assertRaisesRegex(ValueError, "CHART_BYTES_MISMATCH"):
            validate_cases(bad, self.root)

    def test_future_bar_is_rejected(self):
        bad = copy.deepcopy(self.dataset)
        bad["cases"][0]["cutoff_utc"] = "2026-07-01T00:00:00Z"
        with self.assertRaisesRegex(ValueError, "FUTURE_OR_NONMONOTONIC_BAR"):
            validate_cases(bad, self.root)

    def test_duplicate_snapshots_are_rejected(self):
        bad = copy.deepcopy(self.dataset)
        bad["cases"][1]["chart_sha256"] = bad["cases"][0]["chart_sha256"]
        with self.assertRaisesRegex(ValueError, "DUPLICATED_CHART_SNAPSHOT"):
            validate_cases(bad, self.root)

    def test_duplicate_external_provenance_is_rejected(self):
        bad = copy.deepcopy(self.dataset)
        bad["cases"][1]["provenance_id"] = bad["cases"][0]["provenance_id"]
        with self.assertRaisesRegex(ValueError, "DUPLICATE_PROVENANCE"):
            validate_cases(bad, self.root)

    def test_missing_fourth_book_is_rejected(self):
        bad = copy.deepcopy(self.dataset)
        bad["cases"] = [x for x in bad["cases"] if x["edition"] != "COURSE"]
        with self.assertRaisesRegex(ValueError, "MIN_32_REAL"):
            validate_cases(bad, self.root)

    @unittest.skipUnless(HAS_CRYPTO, "optional cryptography package not installed")
    def test_unsigned_unvetted_expert_does_not_grade(self):
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        manifest = validate_cases(self.dataset, self.root)
        review = self.signed("unknown", Ed25519PrivateKey.generate(), manifest)
        with self.assertRaisesRegex(ValueError, "REVIEWER_NOT_TRUSTED"):
            evaluate(self.dataset, manifest, review, review, {}, {}, chart_root=self.root)

    @unittest.skipUnless(HAS_CRYPTO, "optional cryptography package not installed")
    def test_signed_review_and_matching_protocol_still_never_promote(self):
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        from cryptography.hazmat.primitives import serialization
        manifest = validate_cases(self.dataset, self.root)
        keys = {name: Ed25519PrivateKey.generate() for name in ("a", "b")}
        trust = {}
        for name, key in keys.items():
            trust[name] = {
                "ed25519_public_key_hex": key.public_key().public_bytes(
                    encoding=serialization.Encoding.Raw,
                    format=serialization.PublicFormat.Raw).hex(),
                "externally_vetted_expert": True,
                "independent_of_training": True}
        signed_a = self.signed("a", keys["a"], manifest)
        signed_b = self.signed("b", keys["b"], manifest)
        gold = {x["id"]: x["decision"] for x in self.labels()}
        predictions = {
            "schema": "WILLIAMS_V6_DUAL_MODEL_SUBMISSION_V1",
            "dataset_sha256": manifest["dataset_sha256"],
            "protocol_sha256": "1" * 64,
            "models": {}}
        for name in ("candidate_v5", "matched_base"):
            predictions["models"][name] = {
                "weights_sha256": ("2" if name == "candidate_v5" else "4") * 64,
                "protocol_sha256": "1" * 64,
                "answers": [{"id": cid, "decision": label, "finish_reason": "stop"}
                            for cid, label in gold.items()]}
        result = evaluate(self.dataset, manifest, signed_a, signed_b,
                          trust, predictions, chart_root=self.root)
        self.assertEqual(result["agreed_reviewable_cases"], 32)
        self.assertEqual(result["matched_scores"]["candidate_v5"]["correct"], 32)
        self.assertTrue(result["case_diversity_gate"])
        self.assertFalse(result["95pct_research_threshold_met"])
        self.assertLess(result["wilson_95pct_lower_accuracy"], 0.95)
        self.assertFalse(result["execution_allowed"])
        self.assertFalse(result["automatic_model_promotion_allowed"])
        self.assertFalse(result["independent_expert_competence_externally_audited_by_this_code"])

        forged = copy.deepcopy(signed_a)
        forged["payload"]["labels"][0]["decision"] = "NO_SETUP"
        with self.assertRaisesRegex(ValueError, "REVIEW_SIGNATURE_INVALID"):
            evaluate(self.dataset, manifest, forged, signed_b,
                     trust, predictions, chart_root=self.root)

        wrong_protocol = copy.deepcopy(predictions)
        wrong_protocol["models"]["candidate_v5"]["protocol_sha256"] = "3" * 64
        with self.assertRaisesRegex(ValueError, "MISMATCHED_INFERENCE_PROTOCOL"):
            evaluate(self.dataset, manifest, signed_a, signed_b,
                     trust, wrong_protocol, chart_root=self.root)

        identical = copy.deepcopy(predictions)
        identical["models"]["matched_base"]["weights_sha256"] = "2" * 64
        with self.assertRaisesRegex(ValueError, "IDENTICAL_WEIGHTS"):
            evaluate(self.dataset, manifest, signed_a, signed_b,
                     trust, identical, chart_root=self.root)


if __name__ == "__main__":
    unittest.main()
