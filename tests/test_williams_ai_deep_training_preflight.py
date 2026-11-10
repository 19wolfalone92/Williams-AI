"""Deterministic offline contract tests: no private data or GPU required."""
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import warnings
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "ai_deep_training_preflight", ROOT / "scripts" / "ai_deep_training_preflight.py"
)
preflight = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(preflight)


def fake_audit(kind):
    pin = preflight.PINNED[kind]
    a = {
        "source_commit": pin["source_commit"],
        "candidate_adapter_sha256": pin["candidate_sha256"],
        "actual_updates": pin["actual_updates"],
        "execution_allowed": False,
        "trading_qualification": False,
        "selected_checkpoint": "checkpoint-512",
        "text_gates": {"TC1": {"accepted": False}},
    }
    if kind == "closed":
        a.update(
            cohort_gates={k:False for k in ("campaign","context","precision","waves")},
            research_gate_passed=False,
        )
    else:
        a.update(transfer_gates={"boundary":False, "numeric":False},
                 pilot_research_gates_passed=False)
    return a


class DeepTrainingPreflightTests(unittest.TestCase):
    def test_old_research_is_not_admitted(self):
        for kind in preflight.PINNED:
            report = preflight.assess_audit(fake_audit(kind),kind)
            self.assertEqual(report["research_admission"], "FAILED")
            self.assertFalse(report["trade_activation"])

    def test_forged_trading_permission_or_changed_receipt_rejected(self):
        for kind in preflight.PINNED:
            for key, value in [
                ("execution_allowed", True),
                ("trading_qualification", True),
                ("actual_updates", 1),
                ("source_commit", "f"*40),
                ("candidate_adapter_sha256", "f"*64),
                ("selected_checkpoint", ""),
            ]:
                with self.subTest(kind=kind,key=key):
                    a=fake_audit(kind)
                    a[key]=value
                    with self.assertRaises(preflight.PreflightError):
                        preflight.assess_audit(a,kind)

    def test_missing_research_or_book_gate_rejected(self):
        for kind in preflight.PINNED:
            a=fake_audit(kind)
            a["research_gate_passed" if kind=="closed" else "pilot_research_gates_passed"] = True
            with self.assertRaises(preflight.PreflightError):
                preflight.assess_audit(a,kind)
            a=fake_audit(kind)
            cohort="cohort_gates" if kind=="closed" else "transfer_gates"
            a[cohort].pop(next(iter(a[cohort])))
            with self.assertRaises(preflight.PreflightError):
                preflight.assess_audit(a,kind)
            a=fake_audit(kind);a.pop("text_gates")
            with self.assertRaises(preflight.PreflightError):
                preflight.assess_audit(a,kind)

    def test_zip_slip_duplicate_and_nonobject_json_rejected(self):
        with TemporaryDirectory() as d:
            p=Path(d)/"tmp.zip"
            with ZipFile(p,"w") as z:z.writestr("../escape", "bad")
            with ZipFile(p) as z:
                with self.assertRaises(preflight.PreflightError):
                    preflight.check_zip(z)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore",UserWarning)
                with ZipFile(p,"w") as z:
                    z.writestr("same","1");z.writestr("same","2")
            with ZipFile(p) as z:
                with self.assertRaises(preflight.PreflightError):
                    preflight.check_zip(z)
            with ZipFile(p,"w") as z:
                z.writestr("report.json",json.dumps([1,2]))
            with ZipFile(p) as z:
                with self.assertRaises(preflight.PreflightError):
                    preflight.read_json(z,"report.json")


if __name__ == "__main__":
    unittest.main()
