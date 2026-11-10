"""Read-only fail-closed validation of old research before any new QLoRA run.

Never launches GPU training, unseals private artifacts, sends orders, or
copies private archives into the public repository.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import stat
from tempfile import SpooledTemporaryFile
from zipfile import ZipFile, BadZipFile

PINNED = {
    "joint": {
        "archive_sha256": "fa5ef2c7c6cfd4c3885b5640d8146cd80571445a7948b0161eed8a1dfa537222",
        "audit": "independent_joint_rule_audit.json",
        "source_commit": "26b5cabe8c11dde70c8f1c9792223d457da5ad1a",
        "candidate_sha256": "d961d7e7406d255e699da3959a3c67606330fee191807627b15de35ac4508ca7",
        "actual_updates": 2048,
    },
    "closed": {
        "archive_sha256": "934e0f3205d931a753c3ddf493452aaf4fcf9cec5a16ac793c119f0958db2c99",
        "audit": "independent_market_decision_audit.json",
        "source_commit": "bb6399e216e6f3d28a294ce015256f3e22eadb51",
        "candidate_sha256": "eebf29a3a7ab143dc162bbcab11240de0c66c2aa0d3da94f496f11abd542ddfb",
        "actual_updates": 1024,
    },
}


class PreflightError(ValueError):
    pass


def digest(path: Path) -> str:
    h = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def check_zip(z: ZipFile) -> set[str]:
    names = set()
    total = 0
    for entry in z.infolist():
        name = entry.filename
        path = PurePosixPath(name)
        if (not name or name.startswith("/") or "\\" in name or
                any(part in ("", ".", "..") for part in name.split("/")) or
                (entry.create_system == 3 and stat.S_ISLNK(entry.external_attr >> 16)) or
                name in names):
            raise PreflightError("UNSAFE_OR_DUPLICATE_ARCHIVE_ENTRY")
        if entry.flag_bits & 1:
            raise PreflightError("ENCRYPTED_TOP_LEVEL_ARCHIVE_UNSUPPORTED")
        names.add(name)
        total += entry.file_size
        if entry.file_size > 300_000_000 or total > 800_000_000:
            raise PreflightError("RESEARCH_ARCHIVE_SIZE_BOUND_EXCEEDED")
    return names


def read_json(z: ZipFile, path: str) -> dict:
    info = z.getinfo(path)
    if info.file_size > 2_000_000:
        raise PreflightError("RESEARCH_RECEIPT_TOO_LARGE")
    data = json.loads(z.read(info))
    if not isinstance(data, dict):
        raise PreflightError("RESEARCH_RECEIPT_MUST_BE_OBJECT")
    return data


def assess_audit(audit: dict, kind: str) -> dict:
    """Check pinned immutable research audit without trusting its trading claims."""
    pin = PINNED[kind]
    for field, expected in (
        ("source_commit", pin["source_commit"]),
        ("candidate_adapter_sha256", pin["candidate_sha256"]),
        ("actual_updates", pin["actual_updates"]),
        ("execution_allowed", False),
        ("trading_qualification", False),
    ):
        if type(audit.get(field)) is not type(expected) or audit[field] != expected:
            raise PreflightError("INVALID_OR_UNPINNED_" + field.upper())
    if not (isinstance(audit.get("selected_checkpoint"), str)
            and audit["selected_checkpoint"].startswith("checkpoint-")):
        raise PreflightError("MISSING_SELECTED_CHECKPOINT")
    if kind == "closed":
        gates = audit.get("cohort_gates")
        if not isinstance(gates, dict) or set(gates) != {"context", "waves", "campaign", "precision"}:
            raise PreflightError("CLOSED_COHORTS_MISSING")
        if audit.get("research_gate_passed") is not False:
            raise PreflightError("HISTORICAL_ADMISSION_UNEXPECTEDLY_CHANGED")
    else:
        gates = audit.get("transfer_gates")
        if not isinstance(gates, dict) or set(gates) != {"numeric", "boundary"}:
            raise PreflightError("JOINT_COHORTS_MISSING")
        if audit.get("pilot_research_gates_passed") is not False:
            raise PreflightError("JOINT_ADMISSION_UNEXPECTEDLY_CHANGED")
    if any(type(v) is not bool for v in gates.values()):
        raise PreflightError("NONBOOLEAN_COHORT_GATE")
    books = audit.get("text_gates")
    if not isinstance(books, dict) or not books or any(
        not isinstance(x, dict) or type(x.get("accepted")) is not bool
        for x in books.values()
    ):
        raise PreflightError("BOOK_REGRESSION_REPORT_MISSING")
    return {
        "source_commit": pin["source_commit"],
        "candidate_sha256": pin["candidate_sha256"],
        "actual_updates": pin["actual_updates"],
        "selected_checkpoint": audit["selected_checkpoint"],
        "cohort_gates": gates,
        "failed_book_gates": sorted(k for k, v in books.items() if not v["accepted"]),
        "research_admission": "FAILED",
        "trade_activation": False,
    }


def inspect(path: Path, kind: str) -> dict:
    pin = PINNED[kind]
    if not path.is_file() or path.is_symlink():
        raise PreflightError("PRIVATE_ARCHIVE_NOT_FOUND_OR_SYMLINK")
    actual_hash = digest(path)
    if actual_hash != pin["archive_sha256"]:
        raise PreflightError("PINNED_RESEARCH_ARCHIVE_SHA256_MISMATCH")
    try:
        with ZipFile(path) as z:
            names = check_zip(z)
            if pin["audit"] not in names:
                raise PreflightError("INDEPENDENT_AUDIT_NOT_FOUND")
            report = assess_audit(read_json(z, pin["audit"]), kind)
            if kind == "joint":
                member = "Williams_Qwen3_4B_JOINT_RULES_20261106.zip"
                if member not in names:
                    raise PreflightError("PINNED_BASELINE_ADAPTER_ARCHIVE_MISSING")
                with SpooledTemporaryFile(max_size=8 * 1024 * 1024) as tmp:
                    with z.open(member) as stream:
                        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                            tmp.write(block)
                    tmp.seek(0)
                    with ZipFile(tmp) as nested:
                        inside = check_zip(nested)
                        weight = "candidate_adapter/adapter_model.safetensors"
                        if weight not in inside:
                            raise PreflightError("JOINT_ADAPTER_WEIGHTS_MISSING")
                        h = sha256()
                        with nested.open(weight) as stream:
                            for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                                h.update(block)
                        if h.hexdigest() != pin["candidate_sha256"]:
                            raise PreflightError("JOINT_ADAPTER_WEIGHT_SHA256_MISMATCH")
                report["adapter_weights_recovered_and_hashed"] = True
            else:
                if "Williams_Qwen3_4B_CLOSED_CHART_20261107.zip.sealed" not in names:
                    raise PreflightError("CLOSED_CHART_SEALED_ADAPTER_MISSING")
                report["adapter_weights_recovered_and_hashed"] = False
                report["adapter_access_blocker"] = "SEALED_PAYLOAD_REQUIRES_AUTHORIZED_DECRYPTION"
        report["archive_sha256"] = actual_hash
        report["archive_integrity"] = "PINNED_HASH_VERIFIED"
        return report
    except (BadZipFile, OSError, json.JSONDecodeError, KeyError) as exc:
        raise PreflightError("CORRUPT_OR_INCOMPLETE_RESEARCH_ARCHIVE") from exc


def run(joint: Path, closed: Path) -> dict:
    return {
        "mode": "READ_ONLY_DEEP_QLORA_PREPARATION",
        "research_archives": {
            "2026-11-06": inspect(joint, "joint"),
            "2026-11-07": inspect(closed, "closed"),
        },
        "ready_for_next_gpu_training": False,
        "missing_prerequisites": [
            "AUTHORIZED_RECOVERY_OF_LATEST_SEALED_ADAPTER",
            "FRESH_UNEXPOSED_SOURCE_AND_CHART_HOLDOUT_PRECOMMITTED",
            "INDEPENDENT_REVIEW_OF_RISK_STOP_AC_LABELS",
            "CURRENT_GPU_QUOTA_AND_CHECKPOINT_RECOVERY_VERIFIED",
        ],
        "research_admission": "FAILED",
        "trading_qualification": False,
        "execution_allowed": False,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--joint-archive", type=Path, required=True)
    p.add_argument("--closed-archive", type=Path, required=True)
    a = p.parse_args()
    try:
        print(json.dumps(run(a.joint_archive, a.closed_archive), ensure_ascii=False, indent=2))
    except PreflightError as exc:
        print(json.dumps({"status":"BLOCKED","reason":str(exc),"execution_allowed":False}))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
