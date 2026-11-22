"""Prepare/optionally sign PRIVATE independent v6 expert review packets.

This tool NEVER certifies expertise, never fabricates review outcomes, and
never reads model predictions. Private signing seeds must stay offline;
do not upload seeds, review drafts or signed gold to GitHub.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re

from .independent_exam_v6 import canonical, validate_cases

_ID = re.compile(r"[A-Za-z0-9_-]{3,64}\Z")


def require(ok, code):
    if not ok:
        raise ValueError("V6_EXPERT_REVIEW_" + code)


def private_create(path: Path, data: bytes) -> None:
    """Create-only 0600, refuse overwrites and directory links."""
    parent = path.parent.resolve(strict=True)
    target = parent / path.name
    require(target == path.resolve(strict=False), "PRIVATE_PATH_NOT_LOCAL")
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
    except BaseException:
        target.unlink(missing_ok=True)
        raise


def template(cases: dict, manifest: dict, root: str | Path,
             reviewer_id: str) -> dict:
    require(type(reviewer_id) is str and _ID.fullmatch(reviewer_id) is not None,
            "INVALID_REVIEWER_ID")
    require(validate_cases(cases, root) == manifest, "CHART_COHORT_CHANGED")
    rows = []
    for case in cases["cases"]:
        rows.append({
            "id": case["id"],
            "decision": "INSUFFICIENT_DATA",
            "pages": list(case["pdf_pages"]),
            "reviewable": False,
            "rationale": (
                "NOT_REVIEWED: independently inspect the cited original book pages "
                "and cutoff-aligned chart; do not inspect AI predictions.")
        })
    return {"schema": "WILLIAMS_V6_EXPERT_REVIEW_DRAFT_V1",
            "payload": {
                "schema": "WILLIAMS_V6_EXPERT_REVIEW_V1",
                "dataset_sha256": manifest["dataset_sha256"],
                "reviewer_id": reviewer_id,
                "labels": rows},
            "instructions": [
                "PRIVATE reviewer-only draft; do not send to the model or GitHub.",
                "Check actual book PDF pages, original charts and independent novelty.",
                "For each case choose book-specific decision with rationale >=20 chars.",
                "Only set reviewable=true for cases fully substantiated by book and data.",
                "If uncertain, use INSUFFICIENT_DATA or reviewable=false; never fake a signal.",
                "Do not learn the candidate/base predictions before signing.",
            ]}


def keypair(private_file: Path, public_file: Path) -> None:
    require(not private_file.exists() and not public_file.exists(),
            "IDENTITY_PATH_ALREADY_EXISTS")
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives import serialization
    key = Ed25519PrivateKey.generate()
    secret = key.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption())
    public = key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw)
    private_create(private_file, secret)
    try:
        private_create(public_file, public.hex().encode("ascii") + b"\n")
    except BaseException:
        private_file.unlink(missing_ok=True)
        raise


def sign(draft: dict, private_file: Path) -> dict:
    require(type(draft) is dict and
            set(draft) == {"schema", "payload", "instructions"} and
            draft["schema"] == "WILLIAMS_V6_EXPERT_REVIEW_DRAFT_V1",
            "MALFORMED_DRAFT")
    payload = draft["payload"]
    require(type(payload) is dict and
            set(payload) == {"schema", "dataset_sha256", "reviewer_id", "labels"} and
            payload["schema"] == "WILLIAMS_V6_EXPERT_REVIEW_V1" and
            type(payload["labels"]) is list and len(payload["labels"]) >= 32,
            "INVALID_REVIEW_PAYLOAD")
    require(type(payload["reviewer_id"]) is str and
            _ID.fullmatch(payload["reviewer_id"]) is not None,
            "INVALID_REVIEWER_ID")
    ids = set()
    from .independent_exam_v6 import FAMILIES
    permitted = {x for family in FAMILIES.values() for x in family}
    for row in payload["labels"]:
        require(type(row) is dict and
                set(row) == {"id", "decision", "pages", "rationale", "reviewable"},
                "INVALID_REVIEW_ROW")
        require(row["id"] not in ids, "DUPLICATE_REVIEW_CASE")
        ids.add(row["id"])
        require(row["decision"] in permitted and
                type(row["reviewable"]) is bool and
                type(row["rationale"]) is str and
                20 <= len(row["rationale"]) <= 3000 and
                not row["rationale"].startswith("NOT_REVIEWED") and
                type(row["pages"]) is list and bool(row["pages"]) and
                all(type(p) is int and p > 0 for p in row["pages"]),
                "REVIEW_STILL_UNEXAMINED_OR_INVALID")
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    raw = private_file.read_bytes()
    require(len(raw) == 32, "INVALID_PRIVATE_SIGNING_KEY")
    signature = Ed25519PrivateKey.from_private_bytes(raw).sign(canonical(payload))
    return {"payload": payload, "signature_hex": signature.hex()}


def trust_scaffold(a_id: str, a_public: Path,
                   b_id: str, b_public: Path) -> dict:
    require(type(a_id) is str and type(b_id) is str and
            a_id != b_id and _ID.fullmatch(a_id) and _ID.fullmatch(b_id),
            "REVIEW_IDENTITIES_NOT_DISTINCT")
    a, b = a_public.read_text(encoding="ascii").strip(), b_public.read_text(encoding="ascii").strip()
    require(re.fullmatch(r"[0-9a-f]{64}", a) and
            re.fullmatch(r"[0-9a-f]{64}", b) and a != b,
            "REVIEW_KEYS_NOT_DISTINCT")
    return {
        name: {"ed25519_public_key_hex": pub,
               "externally_vetted_expert": False,
               "independent_of_training": False}
        for name, pub in ((a_id, a), (b_id, b))
    }


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    cmds = p.add_subparsers(dest="cmd", required=True)
    t = cmds.add_parser("template", help="Create a private unsigned gold-free reviewer draft")
    t.add_argument("--cases", type=Path, required=True)
    t.add_argument("--manifest", type=Path, required=True)
    t.add_argument("--chart-root", type=Path, required=True)
    t.add_argument("--reviewer-id", required=True)
    t.add_argument("--out", type=Path, required=True)
    k = cmds.add_parser("new-local-signing-key", help="Make 0600 offline Ed25519 signing seed")
    k.add_argument("--private-key", type=Path, required=True)
    k.add_argument("--public-key", type=Path, required=True)
    s = cmds.add_parser("sign", help="Sign an independently completed review")
    s.add_argument("--draft", type=Path, required=True)
    s.add_argument("--private-key", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--confirm-full-independent-review", action="store_true")
    trust = cmds.add_parser("trust-scaffold", help="Export public keys with vetting FALSE")
    trust.add_argument("--reviewer-a", required=True)
    trust.add_argument("--public-a", type=Path, required=True)
    trust.add_argument("--reviewer-b", required=True)
    trust.add_argument("--public-b", type=Path, required=True)
    trust.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.cmd == "new-local-signing-key":
        keypair(a.private_key, a.public_key)
        return
    require(not a.out.exists(), "DO_NOT_OVERWRITE")
    if a.cmd == "template":
        value = template(read_json(a.cases), read_json(a.manifest),
                         a.chart_root, a.reviewer_id)
    elif a.cmd == "sign":
        require(a.confirm_full_independent_review, "EXPLICIT_HUMAN_REVIEW_CONFIRMATION_REQUIRED")
        value = sign(read_json(a.draft), a.private_key)
    else:
        value = trust_scaffold(a.reviewer_a, a.public_a, a.reviewer_b, a.public_b)
    private_create(a.out, json.dumps(value, ensure_ascii=False,
                                    sort_keys=True, indent=2).encode("utf-8") + b"\n")


if __name__ == "__main__":
    main()
