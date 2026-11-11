"""Ed25519 verification of two separately keyed Course expert attestations.

Cryptographic authentication shows two *trusted signing keys* approved the
exact market case and original page image. It cannot prove the reviewers
are competent, independent people or that the interpretation is correct.
Private signing keys NEVER enter ChatGPT, this repository or CI artifacts.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

from .reasoning_research_v2 import canonical, digest, utc


def approval_message(*,case_sha256,pdf_page,image_sha256,source_sha256,
                     reviewer_id,reviewed_at,verdict):
    payload={"schema":"WILLIAMS_AI_COURSE_REVIEW_SIGNATURE_V3",
             "case_sha256":case_sha256,"pdf_page":pdf_page,
             "image_sha256":image_sha256,"source_sha256":source_sha256,
             "reviewer_id":reviewer_id,"reviewed_at":reviewed_at,
             "verdict":verdict}
    if (type(pdf_page) is not int or not 1<=pdf_page<=122 or
            verdict!="APPROVED" or
            not isinstance(reviewer_id,str) or not 5<=len(reviewer_id)<=100):
        raise ValueError("EXPLICIT_APPROVED_COURSE_CASE_AND_REVIEWER")
    for k in ("case_sha256","image_sha256","source_sha256"):
        v=payload[k]
        if not isinstance(v,str) or len(v)!=64 or any(c not in "0123456789abcdef" for c in v):
            raise ValueError("CRYPTOGRAPHIC_COURSE_CASE_DIGEST_REQUIRED")
    signed=utc(reviewed_at)
    if signed>datetime.now(timezone.utc):
        raise ValueError("FUTURE_DATED_HUMAN_REVIEW_FORBIDDEN")
    return canonical(payload).encode("utf-8")


def verify_two_ed25519_course_approvals(db,*,case_sha256,pdf_page,
                                       image_sha256,source_sha256,
                                       expected_reviewers):
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    except ImportError as exc:
        raise RuntimeError("CRYPTOGRAPHY_ED25519_REQUIRED_FOR_COURSE_GOLD") from exc
    if not isinstance(expected_reviewers,set) or len(expected_reviewers)!=2:
        raise ValueError("EXACT_TWO_INDEPENDENT_REVIEWER_IDENTITIES_REQUIRED")
    tables={r[0] for r in db.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    if not {"trusted_reviewer_keys_v3","course_expert_case_reviews_v3"}<=tables:
        raise ValueError("TRUSTED_SIGNER_REGISTRY_AND_CASE_ATTESTATIONS_REQUIRED")
    rows=db.execute(
        "SELECT c.reviewer_id,c.reviewed_at,c.verdict,c.signature_hex,"
        "t.public_key_hex FROM course_expert_case_reviews_v3 AS c "
        "JOIN trusted_reviewer_keys_v3 AS t ON t.reviewer_id=c.reviewer_id "
        "WHERE c.case_sha256=? AND c.pdf_page=? AND c.image_sha256=? "
        "AND c.source_sha256=?",
        (case_sha256,pdf_page,image_sha256,source_sha256)
    ).fetchall()
    validated=set()
    keys=set()
    for reviewer,at,verdict,signature_hex,public_key_hex in rows:
        if reviewer not in expected_reviewers or reviewer in validated:
            continue
        try:
            pub=bytes.fromhex(public_key_hex)
            sig=bytes.fromhex(signature_hex)
            if len(pub)!=32 or len(sig)!=64:
                continue
            Ed25519PublicKey.from_public_bytes(pub).verify(
                sig,approval_message(
                    case_sha256=case_sha256,pdf_page=pdf_page,
                    image_sha256=image_sha256,source_sha256=source_sha256,
                    reviewer_id=reviewer,reviewed_at=at,verdict=verdict))
        except (ValueError,TypeError,OverflowError):
            continue
        except Exception as exc:
            # cryptography.exceptions.InvalidSignature is a standard
            # negative attestation, not an authenticated approval.
            if type(exc).__name__=="InvalidSignature":
                continue
            raise
        if pub in keys:
            raise ValueError("ONE_SIGNING_KEY_REUSED_FOR_TWO_REVIEWERS")
        keys.add(pub)
        validated.add(reviewer)
    if validated!=expected_reviewers:
        raise ValueError("TWO_VALID_TRUSTED_COURSE_SIGNER_ATTESTATIONS_REQUIRED")
    return {"source_attested_by_two_distinct_keys":True,
            "signer_identities":sorted(validated),
            "human_expertise_not_proven_by_cryptography":True,
            "trade_execution_allowed":False}
