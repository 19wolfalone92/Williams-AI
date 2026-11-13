"""Fail-closed, CPU-only intake/grading for genuinely unseen Williams expert charts.

No neural inference, exchange access, model promotion, GPU, or training writes.
Experts and independence MUST be vetted outside this code; cryptographic review
signatures authenticate registered keys, not a person's expertise.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import re

SCHEMA = "WILLIAMS_V6_UNSEEN_EXPERT_EXAM_V1"
BOOKS = ("TC1", "NTD", "TC2", "COURSE")
FAMILIES = {"ENTRY": ("LONG_STOP_ENTRY", "SHORT_STOP_ENTRY", "NO_SETUP", "INSUFFICIENT_DATA"),
            "EXIT": ("EXIT_LONG", "EXIT_SHORT", "HOLD", "INSUFFICIENT_DATA"),
            "WAVE": ("WAVE_CONFIRMED", "WAVE_HYPOTHESIS", "WAVE_INVALID", "INSUFFICIENT_DATA")}
PDF_SHA = {
 "TC1": "36f6822811088a06254009181480aa1c27af705a76960ecf0342023df803331c",
 "NTD": "855eb052353c852f208436952b12e46d9a18badf7c1dac7ade7580d57b9f4bb4",
 "TC2": "055b7924f9a4d165294bf3e12069e74b121f6ca510f25ad310a07c94f0a6107f",
 "COURSE": "fc57f707378087bfc5146db5e242c51ce8135e0b359f0c607305b364e0decfca"}
PAGES = {"TC1": 140, "NTD": 156, "TC2": 224, "COURSE": 122}
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_ID = re.compile(r"E6-[A-Za-z0-9_-]{8,48}\Z")


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False,
                      separators=(",", ":")).encode("utf-8")


def digest(value):
    return sha256(canonical(value)).hexdigest()


def need(ok, reason):
    if not ok:
        raise ValueError("EXPERT_V6_" + reason)


def _utc(value):
    need(isinstance(value, str) and value.endswith("Z"), "TIME_NOT_UTC")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as e:
        raise ValueError("EXPERT_V6_INVALID_TIME") from e
    need(parsed.tzinfo is not None, "TIME_UNAWARE")
    return parsed.astimezone(timezone.utc)


def _no_gold(value):
    """Public cases cannot contain labels/gold/predictions anywhere, even nested."""
    if isinstance(value, dict):
        for k, v in value.items():
            need(k not in {"gold", "expected", "correct", "answer", "label", "decision",
                           "review", "expert_vote", "prediction"}, "GOLD_LEAK_IN_PUBLIC_CASE")
            _no_gold(v)
    elif isinstance(value, list):
        for v in value:
            _no_gold(v)


def _load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_cases(data, chart_root, *, current_time=None):
    # Optional clock injection is for offline mock tests. CLI uses UTC now.
    if current_time is None:
        current_time = datetime.now(timezone.utc)
    need(isinstance(current_time, datetime) and
         current_time.tzinfo is not None, "INVALID_VALIDATION_CLOCK")
    need(type(data) is dict and data.get("schema") == SCHEMA, "BAD_CASE_SCHEMA")
    need(set(data) == {"schema", "dataset_id", "cases", "provenance_note"}, "UNEXPECTED_CASE_FIELDS")
    need(type(data["dataset_id"]) is str and 8 <= len(data["dataset_id"]) <= 100,
         "DATASET_ID")
    cases = data["cases"]
    need(type(cases) is list and len(cases) >= 32, "MIN_32_REAL_EXPERT_CANDIDATES")
    need(len(cases) <= 4096, "TOO_MANY_CASES")
    _no_gold(data)
    root = Path(chart_root).resolve(strict=True)
    seen_ids, seen_charts, seen_provenance, by_book, by_family, dates = (
        set(), set(), set(), Counter(), Counter(), set())
    public_fields = {"id", "edition", "family", "chart_file", "chart_sha256", "cutoff_utc",
                     "source_pdf_sha256", "pdf_pages", "question", "provenance_id"}
    for row in cases:
        need(type(row) is dict and set(row) == public_fields, "UNEXPECTED_PUBLIC_CASE_FIELDS")
        cid = row["id"]
        need(type(cid) is str and _ID.fullmatch(cid), "INVALID_OR_REUSED_CASE_ID")
        need(cid not in seen_ids, "DUPLICATE_CASE_ID")
        seen_ids.add(cid)
        ed, family = row["edition"], row["family"]
        need(ed in BOOKS and family in FAMILIES, "UNKNOWN_BOOK_OR_FAMILY")
        need(row["source_pdf_sha256"] == PDF_SHA[ed], "UNPINNED_SOURCE_EDITION")
        pages = row["pdf_pages"]
        need(type(pages) is list and bool(pages) and all(
             type(p) is int and 1 <= p <= PAGES[ed] for p in pages), "INVALID_PDF_PAGE")
        need(type(row["question"]) is str and 35 <= len(row["question"]) <= 1500,
             "QUESTION_NOT_FROZEN")
        need(type(row["provenance_id"]) is str and len(row["provenance_id"]) >= 8,
             "MISSING_EXTERNAL_PROVENANCE")
        need(row["provenance_id"] not in seen_provenance, "DUPLICATE_PROVENANCE")
        seen_provenance.add(row["provenance_id"])
        cutoff = _utc(row["cutoff_utc"])
        need(cutoff <= current_time, "FUTURE_CUTOFF")
        dates.add(cutoff.date().isoformat())
        chart_sha = row["chart_sha256"]
        need(type(chart_sha) is str and _HEX64.fullmatch(chart_sha), "INVALID_CHART_DIGEST")
        need(chart_sha not in seen_charts, "DUPLICATED_CHART_SNAPSHOT")
        seen_charts.add(chart_sha)
        rel = row["chart_file"]
        need(type(rel) is str and rel and not Path(rel).is_absolute(), "INVALID_CHART_RELATIVE_PATH")
        file = (root / rel).resolve(strict=True)
        need(file.is_relative_to(root) and file.is_file() and file.suffix.lower() == ".json",
             "UNSAFE_CHART_PATH")
        raw = file.read_bytes()
        need(len(raw) <= 3_000_000 and sha256(raw).hexdigest() == chart_sha,
             "CHART_BYTES_MISMATCH")
        chart = json.loads(raw)
        need(type(chart) is dict and set(chart) == {"symbol", "bars"}, "INVALID_CHART_SCHEMA")
        need(type(chart["symbol"]) is str and 3 <= len(chart["symbol"]) <= 32,
             "INVALID_SYMBOL")
        bars = chart["bars"]
        need(type(bars) is list and len(bars) >= 64, "INSUFFICIENT_CHART_BARS")
        last = None
        for bar in bars:
            need(type(bar) is dict and set(bar) == {
                "open_utc", "close_utc", "open", "high", "low", "close", "volume"},
                "BAD_BAR_SCHEMA")
            start, end = _utc(bar["open_utc"]), _utc(bar["close_utc"])
            need(start < end <= cutoff and (last is None or start >= last),
                 "FUTURE_OR_NONMONOTONIC_BAR")
            need(end-start == timedelta(hours=1)
                 and (last is None or start == last),
                 "NONCONTIGUOUS_H1_BARS")
            last = end
            nums = [bar[k] for k in ("open", "high", "low", "close", "volume")]
            need(all(type(x) in (int, float) and math.isfinite(x) for x in nums),
                 "NONFINITE_PRICE_OR_VOLUME")
            o, high, low, close, volume = nums
            need(0 < low <= min(o, close) <= max(o, close) <= high and volume >= 0,
                 "INVALID_OHLCV")
        need(last == cutoff, "CHART_DOES_NOT_END_AT_FROZEN_CUTOFF")
        by_book[ed] += 1
        by_family[family] += 1
    need(all(by_book[b] >= 8 for b in BOOKS), "UNBALANCED_FOUR_BOOKS")
    need(all(by_family[f] >= 4 for f in FAMILIES), "INSUFFICIENT_TASK_DIVERSITY")
    need(len(dates) >= 8, "INSUFFICIENT_DATE_DIVERSITY")
    return {"schema": "WILLIAMS_V6_EXPERT_FROZEN_MANIFEST_V1", "dataset_id": data["dataset_id"],
            "case_count": len(cases), "dataset_sha256": digest(data),
            "by_book": dict(by_book), "by_family": dict(by_family),
            "distinct_utc_dates": len(dates),
            "public_case_schema_does_not_contain_gold": True, "expert_labels_attached": False,
            "case_id_namespace_excludes_v5": True,
            "novelty_vs_training_or_prior_exams_externally_unverified": True,
            "execution_allowed": False, "training_allowed": False}


def _verify_review(bundle, case_ids, manifest, trust, book_of):
    need(type(bundle) is dict and set(bundle) == {"payload", "signature_hex"}, "BAD_REVIEW_ENVELOPE")
    payload = bundle["payload"]
    need(type(payload) is dict and set(payload) == {
        "schema", "dataset_sha256", "reviewer_id", "labels"}, "INVALID_REVIEW_FIELDS")
    need(payload["schema"] == "WILLIAMS_V6_EXPERT_REVIEW_V1" and
         payload["dataset_sha256"] == manifest["dataset_sha256"], "REVIEW_NOT_FOR_COHORT")
    rid = payload["reviewer_id"]
    who = trust.get(rid) if isinstance(trust, dict) else None
    need(type(who) is dict and who.get("externally_vetted_expert") is True and
         who.get("independent_of_training") is True, "REVIEWER_NOT_TRUSTED")
    pub = who.get("ed25519_public_key_hex")
    need(type(pub) is str and re.fullmatch(r"[0-9a-f]{64}", pub), "INVALID_PUBLIC_KEY")
    sig = bundle["signature_hex"]
    need(type(sig) is str and re.fullmatch(r"[0-9a-f]{128}", sig), "UNSIGNED_REVIEW")
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        Ed25519PublicKey.from_public_bytes(bytes.fromhex(pub)).verify(
            bytes.fromhex(sig), canonical(payload))
    except Exception as exc:
        raise ValueError("EXPERT_V6_REVIEW_SIGNATURE_INVALID") from exc
    labels = payload["labels"]
    need(type(labels) is list and len(labels) == len(case_ids), "REVIEW_COUNT_MISMATCH")
    need(len({v.get("id") for v in labels}) == len(labels) and
         {v["id"] for v in labels} == case_ids, "REVIEW_CASE_IDS_MISMATCH")
    result = {}
    for v in labels:
        need(set(v) == {"id", "decision", "pages", "rationale", "reviewable"},
             "EXTRA_REVIEW_FIELDS")
        family = book_of[v["id"]][1]
        decision = v["decision"]
        need(decision in FAMILIES[family], "UNSUPPORTED_EXPERT_DECISION")
        need(type(v["reviewable"]) is bool, "INVALID_REVIEWABLE")
        need(type(v["rationale"]) is str and 20 <= len(v["rationale"]) <= 3000,
             "MISSING_EXPERT_RATIONALE")
        need(type(v["pages"]) is list and all(
             type(x) is int and x in book_of[v["id"]][2] for x in v["pages"])
             and bool(v["pages"]), "UNSUPPORTED_BOOK_PAGES")
        result[v["id"]] = v
    return rid, result


def evaluate(cases, manifest, review_a, review_b, trust, predictions, *, chart_root):
    frozen = validate_cases(cases, chart_root)
    need(frozen == manifest, "FROZEN_MANIFEST_NOT_IDENTICAL")
    by_id = {c["id"]: (c["edition"], c["family"], c["pdf_pages"]) for c in cases["cases"]}
    ids = set(by_id)
    a_id, a = _verify_review(review_a, ids, manifest, trust, by_id)
    b_id, b = _verify_review(review_b, ids, manifest, trust, by_id)
    need(a_id != b_id and trust[a_id]["ed25519_public_key_hex"] !=
         trust[b_id]["ed25519_public_key_hex"], "REVIEWERS_NOT_INDEPENDENT_IDENTITIES")
    need(type(predictions) is dict and set(predictions) == {
        "schema", "dataset_sha256", "protocol_sha256", "models"},
        "INVALID_SUBMISSION_FIELDS")
    need(predictions["schema"] == "WILLIAMS_V6_DUAL_MODEL_SUBMISSION_V1" and
         predictions["dataset_sha256"] == manifest["dataset_sha256"], "PREDICTION_COHORT_CHANGED")
    protocol = predictions["protocol_sha256"]
    need(type(protocol) is str and _HEX64.fullmatch(protocol), "UNPINNED_DECODING_PROTOCOL")
    models = predictions["models"]
    need(type(models) is dict and set(models) == {"candidate_v5", "matched_base"},
         "MATCHED_MODEL_CONTROL_REQUIRED")
    need(models["candidate_v5"].get("weights_sha256") !=
         models["matched_base"].get("weights_sha256"), "IDENTICAL_WEIGHTS_NOT_COMPARABLE")
    answers = {}
    for name, payload in models.items():
        need(type(payload) is dict and set(payload) == {
            "weights_sha256", "protocol_sha256", "answers"}, "MODEL_SUBMISSION_FIELDS")
        need(type(payload["weights_sha256"]) is str and
             _HEX64.fullmatch(payload["weights_sha256"]) and
             payload["protocol_sha256"] == protocol, "MISMATCHED_INFERENCE_PROTOCOL")
        rows = payload["answers"]
        need(type(rows) is list and len(rows) == len(ids) and
             len({r.get("id") for r in rows}) == len(ids), "INCOMPLETE_MODEL_SUBMISSION")
        need({r["id"] for r in rows} == ids, "MODEL_CASE_ID_MISMATCH")
        ans = {}
        for item in rows:
            need(type(item) is dict and set(item) == {
                "id", "decision", "finish_reason"}, "UNEXPECTED_MODEL_ANSWER_FIELDS")
            family = by_id[item["id"]][1]
            need(item["decision"] in (*FAMILIES[family], "ABSTAIN") and
                 item["finish_reason"] in ("stop", "length", "error"),
                 "INVALID_MODEL_RESPONSE")
            if item["finish_reason"] != "stop":
                need(item["decision"] == "ABSTAIN", "TRUNCATION_LAUNDERED_AS_CORRECT")
            ans[item["id"]] = item["decision"]
        answers[name] = ans
    decisional = {cid: a[cid]["decision"] for cid in ids
                  if a[cid]["reviewable"] and b[cid]["reviewable"] and
                  a[cid]["decision"] == b[cid]["decision"]}
    disagreements = len(ids) - len(decisional)
    counts = Counter(by_id[cid][0] for cid in decisional)
    dates = {next(c["cutoff_utc"][:10] for c in cases["cases"] if c["id"] == cid)
             for cid in decisional}
    diversity = Counter(decisional.values())
    enough = (len(decisional) >= 32 and all(counts[b] >= 8 for b in BOOKS)
              and len(dates) >= 8 and len(diversity) >= 3
              and any(d in diversity for d in ("LONG_STOP_ENTRY", "SHORT_STOP_ENTRY"))
              and "NO_SETUP" in diversity)
    scores = {}
    for name, result in answers.items():
        by_book = {b: {"correct": 0, "total": counts[b], "abstained": 0,
                       "critical_overcalls": 0} for b in BOOKS}
        for cid, expected in decisional.items():
            ed, family, _ = by_id[cid]
            choice = result[cid]
            by_book[ed]["correct"] += choice == expected
            by_book[ed]["abstained"] += choice == "ABSTAIN"
            by_book[ed]["critical_overcalls"] += (
                family == "ENTRY" and (
                    (expected in ("NO_SETUP", "INSUFFICIENT_DATA") and
                     choice in ("LONG_STOP_ENTRY", "SHORT_STOP_ENTRY")) or
                    (expected == "LONG_STOP_ENTRY" and choice == "SHORT_STOP_ENTRY") or
                    (expected == "SHORT_STOP_ENTRY" and choice == "LONG_STOP_ENTRY")))
        scores[name] = {"correct": sum(b["correct"] for b in by_book.values()),
                        "total": len(decisional),
                        "critical_overcalls": sum(b["critical_overcalls"] for b in by_book.values()),
                        "by_edition": by_book}
    candidate, base = scores["candidate_v5"], scores["matched_base"]
    nonregress = all(candidate["by_edition"][b]["correct"] >=
                     base["by_edition"][b]["correct"] for b in BOOKS)
    n = len(decisional)
    p = candidate["correct"] / n if n else 0.0
    z = 1.959963984540054
    denom = 1 + z*z/n if n else 1
    wilson_lower = ((p + z*z/(2*n) - z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))) / denom
                    if n else 0.0)
    meets = (enough and p >= 0.95 and wilson_lower >= 0.95
             and candidate["correct"] > base["correct"]
             and candidate["critical_overcalls"] == 0 and nonregress)
    return {"schema": "WILLIAMS_V6_EXPERT_RESEARCH_SCORE_V1",
            "dataset_sha256": manifest["dataset_sha256"],
            "protocol_sha256": protocol, "registered_distinct_signed_reviewers": 2,
            "independent_expert_competence_externally_audited_by_this_code": False,
            "agreed_reviewable_cases": len(decisional), "disputed_or_unreviewable_cases": disagreements,
            "per_book_agreement": dict(counts), "class_distribution": dict(diversity),
            "case_diversity_gate": enough, "matched_scores": scores,
            "95pct_research_threshold_met": meets,
            "wilson_95pct_lower_accuracy": wilson_lower,
            "wilson_assumes_independent_cases_not_established_by_this_tool": True,
            "overlapping_1024h_windows_and_same_day_clusters_need_external_review": True,
            "strictly_better_count_than_base_required": True,
            "automatic_model_promotion_allowed": False,
            "training_allowed": False, "GPU_allowed": False,
            "execution_allowed": False, "main_merge_allowed": False}


def save_exclusive(out, data):
    path = Path(out)
    with path.open("x", encoding="utf-8") as fp:
        json.dump(data, fp, ensure_ascii=False, indent=2, sort_keys=True)
        fp.write("\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    s = p.add_subparsers(dest="cmd", required=True)
    freeze = s.add_parser("freeze", help="Validate closed OHLCV and freeze gold-free case manifest")
    freeze.add_argument("--cases", required=True)
    freeze.add_argument("--chart-root", required=True)
    freeze.add_argument("--out", required=True)
    grade = s.add_parser("grade", help="Verify two Ed25519 expert reviewers and matched model scores")
    for flag in ("cases", "chart-root", "manifest", "review-a", "review-b", "trust", "predictions", "out"):
        grade.add_argument("--" + flag, required=True)
    a = p.parse_args()
    if a.cmd == "freeze":
        save_exclusive(a.out, validate_cases(_load(a.cases), a.chart_root))
    else:
        save_exclusive(a.out, evaluate(_load(a.cases), _load(a.manifest), _load(a.review_a),
                                       _load(a.review_b), _load(a.trust), _load(a.predictions),
                                       chart_root=a.chart_root))


if __name__ == "__main__":
    main()
