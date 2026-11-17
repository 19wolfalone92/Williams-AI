"""Build private expert-agreed train/dev candidates; sealed holdout kept separate.

Never reads model predictions. Requires frozen gold-free cases, pre-registered
split assignment, book-rule links, and two distinct Ed25519 reviewers already
marked as externally vetted and training-independent. Outputs are create-only
0600 local files, never written to GitHub or workflow artifacts.
"""
from __future__ import annotations
import argparse,json,os,re
from collections import Counter
from datetime import datetime
from pathlib import Path

from .independent_exam_v6 import (
    BOOKS,PDF_SHA,canonical,digest,validate_cases,_verify_review
)
from .book_rules import registry

SCHEMA="WILLIAMS_V7_HUMAN_GROUNDED_TRAINING_CANDIDATES_V1"
SPLIT_SCHEMA="WILLIAMS_V7_PRE_REGISTERED_SPLIT_PLAN_V1"
RULEMAP_SCHEMA="WILLIAMS_V7_CASE_TO_SOURCE_RULE_MAP_V1"
FORBIDDEN={"prediction","model_prediction","model_output","assistant_answer","corrected_by_model",
 "auto_label","pseudo_gold","synthetic_gold","gold_from_model","qwen_answer"}

def need(ok,why):
    if not ok:raise ValueError("V7_TRAIN_DATA_"+why)

def no_model_generated(value):
    if isinstance(value,dict):
        for k,v in value.items():
            need(k not in FORBIDDEN,"MODEL_PSEUDO_LABEL_FIELD")
            no_model_generated(v)
    elif isinstance(value,list):
        for x in value:no_model_generated(x)

def _load(p):return json.loads(Path(p).read_text(encoding="utf-8"))

def _private_create(path,obj):
    dest=Path(path)
    parent=dest.parent.resolve(strict=True)
    need(dest.resolve(strict=False)==parent/dest.name,"OUTPUT_NOT_LOCAL")
    raw=json.dumps(obj,ensure_ascii=False,sort_keys=True,indent=2).encode()+b"\n"
    fd=os.open(dest,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    try:
        with os.fdopen(fd,"wb") as f:f.write(raw);f.flush();os.fsync(f.fileno())
    except BaseException:
        dest.unlink(missing_ok=True);raise
    need((dest.stat().st_mode & 0o777)==0o600,"OUTPUT_NOT_PRIVATE")

def _utc(s):
    return datetime.fromisoformat(s.replace("Z","+00:00"))

def _window(chart):
    bars=chart.get("bars")
    need(isinstance(bars,list) and len(bars)>=64,"BAD_CHART_BARS")
    return _utc(bars[0]["open_utc"]),_utc(bars[-1]["close_utc"])

def _validate_split(plan,cases,charts):
    need(type(plan) is dict and set(plan)=={"schema","dataset_sha256","assignments","frozen_before_reviews"}
         and plan["schema"]==SPLIT_SCHEMA,"SPLIT_SCHEMA")
    ids={x["id"] for x in cases["cases"]}
    need(plan["frozen_before_reviews"] is True,"SPLIT_NOT_PRE_REGISTERED")
    need(type(plan["assignments"]) is list and len(plan["assignments"])==len(ids),
         "SPLIT_CASE_COUNT")
    need(plan["dataset_sha256"]==digest(cases),"SPLIT_COHORT_SHA")
    rows={};groups={}
    for row in plan["assignments"]:
        need(type(row) is dict and set(row)=={"case_id","split","group_id"},
             "SPLIT_ASSIGNMENT_FIELDS")
        cid=row["case_id"];split=row["split"];gid=row["group_id"]
        need(cid in ids and cid not in rows,"UNKNOWN_OR_DUPLICATE_SPLIT_CASE")
        need(split in {"train","dev","holdout"},"UNKNOWN_SPLIT")
        need(type(gid) is str and re.fullmatch(r"[A-Za-z0-9_-]{3,100}",gid),
             "INVALID_LEAKAGE_GROUP")
        rows[cid]=row
        if gid in groups:need(groups[gid]==split,"GROUP_SPLIT_LEAKAGE")
        groups[gid]=split
    need(set(rows)==ids and {x["split"] for x in rows.values()}=={"train","dev","holdout"},
         "INCOMPLETE_SPLITS")
    # Reject overlapping windows for the same instrument when assigned to
    # different splits. This avoids the most obvious near-duplicate leakage.
    symbols={}
    for c in cases["cases"]:
        chart=charts[c["id"]];start,end=_window(chart)
        symbols.setdefault(chart["symbol"],[]).append((start,end,rows[c["id"]]["split"],c["id"]))
    for symbol,items in symbols.items():
        for i,a in enumerate(items):
            for b in items[i+1:]:
                if max(a[0],b[0])<min(a[1],b[1]):
                    need(a[2]==b[2],"OVERLAPPING_SAME_SYMBOL_WINDOWS_CROSS_SPLIT")
    # Future prospective H1 cohorts are permanently test-only, not training input.
    for cid,row in rows.items():
        if cid.startswith("E6-FWD-"):
            need(row["split"]=="holdout","FUTURE_H1_NOT_SEALED_HOLDOUT")
    return rows

def _validate_rule_map(data,cases):
    need(type(data) is dict and set(data)=={"schema","dataset_sha256","items"}
         and data["schema"]==RULEMAP_SCHEMA,"RULE_MAP_SCHEMA")
    need(data["dataset_sha256"]==digest(cases),"RULE_MAP_COHORT_SHA")
    ids={x["id"] for x in cases["cases"]}
    need(type(data["items"]) is list and len(data["items"])==len(ids),"RULE_MAP_COUNT")
    cards=registry()["rule_cards"];allowed={x["id"]:x["source"] for x in cards}
    out={}
    for row in data["items"]:
        need(type(row) is dict and set(row)=={"case_id","rule_ids"},"RULE_MAP_FIELDS")
        cid=row["case_id"]
        need(cid in ids and cid not in out,"RULE_MAP_DUPLICATE_OR_UNKNOWN")
        case=next(x for x in cases["cases"] if x["id"]==cid)
        rules=row["rule_ids"]
        need(type(rules) is list and bool(rules) and
             len(set(rules))==len(rules),"RULE_MAP_EMPTY_OR_DUPLICATE")
        need(all(r in allowed and allowed[r]==case["edition"] for r in rules),
             "RULE_ID_NOT_IN_CORRECT_BOOK")
        out[cid]=rules
    need(set(out)==ids,"RULE_MAP_INCOMPLETE")
    return out

def build(cases,manifest,review_a,review_b,trust,split_plan,rule_map,chart_root):
    frozen=validate_cases(cases,chart_root)
    need(frozen==manifest,"FROZEN_COHORT_MISMATCH")
    no_model_generated(cases);no_model_generated(split_plan);no_model_generated(rule_map)
    charts={}
    root=Path(chart_root).resolve(strict=True)
    for c in cases["cases"]:
        charts[c["id"]]=json.loads((root/c["chart_file"]).read_bytes())
    split=_validate_split(split_plan,cases,charts)
    rules=_validate_rule_map(rule_map,cases)
    by_id={c["id"]:(c["edition"],c["family"],c["pdf_pages"]) for c in cases["cases"]}
    ids=set(by_id)
    aid,a=_verify_review(review_a,ids,manifest,trust,by_id)
    bid,b=_verify_review(review_b,ids,manifest,trust,by_id)
    need(aid!=bid and trust[aid]["ed25519_public_key_hex"]!=
         trust[bid]["ed25519_public_key_hex"],"REVIEWERS_NOT_INDEPENDENT")
    train_dev=[];sealed=[];excluded=Counter()
    for cid in sorted(ids):
        la,lb=a[cid],b[cid]
        if not la["reviewable"] or not lb["reviewable"]:
            excluded["unreviewable"]+=1;continue
        if la["decision"]!=lb["decision"]:
            excluded["reviewer_disagreement"]+=1;continue
        common_pages=sorted(set(la["pages"])&set(lb["pages"]))
        if not common_pages:
            excluded["no_common_source_pages"]+=1;continue
        split_name=split[cid]["split"]
        ed,family,pages=by_id[cid]
        record={
          "id":cid,"edition":ed,"family":family,"source_pdf_sha256":PDF_SHA[ed],
          "source_pages":common_pages,"source_rule_ids":rules[cid],
          "chart_file":next(x["chart_file"] for x in cases["cases"] if x["id"]==cid),
          "chart_sha256":next(x["chart_sha256"] for x in cases["cases"] if x["id"]==cid),
          "cutoff_utc":next(x["cutoff_utc"] for x in cases["cases"] if x["id"]==cid),
          "question":next(x["question"] for x in cases["cases"] if x["id"]==cid),
          "gold_decision":la["decision"],
          "independent_rationales":[{"reviewer_id":aid,"text":la["rationale"]},
                                   {"reviewer_id":bid,"text":lb["rationale"]}],
          "two_signed_reviewers_agree":True,
          "split":split_name,
        }
        if split_name=="holdout":
            sealed.append({k:v for k,v in record.items() if k!="split"})
        else:
            train_dev.append(record)
    need(bool(train_dev),"NO_AGREED_TRAIN_DEV_EXAMPLES")
    need(bool(sealed),"NO_SEALED_HOLDOUT_EXAMPLES")
    counts={part:dict(Counter(x["edition"] for x in
        ([r for r in train_dev if r["split"]==part] if part in {"train","dev"} else sealed)))
        for part in ("train","dev","holdout")}
    need(all(counts[p].get(book,0)>0 for p in ("train","dev","holdout") for book in BOOKS),
         "SPLITS_NOT_REPRESENTED_IN_ALL_BOOKS")
    train_dev.sort(key=lambda x:x["id"]);sealed.sort(key=lambda x:x["id"])
    return {
      "training_candidate_schema":SCHEMA,
      "cohort_sha256":manifest["dataset_sha256"],
      "split_plan_sha256":digest(split_plan),
      "rule_map_sha256":digest(rule_map),
      "review_signature_sha256":{"review_a":digest(review_a),"review_b":digest(review_b)},
      "reviewer_ids":[aid,bid],"externally_vetted_status_claimed_by_trust_registry":True,
      "independent_human_agreed_examples":len(train_dev)+len(sealed),
      "excluded":{"unreviewable":excluded["unreviewable"],
        "reviewer_disagreement":excluded["reviewer_disagreement"],
        "no_common_source_pages":excluded["no_common_source_pages"]},
      "counts_by_split_and_book":counts,
      "candidate_train_dev":{"schema":"WILLIAMS_V7_TRAIN_DEV_CANDIDATES_PRIVATE_V1",
        "examples":train_dev},
      "sealed_holdout":{"schema":"WILLIAMS_V7_SEALED_HOLDOUT_PRIVATE_V1",
        "labels":sealed,"model_access_forbidden":True},
      "model_predictions_used_as_labels":False,
      "historical100_or_prospective_exam_reused_as_training":False,
      "training_authorized":False,"gpu_training_authorized":False,
      "model_promotion_allowed":False,"trading_allowed":False,
      "expert_competence_independently_audited_by_software":False,
      "external_expert_competence_and_identity_audit_still_required":True,
    }

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for f in ("cases","chart-root","manifest","review-a","review-b","trust","split-plan","rule-map","out-train-dev","out-holdout"):
        p.add_argument("--"+f,required=True)
    a=p.parse_args()
    result=build(_load(a.cases),_load(a.manifest),_load(a.review_a),_load(a.review_b),
        _load(a.trust),_load(a.split_plan),_load(a.rule_map),a.chart_root)
    _private_create(a.out_train_dev,result["candidate_train_dev"])
    try:_private_create(a.out_holdout,result["sealed_holdout"])
    except BaseException:
        Path(a.out_train_dev).unlink(missing_ok=True);raise
    summary={k:v for k,v in result.items() if k not in {"candidate_train_dev","sealed_holdout"}}
    print(json.dumps(summary,ensure_ascii=False,sort_keys=True,indent=2))
if __name__=="__main__":main()
