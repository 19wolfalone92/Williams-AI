"""Make deterministic blind, private, UNLABELLED reviewer packets from v7 book probes.

A/B are packet slots, not people and do not establish expert independence.
No chart gold, model predictions, training or trading in this module.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path

from .v7_reasoning_review_contract import validate as validate_probes

SCHEMA = "WILLIAMS_V7_SOURCE_ONLY_BLIND_REVIEW_PACKET_V1"
ISSUE_SCHEMA = "WILLIAMS_V7_SOURCE_CONFLICTS_NOT_EXPERT_ADJUDICATED_V1"

def canon(value):
    return json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False,
                      separators=(",",":")).encode("utf-8")

def digest(value):
    return sha256(canon(value)).hexdigest()

def require(ok,reason):
    if not ok:
        raise ValueError("V7_BLIND_REVIEW_" + reason)

def build_packet(source, slot):
    validate_probes(source)
    require(slot in ("A","B"),"INVALID_REVIEWER_SLOT")
    source_hash=digest(source)
    rows=[]
    for pair in source["pairs"]:
        for variant in ("a","b"):
            row=pair["variant_"+variant]
            rows.append({
                "review_item_id":pair["pair_id"]+"-"+variant.upper(),
                "source_pair_id":pair["pair_id"],
                "edition":pair["edition"],
                "source_pdf_sha256":pair["source_pdf_sha256"],
                "source_pages":pair["source_pages"],
                "source_rule_ids":pair["rule_ids"],
                "source_review_ref":pair["source_review_ref"],
                "question":pair["question"],
                "scenario":row["scenario"],
                "unavailable_facts":row["unavailable_facts"],
            })
    rows.sort(key=lambda item:sha256((slot+":"+source_hash+":"+
                   item["review_item_id"]).encode("ascii")).hexdigest())
    return {
      "schema":SCHEMA,
      "reviewer_slot_not_identity":slot,
      "source_candidate_sha256":source_hash,
      "expert_identity_vetted":False,
      "independent_human_reviews_received":0,
      "ground_truth_labels_included":False,
      "model_predictions_included":False,
      "future_holdout_or_training_data":False,
      "sft_approved":False,
      "trading_authorized":False,
      "items":rows,
    }

def validate_packet(packet,source):
    require(type(packet) is dict,"PACKET_TYPE")
    slot=packet.get("reviewer_slot_not_identity")
    require(slot in ("A","B"),"INVALID_REVIEWER_SLOT")
    require(packet==build_packet(source,slot),"PACKET_TAMPER_OR_LEAKAGE")
    return {"review_items":30,"slot":slot,"gold_labels":0,
            "expert_reviews":0,"training_authorized":False}

def validate_issues(data,source):
    validate_probes(source)
    require(type(data) is dict and
            set(data)=={"schema","expert_approved","sft_ready",
                        "labels_generated","independent_experts","issues"} and
            data.get("schema")==ISSUE_SCHEMA,"ISSUE_SCHEMA")
    require(data["expert_approved"] is False and data["sft_ready"] is False and
            data["labels_generated"] is False and
            type(data["independent_experts"]) is int and
            data["independent_experts"]==0,"FALSE_SOURCE_APPROVAL")
    issues=data["issues"]
    require(type(issues) is list and len(issues)==4,"ISSUE_COUNT")
    ids={p["pair_id"] for p in source["pairs"]}
    seen=set()
    for issue in issues:
        require(type(issue) is dict and set(issue)=={
          "id","severity","source_pdf_sha256","source_edition",
          "source_pages","conflicting_source_evidence","affected_probe_ids",
          "unresolved_question","review_required","suggested_resolution_status",
          "training_approved"},"ISSUE_FIELDS")
        k=issue["id"]
        require(type(k) is str and k not in seen,"ISSUE_ID")
        seen.add(k)
        require(issue["severity"] in ("BLOCKING","INTERPRETATION_REVIEW"),
                "SEVERITY")
        require(issue["source_edition"] in ("TC1","NTD","TC2_EN","COURSE"),
                "SOURCE_EDITION")
        require(type(issue["source_pdf_sha256"]) is str and
                len(issue["source_pdf_sha256"])==64 and
                all(c in "0123456789abcdef" for c in issue["source_pdf_sha256"]),
                "SOURCE_SHA")
        require(type(issue["source_pages"]) is list and
                bool(issue["source_pages"]) and
                all(type(p) is int and p>0 for p in issue["source_pages"]),
                "SOURCE_PAGES")
        require(type(issue["conflicting_source_evidence"]) is list and
                len(issue["conflicting_source_evidence"])>=2 and
                all(type(s) is str and len(s.strip())>=12
                    for s in issue["conflicting_source_evidence"]),
                "EVIDENCE")
        require(type(issue["affected_probe_ids"]) is list and
                bool(issue["affected_probe_ids"]) and
                set(issue["affected_probe_ids"])<=ids,"PROBE_ID")
        require(type(issue["unresolved_question"]) is str and
                len(issue["unresolved_question"])>15,
                "QUESTION")
        require(issue["review_required"] is True and
                issue["training_approved"] is False and
                issue["suggested_resolution_status"]=="OPEN_NOT_GOLD",
                "PREMATURE_ADJUDICATION")
    return {"open_source_issues":len(seen),"independent_experts":0,
            "sft_ready":False,"training_approved":False}

def create_private(path,obj):
    dest=Path(path)
    require(dest.parent.is_dir(),"OUTPUT_DIRECTORY_NOT_FOUND")
    fd=os.open(dest,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    try:
        with os.fdopen(fd,"wb") as f:
            f.write(json.dumps(obj,ensure_ascii=False,sort_keys=True,
                     indent=2).encode("utf-8")+b"\\n")
            f.flush();os.fsync(f.fileno())
    except BaseException:
        dest.unlink(missing_ok=True)
        raise
    require((dest.stat().st_mode&0o777)==0o600,
            "PRIVATE_OUTPUT_PERMISSIONS")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",type=Path,required=True)
    p.add_argument("--slot",choices=("A","B"),required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    data=json.loads(a.input.read_text(encoding="utf-8"))
    obj=build_packet(data,a.slot)
    validate_packet(obj,data)
    create_private(a.out,obj)
    print(json.dumps({"packet_written":True,"slot":a.slot,
                      "items":30,"expert_reviews":0,
                      "sft_approved":False},sort_keys=True))

if __name__=="__main__":
    main()
