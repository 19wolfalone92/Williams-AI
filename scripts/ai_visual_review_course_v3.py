"""Local-only Qwen3-VL visual analysis or queue creation for scanned Course.

No remote inference and no browser uploads. Pages and full book remain outside
the GitHub repository. Visual outputs are proposals, NEVER expert training gold.

python scripts/ai_visual_review_course_v3.py --book-index /private/four_books.sqlite3 \
  --queue-only --out /private/course_visual_queue.jsonl

python scripts/ai_visual_review_course_v3.py --book-index /private/four_books.sqlite3 \
  --model-dir /private/Qwen3-VL-4B-Instruct \
  --pages 47,56,71 --out /private/visual_model_proposals.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from williams_ai.visual_vision_review_v3 import (
    LocalQwenVLBackend,prepare_visual_jobs,save_private_visual_proposals,
)
from williams_ai.book_index import ROOT


def new_private_path(location):
    p=Path(location).expanduser().resolve()
    if (p==ROOT or ROOT in p.parents or p.exists()
            or p.is_symlink() or not p.parent.is_dir()):
        raise ValueError("NEW_PRIVATE_REPORT_OUTSIDE_GIT_REQUIRED")
    return p


def main():
    p=argparse.ArgumentParser(description="Private scanned Course visual review; no model self-certification")
    p.add_argument("--book-index",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--queue-only",action="store_true")
    p.add_argument("--model-dir",type=Path)
    p.add_argument("--pages",help="Comma-separated 1..122; empty = all pages")
    args=p.parse_args()
    if args.pages:
        pages=[int(x) for x in args.pages.split(",")]
    else:
        pages=None
    if args.queue_only:
        if args.model_dir:
            p.error("Use --queue-only OR --model-dir, not both")
        jobs=prepare_visual_jobs(args.book_index,pages=pages)
        out=new_private_path(args.out)
        fd=os.open(out,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(fd,"w",encoding="utf-8") as stream:
            for job in jobs:
                stream.write(json.dumps(job,ensure_ascii=False)+"\n")
        result={"review_queue_pages":len(jobs),"vision_inference_completed":0,
                "human_gold_approval":False}
    else:
        if not args.model_dir:
            p.error("--model-dir is required unless --queue-only")
        backend=LocalQwenVLBackend(args.model_dir)
        result=save_private_visual_proposals(
            args.book_index,backend,args.out,pages=pages)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
