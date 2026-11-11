"""Merge the user's locally OCR-indexed Course into a COPY of their private 4-book index.

Example (never inside public Git checkout):
python scripts/ai_merge_private_course.py \
  --existing /private/williams_books.db \
  --course /private/Williams_Course/private_course_ai.sqlite3 \
  --output /private/Williams_Course/williams_all_four_private.sqlite3

The original database and private page images are never modified.
"""
import argparse,json
from pathlib import Path
from williams_ai.course_visual_private import combine_with_existing_private_books

def main():
    p=argparse.ArgumentParser(description="Local personal Course OCR index integration")
    p.add_argument("--existing",type=Path,required=True)
    p.add_argument("--course",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    print(json.dumps(combine_with_existing_private_books(
        args.existing,args.course,destination=args.output),
        ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()
