"""CLI for PRIVATE book ingestion. Never commit original PDF or indexed DB."""
from __future__ import annotations
import argparse
import json
from .book_index import index_pdf, index_report, search_pages
from .book_rules import coverage, lookup_rules

def main():
    p = argparse.ArgumentParser(description="Williams AI books, offline private knowledge")
    p.add_argument("--db", required=True, help="Absolute DB path outside the public checkout")
    sub = p.add_subparsers(dest="action", required=True)
    ingest = sub.add_parser("ingest")
    ingest.add_argument("edition", choices=["TC1","NTD","TC2","COURSE"])
    ingest.add_argument("pdf")
    ingest.add_argument("--ocr-dir", default=None, help="Optional human-reviewed OCR sidecar txt files")
    search = sub.add_parser("search")
    search.add_argument("question")
    search.add_argument("--edition", choices=["TC1","NTD","TC2","COURSE"], default=None)
    sub.add_parser("report")
    opts = p.parse_args()
    if opts.action == "ingest":
        result = index_pdf(opts.db, opts.edition, opts.pdf, optional_ocr_dir=opts.ocr_dir)
    elif opts.action == "search":
        result = {"page_matches": search_pages(opts.db, opts.question, edition=opts.edition),
                  "rule_cards": lookup_rules(opts.question, edition=opts.edition)}
    else:
        result = {"books_index": index_report(opts.db), "rule_cards": coverage()}
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
