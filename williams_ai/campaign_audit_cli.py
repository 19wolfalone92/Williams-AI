"""Verify a *supplied* TC2 exchange-event journal without making orders.

python -m williams_ai.campaign_audit_cli examples/ai_campaign_journal_example.json

The journal is untrusted; independent Binance execution/reconciliation is
required before concluding that a fill, cancellation or stop actually occurred.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .campaign_audit import audit_campaign


def main() -> None:
    parser = argparse.ArgumentParser(description="Williams AI: read-only campaign lifecycle audit")
    parser.add_argument("journal", type=Path, help="JSON journal without API credentials")
    args = parser.parse_args()
    raw = json.loads(args.journal.read_text(encoding="utf-8"))
    result = audit_campaign(raw)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
