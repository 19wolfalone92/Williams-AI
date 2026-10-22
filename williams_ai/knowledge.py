"""Small, local-only text search for user-supplied trading notes.

The bundled notes are project specifications, not verified verbatim book quotes.
No cloud embeddings, scraping or network access.
"""
from __future__ import annotations

from pathlib import Path
import re
from typing import Any

_STOP = {"это", "как", "где", "для", "если", "что", "почему", "какой", "какая",
         "which", "where", "what", "about", "the", "and", "our", "with"}


def _words(text: str) -> set[str]:
    return {word for word in re.findall(r"[^\W_]+", text.casefold(), re.UNICODE)
            if len(word) >= 2 and word not in _STOP}


def retrieve_notes(question: str, folder: str | Path, *, limit: int = 4) -> list[dict[str, Any]]:
    """Return relevant snippets from locally supplied .md/.txt (never follow symlinks)."""
    path = Path(folder).expanduser()
    if not path.is_dir():
        raise FileNotFoundError(f"Knowledge directory not found: {path}")
    if not isinstance(question, str) or len(question) > 1200:
        raise ValueError("Invalid question")
    if not (1 <= limit <= 10):
        raise ValueError("limit must be 1..10")

    terms = _words(question)
    if not terms:
        return []
    results = []
    files = sorted([*path.glob("*.md"), *path.glob("*.txt")])
    for source in files[:100]:
        if source.is_symlink() or not source.is_file() or source.stat().st_size > 500_000:
            continue
        lines = source.read_text(encoding="utf-8").splitlines()
        for index, line in enumerate(lines):
            score = len(terms & _words(line))
            if not score:
                continue
            snippet = " ".join(lines[max(0, index - 1): index + 2]).strip()[:1000]
            results.append({
                "source": f"{source.name}:{index + 1}",
                "excerpt": snippet,
                "_score": score,
            })
    results.sort(key=lambda entry: (-entry["_score"], entry["source"]))
    return [{"source": item["source"], "excerpt": item["excerpt"]}
            for item in results[:limit]]
