"""Resumable local OCR for licensed scanned sources; no cloud and no shell.

Requires Poppler + Tesseract. OCR remains unreviewed and cannot certify charts.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

from .book_index import ROOT, _known_book, verify_original


def ocr_book(edition: str, pdf: str | Path, output: str | Path, *,
             start_page: int = 1, limit: int = 8, language: str = "eng",
             workers: int = 2) -> dict:
    verified = verify_original(edition, pdf)
    total = _known_book(edition)["pages"]
    if (not 1 <= start_page <= total or not 1 <= limit <= total
            or not 1 <= workers <= 4 or not re.fullmatch(r"[a-z]{3}(\+[a-z]{3})*", language)):
        raise ValueError("Invalid OCR budget, page or language")
    dest = Path(output).expanduser().resolve()
    if dest == ROOT or ROOT in dest.parents:
        raise ValueError("OCR output must remain outside the public repository")
    for program in ("pdftoppm", "tesseract"):
        if not shutil.which(program):
            raise RuntimeError("Install local OCR dependency: " + program)
    dest.mkdir(parents=True, exist_ok=True, mode=0o700)
    marker = dest / "source.sha256"
    signature = verified["verified_sha256"] + ":" + language + ":2200px-v1"
    if marker.exists() and marker.read_text() != signature:
        raise ValueError("OCR directory belongs to another source or configuration")
    if not marker.exists():
        # No reuse of arbitrary pre-existing OCR without its source passport.
        if list(dest.glob("page_*.txt")):
            raise ValueError("Existing OCR lacks a source passport")
        marker.write_text(signature)

    def one(page: int) -> tuple[int, str]:
        target = dest / f"page_{page:04d}.txt"
        if target.is_file() and not target.is_symlink():
            return page, "CACHED"
        try:
            with tempfile.TemporaryDirectory(prefix="williams-ocr-") as temp:
                prefix = str(Path(temp) / "page")
                subprocess.run(["pdftoppm", "-f", str(page), "-l", str(page),
                    "-singlefile", "-scale-to", "2200", "-gray", "-png",
                    str(verified["pdf"]), prefix], check=True, capture_output=True, timeout=60)
                result = subprocess.run(["tesseract", prefix + ".png", "stdout", "-l", language,
                    "--psm", "3"], check=True, capture_output=True, timeout=60)
                text = result.stdout.decode("utf-8", errors="replace")
                if len(text) > 100000:
                    raise ValueError("OCR page too large")
                fd, temporary = tempfile.mkstemp(prefix=".ocr-", dir=dest)
                try:
                    with os.fdopen(fd, "w", encoding="utf-8") as output_file:
                        output_file.write(text)
                        output_file.flush()
                        os.fsync(output_file.fileno())
                    os.replace(temporary, target)
                finally:
                    if os.path.exists(temporary):
                        os.unlink(temporary)
                return page, "EXTRACTED_UNREVIEWED" if text.strip() else "EMPTY_SCAN"
        except (subprocess.SubprocessError, OSError, ValueError) as exc:
            return page, "ERROR:" + type(exc).__name__

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(one, range(start_page, min(total + 1, start_page + limit))))
    return {"edition": edition, "pages": [{"page": p, "status": s} for p, s in results],
            "human_reviewed": False, "output": str(dest), "trade_action": None}
