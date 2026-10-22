"""Opt-in public research, primary-document catalog + keyless Wikipedia search.

No exchange API, credentials, model-chosen URLs or redirects. No book uploads.
"""
from __future__ import annotations

from datetime import datetime, timezone
from html.parser import HTMLParser
import json
import re
from typing import Callable
from urllib.parse import urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

CATALOG = (
    ("Binance Futures documentation", "https://developers.binance.com/en/docs/", ("binance", "бинанс", "futures", "фьючерс", "api", "стоп", "ордер")),
    ("Binance USD-M market data", "https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/market-data", ("свеч", "kline", "истори", "market", "ohlcv")),
    ("llama-cpp-python documentation", "https://llama-cpp-python.readthedocs.io/en/stable/", ("gguf", "llama", "модел", "нейросет")),
    ("pypdf text extraction", "https://pypdf.readthedocs.io/en/latest/user/extract-text.html", ("pdf", "ocr", "скан", "извлеч", "книг")),
    ("Backtest overfitting research", "https://scholarworks.wmich.edu/math_pubs/42/", ("overfit", "переобуч", "backtest", "бэктест", "выборк")),
)
HOSTS = {urlsplit(item[1]).hostname for item in CATALOG} | {"ru.wikipedia.org"}
_SECRET = re.compile(r"(?i)(api[ _-]?key|secret|password|парол|секрет|sk-[a-z0-9]{12,}|[a-z0-9_-]{40,}|[\w.+-]+@[\w.-]+\.[a-z]{2,}|\b\d{9,}\b)")


def safe_query(question: str) -> str:
    if not isinstance(question, str) or not 3 <= len(question.strip()) <= 200:
        raise ValueError("Public research question must have 3..200 characters")
    # Inspect the entire question before normalization or truncation.
    if _SECRET.search(question) or any(ord(c) < 32 for c in question):
        raise ValueError("Query may contain private data")
    return question.strip()


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("Research redirects are disabled")


def _get(url: str) -> str:
    parsed = urlsplit(url)
    if (parsed.scheme != "https" or parsed.hostname not in HOSTS or parsed.port is not None
            or parsed.username or parsed.password or parsed.fragment):
        raise ValueError("Research URL outside allowlist")
    req = Request(url, headers={"User-Agent": "WilliamsAI-Research/0.7", "Accept": "text/html,application/json"})
    with build_opener(_NoRedirect()).open(req, timeout=8) as reply:
        data = reply.read(256 * 1024 + 1)
        if len(data) > 256 * 1024:
            raise ValueError("Research response too large")
        return data.decode("utf-8", errors="replace")


class _Text(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}:
            self.skip += 1

    def handle_endtag(self, tag):
        if tag in {"script", "style"} and self.skip:
            self.skip -= 1

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)


def _excerpt(html: str, query: str) -> str:
    parser = _Text()
    parser.feed(html)
    plain = re.sub(r"\s+", " ", " ".join(parser.parts)).strip()
    terms = re.findall(r"[^\W_]{3,}", query.casefold())
    offsets = [plain.casefold().find(t) for t in terms if t in plain.casefold()]
    start = max(0, min(offsets) - 80) if offsets else 0
    return plain[start:start + 1000]


def research(question: str, *, enabled: bool = False,
             fetch: Callable[[str], str] = _get) -> dict:
    if not enabled:
        return {"status": "WEB_DISABLED", "sources": [], "errors": [], "trade_action": None}
    query = safe_query(question)
    found, errors = [], []
    now = datetime.now(timezone.utc).isoformat()
    for title, url, keywords in CATALOG:
        if len(found) >= 3:
            break
        if not any(t in query.casefold() for t in keywords):
            continue
        try:
            text = _excerpt(fetch(url), query)
            if not text:
                raise ValueError("Empty public document")
            found.append({"title": title, "source": url, "excerpt": text,
                          "retrieved_at": now, "status": "PRIMARY_DOCUMENT_UNVERIFIED_INTERPRETATION"})
        except Exception as exc:
            errors.append({"source": url, "error_type": type(exc).__name__})
    try:
        url = "https://ru.wikipedia.org/w/api.php?" + urlencode({"action": "query", "list": "search", "srsearch": query, "format": "json", "srlimit": 3})
        results = json.loads(fetch(url)).get("query", {}).get("search", [])
        for item in results[:3]:
            page = item.get("pageid")
            if not isinstance(page, int) or page <= 0:
                continue
            found.append({"title": str(item.get("title", ""))[:150],
                          "source": "https://ru.wikipedia.org/?curid=" + str(page),
                          "excerpt": _excerpt(str(item.get("snippet", "")), query)[:500],
                          "retrieved_at": now, "status": "SECONDARY_SOURCE_UNVERIFIED"})
    except Exception as exc:
        errors.append({"source": "https://ru.wikipedia.org/", "error_type": type(exc).__name__})
    return {"status": "EVIDENCE_FOUND" if found else "NO_EVIDENCE", "sources": found,
            "errors": errors, "trade_action": None,
            "note": "Research is not verification. Web text never changes book rules or calls execution."}
