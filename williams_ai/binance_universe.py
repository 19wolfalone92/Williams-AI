"""All available Binance USD-M PERPETUAL research symbols, not a 15-coin whitelist.

Snapshot exchangeInfo with strict venue provenance. A current list cannot
recover delisted historical tokens: tests MUST flag survivorship bias.
No signed account data, model-supplied host, POST or trade execution.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
from typing import Any, Callable

from .binance_wave_history import VENUES, HistoryUnavailable, _read_market


@dataclass(frozen=True)
class Universe:
    venue: str
    observed_server_ms: int
    instruments: tuple[str, ...]
    raw_exchangeinfo_hash: str

    def public_report(self) -> dict[str, Any]:
        return {
            "mode": "PUBLIC_BINANCE_USDM_PERPETUAL_UNIVERSE_SNAPSHOT",
            "venue": self.venue,
            "observed_server_ms": self.observed_server_ms,
            "eligible_symbols_count": len(self.instruments),
            "eligible_symbols": list(self.instruments),
            "source_sha256": self.raw_exchangeinfo_hash,
            "historically_delisted_symbols_included": False,
            "historical_survivorship_bias_ruled_out": False,
            "auto_live_universe_substitution": False,
            "execution_allowed": False, "trade_action": None,
        }


def discover_universe(
    venue: str = "PUBLIC_USDM", *,
    fetch: Callable[..., Any] = _read_market,
    now_ms: int | None = None,
) -> Universe:
    if venue not in VENUES:
        raise HistoryUnavailable("UNKNOWN_FUTURES_PUBLIC_VENUE")
    t = fetch(venue, "/fapi/v1/time", None)
    actual = t.get("serverTime") if isinstance(t, dict) else None
    if not isinstance(actual, int) or isinstance(actual, bool) or actual <= 0:
        raise HistoryUnavailable("UNVERIFIED_EXCHANGE_UNIVERSE_CLOCK")
    if now_ms is None:
        import time
        now_ms = int(time.time() * 1000)
    if (not isinstance(now_ms, int) or isinstance(now_ms, bool)
            or abs(now_ms - actual) > 300_000):
        raise HistoryUnavailable("SOURCE_LOCAL_CLOCK_DIVERGENCE")
    raw = fetch(venue, "/fapi/v1/exchangeInfo", None)
    listing = raw.get("symbols") if isinstance(raw, dict) else None
    if not isinstance(listing, list) or not listing or len(listing) > 10_000:
        raise HistoryUnavailable("NO_VERIFIED_EXCHANGEINFO_UNIVERSE")
    eligible = set()
    for item in listing:
        if not isinstance(item, dict):
            raise HistoryUnavailable("MALFORMED_EXCHANGEINFO_SYMBOL")
        if not (item.get("contractType") == "PERPETUAL" and
                item.get("quoteAsset") == "USDT" and
                item.get("status") == "TRADING"):
            continue
        symbol = item.get("symbol")
        if (not isinstance(symbol, str) or not 5 <= len(symbol) <= 24
                or not symbol.endswith("USDT")
                or not all(c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
                           for c in symbol)):
            raise HistoryUnavailable("INVALID_TRADABLE_SYMBOL")
        if symbol in eligible:
            raise HistoryUnavailable("DUPLICATED_EXCHANGE_SYMBOL")
        eligible.add(symbol)
    if not eligible:
        raise HistoryUnavailable("NO_ACTIVE_USDT_PERPETUAL_CONTRACTS")
    source = hashlib.sha256(json.dumps(
        raw, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return Universe(venue, actual, tuple(sorted(eligible)), source)


def write_universe_snapshot(public: Universe, path: Path) -> Path:
    """Immutable dated private snapshot; future listings never rewrite prior eras."""
    if public.venue not in VENUES or not public.instruments:
        raise ValueError("Invalid universe")
    dest = Path(path).expanduser()
    if dest.suffix != ".json" or dest.is_symlink():
        raise ValueError("Private JSON snapshot path expected")
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(public.public_report(), sort_keys=True, indent=2)
    fd = os.open(dest, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(payload + "\n")
        f.flush()
        os.fsync(f.fileno())
    return dest


def read_universe_snapshot(path: Path, *, venue: str) -> Universe:
    """Never let model text or an arbitrary JSON file add trading symbols.

    Only earlier immutable PUBLIC exchangeInfo universe snapshot form is
    accepted. Validation catches duplicate symbols and venue switching.
    """
    dest=Path(path).expanduser()
    if dest.is_symlink() or not dest.is_file():
        raise ValueError("Immutable market universe snapshot file required")
    data=json.loads(dest.read_text(encoding="utf8"))
    if not isinstance(data,dict) or data.get("mode")!=(
            "PUBLIC_BINANCE_USDM_PERPETUAL_UNIVERSE_SNAPSHOT") or (
            data.get("venue")!=venue or venue not in VENUES):
        raise HistoryUnavailable("UNTRUSTED_OR_WRONG_VENUE_UNIVERSE")
    names=data.get("eligible_symbols")
    if (not isinstance(names,list) or not 1<=len(names)<=10_000
            or len(names)!=data.get("eligible_symbols_count")
            or names!=sorted(set(names))):
        raise HistoryUnavailable("UNVERIFIED_OR_DUPLICATED_UNIVERSE")
    for name in names:
        if (not isinstance(name,str) or not name.endswith("USDT")
                or not 5<=len(name)<=24 or not all(
                    c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789" for c in name)):
            raise HistoryUnavailable("UNVERIFIED_FUTURES_SYMBOL")
    hash_value=data.get("source_sha256")
    if (not isinstance(hash_value,str) or len(hash_value)!=64
            or any(ch not in "0123456789abcdef" for ch in hash_value)):
        raise HistoryUnavailable("INVALID_PUBLIC_SNAPSHOT_FINGERPRINT")
    ms=data.get("observed_server_ms")
    if not isinstance(ms,int) or isinstance(ms,bool) or ms<=0:
        raise HistoryUnavailable("INVALID_PUBLIC_SNAPSHOT_CLOCK")
    if data.get("historical_survivorship_bias_ruled_out") is not False:
        raise HistoryUnavailable("UNSUPPORTED_NO_SURVIVORSHIP_BIAS_ASSERTION")
    return Universe(venue,ms,tuple(names),hash_value)
