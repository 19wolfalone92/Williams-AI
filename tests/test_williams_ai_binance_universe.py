"""Current Binance perpetual universe is not a survivorship-free past."""
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest

from williams_ai.binance_universe import discover_universe,write_universe_snapshot,read_universe_snapshot
from williams_ai.binance_wave_history import HistoryUnavailable

NOW=1_800_000_000_000

def market(venue,path,params):
    if venue!="PUBLIC_USDM": raise AssertionError("venue switched")
    if path=="/fapi/v1/time":return {"serverTime":NOW}
    if path=="/fapi/v1/exchangeInfo":
        return {"symbols":[
            {"symbol":"ETHUSDT","contractType":"PERPETUAL","quoteAsset":"USDT","status":"TRADING"},
            {"symbol":"BTCUSDT","contractType":"PERPETUAL","quoteAsset":"USDT","status":"TRADING"},
            {"symbol":"SOLUSDT","contractType":"PERPETUAL","quoteAsset":"USDT","status":"TRADING"},
            {"symbol":"SOLUSDC","contractType":"PERPETUAL","quoteAsset":"USDC","status":"TRADING"},
            {"symbol":"ADAUSDT","contractType":"CURRENT_QUARTER","quoteAsset":"USDT","status":"TRADING"},
            {"symbol":"XRPUSDT","contractType":"PERPETUAL","quoteAsset":"USDT","status":"BREAK"},
        ]}
    raise AssertionError("No signed routes allowed")

class UniverseProofTests(unittest.TestCase):
    def test_all_exchange_pairs_without_hardcoded_whitelist(self):
        result=discover_universe(fetch=market,now_ms=NOW)
        self.assertEqual(result.instruments,("BTCUSDT","ETHUSDT","SOLUSDT"))
        info=result.public_report()
        self.assertFalse(info["historical_survivorship_bias_ruled_out"])
        self.assertFalse(info["historically_delisted_symbols_included"])
        self.assertFalse(info["execution_allowed"])
        with TemporaryDirectory() as d:
            file=Path(d)/"universe_2026.json"
            write_universe_snapshot(result,file)
            self.assertEqual(json.loads(file.read_text())["eligible_symbols_count"],3)
            with self.assertRaises(FileExistsError):
                write_universe_snapshot(result,file)
            reopened=read_universe_snapshot(file,venue="PUBLIC_USDM")
            self.assertEqual(reopened.instruments,result.instruments)
            with self.assertRaises(HistoryUnavailable):
                read_universe_snapshot(file,venue="DEMO_USDM")
            invalid=json.loads(file.read_text())
            invalid["eligible_symbols"]=["SOLUSDT","BTCUSDT"]
            other=Path(d)/"fake.json"
            other.write_text(json.dumps(invalid))
            with self.assertRaises(HistoryUnavailable):
                read_universe_snapshot(other,venue="PUBLIC_USDM")

    def test_mismatched_clock_duplicate_symbol_and_http451_rejected(self):
        with self.assertRaisesRegex(HistoryUnavailable,"CLOCK"):
            discover_universe(fetch=market,now_ms=NOW-500_000)
        def dup(venue,path,params):
            a=market(venue,path,params)
            if path.endswith("exchangeInfo"):
                a["symbols"].append(dict(a["symbols"][0]))
            return a
        with self.assertRaisesRegex(HistoryUnavailable,"DUPLICATED"):
            discover_universe(fetch=dup,now_ms=NOW)
        def denied(venue,path,params):
            raise HistoryUnavailable("PUBLIC_USDM_PUBLIC_HTTP_451")
        with self.assertRaisesRegex(HistoryUnavailable,"451"):
            discover_universe(fetch=denied,now_ms=NOW)

if __name__=="__main__":
    unittest.main()
