"""Official public H1 loader: anti-leak, anti-venue-mixing, strict as-of."""
import copy
import json
import unittest
from datetime import datetime, timezone

from williams_ai.public_forward_capture import (
    capture_latest_closed_h1, _public_get, HOUR_MS,
)

OPEN=int(datetime(2026,10,1,tzinfo=timezone.utc).timestamp()*1000)
NUMBER=270
LAST_OPEN=OPEN+(NUMBER-1)*HOUR_MS
SERVER=LAST_OPEN+7*60_000


def make_fetch(*,price="100",rows=None,clock=SERVER,venue="FUTURES"):
    entries=[]
    for i in range(NUMBER):
        start=OPEN+i*HOUR_MS
        px=100+i/100.0
        entries.append([
            start, str(px),str(px+.5),str(px-.5),str(px),str(i+100),
            start+HOUR_MS-1, "0",100,"0","0","0",
        ])
    if rows is not None:
        entries=rows
    urls=[]
    def fetch(url):
        urls.append(url)
        if url.endswith("/time"):
            return {"serverTime":clock}
        if "/exchangeInfo" in url:
            return {"symbols":[{
                "symbol":"BTCUSDT","status":"TRADING",
                "filters":[{"filterType":"PRICE_FILTER","tickSize":"0.01"}]
            }]}
        if "/klines?" in url:
            return entries
        raise AssertionError("Unknown public endpoint")
    return fetch,entries,urls


class BinancePublicForwardCapture(unittest.TestCase):
    def test_futures_new_unfinished_hour_hidden_from_ai(self):
        fetch,original,urls=make_fetch()
        item=capture_latest_closed_h1(
            "BTCUSDT",venue="FUTURES",fetch_json=fetch,clock_ms=SERVER)
        self.assertEqual(item["source"]["venue"],"FUTURES")
        self.assertEqual(item["source"]["fully_closed_h1"],NUMBER-1)
        self.assertTrue(item["source"]["no_future_bar_in_market"])
        self.assertEqual(len(item["market"]["bars"]),NUMBER-1)
        self.assertNotEqual(item["market"]["bars"][-1]["close_time"],
                            datetime.fromtimestamp(original[-1][6]/1000,
                                                   timezone.utc).isoformat())
        self.assertEqual(item["market"]["bars"][-1]["close_time"],
                         datetime.fromtimestamp(original[-2][6]/1000,
                                                timezone.utc).isoformat())
        self.assertTrue(all("fapi.binance.com/fapi/v1" in u for u in urls))
        self.assertFalse(item["source"]["execution_allowed"])
        self.assertIsNone(item["source"]["trade_action"])

    def test_spot_is_not_silently_labeled_futures(self):
        fetch,_,urls=make_fetch()
        item=capture_latest_closed_h1(
            "BTCUSDT",venue="SPOT",fetch_json=fetch,clock_ms=SERVER)
        self.assertEqual(item["source"]["venue"],"SPOT")
        self.assertIn("Spot and futures",item["source"]["limitations"])
        self.assertTrue(all("data-api.binance.vision/api/v3" in u for u in urls))

    def test_missing_h1_interval_is_strictly_rejected(self):
        fetch,rows,_=make_fetch()
        changed=copy.deepcopy(rows)
        changed[210][0]+=HOUR_MS
        changed[210][6]+=HOUR_MS
        corrupt,_,_=make_fetch(rows=changed)
        with self.assertRaisesRegex(ValueError,"hour|Missing"):
            capture_latest_closed_h1(
                "BTCUSDT",venue="FUTURES",fetch_json=corrupt,clock_ms=SERVER)

    def test_unfinalized_candles_are_never_treated_as_closed(self):
        fetch,rows,_=make_fetch()
        for end in rows[:-1]:
            self.assertLess(end[6],SERVER-5000)
        item=capture_latest_closed_h1(
            "BTCUSDT",venue="FUTURES",fetch_json=fetch,clock_ms=SERVER)
        self.assertFalse(any(x["close_time"]==datetime.fromtimestamp(
            rows[-1][6]/1000,timezone.utc).isoformat()
            for x in item["market"]["bars"]))

    def test_stale_feed_is_not_called_new_market(self):
        fetch,_,_=make_fetch(clock=SERVER+5*HOUR_MS)
        with self.assertRaisesRegex(ValueError,"NOT_FRESH"):
            capture_latest_closed_h1(
                "BTCUSDT",venue="SPOT",fetch_json=fetch,
                clock_ms=SERVER+5*HOUR_MS)

    def test_exchange_timestamp_drift_rejected(self):
        fetch,_,_=make_fetch(clock=SERVER-60_000*6)
        with self.assertRaisesRegex(ValueError,"deviates"):
            capture_latest_closed_h1(
                "BTCUSDT",venue="SPOT",fetch_json=fetch,clock_ms=SERVER)

    def test_unapproved_symbol_and_host_rejected(self):
        fetch,_,_=make_fetch()
        for venue in ("CROSS_FALLBACK","UNKNOWN"):
            with self.assertRaises(ValueError):
                capture_latest_closed_h1(
                    "BTCUSDT",venue=venue,fetch_json=fetch,clock_ms=SERVER)
        with self.assertRaises(ValueError):
            capture_latest_closed_h1(
                "BADUSDT",venue="FUTURES",fetch_json=fetch,clock_ms=SERVER)
        with self.assertRaises(ValueError):
            _public_get("https://evil.example/api/v3/klines?symbol=BTCUSDT")

    def test_exchange_tick_missing_and_wrong_status_fails(self):
        fetch,_,_=make_fetch()
        def malformed(url):
            val=fetch(url)
            if "/exchangeInfo" in url:
                val["symbols"][0]["status"]="HALT"
            return val
        with self.assertRaisesRegex(ValueError,"market status"):
            capture_latest_closed_h1(
                "BTCUSDT",venue="FUTURES",fetch_json=malformed,clock_ms=SERVER)

    def test_h1_geometry_and_open_timestamp_checked(self):
        fetch,rows,_=make_fetch()
        altered=copy.deepcopy(rows)
        altered[180][1]="999.9"
        bad,_,_=make_fetch(rows=altered)
        with self.assertRaisesRegex(ValueError,"OHLC geometry"):
            capture_latest_closed_h1(
                "BTCUSDT",venue="FUTURES",fetch_json=bad,clock_ms=SERVER)
        altered=copy.deepcopy(rows)
        altered[10][0]+=30*60_000
        bad,_,_=make_fetch(rows=altered)
        with self.assertRaisesRegex(ValueError,"Misaligned"):
            capture_latest_closed_h1(
                "BTCUSDT",venue="FUTURES",fetch_json=bad,clock_ms=SERVER)


if __name__=="__main__":
    unittest.main()
