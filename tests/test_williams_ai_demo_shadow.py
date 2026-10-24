"""Fresh market-state observer cannot turn Qwen model words into DEMO order."""
import copy
import unittest
from datetime import datetime,timezone

from williams_ai.demo_shadow import capture_demo_h1,analyze_demo_shadow,HOUR_MS
from williams_ai.demo_pilot_readiness import DemoSafetyError

CLOCK=2_000_000_000_000


def test_market_fixture(*,server=CLOCK,noise=None,break_index=None):
    cur=(CLOCK//HOUR_MS)*HOUR_MS
    candles=[]
    for i in range(500):
        op=cur-(499-i)*HOUR_MS
        price=100+ i*.01
        candles.append([op,str(price),str(price+.5),str(price-.5),
                        str(price+.1),"100",op+HOUR_MS-1, "0",0,"0","0","0"])
    if noise:
        candles[-1][4]=str(noise)
    if break_index is not None:
        candles[break_index][0]+=1_800_000
    calls=[]
    def fetch(path,query):
        calls.append((path,query))
        if path=="/fapi/v1/time":
            return {"serverTime":server}
        if path=="/fapi/v1/exchangeInfo":
            return {"symbols":[{"symbol":"ETHUSDT","status":"TRADING",
                "filters":[{"filterType":"PRICE_FILTER","tickSize":"0.01"},
                           {"filterType":"MARKET_LOT_SIZE","stepSize":"0.001",
                            "minQty":"0.001","maxQty":"500"}]}]}
        if path=="/fapi/v1/premiumIndex":
            return {"symbol":"ETHUSDT","markPrice":"105","time":server}
        if path=="/fapi/v1/klines":
            self_ok=(query=={"symbol":"ETHUSDT","interval":"1h","limit":"500"})
            if not self_ok:
                raise AssertionError("Only approved DEMO H1 source allowed")
            return candles
        raise AssertionError("Unauthorized endpoint in public shadow")
    return fetch,candles,calls


class NoTradingModel:
    def __init__(self):
        self.views=[]
    def propose_chart_reasoning(self,obs,edition="TC2",**kwargs):
        self.views.append({"as_of":obs["as_of"],"symbol":obs["symbol"],
                           "edition":edition,"bar_count":obs["bars_used"]})
        return {
            "symbol":obs["symbol"],"as_of":obs["as_of"],"edition":"TC2",
            "setup":"NO_SETUP","side":"NONE","entry_state":"NO_SETUP",
            "rule_ids":[],"exit_basis":"NONE","wave_degree_verified":False,
            "exchange_fill_confirmed":False,"exchange_sl_confirmed":False,
            "trade_action":None,"execution_allowed":False,
            "reason":"No current confirmed trade signal, research-only.",
        }


class DemoShadowReadOnlyTests(unittest.TestCase):
    def test_demo_source_is_not_spot_and_unfinished_candle_not_sent(self):
        fetch,candles,calls=test_market_fixture(noise=999_999_999)
        data=capture_demo_h1(fetch=fetch,now_ms=CLOCK)
        self.assertEqual(data["venue"],"BINANCE_FUTURES_DEMO")
        self.assertEqual(data["symbol"],"ETHUSDT")
        self.assertEqual(len(data["bars"]),499)
        self.assertFalse(data["execution_allowed"])
        self.assertFalse(any(float(x["close"])==999_999_999 for x in data["bars"]))
        self.assertEqual(len(calls),4)
        self.assertTrue(all(path.startswith("/fapi/v1/") for path,query in calls))
        self.assertTrue(all(x["closed"] is True for x in data["bars"]))

    def test_one_actual_chart_pipeline_optional_model_never_sends_orders(self):
        fetch,rows,calls=test_market_fixture()
        model=NoTradingModel()
        report=analyze_demo_shadow(fetch=fetch,now_ms=CLOCK,model=model)
        self.assertEqual(report["mode"],"BINANCE_FUTURES_DEMO_WILLIAMS_READ_ONLY_SHADOW")
        self.assertEqual(report["closed_h1_bars"],499)
        self.assertGreaterEqual(report["closed_h4_bars"],100)
        self.assertLess(report["closed_d1_bars"],45)
        self.assertEqual(report["independent_gguf_book_claim_audit"]["verdict"],
                         "PROVISIONAL_REASONING_ONLY")
        self.assertEqual(report["independent_gguf_book_claim_audit"]["setup"],
                         "NO_SETUP")
        self.assertEqual(len(model.views),1)
        self.assertEqual(model.views[0]["bar_count"],499)
        self.assertEqual(model.views[0]["as_of"],report["as_of"])
        self.assertTrue(report["manual_pilot_not_authorized_by_shadow"])
        self.assertFalse(report["execution_allowed"])
        self.assertIsNone(report["trade_action"])
        self.assertEqual(len(calls),4)
        self.assertFalse(any(p=="/fapi/v1/order" for p,_ in calls))

    def test_nonhourly_and_gap_rejected(self):
        fetch,candles,calls=test_market_fixture(break_index=200)
        with self.assertRaisesRegex(DemoSafetyError,"MISALIGNED"):
            capture_demo_h1(fetch=fetch,now_ms=CLOCK)

    def test_stale_demo_h1_readiness_fails_closed(self):
        fetch,_,_=test_market_fixture(server=CLOCK+5*HOUR_MS)
        with self.assertRaisesRegex(DemoSafetyError,"STALE"):
            capture_demo_h1(fetch=fetch,now_ms=CLOCK+5*HOUR_MS)

    def test_extra_market_get_not_allowed_for_fake_ai(self):
        fetch,_,calls=test_market_fixture()
        report=analyze_demo_shadow(fetch=fetch,now_ms=CLOCK)
        self.assertIsNone(report["independent_gguf_book_claim_audit"])
        self.assertEqual(len(calls),4)
        self.assertEqual(report["real_trading_authorized"],False)


if __name__=="__main__":
    unittest.main()
