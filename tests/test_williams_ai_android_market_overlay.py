"""Separate Android app's *structural* Binance public-only safety contract.

These are checks of source wiring only; they do NOT prove Android launches
on a device or that direct Binance HTTPS is accessible (HTTP451 on CI).
"""
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
OVERLAY=ROOT/"mobile"/"overlay"/"app"/"src"/"main"
NAMESPACE="{http://schemas.android.com/apk/res/android}"

class AndroidNativePublicChartContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.main=(OVERLAY/"java/com/example/llama/MainActivity.kt").read_text()
        cls.store=(OVERLAY/"java/com/example/llama/NativeWaveHistory.kt").read_text()
        cls.studies=(OVERLAY/"java/com/example/llama/NativeWaveStudy.kt").read_text()
        cls.service=(OVERLAY/"java/com/example/llama/NativeWaveSyncService.kt").read_text()
        cls.scheduler=(OVERLAY/"java/com/example/llama/NativeWaveScheduledJob.kt").read_text()
        cls.notebook=(OVERLAY/"java/com/example/llama/NativeWaveNotebook.kt").read_text()
        cls.build=(ROOT/"mobile/build_offline_android.sh").read_text()
        cls.manifest=ET.fromstring((OVERLAY/"AndroidManifest.xml").read_text())

    def test_pinned_public_url_only_and_no_signed_exchange_calls(self):
        self.assertIn('fapi.binance.com',self.store)
        self.assertIn('conn.requestMethod="GET"',self.store)
        self.assertIn("NativeWaveConfig.verifyPrivatePublicUrl(url)",self.store)
        self.assertNotIn('fapi.binance.com/fapi/v1/order',self.store)
        self.assertNotIn('POST',self.store)
        self.assertNotIn('api_key',self.store.lower())
        self.assertNotIn('api_secret',self.store.lower())
        self.assertNotIn('demo-fapi.binance.com',self.store)

    def test_15_native_timescales_and_15_known_market_pairs(self):
        self.assertIn('const val VENUE = "PUBLIC_USDM"',self.store)
        self.assertIn('"3d" to 259_200_000L',self.store)
        self.assertIn('"1w" to 604_800_000L',self.store)
        self.assertIn('"1M" to -1L',self.store)
        for sym in ("BTCUSDT","ETHUSDT","SOLUSDT","XRPUSDT",
                    "DOGEUSDT","AVAXUSDT","SUIUSDT","TRXUSDT"):
            self.assertIn(sym,self.store)
        self.assertIn("NativeWaveConfig.nextOpen",self.store)

    def test_causal_sqlite_snapshot_and_progress_after_reboot(self):
        self.assertIn("close_ms<=?",self.store)
        self.assertIn("PRIMARY KEY (venue,symbol,tf,open_ms)",self.store)
        self.assertIn("wave_notes",self.store)
        self.assertIn("beginTransaction()",self.store)
        self.assertIn('lastOpen(symbol,tf) ?: 0L',self.store)
        self.assertIn("last().closeTime",self.studies)
        self.assertIn("asOf",self.studies)
        self.assertIn("UNKNOWN",self.studies)
        self.assertIn("wave_degree",self.studies)
        self.assertIn("OFFSET 44",self.notebook)
        self.assertIn("close_ms>?",self.notebook)
        self.assertIn("NativeWaveStudy.archiveNote(bars)",self.notebook)

    def test_fgs_is_opt_in_not_boot_autotrading(self):
        names=[x.get(NAMESPACE+"name") for x in self.manifest.findall("uses-permission")]
        self.assertEqual(names,[
            "android.permission.INTERNET",
            "android.permission.RECEIVE_BOOT_COMPLETED",
            "android.permission.FOREGROUND_SERVICE",
            "android.permission.FOREGROUND_SERVICE_DATA_SYNC",
        ])
        app=self.manifest.find("application")
        self.assertEqual(app.get(NAMESPACE+"allowBackup"),"false")
        service=app.find("service")
        self.assertEqual(service.get(NAMESPACE+"exported"),"false")
        self.assertEqual(service.get(NAMESPACE+"foregroundServiceType"),"dataSync")
        services=app.findall("service")
        self.assertEqual(len(services),2)
        job=services[1]
        self.assertEqual(job.get(NAMESPACE+"name"),
                         "com.example.llama.NativeWaveScheduledJob")
        self.assertEqual(job.get(NAMESPACE+"permission"),
                         "android.permission.BIND_JOB_SERVICE")
        self.assertEqual(job.get(NAMESPACE+"exported"),"true")
        self.assertIn("setPersisted(true)",self.scheduler)
        self.assertIn("getBoolean(OPTED_IN, false)",self.scheduler)
        self.assertIn("pagesPerInterval = 1",self.scheduler)
        self.assertIn("maxDurationMs = 4L * 60_000L",self.scheduler)
        self.assertIn("marketSyncMutex",self.scheduler)
        self.assertIn("marketAuto.setOnCheckedChangeListener",self.main)
        self.assertIn("NativeWaveScheduledJob.setEnabled",self.main)
        self.assertNotIn('requestMethod="POST"',self.scheduler)
        self.assertNotIn('/fapi/v1/order',self.scheduler)
        self.assertIsNone(app.find("receiver"))
        self.assertIn("START_NOT_STICKY",self.service)
        self.assertIn("startForeground(6221",self.service)
        self.assertIn("NativeWaveDownloader(applicationContext).onePass",self.service)
        self.assertIn("marketStart.setOnClickListener",self.main)
        self.assertIn("marketStop.setOnClickListener",self.main)
        self.assertIn("NativeWaveStudy.prompt(applicationContext,lookupQuestion)",self.main)
        self.assertIn("QwenSession.isFollowUp(question)",self.main)
        self.assertIn('val lookupQuestion = question + if(followUp) " " + lastChatQuestion else ""',self.main)

    def test_debug_apk_is_still_distinct_from_original_williams_bot(self):
        self.assertIn('com.williamsai.offline',self.build)
        self.assertNotIn("com.williamsbot",self.build)
        for name in ("NativeWaveHistory","NativeWaveStudy",
                     "NativeWaveNotebook","NativeWaveSyncService",
                     "NativeWaveScheduledJob"):
            self.assertIn(name+".kt",self.build)
        self.assertIn("AndroidManifest.xml",self.build)

if __name__=="__main__":
    unittest.main()
