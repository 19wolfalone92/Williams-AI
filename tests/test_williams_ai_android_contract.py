"""Static no-trade Android release gates. No network, no emulators needed."""
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
MOBILE = ROOT / "mobile"


class AndroidOfflineContract(unittest.TestCase):
    def test_android_package_is_not_trading_app(self):
        script = (MOBILE / "build_offline_android.sh").read_text(encoding="utf-8")
        self.assertIn('applicationId = "com.williamsai.offline"', script)
        self.assertNotIn("com.williamsbot", script)
        self.assertNotIn("19wolfalone92/Williams.git", script)

    def test_opt_in_internet_search_permission(self):
        manifest = (MOBILE / "overlay/app/src/main/AndroidManifest.xml").read_text(encoding="utf-8")
        self.assertIn('android.permission.INTERNET', manifest)
        self.assertNotIn("ACCESS_FINE_LOCATION", manifest)
        self.assertIn('android:allowBackup="false"', manifest)

    def test_pin_is_exact_commit_and_never_branch(self):
        pin = (MOBILE / "UPSTREAM_LLAMA_CPP_SHA").read_text().strip()
        self.assertRegex(pin, r"^[0-9a-f]{40}$")
        script = (MOBILE / "build_offline_android.sh").read_text()
        self.assertIn('rev-parse HEAD', script)
        self.assertIn('= "$pin"', script)

    def test_native_builder_copies_research_layout_and_checks_permissions(self):
        script = (MOBILE / "build_offline_android.sh").read_text(encoding="utf-8")
        for path in (
            "app/src/main/java/com/example/llama/WebResearch.kt",
            "app/src/main/res/layout/activity_main.xml",
        ):
            self.assertIn(path, script)
        self.assertIn('"android.permission.INTERNET",', script)
        self.assertIn('"android.permission.FOREGROUND_SERVICE",', script)
        self.assertIn('"android.permission.FOREGROUND_SERVICE_DATA_SYNC",', script)
        # BIND_JOB_SERVICE is a permission REQUIRED OF THE SYSTEM CALLER
        # on one exported service, not a permission requested by the app.
        import xml.etree.ElementTree as ET
        root = ET.parse(MOBILE / "overlay/app/src/main/AndroidManifest.xml").getroot()
        ns = "{http://schemas.android.com/apk/res/android}"
        requested = [node.get(ns + "name") for node in root.findall("uses-permission")]
        self.assertNotIn("android.permission.BIND_JOB_SERVICE", requested)
        self.assertIn("android.permission.RECEIVE_BOOT_COMPLETED", requested)
        service = next(s for s in root.findall("application/service") if
                       s.get(ns + "name") == "com.example.llama.NativeWaveScheduledJob")
        self.assertEqual(service.get(ns + "permission"),
                         "android.permission.BIND_JOB_SERVICE")
        layout = (MOBILE / "overlay/app/src/main/res/layout/activity_main.xml").read_text(encoding="utf-8")
        self.assertIn('android:id="@+id/web_enabled"', layout)
        self.assertIn('android:id="@+id/book_button"', layout)
        self.assertIn("BookKnowledge.kt", script)

    def test_secure_local_model_import(self):
        kotlin = (MOBILE / "overlay/app/src/main/java/com/example/llama/MainActivity.kt").read_text(encoding="utf-8")
        for required in (
            "MessageDigest.getInstance", "SHA-256", "File.createTempFile",
            "contentResolver.openInputStream", "partial.renameTo(destination)",
            "QwenSession.answer", "AiChat.getInferenceEngine",
        ):
            self.assertIn(required, kotlin)
        self.assertIn("WebResearch.shouldLookup", kotlin)
        self.assertIn("BookKnowledge.retrieve", kotlin)
        self.assertIn("bookPicker", kotlin)
        self.assertIn("webToggle.isChecked", kotlin)
        session = (MOBILE / "overlay/app/src/main/java/com/example/llama/QwenSession.kt").read_text()
        self.assertIn("engine.resetSession()", session)
        self.assertIn("engine.sendUserPrompt", session)
        for forbidden in (
            "BinanceFuturesClient", "api.binance.com", "OPENAI_API_KEY",
            "api.binance.com", "orderEndpoint", "POST /fapi",
        ):
            self.assertNotIn(forbidden, kotlin)

    def test_resumable_private_study_and_curriculum_are_packaged(self):
        main = (MOBILE / "overlay/app/src/main/java/com/example/llama/MainActivity.kt").read_text()
        study = (MOBILE / "overlay/app/src/main/java/com/example/llama/StudyMemory.kt").read_text()
        script = (MOBILE / "build_offline_android.sh").read_text()
        self.assertIn("StudyMemory.kt", script)
        self.assertIn("LEARNING_CURRICULUM_RU.md", script)
        self.assertIn("StudyMemory.save", main)
        self.assertIn("StudyMemory.retrieve", main)
        self.assertIn("StudyMemory.curriculum", main)
        self.assertIn("generation?.cancel()", main)
        self.assertIn("PRIMARY KEY(source_key,model_id)", study)
        self.assertIn("contains(normalized(support))", study)
        self.assertIn("android.system.Os.rename", main)


if __name__ == "__main__":
    unittest.main()
