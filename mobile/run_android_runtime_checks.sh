#!/usr/bin/env bash
# Disposable emulator only: app-private SQLite rollback and real process restart.
# Neither APK contains account keys, signed endpoints nor trading functions.
set -euo pipefail
task_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$task_root/mobile/.build/llama.cpp/examples/llama.android"
./gradlew :app:connectedDebugAndroidTest \
  -Pandroid.testInstrumentationRunnerArguments.class=com.example.llama.NativeMarketRuntimeTest,com.example.llama.BookPracticeRuntimeTest,com.example.llama.NativeQwenSessionRuntimeTest \
  --no-daemon --max-workers=2

# Gradle's connectedDebugAndroidTest may UNINSTALL its test APK after running.
# Explicitly install BOTH test + target before our two independent instrument
# sessions. Never guess the test app ID (upstream llama.cpp can pin one).
main_apk="app/build/outputs/apk/debug/app-debug.apk"
test -s "$main_apk"
mapfile -t test_apks < <(find app/build/outputs/apk/androidTest -type f -name '*.apk' -print)
if [[ "${#test_apks[@]}" -ne 1 ]] || [[ ! -s "${test_apks[0]}" ]]; then
  echo "Expected exactly one nonempty Android instrumented APK" >&2
  exit 1
fi
adb install -r "$main_apk" >/dev/null
adb install -r "${test_apks[0]}" >/dev/null

# Bind only to the test runner that reports target=our isolated AI app.
# No fallback to another runner; failure is visible and blocks APK acceptance.
listing="$(adb shell pm list instrumentation | tr -d '\r')"
mapfile -t matches < <(printf '%s\n' "$listing" | \
  grep -E '^instrumentation:[A-Za-z0-9_.]+/androidx\.test\.runner\.AndroidJUnitRunner \(target=com\.williamsai\.offline\)$' || true)
if [[ "${#matches[@]}" -ne 1 ]]; then
  echo "No unique Android instrumentation bound to Williams AI: $listing" >&2
  exit 1
fi
runner="${matches[0]#instrumentation:}"
# Remove the pm-display annotation: am instrument accepts only pkg/runner.
runner="${runner%% *}"

adb shell am instrument -w -e class \
  'com.example.llama.NativeMarketProcessCheckpointTest#seedBeforeProcessStop' \
  "$runner" | tee /tmp/williams-ai-seed.txt
grep -F 'OK (1 test)' /tmp/williams-ai-seed.txt

adb shell am force-stop com.williamsai.offline

adb shell am instrument -w -e class \
  'com.example.llama.NativeMarketProcessCheckpointTest#verifyAfterProcessStop' \
  "$runner" | tee /tmp/williams-ai-restart.txt
grep -F 'OK (1 test)' /tmp/williams-ai-restart.txt
