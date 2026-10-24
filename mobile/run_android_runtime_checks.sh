#!/usr/bin/env bash
# Disposable emulator only: exercises private test databases, no model or account.
set -euo pipefail
task_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$task_root/mobile/.build/llama.cpp/examples/llama.android"
./gradlew :app:connectedDebugAndroidTest \
  -Pandroid.testInstrumentationRunnerArguments.class=com.example.llama.NativeMarketRuntimeTest \
  --no-daemon --max-workers=2
adb shell am instrument -w -e class \
  'com.example.llama.NativeMarketProcessCheckpointTest#seedBeforeProcessStop' \
  com.williamsai.offline.test/androidx.test.runner.AndroidJUnitRunner | tee /tmp/williams-ai-seed.txt
grep -F 'OK (1 test)' /tmp/williams-ai-seed.txt
adb shell am force-stop com.williamsai.offline
adb shell am instrument -w -e class \
  'com.example.llama.NativeMarketProcessCheckpointTest#verifyAfterProcessStop' \
  com.williamsai.offline.test/androidx.test.runner.AndroidJUnitRunner | tee /tmp/williams-ai-restart.txt
grep -F 'OK (1 test)' /tmp/williams-ai-restart.txt
