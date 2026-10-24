#!/usr/bin/env bash
# Build only the independent Williams AI app from a pinned upstream llama.cpp example.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
pin="$(tr -d '\r\n' < "$root/mobile/UPSTREAM_LLAMA_CPP_SHA")"
if [[ ! "$pin" =~ ^[0-9a-f]{40}$ ]]; then echo "Invalid pinned SHA" >&2; exit 1; fi
work="$root/mobile/.build/llama.cpp"
mkdir -p "$work"
if [[ ! -d "$work/.git" ]]; then
  git -C "$work" init -q
  git -C "$work" remote add origin https://github.com/ggml-org/llama.cpp.git
fi
git -C "$work" fetch --depth 1 origin "$pin"
git -C "$work" checkout --detach -f FETCH_HEAD
test "$(git -C "$work" rev-parse HEAD)" = "$pin"

project="$work/examples/llama.android"
for file in \
  app/src/main/java/com/example/llama/MainActivity.kt \
  app/src/main/java/com/example/llama/WebResearch.kt \
  app/src/main/java/com/example/llama/BookKnowledge.kt \
  app/src/main/java/com/example/llama/StudyMemory.kt \
  app/src/main/java/com/example/llama/NativeWaveHistory.kt \
  app/src/main/java/com/example/llama/NativeWaveStudy.kt \
  app/src/main/java/com/example/llama/NativeWaveNotebook.kt \
  app/src/main/java/com/example/llama/NativeWaveSyncService.kt \
  app/src/main/java/com/example/llama/NativeWaveScheduledJob.kt \
  app/src/main/res/values/strings.xml \
  app/src/main/res/layout/activity_main.xml \
  app/src/main/AndroidManifest.xml; do
  cp "$root/mobile/overlay/$file" "$project/$file"
done

mkdir -p "$project/app/src/main/assets"
cp "$root/knowledge/LEARNING_CURRICULUM_RU.md" "$project/app/src/main/assets/LEARNING_CURRICULUM_RU.md"

python3 - "$project/app/build.gradle.kts" <<'PY'
from pathlib import Path
import sys
p = Path(sys.argv[1])
data = p.read_text()
needle = 'applicationId = "com.example.llama.aichat"'
assert data.count(needle) == 1, "Upstream changed: review before patching"
p.write_text(data.replace(needle, 'applicationId = "com.williamsai.offline"'))
PY

python3 - "$project/app/src/main/AndroidManifest.xml" <<'PY'
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
root = ET.fromstring(Path(sys.argv[1]).read_text())
ns = "{http://schemas.android.com/apk/res/android}"
permissions = [x.attrib.get(ns + "name") for x in root.findall("uses-permission")]
assert permissions == [
    "android.permission.INTERNET",
    "android.permission.RECEIVE_BOOT_COMPLETED",
    "android.permission.FOREGROUND_SERVICE",
    "android.permission.FOREGROUND_SERVICE_DATA_SYNC",
], f"Unexpected permissions: {permissions}"
app = root.find("application")
services = {item.attrib.get(ns+"name"): item
            for item in app.findall("service")}
assert set(services) == {
    "com.example.llama.NativeWaveSyncService",
    "com.example.llama.NativeWaveScheduledJob",
}, f"Unexpected Android services: {services}"
assert services["com.example.llama.NativeWaveSyncService"].attrib.get(
    ns+"exported") == "false"
assert services["com.example.llama.NativeWaveScheduledJob"].attrib.get(
    ns+"permission") == "android.permission.BIND_JOB_SERVICE"
assert services["com.example.llama.NativeWaveScheduledJob"].attrib.get(
    ns+"exported") == "true"
PY

cd "$project"
chmod +x ./gradlew
./gradlew :app:assembleDebug --no-daemon --max-workers=2

output="$root/mobile/out"
mkdir -p "$output"
apk="$project/app/build/outputs/apk/debug/app-debug.apk"
test -s "$apk"
cp "$apk" "$output/WilliamsAI-offline-debug.apk"
cd "$output"
sha256sum WilliamsAI-offline-debug.apk > WilliamsAI-offline-debug.sha256
cat WilliamsAI-offline-debug.sha256
