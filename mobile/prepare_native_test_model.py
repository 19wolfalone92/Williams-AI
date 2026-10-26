"""Small pinned GGUF for emulator plumbing tests only; never bundled in the app."""
from hashlib import sha256
from pathlib import Path
import sys
from urllib.request import urlopen

EXPECTED = "66967fbece6dbe97886593fdbb73589584927e29119ec31f08090732d1861739"
URL = ("https://huggingface.co/ggml-org/models-moved/resolve/"
       "499bc8821c6b12b4e53c5bffcb21ec206f212d81/tinyllamas/stories15M-q4_0.gguf")


def prepare(project: Path):
    target = project / "app/src/androidTest/assets/native-runtime-test.gguf"
    target.parent.mkdir(parents=True, exist_ok=True)
    raw = target.read_bytes() if target.exists() else urlopen(URL, timeout=90).read(20_000_001)
    if len(raw) != 19_077_344 or sha256(raw).hexdigest() != EXPECTED:
        raise ValueError("NATIVE_TEST_MODEL_DIGEST_MISMATCH")
    if not target.exists():
        target.write_bytes(raw)


if __name__ == "__main__": prepare(Path(sys.argv[1]))
