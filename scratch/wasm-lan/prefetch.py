#!/usr/bin/env python3
"""Pre-download the HuggingFace assets the demo needs into serve.py's cache.

Optional: serve.py fetches on first request anyway. Running this first just
means the phone never waits on the WAN.
"""

from pathlib import Path
import sys
import urllib.request

HERE = Path(__file__).resolve().parent
CACHE = HERE / "hf-cache"
HOST = "https://huggingface.co"

BASE = "kyutai/pocket-tts-without-voice-cloning/resolve/main"
BASE_Q8 = "lmz/pocket-tts-without-voice-cloning-q8/resolve/main"
VOICES = ["alba", "marius", "javert", "fantine", "cosette", "eponine", "azelma"]

PATHS = [
    f"{BASE}/tokenizer.model",
    f"{BASE_Q8}/tts_b6369a24.gguf",
    *[f"{BASE}/embeddings_v2/{v}.safetensors" for v in VOICES],
]
if "--f32" in sys.argv:
    PATHS.append(f"{BASE}/tts_b6369a24.safetensors")


def fetch(rel: str) -> None:
    dest = CACHE / rel
    if dest.is_file():
        print(f"  have  {rel}  ({dest.stat().st_size / 1e6:.1f} MB)")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(f"{HOST}/{rel}", headers={"User-Agent": "ptts-lan-serve"})
    with urllib.request.urlopen(req, timeout=60) as resp, part.open("wb") as out:
        total = int(resp.headers.get("content-length") or 0)
        done = 0
        while chunk := resp.read(1 << 20):
            out.write(chunk)
            done += len(chunk)
            print(f"\r  get   {rel}  {done / 1e6:7.1f} / {total / 1e6:.1f} MB", end="", flush=True)
    print()
    part.replace(dest)


for path in PATHS:
    fetch(path)
print(f"\ncache: {CACHE}")
