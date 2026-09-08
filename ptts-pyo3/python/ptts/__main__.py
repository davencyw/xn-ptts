"""Command line for the Python package: `python -m ptts "hello world"`.

The Rust CLI (`cargo install ptts-cli`) is the richer front end; this exists so
that `pip install ptts` alone is enough to make a sound.
"""

from __future__ import annotations

import argparse
import sys

from . import TTS, __version__, available_devices, available_quants


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ptts", description="Generate speech from text using Pocket TTS."
    )
    parser.add_argument("text", help="text to synthesize")
    parser.add_argument(
        "-o", "--output", default="out.wav", help="output WAV path (default: out.wav)"
    )
    parser.add_argument("-v", "--voice", help="voice name, or a path to a voice embedding")
    parser.add_argument("--list-voices", action="store_true", help="list voices and exit")
    parser.add_argument("--dir", help="local model directory instead of downloading")
    parser.add_argument(
        "--device", choices=available_devices(), help="device to run on (default: auto)"
    )
    parser.add_argument("--quant", choices=available_quants(), help="weight format; CPU only")
    parser.add_argument("-t", "--temperature", type=float, default=0.7)
    parser.add_argument("-s", "--seed", type=int, default=4242424242424242)
    parser.add_argument("--threads", type=int, help="CPU threads for tensor ops")
    parser.add_argument("--version", action="version", version=f"ptts {__version__}")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.threads is not None:
        # Must precede loading: it sizes a pool that is built once.
        from . import set_num_threads

        set_num_threads(args.threads)

    voice = args.voice
    # A path is registered under a name; a bare name is used as-is.
    embedding = voice if voice and voice.endswith(".safetensors") else None

    tts = TTS(
        dir=args.dir,
        device=args.device,
        quant=args.quant,
        temperature=args.temperature,
        seed=args.seed,
    )
    if embedding is not None:
        tts.add_voice("custom", embedding)
        voice = "custom"

    if args.list_voices:
        print("\n".join(tts.voices))
        return 0

    try:
        seconds = tts.save(args.output, args.text, voice=voice)
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    print(f"wrote {args.output} ({seconds:.2f}s of audio)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
