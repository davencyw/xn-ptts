"""Pocket TTS: text to 24 kHz speech, on device.

    import ptts

    tts = ptts.TTS()
    tts.save("out.wav", "Hello world")

See `python -m ptts --help` for the command line.
"""

from ._ptts import (
    AudioStream,
    TTS,
    __version__,
    available_devices,
    available_quants,
    build_info,
    get_num_threads,
    set_num_threads,
)

__all__ = [
    "TTS",
    "AudioStream",
    "__version__",
    "available_devices",
    "available_quants",
    "build_info",
    "get_num_threads",
    "set_num_threads",
]
