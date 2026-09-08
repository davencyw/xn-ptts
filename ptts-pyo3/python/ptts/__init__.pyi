"""Type stubs for the `ptts` extension module.

Kept next to `Cargo.toml`; maturin ships `<module>.pyi` alongside the compiled
module so editors can see signatures and docstrings.
"""

from types import TracebackType
from typing import Iterator, Sequence

import numpy as np
from numpy.typing import NDArray

__version__: str

class TTS:
    """A loaded Pocket TTS model.

    Constructing one downloads the checkpoint on first use and caches it the
    way the `huggingface_hub` CLI does.
    """

    def __init__(
        self,
        dir: str | None = None,
        voice: str | None = None,
        device: str | None = None,
        quant: str | None = None,
        temperature: float = 0.7,
        seed: int = 4242424242424242,
        cfg_coef: float | None = None,
        eos_threshold: float | None = None,
    ) -> None:
        """Load a model.

        Args:
            dir: Local directory with `config.json`, weights, tokenizer and
                `voices/`. Defaults to downloading from the Hugging Face Hub.
            voice: Default voice for calls that do not name one.
            device: `"auto"` (default), `"cpu"`, `"cuda"`, `"vulkan"` or
                `"metal"`. See `available_devices()`.
            quant: Weight format, e.g. `"q8_0"`. CPU only; see
                `available_quants()`.
            temperature: Default sampling temperature.
            seed: Default sampling seed.
            cfg_coef: Classifier-free guidance coefficient; 1.0 disables it.
            eos_threshold: Override the config's end-of-speech threshold.
        """

    @property
    def sample_rate(self) -> int:
        """Sample rate of the audio this model produces, in Hz."""

    @property
    def voices(self) -> list[str]:
        """Names of the registered voices, sorted."""

    @property
    def device(self) -> str:
        """Device the model is running on, e.g. `"cpu"`."""

    @property
    def quant(self) -> str:
        """Weight format actually loaded, e.g. `"q8_0"`."""

    @property
    def voice_prompt_sample_rate(self) -> int:
        """Sample rate `clone_voice` expects its PCM in, in Hz."""

    @property
    def supports_voice_cloning(self) -> bool:
        """True if this checkpoint can clone voices from audio."""

    def synth(
        self,
        text: str,
        *,
        voice: str | None = None,
        temperature: float | None = None,
        seed: int | None = None,
        cfg_coef: float | None = None,
    ) -> NDArray[np.float32]:
        """Synthesize `text` and return the whole waveform."""

    def save(
        self,
        path: str,
        text: str,
        *,
        voice: str | None = None,
        temperature: float | None = None,
        seed: int | None = None,
        cfg_coef: float | None = None,
    ) -> float:
        """Synthesize `text` straight to a mono 16-bit WAV file.

        Returns the duration written, in seconds.
        """

    def stream(
        self,
        text: str,
        *,
        voice: str | None = None,
        temperature: float | None = None,
        seed: int | None = None,
        cfg_coef: float | None = None,
    ) -> AudioStream:
        """Synthesize `text`, yielding chunks as the decoder produces them."""

    def tokenize(self, text: str) -> list[int]:
        """Tokenize `text` the way `synth` would."""

    def synth_tokens(
        self,
        tokens: Sequence[int],
        *,
        frames_after_eos: int = 1,
        voice: str | None = None,
        temperature: float | None = None,
        seed: int | None = None,
        cfg_coef: float | None = None,
    ) -> NDArray[np.float32]:
        """Synthesize from token ids produced elsewhere."""

    def add_voice(self, name: str, path: str) -> None:
        """Register a voice from a precomputed embedding file."""

    def add_voice_from_embedding(
        self,
        name: str,
        embedding: NDArray[np.float32],
        *,
        null_embedding: NDArray[np.float32] | None = None,
    ) -> None:
        """Register a voice from an embedding of shape `[T, dim]` or `[1, T, dim]`."""

    def clone_voice(self, name: str, pcm: NDArray[np.float32]) -> None:
        """Clone a voice from ~10s of float32 PCM at `voice_prompt_sample_rate`."""

class AudioStream:
    """An in-progress generation. Iterate it for float32 chunks."""

    @property
    def sample_rate(self) -> int:
        """Sample rate of the chunks, in Hz."""

    def close(self) -> None:
        """Stop generating and release the worker threads."""

    def __iter__(self) -> Iterator[NDArray[np.float32]]: ...
    def __next__(self) -> NDArray[np.float32]: ...
    def __enter__(self) -> AudioStream: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool: ...

def get_num_threads() -> int:
    """Number of CPU threads used for tensor ops."""

def set_num_threads(num_threads: int) -> None:
    """Set the CPU thread count. Call before loading a model."""

def available_devices() -> list[str]:
    """Device names this build accepts, most capable first."""

def available_quants() -> list[str]:
    """Weight formats `quant=` accepts. All are CPU-only."""

def build_info() -> dict[str, object]:
    """The runtime's build configuration, for bug reports."""
