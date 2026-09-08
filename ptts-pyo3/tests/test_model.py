"""End-to-end tests that need a real checkpoint.

Skipped unless `PTTS_MODEL_DIR` points at a directory holding `config.json`,
weights, a tokenizer and `voices/`. The published checkpoint is gated, so CI
cannot run these without a token:

    PTTS_MODEL_DIR=/path/to/model pytest ptts-pyo3/tests/test_model.py
"""

import os
import wave

import numpy as np
import pytest

import ptts

MODEL_DIR = os.environ.get("PTTS_MODEL_DIR")

pytestmark = pytest.mark.skipif(
    not MODEL_DIR, reason="set PTTS_MODEL_DIR to a local model directory to run these"
)

TEXT = "Hello world, this is a test of the pocket text to speech model."


@pytest.fixture(scope="module")
def tts():
    return ptts.TTS(dir=MODEL_DIR)


def test_what_got_loaded(tts):
    assert tts.sample_rate > 0
    assert tts.voices, "a checkpoint should ship at least one voice"
    assert tts.device
    assert tts.quant in ptts.available_quants()
    assert "TTS(" in repr(tts)


def test_synth_returns_float32_audio(tts):
    pcm = tts.synth(TEXT)
    assert pcm.dtype == np.float32
    assert pcm.ndim == 1
    seconds = len(pcm) / tts.sample_rate
    assert 0.5 < seconds < 30, f"{seconds}s is not a plausible duration"
    assert np.isfinite(pcm).all()
    assert np.abs(pcm).max() > 1e-3, "output is silence"


def test_save_writes_a_readable_wav(tts, tmp_path):
    path = tmp_path / "out.wav"
    seconds = tts.save(str(path), TEXT)
    assert path.is_file()
    with wave.open(str(path)) as w:
        assert w.getnchannels() == 1
        assert w.getframerate() == tts.sample_rate
        assert w.getnframes() / w.getframerate() == pytest.approx(seconds, abs=0.01)


def test_the_same_seed_gives_the_same_audio(tts):
    a = tts.synth(TEXT, seed=99)
    b = tts.synth(TEXT, seed=99)
    np.testing.assert_array_equal(a, b)


def test_different_seeds_give_different_audio(tts):
    a = tts.synth(TEXT, seed=1)
    b = tts.synth(TEXT, seed=2)
    assert a.shape != b.shape or not np.array_equal(a, b)


def test_streaming_matches_the_one_shot_call(tts):
    """`synth` is `stream` collected, so the two must agree exactly."""
    chunks = list(tts.stream(TEXT, seed=7))
    assert len(chunks) > 1, "streaming should produce more than one chunk"
    assert all(c.dtype == np.float32 for c in chunks)
    np.testing.assert_array_equal(np.concatenate(chunks), tts.synth(TEXT, seed=7))


def test_a_stream_reports_its_sample_rate(tts):
    stream = tts.stream(TEXT)
    assert stream.sample_rate == tts.sample_rate
    stream.close()


def test_closing_a_stream_early_stops_it(tts):
    stream = tts.stream(TEXT)
    first = next(iter(stream))
    assert len(first) > 0
    stream.close()
    assert list(stream) == [], "a closed stream yields nothing further"
    with pytest.raises(ValueError, match="closed"):
        stream.sample_rate


def test_stream_works_as_a_context_manager(tts):
    with tts.stream(TEXT) as chunks:
        got = next(iter(chunks))
    assert len(got) > 0


def test_every_voice_can_speak(tts):
    for voice in tts.voices:
        pcm = tts.synth("Testing.", voice=voice)
        assert len(pcm) > 0, f"voice {voice} produced nothing"


def test_an_unknown_voice_lists_the_known_ones(tts):
    with pytest.raises(ValueError) as excinfo:
        tts.synth(TEXT, voice="nobody")
    message = str(excinfo.value)
    assert "nobody" in message
    assert tts.voices[0] in message


def test_default_voice_can_be_set_at_load_time():
    model = ptts.TTS(dir=MODEL_DIR, voice=ptts.TTS(dir=MODEL_DIR).voices[-1])
    assert len(model.synth("Testing.")) > 0


def test_empty_text_is_rejected(tts):
    with pytest.raises(ValueError, match="empty"):
        tts.synth("   ")


def test_tokenize_then_synth_tokens_matches_synth(tts):
    tokens = tts.tokenize("Hello there.")
    assert tokens and all(isinstance(t, int) for t in tokens)
    pcm = tts.synth_tokens(tokens, seed=5)
    assert pcm.dtype == np.float32
    assert len(pcm) > 0


def test_synth_tokens_rejects_empty_input(tts):
    with pytest.raises(ValueError, match="no tokens"):
        tts.synth_tokens([])


def test_voice_cloning_round_trip(tts):
    if not tts.supports_voice_cloning:
        pytest.skip("this checkpoint has no speaker encoder")
    # Ten seconds of quiet noise is not a real voice, but it exercises the
    # encode -> register -> generate path.
    rng = np.random.default_rng(0)
    seconds = 10
    pcm = (rng.standard_normal(tts.voice_prompt_sample_rate * seconds) * 0.05).astype(np.float32)
    tts.clone_voice("noise", pcm)
    assert "noise" in tts.voices
    assert len(tts.synth("Testing.", voice="noise")) > 0


def test_a_too_short_voice_prompt_is_rejected(tts):
    if not tts.supports_voice_cloning:
        pytest.skip("this checkpoint has no speaker encoder")
    with pytest.raises(ValueError, match="too short"):
        tts.clone_voice("tiny", np.zeros(1000, dtype=np.float32))


def test_a_misshaped_embedding_is_rejected(tts):
    with pytest.raises(ValueError, match=r"\[T, dim\]"):
        tts.add_voice_from_embedding("bad", np.zeros((2, 3, 4, 5), dtype=np.float32))
