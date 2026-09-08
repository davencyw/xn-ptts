# ptts

Pocket TTS from Python: text to 24 kHz speech, on device. No PyTorch, no server,
no GPU required — the inference runs in Rust and ships as a compiled wheel.

```bash
pip install ptts
```

```python
import ptts

tts = ptts.TTS()
tts.save("out.wav", "Hello world")
```

That downloads the checkpoint on first use and caches it the way
`huggingface_hub` does.

## Getting the samples instead of a file

`synth` returns a 1-D `float32` numpy array at `tts.sample_rate`:

```python
pcm = tts.synth("Hello world", voice="alba")
print(pcm.shape, tts.sample_rate)
```

## Streaming

`stream` yields chunks as the decoder produces them, so playback can start
before generation finishes:

```python
with tts.stream("A longer piece of text, streamed as it is generated.") as chunks:
    for pcm in chunks:
        play(pcm)          # each chunk is one codec frame of float32
```

Leaving the `with` block, or breaking out of the loop, stops the generation.

## Voices

```python
tts.voices                                  # ['alba', 'azelma', 'cosette', ...]
tts.synth("Hello", voice="marius")

tts = ptts.TTS(voice="marius")               # or set the default at load time
```

Clone a voice from about ten seconds of speech:

```python
import soundfile as sf

pcm, sr = sf.read("me.wav", dtype="float32")
assert sr == tts.voice_prompt_sample_rate
tts.clone_voice("me", pcm)
tts.save("out.wav", "Now in my voice.", voice="me")
```

`add_voice(name, path)` registers a precomputed embedding file instead, and
`add_voice_from_embedding(name, array)` one you already have in memory.

## Speed and size

Quantized weights are smaller and faster on CPU, at some cost in quality:

```python
tts = ptts.TTS(quant="q8_0")            # see ptts.available_quants()
tts = ptts.TTS(device="cuda")           # see ptts.available_devices()

ptts.set_num_threads(4)                 # call before loading
```

## Reproducibility

`temperature` and `seed` are set once at load time and can be overridden per
call:

```python
tts = ptts.TTS(temperature=0.7, seed=1234)
a = tts.synth("Hello", seed=1)
b = tts.synth("Hello", seed=1)          # identical to `a`
```

## Reference

| | |
|---|---|
| `TTS(dir, voice, device, quant, temperature, seed, cfg_coef, eos_threshold)` | load a model |
| `tts.save(path, text, **opts) -> float` | write a WAV, return its duration |
| `tts.synth(text, **opts) -> np.ndarray` | the whole waveform |
| `tts.stream(text, **opts) -> AudioStream` | chunks as they are decoded |
| `tts.tokenize(text) -> list[int]` | the tokens `synth` would use |
| `tts.synth_tokens(tokens, **opts)` | synthesize from your own tokens |
| `tts.add_voice(name, path)` | register a voice embedding file |
| `tts.add_voice_from_embedding(name, array)` | register an in-memory embedding |
| `tts.clone_voice(name, pcm)` | clone from PCM at `voice_prompt_sample_rate` |
| `tts.voices`, `tts.sample_rate`, `tts.device`, `tts.quant` | what got loaded |
| `ptts.available_devices()`, `ptts.available_quants()`, `ptts.build_info()` | what this build supports |

`**opts` is `voice`, `temperature`, `seed` and `cfg_coef`, each overriding the
value given to `TTS(...)` for that one call.

Ctrl-C interrupts a generation between chunks, so a long call stays
interruptible.

## License

MIT or Apache-2.0, at your option.
