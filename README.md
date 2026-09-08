# Phonon

Local text-to-speech in Rust. Runs [Kyutai's Pocket TTS](https://github.com/kyutai-labs/pocket-tts)
on the [`xn`](https://github.com/LaurentMazare/xn) tensor crate — on the CPU, with no server and no
Python in the loop. Fast enough to stream faster than realtime on a laptop, and small enough to run
in a browser tab.

## Command line

```bash
cargo install phonon-cli
phonon "hello world" -o out.wav
```

The crate is `phonon-cli`; the binary it installs is `phonon`. Weights download on first run.

```bash
phonon --list-voices
phonon "hello world" --voice marius
phonon "hello world" --voice sample.wav   # ~10s of audio to clone
```

Bundled voices: `alba`, `marius`, `javert`, `jean`, `fantine`, `cosette`, `eponine`, `azelma`.

## In the browser

`ptts-wasm/` builds a WebAssembly bundle that runs the whole pipeline client-side. See
[`ptts-wasm/README.md`](ptts-wasm/README.md). Upstream hosts a demo of the same model
[here](https://laurentmazare.github.io/pocket-tts).

## Workspace

| Crate | What it is |
|---|---|
| [`ptts/`](ptts/) | the library — `Synth` is the entry point |
| [`phonon-cli/`](phonon-cli/) | the `phonon` binary |
| [`ptts-wasm/`](ptts-wasm/) | WebAssembly build + browser demo |
| [`ptts-pyo3/`](ptts-pyo3/) | Python bindings, built with maturin |
| [`ptts-ws-server/`](ptts-ws-server/) | WebSocket streaming server |

## Naming

The product is **Phonon**. The model it runs is Kyutai's Pocket TTS, and HuggingFace repo paths
still reference `kyutai/pocket-tts` because that is where the weights live. The library crate is
published as `ptts`; `phonon` on crates.io belongs to an unrelated audio project, which is why the
CLI crate is `phonon-cli`.
