# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Workspace layout

Cargo workspace (resolver "3", edition 2024) with three members:

- `ptts/` — core TTS library. Pure Rust, depends on the `xn` tensor/nn crate. Examples live under `ptts/examples/`: `pocket_tts` (end-to-end CLI) and `bench` (benchmark harness) both require the `sp` feature for SentencePiece; `quantize` (safetensors → GGUF converter that selectively quantizes `flow_lm.transformer.layers.*` weights) and `create_voice` (voice embeddings from audio samples) do not. `audio_helpers.rs` and `model_helpers.rs` are not examples — they are shared modules each example pulls in with `#[path = "..."] mod`, so `autoexamples = false` and every example is listed explicitly in `Cargo.toml`.
- `ptts-pyo3/` — PyO3 bindings exposing `TTSModel` to Python. Built with maturin; the cdylib is named `ptts`. Has its own `pyproject.toml` and `uv.lock`.
- `ptts-wasm/` — browser build via `wasm-bindgen` / `wasm-pack`. Ships a demo in `ptts-wasm/www/` (`index.html` + `worker.js`).

Shared dependency versions (notably `xn`) and the workspace version live in the top-level `Cargo.toml`. Bumping the release version means editing `workspace.package.version` and the `ptts` workspace dep.

## Build / test / lint

CI (`.github/workflows/rust-ci.yml`) runs on stable + nightly across Linux/macOS/Windows and is the source of truth:

```
cargo check
cargo test
cargo fmt --all -- --check        # rustfmt.toml: use_small_heuristics = "Max", edition 2024
cargo clippy -- -D warnings
```

CI deletes `.cargo/config.toml` before building because it pins `target-cpu=native`, which breaks portable dependency builds. If you reproduce CI failures locally, do the same (`rm -f .cargo/config.toml`) — otherwise keep the file in place for fast local builds.

Cargo features that gate optional functionality:

- `ptts`: `sp` (SentencePiece tokenizer, required by the `pocket_tts` and `bench` examples), `cuda`, `accelerate`, `xnnpack`.
- `ptts-pyo3`: `cuda`, `accelerate` (each forwards to both `xn/*` and `ptts/*`).

Run the CLI example:

```
cargo run --release --example pocket_tts --features sp -- "hello world" -o out.wav
```

It downloads weights from the `kyutai/pocket-tts` HuggingFace repo on first run. Built-in voice IDs: `alba`, `marius`, `javert`, `jean`, `fantine`, `cosette`, `eponine`, `azelma`. `--voice` also accepts a path to a 10s audio file or a precomputed voice safetensors.

Benchmark a local model:

```
cargo run --release --features sp,accelerate --example bench -- \
  --model model/model.q8.gguf --config model/config.json --quant q8 \
  --voice voices/freya.safetensors --threads 8 --iters 20
```

`bench` takes explicit paths and a precomputed voice embedding, never downloads, and reports time-to-first-audio, per-frame time, total generate time and RTF over `--iters` runs, excluding the one-off model load and voice conditioning. It decodes each frame on the generating thread rather than overlapping Mimi with the next frame's sampling as `pocket_tts` does, so its RTF reads lower than `pocket_tts` for the same weights — don't compare the two directly. `--threads` defaults to xn's one-per-logical-core, usually too many for a single autoregressive stream. For profiling rather than measuring, `pocket_tts --chrome-tracing` writes a Chrome trace for https://ui.perfetto.dev.

`--mimi-batch N` decodes N frames of latent per Mimi call instead of one, trading time-to-first-audio for throughput. Mimi expands each latent into 16 timesteps, so decoding one frame at a time hands every gemm in the decoder `m=16` — a single microkernel row panel, with almost no reuse of the packed weight — and the decoder is where most of the per-frame time goes on a small CPU. Batching is exact rather than an approximation: the decoder transformer is causal and the seanet convolutions are streaming, so N latents in one call produce the same samples as N successive calls (verified bit-comparable up to f32 summation order, including across the 250-frame context trim). Always report the `--mimi-batch 1` number alongside any batched one, since the two are different latency/throughput operating points rather than a before and after.

### XNNPACK f32 matmul

`--features xnnpack` routes f32 matmul through XNNPACK's `fully_connected` operator instead of the `gemm` crate. It matters here because `gemm` re-packs the weight panel on every call, and batch-1 decode gives it only 16 rows of output to amortise that over (Mimi expands one latent into 16 timesteps). XNNPACK packs the weights once inside `xnn_create_*` and caches the operator, so it reaches at `m = 16` the throughput `gemm` needs `m = 128` for. Measured on a Pi 5, Mimi decode drops from 30.5 to 13.8 ms per frame and RTF goes 1.65 -> 2.57 at `--threads 4`.

It needs a prebuilt XNNPACK, located from `XNNPACK_DIR` or from an `XNNPACK` directory sitting next to this workspace:

```
git clone --depth 1 https://github.com/google/XNNPACK.git
cd XNNPACK
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DXNNPACK_LIBRARY_TYPE=static \
  -DXNNPACK_BUILD_TESTS=OFF -DXNNPACK_BUILD_BENCHMARKS=OFF \
  -DXNNPACK_BUILD_ALL_MICROKERNELS=OFF -DXNNPACK_ENABLE_KLEIDIAI=OFF
ninja -C build XNNPACK
```

`XN_XNNPACK=0` disables the path at runtime, so one binary serves both sides of an A/B. `XN_XNNPACK_STATS=1` prints each distinct gemm shape the first time it is taken or declined, which is how to check the path is carrying the traffic rather than quietly declining it.

Two caveats. XNNPACK changes the generated audio: the gemm is correct to under 1e-5 relative against an f64 reference on every shape the model issues, but a last-bit difference in a sampled latent feeds back through the autoregressive loop, so the output is a different (equally valid) sample rather than a bit-identical one. And enabling it makes `--mimi-batch` mostly redundant, since batching existed to work around the same small-`m` problem.

## Threads

`--threads` sizes xn's *intra-operator* pool; it does not cap how many cores the process uses. `--pipeline` adds a decode thread outside that pool, so `--threads 1 --pipeline` runs on about 1.3 cores, not 1. For a genuine single-core number, pin with `taskset -c 0`.

`--split SAMPLE:DECODE` gives each pipeline stage its own xn pool instead of sharing the process-wide one. Sharing is why `--pipeline` alone buys little: the pool has a single job slot, so whichever stage publishes first fans out and the other silently runs serially. The stages scale very differently -- the flow LM is memory-bound (about 125 MB of weights per frame against a ~14 GB/s ceiling that does not scale with cores) and barely improves past one core, while the Mimi decoder scales well -- so the useful splits give Mimi most of the machine, e.g. `--split 1:2` on four cores.

## WASM build

From `ptts-wasm/`:

```
make build        # wasm-pack build --target web --release, then copies www/ into pkg/
make profiling    # same but --profiling (no wasm-opt)
```

Requires `wasm-pack` (`cargo install wasm-pack`). Serve `pkg/` with any static server (e.g. `python3 -m http.server 8080`). The demo downloads model weights (~240 MB) from HuggingFace and caches them. Wasm SIMD flags (`+simd128,+relaxed-simd`) and `getrandom_backend="wasm_js"` come from `.cargo/config.toml`.

## Python build

From the repo root:

```
maturin develop --manifest-path ptts-pyo3/Cargo.toml          # local install
maturin build --release --manifest-path ptts-pyo3/Cargo.toml  # produce wheel
```

Release wheels are produced by `.github/workflows/maturin-pub.yml` (Linux x86_64 manylinux + musllinux, Windows x64, macOS aarch64, sdist). It is autogenerated — regenerate with `maturin generate-ci github -m ptts-pyo3/Cargo.toml` rather than hand-editing.

## Architecture

The library implements Pocket TTS: text → tokens → flow-matching language model produces Mimi codec latents → Mimi decoder produces 24 kHz PCM audio.

`ptts/src/lib.rs` exposes a single `Tokenizer` trait (`encode` / `decode`) so each binding plugs in its own implementation:

- `pocket_tts` example: `SpTokenizer` wrapping `sentencepiece::SentencePieceProcessor`.
- `ptts-pyo3`: tokenizer is built from `tokenizer.model` shipped in the HF repo.
- `ptts-wasm`: `PresetTokenizer` — JS tokenizes in the browser and pushes IDs into the Rust state before each step.

Top-level orchestrator is `tts_model::TTSModel<Q>`, generic over a backend-quantization parameter `Q: BackendQ` from `xn`. It owns:

- `flow_lm: FlowLM<Q>` — token-conditioned flow-matching transformer that emits Mimi latents (`flow_lm.rs`, `transformer.rs`, `rope.rs`, `mlp.rs`, `layer_scale.rs`, `conditioners.rs`).
- `mimi: MimiDecoder<Unquantized<f32, Q::B>>` — neural audio codec decoder (`mimi.rs`, `seanet.rs`, `conv.rs`, `resample.rs`, `dummy_quantizer.rs`). The encoder side (`MimiEncoder` / `MimiEnc`) is used only for voice-prompt embedding from a 10s audio sample.

Generation is streaming and stateful: callers `init_flow_lm_state(batch, seq_len)`, then `prompt_text*` / `prompt_audio` to seed the state, then step-decode latents and feed them into `MimiDecoderState`. `lsd_decode_steps` controls flow-matching solver steps; `eos_threshold` controls termination. The default `TTSConfig::v202601` configuration is the canonical one consumed by all three frontends.

Quantization story: only `flow_lm.transformer.layers.*.{linear1,linear2,self_attn.in_proj,self_attn.out_proj}.weight` get GGML-quantized (see `examples/quantize.rs`); Mimi stays in `Unquantized<f32>`. The Mimi quantizer codebook tensors (`mimi.quantizer.*` except `output_proj`) are excluded from output GGUFs since the runtime uses `dummy_quantizer.rs`.

## Conventions to be aware of

- `target-cpu=native` is on by default for host builds and `apple-m1` on the macOS CI release lane; do not assume binaries are portable.
- macOS x86_64 disables AVX/AVX2 (`.cargo/config.toml`), keep that in mind when benchmarking.
- The single workspace version (`workspace.package.version`) is shared by all three crates and the `ptts` workspace dep — update them together.
