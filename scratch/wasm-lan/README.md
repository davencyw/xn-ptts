# wasm-lan — serve the Phonon browser demo to a phone

Scratch tooling for demoing Phonon in a browser on a device on the same network.
Nothing here is part of the shipped crates.

## Quick start

The demo loads its checkpoint from the same origin as the page, so the server has to
point at a model repo laid out as `model/` (weights, config, tokenizer) plus
`voices-prepared/`:

```bash
cd scratch/prep-voices && cargo run --release     # one-off: precompute the voices
cd ../wasm-lan && make build                      # builds ptts-wasm, then serves it
```

Then open the printed `http://<your-ip>:8080` on the phone. `make serve` skips the
build. `PORT=9000 make serve` moves it. `--models <dir>` points at a different repo
(default: the `phonon-inference` checkout beside this one).

More than one address may be printed — the first is the default route, which is
usually the one you want; a VPN like Tailscale contributes others.

## Why the voices are precomputed

A voice ships as an embedding that has to be run through `prompt_audio` before use —
a 125-frame prefill through the whole backbone. Doing that on a phone, once per
voice, at page load spends the device's thermal budget *before* the user asks for any
speech, and everything after that runs throttled.

`scratch/prep-voices` does the conditioning on the host and writes the finished KV
cache in the format `ptts-wasm` loads directly, so registering a voice in the browser
is a tensor copy with no arithmetic. The page also fetches voices lazily — one is
~9 MB and only one is ever heard at a time.

Output is verified equivalent to conditioning on the device: same rms, peak within
1.3% (float-ordering drift through autoregressive sampling).

## Why not just `python3 -m http.server`

That nearly works — the demo is single-threaded, uses no `SharedArrayBuffer` and needs
no cross-origin isolation. `serve.py` adds what bites when the client is a phone:

- **Prints the LAN URLs**, so you know what to type.
- **Threaded.** `http.server` is single-threaded, and the worker's 136 MB model
  download would otherwise block every other request for its whole duration.
- **`no-store` on everything in `pkg/`.** Mobile Safari will happily keep serving a
  stale `worker.js` after a rebuild. Local models get `no-cache` instead: cacheable
  but revalidated, so a reload costs one 304 rather than 136 MB.
- **`application/wasm`.** `WebAssembly.compileStreaming` rejects any other type.
  Stock `http.server` only gets this right on Python 3.10+.
- **`/models/`** serves the model repo, and **`/hf/`** proxies and disk-caches
  HuggingFace (`--no-proxy-hf` to disable) for checkpoints that still live there.
- **Warns about the macOS firewall.** See below.

## macOS firewall

With the firewall on, inbound LAN connections only reach allowed binaries. The system
`python3` is allowed out of the box; a Homebrew or pyenv one is not, and with stealth
mode the phone just hangs instead of being refused — it looks like a network fault.

The `Makefile` runs `/usr/bin/python3` when it exists. To use another interpreter:

```bash
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --add "$(which python3)"
```

`serve.py` prints this hint itself when it detects the situation.

## Measuring on the device

`serve.py` logs every request, which doubles as a result channel: a page can `fetch`
a `/beacon/<message>` URL and the message lands in the log. That is how the device
benchmark reports back, because `--dump-dom` snapshots before the wasm finishes.

## Notes

- Plain HTTP is fine. Nothing needs a secure context: `crypto.getRandomValues` (what
  `getrandom`'s `wasm_js` backend uses) works on insecure origins, unlike
  `crypto.subtle`. `--tls` exists if you want HTTPS, but the self-signed cert has to
  be installed *and* trusted on iOS under
  Settings → General → About → Certificate Trust Settings.
- `+relaxed-simd` is in the build flags, so this needs Safari 18+ / Chrome 114+.
- The page needs a tap before audio plays — `AudioContext` requires a user gesture.

## Files

| | |
|---|---|
| `serve.py` | the server; stdlib only, `--help` for flags |
| `prefetch.py` | warms `hf-cache/` from HuggingFace |
| `Makefile` | `build` / `serve` / `prefetch` / `clean` |
| `hf-cache/` | proxied weights (gitignored) |
