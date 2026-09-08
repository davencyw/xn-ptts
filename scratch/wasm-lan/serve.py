#!/usr/bin/env python3
"""Serve the ptts-wasm build as a static site on the LAN.

Beyond `python3 -m http.server`, this adds the things that actually matter when
the client is a phone on the same network:

  * binds every interface and prints the URLs to type into the phone
  * threaded, so the page and its worker can fetch concurrently
  * `application/wasm` + `text/javascript` MIME types, no-store on everything
    (the phone always picks up a fresh `make build`)
  * an optional caching proxy for the HuggingFace weights, so the ~110-240 MB
    download crosses the WAN once and reaches the phone over LAN

Stdlib only.
"""

from __future__ import annotations

import argparse
import mimetypes
import os
import re
import shutil
import socket
import socketserver
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent

HF_HOST = "https://huggingface.co"
MODELS_PREFIX = "/models/"
HF_PREFIX = "/hf/"
# Rewritten into worker.js when proxying, so the phone never talks to HF directly.
REWRITES = [(f"'{HF_HOST}/", f"'{HF_PREFIX}"), (f'"{HF_HOST}/', f'"{HF_PREFIX}')]


class Server(ThreadingHTTPServer):
    daemon_threads = True

    def server_bind(self):
        # http.server's server_bind() resolves the bind address with
        # socket.getfqdn(), a reverse lookup that blocks for five seconds on a
        # host with no resolvable name — five seconds of the phone waiting.
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = self.server_address[:2]

    def handle_error(self, request, client_address):
        exc = sys.exc_info()[1]
        if isinstance(exc, (BrokenPipeError, ConnectionResetError)):
            return  # client hung up mid-transfer; nothing to report
        super().handle_error(request, client_address)


_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def cache_lock(key: str) -> threading.Lock:
    """One lock per upstream path, so two tabs don't race on the same file."""
    with _locks_guard:
        return _locks.setdefault(key, threading.Lock())


class Handler(SimpleHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    extensions_map = {
        **SimpleHTTPRequestHandler.extensions_map,
        ".wasm": "application/wasm",
        ".js": "text/javascript",
        ".mjs": "text/javascript",
        ".model": "application/octet-stream",
        ".safetensors": "application/octet-stream",
        ".gguf": "application/octet-stream",
    }

    def __init__(self, *a, config, **kw):
        self.config = config
        super().__init__(*a, directory=str(config.root), **kw)

    # -- logging -------------------------------------------------------------
    def log_message(self, fmt, *args):
        sys.stderr.write(f"  {self.address_string()}  {fmt % args}\n")

    def end_headers(self):
        # Everything out of pkg/ is small and must never go stale on the phone;
        # the proxied weights set their own long-lived Cache-Control. Local
        # models are big but do get re-quantized under the same filename, so
        # they are cacheable-but-revalidated: a reload costs one 304, not 88 MB.
        if self.path.startswith(MODELS_PREFIX):
            self.send_header("Cache-Control", "no-cache")
        elif not self.path.startswith(HF_PREFIX):
            self.send_header("Cache-Control", "no-store")
        super().end_headers()

    # -- routing -------------------------------------------------------------
    def do_GET(self):
        if self.config.proxy_hf and self.path.startswith(HF_PREFIX):
            self.serve_hf(self.path[len(HF_PREFIX):], body=True)
        else:
            super().do_GET()

    def do_HEAD(self):
        if self.config.proxy_hf and self.path.startswith(HF_PREFIX):
            self.serve_hf(self.path[len(HF_PREFIX):], body=False)
        else:
            super().do_HEAD()

    def translate_path(self, path):
        if not path.startswith(MODELS_PREFIX):
            return super().translate_path(path)
        saved, self.directory = self.directory, str(self.config.models)
        try:
            return super().translate_path(path[len(MODELS_PREFIX) - 1:])
        finally:
            self.directory = saved

    def send_head(self):
        """Rewrite HF URLs in worker.js on the way out."""
        if not (self.config.proxy_hf and self.path.split("?")[0].endswith(".js")):
            return super().send_head()

        path = Path(self.translate_path(self.path))
        if not path.is_file():
            return super().send_head()

        text = path.read_text(encoding="utf-8")
        for old, new in REWRITES:
            text = text.replace(old, new)
        blob = text.encode("utf-8")

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/javascript")
        self.send_header("Content-Length", str(len(blob)))
        self.end_headers()
        if self.command == "HEAD":
            return None
        import io

        return io.BytesIO(blob)

    # -- HuggingFace proxy ---------------------------------------------------
    def serve_hf(self, rel: str, *, body: bool):
        rel = rel.split("?")[0].lstrip("/")
        if not rel or ".." in rel.split("/"):
            self.send_error(HTTPStatus.BAD_REQUEST, "bad upstream path")
            return

        cached = self.config.hf_cache / rel
        with cache_lock(rel):
            if not cached.is_file():
                try:
                    self.fetch_upstream(rel, cached)
                except (urllib.error.URLError, OSError) as exc:
                    self.log_message("upstream failed %s: %s", rel, exc)
                    self.send_error(HTTPStatus.BAD_GATEWAY, f"upstream: {exc}")
                    return

        size = cached.stat().st_size
        ctype = self.guess_type(str(cached))
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(size))
        self.send_header("Cache-Control", "public, max-age=31536000, immutable")
        self.end_headers()
        if body:
            with cached.open("rb") as fh:
                shutil.copyfileobj(fh, self.wfile, length=1 << 20)

    def fetch_upstream(self, rel: str, dest: Path):
        url = f"{HF_HOST}/{rel}"
        self.log_message("cache miss, fetching %s", url)
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_suffix(dest.suffix + ".part")
        req = urllib.request.Request(url, headers={"User-Agent": "ptts-lan-serve"})
        with urllib.request.urlopen(req, timeout=60) as resp, part.open("wb") as out:
            total = int(resp.headers.get("content-length") or 0)
            done = 0
            while chunk := resp.read(1 << 20):
                out.write(chunk)
                done += len(chunk)
                if total:
                    sys.stderr.write(
                        f"\r    {rel}  {done / 1e6:7.1f} / {total / 1e6:.1f} MB"
                    )
                    sys.stderr.flush()
        if total:
            sys.stderr.write("\n")
        part.replace(dest)


def lan_addresses() -> list[str]:
    """This host's LAN IPv4 addresses, most-likely-useful first.

    Deliberately DNS-free: resolving this machine's own `.local` hostname blocks
    for five seconds before failing, which would delay serving by that long.
    """
    addrs = []
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect(("192.0.2.1", 1))  # TEST-NET-1; nothing is actually sent
        addrs.append(sock.getsockname()[0])  # the address of the default route
        sock.close()
    except OSError:
        pass

    # Secondary interfaces (wired alongside wifi, say) so a phone on either can
    # still be pointed somewhere that works.
    cmd = ["ifconfig"] if sys.platform == "darwin" else ["ip", "-4", "-o", "addr"]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        out = ""
    for ip in re.findall(r"inet (?:addr:)?(\d+\.\d+\.\d+\.\d+)", out):
        if ip not in addrs and not ip.startswith(("127.", "169.254.")):
            addrs.append(ip)
    return addrs


# Apple-shipped interpreters the firewall already covers via its /usr/bin/python3
# rule. `sys.executable` for the system python3 is the CommandLineTools path, not
# the /usr/bin shim, so matching on the shim alone would false-positive.
ALLOWED_PREFIXES = ("/usr/bin/", "/Library/Developer/", "/Applications/Xcode")


def firewall_hint() -> None:
    """macOS drops inbound LAN traffic to binaries the firewall hasn't allowed.

    The system python3 is allowed out of the box; a Homebrew or pyenv
    interpreter is not, and with stealth mode on the failure looks like a
    network problem rather than a block. Warn instead of letting the phone
    time out. Runs on a background thread — `socketfilterfw` takes seconds and
    must not delay serving.
    """
    executable = os.path.realpath(sys.executable)
    if sys.platform != "darwin" or executable.startswith(ALLOWED_PREFIXES):
        return
    fw = "/usr/libexec/ApplicationFirewall/socketfilterfw"
    try:
        if "State = 1" not in subprocess.run(
            [fw, "--getglobalstate"], capture_output=True, text=True, timeout=10
        ).stdout:
            return
        listed = subprocess.run(
            [fw, "--listapps"], capture_output=True, text=True, timeout=10
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return
    if executable in listed:
        return
    print(
        f"warning: the macOS firewall is on and has no rule for\n"
        f"           {executable}\n"
        f"         inbound LAN connections will be dropped and the phone will just hang.\n"
        f"         Either re-run with the system interpreter, which is allowed by default:\n"
        f"           /usr/bin/python3 {' '.join(sys.argv)}\n"
        f"         or allow this one once:\n"
        f"           sudo {fw} --add {executable}\n",
        file=sys.stderr,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=REPO / "ptts-wasm" / "pkg",
                    help="directory to serve (default: ptts-wasm/pkg)")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--bind", default="0.0.0.0")
    ap.add_argument("--no-proxy-hf", dest="proxy_hf", action="store_false",
                    help="let the phone download weights from HuggingFace directly")
    ap.add_argument("--hf-cache", type=Path, default=HERE / "hf-cache",
                    help="where proxied HuggingFace files are cached on disk")
    ap.add_argument("--models", type=Path, default=HERE.parent.parent.parent / "phonon-inference",
                    help="model repo served under /models/ (expects model/ and voices/)")
    ap.add_argument("--tls", action="store_true",
                    help="serve https with a self-signed cert (needs manual trust on iOS)")
    cfg = ap.parse_args()

    cfg.root = cfg.root.resolve()
    if not (cfg.root / "index.html").is_file():
        print(f"error: {cfg.root} has no index.html — run `make build` in ptts-wasm/ first",
              file=sys.stderr)
        return 1
    cfg.models = cfg.models.resolve()
    cfg.hf_cache = cfg.hf_cache.resolve()
    cfg.hf_cache.mkdir(parents=True, exist_ok=True)

    mimetypes.add_type("application/wasm", ".wasm")

    def factory(*a, **kw):
        return Handler(*a, config=cfg, **kw)

    server = Server((cfg.bind, cfg.port), factory)
    scheme = "http"
    if cfg.tls:
        import ssl

        cert, key = ensure_cert(HERE, lan_addresses())
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(cert, key)
        server.socket = ctx.wrap_socket(server.socket, server_side=True)
        scheme = "https"

    print(f"serving {cfg.root}")
    print(f"  HuggingFace weights: {'proxied + cached in ' + str(cfg.hf_cache) if cfg.proxy_hf else 'fetched directly by the browser'}")
    local = sorted(cfg.models.rglob("*.gguf")) if cfg.models.is_dir() else []
    voices = sorted(cfg.models.glob("voices/*.safetensors")) if cfg.models.is_dir() else []
    print(f"  /models/ -> {cfg.models}")
    print(f"    weights: {', '.join(m.name for m in local) or 'none'}"
          f"   voices: {len(voices)}")
    print("\nopen on your phone:")
    for ip in lan_addresses():
        print(f"  {scheme}://{ip}:{cfg.port}")
    print(f"  {scheme}://localhost:{cfg.port}   (this machine)\n", flush=True)

    threading.Thread(target=firewall_hint, daemon=True).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
