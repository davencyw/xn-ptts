WASM_DIR := ../../ptts-wasm
PORT     ?= 8080
# The macOS firewall allows /usr/bin/python3 inbound by default; a Homebrew
# interpreter is silently blocked, which looks like the phone can't connect.
PYTHON   ?= $(shell test -x /usr/bin/python3 && echo /usr/bin/python3 || echo python3)

.PHONY: serve build rebuild prefetch clean

# Serve whatever is already built.
serve:
	$(PYTHON) serve.py --port $(PORT)

# Build the wasm bundle, then serve it.
build:
	$(MAKE) -C $(WASM_DIR) build
	$(PYTHON) serve.py --port $(PORT)

rebuild: clean build

# Warm the LAN weight cache before the phone ever asks for it.
prefetch:
	$(PYTHON) prefetch.py

clean:
	$(MAKE) -C $(WASM_DIR) clean
