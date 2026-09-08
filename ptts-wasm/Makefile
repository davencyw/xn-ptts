.PHONY: build profiling clean

build:
	wasm-pack build --target web --release
	cp www/index.html www/worker.js www/logo.png pkg/

profiling:
	wasm-pack build --target web --profiling
	cp www/index.html www/worker.js www/logo.png pkg/

clean:
	rm -rf pkg
