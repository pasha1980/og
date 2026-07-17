ODIN ?= odin

.PHONY: build test run clean

build:
	$(ODIN) build ./src -out:og

test:
	$(ODIN) test ./src/tests/ -file

clean:
	rm -f og

run: build
	./og help
