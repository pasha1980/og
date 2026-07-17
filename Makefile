ODIN ?= odin

.PHONY: build test run clean

build:
	$(ODIN) build . -out:og

test:
	$(ODIN) test tests/ -file

clean:
	rm -f og

run: build
	./og help
