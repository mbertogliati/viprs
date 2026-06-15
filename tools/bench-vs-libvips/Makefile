CXX := c++
CC := cc

# Maximum optimisation flags — must match what libvips would use in production.
# -O3:          enable all loop/auto-vectorisation passes (libvips uses -O3 internally)
# -march=native: generate NEON/AVX2 for the exact CPU running the bench — same as Rust's target-cpu=native
# -ffast-math:  allow reassociation, flush denormals (libvips enables this via -ffast-math in its own build)
# -flto:        link-time optimisation — eliminates dead libvips wrappers, inlines hot paths
# These flags mirror what a production distro package or self-compiled libvips would use.
OPT_FLAGS := -O3 -march=native -ffast-math -flto -g

CFLAGS := $(shell pkg-config --cflags vips) $(OPT_FLAGS) -Wall -pthread
CPPFLAGS := $(shell pkg-config --cflags vips OpenEXR) $(OPT_FLAGS) -Wall -std=c++17
LDFLAGS := $(shell pkg-config --libs vips) -flto -pthread
OPENEXR_LDFLAGS := $(shell pkg-config --libs vips OpenEXR) -flto

.PHONY: all clean

all: libvips-runner openexr-runner libvips-web-runner

libvips-runner: libvips_runner.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

libvips-web-runner: libvips_web_runner.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

openexr-runner: openexr_runner.cpp
	$(CXX) $(CPPFLAGS) -o $@ $< $(OPENEXR_LDFLAGS)

clean:
	rm -f libvips-runner openexr-runner libvips-web-runner
