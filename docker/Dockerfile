# syntax=docker/dockerfile:1
#
# Multi-arch benchmark container for viprs vs libvips profiling.
# Supports linux/arm64 and linux/amd64.
#
# Includes:
#   - cachegrind: deterministic cache simulation (always works)
#   - perf stat: real PMU counters (needs kernel PMU access)
#   - DHAT: heap allocation profiling with call stacks
#
# Build:
#   docker buildx build --platform linux/arm64 -t viprs-perf:arm64 -f docker/Dockerfile .
#   docker buildx build --platform linux/amd64 -t viprs-perf:amd64 -f docker/Dockerfile .
#
# Run:
#   docker run --privileged --rm -v $(pwd)/tests/fixtures/images:/data/input \
#     viprs-perf:arm64 /data/input/sample.jpg invert --iterations 10 --metrics all

FROM rust:1.96-bookworm AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ninja-build \
    git \
    pkg-config \
    libbrotli-dev \
    libpng-dev \
    libtiff-dev \
    libjpeg-dev \
    zlib1g-dev \
    liblzma-dev \
    libvips-dev \
    libturbojpeg-dev \
    libwebp-dev \
    libheif-dev \
    libopenjp2-7-dev \
    liblcms2-dev \
    libopenslide-dev \
    linux-perf \
    valgrind \
    binutils \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Build and install libjxl from source (required for jpegxl-sys)
RUN git clone --depth 1 https://github.com/libjxl/libjxl.git /tmp/libjxl \
    && mkdir -p /tmp/libjxl/build && cd /tmp/libjxl/build \
    && cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DBUILD_TESTING=OFF .. \
    && ninja -v && ninja install && ldconfig && rm -rf /tmp/libjxl

WORKDIR /workspace
COPY Cargo.toml Cargo.lock ./
COPY src/ src/
COPY xtask/ xtask/
COPY .cargo/ .cargo/
COPY tools/bench-vs-libvips/libvips_runner.c tools/bench-vs-libvips/Makefile tools/bench-vs-libvips/

# Build with counting allocator for allocation profiling
RUN cargo build --release -p xtask --features count-alloc

# Build libvips C runner
RUN make -C tools/bench-vs-libvips libvips-runner

# ----- Runtime stage -----
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libvips42 \
    libturbojpeg0-dev \
    libwebp-dev \
    libheif-dev \
    libopenjp2-7-dev \
    liblcms2-dev \
    libopenslide-dev \
    libbrotli-dev \
    libjpeg-dev \
    libtiff-dev \
    linux-perf \
    valgrind \
    binutils \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /workspace/target/release/xtask /usr/local/bin/viprs-xtask
COPY --from=builder /workspace/tools/bench-vs-libvips/libvips-runner /usr/local/bin/libvips-runner
COPY docker/scripts/ /opt/bench/

WORKDIR /data
ENTRYPOINT ["/opt/bench/entrypoint.sh"]
