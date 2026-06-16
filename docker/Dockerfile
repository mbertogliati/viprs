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
ARG MAKE_JOBS=1

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
    && cd /tmp/libjxl && ./deps.sh && mkdir -p build && cd build \
    && cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DBUILD_TESTING=OFF .. \
    && ninja -v && ninja install && ldconfig && rm -rf /tmp/libjxl

# Build and install newer libheif (>=1.16) from source to satisfy libheif-sys
RUN apt-get update && apt-get install -y --no-install-recommends \
    libde265-dev \
    libx265-dev \
    libaom-dev \
    && rm -rf /var/lib/apt/lists/* \
    && git clone --depth 1 https://github.com/strukturag/libheif.git /tmp/libheif \
    && mkdir -p /tmp/libheif/build && cd /tmp/libheif/build \
    && cmake -DCMAKE_BUILD_TYPE=Release .. \
    && make -j${MAKE_JOBS} && make install && ldconfig && rm -rf /tmp/libheif

# Build libjpeg-turbo >= 3.0 from source so turbojpeg-sys can find libturbojpeg >= 3.0
RUN apt-get update && apt-get install -y --no-install-recommends \
    nasm \
    && rm -rf /var/lib/apt/lists/* \
    && git clone --depth 1 --branch 3.0.0 https://github.com/libjpeg-turbo/libjpeg-turbo.git /tmp/libjpeg-turbo \
    && mkdir -p /tmp/libjpeg-turbo/build && cd /tmp/libjpeg-turbo/build \
    && cmake -DCMAKE_BUILD_TYPE=Release -DWITH_JPEG8=1 .. \
    && make -j${MAKE_JOBS} && make install && ldconfig && rm -rf /tmp/libjpeg-turbo


WORKDIR /workspace
COPY Cargo.toml Cargo.lock ./
COPY src/ src/
COPY xtask/ xtask/
COPY .cargo/ .cargo/
COPY tools/bench-vs-libvips/libvips_runner.c tools/bench-vs-libvips/Makefile tools/bench-vs-libvips/

# Ensure pkg-config can find libraries installed into /opt/libjpeg-turbo
ENV PKG_CONFIG_PATH=/opt/libjpeg-turbo/lib/pkgconfig:/opt/libjpeg-turbo/lib64/pkgconfig

# NOTE: In this hybrid approach the OCI image is a "native-deps" base image.
# Cargo/Rust builds (xtask and viprs) will be executed inside CI jobs that use
# this image as a container. Skipping cargo build here avoids repeated failures
# during Docker image iteration while still providing all native libraries.

# ----- Runtime stage -----
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
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

# xtask and bench runners are expected to be built in CI using this image as the base.
# Keep helper scripts in the image.
COPY docker/scripts/ /opt/bench/

WORKDIR /data
ENTRYPOINT ["/opt/bench/entrypoint.sh"]
