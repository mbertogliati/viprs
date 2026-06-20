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

FROM debian:trixie-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    cmake \
    make \
    git \
    git-lfs \
    jq \
    bc \
    curl \
    ca-certificates \
    libvips42 \
    libvips-dev \
    libturbojpeg0-dev \
    libwebp-dev \
    libspng-dev \
    libpng-dev \
    libheif-dev \
    libheif-plugin-aomdec \
    libheif-plugin-aomenc \
    libjxl-dev \
    libopenjp2-7-dev \
    liblcms2-dev \
    libopenslide-dev \
    libbrotli-dev \
    libjpeg-dev \
    libtiff-dev \
    linux-perf \
    valgrind \
    binutils \
    && rm -rf /var/lib/apt/lists/*

RUN pkg-config --modversion libturbojpeg && \
    pkg-config --modversion spng && \
    pkg-config --modversion libjxl && \
    pkg-config --modversion libheif && \
    pkg-config --modversion lcms2 && \
    pkg-config --modversion openslide && \
    pkg-config --modversion vips

COPY docker/scripts/ /opt/bench/

WORKDIR /data
ENTRYPOINT ["/opt/bench/entrypoint.sh"]
