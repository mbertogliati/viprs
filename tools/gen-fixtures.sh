#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VIPS_BIN="${VIPS_BIN:-$(command -v vips || true)}"
CURL_BIN="${CURL_BIN:-$(command -v curl || true)}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python || command -v python3 || true)}"

# libvips handles all geometric transforms and standard format writes. The local
# libvips build exposes openexrload but not openexrsave, so EXR fixtures are
# emitted from vips raw bytes via Python OpenEXR.

WORK_DIR="$ROOT_DIR/tools/.fixture-work"
DOWNLOAD_DIR="$WORK_DIR/downloads"
RGB_DIR="$WORK_DIR/rgb"
RAW_DIR="$WORK_DIR/raw"
VIPS_DIR="$WORK_DIR/vips"

require_cmd() {
  local name="$1"
  local path="$2"
  if [[ -z "$path" ]]; then
    echo "missing required tool: $name" >&2
    exit 1
  fi
}

require_python_modules() {
  "$PYTHON_BIN" - <<'PY'
from importlib.util import find_spec
missing = [name for name in ("OpenEXR", "Imath", "numpy") if find_spec(name) is None]
if missing:
    raise SystemExit(
        "missing python modules: "
        + ", ".join(missing)
        + ". Install them with: python -m pip install OpenEXR Imath numpy"
    )
PY
}

cleanup() {
  rm -rf "$WORK_DIR"
}

trap cleanup EXIT

require_cmd "vips" "$VIPS_BIN"
require_cmd "curl" "$CURL_BIN"
require_cmd "python" "$PYTHON_BIN"
require_python_modules

mkdir -p "$DOWNLOAD_DIR" "$RGB_DIR" "$RAW_DIR" "$VIPS_DIR"

download_kodak() {
  "$CURL_BIN" -L --fail --silent --show-error \
    https://r0k.us/graphics/kodak/kodak/kodim23.png \
    -o "$DOWNLOAD_DIR/kodim23.png"
  "$CURL_BIN" -L --fail --silent --show-error \
    https://r0k.us/graphics/kodak/kodak/kodim05.png \
    -o "$DOWNLOAD_DIR/kodim05.png"
}

save_copy() {
  local src="$1"
  local dst="$2"
  local opts="$3"
  "$VIPS_BIN" copy "$src" "${dst}${opts}"
}

write_exr_from_png() {
  local src_png="$1"
  local dst_exr="$2"
  local width="$3"
  local height="$4"
  local raw_path="$RAW_DIR/$(basename "$dst_exr").raw"

  "$VIPS_BIN" rawsave "$src_png" "$raw_path"

  RAW_PATH="$raw_path" WIDTH="$width" HEIGHT="$height" DST_PATH="$dst_exr" "$PYTHON_BIN" - <<'PY'
import os

import Imath
import OpenEXR
import numpy as np

raw_path = os.environ["RAW_PATH"]
width = int(os.environ["WIDTH"])
height = int(os.environ["HEIGHT"])
dst_path = os.environ["DST_PATH"]

pixels = np.fromfile(raw_path, dtype=np.uint8).reshape((height, width, 3))
header = OpenEXR.Header(width, height)
pixel_type = Imath.PixelType(Imath.PixelType.HALF)
header["channels"] = {channel: Imath.Channel(pixel_type) for channel in "RGB"}

scale = np.float16(1.0 / 255.0)
exr_file = OpenEXR.OutputFile(dst_path, header)
exr_file.writePixels(
    {
        channel: (pixels[:, :, index].astype(np.float16) * scale).tobytes()
        for index, channel in enumerate("RGB")
    }
)
exr_file.close()
PY
}

create_square_sources() {
  "$VIPS_BIN" crop "$DOWNLOAD_DIR/kodim23.png" "$RGB_DIR/kodim23_square.png" 128 0 512 512
  "$VIPS_BIN" crop "$DOWNLOAD_DIR/kodim05.png" "$RGB_DIR/kodim05_square.png" 128 0 512 512
  "$VIPS_BIN" flip \
    "$RGB_DIR/kodim23_square.png" \
    "$RGB_DIR/kodim23_square_flip.png" \
    horizontal
  "$VIPS_BIN" rot "$RGB_DIR/kodim05_square.png" "$RGB_DIR/kodim05_square_rot.png" d90
  "$VIPS_BIN" arrayjoin \
    "$RGB_DIR/kodim23_square.png $RGB_DIR/kodim05_square.png $RGB_DIR/kodim23_square_flip.png $RGB_DIR/kodim05_square_rot.png" \
    "$RGB_DIR/master_square_1024.png" \
    --across 2

  "$VIPS_BIN" resize "$RGB_DIR/master_square_1024.png" "$RGB_DIR/bench_512x512_rgb.png" 0.5
  "$VIPS_BIN" copy "$RGB_DIR/master_square_1024.png" "$RGB_DIR/bench_1024x1024_rgb.png"
  "$VIPS_BIN" resize "$RGB_DIR/master_square_1024.png" "$RGB_DIR/bench_2048x2048_rgb.png" 2
  "$VIPS_BIN" resize "$RGB_DIR/master_square_1024.png" "$RGB_DIR/bench_8192x8192_rgb.png" 8
}

create_wide_source() {
  "$VIPS_BIN" arrayjoin \
    "$DOWNLOAD_DIR/kodim23.png $DOWNLOAD_DIR/kodim05.png" \
    "$RGB_DIR/wide_strip.png" \
    --across 2
  "$VIPS_BIN" resize "$RGB_DIR/wide_strip.png" "$RGB_DIR/wide_999x333.png" 0.650390625
  "$VIPS_BIN" crop "$RGB_DIR/wide_999x333.png" "$RGB_DIR/bench_777x333_rgb.png" 111 0 777 333
}

generate_standard_family() {
  local size="$1"
  local src_png="$RGB_DIR/bench_${size}x${size}_rgb.png"
  local prefix="$ROOT_DIR/tests/fixtures/images/bench_${size}x${size}"

  save_copy "$src_png" "${prefix}.jpg" "[Q=90,strip]"
  save_copy "$src_png" "${prefix}.png" "[compression=9,strip]"
  save_copy "$src_png" "${prefix}.webp" "[Q=90,strip]"
  save_copy "$src_png" "${prefix}.avif" "[Q=70,strip]"
  save_copy "$src_png" "${prefix}.heic" "[Q=70,strip]"
  save_copy "$src_png" "${prefix}.jp2" "[Q=90,strip]"
  save_copy "$src_png" "${prefix}.jxl" "[Q=90,strip]"
  save_copy "$src_png" "${prefix}.tif" "[strip]"
  save_copy "$src_png" "${prefix}.gif" "[strip]"
  write_exr_from_png "$src_png" "${prefix}.exr" "$size" "$size"
}

generate_jpeg_only_fixtures() {
  save_copy \
    "$RGB_DIR/bench_1024x1024_rgb.png" \
    "$ROOT_DIR/tests/fixtures/images/bench_1024x1024.jpg" \
    "[Q=90,strip]"
  save_copy \
    "$RGB_DIR/bench_777x333_rgb.png" \
    "$ROOT_DIR/tests/fixtures/images/bench_777x333.jpg" \
    "[Q=90,strip]"
}

generate_derived_bench_variants() {
  for size in 512 2048 8192; do
    local src_png="$RGB_DIR/bench_${size}x${size}_rgb.png"

    "$VIPS_BIN" colourspace \
      "$src_png" \
      "$ROOT_DIR/tests/fixtures/images/bench_${size}x${size}_gray.png" \
      b-w

    "$VIPS_BIN" bandjoin_const \
      "$src_png" \
      "$ROOT_DIR/tests/fixtures/images/bench_${size}x${size}_rgba.png" \
      192

    "$VIPS_BIN" linear "$src_png" "$VIPS_DIR/bench_${size}_u16f.v" 257 0
    "$VIPS_BIN" cast "$VIPS_DIR/bench_${size}_u16f.v" "$VIPS_DIR/bench_${size}_u16.v" ushort
    save_copy \
      "$VIPS_DIR/bench_${size}_u16.v" \
      "$ROOT_DIR/tests/fixtures/images/bench_${size}x${size}_u16.tif" \
      "[compression=lzw,strip]"
  done

  "$VIPS_BIN" cast "$RGB_DIR/bench_512x512_rgb.png" "$VIPS_DIR/bench_512_f32.v" float
  save_copy \
    "$VIPS_DIR/bench_512_f32.v" \
    "$ROOT_DIR/tests/fixtures/images/bench_512x512_f32.tif" \
    "[strip]"

  save_copy \
    "$RGB_DIR/bench_512x512_rgb.png" \
    "$ROOT_DIR/tests/fixtures/images/bench_512x512_jpeg.tif" \
    "[compression=jpeg,Q=90,strip]"
  save_copy \
    "$RGB_DIR/bench_2048x2048_rgb.png" \
    "$ROOT_DIR/tests/fixtures/images/bench_2048x2048_lzw.tif" \
    "[compression=lzw,strip]"
}

generate_e2e_goldens() {
  "$VIPS_BIN" invert \
    "$ROOT_DIR/tests/fixtures/images/sample.png" \
    "$VIPS_DIR/png_invert.v"
  "$VIPS_BIN" rawsave "$VIPS_DIR/png_invert.v" "$ROOT_DIR/tests/fixtures/e2e/png_invert.bin"

  "$VIPS_BIN" flip \
    "$ROOT_DIR/tests/fixtures/images/sample.png" \
    "$VIPS_DIR/png_flip_horizontal.v" \
    horizontal
  "$VIPS_BIN" rawsave \
    "$VIPS_DIR/png_flip_horizontal.v" \
    "$ROOT_DIR/tests/fixtures/e2e/png_flip_horizontal.bin"

  "$VIPS_BIN" rot \
    "$ROOT_DIR/tests/fixtures/images/sample.png" \
    "$VIPS_DIR/png_rotate90.v" \
    d90
  "$VIPS_BIN" rawsave \
    "$VIPS_DIR/png_rotate90.v" \
    "$ROOT_DIR/tests/fixtures/e2e/png_rotate90.bin"

  "$VIPS_BIN" invert \
    "$ROOT_DIR/tests/fixtures/images/sample.jpg" \
    "$VIPS_DIR/jpeg_invert.v"
  "$VIPS_BIN" rawsave "$VIPS_DIR/jpeg_invert.v" "$ROOT_DIR/tests/fixtures/e2e/jpeg_invert.bin"
}

print_summary() {
  python - <<'PY'
from pathlib import Path

root = Path("tests/fixtures/images")
for name in [
    "bench_512x512.jpg",
    "bench_2048x2048.jpg",
    "bench_8192x8192.jpg",
    "bench_1024x1024.jpg",
    "bench_777x333.jpg",
]:
    path = root / name
    print(f"{name}\t{path.stat().st_size}")
PY
}

download_kodak
create_square_sources
create_wide_source
generate_standard_family 512
generate_standard_family 2048
generate_standard_family 8192
generate_jpeg_only_fixtures
generate_derived_bench_variants
generate_e2e_goldens
print_summary

echo "Regenerated benchmark fixtures and e2e goldens."
