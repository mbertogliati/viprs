#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VIPS_BIN="${VIPS_BIN:-/opt/homebrew/bin/vips}"
WORK_DIR="$ROOT_DIR/tests/fixtures/.golden-script-work"

if [[ ! -x "$VIPS_BIN" ]]; then
  echo "libvips CLI not found at $VIPS_BIN" >&2
  exit 1
fi

rm -rf "$WORK_DIR"
mkdir -p "$WORK_DIR"
trap 'rm -rf "$WORK_DIR"' EXIT

python3 - <<'PY'
from pathlib import Path
import struct

root = Path("tests/fixtures/.golden-script-work")
root.mkdir(parents=True, exist_ok=True)

base_u8 = bytearray()
rhs_u8 = bytearray()
signed_f32 = bytearray()
rhs_f32 = bytearray()
fractional_f32 = bytearray()
subtract_uniform_source_f32 = bytearray()
subtract_uniform_rhs_f32 = bytearray()
subtract_sequential_source_f32 = bytearray()
subtract_half_rhs_f32 = bytearray()

for y in range(8):
    for x in range(8):
        base = (x * 17 + y * 13 + 5) % 256
        rhs = ((x % 5) + 1) * 9 + (y % 4) * 3
        signed = x * 0.75 - y * 1.1 - 7.25
        fractional = x * 0.5 - y * 0.75 - 3.125

        base_u8.append(base)
        rhs_u8.append(rhs)
        signed_f32.extend(struct.pack("<f", signed))
        rhs_f32.extend(struct.pack("<f", rhs / 10.0))
        fractional_f32.extend(struct.pack("<f", fractional))

for value in [10.0] * 16:
    subtract_uniform_source_f32.extend(struct.pack("<f", value))

for value in [3.0] * 16:
    subtract_uniform_rhs_f32.extend(struct.pack("<f", value))

for value in range(1, 17):
    subtract_sequential_source_f32.extend(struct.pack("<f", float(value)))

for value in [0.5] * 16:
    subtract_half_rhs_f32.extend(struct.pack("<f", value))

(root / "base_u8.raw").write_bytes(base_u8)
(root / "rhs_u8.raw").write_bytes(rhs_u8)
(root / "signed_f32.raw").write_bytes(signed_f32)
(root / "rhs_f32.raw").write_bytes(rhs_f32)
(root / "fractional_f32.raw").write_bytes(fractional_f32)
(root / "subtract_uniform_source_f32.raw").write_bytes(subtract_uniform_source_f32)
(root / "subtract_uniform_rhs_f32.raw").write_bytes(subtract_uniform_rhs_f32)
(root / "subtract_sequential_source_f32.raw").write_bytes(subtract_sequential_source_f32)
(root / "subtract_half_rhs_f32.raw").write_bytes(subtract_half_rhs_f32)
PY

rawload_spec() {
  local raw_path="$1"
  local image_path="$2"
  local width="$3"
  local height="$4"
  local bands="$5"
  local format="$6"
  "$VIPS_BIN" rawload "$raw_path" "$image_path" "$width" "$height" "$bands" --format "$format"
}

rawload() {
  local raw_path="$1"
  local image_path="$2"
  local format="$3"
  rawload_spec "$raw_path" "$image_path" 8 8 1 "$format"
}

save_raw() {
  local image_path="$1"
  local fixture_path="$2"
  mkdir -p "$(dirname "$fixture_path")"
  "$VIPS_BIN" rawsave "$image_path" "$fixture_path"
}

cd "$ROOT_DIR"

rawload "$WORK_DIR/base_u8.raw" "$WORK_DIR/base_u8.v" uchar
rawload "$WORK_DIR/rhs_u8.raw" "$WORK_DIR/rhs_u8.v" uchar
rawload "$WORK_DIR/signed_f32.raw" "$WORK_DIR/signed_f32.v" float
rawload "$WORK_DIR/rhs_f32.raw" "$WORK_DIR/rhs_f32.v" float
rawload "$WORK_DIR/fractional_f32.raw" "$WORK_DIR/fractional_f32.v" float
rawload_spec "$WORK_DIR/subtract_uniform_source_f32.raw" "$WORK_DIR/subtract_uniform_source_f32.v" 4 4 1 float
rawload_spec "$WORK_DIR/subtract_uniform_rhs_f32.raw" "$WORK_DIR/subtract_uniform_rhs_f32.v" 4 4 1 float
rawload_spec "$WORK_DIR/subtract_sequential_source_f32.raw" "$WORK_DIR/subtract_sequential_source_f32.v" 4 4 1 float
rawload_spec "$WORK_DIR/subtract_half_rhs_f32.raw" "$WORK_DIR/subtract_half_rhs_f32.v" 4 4 1 float

"$VIPS_BIN" abs "$WORK_DIR/signed_f32.v" "$WORK_DIR/abs.v"
save_raw "$WORK_DIR/abs.v" "$ROOT_DIR/tests/fixtures/abs/ramp_signed.bin"

"$VIPS_BIN" add "$WORK_DIR/signed_f32.v" "$WORK_DIR/rhs_f32.v" "$WORK_DIR/add.v"
save_raw "$WORK_DIR/add.v" "$ROOT_DIR/tests/fixtures/add/ramp_plus_rhs.bin"

"$VIPS_BIN" subtract "$WORK_DIR/signed_f32.v" "$WORK_DIR/rhs_f32.v" "$WORK_DIR/subtract.v"
save_raw "$WORK_DIR/subtract.v" "$ROOT_DIR/tests/fixtures/subtract/ramp_minus_rhs.bin"

"$VIPS_BIN" subtract "$WORK_DIR/subtract_uniform_source_f32.v" "$WORK_DIR/subtract_uniform_rhs_f32.v" "$WORK_DIR/subtract_uniform_constant.v"
save_raw "$WORK_DIR/subtract_uniform_constant.v" "$ROOT_DIR/tests/fixtures/subtract/uniform_constant.bin"

"$VIPS_BIN" subtract "$WORK_DIR/subtract_sequential_source_f32.v" "$WORK_DIR/subtract_half_rhs_f32.v" "$WORK_DIR/subtract_sequential_minus_half.v"
save_raw "$WORK_DIR/subtract_sequential_minus_half.v" "$ROOT_DIR/tests/fixtures/subtract/sequential_minus_half.bin"

"$VIPS_BIN" multiply "$WORK_DIR/signed_f32.v" "$WORK_DIR/rhs_f32.v" "$WORK_DIR/multiply.v"
save_raw "$WORK_DIR/multiply.v" "$ROOT_DIR/tests/fixtures/multiply/ramp_times_rhs.bin"

"$VIPS_BIN" divide "$WORK_DIR/signed_f32.v" "$WORK_DIR/rhs_f32.v" "$WORK_DIR/divide.v"
save_raw "$WORK_DIR/divide.v" "$ROOT_DIR/tests/fixtures/divide/ramp_divided_by_rhs.bin"

"$VIPS_BIN" linear "$WORK_DIR/signed_f32.v" "$WORK_DIR/linear.v" 1.75 -- -2.5
save_raw "$WORK_DIR/linear.v" "$ROOT_DIR/tests/fixtures/linear/ramp_scale_1_75_offset_-2_5.bin"

"$VIPS_BIN" invert "$WORK_DIR/base_u8.v" "$WORK_DIR/invert.v"
save_raw "$WORK_DIR/invert.v" "$ROOT_DIR/tests/fixtures/invert/grayscale_ramp.bin"

"$VIPS_BIN" round "$WORK_DIR/fractional_f32.v" "$WORK_DIR/round.v" rint
save_raw "$WORK_DIR/round.v" "$ROOT_DIR/tests/fixtures/round/fractional_signed_ramp.bin"

"$VIPS_BIN" sign "$WORK_DIR/fractional_f32.v" "$WORK_DIR/sign.v"
save_raw "$WORK_DIR/sign.v" "$ROOT_DIR/tests/fixtures/sign/fractional_signed_ramp.bin"

"$VIPS_BIN" flip "$WORK_DIR/base_u8.v" "$WORK_DIR/flip_h.v" horizontal
save_raw "$WORK_DIR/flip_h.v" "$ROOT_DIR/tests/fixtures/flip_h/grayscale_ramp.bin"

"$VIPS_BIN" flip "$WORK_DIR/base_u8.v" "$WORK_DIR/flip_v.v" vertical
save_raw "$WORK_DIR/flip_v.v" "$ROOT_DIR/tests/fixtures/flip_v/grayscale_ramp.bin"

"$VIPS_BIN" rot "$WORK_DIR/base_u8.v" "$WORK_DIR/rotate90.v" d90
save_raw "$WORK_DIR/rotate90.v" "$ROOT_DIR/tests/fixtures/rotate90/grayscale_ramp_8x8.bin"

"$VIPS_BIN" rot "$WORK_DIR/base_u8.v" "$WORK_DIR/rotate180.v" d180
save_raw "$WORK_DIR/rotate180.v" "$ROOT_DIR/tests/fixtures/rotate180/grayscale_ramp.bin"

"$VIPS_BIN" rot "$WORK_DIR/base_u8.v" "$WORK_DIR/rotate270.v" d270
save_raw "$WORK_DIR/rotate270.v" "$ROOT_DIR/tests/fixtures/rotate270/grayscale_ramp.bin"

"$VIPS_BIN" embed "$WORK_DIR/base_u8.v" "$WORK_DIR/embed.v" 2 1 12 10 --extend black
save_raw "$WORK_DIR/embed.v" "$ROOT_DIR/tests/fixtures/embed/offset_black_12x10.bin"

"$VIPS_BIN" crop "$WORK_DIR/base_u8.v" "$WORK_DIR/extract_area.v" 1 2 4 3
save_raw "$WORK_DIR/extract_area.v" "$ROOT_DIR/tests/fixtures/extract_area/center_crop.bin"

"$VIPS_BIN" gaussblur "$WORK_DIR/fractional_f32.v" "$WORK_DIR/gauss_blur.v" 1.0
save_raw "$WORK_DIR/gauss_blur.v" "$ROOT_DIR/tests/fixtures/gauss_blur/fractional_ramp_sigma_1_0.bin"

echo "Generated libvips golden fixtures under tests/fixtures/"
