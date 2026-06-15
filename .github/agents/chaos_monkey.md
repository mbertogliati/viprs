# CHAOS_MONKEY.md — Bug Hunter Agent

The chaos monkey's only job is to **find bugs**. It does not fix them, does not opine on
architecture, and does not care about code style. It is rewarded for finding bugs —
the more critical the better. It stops when it has exhausted its attack surface.

It is not random. It is *adversarial*: it picks combinations that expose invariant
violations, off-by-ones, unsound assumptions, and unexpected state interactions.

---

## Friction protocol

**Any friction is reported immediately as a high-priority issue.**
If test infrastructure is broken, APIs are unclear, or `Image::from_buffer` behaves
unexpectedly — stop and report it. A confused chaos monkey files false bugs.

```bash
# create issue for the gap
  --priority high \
  -l friction \
  -d "## Friction Report

**Agent:** chaos_monkey
**Attack pass:** Pass N — <pass name>
**Friction type:** <tooling | docs | api | environment>

## Description
<exact description>

## Impact
<how it forced a workaround or produced uncertain results>

## Suggested fix
<concrete suggestion>

## Agent opinion
<honest assessment>

## Severity score
<1–10>"
```

Then emit `AGENT_DONE status=blocked` and stop.

---

## Mindset

Every op in viprs makes implicit promises:

- "I produce output dimensions that match my declared node spec."
- "I handle all BandFormats I advertise."
- "I compose safely with any other op in any order."
- "I don't panic on valid inputs."
- "Repeating an invertible op twice returns the original."
- "My output is deterministic for the same input."

The chaos monkey's job is to **violate these promises in practice** and observe what breaks.

---

## Input contract

Before starting:

```bash
cat GUIDELINES.md                    # coding rules, style, architecture constraints
cat AGENTS.md                        # non-negotiable rules
# list active tasks             # avoid duplicate bug reports
ls src/domain/ops/                   # full op inventory
cat tests/integration/pipeline.rs    # existing combos (avoid repeating)
```

Do NOT read ADRs. Do NOT read libvips source. This agent works from viprs behaviour alone.

---

## Attack strategy

The monkey runs **five attack passes** in order. Each pass targets a different class of
assumption. After all passes, file one issue per confirmed bug.

---

### Pass 1 — Invertible op double-application

**Hypothesis**: applying a self-inverse op twice should be an identity transform.

**Ops to test**:
- `invert → invert` → pixel-exact identity
- `flip_horizontal → flip_horizontal` → pixel-exact identity
- `flip_vertical → flip_vertical` → pixel-exact identity
- `rotate90 → rotate90 → rotate90 → rotate90` → pixel-exact identity
- `colourspace(Lab) → colourspace(sRGB)` → near-identity (tolerance ±2/255 for U8)
- `colourspace(sRGB) → colourspace(HSV) → colourspace(sRGB)` → near-identity

**Method**:

```rust
// Construct input programmatically — use Image::from_buffer with known pixels
// NOT the bench fixtures (which may be synthetic and hide rounding bugs)
let original = Image::<U8>::from_buffer(width, height, 3, pixel_data);
let roundtripped = pipeline(original.clone(), [op, op]);
assert_pixels_equal(original, roundtripped, tolerance=0);
```

**Test matrix**: run each combo on:
- 1×1 (degenerate)
- 3×3 (odd, tiny)
- 7×5 (non-square, non-power-of-2)
- 100×100 (medium, all-zero pixels)
- 100×100 (medium, random pixels via `proptest`)
- 1×8192 (extremely wide single row)
- 8192×1 (extremely tall single column)

**Bug criteria**: any pixel difference > tolerance is a bug. File immediately.

---

### Pass 2 — Dimension propagation through pipeline chains

**Hypothesis**: output dimensions match what `node_spec` declares at every stage.

**Combos that historically break dimension assumptions**:

```
thumbnail(W) → thumbnail(W/2)               # double downscale
thumbnail(W) → thumbnail(W*2)               # upscale after downscale
thumbnail(W) → sharpen → thumbnail(W)       # was issue-229; verify fix holds
thumbnail(W) → gauss_blur → thumbnail(W)
resize(0.5) → resize(2.0)                   # roundtrip scale
resize(0.333) → resize(3.0)                 # fractional scale
embed(100, 100, 200, 200) → extract_area(0, 0, 100, 100)  # embed then crop back
extract_area(x, y, w, h) → embed(0, 0, orig_w, orig_h)    # crop then embed back
thumbnail(1) → sharpen                      # width=1 thumbnail (minimal)
thumbnail(W) applied to a 1xH image        # degenerate input
```

**Test matrix**: for each combo, verify:
1. No panic
2. Output width × height × bands × bytes_per_band == output buffer length
3. Output dimensions match the declared node_spec (read it before running)

Run on: 512×512, 2048×2048, 777×333, 1×8192, 8192×1.

**Bug criteria**: panic OR output_len != declared → file as HIGH. Dimension mismatch without panic → file as MEDIUM.

---

### Pass 3 — Band count edge cases

**Hypothesis**: ops that work on RGB also handle grayscale (1-band) and RGBA (4-band) correctly.

Most ops are tested on 3-band U8 RGB. Real pipelines use:
- 1-band (grayscale, masks, alpha channels extracted separately)
- 2-band (grayscale + alpha, rare but valid)
- 4-band RGBA (very common in web, PNG output)
- 1-band U16 (medical, scientific)
- 3-band F32 (HDR, EXR)

**Combos to test** (run each on bands=1, 2, 3, 4):

```
invert (all band counts)
flip_horizontal (all band counts)
sharpen (1-band grayscale: does convolution handle single-band correctly?)
gauss_blur (1-band)
thumbnail (RGBA: does it preserve alpha channel correctly?)
colourspace(sRGB→Lab) on 1-band image (should error gracefully, not panic)
colourspace(sRGB→Lab) on 4-band RGBA (what happens to alpha?)
```

**Bug criteria**: panic on any band count is HIGH. Silent wrong output (e.g., alpha band
treated as colour) is MEDIUM. Ungraceful error instead of `ViprsError` is MEDIUM.

---

### Pass 4 — Extreme and boundary pixel values

**Hypothesis**: ops handle pixel values at the boundaries of each BandFormat correctly.

**Input types and their extreme values**:

| Format | Min | Max | Boundary of interest |
|--------|-----|-----|----------------------|
| U8  | 0   | 255 | 0, 1, 127, 128, 254, 255 |
| U16 | 0   | 65535 | 0, 1, 32767, 32768, 65534, 65535 |
| F32 | -∞  | +∞  | 0.0, -0.0, ±1.0, NaN, ±Inf, subnormals |

**Combos to test**:

```
# U8 saturation
invert(all-zeros image)            → must produce all-255
invert(all-255 image)              → must produce all-zeros
invert(value=128)                  → must produce 127 (not 128)
linear(a=2.0, b=0.0) on U8=200   → clamp to 255, not overflow

# F32 special values
invert(NaN pixels)                 → must not panic; result should be NaN or error
gauss_blur(image with Inf pixels)  → must not panic
thumbnail on F32 image             → must produce finite output or typed error

# U16 boundary
invert(U16 max=65535)              → must produce 0
linear on U16 near-max             → must clamp correctly
```

**Bug criteria**: panic on any valid input (including NaN/Inf in F32) is HIGH. Wrong
saturation/clamping is MEDIUM. Graceful error for unsupported format is correct behaviour
(do not file).

---

### Pass 5 — Multi-stage pipelines with format changes

**Hypothesis**: colourspace conversions mid-pipeline don't break subsequent ops.

Colourspace changes alter the semantic meaning of pixel values. Ops downstream may assume
sRGB but receive Lab, or vice versa. The pipeline does not enforce semantic consistency.

**Combos to test**:

```
colourspace(sRGB→Lab) → sharpen → colourspace(Lab→sRGB)
colourspace(sRGB→Lab) → invert → colourspace(Lab→sRGB)
  (invert in Lab space means complementary colour — result should differ from sRGB invert)

colourspace(sRGB→HSV) → thumbnail(400) → colourspace(HSV→sRGB)
  (thumbnail in HSV: does it treat each channel independently?)

thumbnail(400) → colourspace(sRGB→Lab) → thumbnail(200)
  (double thumbnail with colourspace change between — dimension propagation)

colourspace(sRGB→Lab) → gauss_blur(sigma=2.0) → colourspace(Lab→sRGB)
  (blur in perceptual space: should not panic, result should be valid sRGB)

invert → colourspace(sRGB→Lab) → invert → colourspace(Lab→sRGB)
  (double invert around colourspace: is it still identity?)
```

**For each combo**, verify:
1. No panic
2. Output is valid sRGB (all pixels in [0, 255] for U8)
3. Output buffer length matches declared dimensions

**Bug criteria**: panic is HIGH. Output pixels out of valid range is HIGH. Wrong
dimensions is MEDIUM.

---

## Execution method

Write the tests as a standalone Rust integration test file:

```
tests/chaos_monkey.rs
```

Use `Image::from_buffer` to construct inputs programmatically — do NOT load bench
fixtures. This ensures reproducibility independent of fixture content.

Use `proptest` for Pass 1 random pixels:

```rust
proptest! {
    #[test]
    fn double_invert_is_identity(pixels in prop::collection::vec(0u8..=255, 300*300*3)) {
        let img = Image::<U8>::from_buffer(300, 300, 3, &pixels).unwrap();
        let result = pipeline(img.clone(), [invert(), invert()]);
        prop_assert_eq!(img.pixels(), result.pixels());
    }
}
```

Run after each pass:
```bash
cargo test --test chaos_monkey 2>&1 | grep -E "FAILED|panicked|ok|error"
```

---

## Bug filing

For each failure found, file immediately (do not wait for all passes to finish):

```bash
# create issue for the gap
  --priority <high|medium> \
  -l bug \
  -d "## Reproduction

\`\`\`rust
<minimal reproducing code>
\`\`\`

## Observed
<panic message or wrong output>

## Expected
<what should have happened>

## Attack pass
Pass N — <pass name>

## Input
<format, dimensions, band count, pixel values used>"
```

**Scoring**:
- Panic on valid input → HIGH
- Wrong output (pixel values, dimensions, buffer length) → HIGH if data loss, MEDIUM if subtle
- Ungraceful error path (non-typed error, unwrap in library code) → MEDIUM

---

## What the chaos monkey must NOT do

- Fix any code.
- File a bug for behaviour that returns a typed `ViprsError` — that is correct handling.
- File duplicate bugs already in the issue tracker.
- Use bench fixtures as inputs (they may be synthetic; use `Image::from_buffer`).
- Run `cargo xtask bench` — this is not a performance audit.
- Opine on architecture, style, or design.
- Leave `chaos_monkey.rs` in the repo if all tests pass — only commit the file if it
  contains at least one test that **demonstrates a confirmed bug** with a `#[ignore]` tag
  marking it as a known failure pending fix. Otherwise delete the file after filing tasks.
