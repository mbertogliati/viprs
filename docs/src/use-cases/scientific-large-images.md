# Scientific And Large Images

Scientific, medical, and geospatial systems often work with images that are too large to
load as one buffer. Examples include microscopy slides, satellite imagery, large TIFF
pyramids, floating-point scientific formats, and region-of-interest analysis.

This environment needs explicit control over regions, tiles, metadata, numeric precision,
and partial evaluation. A simple `open().thumbnail().save()` facade is useful for previews
but not enough for analysis workflows.

`viprs` should support this environment with:

- Public concepts for regions, tiles, and demand hints.
- Precise behavior for integer and floating-point band formats.
- Metadata preservation and format-specific compatibility notes.
- Windowed reads and pyramid generation.
- Tests that document compatibility with libvips for edge cases.

Correctness and reproducibility are as important as speed. Any difference from libvips
should be documented with the reason and the expected impact.
