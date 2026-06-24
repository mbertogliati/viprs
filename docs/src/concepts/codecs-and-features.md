# Codecs And Features

Codecs are infrastructure. They decode external image formats into domain images and
encode domain images back to external formats. Their traits belong in `src/ports/`, while
concrete implementations belong in `src/adapters/`.

Some formats can be supported with pure Rust crates. Others need mature native libraries
for performance, compatibility, or format coverage. `viprs` should make these choices
explicit through feature flags.

Feature flags let applications build only what they need:

- A web service may enable JPEG, PNG, WebP, and AVIF.
- An edge optimizer may choose the smallest possible codec set.
- A scientific tool may enable TIFF, OpenSlide, FITS, NIfTI, or EXR.
- A development build may avoid native dependencies that are not relevant to the task.

Codec errors should remain typed and specific enough for callers to decide whether a
failure is corrupt input, unsupported content, an external dependency problem, or an
infrastructure failure.
