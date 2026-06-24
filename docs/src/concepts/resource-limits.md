# Resource Limits

Image processing libraries must defend production systems from unbounded input. A small
compressed file can expand to many pixels, and public transformation parameters can ask
for expensive work.

Resource limits should be first-class in `viprs`:

- Maximum input bytes.
- Maximum decoded pixels.
- Maximum output dimensions.
- Maximum memory budget.
- Maximum execution time or cancellation signal.
- Codec-specific limits for formats with unusual expansion behavior.

Typed errors are important here. A service needs to distinguish "unsupported format" from
"valid image rejected because it exceeds policy." The first may be a client error, while
the second may be an intentional product rule.

Limits also help performance work. Benchmarks are more meaningful when they run under the
same resource assumptions as production.
