# CDN And Edge Optimization

CDN and edge optimizers transform images on demand. A URL or request parameter set maps
to a deterministic transformation plan and cache key. Common outputs include responsive
resizes, AVIF, WebP, JPEG, and quality-tuned variants negotiated from the `Accept` header.

This environment is sensitive to binary size, cold start, memory ceilings, and latency.
It also needs strict parameter validation because public URLs can otherwise become an
unbounded compute surface.

`viprs` should support this environment with:

- Small feature-gated builds for only the required codecs.
- Canonical operation plans suitable for cache keys.
- Safe presets for dimensions, quality, and format negotiation.
- Early cancellation when clients disconnect.
- Benchmarks that report p50, p95, RSS, and allocation behavior for realistic requests.

The advantage of a native Rust implementation is deployment simplicity: an optimizer can
ship as a single service binary with selected codec support instead of depending on a full
system image stack.
