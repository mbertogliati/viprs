# Web Image Services

Web image services receive image bytes in a request and return transformed bytes in the
response. They are common in marketplaces, content platforms, CMS products, and apps with
user-generated media.

Typical work includes thumbnails, crops, resizes, format conversion, optimization,
watermarks, autorotation, and color profile handling. The service may see high
concurrency and sharp traffic spikes, so memory use and p95 latency matter more than a
single best-case benchmark.

`viprs` should make this environment easy by offering:

- A compact facade for bytes-in and bytes-out handlers.
- Request-level limits for maximum bytes, pixels, memory, and execution time.
- Typed errors that map cleanly to HTTP responses.
- Scheduler configuration that can coexist with the web framework runtime.
- Examples for frameworks such as `axum`, `actix-web`, or `hyper`.

The core promise is demand-driven processing without unnecessary intermediate buffers.
The service should decode and compute only what the requested output requires.
