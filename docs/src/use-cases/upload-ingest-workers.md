# Upload And Ingest Workers

Upload and ingest workers process images after an upload completes. They usually read an
original from disk or object storage, generate derivatives, extract metadata, validate
content, and write results back to storage.

This environment values throughput, idempotence, and predictable resource use. A worker
should tolerate corrupt files, retry transient infrastructure failures, and avoid loading
huge originals into memory when only small derivatives are needed.

`viprs` should support this environment with:

- Clear source and sink abstractions for files, memory, and object storage adapters.
- Batch-friendly APIs that can reuse scheduler and buffer configuration.
- Progress and cancellation hooks for long-running jobs.
- Recoverable error categories for retry and dead-letter decisions.
- Recipes for common derivative sets such as thumbnails, previews, and optimized
  display images.

The lower-level pipeline should remain accessible because ingest systems often need to
make storage, metadata, and retry behavior part of the contract.
