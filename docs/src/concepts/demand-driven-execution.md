# Demand-Driven Execution

Demand-driven execution means the output asks the pipeline for the pixels it needs, and
upstream stages compute only the required regions. This is the central idea inherited
from libvips.

In an eager image library, each operation often materializes a complete intermediate
image. That is simple but expensive for large images and chained transformations. In a
demand-driven pipeline, operations describe how to produce output regions from input
regions. Evaluation happens when a sink requests output.

The practical benefits are:

- Fewer full-image intermediate buffers.
- Lower peak memory for thumbnails and crops from large originals.
- More opportunities to parallelize independent tiles.
- Better fit for streaming and partial-output workflows.

The cost is that operation contracts must be precise. Each operation needs to describe
which input region is required for a given output region, and the scheduler needs enough
information to evaluate those regions safely.
