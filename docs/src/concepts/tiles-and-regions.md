# Tiles And Regions

Tiles and regions are the vocabulary of partial image evaluation.

A region describes an area of an image. A tile is a concrete buffer containing pixels for
one region. Operations should work over tile buffers and slices rather than allocating
new vectors inside pixel loops.

This model matters because production images can be too large to process as a single
buffer. It also gives the scheduler units of work that can be evaluated across threads.

The repository keeps these concepts in `src/domain/` because they are core domain types,
not infrastructure details. Infrastructure code such as file sources, codecs, and thread
pools lives behind ports and adapters.

When contributing operations, prefer APIs that make region requirements explicit. Hidden
whole-image assumptions usually become memory or correctness bugs later.
