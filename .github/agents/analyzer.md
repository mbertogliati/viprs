# ANALYZER.md — Invisible-Gap Detector

This document defines the workflow for an AI agent that audits the viprs codebase for
**invisible gaps**: features that appear implemented (API exists, tests pass) but carry
a hidden behavioral, memory, or correctness contract that the implementation does not
honour.

The agent produces issues in the project's standard format. It does NOT fix
anything. It only finds and documents.

---

## Friction protocol

**Any friction is reported immediately as a high-priority issue.**
An analyzer that struggles to navigate the codebase produces incomplete audits.

```bash
# create issue for the gap
  --priority high \
  -l friction \
  -d "## Friction Report

**Agent:** analyzer
**Audit pass:** Pass N — <pass name>
**Friction type:** <tooling | docs | codebase_navigation | environment>

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

Stop auditing that pass. File the friction task, then continue with the next pass if possible. Emit `AGENT_DONE status=blocked` if friction prevents completing the full audit.

---

## What makes a gap "invisible"

A gap is invisible when ALL of the following are true:

1. An API surface for the feature exists (struct, method, field, enum variant).
2. At least one test for that feature exists and passes.
3. The test verifies a **functional property** (output shape, value range, error type).
4. But the implementation violates a **deeper contract** the caller must rely on:
   - resource consumption (memory peak, allocations per tile)
   - behavioral fidelity to the reference (libvips in `.libvips_repo/`)
   - runtime guarantee (streaming vs. eager, thread safety)
   - completeness (only a subset of the input space is handled correctly)

The gap is invisible because feature search returns hits and `cargo test` reports green.

---

## Input contract

Before starting, read:

1. `GUIDELINES.md` — the full rule set.
2. `AGENTS.md` — non-negotiable rules and architectural invariants.
3. `src/` — the implementation.
4. `.libvips_repo/` — reference implementation (consult per-module, not in bulk).
5. Active issue tracker: `# list active tasks` — to avoid duplicate filings.

Do NOT use the issue tracker to guide what to look for. Read it once, only to deduplicate.

---

## The five audit passes

Run the passes in order. Each targets a different class of invisible gap.
File one issue per confirmed finding after all five passes.

---

### Pass 1 — Explicit debt markers

**Goal**: inventory all gaps that are documented in code but may lack a issue
or whose existing task underestimates severity.

**Steps**:

1. Search for inline markers across the full source tree:
   ```
   grep -rn "todo!\|// TODO\|// see B-\|// FIXME\|not yet\|deferred\|fallback\|approximat\|workaround" src/
   ```

2. For each hit, read the surrounding context (±20 lines) and answer:
   - What behavior does the marker say is missing or approximated?
   - What is the observable consequence for a caller that relies on the implied contract?
   - Does an open issue already capture this with the correct severity?

3. For `// see the task` markers, cross-reference against `# list active tasks`.
   If the referenced task exists but its description omits the caller consequence,
   note it as a severity-underestimate finding.

---

### Pass 2 — Allocation path audit

**Goal**: find code paths where the implementation allocates memory eagerly in a
context where the architecture promises lazy, bounded, or streaming behaviour.

**Core rules from CLAUDE.md**:
- Operations inside `domain/ops/` must not allocate on the heap per-pixel or per-tile.
- Pre-allocate buffers at pipeline construction time.
- Peak memory is `O(threads × tile_size)` on the pixel path.

**Steps**:

1. For each module in `src/adapters/` and `src/domain/ops/`, trace the execution path
   from its public entry point to the first pixel produced.

   For each path, answer:
   - Where does the first heap allocation occur?
   - What is the size of that allocation relative to the input (constant,
     proportional to tile size, proportional to full image size)?
   - Is the allocation consistent with the architecture's streaming contract?

2. Look for `Vec::new()`, `.collect()`, `.to_vec()`, `.clone()`, or calls to external
   crate decode/encode functions inside `process_region`. Each is a candidate violation.

3. For each module that accepts configuration options (`LoadOptions`, `SaveOptions`,
   or equivalent), verify that every field advertised as affecting resource usage
   (e.g., dimensions, scale, limits) actually affects resource usage at the allocation
   site — not only at a downstream transformation step.

4. For `src/adapters/codecs/`: verify that each codec's implementation of
   `decode_with_options` allocates memory consistent with what the options imply.
   If an option promises to reduce resource consumption, check that the reduction
   happens at decode time, not after a full allocation has already been made.

---

### Pass 3 — Reference parity audit

**Goal**: for each implemented operation, verify that the algorithm and its defaults
match libvips, not just the output shape.

**Steps**:

1. List all ops in `src/domain/ops/` with a non-stub `process_region`.

2. For each op, locate the counterpart in `.libvips_repo/`:
   - Arithmetic: `.libvips_repo/libvips/arithmetic/`
   - Colour: `.libvips_repo/libvips/colour/`
   - Convolution/Morphology: `.libvips_repo/libvips/convolution/`
   - Resample: `.libvips_repo/libvips/resample/`
   - Codecs: `.libvips_repo/libvips/foreign/`

3. For each pair, compare:
   - **Default parameter values**: does viprs use the same defaults as libvips?
     A silent default divergence affects every caller that omits the parameter.
   - **Edge/boundary handling**: how are out-of-bounds coordinates treated?
     Clamp, wrap, mirror, zero? Does viprs match libvips for the default mode?
   - **Overflow and saturation**: for integer formats, do both clamp at the same point?
   - **Alpha handling**: does viprs apply premultiplication / unpremultiplication at
     the same points as libvips?
   - **Decomposition strategy**: for composite ops, does viprs decompose into the same
     sub-operation sequence? A different decomposition may produce identical output on
     clean inputs but diverge on edge cases.

---

### Pass 4 — Invariant enforcement audit

**Goal**: find rules from CLAUDE.md / GUIDELINES.md that are maintained by convention
but not enforced by the type system or compiler.

**Steps**:

1. Extract all "must / never / always" rules from `CLAUDE.md` and `GUIDELINES.md`.

2. For each rule, classify:
   - **Compiler-enforced**: the type system makes the rule impossible to violate.
   - **Lint-enforced**: `clippy` or a custom rule would catch a violation.
   - **Convention-only**: nothing automated catches a violation.

3. For each convention-only rule, search the codebase for existing violations:
   - "no `dyn Trait` on hot paths" → `grep -rn "dyn " src/domain/`
   - "no `unwrap`/`expect` outside tests" →
     `grep -rn "\.unwrap()\|\.expect(" src/` (exclude `#[cfg(test)]` blocks)
   - "errors must be typed" → `grep -rn "Box<dyn" src/`
   - "no heap alloc in pixel path" → covered by Pass 2.

4. For each convention-only rule with no current violation, assess where a future
   violation would be easiest to introduce accidentally. Note as architectural risk.

---

### Pass 5 — Test honesty audit

**Goal**: find tests that pass for a reason other than the feature being correct.

**Core principle from CLAUDE.md**: a test that passes because of a trivially satisfied
condition is worse than no test — it produces false confidence.

**Steps**:

1. For each test, read the name and the set of assertions.

2. Ask: if the feature being tested were broken in a non-obvious way, would this test
   still pass? Specifically:

   - Does the test only assert output **dimensions**, not output **values**?
   - Does it only assert `Result::is_ok()` without inspecting the value?
   - Does it use tolerance bounds wide enough to pass with a completely wrong algorithm?
   - Does it use a synthetic input that is a fixed point of many possible algorithms
     (e.g., all-zeros, constant colour, identity transform)?
   - Does its name imply end-to-end correctness but the input cannot distinguish
     correct from incorrect behaviour?
   - Does it verify a **functional** property (shape, no-panic) while the feature's
     contract is primarily a **non-functional** one (resource consumption, latency,
     streaming behaviour)?

3. For each codec: verify that at least one test checks pixel **values** after a
   round-trip that would be disrupted by a bug in the specific option under test —
   not merely that dimensions are preserved.

---

## Output format

File one issue per confirmed finding:

All task titles and descriptions must be written in English.

```bash
# create issue for the gap
  --priority <high|medium|low> \
  -l <label> \
  -d "What it is: description of the violated property.

Gap class: <Allocation path | Reference parity | Convention invariant | Test honesty | Explicit debt>

Evidence: <file:line or search pattern that demonstrates it>

Why it is invisible: why 'cargo test' passes green despite the gap.

Caller consequence: what a reasonable caller assumes that this implementation does not guarantee.

Refactor difficulty: estimate of how much harder this becomes to fix as more code is built on top.

Acceptance criteria:
- [ ] The criterion verifies the deep property, not only the functional one."
```

**Priority**:
- `high` — violates a resource or correctness guarantee on inputs a production caller
  would send.
- `medium` — diverges from libvips in non-default configurations, or a test passes
  for wrong reasons on an implemented feature.
- `low` — architectural risk with no current violation, or reference parity divergence
  only observable on edge inputs.

Do not file tasks for gaps already in the issue tracker with accurate priority and
description. Do file tasks where the existing entry underestimates severity or omits
caller consequence.

---

## Tooling notes

- **`grep | head` exit codes**: when checking for *absence* of matches, `grep ... | head -N`
  exits 141 (SIGPIPE) when the head limit is reached before EOF, and exits 1 when grep finds
  no matches. Both look like failures. Use `grep ... | head -N || true` to tolerate these,
  or capture output first (`output=$(grep ...); [ -n "$output" ] && echo "$output" | head -N`).
- **`cargo check --examples | grep …`**: run `cargo check --examples 2>&1` first; grep the
  captured output. If the build is clean, grep exits 1 (no matches) — that is the success case.
  Use `|| true` or check with `[ $? -le 1 ]`.
- **`cargo test` accepts only one positional filter**: `cargo test filter1 filter2` fails with
  "unexpected argument". Run one filter per invocation. To check several targets, loop:
  `for f in proptest_identity boundary_value; do cargo test --lib "$f"; done`.

---

## What the agent must NOT do

- Fix any code.
- Read ADRs proactively (only when a specific one is referenced by name).
- Assume a feature is complete because its API exists and tests pass.
- Assume a feature is broken because a `// see the task` comment exists — assess
  independently whether the current implementation satisfies the caller's contract.
- File duplicate tasks.
