@AGENTS.md

---

# Working-style instructions — Spec-Decode Confidence Gating Research

These rules are session-derived from the project owner's explicit
instructions for the spec-decode confidence-gating work. They override
defaults; they do not override AGENTS.md.

Project memory and findings: `Memory.md`.
Plan file: `/root/.claude/plans/prepended-section-0-with-foamy-pearl.md`.

---

## Don't come to conclusions before checking

The most common failure mode is writing code or reporting a number
before the underlying premise has been verified. Concrete rules:

1. **No engineering before data.** The whole project is gated on
   whether `P(accept | gap)` has signal on Impala's actual workloads.
   That is a *data question*, not a code question. The first ~50 LOC
   are passive instrumentation that dumps `(logit_gap, accept_outcome)`
   tuples through the existing path. **No gate logic, no scheduler
   changes, no kernels are written until offline calibration on real
   traces clears decision gate G0.**

2. **Re-verify file:line anchors after every upstream merge.** The
   v0.20 fast-forward in this session moved hundreds of files; every
   anchor in the original plan had to be re-found. Any plan written
   against an older `HEAD` is suspect until a re-read confirms it.
   Don't trust your own previous notes blindly across a `git merge`.

3. **Separate exploration from solutioning.** When asked "what could
   break?", report failure modes with file:line evidence. Do *not*
   propose fixes in the same response unless asked. The owner asked
   for the failure list first, then "how would you fix this" as a
   second step. That ordering is intentional — premature fixes
   anchor the discussion before the problem space is mapped.

4. **Don't claim numbers you didn't measure.** Performance uplift,
   acceptance rate, throughput regression — never cite a figure unless
   it came from an actual run. On this machine there is no GPU, so
   *every* performance claim must come from an operator-run benchmark
   reported back. The PR description carries the numbers, the AI
   assistant does not.

5. **Honest negative results beat engineered-around problems.** If
   G0 fails (oracle bound <10%), the project ends. The kill condition
   exists for a reason. Failing a gate is the right outcome, not a
   problem to engineer around by adding features (entropy, mixture
   models, etc.) until something looks promising.

6. **Trust but verify subagents.** When a subagent claims a file
   exists or a function has a particular signature, open the file and
   confirm before relying on it. This session caught one stale claim
   (`NotImplementedError` for full draft-model — actually implemented
   in v0.20).

---

## How to test

Four orthogonal test layers, run in this order, each catching a
different class of bug:

### Layer 1 — Offline replay (CPU, every commit, seconds)
The gate is a pure function on `(gap, num_draft_tokens) -> per-request k`.
Exercise it against the dumped traces from Step 1a. Bitwise-identical
decisions to the offline calibration notebook. Lives in
`tests/v1/spec_decode/test_gate_unit.py` (to be added). **CI-blocking
on every commit.**

### Layer 2 — Synthetic single-batch integration (GPU, mandatory pre-merge)
**The single most important correctness check.** Do not ship without
it.

- With `gate_threshold = -inf`: outputs must be **bitwise identical**
  to non-gated baseline (gate is a no-op).
- With `gate_threshold = +inf`: outputs must be bitwise identical to
  non-spec baseline (gate trims everything; we fall through to base
  model).

Failure here means the verification path is broken under ragged `k`.
Block the merge.

### Layer 3 — E2E benchmarks (GPU, integration branch only)
Three workloads: ShareGPT (chat), HumanEval (code), internal agentic
trace. The gated config must beat the **strongest** of:

- static `k=4`
- static `k=8`
- static tree (`speculative_token_tree` configured)

Beating only the weakest baseline is not shippable. v1 of vLLM does
not have an adaptive (EAGLE-2-style) tree, so adaptive-tree is not in
the baseline set today.

### Layer 4 — 24h soak (GPU, before ops-on-by-default)
- **Calibration drift**: per-bin gate-fire rate stays within 2σ of the
  offline distribution.
- **Tail latency**: p99 within budget.
- **CUDA-graph cache bloat**: variant count bounded; eviction working.

### Blind-implementation discipline (no GPU here)

Every change classifies into one of three buckets, and the bucket
determines the proof-of-correctness shipped with the commit:

- **Bucket A — CPU-validatable** (gate logic, telemetry, calibration
  analysis): ships with green CPU-only CI.
- **Bucket B — GPU-required, behavior-deterministic** (worker hooks,
  drafter return-value changes, metadata extensions): ships with
  aggressive `assert tensor.shape == ...` and `assert tensor.dtype ==
  ...` lines plus a one-page `GPU_RUN_PLAN.md` listing exact commands,
  expected outputs, abort conditions. Layer-2 bitwise tests are
  mandatory.
- **Bucket C — GPU-required, performance-only** (CUDA-graph variants,
  Triton kernel fold-in, soak): designed but not implemented until
  Bucket B has shown correctness on real hardware. Isolated commits,
  revertable independently.

A `GPU_RUN_PLAN.md` accompanies any Bucket-B or Bucket-C PR. The
operator dumps the gate-fire histogram on first canary; we re-validate
against the offline distribution before the PR is approved.

### Container strategy
- **Pure Python** (telemetry, gate, scheduler hooks): hotpatch the
  v0.20 container by bind-mounting `vllm/` over
  `/usr/local/lib/python3.12/dist-packages/vllm/`. Sub-second
  iteration. Confirmed install path at `docker/Dockerfile:360, 488`.
- **Kernel / `.so` changes**: full multi-stage rebuild. ~30-45 min
  cold, ~5-10 min incremental. Blackwell dev: pin
  `torch_cuda_arch_list='9.0;10.0'` (the upstream superset is at
  `docker/Dockerfile:152`).

---

## KPIs (decision gates)

Five gates, each a written stop-point. Failing a gate ends the
project at that phase rather than escalating sunk cost.

| Gate | Question | Pass | Stop |
|---|---|---|---|
| **G0** | Does `P(accept \| gap)` have signal? | Oracle uplift ≥10% on ≥2 of 3 workloads | <10% on all → kill |
| **G1** | Which workloads benefit? | Code+agentic clearly above; chat ≥ baseline | All workloads marginal → reduce scope to one model class |
| **G2** | Threshold + ladder pickable from data? | Calibration curve monotonic; replay uplift ≥80% of oracle bound | Curve non-monotonic, or chosen policy ≪ oracle → re-feature (entropy, top-k margin, calibrated logistic on `[gap, entropy, position]`) |
| **G3** | Online matches replay? | E2E throughput within ±15% of offline replay; bitwise-identical at `τ=−∞` and `τ=+∞` | Drift >15% → bug in integration, do not roll out |
| **G4** | Stable under load? | 24h soak: no calibration drift, p99 within budget, CUDA-graph cache stable | Drift, OOM, graph-cache thrash → kernel/graph rework |

### G0 threshold note

The plan as originally written sets G0 at ≥10% oracle bound. The risk
audit in `Memory.md §7` argues for **tightening G0 to ≥25% oracle
bound**, because realized uplift is the oracle bound multiplied
through several lossy factors:

```
realized  ≈  oracle
            × (drafter_share is low)         # gating doesn't trim drafter
            × (max(k_i) actually shrinks)    # not just mean
            × (graph cache holds up)         # ragged k is hostile
            × (sync cost amortized)          # gate decision is a barrier
            × (compute-bound regime)         # bandwidth-bound kills the premise
            × (gate floor doesn't bite)      # k_min preserves bonus
            × (workload has gap entropy)     # RAG/agentic concentrate
```

Two of those at 0.5 puts a 12% oracle bound below noise on a 24h soak.

### Secondary metrics (non-gate, for diagnostics)

- **Drafter time / step time ratio.** Can read off the existing
  `SpecDecodingLogging` throughput numbers in
  `vllm/v1/spec_decode/metrics.py:74-117`. If drafter is >40% of step
  time, gating uplift is bounded above by the verify share alone.
- **`max(k_i)` vs `mean(k_i)` per batch** post-gate. Verify-time
  savings track total-position reduction (close to mean), not max —
  but graph-cache pressure tracks max. Both must move favorably.
- **Bonus-token retention rate.** Gate floor `k_min ≥ 1` means we
  always speculate at least once. Should be ~100%; if it drops, the
  gate is over-trimming.
- **Per-rank trace duplication.** On TP>1 every rank logs; raw row
  counts must match `world_size`-deduplicated (or: only rank 0 logs).

---

## Always do this

- `git merge-base --is-ancestor HEAD <tag>` before merging upstream.
  Confirms a fast-forward is possible. Avoids a 6,000-commit
  conflict-mining exercise when none is needed.
- Run an explicit `python -c "import ast; ast.parse(open(...).read())"`
  syntax check on every modified file when no test runner is available
  locally. Catches careless typos before the commit.
- After any non-trivial change to a hot path, document in the commit
  message what is bitwise-identical when the feature is *off*. The
  off-state is the contract; the on-state is the optimization.
- When a Python list flows from CPU to GPU side metadata
  (`SpecDecodeMetadata.num_draft_tokens: list[int]` is the canonical
  example), check whether the consumer actually needs the CPU form or
  whether the GPU tensor (`cu_num_draft_tokens`) suffices. Lists force
  a sync; tensors do not.
- Use `.venv/bin/python` and `uv` per AGENTS.md. Never system
  `python3` or bare `pip`.

## Never do this

- Don't push directly to `main`. Use the designated branch for the
  task (`claude/research-spec-decode-gating-*` for this project).
- Don't claim a perf number without a benchmark run on real hardware.
  Operator runs benchmarks; we report what they reported.
- Don't disable a failing test to make CI green. The test is the
  product spec; the failure is information.
- Don't ship a commit that touches the spec-decode hot path without a
  Layer-2 bitwise-identity test attached.
- Don't trust an old plan after `git merge`. Re-verify anchors first.
- Don't engineer around a failed decision gate. Stop.
