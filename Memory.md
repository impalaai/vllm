# Memory.md — Spec-Decode Confidence Gating Research

Working notes for the speculative-decode confidence-gating project on the
`impalaai/vllm` fork. This file is the durable summary of research so
far; pick up from here.

> Branch: this Memory file lives on `claude/research-spec-decode-gating-memory`.
> Implementation work (v0.20 fast-forward + Step 1a instrumentation) lives on
> `claude/research-spec-decode-gating-5n0eP`.

---

## 1. Goal

Per-position dynamic gating of speculative decoding: at each draft
position, predict `P(accept | gap)` where `gap = top1_logit − top2_logit`
from the draft model, and skip verification compute on positions whose
predicted acceptance is below threshold. Hypothesis: low-confidence
draft positions waste verify FLOPs, so trimming them yields throughput
uplift over a static-`k` baseline.

The whole project is gated on whether `P(accept | gap)` has real
discriminative power on Impala's actual workloads — a data question,
answerable offline from passive instrumentation in days. **No gating
code ships until the offline calibration clears decision gate G0.**

Plan file (full structure, decision gates, blind-implementation
discipline, container strategy, per-model matrix):
`/root/.claude/plans/prepended-section-0-with-foamy-pearl.md`.

---

## 2. Decision gates (kill conditions)

| Gate | Question | Pass | Stop |
|---|---|---|---|
| G0 | Does `P(accept | gap)` have signal? | Oracle uplift ≥10% on ≥2 of 3 workloads | <10% on all → kill |
| G1 | Which workloads? | Code+agentic clearly above; chat ≥ baseline | All marginal → reduce scope |
| G2 | Threshold + ladder pickable? | Replay ≥80% of oracle | Curve non-monotonic → re-feature |
| G3 | Online matches replay? | E2E within ±15%; bitwise-identical at τ=±∞ | Drift >15% → bug, no rollout |
| G4 | Stable under load? | 24h soak: drift, p99, graph-cache OK | Any failure → rework |

**G0 should probably be tightened to ≥25% oracle bound** based on the
risk audit below — see §6.

---

## 3. What landed on `claude/research-spec-decode-gating-5n0eP`

### Step 0: fork advanced to vLLM v0.20.0

Clean fast-forward from `51d4126` → `88d34c6` (v0.20.0). 6,740 upstream
commits, zero impala-specific patches to preserve (verified pre-merge
with `git merge-base --is-ancestor HEAD v0.20.0`).

This pre-merge verification mattered: the original plan's anchors were
pinned to `51d4126`-era code, but v0.20 refactored the spec-decode
subsystem extensively. New layout:

- `vllm/v1/spec_decode/eagle.py` — gutted to a 22-line shell extending
  `SpecDecodeBaseProposer`.
- `vllm/v1/spec_decode/llm_base_proposer.py` — 1809 lines, the actual
  drafter logic.
- `vllm/v1/worker/gpu_model_runner.py` — gone, replaced by
  `vllm/v1/worker/gpu/model_runner.py` (1374 lines) plus a `gpu/`
  subtree (`spec_decode/`, `sample/`, `metrics/`, etc.).
- `vllm/config.py` — became `vllm/config/` package; SpeculativeConfig
  fields restructured.
- New drafters: `dflash.py`, `draft_model.py` (full draft-model path is
  now implemented; the `NotImplementedError` flag in earlier notes is
  stale), `suffix_decoding.py`, `ngram_proposer_gpu.py`.
- Two rejection samplers exist: original
  `vllm/v1/sample/rejection_sampler.py` (now `MAX_SPEC_LEN = 128`,
  raised from 32) and the canonical-for-GPU
  `vllm/v1/worker/gpu/spec_decode/rejection_sampler.py` (230 lines)
  with three modes: `strict`, `probabilistic`, `synthetic`.

### Step 1a: passive instrumentation (commit `08de8e1`)

Eight files, ~1k lines total, fully no-op when env var unset.

**New files:**
- `vllm/v1/spec_decode/_instrumentation.py` — `GapAcceptanceTracer`
  singleton, JSONL append-only shard writer, `compute_top1_top2_gap`
  helper.
- `tests/v1/spec_decode/test_instrumentation.py` — CPU-runnable unit
  tests (cannot execute locally; no torch/numpy/pytest in the dev env).
- `tools/spec_gate_calibrate.py` — offline calibration analysis script
  (numpy + pandas + scipy). Computes Spearman ρ, oracle uplift,
  realizable uplift, emits the G0 verdict.
- `tools/SPEC_GATE_TRACE_GPU_RUN_PLAN.md` — operator runbook for the
  GPU-side validation: bitwise-identity check, schema sanity assertions,
  overhead acceptance criterion, calibration dump procedure.

**Modified files:**
- `vllm/envs.py` — three new env vars:
  - `VLLM_SPEC_GATE_TRACE` (bool, default 0)
  - `VLLM_SPEC_GATE_TRACE_DIR` (default `/tmp/vllm-spec-gate-trace`)
  - `VLLM_SPEC_GATE_TRACE_FLUSH_EVERY` (int, default 4096)
- `vllm/v1/spec_decode/llm_base_proposer.py` —
  `_greedy_sample(hidden_states, position=0)` records gaps when tracer
  is on; called from `propose()` first-position site (line 493/516)
  with `position=0` and from inner-loop site (line 650) with
  `position=token_index+1`.
- `vllm/v1/spec_decode/medusa.py` — per-head gap recording before
  argmax stack.
- `vllm/v1/core/sched/scheduler.py` —
  `make_spec_decoding_stats` records per-request `(num_drafted,
  num_accepted)`; `update_from_output` calls `step_boundary()` once per
  step.

**Verified locally:** Python syntax on all modified files. **Not
verified locally:** import resolution, runtime behavior (no torch/numpy
in dev env). Hardware validation runbook in
`tools/SPEC_GATE_TRACE_GPU_RUN_PLAN.md`.

---

## 4. Critical file:line anchors (v0.20.0 verified)

For future implementation work in §§1–12 of the plan:

| Concern | File:line | Notes |
|---|---|---|
| Drafter logit interception (chain) | `vllm/v1/spec_decode/llm_base_proposer.py:407-411` | Single `_greedy_sample` site covers first + inner-loop |
| Drafter logit interception (tree) | `llm_base_proposer.py:504, 1143` | `propose_tree` separate sites |
| Inner-loop drafter call | `llm_base_proposer.py:554-655` | `for token_index in range(num_speculative_tokens - 1)` |
| Medusa interception | `vllm/v1/spec_decode/medusa.py:49-53` | List of logits, one per head |
| Rejection sampler (GPU) | `vllm/v1/worker/gpu/spec_decode/rejection_sampler.py:100-230` | Strict/probabilistic/synthetic modes |
| Rejection sampler kernel | same file, lines 23-77 | `_strict_rejection_sample_kernel` per-request loop |
| Rejection-sampler call site | `vllm/v1/worker/gpu/model_runner.py:896-898` | `self.rejection_sampler(...)` |
| SpecDecodeMetadata | `vllm/v1/spec_decode/metadata.py:9-27` | `num_draft_tokens: list[int]` (CPU), `cu_num_draft_tokens: torch.Tensor` (GPU), `cu_num_sampled_tokens: torch.Tensor` (new in v0.20) |
| Per-request stats hook | `vllm/v1/core/sched/scheduler.py:1370-1391, 1977-1994` | `make_spec_decoding_stats(..., request_id=req_id)` already plumbed |
| SpecDecodingStats | `vllm/v1/spec_decode/metrics.py:17-45` | `observe_draft(num_draft, num_accepted)` |
| Draft method dispatch | `llm_base_proposer.py:434-446` | `eagle3`, `dflash`, plus MTP variants |
| EAGLE3 hidden combine | `llama_eagle3.py:168-170` | Required for size-mismatched draft (e.g. Llama 70B) |
| MTP single-layer constraint | `llm_base_proposer.py:250-252` | Comment notes only one MTP layer supported |
| `use_local_argmax_reduction` | `llm_base_proposer.py:112-114, 407-410` | Skips logit return path |
| Parallel drafting | `llm_base_proposer.py:99-102, 492-494` | Single forward, all positions; gate-incompatible at the inner-loop level |
| Static tree drafting | `vllm/config/__init__.py:2228-2240`, `vllm/v1/attention/backends/tree_attn.py` | v1 has only static tree (not adaptive EAGLE-2) |
| CUDA graph dispatcher | `llm_base_proposer.py:142, 395-405` | `cudagraph_dispatcher.initialize_cudagraph_keys` — gate must coexist with this |

---

## 5. Drafter / target compatibility

Recovered from v0.20 registry inspection at `vllm/config/__init__.py:2050-2240`:

| Target | EAGLE/EAGLE3 | Medusa | MTP | MLP-Spec | n-gram | Phase |
|---|---|---|---|---|---|---|
| Llama 3.1/3.3 8B | ✓ | ✓ | ✗ | ✓ | ✓ | **1** |
| Llama 3.1/3.3 70B | ✓ (EAGLE3 + combine_hs) | ✓ | ✗ | — | ✓ | **1** |
| Llama 4 | `llama4_eagle.py` | — | — | — | ✓ | 2 |
| Qwen 2.5 / 3 | ✓ | ✓ | ✗ | — | ✓ | **1** |
| Qwen 3-Next | — | — | ✓ (`qwen3_next_mtp`) | — | ✓ | 2 |
| DeepSeek-V3 / R1 | `deepseek_eagle.py` | — | ✓ (`deepseek_mtp`) | — | ✓ | 2 |
| Mixtral 8x7B/8x22B | ✓ | ✓ | ✗ | — | ✓ | 2 |
| MiniCPM | `minicpm_eagle.py` | — | — | — | ✓ | 3 |
| Ernie 4.5 MoE | — | — | ✓ (`ernie_mtp`) | — | ✓ | 3 |
| MiMo / GLM-4 MoE | — | — | ✓ | — | ✓ | 3 |
| DFlash (Qwen3) | — (parallel-drafting path) | — | — | — | ✓ | 3 |

Full draft-model path (small model drafts large) **is** implemented in
v0.20 (`draft_model.py`), unlike earlier notes. Add to phase 3 if a
real Impala use case appears.

---

## 6. Risk audit — what breaks on real GPU

The instrumentation lives in two places that don't run locally — the
drafter forward and the scheduler. Concrete risks ranked by severity:

### Critical (block ship until fixed)

1. **D2H copy inside CUDA graph replay** —
   `_greedy_sample` runs *inside* the `set_forward_context(...,
   cudagraph_runtime_mode=...)` block at `llm_base_proposer.py:472-481`.
   Calling `gap.cpu().numpy()` inside graph replay is undefined
   behavior — either hangs or returns garbage. Must move the D2H to
   *after* graph context exits, or capture gap into a pre-allocated GPU
   tensor and only D2H once outside the captured region.
2. **Tensor-parallel logit sharding** —
   `compute_logits` returns vocab-sharded logits when TP>1
   (`vllm/model_executor/layers/logits_processor.py:75-99` does the
   gather *only in the sampling path*). My hook computes top-2 over a
   shard, not the full vocab — gap values are garbage on TP>1. Fix:
   gate the hook on `get_tensor_model_parallel_rank() == 0` AND ensure
   the full-gather has happened before the hook fires.

### High (fix before producing real calibration data)

3. **Duplicate rows on TP ranks** — every rank runs the drafter and
   records gaps. Trace bias toward over-confidence. Fix: rank-0-only
   recording.
4. **Parallel-drafting mode (DFlash, etc.)** — single forward predicts
   all K positions; my hook records `position=0` only. Schema expects
   per-position rows. Fix: extract per-head/per-position logits inside
   the parallel-drafting forward.

### Medium

5. **Async scheduling race** — `step_boundary` increments at end of
   `update_from_output`; with `use_async_scheduling=True`, the next
   step's drafter has already fired. Fix: drain async queue before
   step_boundary, or accept that step counter is approximate for
   async-mode traces.
6. **`log_stats=False` silent drop** — `make_spec_decoding_stats`
   early-returns when `log_stats=False`; my acceptance hook lives below
   that early return. With `log_stats=False`, gap rows are recorded
   without matching accept rows. Fix: move acceptance hook above the
   early return, or document the requirement.
7. **Thread safety in `record_gaps` / `record_acceptance`** — append
   without lock; flush takes lock. Multi-threaded executor could lose
   rows between append and flush. Fix: lock around append (cheap, the
   GIL serializes anyway).

### Low

8. **Pipeline-parallel rank gating** — drafter runs on last PP rank
   only; non-last ranks see no logits. Hook should `if not
   get_pp_group().is_last_rank: return`.
9. **Vocab padding** — TP/quantization may pad vocab to multiple of
   256. Top-2 over padded vocab is benign (padding rows have very
   negative logits); minor noise floor.
10. **Performance overhead** — `topk(2, vocab=128k)` on `[batch, vocab]`
    is ~free on GPU. The `.cpu()` is a sync barrier; budget 1-3%
    slowdown at default flush rate. Acceptable; document in PR.

The full GPU run plan (`tools/SPEC_GATE_TRACE_GPU_RUN_PLAN.md`)
encodes the bitwise-identity check that catches risks 1, 2, 7, 8 at
operator-run time before the trace is trusted.

---

## 7. What could negate the gating uplift itself

Even with the instrumentation correct, the project's premise has
multiple ways to fail at deploy time. Severity-ranked:

1. **Drafter compute is unchanged by post-hoc gating.** Step time =
   drafter + verify + reject. Gating only trims verify. If drafter is
   30% of step time (typical for non-trivial EAGLE drafters), 50%
   verify savings → 35% step savings → 50% throughput uplift, not the
   100% the verify-savings ratio implied. **Fix: gate inside the
   drafter inner loop at `llm_base_proposer.py:554`, breaking early
   when threshold isn't met. This saves both drafter and verify
   compute.**
2. **Memory-bandwidth-bound regime.** Spec-decode wins by converting
   memory-bound autoregressive decode into compute-bound parallel
   verify. Gating reduces FLOPs but not bytes-loaded; on low-batch
   large-model serving, throughput is bandwidth-bound and gating saves
   ≈zero. **Fix: detect regime at startup (use `SpecDecodingLogging`
   throughput numbers in `vllm/v1/spec_decode/metrics.py:74-117`); log
   a warning and leave gate off when bandwidth-bound.**
3. **CUDA-graph cache thrash.** Per-request ragged `k` adds a second
   cardinality axis to the graph cache key. Either capture variants
   (eviction churn) or fall back to eager (lose graph speedup, often
   ~30% of throughput). **Fix: bucket `k ∈ {0, 1, 2, 4, 8}` and capture
   five graph variants per batch-size bucket. Lose ~5-10% theoretical
   uplift to bucketing rounding; keep graph speedup.**
4. **Gate-decision sync barrier.** Computing gap on GPU, copying to
   CPU, building per-request `k_i` list: ~50-200 µs round-trip — 5-10%
   of step time on small batches. **Fix: keep `cu_num_draft_tokens` on
   GPU end-to-end; replace `num_draft_tokens: list[int]` with a lazy
   property that only syncs when actually consumed (Triton kernel
   already takes the GPU tensor).**
5. **Probabilistic vs strict rejection mode.** Strict rejection
   (`vllm/v1/worker/gpu/spec_decode/rejection_sampler.py:166-174`) is
   index-ops, ~free. Probabilistic mode does a real GEMM via
   `processed_logits = sampler.apply_sampling_params(...)`. Gating
   helps probabilistic mode disproportionately. **G0 must be measured
   in the rejection mode that production uses.**
6. **Bonus-token loss.** If gate trims `k=0` (low gap on first
   position), no bonus token, no spec uplift for that request.
   **Fix: gate floor `k_min = 1`. One line.**
7. **KV-cache slots are pre-gate.** Scheduler allocates slots for full
   `num_speculative_tokens` regardless. Gating doesn't improve
   concurrency; only wallclock. Document.
8. **Workload concentration.** RAG/agentic workloads: 80% of requests
   share a prompt template → near-degenerate gap distribution → no
   information for the gate. **Most likely G0 failure mode in
   production. Not engineerable.**
9. **Correlated outcomes.** `P(accept | gap)` is a marginal; per-
   request acceptance is correlated. Threshold tuned on the marginal
   gates poorly on the bimodal mixture. **Fix: include `last_accept`
   feature; train a 3-feature logistic head `(prob_margin, position,
   last_accept)`.**
10. **Logit-space `gap` doesn't transfer across temperatures.** **Fix:
    use probability margin `softmax(top1) - softmax(top2)` instead of
    raw logit gap. Calibrated [0,1] feature, transfers across
    workloads. Same hot-path cost (two extra divisions).**

### Architectural reframe

The plan's §5 ("worker hook after `propose_draft_token_ids` returns")
is wrong given fix #1. The gate belongs **inside** `propose()`, between
inner-loop iterations:

```python
# llm_base_proposer.py, around line 650
last_hidden_states, hidden_states = ret_hidden_states
next_draft, gap = self._greedy_sample_with_gap(last_hidden_states[:batch_size])
draft_token_ids_list.append(next_draft)
if self.gate_policy != "off" and not gate.should_continue(gap, request_state).any():
    break  # batch-wide early exit; saves drafter forwards too
```

This collapses risks 1, 3 (because K decreases batch-wide → graph
shapes are bounded by `num_speculative_tokens` choices, not k-
histograms), and partially 4 (the inner loop already syncs at
`input_ids = draft_token_ids_list[-1].int()` line 558; gate piggy-
backs on the existing barrier).

§6 ("Python rejection mask first") becomes unnecessary — the gate
produces ragged `k` that the existing Triton kernel already handles.

§7 ("CUDA-graph compatibility") simplifies from "capture per
k-histogram" to "capture per K-bucket" because of fix #3.

Net effect: gate is ~80 LOC class + 30 LOC integration, not the
multi-section plan I wrote.

### What to skip

- Don't fix the bandwidth-bound regime (#2) by engineering. Detect and
  disable.
- Don't fix async scheduling (#10 from §6 above) by serializing.
  Disable async when gate is on.

---

## 8. Open work

Resume sequence:

1. **Operator runs `tools/SPEC_GATE_TRACE_GPU_RUN_PLAN.md` Steps 1-3**
   on a Llama-3.1-8B EAGLE config:
   - Step 1: bitwise-identity baseline (mandatory, blocks merge if
     non-empty diff).
   - Step 2: schema sanity on emitted JSONL shards.
   - Step 3: overhead measurement, target ≤5% throughput drop.
2. **Apply risk fixes 1-4 from §6** (CUDA graph context, TP rank
   gating, parallel-drafting positions, log_stats placement) before
   producing real calibration data.
3. **Fix the feature**: replace logit gap with probability margin
   (`softmax(top1) - softmax(top2)`) before the trace dump. The G0
   number will be more representative.
4. **24h trace dump** across {chat, code, agentic} workloads.
5. **Run `tools/spec_gate_calibrate.py`**, get the G0 verdict per
   workload.
6. **Decision point.** If G0 fails → write up the negative result and
   stop. If G0 passes → proceed to §§1–12 with the architectural
   revision in §7 (gate inside `propose()`, GPU-resident
   `cu_num_draft_tokens`, K-bucketing for graphs).

If §§1–12 begin: the §1 config surface, §3 gate module,
§4 drafter integration, and §5 worker hook should all be **revised**
per the architectural reframe in §7 above. The plan file's appendix
notes the v0.20 anchor corrections; the §7 reframe in this Memory file
is the more important correction.

---

## 9. Knowns to revisit

- **Per-request request_id in the drafter hook.** Today the drafter
  records `(step, batch_idx, position, gap)`; offline join is
  approximate (per-step aggregate, not per-request). For the
  per-(request, position) granularity the original plan called for, we
  need to plumb `request_ids` from the input batch into the drafter
  forward. Not blocking for G0; blocking for the per-request feature
  in fix #9 above.
- **Multi-process correlation.** Ray executor spawns workers in
  separate processes; each gets its own tracer singleton with
  process-local step counter. Calibration script currently treats
  shards independently. Cross-process join is a follow-up; document as
  known limitation.
- **Tree-drafting hooks.** `propose_tree` at `llm_base_proposer.py:990,
  1143` is not hooked. Tree-drafted runs miss draft rows for tree
  positions. Out of scope for chain-drafting calibration; required
  before measuring tree configurations.
- **MoE / EPLB interaction.** Gating perturbs the sequence of tokens
  hitting the MoE gate. With `enable_eplb=True`, may trigger more
  rebalances. Doesn't kill uplift directly but adds variance. Worth
  measuring on Mixtral / DeepSeek-V3 / Qwen3-Next configs specifically.

---

## 10. References

- Plan file: `/root/.claude/plans/prepended-section-0-with-foamy-pearl.md`
- Implementation branch: `claude/research-spec-decode-gating-5n0eP`
- Implementation commit: `08de8e1` (spec-decode passive tracer)
- v0.20 base: `88d34c6` (`v0.20.0` upstream tag)
- AGENTS.md / CLAUDE.md: project contribution policy (duplicate-work
  checks, accountability, dev workflow with `uv` + `.venv`).
