# GPU Run Plan — Spec-Gate Tracing (Step 1a)

This is the operator-side runbook for the passive-instrumentation phase
of the spec-gate research project. Local development happens without a
GPU; this document is the contract that an operator with hardware
follows to validate the changes and produce the data the offline
calibration analysis needs.

## What this run validates

1. **Bitwise-identity (mandatory)** — outputs with `VLLM_SPEC_GATE_TRACE`
   unset must be identical to upstream `v0.20.0` outputs. This proves
   the instrumentation is a true no-op when off.
2. **Trace correctness** — with the tracer on, the JSONL shards under
   `VLLM_SPEC_GATE_TRACE_DIR` are well-formed and contain
   per-(step, batch_idx, position) gaps and per-(step, request_id)
   acceptance counts.
3. **Overhead budget** — throughput with tracer on must be within 5% of
   tracer off at the chosen flush rate.

## Container

```
docker build -f docker/Dockerfile -t vllm-spec-gate-trace:v0.20-instrumented .
```

The build pins `torch_cuda_arch_list='7.0 7.5 8.0 8.9 9.0 10.0 12.0'`
(see `docker/Dockerfile:152`). For Blackwell-only dev boxes, narrow to
`'9.0;10.0'` to cut build time.

For pure-Python iteration (this PR is pure Python), the operator can
hotpatch the running container instead of rebuilding:

```
docker run --gpus all --rm -it \
    -v $(pwd)/vllm:/usr/local/lib/python3.12/dist-packages/vllm \
    -v /tmp/vllm-spec-gate-trace:/tmp/vllm-spec-gate-trace \
    vllm-spec-gate-trace:v0.20-instrumented bash
```

## Step 1: bitwise-identity baseline

Run any spec-decode example with `VLLM_SPEC_GATE_TRACE` unset. The
output token IDs and logprobs must match upstream `v0.20.0` exactly on
the same prompt + temperature=0 + same seed.

```
VLLM_SPEC_GATE_TRACE=0 python examples/offline_inference/eagle.py \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --num-spec-tokens 4 \
    --temperature 0 --seed 0 \
    --num-prompts 16 \
    > /tmp/baseline_off.txt
```

Compare against an unmodified `v0.20.0` build:

```
diff /tmp/baseline_off.txt /tmp/baseline_v0.20.0.txt
# expected: empty diff
```

**If this diff is non-empty, do not proceed.** It means the patch is
not a true no-op. Block the PR.

## Step 2: tracer correctness

```
mkdir -p /tmp/vllm-spec-gate-trace
VLLM_SPEC_GATE_TRACE=1 \
VLLM_SPEC_GATE_TRACE_DIR=/tmp/vllm-spec-gate-trace \
VLLM_SPEC_GATE_TRACE_FLUSH_EVERY=1024 \
python examples/offline_inference/eagle.py \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --num-spec-tokens 4 \
    --num-prompts 256 \
    > /tmp/with_tracer.txt
```

Check the shards:

```
ls /tmp/vllm-spec-gate-trace/
# Expected:
#   shard_<epoch>_<id>_drafts.jsonl
#   shard_<epoch>_<id>_accepts.jsonl

head -3 /tmp/vllm-spec-gate-trace/shard_*_drafts.jsonl
# Expected schema: {"step": int, "batch_idx": int, "position": int, "gap": float}

head -3 /tmp/vllm-spec-gate-trace/shard_*_accepts.jsonl
# Expected schema: {"step": int, "request_id": str, "batch_idx": int,
#                   "num_drafted": int, "num_accepted": int}
```

Sanity assertions an operator can run:

```
python - <<'PY'
import json, glob
draft_rows = [json.loads(l) for p in glob.glob('/tmp/vllm-spec-gate-trace/*_drafts.jsonl')
              for l in open(p)]
accept_rows = [json.loads(l) for p in glob.glob('/tmp/vllm-spec-gate-trace/*_accepts.jsonl')
               for l in open(p)]
print('draft rows:', len(draft_rows))
print('accept rows:', len(accept_rows))
gaps = [r['gap'] for r in draft_rows]
print('gap min / median / max:',
      min(gaps), sorted(gaps)[len(gaps)//2], max(gaps))
assert all(g >= 0 for g in gaps), 'gap must be non-negative'
assert all(0 <= r['num_accepted'] <= r['num_drafted'] for r in accept_rows)
print('OK')
PY
```

Expected output: gaps non-negative; num_accepted within [0, num_drafted].

## Step 3: overhead measurement

Measure throughput in a single canonical config, with and without the
tracer, on the same prompt set:

```
# tracer off
VLLM_SPEC_GATE_TRACE=0 vllm bench latency \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --speculative-config '{"method":"eagle","num_speculative_tokens":4}' \
    --num-iters 30 > /tmp/perf_off.txt

# tracer on
VLLM_SPEC_GATE_TRACE=1 vllm bench latency \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --speculative-config '{"method":"eagle","num_speculative_tokens":4}' \
    --num-iters 30 > /tmp/perf_on.txt
```

Acceptance criterion: throughput drop with tracer on must be < 5% at
the default `VLLM_SPEC_GATE_TRACE_FLUSH_EVERY=4096`. If higher, raise
`VLLM_SPEC_GATE_TRACE_FLUSH_EVERY` and retest; document the chosen
value in the PR.

## Step 4: produce the data dump for G0

Once Steps 1–3 are green, dump 24h of traffic across the three
workloads in the plan (chat, code, agentic). Recommended:

```
VLLM_SPEC_GATE_TRACE=1 \
VLLM_SPEC_GATE_TRACE_DIR=/data/spec-gate-trace/llama3-8b-chat \
VLLM_SPEC_GATE_TRACE_FLUSH_EVERY=4096 \
vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct \
    --speculative-config '{"method":"eagle","num_speculative_tokens":4}' \
    --tensor-parallel-size 1 \
    ... # rest of the prod config
```

Tag the workload by writing into `VLLM_SPEC_GATE_TRACE_DIR` per
deployment slice (chat / code / agentic). The offline analysis script
takes a single trace-dir, so split by directory.

## Step 5: offline analysis (laptop)

After copying the shards to a laptop, run:

```
python tools/spec_gate_calibrate.py \
    --trace-dir /data/spec-gate-trace/llama3-8b-chat \
    --workload-tag chat \
    --num-spec-tokens 4 \
    --output-dir ./spec_gate_calibration/llama3-8b-chat
```

The script writes:
* `summary.json` — Spearman rho, oracle uplift, realizable uplift, G0 verdict.
* `calibration_curve.csv` — bin-level P(accept | gap_bin).

**G0 verdict (project decision gate):** if
`g0_passes == true` on at least 2 of 3 workloads, proceed to Step 2
(workload-conditional analysis) of the plan. Otherwise the project
stops here.

## Known caveats this run will surface

1. **`use_local_argmax_reduction`** — when this config flag is set, the
   drafter takes a different path that does not expose the full
   vocabulary distribution; gaps are not recorded. The shard counts
   will be correspondingly lower for affected requests. Document the
   actual fraction of requests affected in the PR.

2. **Multi-process workers** — drafter-side and scheduler-side records
   end up in different shard files in Ray-executor setups, with
   independent step counters. The current calibration script aggregates
   per-shard. For the G0 measurement this is acceptable. If the
   per-request join is needed later, a follow-up commit threads
   `request_id` through the drafter hook.

3. **Tree drafting path** — `propose_tree` uses separate `compute_logits`
   sites (`llm_base_proposer.py:504, 1143`). Tracer is *not* hooked
   there in this PR; tree-drafted runs will be missing draft rows for
   the tree positions. Out of scope for the chain-drafting calibration;
   add hooks before measuring tree configurations.
