# Pipeline Parallelism: How vLLM Avoids Bubbles

This document analyzes how vLLM's V1 engine balances pipeline parallel (PP)
execution so that pipeline stages remain busy and "bubbles" (idle gaps where a
stage waits for upstream work) are minimized. It walks through the relevant
code paths and explains the design choices.

The analysis covers the state of `main` at commit `51d4126` (latest at the
time of writing).

## 1. The pipeline-bubble problem in a nutshell

In a K-stage pipeline, a single batch traversing the pipeline keeps only
**one** stage busy at a time. The other K-1 stages are idle, giving an
efficiency of `1/K`. Classic remedies (GPipe, 1F1B, interleaved 1F1B) split a
batch into many micro-batches and stagger them across stages, so that all K
stages are doing useful work in steady state.

vLLM takes a different — and architecturally simpler — route: it does not
split a single autoregressive step into micro-batches. Instead, it keeps
**multiple decoding steps in flight at once**, where each step is itself a
batch of requests. The "micro-batch" granularity of LLM serving is naturally
a *step* (a single token-generation iteration over many requests), and the
engine pipelines those steps.

## 2. The headline mechanism: a depth-`pp_size` batch queue

The bubble-avoidance machinery lives in the engine core, not in the
distributed layer. The constructor of `EngineCore` allocates a `deque` whose
depth equals the number of PP stages:

```python
# vllm/v1/engine/core.py:137-147
# Setup batch queue for pipeline parallelism.
# Batch queue for scheduled batches. This enables us to asynchronously
# schedule and execute batches, and is required by pipeline parallelism
# to eliminate pipeline bubbles.
self.batch_queue_size = self.model_executor.max_concurrent_batches
self.batch_queue: Optional[deque[tuple[Future[ModelRunnerOutput],
                                       SchedulerOutput]]] = None
if self.batch_queue_size > 1:
    logger.info("Batch queue is enabled with size %d",
                self.batch_queue_size)
    self.batch_queue = deque(maxlen=self.batch_queue_size)
```

`max_concurrent_batches` is supplied by the executor and is exactly
`pipeline_parallel_size` for both supported multi-worker backends:

```python
# vllm/v1/executor/multiproc_executor.py:325-328
@property
def max_concurrent_batches(self) -> int:
    if self.scheduler_config.async_scheduling:
        return 2
    return self.parallel_config.pipeline_parallel_size
```

```python
# vllm/v1/executor/ray_distributed_executor.py:57-64
@property
def max_concurrent_batches(self) -> int:
    """Ray distributed executor supports pipeline parallelism,
    meaning that it allows PP size batches to be executed concurrently.
    """
    if self.scheduler_config.async_scheduling:
        return 2
    return self.parallel_config.pipeline_parallel_size
```

So with `pp_size=4`, the engine is allowed to hold four `(Future,
SchedulerOutput)` pairs in flight simultaneously — one for each stage of the
pipeline.

The engine picks its step function once at startup based on whether the queue
exists:

```python
# vllm/v1/engine/core.py:537-538
self.step_fn = (self.step if self.batch_queue is None else
                self.step_with_batch_queue)
```

PP > 1 ⇒ queued stepping. PP == 1 ⇒ ordinary blocking stepping.

## 3. The non-blocking step loop

`step_with_batch_queue` is the heart of the design. It implements a simple
two-rule loop: **fill the queue before reading from it**, and **only block
when you have no other useful work to do**.

```python
# vllm/v1/engine/core.py:308-359
def step_with_batch_queue(
        self) -> tuple[Optional[dict[int, EngineCoreOutputs]], bool]:
    """Schedule and execute batches with the batch queue.
    Note that if nothing to output in this step, None is returned.

    The execution flow is as follows:
    1. Try to schedule a new batch if the batch queue is not full.
    If a new batch is scheduled, directly return an empty engine core
    output. In other words, fulfilling the batch queue has a higher priority
    than getting model outputs.
    2. If there is no new scheduled batch, meaning that the batch queue
    is full or no other requests can be scheduled, we block until the first
    batch in the job queue is finished.
    3. Update the scheduler from the output.
    """
    batch_queue = self.batch_queue
    assert batch_queue is not None
    assert len(batch_queue) < self.batch_queue_size

    model_executed = False
    if self.scheduler.has_requests():
        scheduler_output = self.scheduler.schedule()
        future = self.model_executor.execute_model(scheduler_output)
        batch_queue.appendleft((future, scheduler_output))

        model_executed = scheduler_output.total_num_scheduled_tokens > 0
        if model_executed and len(batch_queue) < self.batch_queue_size \
            and not batch_queue[-1][0].done():
            # Don't block on next worker response unless the queue is full
            # or there are no more requests to schedule.
            return None, True

    elif not batch_queue:
        return None, False

    # Block until the next result is available.
    future, scheduler_output = batch_queue.pop()
    model_output = self.execute_model_with_error_logging(
        lambda _: future.result(), scheduler_output)

    engine_core_outputs = self.scheduler.update_from_output(
        scheduler_output, model_output)

    return engine_core_outputs, model_executed
```

The crucial line is `return None, True` after `batch_queue.appendleft(...)`.
That early return is what keeps the pipeline full: the engine launches batch
N+1 without ever waiting on batch N to come back. Only when the queue is full
(or there is genuinely nothing else to schedule) does the engine `pop()` the
oldest batch and synchronously wait on its `Future`.

In steady state with sufficient load, the loop alternates between scheduling
a new batch and harvesting a completed one — and at every moment all
`pp_size` stages have a batch they are processing.

## 4. Why this is equivalent to 1F1B at step granularity

A K-stage 1F1B pipeline runs by sending micro-batch *i* through stage 0,
then sending micro-batch *i+1* into stage 0 while *i* moves to stage 1, and so
on. In steady state, micro-batch *j* is at stage `j mod K`.

vLLM does the same thing, but treats one *autoregressive decoding step* as
the micro-batch. Each call to `execute_model(scheduler_output)` is a single
forward pass that, on the worker side, traverses one PP stage at a time
through the layers it owns. The `Future` returned to the engine resolves only
when the *last* PP rank has produced sampled tokens; meanwhile, the engine
has already submitted the next step. With `pp_size` outstanding `Future`s,
each PP rank always has a step to compute.

What vLLM does **not** do is split a single decoding step into smaller
sub-batches that get interleaved within a single iteration. This is partly
because LLM serving naturally produces a steady stream of new steps — the
"interleaving" comes from the temporal flow of decoding rather than from
splitting one step.

## 5. Closing the loop: routing sampled tokens through the scheduler

A subtle PP problem: only the **last** PP rank actually samples the next
token, but the **first** PP rank needs that token to start step N+1. In a
naive design you would add a direct last-rank → first-rank link. vLLM
sidesteps this by making the scheduler the carrier:

```python
# vllm/v1/core/sched/scheduler.py:680-688
if self.use_pp:
    # When using PP, the scheduler sends the sampled tokens back,
    # because there's no direct communication between the first-
    # stage worker and the last-stage worker.
    token_ids = req.all_token_ids[req.num_computed_tokens:
                                  req.num_computed_tokens + num_tokens]
    new_token_ids.append(token_ids)
```

When a step completes, the engine calls `scheduler.update_from_output(...)`,
which records the new tokens. When the next batch is scheduled, the scheduler
includes those tokens in the `SchedulerOutput` it ships to the first PP rank.
This keeps the topology to a single client→worker tree instead of requiring
a return path from rank `K-1` to rank `0`.

The scheduler also has to be careful not to stall when it has already
scheduled all prompt tokens for a request that has not yet finished
computing across PP stages:

```python
# vllm/v1/core/sched/scheduler.py:236-250 (paraphrased context)
if num_new_tokens == 0:
    # PP>1 and we have already scheduled all prompt tokens
    # but they are not finished yet.
    req_index += 1   # Skip this request, look for lower-priority work.
```

Without this, the scheduler would block on a request waiting for upstream
data and would starve the pipeline of new work.

## 6. Balancing the *compute* across stages: `get_pp_indices`

A queue cannot hide bubbles caused by *unequal* stage cost: if stage 0 takes
twice as long as the others, the others will idle waiting for it regardless
of how many steps are in flight. vLLM tries to make stages cost-balanced via
its layer partitioner:

```python
# vllm/distributed/utils.py:88-134
def get_pp_indices(num_hidden_layers: int, pp_rank: int,
                   pp_size: int) -> tuple[int, int]:
    """Try to evenly distribute layers across partitions.

    If the number of layers is not divisible by the number of partitions,
    the remaining layers are evenly distributed across all but the last
    partition. The last partition is excluded because it often contains an
    additional norm layer and we are attempting to balance compute.

    If `pp_size > 2` and the number of remaining layers is
    `0 < x <= pp_size - 2` then the remaining layers are evenly distributed
    across the middle partitions. The first and last partitions are excluded
    because they contain the input and output embeddings respectively and we
    are attempting to reduce maximum memory consumption across partitions.
    """
    ...
    layers_per_partition = num_hidden_layers // pp_size
    partitions = [layers_per_partition for _ in range(pp_size)]
    if remaining_layers := num_hidden_layers % pp_size:
        for i in range(2, remaining_layers + 2):
            partitions[-i] += 1
    ...
    start_layer = sum(partitions[:pp_rank])
    end_layer = start_layer + partitions[pp_rank]
    return (start_layer, end_layer)
```

Two design points worth highlighting:

- The first and last ranks are deliberately given the *floor* number of
  decoder blocks. The first rank also owns the embedding table; the last
  rank owns the final norm + LM head. Pushing the remainder onto middle
  ranks balances both compute *and* peak memory.
- The split is overridable per-deployment via the `VLLM_PP_LAYER_PARTITION`
  environment variable, which accepts a comma-separated list summing to the
  layer count. This is the escape hatch when a model has heterogeneous
  layers (e.g. some are MoE and far more expensive).

Layers outside the local range are instantiated as `PPMissingLayer`
placeholders (in `vllm/model_executor/models/utils.py::make_layers`), so the
model code can iterate uniformly without rank-specific branches.

## 7. Activations on the wire: `IntermediateTensors`

Between stages, each rank sends hidden states (and optionally residuals) to
the next rank using a NCCL P2P send/recv on the PP group:

```python
# vllm/v1/worker/gpu_worker.py:425-461
@torch.inference_mode()
def execute_model(
    self,
    scheduler_output: "SchedulerOutput",
) -> Optional[Union[ModelRunnerOutput, AsyncModelRunnerOutput]]:
    intermediate_tensors = None
    forward_pass = scheduler_output.total_num_scheduled_tokens > 0
    if forward_pass and not get_pp_group().is_first_rank:
        intermediate_tensors = IntermediateTensors(
            get_pp_group().recv_tensor_dict(
                all_gather_group=get_tp_group()))

    output = self.model_runner.execute_model(scheduler_output,
                                             intermediate_tensors)
    if isinstance(output, (ModelRunnerOutput, AsyncModelRunnerOutput)):
        return output

    assert isinstance(output, IntermediateTensors)
    ...
    get_pp_group().send_tensor_dict(output.tensors,
                                    all_gather_group=get_tp_group())
```

`IntermediateTensors` (`vllm/sequence.py`) is a typed wrapper around a
`dict[str, torch.Tensor]` whose contents are determined by each model's
`make_empty_intermediate_tensors` factory. The factory lets the worker
**pre-allocate** the receive buffers once, so the per-step path has no
allocations on the activation channel.

Two implications:

- The `send_tensor_dict` / `recv_tensor_dict` calls are **synchronous** —
  the receiving rank cannot start its layer compute until the previous
  rank has finished. There is currently no overlap of compute with
  inter-stage communication within a single step.
- The `all_gather_group=get_tp_group()` argument enables a small but
  important optimization: when the activation tensor is divisible by the
  TP group size, only one TP shard sends the data and the rest of the
  group recovers it via an all-gather. This reduces the per-stage P2P
  payload by the TP factor.

The bubble-avoidance argument therefore relies on the batch queue, not on
overlapped P2P: each rank stays busy because *another* step's activations
arrive while it is still computing the *previous* step.

## 8. Pipeline arithmetic

Let `t_stage` be the time a single PP rank spends per step (compute +
intra-step P2P send/recv). For one step, end-to-end latency is roughly
`K · t_stage` where `K = pp_size`. With the batch queue full of `K` steps,
the engine completes one step every `t_stage`, giving throughput

```
throughput_pp ≈ throughput_single_stage  (after pipeline fill)
```

Bubbles occur in three remaining places:

1. **Fill** (first `K-1` steps after the server starts or after the queue
   drains) — at most `(K-1) · t_stage` of bubble per drain event.
2. **Drain** at shutdown / very-low-load periods.
3. **Imbalance**: if any rank's `t_stage` is larger than the others, the
   faster ranks idle by the difference each step. The `get_pp_indices`
   heuristic targets this, and the env var override is the escape hatch.

For long-running serving workloads with steady demand, the fill/drain costs
amortize to ~0 and the imbalance term is the only persistent bubble.

## 9. Async scheduling: a related, separate mechanism

Independent of PP, vLLM supports asynchronous scheduling (overlap of
scheduling with the previous step's model execution). When enabled
(`async_scheduling=True`), the executor reports `max_concurrent_batches = 2`
even for `pp_size = 1`:

```python
# vllm/v1/executor/multiproc_executor.py:325-328
if self.scheduler_config.async_scheduling:
    return 2
return self.parallel_config.pipeline_parallel_size
```

So the batch-queue machinery has two distinct uses: hiding PP latency
(`depth = pp_size`) and hiding scheduler latency (`depth = 2`). The
companion `AsyncScheduler` (`vllm/v1/core/sched/async_scheduler.py`)
pre-reserves slots for tokens that have not yet been sampled so the next
batch can be planned without the previous one's output in hand:

```python
# vllm/v1/core/sched/async_scheduler.py (paraphrased)
if (request.num_computed_tokens ==
        request.num_tokens + request.num_output_placeholders):
    # The request will generate a new token in this scheduling step.
    request.num_output_placeholders += 1
```

When both PP > 1 and async scheduling are configured, the queue depth still
follows `pp_size` (the larger of the two), so the PP guarantee is preserved.

## 10. What vLLM intentionally does *not* do

- **No 1F1B within a step.** There is no decomposition of a single
  decoding step into smaller micro-batches that get interleaved across
  ranks. The unit of pipelining is the full step.
- **No comm/compute overlap between stages.** Stage transitions are blocking
  `send_tensor_dict` / `recv_tensor_dict` calls. A TODO in the model runner
  flags this as future work:

  ```python
  # vllm/v1/worker/gpu_model_runner.py (near line 2079)
  # TODO: Support overlapping mirco-batches
  # https://github.com/vllm-project/vllm/issues/18019
  ```

- **No special "interleaved" PP schedule** (à la Megatron's V-cycle).
  Every step traverses ranks `0..K-1` in the same order.

These omissions reflect a deliberate trade-off: the dominant bubble for
serving workloads is the inter-step gap, which the queue closes cleanly. The
intra-step overlap that 1F1B and interleaved schedules add costs significant
complexity for a smaller marginal gain in the LLM serving regime, where each
"micro-batch" is already a large multi-request step.

## 11. Code map (quick reference)

| Concern | File | Symbol |
|---|---|---|
| Per-engine queue allocation | `vllm/v1/engine/core.py:137-147` | `EngineCore.__init__` |
| Non-blocking step loop | `vllm/v1/engine/core.py:308-359` | `step_with_batch_queue` |
| Step function selection | `vllm/v1/engine/core.py:537-538` | `step_fn` assignment |
| Queue depth (multiproc) | `vllm/v1/executor/multiproc_executor.py:325-328` | `max_concurrent_batches` |
| Queue depth (ray) | `vllm/v1/executor/ray_distributed_executor.py:57-64` | `max_concurrent_batches` |
| Token feedback in PP | `vllm/v1/core/sched/scheduler.py:680-688` | `Scheduler.schedule` |
| Layer balancing | `vllm/distributed/utils.py:88-134` | `get_pp_indices` |
| Manual layer override | `VLLM_PP_LAYER_PARTITION` env var | — |
| Activation envelope | `vllm/sequence.py` | `IntermediateTensors` |
| Stage send/recv | `vllm/v1/worker/gpu_worker.py:425-461` | `Worker.execute_model` |
| Async scheduling reservation | `vllm/v1/core/sched/async_scheduler.py` | `AsyncScheduler` |

## 12. Summary

vLLM's PP design is a study in moving complexity to where it is cheapest. The
distributed layer stays simple — synchronous P2P of `IntermediateTensors`
between adjacent ranks — and the model layer stays simple — even-ish layer
partitioning with a memory-aware bias toward middle ranks. The interesting
work happens in the engine, where a `pp_size`-deep `deque` of in-flight
`(Future, SchedulerOutput)` pairs and a "fill before drain" step loop keep
every stage busy without any per-step micro-batch interleaving.

The result is that, in steady-state serving, every PP rank is computing on
behalf of *some* decoding step at every moment, and the only residual
bubbles come from stage-cost imbalance — which the layer partitioner and the
`VLLM_PP_LAYER_PARTITION` override are there to mitigate.
