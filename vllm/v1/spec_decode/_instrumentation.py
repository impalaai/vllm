# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Passive instrumentation for offline P(accept | gap) calibration.

This is the Step 1a "~50 LOC" instrumentation that gates the whole
speculative-decode confidence-gating project. It emits two streams:

  * draft stream: (step, batch_idx, position, gap) where
        gap = top1_logit - top2_logit at the draft model's compute_logits
        output, before argmax. Recorded once per drafted position per
        request in the batch.

  * accept stream: (step, request_id, batch_idx, num_drafted, num_accepted)
        Recorded once per request per step from the scheduler side, after
        the rejection sampler has run.

Both streams are joined offline on (step, batch_idx) — the model runner
records the same step counter on both sides so the join is exact.

The tracer is fully no-op when VLLM_SPEC_GATE_TRACE is not set: callers
hit a single `is None` check and return. With the env var on, the only
hot-path cost is a topk(2) on the existing logits tensor plus one device
-> host copy per drafter invocation.

NEVER make this module import torch lazily — it is imported by the
drafter's hot path on every step.
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class _DraftRow:
    step: int
    batch_idx: int
    position: int
    gap: float


@dataclass
class _AcceptRow:
    step: int
    request_id: str
    batch_idx: int
    num_drafted: int
    num_accepted: int


class GapAcceptanceTracer:
    """Append-only per-process tracer for (gap, accept) tuples.

    Buffers in memory; flushes to JSONL shard files when either buffer
    crosses ``flush_every`` rows or when ``flush()`` is called explicitly.
    JSONL is chosen over Parquet for Step 1a because pyarrow is an
    optional dependency and we never want a missing dep to disable
    tracing silently. Step 1b's offline notebook converts to Parquet.
    """

    def __init__(self, dump_dir: str, flush_every: int):
        self.dump_dir = Path(dump_dir)
        self.dump_dir.mkdir(parents=True, exist_ok=True)
        self.flush_every = flush_every
        self._draft_rows: list[_DraftRow] = []
        self._accept_rows: list[_AcceptRow] = []
        self._step = 0
        self._lock = threading.Lock()
        # Shard-id is process-startup epoch seconds. Multiple workers
        # writing to the same dump dir disambiguate via this id.
        self._shard = f"{int(time.time())}_{id(self):x}"

    @property
    def step(self) -> int:
        return self._step

    def record_gaps(self, gaps_cpu: np.ndarray, position: int) -> None:
        """Append per-batch gaps for a single draft position.

        ``gaps_cpu`` must already be on CPU (typically via
        ``compute_top1_top2_gap(logits).cpu().numpy()`` at the call
        site). Shape: ``[batch_size]``.
        """
        step = self._step
        rows = self._draft_rows
        for batch_idx, gap in enumerate(gaps_cpu.tolist()):
            rows.append(
                _DraftRow(
                    step=step,
                    batch_idx=batch_idx,
                    position=position,
                    gap=float(gap),
                )
            )
        if len(rows) >= self.flush_every:
            self.flush()

    def record_acceptance(
        self,
        request_id: str,
        batch_idx: int,
        num_drafted: int,
        num_accepted: int,
    ) -> None:
        self._accept_rows.append(
            _AcceptRow(
                step=self._step,
                request_id=request_id,
                batch_idx=batch_idx,
                num_drafted=num_drafted,
                num_accepted=num_accepted,
            )
        )

    def step_boundary(self) -> None:
        """Advance the step counter. Call exactly once per scheduler step.

        Drafter records emitted *before* this call belong to the current
        step; records after belong to the next step.
        """
        self._step += 1

    def flush(self) -> None:
        with self._lock:
            if not self._draft_rows and not self._accept_rows:
                return
            path_d = self.dump_dir / f"shard_{self._shard}_drafts.jsonl"
            path_a = self.dump_dir / f"shard_{self._shard}_accepts.jsonl"
            try:
                if self._draft_rows:
                    with path_d.open("a") as f:
                        for r in self._draft_rows:
                            f.write(
                                json.dumps(
                                    {
                                        "step": r.step,
                                        "batch_idx": r.batch_idx,
                                        "position": r.position,
                                        "gap": r.gap,
                                    }
                                )
                                + "\n"
                            )
                    self._draft_rows.clear()
                if self._accept_rows:
                    with path_a.open("a") as f:
                        for r in self._accept_rows:
                            f.write(
                                json.dumps(
                                    {
                                        "step": r.step,
                                        "request_id": r.request_id,
                                        "batch_idx": r.batch_idx,
                                        "num_drafted": r.num_drafted,
                                        "num_accepted": r.num_accepted,
                                    }
                                )
                                + "\n"
                            )
                    self._accept_rows.clear()
            except OSError as e:
                # Trace-write failures must never crash the engine.
                logger.warning("spec-gate-trace flush failed: %s", e)


_TRACER: Optional[GapAcceptanceTracer] = None
_TRACER_INIT_LOCK = threading.Lock()


def get_tracer() -> Optional[GapAcceptanceTracer]:
    """Return the global tracer if VLLM_SPEC_GATE_TRACE is set, else None.

    The fast path is a single ``_TRACER is not None`` check after the
    first call; the env-var read happens at most once per process.
    """
    global _TRACER
    if _TRACER is not None:
        return _TRACER
    if not envs.VLLM_SPEC_GATE_TRACE:
        return None
    with _TRACER_INIT_LOCK:
        if _TRACER is not None:
            return _TRACER
        _TRACER = GapAcceptanceTracer(
            dump_dir=envs.VLLM_SPEC_GATE_TRACE_DIR,
            flush_every=envs.VLLM_SPEC_GATE_TRACE_FLUSH_EVERY,
        )
        logger.info(
            "spec-gate-trace enabled: dir=%s flush_every=%d",
            _TRACER.dump_dir,
            _TRACER.flush_every,
        )
        return _TRACER


def compute_top1_top2_gap(logits: torch.Tensor) -> torch.Tensor:
    """Return per-row ``top1 - top2`` logit gap.

    Input shape ``[N, vocab_size]``; output shape ``[N]``. The output is
    detached so it does not enter any autograd graph (the drafter is in
    inference mode but defence in depth is cheap).
    """
    top2 = torch.topk(logits, 2, dim=-1).values
    return (top2[:, 0] - top2[:, 1]).detach()


def _reset_tracer_for_tests() -> None:
    """Test-only hook to clear the singleton between unit tests."""
    global _TRACER
    with _TRACER_INIT_LOCK:
        if _TRACER is not None:
            _TRACER.flush()
        _TRACER = None
