# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-runnable unit tests for the spec-gate calibration tracer.

These tests exist because we cannot run vLLM on a GPU in the development
environment. They exercise the tracer with synthetic CPU tensors so that
the gate logic — and the drafter-side hot path that feeds it — can be
validated against expected behavior on every commit.

What is NOT covered here:
  * Actual interception inside the drafter forward pass (that requires a
    GPU run; see GPU_RUN_PLAN.md).
  * Multi-process correlation across drafter and scheduler when running
    under Ray/multiprocess executor.

Anything that reaches into vllm.v1.spec_decode.llm_base_proposer or the
scheduler is GPU-required and lives in a separate test file.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from vllm.v1.spec_decode import _instrumentation


@pytest.fixture(autouse=True)
def _clean_tracer(monkeypatch, tmp_path):
    """Reset the global tracer and point it at a tmp dir for every test."""
    monkeypatch.setenv("VLLM_SPEC_GATE_TRACE", "0")
    monkeypatch.setenv("VLLM_SPEC_GATE_TRACE_DIR", str(tmp_path))
    monkeypatch.setenv("VLLM_SPEC_GATE_TRACE_FLUSH_EVERY", "8")
    # Re-read envs — the env-var lookup table is built at import time.
    import vllm.envs

    # Clear any cached env values inside vllm.envs.
    for attr in (
        "VLLM_SPEC_GATE_TRACE",
        "VLLM_SPEC_GATE_TRACE_DIR",
        "VLLM_SPEC_GATE_TRACE_FLUSH_EVERY",
    ):
        if attr in vars(vllm.envs):
            delattr(vllm.envs, attr)
    _instrumentation._reset_tracer_for_tests()
    yield
    _instrumentation._reset_tracer_for_tests()


def _enable_tracer(monkeypatch, tmp_path: Path, flush_every: int = 8) -> None:
    monkeypatch.setenv("VLLM_SPEC_GATE_TRACE", "1")
    monkeypatch.setenv("VLLM_SPEC_GATE_TRACE_DIR", str(tmp_path))
    monkeypatch.setenv("VLLM_SPEC_GATE_TRACE_FLUSH_EVERY", str(flush_every))
    import vllm.envs

    for attr in (
        "VLLM_SPEC_GATE_TRACE",
        "VLLM_SPEC_GATE_TRACE_DIR",
        "VLLM_SPEC_GATE_TRACE_FLUSH_EVERY",
    ):
        if attr in vars(vllm.envs):
            delattr(vllm.envs, attr)
    _instrumentation._reset_tracer_for_tests()


def test_tracer_disabled_by_default():
    """When VLLM_SPEC_GATE_TRACE is not set, get_tracer returns None and no
    files are created. This is the bitwise-no-op contract."""
    assert _instrumentation.get_tracer() is None


def test_tracer_enabled_returns_singleton(monkeypatch, tmp_path):
    _enable_tracer(monkeypatch, tmp_path)
    t1 = _instrumentation.get_tracer()
    t2 = _instrumentation.get_tracer()
    assert t1 is not None
    assert t1 is t2


def test_compute_top1_top2_gap_basic_shape_and_values():
    logits = torch.tensor(
        [
            [1.0, 5.0, 3.0, 2.0],  # gap = 5 - 3 = 2
            [0.5, 0.5, 1.5, 0.0],  # gap = 1.5 - 0.5 = 1
            [-1.0, -2.0, -3.0, -4.0],  # gap = -1 - -2 = 1
        ]
    )
    gap = _instrumentation.compute_top1_top2_gap(logits)
    assert gap.shape == (3,)
    np.testing.assert_allclose(gap.numpy(), [2.0, 1.0, 1.0], atol=1e-6)


def test_compute_top1_top2_gap_is_nonnegative():
    """Gap = top1 - top2 must always be >= 0 by construction. Property test
    over random logits."""
    torch.manual_seed(0)
    logits = torch.randn(64, 1024)
    gap = _instrumentation.compute_top1_top2_gap(logits)
    assert (gap >= 0).all()


def test_record_gaps_appends_rows(monkeypatch, tmp_path):
    _enable_tracer(monkeypatch, tmp_path, flush_every=10**9)
    tracer = _instrumentation.get_tracer()
    assert tracer is not None
    gaps = np.array([1.0, 2.0, 3.0])
    tracer.record_gaps(gaps, position=0)
    tracer.record_gaps(np.array([0.5, 0.5, 0.5]), position=1)
    assert len(tracer._draft_rows) == 6
    # Position 0 rows match the input order.
    pos0 = [r for r in tracer._draft_rows if r.position == 0]
    assert [r.batch_idx for r in pos0] == [0, 1, 2]
    np.testing.assert_allclose([r.gap for r in pos0], [1.0, 2.0, 3.0])


def test_record_acceptance_appends_rows(monkeypatch, tmp_path):
    _enable_tracer(monkeypatch, tmp_path, flush_every=10**9)
    tracer = _instrumentation.get_tracer()
    assert tracer is not None
    tracer.record_acceptance(
        request_id="req-A", batch_idx=0, num_drafted=4, num_accepted=2
    )
    tracer.record_acceptance(
        request_id="req-B", batch_idx=1, num_drafted=4, num_accepted=4
    )
    assert len(tracer._accept_rows) == 2
    assert tracer._accept_rows[0].request_id == "req-A"
    assert tracer._accept_rows[1].num_accepted == 4


def test_step_boundary_advances_counter(monkeypatch, tmp_path):
    _enable_tracer(monkeypatch, tmp_path, flush_every=10**9)
    tracer = _instrumentation.get_tracer()
    assert tracer is not None
    tracer.record_gaps(np.array([1.0]), position=0)
    assert tracer._draft_rows[0].step == 0
    tracer.step_boundary()
    tracer.record_gaps(np.array([2.0]), position=0)
    assert tracer._draft_rows[1].step == 1


def test_flush_writes_jsonl_shards(monkeypatch, tmp_path):
    _enable_tracer(monkeypatch, tmp_path, flush_every=2)
    tracer = _instrumentation.get_tracer()
    assert tracer is not None
    tracer.record_gaps(np.array([1.0, 2.0]), position=0)  # 2 rows triggers flush
    tracer.record_acceptance(
        request_id="r0", batch_idx=0, num_drafted=2, num_accepted=1
    )
    tracer.flush()

    drafts = list(tmp_path.glob("shard_*_drafts.jsonl"))
    accepts = list(tmp_path.glob("shard_*_accepts.jsonl"))
    assert len(drafts) == 1, drafts
    assert len(accepts) == 1, accepts

    rows = [json.loads(l) for l in drafts[0].read_text().splitlines()]
    assert len(rows) == 2
    assert {r["batch_idx"] for r in rows} == {0, 1}
    assert all(r["position"] == 0 for r in rows)
    assert all(r["step"] == 0 for r in rows)

    accept_rows = [json.loads(l) for l in accepts[0].read_text().splitlines()]
    assert len(accept_rows) == 1
    assert accept_rows[0]["request_id"] == "r0"
    assert accept_rows[0]["num_drafted"] == 2
    assert accept_rows[0]["num_accepted"] == 1


def test_flush_is_idempotent_on_empty(monkeypatch, tmp_path):
    _enable_tracer(monkeypatch, tmp_path, flush_every=10**9)
    tracer = _instrumentation.get_tracer()
    assert tracer is not None
    tracer.flush()
    tracer.flush()
    # No files created from empty flushes.
    assert not list(tmp_path.glob("shard_*_drafts.jsonl"))
    assert not list(tmp_path.glob("shard_*_accepts.jsonl"))


def test_auto_flush_at_threshold(monkeypatch, tmp_path):
    _enable_tracer(monkeypatch, tmp_path, flush_every=4)
    tracer = _instrumentation.get_tracer()
    assert tracer is not None
    # 5 gap rows -> auto-flush after the 4th.
    tracer.record_gaps(np.array([1.0, 2.0, 3.0, 4.0]), position=0)
    drafts = list(tmp_path.glob("shard_*_drafts.jsonl"))
    assert len(drafts) == 1
    rows = [json.loads(l) for l in drafts[0].read_text().splitlines()]
    assert len(rows) == 4


def test_failed_write_does_not_raise(monkeypatch, tmp_path):
    """If the dump directory becomes unwritable, flush must warn but not
    raise — the engine must never crash because of telemetry."""
    _enable_tracer(monkeypatch, tmp_path, flush_every=10**9)
    tracer = _instrumentation.get_tracer()
    assert tracer is not None
    tracer.record_gaps(np.array([1.0]), position=0)
    # Make the dump dir unwritable.
    os.chmod(tmp_path, 0o555)
    try:
        tracer.flush()  # Must not raise.
    finally:
        os.chmod(tmp_path, 0o755)


def test_logit_gap_via_topk_matches_manual_for_small_vocab():
    """End-to-end: compute the gap on a small synthetic logit tensor and
    verify against a manual top-2 computation. This is the contract that
    the drafter hot path relies on."""
    torch.manual_seed(42)
    logits = torch.randn(8, 64)
    gap = _instrumentation.compute_top1_top2_gap(logits)
    sorted_vals, _ = torch.sort(logits, dim=-1, descending=True)
    expected = sorted_vals[:, 0] - sorted_vals[:, 1]
    torch.testing.assert_close(gap, expected)


def test_record_gaps_handles_empty_batch(monkeypatch, tmp_path):
    _enable_tracer(monkeypatch, tmp_path, flush_every=10**9)
    tracer = _instrumentation.get_tracer()
    assert tracer is not None
    tracer.record_gaps(np.array([], dtype=np.float32), position=0)
    assert len(tracer._draft_rows) == 0
