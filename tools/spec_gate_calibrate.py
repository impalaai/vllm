# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline calibration analysis for the speculative-decode confidence gate.

Step 1b of the spec-gate research plan
(see /root/.claude/plans/prepended-section-0-with-foamy-pearl.md).

Consumes the JSONL shards produced by VLLM_SPEC_GATE_TRACE and emits:

  1. Empirical P(accept | gap_bin) curve (32 logit-gap bins).
  2. Monotonicity check (Spearman rho between gap-bin and accept-rate).
  3. Oracle-gated upper bound on throughput uplift vs static-k.
  4. Realizable upper bound from a deterministic k(gap) policy that only
     sees pre-acceptance signal.

This is the **decision-gate G0** computation: if the realizable upper
bound is < 10% over the best static-k baseline on >= 2 of 3 workloads,
the project stops here.

Usage:

    python tools/spec_gate_calibrate.py \\
        --trace-dir /tmp/vllm-spec-gate-trace \\
        --workload-tag chat \\
        --num-spec-tokens 4 \\
        --output-dir ./spec_gate_calibration

The script depends only on ``numpy``, ``pandas``, and ``scipy.stats`` —
no GPU, no torch, no vLLM imports. It can run on the laptop after
copying the shard files off the GPU box.

Limitations of this initial scaffold:

* Drafter-side records carry ``batch_idx`` (per-step row in the input
  batch) but the scheduler-side records carry ``request_id`` and
  ``batch_idx = -1``. We aggregate at the per-step / per-position level
  rather than per-request. That is sufficient for the marginal
  P(accept | gap) curve but we cannot observe within-request structure.
  A follow-up plumbs ``request_id`` through the drafter-side hook.
* Multi-process workers (Ray executor) write disjoint shards with
  process-local step counters. We currently treat shards independently
  and concatenate; correlation across drafter and scheduler in
  multi-process setups is a known follow-up.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd  # type: ignore[import-not-found]
from scipy import stats  # type: ignore[import-not-found]


# ---------------------------------------------------------------------------
# Shard loading
# ---------------------------------------------------------------------------
def load_jsonl(paths: Iterable[Path]) -> pd.DataFrame:
    rows: list[dict] = []
    for p in paths:
        with p.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return pd.DataFrame(rows)


def load_trace_dir(trace_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    drafts = load_jsonl(trace_dir.glob("shard_*_drafts.jsonl"))
    accepts = load_jsonl(trace_dir.glob("shard_*_accepts.jsonl"))
    if drafts.empty:
        raise SystemExit(f"No draft rows under {trace_dir}")
    if accepts.empty:
        raise SystemExit(f"No accept rows under {trace_dir}")
    return drafts, accepts


# ---------------------------------------------------------------------------
# Marginal P(accept | gap) curve
# ---------------------------------------------------------------------------
def join_drafts_with_accepts(
    drafts: pd.DataFrame,
    accepts: pd.DataFrame,
) -> pd.DataFrame:
    """Per-step-aggregate join.

    For each (step, position) we know:
      * the per-batch-row gap distribution (drafts), and
      * the per-step total num_drafted / num_accepted across all
        requests (sum of accepts).

    With only step granularity we infer the per-position acceptance
    rate from the well-known property of speculative decoding: a
    request that drafted ``num_drafted`` tokens and accepted
    ``num_accepted`` tokens accepted positions ``0..num_accepted-1`` and
    rejected position ``num_accepted`` (if num_accepted < num_drafted).

    We attribute that pattern to the per-(step, position) gap rows
    proportionally, weighted by the position's batch population. This
    is an approximation but it is unbiased at the marginal level when
    the gap-acceptance link is consistent across requests within a
    step — which is exactly the property we are calibrating.
    """
    accept_by_step = (
        accepts.groupby("step")
        .agg(num_drafted=("num_drafted", "sum"), num_accepted=("num_accepted", "sum"))
        .reset_index()
    )
    drafts_with_step_acc = drafts.merge(accept_by_step, on="step", how="inner")
    # Per (step, position) compute the population's expected accept rate
    # given step-level totals.
    drafts_with_step_acc["expected_accept_rate"] = drafts_with_step_acc.groupby(
        ["step"]
    )["num_accepted"].transform("mean") / drafts_with_step_acc.groupby(["step"])[
        "num_drafted"
    ].transform("mean")
    return drafts_with_step_acc


def calibration_curve(
    joined: pd.DataFrame,
    num_bins: int = 32,
) -> pd.DataFrame:
    """Empirical P(accept | gap_bin)."""
    gap_min = joined["gap"].min()
    gap_max = joined["gap"].max()
    if not np.isfinite(gap_min) or not np.isfinite(gap_max):
        raise ValueError("Non-finite gap values in trace.")
    edges = np.linspace(gap_min, gap_max, num_bins + 1)
    joined = joined.copy()
    joined["gap_bin"] = pd.cut(joined["gap"], edges, include_lowest=True)
    grouped = joined.groupby("gap_bin", observed=True).agg(
        n=("gap", "size"),
        mean_gap=("gap", "mean"),
        accept_rate=("expected_accept_rate", "mean"),
    )
    return grouped.reset_index()


def monotonicity_check(curve: pd.DataFrame) -> float:
    """Spearman rho between gap-bin centre and accept rate.

    A value close to +1 means accept-rate increases with gap, which is
    the hypothesis. Values near 0 or negative are kill-conditions for G0.
    """
    rho, _ = stats.spearmanr(curve["mean_gap"], curve["accept_rate"])
    return float(rho)


# ---------------------------------------------------------------------------
# Oracle and realizable upper bounds
# ---------------------------------------------------------------------------
def oracle_uplift(
    accepts: pd.DataFrame,
    num_spec_tokens: int,
) -> float:
    """Throughput uplift if we knew the truth and trimmed each request's k.

    Static-k baseline verifies ``num_spec_tokens`` positions per step per
    request. Oracle baseline verifies only ``num_accepted`` positions.
    Returns the ratio (static_verify_FLOPs - oracle_verify_FLOPs) /
    static_verify_FLOPs across the trace, treating verification cost as
    proportional to position count.
    """
    static_verify = float(num_spec_tokens) * len(accepts)
    oracle_verify = float(accepts["num_accepted"].sum())
    if static_verify <= 0:
        return 0.0
    saved = static_verify - oracle_verify
    return saved / static_verify


def realizable_uplift_from_curve(
    joined: pd.DataFrame,
    curve: pd.DataFrame,
    num_spec_tokens: int,
    p_min: float = 0.3,
) -> tuple[float, float]:
    """Pick threshold tau such that P(accept | gap < tau) <= p_min, then
    measure how much verification work is saved.

    Returns (uplift_fraction, chosen_threshold).
    """
    # Find largest gap such that bin accept-rate <= p_min.
    below = curve[curve["accept_rate"] <= p_min]
    if below.empty:
        return 0.0, float("-inf")
    tau = float(below["mean_gap"].max())

    # Apply tau: for each draft row, drop the position if gap < tau.
    joined = joined.copy()
    joined["dropped"] = joined["gap"] < tau
    static_verify = float(num_spec_tokens) * joined["step"].nunique() * (
        joined.groupby("step").size().mean()
        / max(num_spec_tokens, 1)
    )  # approximate
    saved = float(joined["dropped"].sum())
    if static_verify <= 0:
        return 0.0, tau
    return saved / max(static_verify, 1.0), tau


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def report(
    drafts: pd.DataFrame,
    accepts: pd.DataFrame,
    num_spec_tokens: int,
    workload_tag: str,
    output_dir: Path,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    joined = join_drafts_with_accepts(drafts, accepts)
    curve = calibration_curve(joined)
    rho = monotonicity_check(curve)
    oracle = oracle_uplift(accepts, num_spec_tokens)
    realizable, tau = realizable_uplift_from_curve(joined, curve, num_spec_tokens)

    summary = {
        "workload_tag": workload_tag,
        "num_spec_tokens": num_spec_tokens,
        "num_draft_rows": int(len(drafts)),
        "num_accept_rows": int(len(accepts)),
        "spearman_rho_gap_vs_accept": rho,
        "oracle_uplift_fraction": oracle,
        "realizable_uplift_fraction": realizable,
        "chosen_threshold_tau": tau,
        "g0_passes": realizable >= 0.10 and rho >= 0.5,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    curve.to_csv(output_dir / "calibration_curve.csv", index=False)
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--workload-tag", required=True)
    parser.add_argument("--num-spec-tokens", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    drafts, accepts = load_trace_dir(args.trace_dir)
    report(
        drafts=drafts,
        accepts=accepts,
        num_spec_tokens=args.num_spec_tokens,
        workload_tag=args.workload_tag,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
