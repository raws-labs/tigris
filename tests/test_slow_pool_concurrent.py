"""Slow-tier (PSRAM) budget must model concurrent residency, not just the
coarse per-stage input+output max. A tensor that crosses several stage
boundaries (a long-lived skip) stays resident in slow memory for every
step in between, and can coexist with a middle stage's own boundary
tensors in a way no single stage's own input+output sum captures.
"""

from onnx import TensorProto

from tigris.analysis.validation import slow_pool_usage
from tigris.graph.ir import AnalyzedGraph, MemoryBudget, OpNode, Stage, TensorInfo


def build_long_lived_skip_graph() -> AnalyzedGraph:
    """Four single-op stages. "skip" is born at stage 0's output and only
    consumed by stage 3, so it stays slow-resident through stage 1 and
    stage 2. Each stage has exactly one op, so op step index == stage
    op_indices == lifetime step index.

    Tensor sizes (bytes; INT8 dtype so num_elements == size_bytes):
      input=64, skip=1000 (the long-lived skip), mid1=2000, mid2=64,
      output=64.

    Old coarse per-stage peak (that stage's own input_tensors +
    output_tensors sizes, summed independently per stage):
      stage0: input(64) + skip(1000)             = 1064
      stage1: input(64) + mid1(2000)              = 2064
      stage2: mid1(2000) + mid2(64)                = 2064
      stage3: (mid2(64) + skip(1000)) + output(64) = 1128
      old coarse peak = max(...) = 2064

    Concurrent peak (closed-closed birth_step <= t <= death_step, matching
    _live_bytes_by_step / stage.peak_bytes in partition_temporal.py;
    lifetimes: input birth=-1 death=1, skip birth=0 death=3, mid1 birth=1
    death=2, mid2 birth=2 death=3, output birth=3 death=4):
      t=0 (stage0): input(64) alive; skip born this step, already live
                    -> 64 + 1000 = 1064
      t=1 (stage1): input(64, dies here) + skip(1000, alive)
                    + mid1(2000, born this step, already live)
                    -> 64 + 1000 + 2000 = 3064
      t=2 (stage2): skip(1000, still alive) + mid1(2000, dies here)
                    + mid2(64, born this step, already live)
                    -> 1000 + 2000 + 64 = 3064
      t=3 (stage3): skip(1000, dies here) + mid2(64, dies here)
                    + output(64, born this step, already live)
                    -> 1000 + 64 + 64 = 1128
      concurrent peak = max(1064, 3064, 3064, 1128) = 3064

    3064 > 2064: at stage 1's own op step, "skip" (produced by stage 0,
    not yet consumed by stage 3) is concurrently resident alongside both
    of stage 1's own boundary tensors ("input" and "mid1"), and the
    coarse per-stage method never sums a boundary tensor from one stage
    against a different stage's own boundary tensors.
    """
    op0 = OpNode(name="tap", op_type="Conv", inputs=["input"], outputs=["skip"], step=0)
    op1 = OpNode(name="branch", op_type="Conv", inputs=["input"], outputs=["mid1"], step=1)
    op2 = OpNode(name="mid_conv", op_type="Conv", inputs=["mid1"], outputs=["mid2"], step=2)
    op3 = OpNode(name="merge", op_type="Add", inputs=["mid2", "skip"], outputs=["output"], step=3)

    stage0 = Stage(
        stage_id=0, op_indices=[0],
        input_tensors=["input"], output_tensors=["skip"],
        peak_bytes=100_000,
    )
    stage1 = Stage(
        stage_id=1, op_indices=[1],
        input_tensors=["input"], output_tensors=["mid1"],
        peak_bytes=100_000,
    )
    stage2 = Stage(
        stage_id=2, op_indices=[2],
        input_tensors=["mid1"], output_tensors=["mid2"],
        peak_bytes=100_000,
    )
    stage3 = Stage(
        stage_id=3, op_indices=[3],
        input_tensors=["mid2", "skip"], output_tensors=["output"],
        peak_bytes=100_000,
    )

    return AnalyzedGraph(
        ops=[op0, op1, op2, op3],
        stages=[stage0, stage1, stage2, stage3],
        model_inputs=["input"],
        model_outputs=["output"],
        tensors={
            "input": TensorInfo("input", (1, 64), TensorProto.INT8),
            "skip": TensorInfo("skip", (1, 1000), TensorProto.INT8),
            "mid1": TensorInfo("mid1", (1, 2000), TensorProto.INT8),
            "mid2": TensorInfo("mid2", (1, 64), TensorProto.INT8),
            "output": TensorInfo("output", (1, 64), TensorProto.INT8),
        },
        budget=MemoryBudget(fast=64),
    )


def test_slow_pool_counts_long_lived_skip_concurrently():
    # Interval-overlap over each single-op stage's op-step interval reproduces
    # the same concurrent peak (3064) the closed-closed per-step model gave for
    # this all-single-op graph, so this expectation is unchanged by the
    # multi-op interval-overlap fix.
    expected_concurrent_peak = 3064
    old_coarse_peak = 2064
    assert expected_concurrent_peak > old_coarse_peak

    ag = build_long_lived_skip_graph()
    ag.budget = MemoryBudget(fast=ag.budget.fast, slow=expected_concurrent_peak)
    usage = slow_pool_usage(ag)
    assert usage.slow_peak_bytes == expected_concurrent_peak
    assert usage.overflow_stage_ids == ()

    ag.budget = MemoryBudget(fast=ag.budget.fast, slow=expected_concurrent_peak - 1)
    usage_tight = slow_pool_usage(ag)
    assert usage_tight.overflow_stage_ids != ()


def build_multi_op_tiled_stage_graph() -> AnalyzedGraph:
    """One tiled stage of two ops: Conv(in)->c, Sigmoid(c)->out. Both ``in``
    and ``out`` are stage-boundary tensors (equal size); ``c`` is intra-stage
    (produced and consumed inside the stage), so it is not slow-resident.

    The stage is tiled: peak_bytes 100_000 exceeds the fast pool (64). PSRAM is
    freed at STAGE granularity, but under the OLD per-op-step sampling ``in``
    dies at the Conv step (step 0) and ``out`` is born only at the Sigmoid step
    (step 1), so no single sampled step counts both and the stage reports
    max(in, out). Interval-overlap over the stage op-step interval [0, 1]
    counts both, reporting in + out.

    Sizes (INT8, so num_elements == size_bytes): in=1000, out=1000.
    """
    conv = OpNode(name="conv", op_type="Conv", inputs=["in"], outputs=["c"], step=0)
    sig = OpNode(name="sig", op_type="Sigmoid", inputs=["c"], outputs=["out"], step=1)
    stage = Stage(
        stage_id=0, op_indices=[0, 1],
        input_tensors=["in"], output_tensors=["out"],
        peak_bytes=100_000,
    )
    return AnalyzedGraph(
        ops=[conv, sig],
        stages=[stage],
        model_inputs=["in"],
        model_outputs=["out"],
        tensors={
            "in": TensorInfo("in", (1, 1000), TensorProto.INT8),
            "c": TensorInfo("c", (1, 1000), TensorProto.INT8),
            "out": TensorInfo("out", (1, 1000), TensorProto.INT8),
        },
        budget=MemoryBudget(fast=64),
    )


def test_slow_pool_counts_multi_op_stage_boundaries_concurrently():
    in_bytes = 1000
    out_bytes = 1000
    concurrent = in_bytes + out_bytes           # 2000, interval-overlap
    per_op_step = max(in_bytes, out_bytes)       # 1000, the old under-count
    assert concurrent > per_op_step

    ag = build_multi_op_tiled_stage_graph()
    ag.budget = MemoryBudget(fast=ag.budget.fast, slow=concurrent)
    usage = slow_pool_usage(ag)
    assert usage.slow_peak_bytes == concurrent
    assert usage.overflow_stage_ids == ()

    # A slow budget just below in+out must be rejected. The old per-op-step
    # model reports only max(in, out) and would have wrongly accepted it,
    # letting a plan that overflows PSRAM through the fail-closed check.
    ag.budget = MemoryBudget(fast=ag.budget.fast, slow=concurrent - 1)
    assert slow_pool_usage(ag).overflow_stage_ids != ()

    # Just above in+out fits.
    ag.budget = MemoryBudget(fast=ag.budget.fast, slow=concurrent + 1)
    assert slow_pool_usage(ag).overflow_stage_ids == ()
