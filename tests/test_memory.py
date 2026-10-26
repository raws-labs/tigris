"""Tests for tigris.analysis.memory."""

from tigris.graph.ir import AnalyzedGraph, OpNode, TensorLifetime
from tigris.loaders import load_model
from tigris.analysis.lifetime import compute_lifetimes
from tigris.analysis.memory import TENSOR_ALIGN, align_up, compute_memory_timeline


def test_diamond_peak_higher(diamond_path):
    """Diamond has overlapping lifetimes - peak should reflect multiple live tensors."""
    ag = load_model(diamond_path)
    ag = compute_lifetimes(ag)
    ag = compute_memory_timeline(ag)

    # At the Add step, left + right + output (512 bytes each) are live.
    assert ag.peak_memory_bytes == 1536


def test_peak_counts_output_coresident_with_inputs(linear_3op_path):
    """Each op allocates its output BEFORE its consumed inputs are freed (the
    runtime's exec_stage_normal order: alloc out, run kernel, free inputs). So at
    every producing step the output co-resides with its dying input. For the
    linear 3-op chain (all [1,64] f32 = 256 B) the peak is two tensors = 512 B,
    not one. Regression guard for the birth_step+1 undercount."""
    ag = load_model(linear_3op_path)
    ag = compute_lifetimes(ag)
    ag = compute_memory_timeline(ag)
    assert ag.peak_memory_bytes == 512


def test_final_output_counted_in_peak():
    """The final op's output is live at the last step and must be counted, not
    dropped because its alive_from fell past the end of the sweep."""
    ag = AnalyzedGraph(
        ops=[OpNode("a", "Relu", ["input"], ["mid"]), OpNode("b", "Relu", ["mid"], ["output"])],
        lifetimes={
            "input": TensorLifetime("input", -1, 0, 4),
            "mid": TensorLifetime("mid", 0, 1, 4),
            "output": TensorLifetime("output", 1, 2, 100),
        },
    )

    ag = compute_memory_timeline(ag)

    # The last step holds mid and output, each rounded to TENSOR_ALIGN: 32 + 128.
    assert ag.peak_memory_bytes == 160


def test_align_up_rounds_to_multiples():
    assert TENSOR_ALIGN == 32
    assert [align_up(n) for n in (0, 1, 31, 32, 33, 64)] == [0, 32, 32, 32, 64, 64]
    assert [align_up(n, 4) for n in (0, 1, 4, 5)] == [0, 4, 4, 8]


def test_peak_accounts_for_allocator_alignment():
    """The planner must model physical bump allocations, not raw byte sums."""
    ag = AnalyzedGraph(
        ops=[OpNode("op", "Relu", ["input"], ["output"])],
        lifetimes={
            "input": TensorLifetime("input", -1, 0, 5),
            "output": TensorLifetime("output", 0, 1, 5),
        },
    )

    ag = compute_memory_timeline(ag)

    # Both values co-reside at the producing step: 2 * align_up(5, 32).
    assert ag.peak_memory_bytes == 64

