"""Tests for tigris.analysis.memory."""

from tigris.loaders import load_model
from tigris.analysis.lifetime import compute_lifetimes
from tigris.analysis.memory import compute_memory_timeline


def test_timeline_length(linear_3op_path):
    ag = load_model(linear_3op_path)
    ag = compute_lifetimes(ag)
    ag = compute_memory_timeline(ag)

    assert len(ag.timeline) == len(ag.ops)


def test_peak_positive(linear_3op_path):
    ag = load_model(linear_3op_path)
    ag = compute_lifetimes(ag)
    ag = compute_memory_timeline(ag)

    assert ag.peak_memory_bytes > 0


def test_peak_at_least_one_tensor(linear_3op_path):
    ag = load_model(linear_3op_path)
    ag = compute_lifetimes(ag)
    ag = compute_memory_timeline(ag)

    # Peak should be at least one activation tensor size (256 bytes for [1,64] float32)
    assert ag.peak_memory_bytes >= 256


def test_diamond_peak_higher(diamond_path):
    """Diamond has overlapping lifetimes - peak should reflect multiple live tensors."""
    ag = load_model(diamond_path)
    ag = compute_lifetimes(ag)
    ag = compute_memory_timeline(ag)

    single = 1 * 128 * 4  # 512 bytes
    # At the Add step, left + right + output (or at least left + right) should be live
    assert ag.peak_memory_bytes >= 2 * single


def test_no_negative_memory(linear_3op_path):
    ag = load_model(linear_3op_path)
    ag = compute_lifetimes(ag)
    ag = compute_memory_timeline(ag)

    for snap in ag.timeline:
        assert snap.live_bytes >= 0


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


def test_final_output_counted_in_peak(linear_3op_path):
    """The final op's output is live at the last step and must be counted, not
    dropped because its alive_from fell past the end of the sweep."""
    ag = load_model(linear_3op_path)
    ag = compute_lifetimes(ag)
    ag = compute_memory_timeline(ag)
    assert "output" in ag.timeline[-1].live_tensors
