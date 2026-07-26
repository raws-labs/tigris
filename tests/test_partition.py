"""Tests for tigris.analysis.partition_temporal."""

from tigris.graph.ir import AnalyzedGraph, OpNode, TensorLifetime
from tigris.loaders import load_model
from tigris.analysis.lifetime import compute_lifetimes
from tigris.analysis.memory import compute_memory_timeline
from tigris.analysis.partition_temporal import partition_temporal


def _full_pipeline(path):
    ag = load_model(path)
    ag = compute_lifetimes(ag)
    ag = compute_memory_timeline(ag)
    return ag


class _CountingLifetimes(dict):
    """Record full lifetime scans without changing mapping behavior."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.values_calls = 0

    def values(self):
        self.values_calls += 1
        return super().values()


def _linear_graph(num_ops, *, tensor_size=32):
    ops = []
    lifetimes = {
        "input": TensorLifetime("input", -1, 0, tensor_size),
    }
    for step in range(num_ops):
        input_name = "input" if step == 0 else f"t{step - 1}"
        output_name = "output" if step == num_ops - 1 else f"t{step}"
        ops.append(
            OpNode(
                name=f"relu{step}",
                op_type="Relu",
                inputs=[input_name],
                outputs=[output_name],
                step=step,
            )
        )
        death_step = num_ops if step == num_ops - 1 else step + 1
        lifetimes[output_name] = TensorLifetime(
            output_name, step, death_step, tensor_size
        )
    return AnalyzedGraph(ops=ops, lifetimes=lifetimes)


def test_single_stage_large_budget(linear_3op_path):
    ag = _full_pipeline(linear_3op_path)
    ag = partition_temporal(ag, budget=1024 * 1024)  # 1 MiB - everything fits

    assert len(ag.stages) == 1
    assert ag.stages[0].peak_bytes <= 1024 * 1024


def test_all_ops_assigned(linear_3op_path):
    ag = _full_pipeline(linear_3op_path)
    ag = partition_temporal(ag, budget=1024 * 1024)

    assigned = set()
    for s in ag.stages:
        assigned.update(s.op_indices)
    assert assigned == set(range(len(ag.ops)))


def test_multiple_stages_tight_budget(diamond_path):
    ag = _full_pipeline(diamond_path)
    # Diamond: at the Add step, left (512B) + right (512B) are both live = 1024B.
    # With a budget of 512 bytes, should force multiple stages.
    ag = partition_temporal(ag, budget=512)

    assert len(ag.stages) >= 2


def test_stage_inputs_outputs(large_activations_path):
    ag = _full_pipeline(large_activations_path)
    ag = partition_temporal(ag, budget=8192)

    # First stage should have model inputs but no stage inputs from prior stages
    # Later stages should have input_tensors (things they need from slow memory)
    if len(ag.stages) > 1:
        # Non-first stages must have at least one input tensor
        for s in ag.stages[1:]:
            assert len(s.input_tensors) >= 1, f"Stage {s.stage_id} has no inputs"


def test_oversized_tensor_warning(large_activations_path):
    ag = _full_pipeline(large_activations_path)
    # Budget smaller than a single tensor (4096 bytes)
    ag = partition_temporal(ag, budget=2048)

    # Should still produce stages (one op each) with warnings
    warnings = [w for s in ag.stages for w in s.warnings]
    assert len(warnings) > 0


def test_ops_have_stage_assignment(linear_3op_path):
    ag = _full_pipeline(linear_3op_path)
    ag = partition_temporal(ag, budget=1024 * 1024)

    for op in ag.ops:
        assert op.stage >= 0


def test_partition_scans_lifetimes_constant_number_of_times():
    """Guard against rescanning every tensor for every candidate stage."""
    ag = _linear_graph(400)
    ag.lifetimes = _CountingLifetimes(ag.lifetimes)

    ag = partition_temporal(ag, budget=64)

    assert len(ag.stages) == 1
    assert ag.stages[0].peak_bytes == 64
    assert ag.lifetimes.values_calls <= 2


def test_partition_handles_thousands_of_tight_budget_stages():
    """Large graphs remain practical even when every op needs its own stage."""
    ag = _linear_graph(2000)

    ag = partition_temporal(ag, budget=32)

    assert len(ag.stages) == 2000
    assert all(stage.peak_bytes == 64 for stage in ag.stages)
    assert all(len(stage.warnings) == 1 for stage in ag.stages)
