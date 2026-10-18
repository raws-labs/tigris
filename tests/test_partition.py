"""Tests for tigris.analysis.partition_temporal."""

from tigris.graph.ir import AnalyzedGraph, Layout, OpNode, TensorInfo, TensorLifetime
from tigris.analysis.lifetime import compute_lifetimes
from tigris.analysis.partition_temporal import partition_temporal

from conftest import full_pipeline


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
    ag = full_pipeline(linear_3op_path)
    ag = partition_temporal(ag, budget=1024 * 1024)  # 1 MiB - everything fits

    assert len(ag.stages) == 1
    assert ag.stages[0].peak_bytes <= 1024 * 1024
    assert set(ag.stages[0].op_indices) == set(range(len(ag.ops)))
    for op in ag.ops:
        assert op.stage >= 0


def test_multiple_stages_tight_budget(diamond_path):
    ag = full_pipeline(diamond_path)
    # Diamond: at the Add step, left (512B) + right (512B) are both live = 1024B.
    # With a budget of 512 bytes, should force multiple stages.
    ag = partition_temporal(ag, budget=512)

    assert len(ag.stages) >= 2


def test_stage_inputs_outputs(large_activations_path):
    ag = full_pipeline(large_activations_path)
    # A budget of one 4096-byte tensor gives every Relu its own stage.
    ag = partition_temporal(ag, budget=4096)

    names = ["input", "t0", "t1", "t2", "t3", "output"]
    assert [(s.input_tensors, s.output_tensors) for s in ag.stages] == [
        ([names[i]], [names[i + 1]]) for i in range(5)
    ]
    assert all(len(s.warnings) == 1 for s in ag.stages)


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


def test_reshape_alias_is_materialized_at_stage_boundary():
    ag = AnalyzedGraph(
        ops=[OpNode(name="reshape", op_type="Reshape", inputs=["x"], outputs=["r"], step=0),
             OpNode(name="relu", op_type="Relu", inputs=["r"], outputs=["y"], step=1)],
        tensors={name: TensorInfo(name=name, shape=shape, dtype=1, layout=Layout.LINEAR)
                 for name, shape in (("x", (1, 4, 8)), ("r", (8, 1, 4)), ("y", (8, 1, 4)))},
        model_inputs=["x"], model_outputs=["y"],
    )
    compute_lifetimes(ag)
    assert ag.lifetimes["r"].size_bytes == 0
    partition_temporal(ag, budget=256, forced_cuts=frozenset({1}))
    assert [s.peak_bytes for s in ag.stages] == [128, 256]
    partition_temporal(ag, budget=256)
    assert len(ag.stages) == 1 and ag.stages[0].peak_bytes == 256
