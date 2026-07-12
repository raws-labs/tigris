"""Tests for the weights pipeline: extraction, binary round-trip, op references."""


import numpy as np

from tigris.analysis.lifetime import compute_lifetimes
from tigris.analysis.memory import compute_memory_timeline
from tigris.analysis.partition_temporal import partition_temporal
from tigris.analysis.partition_spatial import partition_spatial
from tigris.emitters.binary.defs import NO_WEIGHT
from tigris.emitters.binary.reader import read_binary_plan
from tigris.emitters.binary.writer import emit_binary_bytes
from tigris.graph.ir import AnalyzedGraph, OpNode, TensorInfo
from tigris.loaders import load_model


def _full_pipeline(path, budget=0):
    ag = load_model(path)
    ag = compute_lifetimes(ag)
    ag = compute_memory_timeline(ag)
    if budget > 0:
        ag = partition_temporal(ag, budget)
        ag = partition_spatial(ag)
    return ag


# Weight extraction


def test_weight_data_populated(conv_relu_chain_path):
    """ONNX loader should populate weight_data with numpy arrays."""
    ag = load_model(conv_relu_chain_path)

    assert len(ag.weight_data) > 0
    for name, arr in ag.weight_data.items():
        assert isinstance(arr, np.ndarray)
        assert name in ag.tensors
        assert ag.tensors[name].is_constant


def test_weight_shapes_match(conv_relu_chain_path):
    """Weight array shapes should match tensor info shapes."""
    ag = load_model(conv_relu_chain_path)

    for name, arr in ag.weight_data.items():
        assert arr.shape == ag.tensors[name].shape


def test_no_weights_for_activation_only(diamond_path):
    """A graph with no initializers should have empty weight_data."""
    ag = load_model(diamond_path)
    assert len(ag.weight_data) == 0


# Binary round-trip with weights


def test_weights_in_binary(conv_relu_chain_path):
    """Binary plan should contain a weights section when graph has weights."""
    ag = _full_pipeline(conv_relu_chain_path)
    data = emit_binary_bytes(ag)
    plan = read_binary_plan(data)

    assert plan["num_weights"] == len(ag.weight_data)
    assert len(plan["weights"]) == plan["num_weights"]


def test_weight_entries_correct(conv_relu_chain_path):
    """Weight entries should have correct names and sizes."""
    ag = _full_pipeline(conv_relu_chain_path)
    data = emit_binary_bytes(ag)
    plan = read_binary_plan(data)

    weight_names = set(ag.weight_data.keys())
    plan_names = {w["name"] for w in plan["weights"]}
    assert plan_names == weight_names

    for w in plan["weights"]:
        name = w["name"]
        expected_size = ag.weight_data[name].astype(np.float32).nbytes
        assert w["size_bytes"] == expected_size


def test_weight_blob_data(conv_relu_chain_path):
    """Actual weight bytes in the blob should match the original arrays."""
    ag = _full_pipeline(conv_relu_chain_path)
    data = emit_binary_bytes(ag)
    plan = read_binary_plan(data)

    for w in plan["weights"]:
        name = w["name"]
        blob_start = w["blob_base"] + w["offset"]
        blob_bytes = data[blob_start:blob_start + w["size_bytes"]]

        expected = ag.weight_data[name].astype(np.float32).tobytes()
        assert blob_bytes == expected, f"Weight blob mismatch for {name}"


def test_full_shape_elementwise_constant_is_emitted_in_nhwc_layout():
    constant = np.arange(12, dtype=np.float32).reshape(1, 2, 2, 3)
    graph = AnalyzedGraph(
        ops=[
            OpNode(
                name="constant_first_add",
                op_type="Add",
                inputs=["constant", "input"],
                outputs=["output"],
            )
        ],
        tensors={
            "input": TensorInfo("input", constant.shape, 1),
            "constant": TensorInfo(
                "constant", constant.shape, 1, is_constant=True
            ),
            "output": TensorInfo("output", constant.shape, 1),
        },
        weight_data={"constant": constant},
        model_inputs=["input"],
        model_outputs=["output"],
    )

    data = emit_binary_bytes(graph)
    plan = read_binary_plan(data)
    weight = plan["weights"][0]
    start = weight["blob_base"] + weight["offset"]
    actual = data[start : start + weight["size_bytes"]]

    assert actual == constant.transpose(0, 2, 3, 1).tobytes()
    assert plan["ops"][0]["weight_idx"] == 0


def test_no_weights_section_when_empty(diamond_path):
    """Binary plan should not have a weights section for activation-only graphs."""
    ag = _full_pipeline(diamond_path)
    data = emit_binary_bytes(ag)
    plan = read_binary_plan(data)

    assert plan["num_weights"] == 0
    assert len(plan["weights"]) == 0


# Op-to-weight references


def test_conv_ops_have_weights(conv_relu_chain_path):
    """Conv ops should reference weight entries."""
    ag = _full_pipeline(conv_relu_chain_path)
    data = emit_binary_bytes(ag)
    plan = read_binary_plan(data)

    from tigris.emitters.binary.defs import OP_TYPE_MAP
    for op in plan["ops"]:
        if op["op_type"] == OP_TYPE_MAP["Conv"]:
            assert op["weight_idx"] != NO_WEIGHT, "Conv op should have weight_idx"


def test_relu_has_no_weights(conv_relu_chain_path):
    """Relu ops should have NO_WEIGHT for both weight and bias."""
    ag = _full_pipeline(conv_relu_chain_path)
    data = emit_binary_bytes(ag)
    plan = read_binary_plan(data)

    from tigris.emitters.binary.defs import OP_TYPE_MAP
    for op in plan["ops"]:
        if op["op_type"] == OP_TYPE_MAP["Relu"]:
            assert op["weight_idx"] == NO_WEIGHT
            assert op["bias_idx"] == NO_WEIGHT


def test_conv_weight_bias_distinct(conv_relu_chain_path):
    """Conv weight_idx and bias_idx should point to different weight entries."""
    ag = _full_pipeline(conv_relu_chain_path)
    data = emit_binary_bytes(ag)
    plan = read_binary_plan(data)

    from tigris.emitters.binary.defs import OP_TYPE_MAP
    for op in plan["ops"]:
        if op["op_type"] == OP_TYPE_MAP["Conv"]:
            if op["bias_idx"] != NO_WEIGHT:
                assert op["weight_idx"] != op["bias_idx"]


def test_all_fixtures_with_weights(
    conv_relu_chain_path, conv_with_flatten_path, conv_pool_chain_path,
):
    """All fixtures with weights should produce valid binaries."""
    for path in [conv_relu_chain_path, conv_with_flatten_path, conv_pool_chain_path]:
        ag = _full_pipeline(path, budget=8192)
        data = emit_binary_bytes(ag)
        plan = read_binary_plan(data)
        assert plan["num_weights"] > 0
        assert len(plan["weights"]) == plan["num_weights"]
