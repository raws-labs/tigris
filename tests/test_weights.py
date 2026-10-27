"""Tests for the weights pipeline: extraction, binary round-trip, op references."""


import math

import numpy as np
import pytest

from tigris.emitters.binary.defs import NO_WEIGHT, OP_TYPE_MAP
from tigris.emitters.binary.reader import read_binary_plan
from tigris.emitters.binary.writer import _permute_fc_weight_for_nhwc, emit_binary_bytes
from tigris.graph.ir import AnalyzedGraph, OpNode, TensorInfo
from tigris.loaders import load_model
from tigris.pipeline import plan_model


# Weight extraction


def test_weight_data_populated(conv_relu_chain_path):
    """ONNX loader should populate weight_data with arrays of the tensor shapes."""
    ag = load_model(conv_relu_chain_path)

    assert len(ag.weight_data) > 0
    for name, arr in ag.weight_data.items():
        assert isinstance(arr, np.ndarray)
        assert name in ag.tensors
        assert ag.tensors[name].is_constant
        assert arr.shape == ag.tensors[name].shape


def test_no_weights_for_activation_only(diamond_path):
    """A graph with no initializers has no weight_data and no weight entries."""
    assert len(load_model(diamond_path).weight_data) == 0

    plan = read_binary_plan(emit_binary_bytes(plan_model(diamond_path)))
    assert plan["num_weights"] == 0
    assert len(plan["weights"]) == 0


# Binary round-trip with weights


def test_weights_in_binary(conv_relu_chain_path):
    """Weight entries carry the names, sizes and bytes of the original arrays."""
    ag = plan_model(conv_relu_chain_path)
    data = emit_binary_bytes(ag, names=True)
    plan = read_binary_plan(data)

    assert plan["num_weights"] == len(ag.weight_data)
    assert len(plan["weights"]) == plan["num_weights"]
    assert {w["name"] for w in plan["weights"]} == set(ag.weight_data)

    for w in plan["weights"]:
        name = w["name"]
        expected = ag.weight_data[name].astype(np.float32).tobytes()
        assert w["size_bytes"] == len(expected), f"Weight size mismatch for {name}"
        blob_start = w["blob_base"] + w["offset"]
        blob_bytes = data[blob_start:blob_start + w["size_bytes"]]
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


def test_gemm_after_spatial_reshape_preserves_onnx_element_order():
    weights = np.arange(16, dtype=np.float32).reshape(2, 8)
    graph = AnalyzedGraph(
        ops=[
            OpNode(
                name="flattening_reshape",
                op_type="Reshape",
                inputs=["input"],
                outputs=["vector"],
            ),
            OpNode(
                name="classifier",
                op_type="Gemm",
                inputs=["vector", "weights"],
                outputs=["output"],
                attrs={"transB": 1},
            ),
        ],
        tensors={
            "input": TensorInfo("input", (1, 2, 2, 2), 1),
            "vector": TensorInfo("vector", (1, 8), 1),
            "weights": TensorInfo(
                "weights", weights.shape, 1, is_constant=True
            ),
            "output": TensorInfo("output", (1, 2), 1),
        },
        weight_data={"weights": weights},
        model_inputs=["input"],
        model_outputs=["output"],
    )

    data = emit_binary_bytes(graph)
    plan = read_binary_plan(data)
    weight = plan["weights"][0]
    start = weight["blob_base"] + weight["offset"]
    actual = np.frombuffer(
        data[start : start + weight["size_bytes"]], dtype=np.float32
    ).reshape(2, 8)

    # Runtime order is NHWC: each position alternates its two channels.
    nhwc_to_nchw = [0, 4, 1, 5, 2, 6, 3, 7]
    np.testing.assert_array_equal(actual, weights[:, nhwc_to_nchw])


@pytest.mark.parametrize("shape", [(1, 3, 4, 5), (1, 7, 1, 3), (1, 2, 9), (1, 5, 1), (2, 4, 3)])
def test_fc_weight_on_the_channels_last_flatten_gives_the_model_result(shape):
    columns = math.prod(shape[1:])
    weights = np.random.default_rng(0).integers(-50, 50, (3, columns))
    permuted = _permute_fc_weight_for_nhwc(weights, shape)
    # One input per column: the model flattens it channel-major, the runtime channels-last.
    inputs = np.eye(columns, dtype=np.int64).reshape(columns, *shape[1:])
    channels_last = np.moveaxis(inputs, 1, -1).reshape(columns, columns)
    np.testing.assert_array_equal(channels_last @ permuted.T, inputs.reshape(columns, columns) @ weights.T)


# Op-to-weight references


def test_conv_ops_have_weights(conv_relu_chain_path):
    """Conv ops reference weight entries, with weight and bias distinct."""
    ag = plan_model(conv_relu_chain_path)
    data = emit_binary_bytes(ag)
    plan = read_binary_plan(data)

    for op in plan["ops"]:
        if op["op_type"] == OP_TYPE_MAP["Conv"]:
            assert op["weight_idx"] != NO_WEIGHT, "Conv op should have weight_idx"
            if op["bias_idx"] != NO_WEIGHT:
                assert op["weight_idx"] != op["bias_idx"]


def test_relu_has_no_weights(linear_3op_path):
    """Relu ops should have NO_WEIGHT for both weight and bias."""
    ag = plan_model(linear_3op_path)
    data = emit_binary_bytes(ag)
    plan = read_binary_plan(data)

    relus = [op for op in plan["ops"] if op["op_type"] == OP_TYPE_MAP["Relu"]]
    assert relus
    for op in relus:
        assert op["weight_idx"] == NO_WEIGHT
        assert op["bias_idx"] == NO_WEIGHT
