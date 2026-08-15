"""Tests for tigris.analysis.partition_spatial - op classification, receptive field, tile solver."""

import pytest
import yaml
import numpy as np
from onnx import TensorProto, helper, numpy_helper

from tigris.graph.ir import AnalyzedGraph, OpNode, Stage, TensorInfo
from tigris.loaders import load_model
from tigris.analysis.lifetime import compute_lifetimes
from tigris.analysis.memory import compute_memory_timeline
from tigris.analysis.partition_temporal import partition_temporal
from tigris.analysis.partition_spatial import (
    TileCategory,
    classify_op,
    compute_receptive_field,
    partition_spatial,
    _stage_tile_axis,
)
from tigris.analysis.validation import validate_memory_plan
from tigris.emitters.yaml import emit_yaml_str
from tigris.fixtures import build_tcn


def _write_conv1d(path, length=32):
    model_input = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, 2, length]
    )
    model_output = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, 3, length]
    )
    weights = numpy_helper.from_array(
        np.linspace(-0.75, 0.75, 18, dtype=np.float32).reshape(3, 2, 3),
        "weights",
    )
    bias = numpy_helper.from_array(
        np.array([0.1, -0.2, 0.3], dtype=np.float32), "bias"
    )
    graph = helper.make_graph(
        [
            helper.make_node(
                "Conv",
                ["input", "weights", "bias"],
                ["output"],
                name="conv1d",
                kernel_shape=[3],
                pads=[1, 1],
            )
        ],
        "conv1d_tiling",
        [model_input],
        [model_output],
        [weights, bias],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)]
    )
    path.write_bytes(model.SerializeToString())
    return path


def _write_rank3_pointwise(path, op_type, length=64):
    inputs = [
        helper.make_tensor_value_info(
            "left", TensorProto.FLOAT, [1, 4, length]
        )
    ]
    node_inputs = ["left"]
    if op_type in {"Add", "Mul"}:
        inputs.append(
            helper.make_tensor_value_info(
                "right", TensorProto.FLOAT, [1, 4, length]
            )
        )
        node_inputs.append("right")
    output = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, 4, length]
    )
    model = helper.make_model(
        helper.make_graph(
            [helper.make_node(op_type, node_inputs, ["output"], name="pointwise")],
            f"rank3_{op_type.lower()}",
            inputs,
            [output],
        ),
        opset_imports=[helper.make_opsetid("", 17)],
    )
    path.write_bytes(model.SerializeToString())
    return path


def _full_pipeline(path, budget=0):
    ag = load_model(path)
    ag = compute_lifetimes(ag)
    ag = compute_memory_timeline(ag)
    if budget > 0:
        ag = partition_temporal(ag, budget)
        ag = partition_spatial(ag)
    return ag


# Op classification


class TestClassifyOp:
    def test_conv_is_conv(self):
        assert classify_op("Conv") == TileCategory.CONV

    def test_relu_is_pointwise(self):
        assert classify_op("Relu") == TileCategory.POINTWISE

    def test_flatten_is_untileable(self):
        assert classify_op("Flatten") == TileCategory.UNTILEABLE

    def test_unknown_is_untileable(self):
        assert classify_op("MyCustomOp") == TileCategory.UNTILEABLE

    def test_maxpool_is_pool(self):
        assert classify_op("MaxPool") == TileCategory.POOL

    def test_averagepool_is_pool(self):
        assert classify_op("AveragePool") == TileCategory.POOL

    @pytest.mark.parametrize(
        "op_type",
        [
            "BatchNormalization",
            "GlobalAveragePool",
            "Resize",
        ],
    )
    def test_operators_without_schema_v4_height_contract_are_untileable(
        self, op_type
    ):
        assert classify_op(op_type) == TileCategory.UNTILEABLE

    def test_gemm_is_untileable(self):
        assert classify_op("Gemm") == TileCategory.UNTILEABLE


# Receptive field computation


class TestReceptiveField:
    def test_single_3x3_conv(self):
        """A single 3x3 conv has RF=3 on both axes."""
        ops = [OpNode(name="c", op_type="Conv", inputs=[], outputs=[],
                      attrs={"kernel_shape": [3, 3], "strides": [1, 1]})]
        rf_h, rf_w = compute_receptive_field(ops)
        assert rf_h == 3
        assert rf_w == 3

    def test_two_3x3_convs(self):
        """Two stacked 3x3 convs have RF=5 on both axes."""
        ops = [
            OpNode(name="c0", op_type="Conv", inputs=[], outputs=[],
                   attrs={"kernel_shape": [3, 3], "strides": [1, 1]}),
            OpNode(name="c1", op_type="Conv", inputs=[], outputs=[],
                   attrs={"kernel_shape": [3, 3], "strides": [1, 1]}),
        ]
        rf_h, rf_w = compute_receptive_field(ops)
        assert rf_h == 5
        assert rf_w == 5

    def test_conv_stride2_pool(self):
        """Conv3x3(s=1) + MaxPool2x2(s=2) + Conv3x3(s=1), symmetric kernel/stride.

        reversed: conv1(k=3,s=1): rf=3, j=1 -> pool(k=2,s=2): rf=4, j=2 -> conv0(k=3,s=1): rf=8, j=2
        Kernel and stride are symmetric across height/width, so rf_h == rf_w == 8.
        """
        ops = [
            OpNode(name="c0", op_type="Conv", inputs=[], outputs=[],
                   attrs={"kernel_shape": [3, 3], "strides": [1, 1]}),
            OpNode(name="p", op_type="MaxPool", inputs=[], outputs=[],
                   attrs={"kernel_shape": [2, 2], "strides": [2, 2]}),
            OpNode(name="c1", op_type="Conv", inputs=[], outputs=[],
                   attrs={"kernel_shape": [3, 3], "strides": [1, 1]}),
        ]
        rf_h, rf_w = compute_receptive_field(ops)
        assert rf_h == 8
        assert rf_w == 8

    def test_dilated_conv(self):
        """Conv3x3 with dilation=2: effective_k = 2*(3-1)+1 = 5, RF=5 on both axes."""
        ops = [OpNode(name="c", op_type="Conv", inputs=[], outputs=[],
                      attrs={"kernel_shape": [3, 3], "strides": [1, 1],
                             "dilations": [2, 2]})]
        rf_h, rf_w = compute_receptive_field(ops)
        assert rf_h == 5
        assert rf_w == 5

    def test_pointwise_passthrough(self):
        """Pointwise ops (Relu) don't change RF."""
        ops = [
            OpNode(name="c", op_type="Conv", inputs=[], outputs=[],
                   attrs={"kernel_shape": [3, 3], "strides": [1, 1]}),
            OpNode(name="r", op_type="Relu", inputs=[], outputs=[], attrs={}),
        ]
        rf_h, rf_w = compute_receptive_field(ops)
        assert rf_h == 3  # same as single conv
        assert rf_w == 3

    def test_only_pointwise_rf_is_1(self):
        """A chain of only pointwise ops has RF=1 on both axes."""
        ops = [
            OpNode(name="r0", op_type="Relu", inputs=[], outputs=[], attrs={}),
            OpNode(name="r1", op_type="Add", inputs=[], outputs=[], attrs={}),
        ]
        rf_h, rf_w = compute_receptive_field(ops)
        assert rf_h == 1
        assert rf_w == 1


# Integration: full pipeline with ONNX fixtures


class TestTilingIntegration:
    def test_tileable_stage_gets_plan(self, conv_relu_chain_path):
        """Conv-Relu-Conv chain should be fully tileable with a tight budget."""
        ag = _full_pipeline(conv_relu_chain_path, budget=1024)

        # At least one stage should have a tile plan
        tiled = [s for s in ag.stages if s.tile_plan is not None]
        assert len(tiled) >= 1

        # Plans should be tileable
        for s in tiled:
            tp = s.tile_plan
            assert tp.tileable is True
            assert tp.tile_height >= 1
            assert tp.num_tiles >= 1
            assert tp.halo >= 0
            assert tp.receptive_field >= 1

    def test_untileable_stage_flagged(self, conv_with_flatten_path):
        """Conv-Relu-Flatten should flag the stage containing Flatten as untileable."""
        ag = _full_pipeline(conv_with_flatten_path, budget=256)

        # Find stages with tile plans
        tiled = [s for s in ag.stages if s.tile_plan is not None]
        assert len(tiled) >= 1

        # At least one should be untileable
        untileable = [s for s in tiled if not s.tile_plan.tileable]
        assert len(untileable) >= 1
        for s in untileable:
            assert len(s.tile_plan.untileable_ops) >= 1

    def test_no_tiling_when_fits(self, conv_relu_chain_path):
        """With a large budget, no stages should need tiling."""
        ag = _full_pipeline(conv_relu_chain_path, budget=10 * 1024 * 1024)

        # No stage should have a tile plan
        for s in ag.stages:
            assert s.tile_plan is None

    def test_conv_pool_chain_rf(self, conv_pool_chain_path):
        """Conv-Relu-Pool-Conv chain: stages with conv/pool ops should have RF > 1."""
        ag = _full_pipeline(conv_pool_chain_path, budget=512)

        tiled = [s for s in ag.stages if s.tile_plan is not None and s.tile_plan.tileable]
        assert len(tiled) >= 1

        # Find stages that contain a Conv or Pool op
        spatial_stages = []
        for s in tiled:
            stage_ops = [ag.ops[i] for i in s.op_indices]
            has_spatial = any(op.op_type in ("Conv", "MaxPool") for op in stage_ops)
            if has_spatial:
                spatial_stages.append(s)

        for s in spatial_stages:
            assert s.tile_plan.receptive_field > 1

    def test_conv1d_uses_serialized_length_axis(self, tmp_path):
        ag = _full_pipeline(_write_conv1d(tmp_path / "conv1d.onnx"), budget=128)

        assert len(ag.stages) == 1
        tile_plan = ag.stages[0].tile_plan
        assert tile_plan is not None
        assert tile_plan.tileable
        assert tile_plan.axis == 1
        assert tile_plan.original_height == 32
        assert tile_plan.num_tiles > 1

    @pytest.mark.parametrize("op_type", ["Tanh", "Sigmoid", "Add", "Mul"])
    def test_rank3_pointwise_uses_serialized_length_axis(
        self, tmp_path, op_type
    ):
        model = _write_rank3_pointwise(
            tmp_path / f"rank3_{op_type.lower()}.onnx", op_type
        )
        ag = _full_pipeline(model, budget=256)

        assert len(ag.stages) == 1
        tile_plan = ag.stages[0].tile_plan
        assert tile_plan is not None
        assert tile_plan.tileable
        assert tile_plan.axis == 1
        assert tile_plan.original_height == 64
        assert tile_plan.num_tiles > 1

    @pytest.mark.parametrize("op_type", ["Relu", "Relu6", "Sigmoid", "Tanh"])
    def test_rank3_unary_may_compose_with_one_conv1d(self, op_type):
        ops = [
            OpNode("conv", "Conv1D", ["input"], ["mid"]),
            OpNode("pointwise", op_type, ["mid"], ["output"]),
        ]
        graph = AnalyzedGraph(
            ops=ops,
            stages=[
                Stage(
                    stage_id=0,
                    op_indices=[0, 1],
                    input_tensors=["input"],
                    output_tensors=["output"],
                )
            ],
            tensors={
                name: TensorInfo(name, (1, 4, 64), TensorProto.FLOAT)
                for name in ("input", "mid", "output")
            },
        )

        assert _stage_tile_axis(graph, graph.stages[0], ops) == 1

    @pytest.mark.parametrize("op_type", ["Add", "Mul"])
    def test_rank3_binary_does_not_compose_with_conv1d(self, op_type):
        ops = [
            OpNode("conv", "Conv1D", ["input"], ["mid"]),
            OpNode("binary", op_type, ["mid", "residual"], ["output"]),
        ]
        stage = Stage(
            stage_id=0,
            op_indices=[0, 1],
            input_tensors=["input", "residual"],
            output_tensors=["output"],
        )
        graph = AnalyzedGraph(
            ops=ops,
            stages=[stage],
            tensors={
                name: TensorInfo(name, (1, 4, 64), TensorProto.FLOAT)
                for name in ("input", "mid", "residual", "output")
            },
        )

        assert _stage_tile_axis(graph, stage, ops) == 0

    def test_rank3_concat_remains_untileable(self):
        op = OpNode("concat", "Concat", ["left", "right"], ["output"])
        stage = Stage(
            stage_id=0,
            op_indices=[0],
            input_tensors=["left", "right"],
            output_tensors=["output"],
        )
        graph = AnalyzedGraph(
            ops=[op],
            stages=[stage],
            tensors={
                "left": TensorInfo("left", (1, 2, 64), TensorProto.FLOAT),
                "right": TensorInfo("right", (1, 2, 64), TensorProto.FLOAT),
                "output": TensorInfo("output", (1, 4, 64), TensorProto.FLOAT),
            },
        )

        assert _stage_tile_axis(graph, stage, [op]) == 0

    def test_tcn_rank3_pointwise_stages_fit_16k(self, tmp_path):
        model_path = tmp_path / "tcn.onnx"
        model_path.write_bytes(build_tcn().SerializeToString())

        ag = _full_pipeline(model_path, budget=16 * 1024)
        tiled_pointwise = {
            ag.ops[stage.op_indices[0]].op_type: stage.tile_plan
            for stage in ag.stages
            if len(stage.op_indices) == 1
            and ag.ops[stage.op_indices[0]].op_type
            in {"Tanh", "Sigmoid", "Mul"}
        }

        assert set(tiled_pointwise) == {"Tanh", "Sigmoid", "Mul"}
        assert all(
            tile_plan is not None and tile_plan.tileable
            for tile_plan in tiled_pointwise.values()
        )
        validation = validate_memory_plan(ag)
        assert validation.feasible, [issue.describe() for issue in validation.issues]
        assert validation.scheduled_peak_bytes <= 16 * 1024


# YAML includes tile_plan


class TestTilingYaml:
    def test_yaml_includes_tile_plan(self, conv_relu_chain_path):
        """YAML output should include tile_plan for oversized stages."""
        ag = _full_pipeline(conv_relu_chain_path, budget=1024)
        text = emit_yaml_str(ag)
        plan = yaml.safe_load(text)

        if "stages" in plan:
            stages_with_tp = [s for s in plan["stages"] if "tile_plan" in s]
            assert len(stages_with_tp) >= 1
            for s in stages_with_tp:
                tp = s["tile_plan"]
                assert "tileable" in tp
                assert "tile_height" in tp
                assert "halo" in tp


# Loader extracts attrs


class TestLoaderAttrs:
    def test_conv_has_kernel_shape(self, conv_relu_chain_path):
        """Conv ops should have kernel_shape extracted into attrs."""
        ag = load_model(conv_relu_chain_path)
        conv_ops = [op for op in ag.ops if op.op_type == "Conv"]
        assert len(conv_ops) >= 1
        for op in conv_ops:
            assert "kernel_shape" in op.attrs
            assert op.attrs["kernel_shape"] == [3, 3]

    def test_conv_has_strides(self, conv_relu_chain_path):
        """Conv ops should have strides extracted into attrs."""
        ag = load_model(conv_relu_chain_path)
        conv_ops = [op for op in ag.ops if op.op_type == "Conv"]
        for op in conv_ops:
            assert "strides" in op.attrs
            assert op.attrs["strides"] == [1, 1]

    def test_pool_has_kernel_shape(self, conv_pool_chain_path):
        """MaxPool ops should have kernel_shape extracted."""
        ag = load_model(conv_pool_chain_path)
        pool_ops = [op for op in ag.ops if op.op_type == "MaxPool"]
        assert len(pool_ops) == 1
        assert pool_ops[0].attrs["kernel_shape"] == [2, 2]
        assert pool_ops[0].attrs["strides"] == [2, 2]

    def test_relu_fused_into_conv(self, conv_relu_chain_path):
        """Relu should be absorbed into preceding Conv as fused_activation."""
        ag = load_model(conv_relu_chain_path)
        relu_ops = [op for op in ag.ops if op.op_type == "Relu"]
        assert len(relu_ops) == 0  # all Relu ops fused
        fused = [op for op in ag.ops if op.attrs.get("fused_activation") == "Relu"]
        assert len(fused) >= 1
