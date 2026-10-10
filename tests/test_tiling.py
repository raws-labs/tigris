"""Tests for tigris.analysis.partition_spatial - op classification, receptive field, tile solver."""

import pytest
import json
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
from tigris.emitters.plan_json import plan_json_str
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
        ],
    )
    def test_operators_without_schema_v4_height_contract_are_untileable(
        self, op_type
    ):
        assert classify_op(op_type) == TileCategory.UNTILEABLE

    @pytest.mark.parametrize("op_type", ["Resize", "ResizeLinear"])
    def test_resampling_is_its_own_category(self, op_type):
        """Its output has more rows than its input, so the tile loop divides
        to find the source band instead of multiplying."""
        assert classify_op(op_type) == TileCategory.UPSAMPLE

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

    def test_kernel_inferred_from_weight_when_shape_absent(self):
        """A Conv omitting kernel_shape recovers its 3x3 kernel from the weight.

        Mirrors the emitter's inference so a recomputing chain that drops the
        attribute is not silently treated as a 1x1 kernel (understated halo).
        """
        op = OpNode(name="c", op_type="Conv", inputs=["x", "w"], outputs=["y"],
                    attrs={"strides": [1, 1]})
        # ONNX Conv weight is [out_ch, in_ch/group, kH, kW].
        rf_h, rf_w = compute_receptive_field([op], {"w": (8, 4, 3, 3)})
        assert rf_h == 3
        assert rf_w == 3

    def test_kernel_shape_absent_without_weights_falls_back_to_1(self):
        """Negative control: no weight_shapes keeps the prior 1x1 fallback."""
        op = OpNode(name="c", op_type="Conv", inputs=["x", "w"], outputs=["y"],
                    attrs={"strides": [1, 1]})
        rf_h, rf_w = compute_receptive_field([op])
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
        pointwise = {"Tanh", "Sigmoid", "Mul"}
        placed = set()
        for stage in ag.stages:
            held = {ag.ops[i].op_type for i in stage.op_indices} & pointwise
            if not held:
                continue
            placed |= held
            if stage.peak_bytes <= 16 * 1024:
                continue
            # Which ops share a stage is the partitioner's call; that an
            # oversized one carries a tile plan is not.
            assert stage.tile_plan is not None and stage.tile_plan.tileable, (
                f"stage {stage.stage_id} holds {sorted(held)} over budget "
                "without a tile plan"
            )
        assert placed == pointwise

        validation = validate_memory_plan(ag)
        assert validation.feasible, [issue.describe() for issue in validation.issues]
        assert validation.scheduled_peak_bytes <= 16 * 1024


# The JSON plan includes tile_plan


class TestTilingYaml:
    def test_yaml_includes_tile_plan(self, conv_relu_chain_path):
        """YAML output should include tile_plan for oversized stages."""
        ag = _full_pipeline(conv_relu_chain_path, budget=1024)
        text = plan_json_str(ag)
        plan = json.loads(text)

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


def _gated_map_path(tmp_path, channels=16, side=12):
    """GlobalAveragePool -> 1x1 Conv -> Sigmoid gates the map it came from."""
    weight = np.full((channels, channels, 1, 1), 0.1, dtype=np.float32)
    model = helper.make_model(
        helper.make_graph(
            [helper.make_node("GlobalAveragePool", ["input"], ["pooled"]),
             helper.make_node("Conv", ["pooled", "w"], ["fc"], kernel_shape=[1, 1]),
             helper.make_node("Sigmoid", ["fc"], ["gate"]),
             helper.make_node("Mul", ["gate", "input"], ["output"])],
            "gated_map",
            [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, channels, side, side])],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, channels, side, side])],
            [numpy_helper.from_array(weight, "w")]),
        opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    path = tmp_path / "gated_map.onnx"
    path.write_bytes(model.SerializeToString())
    return path


def test_per_channel_operand_stage_is_banded_over_the_full_operand(tmp_path):
    """The gate is one row high and listed first; the band runs over the map."""
    from tigris.cli import _run_pipeline
    from tigris import TILE_AXIS_HEIGHT_OR_LENGTH

    ag, _ = _run_pipeline(str(_gated_map_path(tmp_path)), ("2K",), report_bindings=False)
    stage = next(s for s in ag.stages
                 if any(ag.ops[i].op_type == "Mul" for i in s.op_indices))
    plan = stage.tile_plan
    assert plan is not None and plan.tileable
    assert plan.axis == TILE_AXIS_HEIGHT_OR_LENGTH and plan.tile_width == 0
    assert plan.original_height == 12 and plan.num_tiles > 1
    assert stage.chain_id == 0xFFFF


def test_half_pixel_bilinear_band_reaches_one_row_further():
    """Half-pixel places output row o at (o + 0.5) / s - 0.5, a row ahead of o / s."""
    half = OpNode(name="up", op_type="ResizeLinear", inputs=["x"], outputs=["y"],
                  attrs={"coordinate_transformation_mode": "half_pixel"})
    asym = OpNode(name="up", op_type="ResizeLinear", inputs=["x"], outputs=["y"],
                  attrs={"coordinate_transformation_mode": "asymmetric"})
    nearest = OpNode(name="up", op_type="Resize", inputs=["x"], outputs=["y"])
    assert compute_receptive_field([half]) == (3, 3)
    assert compute_receptive_field([asym]) == (2, 2)
    assert compute_receptive_field([nearest]) == (1, 1)


@pytest.mark.parametrize("kind", ["ReduceMean", "ReduceMax", "ReduceMin", "ReduceSum", "ReduceAll",
                                  "CumSum", "ArgMax", "ArgMin", "Gather", "GatherND", "EmbeddingLookup",
                                  "StridedSlice", "MirrorPad", "ReverseV2", "DynamicUpdateSlice"])
@pytest.mark.parametrize("quantized", [False, True])
def test_independent_operator_band_fits_256_bytes(tmp_path, kind, quantized):
    import onnx
    from scripts.crossrepo_contract import _arg_case, _movement_case, _reduce_all_case, _reduction_case
    from tigris.cli import _run_pipeline

    if kind in {"ArgMax", "ArgMin"}:
        case = _arg_case(kind, 2, quantized, True, tiled=True)
    elif kind == "ReduceAll":
        case = _reduce_all_case(2, True, tiled=True)
    elif kind in {"ReduceMean", "ReduceMax", "ReduceMin", "ReduceSum", "CumSum"}:
        case = _reduction_case(kind, 2, quantized=quantized, tiled=True)
    else:
        case = _movement_case(kind, quantized, 2 if kind == "Gather" else 0, tiled=True)
    path = tmp_path / "band.onnx"
    onnx.save(case.compile_model, path)
    graph, _ = _run_pipeline(str(path), ("256", "16K"))
    stage = next(s for s in graph.stages if any(graph.ops[i].op_type == kind for i in s.op_indices))
    assert stage.tile_plan and stage.tile_plan.tileable and stage.tile_plan.num_tiles > 1
    assert stage.tile_plan.tiled_peak_bytes <= 256
    assert validate_memory_plan(graph).feasible


@pytest.mark.parametrize("kind", ["ReduceMax", "ReduceMin", "ReduceSum", "CumSum", "ArgMax", "ArgMin",
                                  "Gather", "GatherND", "EmbeddingLookup", "StridedSlice", "MirrorPad",
                                  "ReverseV2", "DynamicUpdateSlice"])
def test_band_cannot_cut_a_reduced_indexed_or_modified_axis(kind):
    from tigris.analysis.partition_spatial import _independent_band
    from tigris.graph.ir import Layout

    tensors = {name: TensorInfo(name, (2, 7, 4), dtype=1, layout=Layout.LINEAR) for name in ("x", "y", "u")}
    tensors["indices"] = TensorInfo("indices", (7,), dtype=6, is_constant=True)
    attributes = {"axes": [1]}
    inputs = ["x"]
    if kind in {"Gather", "EmbeddingLookup"}:
        attributes = {"movement": [1, 0, 1, 7]}
        inputs.append("indices")
    elif kind == "GatherND":
        attributes = {"movement": [2, 7, 2]}
        inputs.append("indices")
    elif kind == "StridedSlice":
        attributes = {"movement": [0, 2, 1, 0, 7, 2, 0, 4, 1]}
    elif kind == "MirrorPad":
        attributes = {"movement": [0, 0, 0, 1, 1, 0, 0]}
    elif kind == "ReverseV2":
        attributes = {"movement": [2]}
    elif kind == "DynamicUpdateSlice":
        attributes = {"movement": [0, 1, 0, 2, 6, 4]}
        inputs.append("u")
    op = OpNode("band", kind, inputs, ["y"], attrs=attributes)
    graph = AnalyzedGraph(tensors=tensors, ops=[op])
    assert not _independent_band(graph, op)
    assert not _independent_band(graph, op, row_tiled=False)


@pytest.mark.parametrize("source,target,expected", [
    ((2, 11, 12), (2, 33, 4), (2, 11, 12, 33, 4)),
    ((2, 33, 4), (2, 11, 12), (2, 33, 4, 11, 12)),
    ((1, 3, 3, 8), (1, 3, 3, 2, 2, 2), (1, 3, 24, 3, 24)),
    ((1, 6, 6, 4), (1, 36, 4), (1, 6, 24, 36, 4)),
    ((1, 6, 6, 4), (36, 1, 4), (1, 6, 24, 36, 4)),
    ((36, 1, 4), (1, 6, 6, 4), (1, 36, 4, 6, 24)),
    ((1, 1, 6, 6, 4), (36, 1, 4), (1, 6, 24, 36, 4)),
])
def test_reshape_band_mapping(source, target, expected):
    from tigris.analysis.partition_spatial import _reshape_band
    from tigris.graph.ir import Layout, Stage

    graph = AnalyzedGraph(tensors={name: TensorInfo(name, shape, dtype=1, layout=Layout.LINEAR)
                                  for name, shape in (("x", source), ("y", target))},
                          ops=[OpNode("reshape", "Reshape", ["x"], ["y"])])
    stage = Stage(0, [0], ["x"], ["y"])
    assert _reshape_band(graph, stage) == expected
    stage.chain_len = 2
    stage.chain_id = 0
    assert _reshape_band(graph, stage) is None
    stage.chain_len = 0
    graph.tensors["x"].layout = Layout.SPATIAL
    assert _reshape_band(graph, stage) is None


def test_reshape_band_solver_preserves_integral_endpoints():
    from tigris.analysis.partition_spatial import _reshape_tile_for_bytes

    mapping = (2, 33, 4, 11, 12)
    plan = _reshape_tile_for_bytes(mapping, 256, 32, 4)
    assert plan.tileable and plan.tile_height == 6 and plan.num_tiles == 6
    assert plan.tiled_peak_bytes == 192
    for start in range(0, 33, plan.tile_height):
        end = min(33, start + plan.tile_height)
        assert start * 4 % 12 == 0 and end * 4 % 12 == 0
    refused = _reshape_tile_for_bytes(mapping, 95, 32, 4)
    assert not refused.tileable and refused.min_tile_bytes == 96


@pytest.mark.parametrize("source,target,expected", [
    ((2, 3, 4), (3, 2, 4), (1, 2, 12, 3, 8)),
    ((3, 2, 4), (2, 3, 4), (1, 3, 8, 2, 12)),
])
def test_reshape_refuses_strided_intervals_as_contiguous(source, target, expected):
    from tigris.analysis.partition_spatial import _reshape_band, _reshape_tile_for_bytes
    from tigris.graph.ir import Layout, Stage

    graph = AnalyzedGraph(tensors={name: TensorInfo(name, shape, dtype=1, layout=Layout.LINEAR)
                                  for name, shape in (("x", source), ("y", target))},
                          ops=[OpNode("reshape", "Reshape", ["x"], ["y"])])
    mapping = _reshape_band(graph, Stage(0, [0], ["x"], ["y"]))
    # Axis 1 has multiple leading blocks on each side. Only whole axis-0
    # intervals are contiguous, and their first integral band is the tensor.
    assert mapping == expected
    assert not _reshape_tile_for_bytes(mapping, 64, 32, 4).tileable


def test_contiguous_reshape_rounds_bands_to_integral_endpoints():
    from tigris.analysis.partition_spatial import _reshape_tile_for_bytes

    mapping = (1, 36, 4, 6, 24)
    plan = _reshape_tile_for_bytes(mapping, 255, 32, 4)
    assert plan.tileable and plan.tile_height == 12 and plan.num_tiles == 3
    for start in range(0, 36, plan.tile_height):
        assert start * 4 % 24 == 0
        assert min(36, start + plan.tile_height) * 4 % 24 == 0
    assert not _reshape_tile_for_bytes(mapping, 95, 32, 4).tileable


@pytest.mark.parametrize("kind", ["ReduceMean", "ReduceMax", "ReduceMin", "ReduceSum", "ReduceAll",
                                  "CumSum", "ArgMax", "ArgMin", "Gather", "GatherND", "EmbeddingLookup",
                                  "StridedSlice", "MirrorPad", "ReverseV2", "DynamicUpdateSlice"])
def test_leading_independent_operator_band(kind):
    from tigris.analysis.partition_spatial import _independent_band, _leading_view, _solve_row_tile
    from tigris.graph.ir import Layout, Stage

    shape = (1, 1, 37, 1, 4) if kind in {"GatherND", "EmbeddingLookup"} else (37, 1, 4)
    rank = len(shape)
    tensors = {n: TensorInfo(n, shape, dtype=1, layout=Layout.LINEAR) for n in ("x", "y", "u")}
    tensors["indices"] = TensorInfo("indices", (1,), dtype=6, is_constant=True)
    inputs, attrs = ["x"], {"axes": [rank - 1]}
    if kind in {"Gather", "EmbeddingLookup"}:
        attrs = {"movement": [2 if kind == "Gather" else 0, 0, 1, 1]}
        inputs.append("indices")
    elif kind == "GatherND":
        attrs = {"movement": [3, 1, 1, 2]}
        inputs.append("indices")
    elif kind == "StridedSlice":
        attrs = {"movement": [0, 37, 1, 0, 1, 1, 0, 4, 1]}
    elif kind == "MirrorPad":
        attrs = {"movement": [0, 0, 0, 0, 0, 0, 0]}
    elif kind == "ReverseV2":
        attrs = {"movement": [4]}
    elif kind == "DynamicUpdateSlice":
        attrs = {"movement": [0, 0, 0, 37, 1, 4]}
        inputs.append("u")
    op = OpNode("band", kind, inputs, ["y"], attrs=attrs)
    graph = AnalyzedGraph(tensors=tensors, ops=[op])
    assert _independent_band(graph, op, leading=True)
    stage = Stage(0, [0], [n for n in inputs if n != "indices"], ["y"])
    tile = _solve_row_tile(graph, stage, [op], 256, _leading_view)
    assert tile.tileable and tile.original_height == 37 and tile.num_tiles > 1
    assert tile.tiled_peak_bytes <= 256
    tensors["y"].shape = (2, *shape[1:])
    assert not _independent_band(graph, op, leading=True)


@pytest.mark.parametrize("kind", ["ReduceMean", "ReduceMax", "ReduceMin", "ReduceSum", "ReduceAll",
                                  "CumSum", "ArgMax", "ArgMin", "ReverseV2"])
def test_leading_band_cannot_cut_the_operator_axis(kind):
    from tigris.analysis.partition_spatial import _independent_band
    from tigris.graph.ir import Layout

    tensors = {n: TensorInfo(n, (37, 1, 4), dtype=1, layout=Layout.LINEAR) for n in ("x", "y")}
    attrs = {"movement": [1]} if kind == "ReverseV2" else {"axes": [0]}
    op = OpNode("band", kind, ["x"], ["y"], attrs=attrs)
    assert not _independent_band(AnalyzedGraph(tensors=tensors, ops=[op]), op, leading=True)


@pytest.mark.parametrize("source,target", [((1, 37, 8), (2, 37, 4)), ((2, 37, 4), (1, 37, 8))])
def test_leading_band_requires_one_block_before_its_axis(source, target):
    from tigris.analysis.partition_spatial import _independent_band
    from tigris.graph.ir import Layout

    tensors = {n: TensorInfo(n, shape, dtype=1, layout=Layout.LINEAR)
               for n, shape in (("x", source), ("y", target))}
    op = OpNode("band", "ReverseV2", ["x"], ["y"], attrs={"movement": [4]})
    assert not _independent_band(AnalyzedGraph(tensors=tensors, ops=[op]), op, leading=True)


@pytest.mark.parametrize("combined", [False, True])
def test_leading_bands_do_not_extend_chains_or_combined_stages(combined):
    from tigris.analysis.partition_spatial import _assign_tile_plans
    from tigris.graph.ir import Layout, MemoryBudget, Stage

    tensors = {n: TensorInfo(n, shape, dtype=1, layout=Layout.LINEAR)
               for n, shape in (("x", (37, 1, 4)), ("y", (37, 1, 1)), ("z", (37, 1, 1)))}
    ops = [OpNode("max", "ReduceMax", ["x"], ["y"], attrs={"axes": [2]})]
    if combined:
        ops.append(OpNode("relu", "Relu", ["y"], ["z"]))
    stage = Stage(0, list(range(len(ops))), ["x"], ["z" if combined else "y"], peak_bytes=1024,
                  chain_len=0 if combined else 2)
    graph = AnalyzedGraph(tensors=tensors, ops=ops, stages=[stage], budget=MemoryBudget(fast=256))
    _assign_tile_plans(graph)
    assert stage.tile_plan is not None and not stage.tile_plan.tileable
