"""Tests for tigris.analysis.partition_spatial - op classification, receptive field, tile solver."""

import pytest
import json
import numpy as np
from onnx import TensorProto, helper, numpy_helper

from tigris.graph.ir import AnalyzedGraph, OpNode, Stage, TensorInfo
from tigris.analysis.partition_spatial import (
    TileCategory,
    classify_op,
    compute_receptive_field,
    _stage_tile_axis,
)
from tigris.analysis.validation import validate_memory_plan
from tigris.emitters.plan_json import plan_json_str
from tigris.fixtures import build_tcn

from conftest import full_pipeline


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


# Op classification


@pytest.mark.parametrize("op_type, category", [
    ("Conv", TileCategory.CONV),
    ("Relu", TileCategory.POINTWISE),
    ("MaxPool", TileCategory.POOL),
    ("AveragePool", TileCategory.POOL),
    # The output has more rows than the input, so the tile loop divides to find the source band.
    ("Resize", TileCategory.UPSAMPLE),
    ("ResizeLinear", TileCategory.UPSAMPLE),
    ("Flatten", TileCategory.UNTILEABLE),
    ("MyCustomOp", TileCategory.UNTILEABLE),
    ("Gemm", TileCategory.UNTILEABLE),
    # No schema v4 height contract.
    ("BatchNormalization", TileCategory.UNTILEABLE),
    ("GlobalAveragePool", TileCategory.UNTILEABLE),
])
def test_classify_op(op_type, category):
    assert classify_op(op_type) == category


# Receptive field computation


_CONV3 = ("Conv", {"kernel_shape": [3, 3], "strides": [1, 1]})


class TestReceptiveField:
    @pytest.mark.parametrize("layers, expected", [
        ([_CONV3], (3, 3)),
        ([_CONV3, _CONV3], (5, 5)),
        # Reversed: conv1 rf=3 j=1, pool(k=2,s=2) rf=4 j=2, conv0 rf=8 j=2.
        ([_CONV3, ("MaxPool", {"kernel_shape": [2, 2], "strides": [2, 2]}), _CONV3], (8, 8)),
        # Effective kernel 2*(3-1)+1 = 5.
        ([("Conv", {"kernel_shape": [3, 3], "strides": [1, 1], "dilations": [2, 2]})], (5, 5)),
        ([_CONV3, ("Relu", {})], (3, 3)),
        ([("Relu", {}), ("Add", {})], (1, 1)),
        ([("Conv", {"kernel_shape": [5, 3], "strides": [2, 1], "dilations": [1, 1]})], (5, 3)),
        # A stride multiplies the extent of every later kernel, so a stride-2 conv
        # before a 3x3 conv reaches 7 input rows and after it only 5.
        ([("Conv", {"kernel_shape": [3, 3], "strides": [2, 2]}), _CONV3], (7, 7)),
        ([_CONV3, ("Conv", {"kernel_shape": [3, 3], "strides": [2, 2]})], (5, 5)),
        # Height and width keep their own strides.
        ([("Conv", {"kernel_shape": [3, 3], "strides": [2, 1]}), _CONV3], (7, 5)),
    ], ids=["conv3x3", "two-conv3x3", "conv-pool-conv", "dilated", "pointwise-passthrough",
            "pointwise-only", "asymmetric", "stride-first", "stride-last", "stride-per-axis"])
    def test_fixed_geometry(self, layers, expected):
        ops = [OpNode(name=f"op{i}", op_type=op_type, inputs=[], outputs=[], attrs=dict(attrs))
               for i, (op_type, attrs) in enumerate(layers)]
        assert compute_receptive_field(ops) == expected

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
    def test_spatial_stages_get_exact_tile_plans(self, conv_relu_chain_path, conv_pool_chain_path):
        """Each stage holds one spatial op and tiles on that op's own receptive field."""
        for path, budget, expected in [
            (conv_relu_chain_path, 1024, [(2, 60, 2, 3), (2, 98, 2, 3)]),
            (conv_pool_chain_path, 512, [(1, 150, 2, 3), (1, 75, 1, 2), (1, 91, 2, 3)]),
        ]:
            ag = full_pipeline(path, budget=budget)
            plans = [s.tile_plan for s in ag.stages]
            assert all(tp is not None and tp.tileable for tp in plans)
            assert [(tp.tile_height, tp.num_tiles, tp.halo, tp.receptive_field)
                    for tp in plans] == expected

    def test_untileable_stage_flagged(self, conv_with_flatten_path):
        """Conv-Relu-Flatten should flag the stage containing Flatten as untileable."""
        ag = full_pipeline(conv_with_flatten_path, budget=256)

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
        ag = full_pipeline(conv_relu_chain_path, budget=10 * 1024 * 1024)

        # No stage should have a tile plan
        for s in ag.stages:
            assert s.tile_plan is None

    def test_conv1d_uses_serialized_length_axis(self, tmp_path):
        ag = full_pipeline(_write_conv1d(tmp_path / "conv1d.onnx"), budget=128)

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
        ag = full_pipeline(model, budget=256)

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

        ag = full_pipeline(model_path, budget=16 * 1024)
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
        ag = full_pipeline(conv_relu_chain_path, budget=1024)
        text = plan_json_str(ag)
        plan = json.loads(text)

        stages_with_tp = [s for s in plan["stages"] if "tile_plan" in s]
        assert len(stages_with_tp) == 2
        for s in stages_with_tp:
            tp = s["tile_plan"]
            assert "tileable" in tp
            assert "tile_height" in tp
            assert "halo" in tp


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


_BAND_KINDS = ["ReduceMean", "ReduceMax", "ReduceMin", "ReduceSum", "ReduceAll",
               "CumSum", "ArgMax", "ArgMin", "Gather", "GatherND", "EmbeddingLookup",
               "StridedSlice", "MirrorPad", "ReverseV2", "DynamicUpdateSlice"]


# The ReduceAll case has no quantized variant.
@pytest.mark.parametrize("quantized, kind", [
    (quantized, kind) for quantized in (False, True) for kind in _BAND_KINDS
    if not (quantized and kind == "ReduceAll")
])
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
    ((2, 11, 12), (2, 33, 4), (2, 11, 12, 33, 4, 1)),
    ((2, 33, 4), (2, 11, 12), (2, 33, 4, 11, 12, 1)),
    ((1, 3, 3, 8), (1, 3, 3, 2, 2, 2), (1, 3, 24, 3, 24, 1)),
    ((1, 6, 6, 4), (1, 36, 4), (1, 6, 24, 36, 4, 1)),
    ((1, 6, 6, 4), (36, 1, 4), (1, 6, 24, 36, 4, 1)),
    ((36, 1, 4), (1, 6, 6, 4), (1, 36, 4, 6, 24, 0)),
    ((1, 1, 6, 6, 4), (36, 1, 4), (1, 6, 24, 36, 4, 2)),
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

    mapping = (2, 33, 4, 11, 12, 1)
    plan = _reshape_tile_for_bytes(mapping, 256, 32, 4)
    assert plan.tileable and plan.tile_height == 6 and plan.num_tiles == 6
    assert plan.original_height == 33 and plan.band_axis == 1
    assert plan.tiled_peak_bytes == 192
    for start in range(0, 33, plan.tile_height):
        end = min(33, start + plan.tile_height)
        assert start * 4 % 12 == 0 and end * 4 % 12 == 0
    refused = _reshape_tile_for_bytes(mapping, 95, 32, 4)
    assert not refused.tileable and refused.min_tile_bytes == 96


@pytest.mark.parametrize("source,target,expected", [
    ((2, 3, 4), (3, 2, 4), (1, 2, 12, 3, 8, 0)),
    ((3, 2, 4), (2, 3, 4), (1, 3, 8, 2, 12, 0)),
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

    mapping = (1, 36, 4, 6, 24, 0)
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
    from tigris.analysis.partition_spatial import (
        _independent_band, _leading_band_axis, _leading_view, _solve_row_tile)
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
    tile = _solve_row_tile(graph, stage, [op], 256, _leading_view,
                           band_axis=_leading_band_axis(graph, op, 1))
    assert tile.tileable and tile.original_height == 37 and tile.num_tiles > 1
    assert tile.band_axis == rank - 3
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


@pytest.mark.parametrize("kind", ["Add", "Sub", "Mul", "Div", "FloorDiv", "FloorMod",
                                  "Max", "Min", "SquaredDifference"])
@pytest.mark.parametrize("constant,row_band", [
    ((64, 8), False), ((64, 1), False), ((8,), True), ((1, 8), True), ((), True)])
def test_row_band_takes_only_a_constant_every_row_reads_alike(kind, constant, row_band):
    from tigris.analysis.partition_spatial import _stage_is_row_tiled

    tensors = {n: TensorInfo(n, (64, 8), dtype=1) for n in ("x", "y")}
    tensors["k"] = TensorInfo("k", constant, dtype=1, is_constant=True)
    op = OpNode("op", kind, ["x", "k"], ["y"])
    graph = AnalyzedGraph(tensors=tensors, ops=[op],
                          weight_data={"k": np.ones(constant, dtype=np.float32)})
    assert _stage_is_row_tiled(graph, Stage(0, [0], ["x"], ["y"]), [op]) is row_band


def test_row_band_operators_are_the_contract_row_mode():
    from tigris.analysis.partition_spatial import _ROW_TILING_OPS
    from tigris.operator_contract import OPERATOR_CONTRACT

    assert _ROW_TILING_OPS == {name for name, row in OPERATOR_CONTRACT.items() if "row" in row.modes}


def test_a_strided_stripe_states_the_output_band_it_runs(strided_conv_path):
    """Three input rows and the halo give a stride-2 conv two output rows, so the plan states
    eight bands of two of the sixteen output rows, not eleven bands of the 32 input rows."""
    from tigris.cli import _run_pipeline
    from tigris.emitters.binary.reader import read_binary_plan
    from tigris.emitters.binary.writer import emit_binary_bytes

    ag, _ = _run_pipeline(str(strided_conv_path), ("1280",), report_bindings=False)
    (stage,) = ag.stages
    tp = stage.tile_plan
    assert (tp.tile_height, tp.num_tiles, tp.original_height, tp.halo) == (2, 8, 16, 2)
    # The band priced is still three input rows plus the halo; seven boundaries read the halo again.
    assert (tp.tiled_peak_bytes, tp.overhead_bytes) == (1280, 7 * 256)
    (record,) = read_binary_plan(emit_binary_bytes(ag))["tile_plans"]
    assert (record["input_band_extent"], record["output_band_extent"]) == (0, 2)
    assert (record["tile_extent"], record["num_tiles"], record["original_extent"],
            record["halo"], record["band_axis"]) == (2, 8, 16, 2, 1)
