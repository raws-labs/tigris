"""Operators an exporter leaves behind that compute nothing at inference."""

import numpy as np
import onnx
import pytest
from click.testing import CliRunner
from onnx import TensorProto, helper, numpy_helper

from tigris.cli import cli
from tigris.emitters.binary.reader import read_binary_plan
from tigris.emitters.binary.writer import emit_binary_bytes
from tigris.loaders import load_model
from tigris.loaders.onnx.normalize.layout import _stored_extent

from conftest import normalized, save_onnx, vi


def test_dropout_and_identity_are_removed(tmp_path):
    ag = normalized(tmp_path, "identities",
                    [helper.make_node("Identity", ["x"], ["a"], name="id1"),
                     helper.make_node("Dropout", ["a"], ["b"], ratio=0.5, name="drop1"),
                     helper.make_node("Relu", ["b"], ["y"], name="relu1")], [vi("x", [1, 8])],
                    [vi("y", [1, 8])], opset=11)

    types = [op.op_type for op in ag.ops]
    assert types == ["Relu"], types
    # The survivor must read the original graph input, not a dropped tensor.
    assert ag.ops[0].inputs[0] == "x"


def test_removed_identity_keeps_the_declared_output_name(tmp_path):
    """An Identity feeding the graph output goes, but the output keeps its name."""
    ag = normalized(tmp_path, "trailing_identity",
                    [helper.make_node("Relu", ["x"], ["activation"], name="relu1"),
                     helper.make_node("Identity", ["activation"], ["y"], name="id1")],
                    [vi("x", [1, 8])], [vi("y", [1, 8])])

    assert [op.op_type for op in ag.ops] == ["Relu"]
    assert ag.model_outputs == ["y"]
    assert ag.ops[0].outputs == ["y"]
    assert "activation" not in ag.tensors


def test_output_that_passes_an_input_through_gets_its_own_tensor(tmp_path):
    """One tensor cannot carry two names, so the output is lowered to a same-shape copy of the input."""
    ag = normalized(tmp_path, "passthrough",
                    [helper.make_node("Identity", ["x"], ["y"], name="id1"),
                     helper.make_node("Relu", ["x"], ["z"], name="relu1")], [vi("x", [1, 8])],
                    [vi("y", [1, 8]), vi("z", [1, 8])])

    assert ag.model_outputs == ["y", "z"]
    assert [(op.name, op.op_type, op.inputs) for op in ag.ops if "y" in op.outputs] == [("y_copy", "Reshape", ["x"])]


def test_dropout_keeping_its_mask_is_left_alone(tmp_path):
    """The mask output only exists for training; an op whose mask is read stays."""
    ag = normalized(tmp_path, "dropout_mask",
                    [helper.make_node("Dropout", ["x"], ["b", "mask"], name="drop1"),
                     helper.make_node("Relu", ["b"], ["y"], name="relu1")], [vi("x", [1, 8])],
                    [vi("y", [1, 8]),
                     helper.make_tensor_value_info("mask", TensorProto.BOOL, [1, 8])], opset=12)

    assert "Dropout" in [op.op_type for op in ag.ops]


def test_squeeze_becomes_reshape_when_element_order_survives(tmp_path):
    """Dropping the trailing unit axes of a pooled tensor is a pure reshape."""
    ag = normalized(tmp_path, "squeeze_safe",
                    [helper.make_node("Squeeze", ["x", "axes"], ["y"], name="sq1")],
                    [vi("x", [1, 6, 1, 1])], [vi("y", [1, 6])],
                    [numpy_helper.from_array(np.array([2, 3], np.int64), "axes")])

    assert [op.op_type for op in ag.ops] == ["Reshape"]
    # The axes operand is gone: the plan takes the shape from the tensor table.
    assert len(ag.ops[0].inputs) == 1


def test_unsqueeze_becomes_reshape_when_element_order_survives(tmp_path):
    ag = normalized(tmp_path, "unsqueeze_safe",
                    [helper.make_node("Unsqueeze", ["x", "axes"], ["y"], name="un1")],
                    [vi("x", [1, 6])], [vi("y", [1, 6, 1, 1])],
                    [numpy_helper.from_array(np.array([2, 3], np.int64), "axes")])

    assert [op.op_type for op in ag.ops] == ["Reshape"]


def test_a_squeeze_that_would_transpose_is_converted_first(tmp_path):
    """Rank 4 to rank 3 dropping the channel axis reorders under NHWC/NLC.

    NHWC [1, H, W, 1] walks (h, w); the resulting NLC [1, W, H] walks (w, h).
    Copying the bytes through would transpose the picture, so the operand is
    converted to the model's own order first and the regrouping then moves
    nothing. The conversions are explicit Transposes around the reshape, and
    the last one puts the model output back in the layout callers expect.
    """
    ag = normalized(tmp_path, "squeeze_unsafe",
                    [helper.make_node("Squeeze", ["x", "axes"], ["y"], name="sq1")],
                    [vi("x", [1, 1, 8, 4])], [vi("y", [1, 8, 4])],
                    [numpy_helper.from_array(np.array([1], np.int64), "axes")])

    types = [op.op_type for op in ag.ops]
    assert types.count("Reshape") == 1
    assert types[0] == "Transpose" and types[-1] == "Transpose"


def test_stored_extent_orders_axes_the_way_the_runtime_serializes_them():
    # rank 4 is NCHW -> NHWC, rank 3 is NCL -> NLC, everything else is as-is.
    assert _stored_extent((1, 1, 8, 4)) == (8, 4)     # N,H,W,C over [1,1,8,4]
    assert _stored_extent((1, 8, 4)) == (4, 8)        # N,L,C over [1,8,4]
    assert _stored_extent((1, 6, 1, 1)) == (6,)
    assert _stored_extent((1, 6)) == (6,)


@pytest.mark.parametrize("index, parts", [
    (0, [(1, 8), (3, 8)]),
    (2, [(2, 8), (1, 8), (1, 8)]),
], ids=["first", "middle"])
def test_a_gather_of_one_index_becomes_a_cut(tmp_path, index, parts):
    """Reading one index out of an axis nothing precedes is a run of bytes.

    The runtime carries no Gather, and it does not need one: the wanted index
    starts a contiguous run, which is what a Split cuts.
    """
    proj = np.eye(8, dtype=np.float32)
    ag = normalized(tmp_path, "gather_one",
                    [helper.make_node("MatMul", ["x", "proj"], ["mixed"], name="mix"),
                     helper.make_node("Gather", ["mixed", "index"], ["y"], axis=1, name="pick")],
                    [vi("x", [1, 4, 8])], [vi("y", [1, 8])],
                    init=[numpy_helper.from_array(proj, "proj"),
                          numpy_helper.from_array(np.array(index, dtype=np.int64), "index")])
    types = [op.op_type for op in ag.ops]
    assert "Gather" not in types
    assert "Split" in types
    cut = ag.ops[types.index("Split")]
    assert [ag.tensors[name].shape for name in cut.outputs] == parts


def test_a_gather_on_an_axis_something_precedes_is_left_alone(tmp_path):
    """The wanted index is then scattered through the stored bytes."""
    proj = np.eye(8, dtype=np.float32)
    ag = normalized(tmp_path, "gather_inner",
                    [helper.make_node("MatMul", ["x", "proj"], ["mixed"], name="mix"),
                     helper.make_node("Gather", ["mixed", "first"], ["y"], axis=2, name="pick")],
                    [vi("x", [1, 4, 8])], [vi("y", [1, 4])],
                    init=[numpy_helper.from_array(proj, "proj"),
                          numpy_helper.from_array(np.array(0, dtype=np.int64), "first")])
    assert "Gather" in [op.op_type for op in ag.ops]


def _boundary_model(tmp_path):
    """uint8 image -> DequantizeLinear -> Conv -> (QuantizeLinear to uint8)."""
    weight = np.full((2, 3, 1, 1), 0.5, dtype=np.float32)
    inits = [
        numpy_helper.from_array(weight, "w"),
        numpy_helper.from_array(np.array(1.0 / 255.0, dtype=np.float32), "img_s"),
        numpy_helper.from_array(np.array(0, dtype=np.uint8), "img_z"),
        numpy_helper.from_array(np.array(0.01, dtype=np.float32), "w_s"),
        numpy_helper.from_array(np.array(0, dtype=np.int8), "w_z"),
        numpy_helper.from_array(np.array(0.02, dtype=np.float32), "out_s"),
        numpy_helper.from_array(np.array(100, dtype=np.uint8), "out_z"),
    ]
    nodes = [
        helper.make_node("DequantizeLinear", ["image", "img_s", "img_z"], ["x"]),
        helper.make_node("QuantizeLinear", ["w", "w_s", "w_z"], ["wq"]),
        helper.make_node("DequantizeLinear", ["wq", "w_s", "w_z"], ["wdq"]),
        helper.make_node("Conv", ["x", "wdq"], ["raw"], kernel_shape=[1, 1]),
        helper.make_node("QuantizeLinear", ["raw", "out_s", "out_z"], ["mask"]),
    ]
    return save_onnx(
        tmp_path, "boundary", nodes,
        [helper.make_tensor_value_info("image", TensorProto.UINT8, [1, 3, 4, 4])],
        [helper.make_tensor_value_info("mask", TensorProto.UINT8, [1, 2, 4, 4])],
        inits, ir_version=8)


def test_quantized_boundaries_fold_onto_int8_tensors(tmp_path):
    """A uint8 input and output keep their places and become int8 tensors.

    The zero points move by 128 with the values, and the dtype the model
    declares stays recorded so the plan converts at the boundary instead of
    carrying a float copy of the image.
    """
    ag = load_model(_boundary_model(tmp_path))
    assert not [op for op in ag.ops if op.op_type in ("QuantizeLinear", "DequantizeLinear")]
    image = ag.tensors[ag.model_inputs[0]]
    mask = ag.tensors[ag.model_outputs[0]]
    assert ag.model_inputs == ["image"]
    assert image.dtype == 3 and int(image.quant.zero_point[0]) == -128
    assert mask.dtype == 3 and int(mask.quant.zero_point[0]) == 100 - 128
    assert ag.model_input_dtypes == [2] and ag.model_output_dtypes == [2]


@pytest.mark.parametrize("attrs, op_type", [
    # F.avg_pool2d(x, x.shape[2:]) with PyTorch's default count_include_pad
    ({"kernel_shape": [5, 7], "strides": [5, 7], "count_include_pad": 1},
     "GlobalAveragePool"),
    ({"kernel_shape": [5, 5], "strides": [5, 5]}, "AveragePool"),
], ids=["whole_map", "partial_window"])
def test_only_a_whole_map_average_pool_becomes_the_global_pool(tmp_path, attrs, op_type):
    ag = normalized(tmp_path, "avg_pool",
                    [helper.make_node("AveragePool", ["input"], ["output"], **attrs)],
                    [vi("input", [1, 3, 5, 7])], [vi("output", [1, 3, 1, 1])], ir_version=8)
    assert [op.op_type for op in ag.ops] == [op_type]



_CONV_WEIGHT = numpy_helper.from_array(
    np.random.default_rng(0).standard_normal((8, 8, 3, 3)).astype(np.float32), "w")


def _onnx(nodes, init=()):
    return lambda tmp_path, name: save_onnx(tmp_path, name, nodes, [vi("x", [1, 8, 64, 64])],
                                            [vi("y", [1, 8, 64, 64])], init)


def _tflite_logistic_nothing_reads(tmp_path, name):
    from tigris.frontends import tflite_writer as tw
    from tigris.frontends.tflite_schema import BUILTIN_OPERATORS

    codes = [BUILTIN_OPERATORS.index(kind) for kind in ("RELU", "LOGISTIC")]
    graph = tw.Subgraph([tw.Tensor([1, 64, 64, 8], 0, 0, tensor) for tensor in (b"x", b"y", b"unread")],
                        [0], [1], [tw.Operator(0, [0], [1]), tw.Operator(1, [0], [2])])
    path = tmp_path / f"{name}.tflite"
    path.write_bytes(tw.write(tw.Model(3, [tw.OperatorCode(code, 1, None, code) for code in codes],
                                       [graph], [None])))
    return path


# Writers of models whose output is Relu(x), x of 128 KiB, that compute more
# than that, with the operators normalization keeps.
DEAD_OPERATORS = {
    "sigmoid_nothing_reads": (_onnx(
        [helper.make_node("Relu", ["x"], ["y"], name="relu"),
         helper.make_node("Sigmoid", ["x"], ["dead"], name="sigmoid")]), ["Relu"]),
    "conv_then_tanh_nothing_reads": (_onnx(
        [helper.make_node("Relu", ["x"], ["y"], name="relu"),
         helper.make_node("Conv", ["x", "w"], ["c"], pads=[1, 1, 1, 1], name="conv"),
         helper.make_node("Tanh", ["c"], ["dead"], name="tanh")], [_CONV_WEIGHT]), ["Relu"]),
    # Folding Shape to a constant takes the Sigmoid's only reader.
    "sigmoid_only_a_shape_reads": (_onnx(
        [helper.make_node("Sigmoid", ["x"], ["s"], name="sigmoid"),
         helper.make_node("Shape", ["s"], ["shape"], name="shape"),
         helper.make_node("Relu", ["x"], ["r"], name="relu"),
         helper.make_node("Reshape", ["r", "shape"], ["y"], name="reshape")]), ["Relu", "Reshape"]),
    "tflite_logistic_nothing_reads": (_tflite_logistic_nothing_reads, ["Relu"]),
}


@pytest.mark.parametrize("case", sorted(DEAD_OPERATORS))
def test_an_operator_whose_result_nothing_reads_is_removed(tmp_path, case):
    """Every operator left computes part of a model output, and the activations
    and weights only removed operators used are gone with them."""
    write, kept = DEAD_OPERATORS[case]
    ag = load_model(write(tmp_path, case))

    assert [op.op_type for op in ag.ops] == kept
    named = {name for op in ag.ops for name in op.inputs + op.outputs}
    assert {name for name, info in ag.tensors.items() if not info.is_constant} == named
    assert not ag.weight_data


@pytest.mark.usefixtures("require_host_library")
@pytest.mark.parametrize("case", sorted(DEAD_OPERATORS))
def test_tigris_compile_writes_a_model_with_an_operator_nothing_reads_at_a_tight_budget(tmp_path, case):
    """At 16K every stage is tiled, and the runtime loader accepts the plan."""
    source = DEAD_OPERATORS[case][0](tmp_path, case)
    plan = tmp_path / f"{case}.tgrs"
    result = CliRunner().invoke(cli, ["compile", str(source), "-m", "16K", "-o", str(plan)])
    assert result.exit_code == 0, result.output
    assert read_binary_plan(plan.read_bytes())["tile_plans"]


@pytest.mark.usefixtures("require_host_library")
@pytest.mark.parametrize("case", sorted(DEAD_OPERATORS))
def test_the_library_compile_of_a_model_with_an_operator_nothing_reads_runs_at_a_tight_budget(
        tmp_path, case):
    """The plan planned for 16K is tiled and computes Relu(x) exactly."""
    from tigris.pipeline import plan_model
    from tigris.runtime import Session

    source = DEAD_OPERATORS[case][0](tmp_path, case)
    plan = tmp_path / f"{case}.tgrs"
    plan.write_bytes(emit_binary_bytes(plan_model(str(source), ("16K",))))
    assert read_binary_plan(plan.read_bytes())["tile_plans"]
    with Session(plan) as session:
        (item,) = session.inputs
        image = np.random.default_rng(0).standard_normal(item["shape"]).astype(np.float32)
        (output,) = session.run({item["name"]: image}).values()
    np.testing.assert_array_equal(output, np.maximum(image, 0))


def test_a_reader_nothing_reads_leaves_a_fold_as_it_is(tmp_path):
    """Relu folds into the Conv whether or not a Sigmoid nobody reads also reads the Conv."""
    def graph(name, extra):
        return normalized(tmp_path, name,
                          [helper.make_node("Conv", ["x", "w"], ["c"], pads=[1, 1, 1, 1], name="conv"),
                           helper.make_node("Relu", ["c"], ["y"], name="relu")] + extra,
                          [vi("x", [1, 8, 16, 16])], [vi("y", [1, 8, 16, 16])], [_CONV_WEIGHT])

    plain = graph("plain", [])
    dead = graph("dead_reader", [helper.make_node("Sigmoid", ["c"], ["dead"], name="sigmoid")])
    assert [(op.op_type, op.attrs.get("fused_activation")) for op in dead.ops] == [("Conv", "Relu")]
    assert [(op.op_type, op.inputs, op.outputs, op.attrs) for op in dead.ops] == [
        (op.op_type, op.inputs, op.outputs, op.attrs) for op in plain.ops]


def test_an_if_nothing_reads_is_removed_with_its_branches(tmp_path):
    """The If that remains names its own branches after the removed If's are gone."""
    def branch(name, op_type):
        return helper.make_graph([helper.make_node(op_type, ["x"], [f"{name}_y"])], name, [],
                                 [vi(f"{name}_y", [1, 8])])

    nodes = [helper.make_node("If", ["cond"], ["unread"], name="dead_if",
                              then_branch=branch("dead_then", "Relu"),
                              else_branch=branch("dead_else", "Tanh")),
             helper.make_node("If", ["cond"], ["y"], name="live_if",
                              then_branch=branch("live_then", "Abs"),
                              else_branch=branch("live_else", "Neg"))]
    ag = normalized(tmp_path, "dead_if", nodes,
                    [vi("x", [1, 8]), helper.make_tensor_value_info("cond", TensorProto.BOOL, [1])],
                    [vi("y", [1, 8])])

    (op,) = ag.ops
    assert op.name == "live_if" and "unread" not in ag.tensors
    assert len(ag.subgraphs) == 2
    assert [[sub.op_type for sub in ag.subgraphs[op.attrs[key]].ops]
            for key in ("then_branch", "else_branch")] == [["Abs"], ["Neg"]]


def _row(name):
    return vi(name, [1, 8, 64, 64])


def _count(name):
    return helper.make_tensor_value_info(name, TensorProto.INT32, [])


def _save_with_tigris_ops(tmp_path, name, nodes, inputs, init=(), value_info=()):
    graph = helper.make_graph(nodes, name, inputs, [_row("y")], list(init), value_info=list(value_info))
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17),
                                                    helper.make_opsetid("tigris", 1)])
    model.ir_version = 8
    path = tmp_path / f"{name}.onnx"
    onnx.save(model, path)
    return path


def _if_branch_with_an_operator_nothing_reads(tmp_path, name):
    then = helper.make_graph([helper.make_node("Relu", ["x"], ["then_y"]),
                              helper.make_node("Sigmoid", ["x"], ["then_dead"])], "then", [], [_row("then_y")])
    other = helper.make_graph([helper.make_node("Abs", ["x"], ["else_y"])], "else", [], [_row("else_y")])
    return _save_with_tigris_ops(
        tmp_path, name, [helper.make_node("If", ["cond"], ["y"], then_branch=then, else_branch=other)],
        [_row("x"), helper.make_tensor_value_info("cond", TensorProto.BOOL, [1])])


def _while_body_with_an_operator_nothing_reads(tmp_path, name):
    condition = helper.make_graph(
        [helper.make_node("Less", ["c_i", "c_three"], ["c_go"])], "condition", [_count("c_i"), _row("c_v")],
        [helper.make_tensor_value_info("c_go", TensorProto.BOOL, [])],
        [numpy_helper.from_array(np.asarray(3, np.int32), "c_three")])
    body = helper.make_graph(
        [helper.make_node("Add", ["b_i", "b_one"], ["b_next"]), helper.make_node("Relu", ["b_v"], ["b_y"]),
         helper.make_node("Sigmoid", ["b_v"], ["b_dead"])], "body", [_count("b_i"), _row("b_v")],
        [_count("b_next"), _row("b_y")], [numpy_helper.from_array(np.asarray(1, np.int32), "b_one")])
    return _save_with_tigris_ops(
        tmp_path, name, [helper.make_node("While", ["zero", "x"], ["count", "y"], domain="tigris",
                                          cond_branch=condition, body_branch=body)],
        [_row("x")], [numpy_helper.from_array(np.asarray(0, np.int32), "zero")], [_count("count")])


# Writers of models whose output is Relu(x), x of 128 KiB, through an If with
# cond true or a While of three steps, with an operator nothing reads in the
# subgraph that computes it; the subgraph's operators normalization keeps.
DEAD_IN_SUBGRAPHS = {
    "if_branch": (_if_branch_with_an_operator_nothing_reads, "then_branch", ["Relu"]),
    "while_body": (_while_body_with_an_operator_nothing_reads, "body_branch", ["Relu", "Add"]),
}


@pytest.mark.parametrize("case", sorted(DEAD_IN_SUBGRAPHS))
def test_an_operator_nothing_reads_in_a_subgraph_is_removed(tmp_path, case):
    write, key, kept = DEAD_IN_SUBGRAPHS[case]
    ag = load_model(write(tmp_path, case))

    (op,) = ag.ops
    assert [sub.op_type for sub in ag.subgraphs[op.attrs[key]].ops] == kept


@pytest.mark.usefixtures("require_host_library")
@pytest.mark.parametrize("case", sorted(DEAD_IN_SUBGRAPHS))
def test_a_subgraph_with_an_operator_nothing_reads_compiles_and_runs_at_a_tight_budget(tmp_path, case):
    """tigris compile and the library compile write a tiled plan at 16K that computes Relu(x) exactly."""
    from tigris.pipeline import plan_model
    from tigris.runtime import Session

    source = DEAD_IN_SUBGRAPHS[case][0](tmp_path, case)
    cli_plan, library_plan = tmp_path / "cli.tgrs", tmp_path / "library.tgrs"
    result = CliRunner().invoke(cli, ["compile", str(source), "-m", "16K", "-o", str(cli_plan)])
    assert result.exit_code == 0, result.output
    library_plan.write_bytes(emit_binary_bytes(plan_model(str(source), ("16K",))))
    image = np.random.default_rng(0).standard_normal((1, 8, 64, 64)).astype(np.float32)
    for plan in (cli_plan, library_plan):
        assert read_binary_plan(plan.read_bytes())["tile_plans"]
        with Session(plan) as session:
            feeds = {item["name"]: np.ones(item["shape"], bool) if item["name"] == "cond"
                     else image.reshape(item["shape"]) for item in session.inputs}
            (output,) = session.run(feeds).values()
        np.testing.assert_array_equal(output.reshape(image.shape), np.maximum(image, 0))
