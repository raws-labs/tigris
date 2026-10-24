"""Tests for tigris.loaders."""

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from tigris.loaders import load_model

from conftest import save_onnx


def test_linear_load(linear_3op_path):
    ag = load_model(linear_3op_path)

    assert ag.model_name == "linear_3op"
    # relu0 must come before add0, add0 before add1
    assert [op.op_type for op in ag.ops] == ["Relu", "Add", "Add"]
    assert ag.model_inputs == ["input"]
    assert ag.model_outputs == ["output"]

    info = ag.tensors["input"]
    assert info.shape == (1, 64)
    assert info.size_bytes == 1 * 64 * 4  # float32

    assert ag.tensors["w0"].is_constant is True
    assert ag.tensors["w1"].is_constant is True
    assert ag.tensors["input"].is_constant is False


def test_loader_extracts_spatial_attrs(conv_relu_chain_path, conv_pool_chain_path):
    conv_ops = [op for op in load_model(conv_relu_chain_path).ops if op.op_type == "Conv"]
    assert len(conv_ops) == 2
    for op in conv_ops:
        assert op.attrs["kernel_shape"] == [3, 3]
        assert op.attrs["strides"] == [1, 1]

    pool_ops = [op for op in load_model(conv_pool_chain_path).ops if op.op_type == "MaxPool"]
    assert len(pool_ops) == 1
    assert pool_ops[0].attrs["kernel_shape"] == [2, 2]
    assert pool_ops[0].attrs["strides"] == [2, 2]


def _save_model(tmp_path, name, nodes, inputs, outputs, initializers):
    return save_onnx(
        tmp_path, name, nodes, inputs, outputs, initializers, opset=17, ir_version=8
    )


def test_reshape_target_shape_is_not_a_weight(tmp_path):
    """A shape vector is metadata; binding it as a weight makes a plan the
    runtime refuses to load."""
    path = _save_model(
        tmp_path,
        "reshape_init",
        [
            helper.make_node("Relu", ["input"], ["act"]),
            helper.make_node("Reshape", ["act", "target_shape"], ["output"]),
        ],
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 2, 2, 2])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 8])],
        [numpy_helper.from_array(np.array([1, -1], dtype=np.int64), "target_shape")],
    )

    ag = load_model(path)

    reshape = next(op for op in ag.ops if op.op_type == "Reshape")
    # The operand is gone, whichever tensor the layout pass leaves feeding it:
    # regrouping a feature map needs it held in the model's own order first.
    assert reshape.inputs == [reshape.inputs[0]]
    assert "target_shape" not in reshape.inputs
    assert "target_shape" not in ag.weight_data


def test_clip_bounds_are_not_weights(tmp_path):
    path = _save_model(
        tmp_path,
        "clip_init",
        [helper.make_node("Clip", ["input", "lo", "hi"], ["output"])],
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])],
        [
            numpy_helper.from_array(np.array(0.0, dtype=np.float32), "lo"),
            numpy_helper.from_array(np.array(6.0, dtype=np.float32), "hi"),
        ],
    )

    ag = load_model(path)

    clip = ag.ops[0]
    assert clip.op_type == "Relu6"
    assert clip.inputs == ["input"]
    assert ag.weight_data == {}


def test_runtime_computed_metadata_operand_is_kept(tmp_path):
    """Only constants are metadata the compiler can drop."""
    path = _save_model(
        tmp_path,
        "reshape_dynamic",
        [
            helper.make_node("Relu", ["input"], ["act"]),
            helper.make_node("Reshape", ["act", "target_shape"], ["output"]),
        ],
        [
            helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 2, 2, 2]),
            helper.make_tensor_value_info("target_shape", TensorProto.INT64, [2]),
        ],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 8])],
        [],
    )

    ag = load_model(path)

    reshape = next(op for op in ag.ops if op.op_type == "Reshape")
    # A shape the graph computes at runtime stays an operand, whichever
    # tensor the layout pass leaves as the data one.
    assert len(reshape.inputs) == 2
    assert reshape.inputs[1] == "target_shape"


def test_orphan_value_info_is_not_a_tensor(tmp_path):
    """A graph optimizer leaves value_info behind for tensors its fusion removed.

    Those entries describe nothing the graph computes. Carrying them into the
    IR makes a fully quantized model look mixed-dtype and be rejected.
    """
    path = _save_model(
        tmp_path,
        "orphan_value_info",
        [helper.make_node("Relu", ["input"], ["output"])],
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])],
        [],
    )
    model = onnx.load(str(path))
    model.graph.value_info.append(
        helper.make_tensor_value_info("fused_away", TensorProto.FLOAT, [1, 4])
    )
    onnx.save(model, path)

    ag = load_model(path)

    assert "fused_away" not in ag.tensors
    assert set(ag.tensors) == {"input", "output"}


def _row(name):
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, [1, 4])


def _cycle(file_order):
    """relu(x) -> a; add and sigmoid feed each other; tanh reads the cycle."""
    nodes = {
        "relu": helper.make_node("Relu", ["x"], ["a"], name="relu"),
        "add": helper.make_node("Add", ["a", "c"], ["b"], name="add"),
        "sigmoid": helper.make_node("Sigmoid", ["b"], ["c"], name="sigmoid"),
        "tanh": helper.make_node("Tanh", ["b"], ["y"], name="tanh"),
    }
    return [nodes[name] for name in file_order], []


def _self_loop():
    return [helper.make_node("Add", ["x", "b"], ["b"], name="add"),
            helper.make_node("Tanh", ["b"], ["y"], name="tanh")], []


def _cycle_through_shape():
    """Folding shape(b) to the stated shape would cut the cycle and leave a valid order."""
    return [helper.make_node("Relu", ["x"], ["a"], name="relu"),
            helper.make_node("Reshape", ["a", "s"], ["d"], name="reshape"),
            helper.make_node("Add", ["a", "d"], ["b"], name="add"),
            helper.make_node("Shape", ["b"], ["s"], name="shape"),
            helper.make_node("Tanh", ["b"], ["y"], name="tanh")], [_row("b"), _row("d")]


def _cycle_through_a_branch():
    """Only the then branch reads d, which depends on the If's own output."""
    def branch(name, node):
        return helper.make_graph([node], name, [], [_row(node.output[0])])

    return [helper.make_node("If", ["cond"], ["f"], name="if",
                             then_branch=branch("then", helper.make_node("Relu", ["d"], ["then_y"])),
                             else_branch=branch("else", helper.make_node("Tanh", ["x"], ["else_y"]))),
            helper.make_node("Sigmoid", ["f"], ["d"], name="sigmoid"),
            helper.make_node("Tanh", ["d"], ["y"], name="tanh")], []


@pytest.mark.parametrize("graph, named", [
    (_cycle(["relu", "add", "sigmoid", "tanh"]), "add"),
    (_cycle(["tanh", "relu", "add", "sigmoid"]), "tanh"),
    (_self_loop(), "add"),
    (_cycle_through_shape(), "reshape"),
    (_cycle_through_a_branch(), "if"),
], ids=["on_cycle", "reads_cycle", "self_loop", "through_shape", "through_branch"])
def test_a_graph_with_a_cycle_is_refused(tmp_path, graph, named):
    """The first node in file order that cannot be placed is named, whether it
    is on the cycle or only reads from it."""
    nodes, value_info = graph
    path = _save_model(tmp_path, "cycle", nodes,
                       [_row("x"), helper.make_tensor_value_info("cond", TensorProto.BOOL, [1])],
                       [_row("y")], [])
    model = onnx.load(str(path))
    model.graph.value_info.extend(value_info)
    onnx.save(model, path)

    with pytest.raises(ValueError) as refusal:
        load_model(path)

    assert str(refusal.value) == (
        f"operator {named} is on a cycle or depends on one, so the graph has no execution order")


def test_a_value_read_inside_a_loop_body_is_produced_before_the_loop(tmp_path):
    """The read sits in a Loop nested in the body, so only the operator's
    subgraphs say it depends on the producer."""
    def body(name, node, initializers=()):
        return helper.make_graph(
            [helper.make_node("Identity", [f"{name}_go"], [f"{name}_go_out"]), node], name,
            [helper.make_tensor_value_info(f"{name}_i", TensorProto.INT64, []),
             helper.make_tensor_value_info(f"{name}_go", TensorProto.BOOL, []), _row(f"{name}_v")],
            [helper.make_tensor_value_info(f"{name}_go_out", TensorProto.BOOL, []),
             _row(node.output[0])], list(initializers))

    inner = helper.make_node("Loop", ["inner_n", "", "outer_v"], ["outer_y"],
                             body=body("inner", helper.make_node("Add", ["inner_v", "act"], ["inner_y"])))
    nodes = [
        helper.make_node("Sigmoid", ["x"], ["act"], name="producer"),
        helper.make_node("Loop", ["n", "", "x"], ["y"], name="loop", body=body(
            "outer", inner, [numpy_helper.from_array(np.asarray(2, np.int64), "inner_n")])),
    ]
    path = _save_model(tmp_path, "outer_read", nodes, [_row("x")], [_row("y")],
                       [numpy_helper.from_array(np.asarray(3, np.int64), "n")])

    assert [op.name for op in load_model(path).ops] == ["producer", "loop"]


def _subgraph(name, nodes, inputs=(), initializers=()):
    return helper.make_graph(nodes, name, [_row(n) for n in inputs], [_row(nodes[-1].output[0])],
                             list(initializers))


def _scalar(name, value):
    return numpy_helper.from_array(np.asarray(value, np.float32), name)


def _if_reading(then_input, else_input, operands=()):
    """An If whose branches apply Relu and Tanh to the named values; the then
    branch holds its own constant then_c."""
    tigris = {"domain": "tigris"} if operands else {}
    return helper.make_node(
        "If", ["cond", *operands], ["y"], name="if", **tigris,
        then_branch=_subgraph("then", [helper.make_node("Relu", [then_input], ["then_y"])],
                              [f"then_{o}" for o in operands], [_scalar("then_c", 1.0)]),
        else_branch=_subgraph("else", [helper.make_node("Tanh", [else_input], ["else_y"])],
                              [f"else_{o}" for o in operands]))


def _nested_if():
    """The inner If takes the outer then branch's input, but its then branch
    reads then_t, a value of the outer then branch."""
    inner = helper.make_node(
        "If", ["then_cond", "then_x"], ["then_y"], name="inner", domain="tigris",
        then_branch=_subgraph("inner_then", [helper.make_node("Relu", ["then_t"], ["inner_then_y"])],
                              ["inner_then_x"]),
        else_branch=_subgraph("inner_else", [helper.make_node("Tanh", ["inner_else_x"], ["inner_else_y"])],
                              ["inner_else_x"]))
    then = helper.make_graph(
        [helper.make_node("Sigmoid", ["then_x"], ["then_t"]), inner], "then", [_row("then_x")],
        [_row("then_y")], [numpy_helper.from_array(np.asarray([True]), "then_cond")])
    return [helper.make_node("If", ["cond", "x"], ["y"], name="if", domain="tigris", then_branch=then,
                             else_branch=_subgraph("else", [helper.make_node("Tanh", ["else_x"], ["else_y"])],
                                                   ["else_x"]))]


def _while_reading_outside():
    """Three steps of v * w, where w = sigmoid(x) is a value of the main graph."""
    def count(name):
        return helper.make_tensor_value_info(name, TensorProto.INT32, [])

    condition = helper.make_graph(
        [helper.make_node("Less", ["c_i", "c_three"], ["c_go"])], "condition", [count("c_i"), _row("c_v")],
        [helper.make_tensor_value_info("c_go", TensorProto.BOOL, [])],
        [numpy_helper.from_array(np.asarray(3, np.int32), "c_three")])
    body = helper.make_graph(
        [helper.make_node("Add", ["b_i", "b_one"], ["b_next"]),
         helper.make_node("Mul", ["b_v", "w"], ["b_y"])], "body", [count("b_i"), _row("b_v")],
        [count("b_next"), _row("b_y")], [numpy_helper.from_array(np.asarray(1, np.int32), "b_one")])
    return [helper.make_node("Sigmoid", ["x"], ["w"]),
            helper.make_node("While", ["zero", "x"], ["count", "y"], name="while", domain="tigris",
                             cond_branch=condition, body_branch=body)]


def _save_outer_read(tmp_path, nodes):
    graph = helper.make_graph(
        nodes, "outer_read", [_row("x"), helper.make_tensor_value_info("cond", TensorProto.BOOL, [1])],
        [_row("y")], [_scalar("k", 2.0), numpy_helper.from_array(np.asarray(0, np.int32), "zero")])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17),
                                                    helper.make_opsetid("tigris", 1)])
    model.ir_version = 8
    path = tmp_path / "outer_read.onnx"
    onnx.save(model, path)
    return path


@pytest.mark.parametrize("nodes, name, operands, then_inputs, else_inputs, copied", [
    ([helper.make_node("Relu", ["x"], ["a"]), _if_reading("a", "a")],
     "if", ["cond", "a"], ["a"], ["a"], []),
    ([_if_reading("then_c", "x")], "if", ["cond", "x"], ["x"], ["x"], []),
    ([helper.make_node("Sigmoid", ["x"], ["b"]), _if_reading("then_x", "b", ["x"])],
     "if", ["cond", "x", "b"], ["then_x", "b"], ["else_x", "b"], []),
    ([_if_reading("k", "else_x", ["x"])], "if", ["cond", "x"], ["then_x"], ["else_x"], ["k"]),
    (_nested_if(), "inner", ["then_cond", "then_x", "then_t"], ["inner_then_x", "then_t"],
     ["inner_else_x", "then_t"], []),
], ids=["then_branch", "else_branch", "beside_operands", "outer_constant", "nested"])
def test_an_if_takes_what_its_branches_read_from_enclosing_graphs(
        tmp_path, nodes, name, operands, then_inputs, else_inputs, copied):
    """A value read becomes an operand after the existing ones and an input of
    both branches; a constant read is copied into the branch."""
    def find(graph):
        for op in graph.ops:
            if op.name == name:
                return graph, op
            for key in ("then_branch", "else_branch"):
                if key in op.attrs and (found := find(graph.subgraphs[op.attrs[key]])):
                    return found
        return None

    graph, op = find(load_model(_save_outer_read(tmp_path, nodes)))
    then, other = (graph.subgraphs[op.attrs[key]] for key in ("then_branch", "else_branch"))

    assert op.op_type == "If"
    assert op.inputs == operands
    assert (then.model_inputs, other.model_inputs) == (then_inputs, else_inputs)
    assert [n for n in copied if n in then.weight_data] == copied


def test_a_while_reading_an_enclosing_value_is_refused(tmp_path):
    """The loop variables are the only values a plan passes into a While."""
    with pytest.raises(ValueError) as refused:
        load_model(_save_outer_read(tmp_path, _while_reading_outside()))

    assert str(refused.value) == (
        "operator while reads w from an enclosing graph in its body_branch; a While condition "
        "and body can read only its loop variables and the values they define")


@pytest.mark.parametrize("node", [_if_reading("nowhere", "x"), _if_reading("nowhere", "else_x", ["x"])],
                         ids=["standard", "with_operands"])
def test_a_branch_reading_a_name_no_graph_defines_is_refused(tmp_path, node):
    with pytest.raises(ValueError) as refused:
        load_model(_save_outer_read(tmp_path, [node]))

    assert str(refused.value) == "operator if reads nowhere, which no graph defines"


def test_a_standard_if_whose_branches_give_different_shapes_is_refused(tmp_path):
    """ONNX's If is checked as the compiler's own: each branch gives its results."""
    nodes = [helper.make_node(
        "If", ["cond"], ["y"], name="if",
        then_branch=_subgraph("then", [helper.make_node("Relu", ["x"], ["then_y"])]),
        else_branch=helper.make_graph(
            [helper.make_node("Concat", ["x", "x"], ["else_y"], axis=1)], "else", [],
            [helper.make_tensor_value_info("else_y", TensorProto.FLOAT, [1, 8])]))]

    with pytest.raises(ValueError) as refused:
        load_model(_save_outer_read(tmp_path, nodes))

    assert str(refused.value) == "If branches must take its operands and give its results"
