"""If and While cases."""

from __future__ import annotations

import numpy as np
from onnx import TensorProto, helper, numpy_helper

from .common import ContractCase, _model, _qdq, _scalars


def _if_case(take_then: bool, standard: bool = False) -> ContractCase:
    """An If on the sign of a sum, each branch a different computation of x;
    the input is chosen so the given branch runs. ``standard`` compiles ONNX's
    If, whose branches read x from the main graph."""
    x = np.asarray([[0.5, -1.25, 2.0, 0.75]], np.float32) * (1.0 if take_then else -1.0)
    weights = [numpy_helper.from_array(np.asarray([2], np.int64), "axes"),
               numpy_helper.from_array(np.asarray([1, 1, 4], np.int64), "rows"),
               numpy_helper.from_array(np.asarray(0.0, np.float32), "zero")]

    def branch(name, nodes, constants, explicit):
        inputs = [helper.make_tensor_value_info(f"{name}_x", TensorProto.FLOAT, [1, 4])] if explicit else []
        return helper.make_graph(nodes(f"{name}_x" if explicit else "x", name), name, inputs,
                                 [helper.make_tensor_value_info(f"{name}_y", TensorProto.FLOAT, [1, 4])],
                                 constants(name))

    def then_nodes(x, name):
        return [helper.make_node("Mul", [x, f"{name}_two"], [f"{name}_twice"]),
                helper.make_node("Add", [f"{name}_twice", f"{name}_one"], [f"{name}_y"])]

    def then_constants(name):
        return [numpy_helper.from_array(np.asarray(2.0, np.float32), f"{name}_two"),
                numpy_helper.from_array(np.asarray(1.0, np.float32), f"{name}_one")]

    def else_nodes(x, name):
        return [helper.make_node("Relu", [x], [f"{name}_relu"]),
                helper.make_node("Sub", [f"{name}_relu", f"{name}_three"], [f"{name}_y"])]

    def else_constants(name):
        return [numpy_helper.from_array(np.asarray(3.0, np.float32), f"{name}_three")]

    # The runtime reduces a rank-3 tensor along one axis.
    condition = [helper.make_node("Reshape", ["x", "rows"], ["row"]),
                 helper.make_node("ReduceSum", ["row", "axes"], ["sum"], keepdims=0),
                 helper.make_node("Greater", ["sum", "zero"], ["positive"])]
    compile_graph = helper.make_graph(
        [*condition, helper.make_node(
            "If", ["positive", "x"], ["y"], domain="tigris",
            then_branch=branch("then", then_nodes, then_constants, True),
            else_branch=branch("else", else_nodes, else_constants, True))],
        "if", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])], weights,
        value_info=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])])
    compile_model = helper.make_model(compile_graph, opset_imports=[
        helper.make_opsetid("", 17), helper.make_opsetid("tigris", 1)])
    compile_model.ir_version = 9
    reference = _model(
        "if_reference",
        [*condition, helper.make_node(
            "If", ["positive"], ["y"],
            then_branch=branch("then", then_nodes, then_constants, False),
            else_branch=branch("else", else_nodes, else_constants, False))],
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])], weights, opset=17)
    return ContractCase(name=f"if_{'standard_' if standard else ''}{'then' if take_then else 'else'}",
                        compile_model=reference if standard else compile_model,
                        reference_model=reference, inputs={"x": x},
                        # The main graph's stages, then each branch's, the else
                        # branch's Sub folded into an Add.
                        expected_operators=("Reshape", "Transpose", "ReduceSum", "Greater", "If",
                                            "Relu", "Add", "Mul", "Add"))


def _if_tiled_case(take_then: bool) -> ContractCase:
    """An If whose branches are convolutions over a feature map too large for
    the budget, so the branch stages run in tiles."""
    rng = np.random.default_rng(87)
    x = rng.uniform(-1.0, 1.0, (1, 4, 8, 8)).astype(np.float32)
    x = np.abs(x) if take_then else -np.abs(x)
    kernels = {part: rng.normal(0.0, 0.3, (4, 4, 3, 3)).astype(np.float32)
               for part in ("then_a", "then_b", "else_a")}
    shape = [1, 4, 8, 8]
    weights = [numpy_helper.from_array(np.asarray([1, 1, 256], np.int64), "rows"),
               numpy_helper.from_array(np.asarray([2], np.int64), "axes"),
               numpy_helper.from_array(np.asarray(0.0, np.float32), "zero")]

    def conv(source, kernel, result):
        return helper.make_node("Conv", [source, kernel], [result], pads=[1, 1, 1, 1])

    def branch(name, explicit):
        x_name = f"{name}_x" if explicit else "x"
        if name == "then":
            nodes = [conv(x_name, "then_a", "then_h"), helper.make_node("Relu", ["then_h"], ["then_r"]),
                     conv("then_r", "then_b", "then_y")]
            constants = ["then_a", "then_b"]
        else:
            nodes = [conv(x_name, "else_a", "else_y")]
            constants = ["else_a"]
        inputs = [helper.make_tensor_value_info(x_name, TensorProto.FLOAT, shape)] if explicit else []
        return helper.make_graph(nodes, name, inputs,
                                 [helper.make_tensor_value_info(f"{name}_y", TensorProto.FLOAT, shape)],
                                 [numpy_helper.from_array(kernels[c], c) for c in constants])

    condition = [helper.make_node("Reshape", ["x", "rows"], ["row"]),
                 helper.make_node("ReduceSum", ["row", "axes"], ["sum"], keepdims=0),
                 helper.make_node("Greater", ["sum", "zero"], ["positive"])]
    compile_graph = helper.make_graph(
        [*condition, helper.make_node("If", ["positive", "x"], ["y"], domain="tigris",
                                      then_branch=branch("then", True), else_branch=branch("else", True))],
        "if_tiled", [helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, shape)], weights,
        value_info=[helper.make_tensor_value_info("y", TensorProto.FLOAT, shape)])
    compile_model = helper.make_model(compile_graph, opset_imports=[
        helper.make_opsetid("", 17), helper.make_opsetid("tigris", 1)])
    compile_model.ir_version = 9
    reference = _model(
        "if_tiled_reference",
        [*condition, helper.make_node("If", ["positive"], ["y"], then_branch=branch("then", False),
                                      else_branch=branch("else", False))],
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, shape)], weights, opset=17)
    return ContractCase(name="if_tiled_then" if take_then else "if_tiled_else",
                        compile_model=compile_model, reference_model=reference, inputs={"x": x},
                        expected_operators=("Reshape", "Transpose", "ReduceSum", "Greater", "If",
                                            "Conv", "Conv", "Conv"),
                        # The then branch's two convolutions run as one tiled,
                        # line-buffered chain; the plan holds both branches.
                        mem_budget="1536", expect_tiled=True, expect_chain=True,
                        expect_line_buffered=True)


def _nested_if_case(standard: bool = False) -> ContractCase:
    """An If whose then branch holds another If; inputs reach each innermost
    branch once across the runs the reference checks. ``standard`` compiles
    ONNX's If, whose branches read x from the main graph."""
    x = np.asarray([[0.5, 2.5, -0.25, 1.0]], np.float32)
    shape = [1, 4]
    weights = [numpy_helper.from_array(np.asarray([1, 1, 4], np.int64), "rows"),
               numpy_helper.from_array(np.asarray([2], np.int64), "axes"),
               numpy_helper.from_array(np.asarray(0.0, np.float32), "zero"),
               numpy_helper.from_array(np.asarray(2.0, np.float32), "two")]

    def info(name):
        return helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)

    def test(value, prefix, op, limit, result, rows="rows"):
        return [helper.make_node("Reshape", [value, rows], [f"{prefix}row"]),
                helper.make_node(op, [f"{prefix}row", "axes"] if op == "ReduceSum" else [f"{prefix}row"],
                                 [f"{prefix}reduced"], keepdims=0,
                                 **({} if op == "ReduceSum" else {"axes": [2]})),
                helper.make_node("Greater", [f"{prefix}reduced", limit], [result])]

    def leaf(name, op, constant, explicit):
        x_name = f"{name}_x" if explicit else "outer_x"
        return helper.make_graph(
            [helper.make_node(op, [x_name, f"{name}_c"], [f"{name}_y"])], name,
            [info(x_name)] if explicit else [], [info(f"{name}_y")],
            [numpy_helper.from_array(np.asarray(constant, np.float32), f"{name}_c")])

    def outer_then(explicit):
        x_name = "then_x" if explicit else "outer_x"
        nodes = [helper.make_node("Identity", ["x" if not explicit else x_name], ["outer_x"])] if not explicit else []
        inner_in = x_name
        # A branch holds its own constants.
        nodes += test(inner_in, "inner_", "ReduceMax", "inner_two", "large", rows="inner_rows")
        nodes.append(helper.make_node(
            "If", ["large", inner_in] if explicit else ["large"], ["then_y"],
            **({"domain": "tigris"} if explicit else {}),
            then_branch=leaf("half", "Mul", 0.5, explicit), else_branch=leaf("more", "Add", 1.0, explicit)))
        return helper.make_graph(
            nodes, "then", [info(x_name)] if explicit else [], [info("then_y")],
            [numpy_helper.from_array(np.asarray([1, 1, 4], np.int64), "inner_rows"),
             numpy_helper.from_array(np.asarray(2.0, np.float32), "inner_two")])

    def outer_else(explicit):
        return leaf("less", "Sub", 3.0, explicit) if explicit else helper.make_graph(
            [helper.make_node("Sub", ["x", "less_c"], ["less_y"])], "less", [], [info("less_y")],
            [numpy_helper.from_array(np.asarray(3.0, np.float32), "less_c")])

    condition = test("x", "outer_", "ReduceSum", "zero", "positive")
    compile_graph = helper.make_graph(
        [*condition, helper.make_node("If", ["positive", "x"], ["y"], domain="tigris",
                                      then_branch=outer_then(True), else_branch=outer_else(True))],
        "nested_if", [info("x")], [info("y")], weights, value_info=[info("y")])
    compile_model = helper.make_model(compile_graph, opset_imports=[
        helper.make_opsetid("", 17), helper.make_opsetid("tigris", 1)])
    compile_model.ir_version = 9
    reference = _model(
        "nested_if_reference",
        [*condition, helper.make_node("If", ["positive"], ["y"], then_branch=outer_then(False),
                                      else_branch=outer_else(False))],
        [info("x")], [info("y")], weights, opset=17)
    return ContractCase(name="nested_if_standard" if standard else "nested_if",
                        compile_model=reference if standard else compile_model, reference_model=reference,
                        inputs={"x": x},
                        expected_operators=("Reshape", "Transpose", "ReduceSum", "Greater", "If",
                                            "Reshape", "Transpose", "ReduceMax", "Greater", "If",
                                            # Sub by a constant folds into Add.
                                            "Mul", "Add", "Add"))


def _int8_if_case(take_then: bool, standard: bool = False) -> ContractCase:
    """An int8 If whose operand and result cross the boundary in float32, as
    the TFLite converter writes it: Dequantize before, Quantize after, each
    branch requantizing its input in its own encoding. ``standard`` compiles
    the QDQ form with ONNX's If, whose branches read xf from the main graph."""
    rng = np.random.default_rng(90)
    data = rng.integers(-128, 128, (1, 4, 4, 2)).astype(np.int8)
    shape = [1, 2, 4, 4]

    def info(name, kind=TensorProto.FLOAT, dims=shape):
        return helper.make_tensor_value_info(name, kind, dims)

    def branch(name, scale, zero, relu, tigris):
        entering = (helper.make_node("Quantize", [f"{name}_x"], [f"{name}_q"], domain="tigris") if tigris
                    else helper.make_node("Identity", [f"{name}_x"], [f"{name}_q"]))
        nodes = [entering, *_qdq(f"{name}_q", f"{name}_s", f"{name}_z", f"{name}_v")]
        last = f"{name}_v"
        if relu:
            nodes += [helper.make_node("Relu", [last], [f"{name}_r"]),
                      *_qdq(f"{name}_r", f"{name}_s", f"{name}_z", f"{name}_rq")]
            last = f"{name}_rq"
        nodes.append(helper.make_node("Dequantize", [last], [f"{name}_y"], domain="tigris") if tigris
                     else helper.make_node("Identity", [last], [f"{name}_y"]))
        return helper.make_graph(nodes, name, [info(f"{name}_x")], [info(f"{name}_y")],
                                 _scalars(**{name: (scale, zero)}),
                                 value_info=[info(f"{name}_q"), info(f"{name}_y")] if tigris else [])

    def model(tigris):
        domain = {"domain": "tigris"} if tigris else {}
        nodes = [helper.make_node("DequantizeLinear", ["image", "in_s", "in_z"], ["x"]),
                 helper.make_node("Dequantize" if tigris else "Identity", ["x"], ["xf"], **domain),
                 helper.make_node("If", ["take", "xf"] if tigris else ["take"], ["yf"],
                                  then_branch=branch("then", 0.03, -5, True, tigris),
                                  else_branch=branch("else", 0.05, 3, False, tigris), **domain),
                 helper.make_node("Quantize" if tigris else "Identity", ["yf"], ["yq"], **domain),
                 helper.make_node("QuantizeLinear", ["yq", "out_s", "out_z"], ["out"])]
        if not tigris:
            # ONNX's If reads the outer value; the branch input is renamed onto it.
            for graph in (nodes[2].attribute[0].g, nodes[2].attribute[1].g):
                prefix = graph.name
                del graph.input[:]
                graph.node.insert(0, helper.make_node("Identity", ["xf"], [f"{prefix}_x"]))
        graph = helper.make_graph(
            nodes, "int8_if", [helper.make_tensor_value_info("image", TensorProto.INT8, shape),
                               helper.make_tensor_value_info("take", TensorProto.BOOL, [1])],
            [helper.make_tensor_value_info("out", TensorProto.INT8, shape)],
            _scalars(**{"in": (0.04, -2), "out": (0.035, -10)}),
            value_info=[info("xf"), info("yf"), info("yq")] if tigris else [])
        built = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)] + (
            [helper.make_opsetid("tigris", 1)] if tigris else []))
        built.ir_version = 9
        return built

    return ContractCase(name=f"int8_if_{'standard_' if standard else ''}{'then' if take_then else 'else'}",
                        compile_model=model(not standard),
                        reference_model=model(False),
                        inputs={"image": data.transpose(0, 3, 1, 2).copy(),
                                "take": np.asarray([take_then])},
                        expected_operators=("Dequantize", "If", "Quantize", "Quantize", "Relu",
                                            "Dequantize", "Quantize", "Dequantize"))


def _while_counter_case() -> ContractCase:
    """A While counting i from a constant 0 to 5 in int32, adding i as a float
    to v * 1.5 each step, against the loop unrolled."""
    x = np.asarray([[0.5, -1.25, 2.0, 0.75]], np.float32)

    def info(name, kind=TensorProto.FLOAT, dims=(1, 4)):
        return helper.make_tensor_value_info(name, kind, list(dims))

    condition = helper.make_graph(
        [helper.make_node("Less", ["c_i", "c_five"], ["c_out"])], "condition",
        [info("c_i", TensorProto.INT32, ()), info("c_v")], [info("c_out", TensorProto.BOOL, ())],
        [numpy_helper.from_array(np.asarray(5, np.int32), "c_five")])
    body = helper.make_graph(
        [helper.make_node("Add", ["b_i", "b_one"], ["b_next"]),
         helper.make_node("Cast", ["b_i"], ["b_f"], to=TensorProto.FLOAT),
         # The runtime broadcasts operands of the output's rank.
         helper.make_node("Reshape", ["b_f", "b_one_by_one"], ["b_f2"]),
         helper.make_node("Mul", ["b_v", "b_rate"], ["b_scaled"]),
         helper.make_node("Add", ["b_scaled", "b_f2"], ["b_out"])], "body",
        [info("b_i", TensorProto.INT32, ()), info("b_v")],
        [info("b_next", TensorProto.INT32, ()), info("b_out")],
        [numpy_helper.from_array(np.asarray(1, np.int32), "b_one"),
         numpy_helper.from_array(np.asarray([1, 1], np.int64), "b_one_by_one"),
         numpy_helper.from_array(np.asarray(1.5, np.float32), "b_rate")])
    compile_graph = helper.make_graph(
        [helper.make_node("While", ["zero", "x"], ["count", "y"], domain="tigris",
                          cond_branch=condition, body_branch=body)],
        "while_counter", [info("x")], [info("y")],
        [numpy_helper.from_array(np.asarray(0, np.int32), "zero")],
        value_info=[info("count", TensorProto.INT32, ()), info("y")])
    compile_model = helper.make_model(compile_graph, opset_imports=[
        helper.make_opsetid("", 17), helper.make_opsetid("tigris", 1)])
    compile_model.ir_version = 9
    nodes, value = [], "x"
    for step in range(5):
        nodes += [helper.make_node("Mul", [value, "rate"], [f"scaled{step}"]),
                  helper.make_node("Add", [f"scaled{step}", f"i{step}"],
                                   ["y" if step == 4 else f"v{step}"])]
        value = f"v{step}"
    reference = _model(
        "while_counter_reference", nodes, [info("x")], [info("y")],
        [numpy_helper.from_array(np.asarray(1.5, np.float32), "rate"),
         *(numpy_helper.from_array(np.asarray(float(i), np.float32), f"i{i}") for i in range(5))])
    return ContractCase(name="while_counter", compile_model=compile_model, reference_model=reference,
                        inputs={"x": x},
                        expected_operators=("While", "Less", "Mul", "Cast", "Reshape", "Add", "Add"))


def _while_case() -> ContractCase:
    """A While growing x until its sum reaches 40, against ONNX's Loop, which
    tests the first condition outside its body and each later one inside."""
    x = np.asarray([[0.5, 1.25, 2.0, 0.75]], np.float32)
    shape = [1, 4]

    def info(name, kind=TensorProto.FLOAT, dims=None):
        return helper.make_tensor_value_info(name, kind, shape if dims is None else dims)

    def constants(prefix):
        return [numpy_helper.from_array(np.asarray([1, 1, 4], np.int64), f"{prefix}rows"),
                numpy_helper.from_array(np.asarray([2], np.int64), f"{prefix}axes"),
                numpy_helper.from_array(np.asarray(40.0, np.float32), f"{prefix}limit"),
                numpy_helper.from_array(np.asarray(1.5, np.float32), f"{prefix}rate"),
                numpy_helper.from_array(np.asarray(0.25, np.float32), f"{prefix}step")]

    def below(value, prefix, result):
        # The runtime reduces a rank-3 tensor along one axis.
        return [helper.make_node("Reshape", [value, f"{prefix}rows"], [f"{prefix}row"]),
                helper.make_node("ReduceSum", [f"{prefix}row", f"{prefix}axes"], [f"{prefix}sum"],
                                 keepdims=0),
                helper.make_node("Less", [f"{prefix}sum", f"{prefix}limit"], [result])]

    def grow(value, prefix, result):
        return [helper.make_node("Mul", [value, f"{prefix}rate"], [f"{prefix}scaled"]),
                helper.make_node("Add", [f"{prefix}scaled", f"{prefix}step"], [result])]

    condition = helper.make_graph(below("c_in", "c_", "c_out"), "condition", [info("c_in")],
                                  [info("c_out", TensorProto.BOOL, [1, 1])], constants("c_"))
    body = helper.make_graph(grow("b_in", "b_", "b_out"), "body", [info("b_in")], [info("b_out")],
                             constants("b_"))
    compile_graph = helper.make_graph(
        [helper.make_node("While", ["x"], ["y"], domain="tigris", cond_branch=condition,
                          body_branch=body)],
        "while", [info("x")], [info("y")], value_info=[info("y")])
    compile_model = helper.make_model(compile_graph, opset_imports=[
        helper.make_opsetid("", 17), helper.make_opsetid("tigris", 1)])
    compile_model.ir_version = 9
    loop_body = helper.make_graph(
        [*grow("v", "l_", "next"), *below("next", "l_", "again_2d"),
         helper.make_node("Squeeze", ["again_2d"], ["again"])],
        "loop_body",
        [info("iteration", TensorProto.INT64, []), info("keep", TensorProto.BOOL, []), info("v")],
        [info("again", TensorProto.BOOL, []), info("next")], constants("l_"))
    reference = _model(
        "while_reference",
        [*below("x", "r_", "first_2d"), helper.make_node("Squeeze", ["first_2d"], ["first"]),
         helper.make_node("Loop", ["", "first", "x"], ["y"], body=loop_body)],
        [info("x")], [info("y")], constants("r_"), opset=17)
    return ContractCase(name="while", compile_model=compile_model, reference_model=reference,
                        inputs={"x": x},
                        # The main graph's While, then the condition's and body's stages.
                        expected_operators=("While", "Reshape", "Transpose", "ReduceSum", "Less",
                                            "Mul", "Add"))
