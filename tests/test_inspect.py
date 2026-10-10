"""Inspection preserves source metadata and never executes model code."""

import json
import re
import struct
from pathlib import Path

import onnx
import pytest
from click.testing import CliRunner
from onnx import TensorProto, helper

from tigris.cli import cli
from tigris.emitters.binary import defs
from tigris.emitters.binary.reader import read_binary_plan
from tigris.inspection import inspect_file


FIXTURES = Path(__file__).parent / "schema_compat"


def test_onnx_original_graph(linear_3op_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Inspection must not infer shapes or load external weights")

    monkeypatch.setattr(onnx.shape_inference, "infer_shapes", forbidden)
    monkeypatch.setattr(onnx, "load_external_data_for_model", forbidden)
    report = inspect_file(linear_3op_path)
    assert report["format"] == "onnx"
    assert report["operator_counts"] == {"Add": 2, "Relu": 1}
    assert report["graph"]["value_info"] == []
    assert report["graph"]["inputs"][0]["shape"] == [1, 64]
    assert report["graph"]["initializers"][0]["size_bytes"] == 256
    assert report["graph"]["operators"][1]["inputs"] == ["t0", "w0"]


def test_symbolic_unknown_scalar_and_external_data(tmp_path):
    x = helper.make_tensor_value_info("[red]x[/red]", TensorProto.FLOAT, ["batch", None, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, None)
    scalar = helper.make_tensor_value_info("scalar", TensorProto.INT64, [])
    weight = TensorProto(name="weight", data_type=TensorProto.FLOAT, dims=[3])
    weight.data_location = TensorProto.EXTERNAL
    item = weight.external_data.add()
    item.key, item.value = "location", "missing-weights.bin"
    model = helper.make_model(helper.make_graph([], "[bold]graph[/bold]", [x], [y, scalar], [weight]))
    path = tmp_path / "not-an-onnx-extension.tgrs"
    path.write_bytes(model.SerializeToString())
    report = inspect_file(path)
    assert report["format"] == "onnx"
    assert report["graph"]["inputs"][0]["shape"] == ["batch", None, 3]
    assert [tensor["shape"] for tensor in report["graph"]["outputs"]] == [None, []]
    assert report["graph"]["initializers"][0]["storage"] == "external"
    result = CliRunner().invoke(cli, ["inspect", str(path)])
    assert result.exit_code == 0, result.output
    assert "[red]x[/red]" in result.output
    assert "[bold]graph[/bold]" in result.output
    assert "unknown shape" in result.output


@pytest.mark.parametrize("path", sorted(FIXTURES.glob("*.tgrs")), ids=lambda path: path.stem)
def test_supported_schemas_and_json(path):
    report = inspect_file(path)
    assert report["format"] == "tgrs"
    assert report["file_size_bytes"] == path.stat().st_size
    assert report["plan"]["ops"]
    result = CliRunner().invoke(cli, ["inspect", str(path), "--json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == report
    verbose = CliRunner().invoke(cli, ["inspect", str(path), "-v"])
    assert verbose.exit_code == 0, verbose.output
    assert "stored tensors" in verbose.output


def test_plan_detected_by_content(tmp_path):
    path = tmp_path / "plan.onnx"
    path.write_bytes((FIXTURES / "schema-v7-tensor-layout.tgrs").read_bytes())
    report = inspect_file(path)
    assert report["format"] == "tgrs"
    assert {tensor["layout"] for tensor in report["plan"]["tensors"]} == {"NLC", "model order"}


def test_interface_dtype_and_attributes():
    report = inspect_file(FIXTURES / "schema-v6-interface-dtype.tgrs")
    inputs = [report["plan"]["tensors"][index] for index in report["plan"]["model_inputs"]]
    assert any(tensor["dtype"] != tensor["interface_dtype"] for tensor in inputs)
    report = inspect_file(FIXTURES / "schema-v8-layer-norm.tgrs")
    assert any(attr["type"] == "epsilon" for op in report["plan"]["ops"] for attr in op["attributes"])


def test_compressed_metadata_without_decompression(linear_3op_path, tmp_path, monkeypatch):
    import lz4.block
    from tigris.cli import _run_pipeline
    from tigris.emitters.binary.writer import emit_binary_bytes

    graph, _ = _run_pipeline(str(linear_3op_path), ("256K",))
    data = emit_binary_bytes(graph, compress="lz4")
    assert read_binary_plan(data)["weight_blocks"][0]["decompressed"]

    def forbidden(*args, **kwargs):
        pytest.fail("Inspection must not decompress weight blocks")

    monkeypatch.setattr(lz4.block, "decompress", forbidden)
    path = tmp_path / "compressed.tgrs"
    path.write_bytes(data)
    report = inspect_file(path)
    assert report["plan"]["weight_blocks_compression"] == defs.COMPRESS_LZ4
    assert all("decompressed" not in block for block in report["plan"]["weight_blocks"])


@pytest.mark.parametrize("data, message", [
    (b"TGRS", "File too small"), (b"", "readable ONNX"),
    (b"not a model", "readable ONNX"),
])
def test_invalid_files_are_cli_errors(tmp_path, data, message):
    path = tmp_path / "model"
    path.write_bytes(data)
    result = CliRunner().invoke(cli, ["inspect", str(path), "--json"])
    assert result.exit_code == 1
    assert message in result.output
    assert "Traceback" not in result.output


def test_unsupported_schema(tmp_path):
    data = bytearray((FIXTURES / "schema-v7-tensor-layout.tgrs").read_bytes())
    struct.pack_into("<I", data, 4, 9999)
    path = tmp_path / "future.tgrs"
    path.write_bytes(data)
    with pytest.raises(ValueError, match="Unsupported schema version"):
        inspect_file(path)


def test_bad_tensor_reference(tmp_path):
    data = bytearray((FIXTURES / "schema-v2-linear.tgrs").read_bytes())
    header = defs.HEADER_STRUCT.unpack_from(data)
    offset = header[3]
    while True:
        kind, start = defs.SECTION_ENTRY_STRUCT.unpack_from(data, offset)
        if kind == defs.SEC_INDEX_POOL:
            break
        assert kind
        offset += defs.SECTION_ENTRY_SIZE
    struct.pack_into("<H", data, start + header[11] * 2, 65534)
    path = tmp_path / "bad.tgrs"
    path.write_bytes(data)
    with pytest.raises(ValueError, match="Invalid tensors reference"):
        inspect_file(path)


def test_tensor_attributes_do_not_include_weight_values(tmp_path):
    value = helper.make_tensor("constant", TensorProto.FLOAT, [2], [123.0, 456.0])
    node = helper.make_node("Constant", [], ["y"], value=value)
    output = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2])
    model = helper.make_model(helper.make_graph([node], "constant", [], [output]))
    path = tmp_path / "constant.onnx"
    path.write_bytes(model.SerializeToString())
    report = inspect_file(path)
    attr = report["graph"]["operators"][0]["attributes"][0]
    assert attr["value"]["size_bytes"] == 8
    assert "123.0" not in json.dumps(report)


def test_chain_mode_and_fused_activation(conv_relu_chain_path, tmp_path):
    from tigris.cli import _run_pipeline
    from tigris.emitters.binary.writer import emit_binary_bytes

    graph, _ = _run_pipeline(str(conv_relu_chain_path), ("16K",))
    path = tmp_path / "chain.tgrs"
    path.write_bytes(emit_binary_bytes(graph, xip=True))
    report = inspect_file(path)
    heads = [(index, stage) for index, stage in enumerate(graph.stages)
             if stage.chain_id == index and stage.chain_len >= 2]
    assert heads, "Fixture must exercise a tiled chain"
    for index, head in heads:
        stored = report["plan"]["stages"][index]
        assert bool(stored["flags"] & defs.STAGE_FLAG_LINE_BUFFERED) == head.line_buffered
        result = CliRunner().invoke(cli, ["inspect", str(path), "-v"])
        assert result.exit_code == 0, result.output
        assert ("line buffered" if head.line_buffered else "tile buffered") in result.output
    assert any(op["fused_activation"] == "Relu" for op in report["plan"]["ops"])
    assert "Activation clamp" not in result.output


def test_nonfinite_attributes_produce_standard_json(tmp_path):
    node = helper.make_node("Custom", ["x"], ["y"], domain="example", alpha=float("inf"))
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])
    model = helper.make_model(helper.make_graph([node], "custom", [x], [y]))
    path = tmp_path / "custom.onnx"
    path.write_bytes(model.SerializeToString())
    result = CliRunner().invoke(cli, ["inspect", str(path), "--json"])
    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report["operator_counts"] == {"example::Custom": 1}
    assert report["graph"]["operators"][0]["attributes"][0]["value"] == "inf"


@pytest.mark.parametrize("kind, field, value", [
    (defs.SEC_TENSORS, 2, 65535),
    (defs.SEC_WEIGHTS, 1, 0xFFFFFFFF),
])
def test_pool_and_weight_bounds(tmp_path, kind, field, value):
    data = bytearray((FIXTURES / "schema-v3-qdq-conv.tgrs").read_bytes())
    offset = defs.HEADER_STRUCT.unpack_from(data)[3]
    while True:
        section, start = defs.SECTION_ENTRY_STRUCT.unpack_from(data, offset)
        if section == kind:
            break
        assert section
        offset += defs.SECTION_ENTRY_SIZE
    record = defs.TENSOR_STRUCT if kind == defs.SEC_TENSORS else defs.WEIGHT_ENTRY_STRUCT
    values = list(record.unpack_from(data, start))
    values[field] = value
    record.pack_into(data, start, *values)
    path = tmp_path / "bad-offset.tgrs"
    path.write_bytes(data)
    with pytest.raises(ValueError, match="Data exceeds section"):
        inspect_file(path)


TFLITE = Path(__file__).parent / "fixtures" / "tflite"


def test_tflite_costs_quantization_and_activations():
    report = inspect_file(TFLITE / "ops" / "conv_bias.tflite")
    # 7x7x6 outputs, each a 3x3x4 dot product.
    assert report["costs"]["operators"] == [
        {"type": "CONV_2D", "count": 1, "weight_bytes": 240, "macs": 7 * 7 * 6 * 3 * 3 * 4}]
    assert report["quantization"] == {"activation_dtypes": ["int8"], "int8_weights": 1,
                                      "per_channel": 1, "asymmetric": 0, "float_islands": 0}
    report = inspect_file(TFLITE / "vww_96_int8.tflite")
    costs = report["costs"]
    assert costs["weight_bytes"] == sum(item["size_bytes"] for item in report["graph"]["initializers"])
    assert costs["macs"] == sum(kind["macs"] for kind in costs["operators"])
    largest = costs["largest_activations"][0]
    assert (largest["shape"], largest["bytes"], largest["step"]) == ([1, 48, 48, 16], 36864, 2)
    assert any(item["step"] is None for item in costs["largest_activations"])
    result = CliRunner().invoke(cli, ["inspect", str(TFLITE / "vww_96_int8.tflite")])
    assert result.exit_code == 0, result.output
    assert "largest activations" in result.output
    assert "int8 weights 27 of 28 per channel, symmetric" in result.output


def test_onnx_costs_mark_undeclared_shapes(conv_relu_chain_path):
    report = inspect_file(conv_relu_chain_path)
    assert report["quantization"] is None
    conv = next(kind for kind in report["costs"]["operators"] if kind["type"] == "Conv")
    # The intermediate shapes are not declared, so the multiply-accumulates are unknown.
    assert conv["macs"] is None and report["costs"]["macs"] is None
    assert conv["weight_bytes"] > 0
    result = CliRunner().invoke(cli, ["inspect", str(conv_relu_chain_path)])
    assert result.exit_code == 0, result.output
    assert re.search(r"Conv +2 +\S+ \S+ +\S+ +\?", result.output)


def test_plan_requirements_name_the_build_limits(tmp_path):
    report = inspect_file(FIXTURES / "schema-v10-state.tgrs")
    needs = report["requirements"]
    assert needs["fast_arena_bytes"] == report["plan"]["budget"]
    assert needs["state_bytes"] == 48
    limits = {item["name"]: item for item in needs["build_limits"]}
    assert limits["TIGRIS_MAX_TENSORS"]["needed"] == len(report["plan"]["tensors"])
    result = CliRunner().invoke(cli, ["inspect", str(FIXTURES / "schema-v10-state.tgrs")])
    assert "default limits suffice" in result.output
    assert "kept between runs" in result.output


def test_qdq_weights_belong_to_the_operator_that_uses_them(qdq_conv_path, tmp_path):
    # Declare the Conv output so its multiply-accumulates are computable.
    model = onnx.load(qdq_conv_path)
    conv = next(node for node in model.graph.node if node.op_type == "Conv")
    model.graph.value_info.append(helper.make_tensor_value_info(conv.output[0], TensorProto.FLOAT, [1, 2, 2, 2]))
    path = tmp_path / "qdq.onnx"
    onnx.save(model, path)
    report = inspect_file(path)
    kinds = {kind["type"]: kind for kind in report["costs"]["operators"]}
    # The weight reaches Conv through QuantizeLinear and DequantizeLinear, and
    # its bytes, scales and zero points count where it is used.
    assert kinds["Conv"]["weight_bytes"] == report["costs"]["weight_bytes"] - kinds["QuantizeLinear"]["weight_bytes"] \
        - kinds["DequantizeLinear"]["weight_bytes"]
    assert kinds["Conv"]["weight_bytes"] >= 2 * 1 * 3 * 3 * 4
    assert kinds["Conv"]["macs"] == 2 * 2 * 2 * 1 * 3 * 3
    names = {item["name"] for item in report["costs"]["activations"]}
    produced_from_constants = {node.output[0] for node in model.graph.node
                               if node.input[0] in {init.name for init in model.graph.initializer}}
    assert not names & produced_from_constants
