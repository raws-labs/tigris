"""Tests for C harness generation and XIP plan metadata."""

import struct

import numpy as np
import onnx
import pytest
from click.testing import CliRunner
from onnx import TensorProto, helper, numpy_helper

from tigris.analysis.lifetime import compute_lifetimes
from tigris.analysis.memory import compute_memory_timeline
from tigris.analysis.partition_spatial import partition_spatial
from tigris.analysis.partition_temporal import partition_temporal
from tigris.cli import cli
from tigris.emitters.binary.defs import (
    FLAG_XIP,
    HEADER_SIZE,
    SECTION_ENTRY_SIZE,
    SEC_TENSORS,
)
from tigris.emitters.binary.reader import read_binary_plan
from tigris.emitters.binary.writer import emit_binary_bytes
from tigris.emitters.codegen import generate_c
from tigris.loaders import load_model


@pytest.fixture
def quantized_matmul_plan(tmp_path):
    """A QDQ MatMul plan: schema-known, but unsupported by every dispatcher."""
    model_input = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, 4]
    )
    model_output = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, 3]
    )

    weight = numpy_helper.from_array(
        np.arange(12, dtype=np.float32).reshape(4, 3) / 16.0,
        name="weight",
    )
    input_scale = numpy_helper.from_array(
        np.array([0.05], dtype=np.float32), "input_scale"
    )
    input_zp = numpy_helper.from_array(
        np.array([0], dtype=np.int8), "input_zp"
    )
    weight_scale = numpy_helper.from_array(
        np.array([0.02], dtype=np.float32), "weight_scale"
    )
    weight_zp = numpy_helper.from_array(
        np.array([0], dtype=np.int8), "weight_zp"
    )
    output_scale = numpy_helper.from_array(
        np.array([0.1], dtype=np.float32), "output_scale"
    )
    output_zp = numpy_helper.from_array(
        np.array([0], dtype=np.int8), "output_zp"
    )

    nodes = [
        helper.make_node(
            "QuantizeLinear",
            ["input", "input_scale", "input_zp"],
            ["input_q"],
        ),
        helper.make_node(
            "DequantizeLinear",
            ["input_q", "input_scale", "input_zp"],
            ["input_dq"],
        ),
        helper.make_node(
            "QuantizeLinear",
            ["weight", "weight_scale", "weight_zp"],
            ["weight_q"],
        ),
        helper.make_node(
            "DequantizeLinear",
            ["weight_q", "weight_scale", "weight_zp"],
            ["weight_dq"],
        ),
        helper.make_node(
            "MatMul", ["input_dq", "weight_dq"], ["matmul_out"], name="matmul"
        ),
        helper.make_node(
            "QuantizeLinear",
            ["matmul_out", "output_scale", "output_zp"],
            ["output_q"],
        ),
        helper.make_node(
            "DequantizeLinear",
            ["output_q", "output_scale", "output_zp"],
            ["output"],
        ),
    ]
    graph = helper.make_graph(
        nodes,
        "quantized_matmul",
        [model_input],
        [model_output],
        initializer=[
            weight,
            input_scale,
            input_zp,
            weight_scale,
            weight_zp,
            output_scale,
            output_zp,
        ],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)]
    )
    model.ir_version = 8
    onnx.checker.check_model(model)

    model_path = tmp_path / "quantized_matmul.onnx"
    onnx.save(model, model_path)
    analyzed = _full_pipeline(model_path, budget=4096)
    assert analyzed.is_quantized
    assert [op.op_type for op in analyzed.ops] == ["MatMul"]

    plan_path = tmp_path / "quantized_matmul.tgrs"
    plan_path.write_bytes(emit_binary_bytes(analyzed))
    return plan_path


def _full_pipeline(path, budget=0):
    ag = load_model(path)
    ag = compute_lifetimes(ag)
    ag = compute_memory_timeline(ag)
    if budget > 0:
        ag = partition_temporal(ag, budget)
        ag = partition_spatial(ag)
    return ag


def test_cli_exposes_codegen_and_xip():
    runner = CliRunner()

    top = runner.invoke(cli, ["--help"])
    assert top.exit_code == 0
    assert "codegen" in top.output

    compile_help = runner.invoke(cli, ["compile", "--help"])
    assert compile_help.exit_code == 0
    assert "--xip" in compile_help.output


def test_xip_sets_plan_flag(linear_3op_path):
    ag = _full_pipeline(linear_3op_path, budget=4096)
    plan = read_binary_plan(emit_binary_bytes(ag, xip=True))

    assert plan["flags"] & FLAG_XIP


def test_codegen_reports_xip_and_loads_plan_at_runtime(linear_3op_path):
    ag = _full_pipeline(linear_3op_path, budget=4096)
    data = emit_binary_bytes(ag, xip=True)

    source = generate_c(data, "reference")

    assert "XIP:     yes" in source
    assert "static uint8_t *load_file" in source
    assert "tigris_plan_load(plan_buf, plan_len, &plan)" in source
    assert "tigris_mem_error_t merr = tigris_mem_init(" in source
    assert "merr = tigris_mem_alloc_slow(" in source
    assert source.count("if (merr != TIGRIS_MEM_OK)") >= 2
    assert "plan.header->peak > UINT32_MAX / 4u" in source
    assert "tigris_run(&plan, &mem, tigris_dispatch_kernel, NULL, &stats)" in source


def test_quantized_esp_codegen_has_valid_includes(qdq_conv_path):
    ag = _full_pipeline(qdq_conv_path, budget=4096)
    source = generate_c(emit_binary_bytes(ag), "esp-nn")

    assert "Kernels: esp-nn -> s8_ref fallback" in source
    assert "if (tigris_esp_nn_prepare(&plan, &mem) != 0)" in source
    assert "ESP-NN preparation failed" in source
    assert "tigris_mem_error_t merr = tigris_mem_init(" in source
    assert "merr = tigris_mem_alloc_slow(" in source
    assert "part->size < sizeof(tigris_file_header_t)" in source
    assert "(size_t)plan_size > part->size" in source
    assert '#include "tigris_kernels_esp_nn.h"' in source
    assert '#include "tigris_kernels_s8.h"' in source
    assert 'tigris_kernels_s8.h"' in source
    assert 'tigris_kernels_s8.h\\"' not in source


def test_quantized_reference_codegen_prints_int8_outputs(qdq_conv_path):
    ag = _full_pipeline(qdq_conv_path, budget=4096)

    source = generate_c(emit_binary_bytes(ag), "reference")

    assert "tigris_dispatch_kernel_s8" in source
    assert "int8_t *out = (int8_t *)ptr" in source
    assert "float *out = (float *)ptr" not in source


def test_codegen_rejects_plan_with_mixed_serialized_dtypes(qdq_conv_path):
    ag = _full_pipeline(qdq_conv_path, budget=4096)
    data = bytearray(emit_binary_bytes(ag))

    section_off = HEADER_SIZE
    tensor_section = None
    while section_off + SECTION_ENTRY_SIZE <= len(data):
        section_type, payload_off = struct.unpack_from("<II", data, section_off)
        if section_type == SEC_TENSORS:
            tensor_section = payload_off
            break
        if section_type == 0:
            break
        section_off += SECTION_ENTRY_SIZE
    assert tensor_section is not None
    data[tensor_section + 11] = TensorProto.FLOAT

    with pytest.raises(
        ValueError, match="cannot select one graph-wide runtime dispatcher"
    ):
        generate_c(bytes(data), "reference")


def test_codegen_rejects_plan_without_executable_stages(linear_3op_path):
    ag = _full_pipeline(linear_3op_path)

    with pytest.raises(ValueError, match="Plan has no executable stages"):
        generate_c(emit_binary_bytes(ag), "reference")


def test_cmsis_codegen_rejects_unrepresentable_static_slow_arena(
    linear_3op_path,
):
    ag = _full_pipeline(linear_3op_path, budget=4096)
    data = bytearray(emit_binary_bytes(ag))
    struct.pack_into("<I", data, 24, 0x80000000)

    with pytest.raises(ValueError, match="static slow arena exceeds"):
        generate_c(bytes(data), "cmsis-nn")


def test_quantized_cmsis_codegen_checks_prepare_and_memory(qdq_conv_path):
    ag = _full_pipeline(qdq_conv_path, budget=4096)

    source = generate_c(emit_binary_bytes(ag), "cmsis-nn")

    assert "if (tigris_cmsis_nn_prepare(&plan, &mem) != 0)" in source
    assert "CMSIS-NN preparation failed" in source
    assert "tigris_mem_error_t merr = tigris_mem_init(" in source
    assert "merr = tigris_mem_alloc_slow(" in source
    assert source.count("if (merr != TIGRIS_MEM_OK)") >= 2


def test_compressed_cmsis_arena_includes_static_weight_reserve(qdq_conv_path):
    ag = _full_pipeline(qdq_conv_path, budget=4096)
    data = emit_binary_bytes(ag, compress="lz4")
    plan = read_binary_plan(data)
    blocks = plan["weight_blocks"]
    assert blocks
    expected_overhead = max(
        (block["uncompressed_size"] + 15) // 16 * 16 for block in blocks
    )

    source = generate_c(data, "cmsis-nn")

    assert (
        f"static uint8_t fast_arena[{plan['budget'] + expected_overhead}]"
        in source
    )
    assert f"if (weight_overhead > {expected_overhead}u)" in source
    assert "fast_size += tigris_weight_decompression_overhead" not in source


@pytest.mark.parametrize("backend", ["esp-nn", "cmsis-nn"])
def test_float_accelerated_backend_declares_reference_fallback(
    linear_3op_path, backend
):
    ag = _full_pipeline(linear_3op_path, budget=4096)

    source = generate_c(emit_binary_bytes(ag), backend)

    assert (
        f"Kernels: reference (explicit float32 fallback; "
        f"{backend} acceleration is int8-only)"
    ) in source
    assert "tigris_dispatch_kernel, NULL, &stats" in source


@pytest.mark.parametrize("backend", ["reference", "esp-nn", "cmsis-nn"])
def test_quantized_matmul_codegen_fails_before_output(
    quantized_matmul_plan, tmp_path, backend
):
    output = tmp_path / f"matmul-{backend}.c"

    result = CliRunner().invoke(
        cli,
        [
            "codegen",
            str(quantized_matmul_plan),
            "--backend",
            backend,
            "--output",
            str(output),
        ],
    )

    assert result.exit_code != 0
    assert f"Backend '{backend}' cannot execute this int8 plan" in result.output
    assert "matmul (MatMul)" in result.output
    assert not output.exists()


def test_codegen_preserves_existing_output_on_capability_failure(
    quantized_matmul_plan, tmp_path
):
    output = tmp_path / "existing.c"
    output.write_text("sentinel")

    result = CliRunner().invoke(
        cli,
        [
            "codegen",
            str(quantized_matmul_plan),
            "--backend",
            "reference",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code != 0
    assert output.read_text() == "sentinel"
