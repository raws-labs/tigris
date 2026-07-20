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
from tigris.emitters.codegen import (
    _executor_workspace_limits,
    generate_c,
    generate_core_header,
)
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

    codegen_help = runner.invoke(cli, ["codegen", "--help"])
    assert codegen_help.exit_code == 0
    assert "--format" in codegen_help.output
    assert "--header" in codegen_help.output
    assert "--name" in codegen_help.output

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
    assert "TIGRIS_GENERATED_EXECUTOR_WORKSPACE_BYTES" in source
    assert (
        "uint8_t executor_workspace"
        "[TIGRIS_GENERATED_EXECUTOR_WORKSPACE_BYTES]" in source
    )
    assert "tigris_run_with_workspace_buffer(" in source
    assert "&plan, &mem, tigris_dispatch_kernel, NULL, &stats," in source
    assert "executor_workspace, sizeof(executor_workspace)" in source


def test_posix_codegen_aligns_plan_and_arenas(linear_3op_path):
    source = generate_c(
        emit_binary_bytes(_full_pipeline(linear_3op_path, budget=4096)),
        "reference",
    )

    assert "#define _POSIX_C_SOURCE 200112L" in source
    assert "posix_memalign(&ptr, alignment, size)" in source
    assert "uint8_t *buf = allocate_aligned" in source
    assert "void *fast_buf = allocate_aligned(fast_size)" in source
    assert "void *slow_buf = allocate_aligned(slow_size)" in source


def test_quantized_esp_codegen_has_valid_includes(qdq_conv_path):
    ag = _full_pipeline(qdq_conv_path, budget=4096)
    source = generate_c(emit_binary_bytes(ag), "esp-nn")

    assert "Kernels: esp-nn -> s8_ref fallback" in source
    assert "if (tigris_esp_nn_prepare(&plan, &mem) != 0)" in source
    assert "ESP-NN preparation failed" in source
    assert "tigris_fast_arena_required(&plan)" in source
    assert "tigris_weight_decompression_overhead(&plan)" not in source
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


@pytest.mark.parametrize(
    ("backend", "dispatch"),
    [
        ("reference", "tigris_dispatch_kernel_s8"),
        ("cmsis-nn", "tigris_dispatch_kernel_cmsis_nn"),
    ],
)
def test_core_codegen_is_embeddable_and_backend_specific(
    qdq_conv_path, backend, dispatch
):
    ag = _full_pipeline(qdq_conv_path, budget=4096)

    source = generate_c(
        emit_binary_bytes(ag), backend, output_format="core",
        core_header="generated_core.h",
    )

    assert '#include "generated_core.h"' in source
    assert "int main" not in source
    assert "malloc(" not in source
    assert "tigris_codegen_load_plan" in source
    assert "tigris_codegen_init" in source
    assert "tigris_codegen_reset" in source
    assert dispatch in source
    if backend == "cmsis-nn":
        assert "tigris_cmsis_nn_fast_arena_required(plan)" in source
        assert "fast_arena_size < cmsis_fast_required" in source
        assert "tigris_cmsis_nn_prepare" in source
        assert "tigris_cmsis_nn_deinit(mem)" in source
        assert source.index("tigris_cmsis_nn_deinit(mem)") < source.index(
            "tigris_mem_init("
        )


def test_core_codegen_header_exposes_embedding_api(qdq_conv_path):
    header = generate_core_header(
        emit_binary_bytes(_full_pipeline(qdq_conv_path, budget=4096))
    )

    assert "tigris_codegen_input_init_fn" in header
    assert "tigris_codegen_load_plan" in header
    assert "tigris_codegen_init" in header
    assert "tigris_codegen_reset" in header
    assert "tigris_codegen_run" in header
    assert "tigris_executor_workspace_t *workspace" in header
    assert "tigris_codegen_run_with_workspace_buffer" in header
    assert "TIGRIS_CODEGEN_TENSOR_CAPACITY" in header
    assert "TIGRIS_CODEGEN_PLAN_TENSOR_ALIGNMENT_BYTES 32u" in header
    assert "TIGRIS_CODEGEN_PLAN_BUDGET_BYTES" in header
    assert "TIGRIS_CODEGEN_WEIGHT_DECOMPRESSION_RESERVE_BYTES" in header
    assert "TIGRIS_CODEGEN_CORE_FAST_ARENA_BYTES" in header
    assert "TIGRIS_CODEGEN_EXECUTOR_WORKSPACE_BYTES" in header
    assert "TIGRIS_EXECUTOR_WORKSPACE_BYTES_FOR_LIMITS" in header


def test_codegen_workspace_limits_include_only_executable_chain_capacity():
    plan = {
        "num_tensors": 9,
        "ops": [
            {"op_type": 1},
            {"op_type": 2},
            {"op_type": 3},
            {"op_type": 1},
        ],
        "stages": [
            {
                "inputs": [0, 1],
                "outputs": [2],
                "ops": [0, 1],
                "chain_id": 0,
                "chain_len": 2,
            },
            {
                "inputs": [2],
                "outputs": [3, 4, 5],
                "ops": [2],
                "chain_id": 0,
                "chain_len": 2,
            },
            {
                "inputs": [5],
                "outputs": [6],
                "ops": [3],
                "chain_id": 0xFFFF,
                "chain_len": 0,
            },
        ],
    }

    assert _executor_workspace_limits(plan) == (9, 2, 3, 2, 2)


def test_compressed_core_header_uses_plan_alignment_for_arena_requirement(
    qdq_conv_path,
):
    data = emit_binary_bytes(
        _full_pipeline(qdq_conv_path, budget=4096), compress="lz4"
    )
    plan = read_binary_plan(data)
    expected_reserve = max(
        (block["uncompressed_size"] + 31) // 32 * 32
        for block in plan["weight_blocks"]
    )

    header = generate_core_header(data)

    assert (
        f"TIGRIS_CODEGEN_WEIGHT_DECOMPRESSION_RESERVE_BYTES "
        f"{expected_reserve}u" in header
    )
    assert (
        f"TIGRIS_CODEGEN_CORE_FAST_ARENA_BYTES "
        f"{plan['budget'] + expected_reserve}u" in header
    )


def test_core_codegen_custom_name_is_linkable_alongside_default(qdq_conv_path):
    data = emit_binary_bytes(_full_pipeline(qdq_conv_path, budget=4096))
    source = generate_c(
        data, "reference", output_format="core", core_header="audio_core.h",
        core_name="audio_codegen",
    )
    header = generate_core_header(data, "audio_codegen")

    assert "audio_codegen_load_plan" in source
    assert "audio_codegen_load_plan" in header
    assert "AUDIO_CODEGEN_TENSOR_CAPACITY" in header
    assert "#ifndef AUDIO_CODEGEN_CORE_H" in header
    assert "tigris_codegen_load_plan" not in source


def test_cli_core_writes_matched_source_and_header(linear_3op_path, tmp_path):
    plan_data = emit_binary_bytes(_full_pipeline(linear_3op_path, budget=4096))
    plan_path = tmp_path / "model.tgrs"
    source_path = tmp_path / "generated.c"
    header_path = tmp_path / "generated.h"
    plan_path.write_bytes(plan_data)

    result = CliRunner().invoke(
        cli,
        [
            "codegen", str(plan_path), "--format", "core",
            "--name", "sensor_codegen", "--output", str(source_path),
            "--header", str(header_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert source_path.is_file()
    assert header_path.is_file()
    assert '#include "generated.h"' in source_path.read_text()
    assert "sensor_codegen_load_plan" in source_path.read_text()
    assert "sensor_codegen_load_plan" in header_path.read_text()


@pytest.mark.parametrize("option", ["--header", "--name"])
def test_cli_rejects_core_only_options_for_app(linear_3op_path, option):
    args = ["codegen", str(linear_3op_path), option]
    if option == "--header":
        args.append("ignored.h")
    else:
        args.append("custom_codegen")

    result = CliRunner().invoke(cli, args)

    assert result.exit_code != 0
    assert "requires --format core" in result.output


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

    assert "tigris_cmsis_nn_fast_arena_required(&plan)" in source
    assert "Increase TIGRIS_CMSIS_NN_SCRATCH_BYTES" in source
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

    expected_core = (plan["budget"] + expected_overhead + 15) // 16 * 16
    assert "#define TIGRIS_CMSIS_NN_SCRATCH_BYTES 4096u" in source
    assert (
        f"static uint8_t fast_arena[{expected_core}u + "
        "TIGRIS_CMSIS_NN_SCRATCH_BYTES]" in source
    )
    assert f"if (weight_overhead > {expected_overhead}u)" in source
    assert "cmsis_fast_required > sizeof(fast_arena)" in source
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
