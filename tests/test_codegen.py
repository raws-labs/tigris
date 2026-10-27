"""Tests for C harness generation and XIP plan metadata."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from click.testing import CliRunner
from onnx import TensorProto, helper, numpy_helper

from tigris.analysis.plan_graph import (
    build_limits_needed,
    executor_workspace_limits,
    slow_arena_required,
)
from tigris.analysis.validation import OperatorSupportValidation
from tigris.cli import cli
from tigris.emitters.binary.defs import (
    FLAG_XIP,
    SEC_TENSORS,
)
from tigris.emitters.binary.reader import read_binary_plan
from tigris.emitters.binary.writer import emit_binary_bytes
from tigris.emitters.codegen import generate_c, generate_core_header
from tigris.pipeline import plan_model

from conftest import save_onnx, section_entries

TFLITE_OPS = Path(__file__).parent / "fixtures" / "tflite" / "ops"


@pytest.fixture
def unsupported_operator_plan(tmp_path):
    """A QDQ plan holding an operator no dispatcher can execute.

    InstanceNormalization has a schema opcode and no kernel on any backend, and normalization
    leaves it alone, so it reaches the plan intact. The operator itself is not
    the point: this fixture exists so codegen's capability check has something
    to refuse. The assertion below says so, because three earlier exemplars
    were chosen and then implemented.
    """
    from tigris.capabilities import KERNEL_CAPABILITIES, effective_operators

    routed = frozenset().union(
        *(effective_operators(backend) for backend in KERNEL_CAPABILITIES))
    assert "InstanceNormalization" not in routed, (
        "InstanceNormalization now has a kernel; pick another unrouted operator "
        "for this fixture")
    model_input = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, 4, 2]
    )
    model_output = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, 4, 2]
    )
    input_scale = numpy_helper.from_array(
        np.array([0.05], dtype=np.float32), "input_scale"
    )
    input_zp = numpy_helper.from_array(
        np.array([0], dtype=np.int8), "input_zp"
    )
    norm_scale = numpy_helper.from_array(np.ones(4, dtype=np.float32), "norm_scale")
    norm_bias = numpy_helper.from_array(np.zeros(4, dtype=np.float32), "norm_bias")
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
            "InstanceNormalization", ["input_dq", "norm_scale", "norm_bias"],
            ["padded"], name="unsupported"
        ),
        helper.make_node(
            "QuantizeLinear",
            ["padded", "output_scale", "output_zp"],
            ["output_q"],
        ),
        helper.make_node(
            "DequantizeLinear",
            ["output_q", "output_scale", "output_zp"],
            ["output"],
        ),
    ]
    model_path = save_onnx(tmp_path, "quantized_unsupported", nodes, [model_input], [model_output],
                           [input_scale, input_zp, norm_scale, norm_bias, output_scale, output_zp],
                           ir_version=8, check=True)

    analyzed = plan_model(model_path, (4096,))
    assert analyzed.is_quantized
    assert [op.op_type for op in analyzed.ops] == ["InstanceNormalization"]

    plan_path = tmp_path / "quantized_unsupported.tgrs"
    # InstanceNormalization has no runtime route on any backend, so validate_operator_support
    # now correctly rejects it at compile time (the fail-closed gate this
    # fixture predates). This fixture exists to exercise codegen's own,
    # separate defense-in-depth capability check against an already-serialized
    # plan (the same check that also guards a plan compiled by an older
    # toolchain version), so bypass only the compile-time gate to construct
    # the plan bytes; codegen's check below is untouched and still runs for
    # real.
    with patch(
        "tigris.analysis.validation.validate_operator_support",
        return_value=OperatorSupportValidation(issues=()),
    ):
        plan_path.write_bytes(emit_binary_bytes(analyzed))
    return plan_path


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
    ag = plan_model(linear_3op_path, (4096,))
    plan = read_binary_plan(emit_binary_bytes(ag, xip=True))

    assert plan["flags"] & FLAG_XIP


def test_codegen_reports_xip_and_loads_plan_at_runtime(linear_3op_path):
    ag = plan_model(linear_3op_path, (4096,))
    data = emit_binary_bytes(ag, xip=True)

    source = generate_c(read_binary_plan(data), "reference")

    assert "XIP:     yes" in source
    assert "static uint8_t *load_file" in source
    assert "tigris_plan_load(plan_buf, plan_len, &plan)" in source
    assert "tigris_mem_error_t merr = tigris_mem_init(" in source
    # Inputs are placed by the runtime; this one-stage plan reads them in place and needs no slow arena.
    assert "tigris_error_t ierr = tigris_input_alloc(&plan, &mem, tidx);" in source
    assert "if (merr != TIGRIS_MEM_OK)" in source and "if (ierr != TIGRIS_OK)" in source
    plan = read_binary_plan(data)
    assert slow_arena_required(plan) == 0
    assert "uint32_t slow_size = 0u;" in source
    assert "TIGRIS_GENERATED_EXECUTOR_WORKSPACE_BYTES" in source
    assert (
        "uint8_t executor_workspace"
        "[TIGRIS_GENERATED_EXECUTOR_WORKSPACE_BYTES]" in source
    )
    assert "tigris_run_with_workspace_buffer(" in source
    assert "&plan, &mem, tigris_dispatch_kernel, NULL, &stats," in source
    assert "executor_workspace, sizeof(executor_workspace)" in source


def test_posix_codegen_aligns_plan_and_arenas(linear_3op_path, conv_relu_chain_path):
    source = generate_c(
        read_binary_plan(emit_binary_bytes(plan_model(linear_3op_path, (4096,)))),
        "reference",
    )

    assert "#define _POSIX_C_SOURCE 200112L" in source
    assert "posix_memalign(&ptr, alignment, size)" in source
    assert "uint8_t *buf = allocate_aligned" in source
    assert "void *fast_buf = allocate_aligned(fast_size)" in source
    assert "void *slow_buf = NULL;" in source
    assert "if (!fast_buf || !tensor_ptrs)" in source
    staged = generate_c(
        read_binary_plan(emit_binary_bytes(plan_model(conv_relu_chain_path, (4096,)))),
        "reference")
    assert "void *slow_buf = allocate_aligned(slow_size)" in staged
    assert "if (!fast_buf || !slow_buf || !tensor_ptrs)" in staged
    assert "tigris_error_t ierr = tigris_input_alloc(&plan, &mem, tidx);" in staged


def test_quantized_esp_codegen_has_valid_includes(qdq_conv_path):
    ag = plan_model(qdq_conv_path, (4096,))
    source = generate_c(read_binary_plan(emit_binary_bytes(ag)), "esp-nn")

    assert "Kernels: esp-nn -> s8_ref fallback" in source
    assert "if (tigris_esp_nn_prepare(&plan, &mem) != 0)" in source
    assert "ESP-NN preparation failed" in source
    assert "tigris_fast_arena_required(&plan)" in source
    assert "tigris_weight_decompression_overhead(&plan)" not in source
    assert "tigris_mem_error_t merr = tigris_mem_init(" in source
    assert "tigris_error_t ierr = tigris_input_alloc(&plan, &mem, tidx);" in source
    assert "void *slow_buf = NULL;" in source
    assert "if (merr != TIGRIS_MEM_OK)" in source and "if (ierr != TIGRIS_OK)" in source
    assert "part->size < sizeof(tigris_file_header_t)" in source
    assert "(size_t)plan_size > part->size" in source
    assert '#include "tigris_kernels_esp_nn.h"' in source
    assert '#include "tigris_kernels_s8.h"' in source
    assert 'tigris_kernels_s8.h\\"' not in source


def test_quantized_codegen_reads_the_declared_interface(qdq_conv_path):
    """The harness prints what the model declares, not what the plan stores."""
    ag = plan_model(qdq_conv_path, (4096,))

    source = generate_c(read_binary_plan(emit_binary_bytes(ag)), "reference")

    assert "tigris_dispatch_kernel_s8" in source
    assert "tigris_output_read(&plan, &mem, tidx, staging, iface_bytes)" in source
    # Nothing reads the arena pointer as a typed array any more.
    assert "*out = (int8_t *)ptr" not in source
    assert "*out = (float *)ptr" not in source


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
    ag = plan_model(qdq_conv_path, (4096,))

    source = generate_c(
        read_binary_plan(emit_binary_bytes(ag)), backend, output_format="core",
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
            "= tigris_mem_init("
        )


def test_core_codegen_header_exposes_embedding_api(qdq_conv_path):
    header = generate_core_header(
        read_binary_plan(emit_binary_bytes(plan_model(qdq_conv_path, (4096,))))
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

    assert executor_workspace_limits(plan) == (9, 2, 3, 2, 2, 0)

    # The chain's spatial operators are summed over its stages, while the
    # build limit stays the most in one stage.
    plan["ops"][2]["op_type"] = 1
    assert executor_workspace_limits(plan) == (9, 2, 3, 2, 3, 0)
    assert build_limits_needed(plan) == (9, 2, 3, 2, 2)


@pytest.mark.parametrize("name,depth", [("float_if", 1), ("float_if_if", 2),
                                        ("float_while_if_through", 2), ("conv_bias", 0)])
def test_codegen_workspace_states_the_nesting_depth(name, depth, tmp_path):
    plan_path = tmp_path / "model.tgrs"
    result = CliRunner().invoke(cli, ["compile", str(TFLITE_OPS / f"{name}.tflite"),
                                      "-m", "256K", "-o", str(plan_path)])
    assert result.exit_code == 0, result.output
    assert executor_workspace_limits(read_binary_plan(plan_path.read_bytes()))[5] == depth


def test_compressed_core_header_uses_plan_alignment_for_arena_requirement(
    qdq_conv_path,
):
    data = emit_binary_bytes(plan_model(qdq_conv_path, (4096,), compress="lz4"), compress="lz4")
    plan = read_binary_plan(data)
    expected_reserve = max(
        (block["uncompressed_size"] + 31) // 32 * 32
        for block in plan["weight_blocks"]
    )

    header = generate_core_header(read_binary_plan(data))

    assert (
        f"TIGRIS_CODEGEN_WEIGHT_DECOMPRESSION_RESERVE_BYTES "
        f"{expected_reserve}u" in header
    )
    # The plan's scheduled peak, not its budget, plus the decompressed weights.
    assert plan["scheduled_peak"] < plan["budget"]
    assert (
        f"TIGRIS_CODEGEN_CORE_FAST_ARENA_BYTES "
        f"{plan['scheduled_peak'] + expected_reserve}u" in header
    )


def test_codegen_reads_the_plan_once_and_leaves_weight_blocks_encoded(
    qdq_conv_path, tmp_path, monkeypatch,
):
    import lz4.block

    import tigris.emitters.binary.reader as reader

    plan_path = tmp_path / "model.tgrs"
    plan_path.write_bytes(emit_binary_bytes(plan_model(qdq_conv_path, (4096,), compress="lz4"), compress="lz4"))
    reads = []
    read = reader.read_binary_plan

    def counted(data, **options):
        reads.append(options)
        return read(data, **options)

    def refuse(*args, **kwargs):
        raise AssertionError("codegen decoded a weight block")

    monkeypatch.setattr(reader, "read_binary_plan", counted)
    monkeypatch.setattr(lz4.block, "decompress", refuse)
    result = CliRunner().invoke(
        cli, ["codegen", str(plan_path), "--format", "core", "-o", str(tmp_path / "core.c")])

    assert result.exit_code == 0, result.output
    assert reads == [{"decompress_weights": False}]


def test_core_codegen_custom_name_is_linkable_alongside_default(qdq_conv_path):
    data = emit_binary_bytes(plan_model(qdq_conv_path, (4096,)))
    source = generate_c(
        read_binary_plan(data), "reference", output_format="core", core_header="audio_core.h",
        core_name="audio_codegen",
    )
    header = generate_core_header(read_binary_plan(data), "audio_codegen")

    assert "audio_codegen_load_plan" in source
    assert "audio_codegen_load_plan" in header
    assert "AUDIO_CODEGEN_TENSOR_CAPACITY" in header
    assert "#ifndef AUDIO_CODEGEN_CORE_H" in header
    assert "tigris_codegen_load_plan" not in source


def test_cli_core_writes_matched_source_and_header(linear_3op_path, tmp_path):
    plan_data = emit_binary_bytes(plan_model(linear_3op_path, (4096,)))
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
    ag = plan_model(qdq_conv_path, (4096,))
    data = bytearray(emit_binary_bytes(ag))

    data[section_entries(data)[SEC_TENSORS][1] + 11] = TensorProto.FLOAT

    with pytest.raises(
        ValueError, match="cannot select one graph-wide runtime dispatcher"
    ):
        generate_c(read_binary_plan(bytes(data)), "reference")


def test_codegen_rejects_plan_without_executable_stages(linear_3op_path):
    ag = plan_model(linear_3op_path)

    with pytest.raises(ValueError, match="Plan has no executable stages"):
        generate_c(read_binary_plan(emit_binary_bytes(ag)), "reference")


def test_cmsis_codegen_rejects_unrepresentable_static_slow_arena(
    linear_3op_path, monkeypatch,
):
    import tigris.emitters.codegen as codegen

    ag = plan_model(linear_3op_path, (4096,))
    monkeypatch.setattr(codegen, "slow_arena_required", lambda plan: 0x100000000)
    with pytest.raises(ValueError, match="static slow arena exceeds"):
        generate_c(read_binary_plan(emit_binary_bytes(ag)), "cmsis-nn")


@pytest.mark.parametrize("backend", ["reference", "cmsis-nn", "esp-nn"])
def test_apps_size_the_slow_arena_from_the_plan(conv_relu_chain_path, backend):
    """The slow arena is what the plan needs, not a multiple of the fast budget."""
    data = emit_binary_bytes(plan_model(conv_relu_chain_path, (4096,)))
    required = slow_arena_required(read_binary_plan(data))
    source = generate_c(read_binary_plan(data), backend)
    assert 0 < required < 256 * 1024
    if backend == "cmsis-nn":
        assert f"static uint8_t slow_arena[{required}]" in source
    else:
        assert f"uint32_t slow_size = {required}u;" in source


def test_quantized_cmsis_codegen_checks_prepare_and_memory(qdq_conv_path):
    ag = plan_model(qdq_conv_path, (4096,))

    source = generate_c(read_binary_plan(emit_binary_bytes(ag)), "cmsis-nn")

    assert "tigris_cmsis_nn_fast_arena_required(&plan)" in source
    assert "Increase TIGRIS_CMSIS_NN_SCRATCH_BYTES" in source
    assert "if (tigris_cmsis_nn_prepare(&plan, &mem) != 0)" in source
    assert "CMSIS-NN preparation failed" in source
    assert "tigris_mem_error_t merr = tigris_mem_init(" in source
    assert "tigris_error_t ierr = tigris_input_alloc(&plan, &mem, tidx);" in source
    assert "slow_arena" not in source
    assert "fast_arena, sizeof(fast_arena), NULL, 0u);" in source
    assert "if (merr != TIGRIS_MEM_OK)" in source and "if (ierr != TIGRIS_OK)" in source


def test_compressed_cmsis_arena_includes_static_weight_reserve(qdq_conv_path):
    ag = plan_model(qdq_conv_path, (4096,), compress="lz4")
    data = emit_binary_bytes(ag, compress="lz4")
    plan = read_binary_plan(data)
    blocks = plan["weight_blocks"]
    assert blocks
    expected_overhead = max(
        (block["uncompressed_size"] + 15) // 16 * 16 for block in blocks
    )

    source = generate_c(read_binary_plan(data), "cmsis-nn")

    expected_core = (plan["scheduled_peak"] + expected_overhead + 15) // 16 * 16
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
    ag = plan_model(linear_3op_path, (4096,))

    source = generate_c(read_binary_plan(emit_binary_bytes(ag)), backend)

    assert (
        f"Kernels: reference (explicit float32 fallback; "
        f"{backend} acceleration is int8-only)"
    ) in source
    assert "tigris_dispatch_kernel, NULL, &stats" in source


@pytest.mark.parametrize("backend", ["reference", "esp-nn", "cmsis-nn"])
def test_unsupported_operator_codegen_fails_before_output(
    unsupported_operator_plan, tmp_path, backend
):
    output = tmp_path / f"unsupported-{backend}.c"

    result = CliRunner().invoke(
        cli,
        [
            "codegen",
            str(unsupported_operator_plan),
            "--backend",
            backend,
            "--output",
            str(output),
        ],
    )

    assert result.exit_code != 0
    assert f"Backend '{backend}' cannot execute this int8 plan" in result.output
    # The plan keeps no operator names, as compile writes it by default.
    assert "Unsupported operators: InstanceNormalization" in result.output
    assert not output.exists()


def test_codegen_preserves_existing_output_on_capability_failure(
    unsupported_operator_plan, tmp_path
):
    output = tmp_path / "existing.c"
    output.write_text("sentinel")

    result = CliRunner().invoke(
        cli,
        [
            "codegen",
            str(unsupported_operator_plan),
            "--backend",
            "reference",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code != 0
    assert output.read_text() == "sentinel"


def _tensor(dtype, flags=0):
    return {"dtype": dtype, "flags": flags}


def test_kernel_macros_route_as_the_runtime_dispatches():
    from tigris.emitters.binary.defs import OP_TYPE_MAP
    from tigris.emitters.codegen import kernel_macros

    def op(name, inputs):
        return {"op_type": OP_TYPE_MAP[name], "inputs": inputs, "outputs": []}

    # int8 activations 0-1, float32 2-3, int32 4, an int8 constant 5.
    int8_plan = {"tensors": [_tensor(3), _tensor(3), _tensor(1), _tensor(1), _tensor(6),
                             _tensor(3, flags=0x01)],
                 "ops": [op("Conv", [0]), op("Dequantize", [1]), op("Add", [2, 3]),
                         op("Add", [4, 4]), op("Quantize", [3]), op("If", [0]),
                         op("Relu", [5])]}
    assert kernel_macros(int8_plan) == [
        "TIGRIS_KERNEL_F32_ADD", "TIGRIS_KERNEL_I32", "TIGRIS_KERNEL_S8_CONV",
        "TIGRIS_KERNEL_S8_DEQUANTIZE", "TIGRIS_KERNEL_S8_QUANTIZE", "TIGRIS_KERNEL_S8_RELU"]

    # Without an int8 activation every kernel is a float32 one; an int8
    # constant does not make a plan int8.
    float_plan = {"tensors": [_tensor(1), _tensor(3, flags=0x01)],
                  "ops": [op("Relu", [0]), op("Gather", [1, 0]), op("While", [0])]}
    assert kernel_macros(float_plan) == ["TIGRIS_KERNEL_F32_GATHER", "TIGRIS_KERNEL_F32_RELU"]


def test_codegen_writes_the_kernel_selection_header(linear_3op_path, tmp_path):
    from tigris.emitters.codegen import generate_kernel_config, kernel_macros

    data = emit_binary_bytes(plan_model(linear_3op_path, (4096,)))
    header = generate_kernel_config(read_binary_plan(data), "MODEL_KERNELS_H")
    assert "#define TIGRIS_KERNEL_DEFAULT 0\n" in header
    macros = kernel_macros(read_binary_plan(data))
    assert macros and all(f"#define {m} 1\n" in header for m in macros)
    assert all(m.startswith("TIGRIS_KERNEL_F32_") for m in macros)
    with pytest.raises(ValueError):
        generate_kernel_config(read_binary_plan(data), "not a guard")

    plan_path = tmp_path / "model.tgrs"
    plan_path.write_bytes(data)
    result = CliRunner().invoke(cli, ["codegen", str(plan_path), "-o", str(tmp_path / "app.c")])
    assert result.exit_code == 0, result.output
    written = (tmp_path / "app_kernels.h").read_text()
    assert written == generate_kernel_config(read_binary_plan(data), "APP_KERNELS_H")
    assert f"kernels {tmp_path / 'app_kernels.h'}" in result.output
