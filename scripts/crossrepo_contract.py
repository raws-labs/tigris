#!/usr/bin/env python3
"""Prove a deterministic compiler -> runtime contract against ONNX Runtime.

Plans emitted by the compiler execute with the runtime's public loader, arena,
executor, and raw I/O contract on the float32 and s8 reference dispatchers.

Every run of an int8 plan on s8 is replayed, with the same arguments and
inputs, on the CMSIS-NN dispatcher and on the ESP-NN dispatcher over both of
ESP-NN's portable C variants, in runners built with the same tensor alignment
and linked against the vendors' portable C kernels under AddressSanitizer and
UndefinedBehaviorSanitizer. Each replay must return the s8 output bytes
exactly, report the s8 run's activation memory figures with only its vendor
scratch added to the allocation, size the CMSIS-NN scratch to what its
vendor calls were given, leave the guard bytes around every arena untouched,
hand no operator its adapter takes to s8, and report the calls each wrapped
vendor function received. Cases state the vendor functions their plan must
call on each dispatcher, or that it calls none, and every wrapped function
must be called by some case, while preparing or running, unless it is listed
as unreachable with the reason. Vector and assembly kernels are not covered on
the host.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import re
import subprocess
import tempfile
from collections import Counter, deque
from concurrent.futures import Future, ThreadPoolExecutor
from collections.abc import Collection, Mapping
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

from tigris import SCHEMA_VERSION, TILE_AXIS_HW
from tigris.analysis.plan_graph import (
    executor_workspace_limits,
    graph_from_plan,
    slow_arena_required,
)
from tigris.analysis.validation import io_in_fast, slow_pool_usage, validate_memory_plan
from tigris.capabilities import KERNEL_CAPABILITIES
from tigris.emitters.binary.defs import (
    COMPRESS_LZ4,
    FLAG_XIP,
    STAGE_FLAG_LINE_BUFFERED,
    TENSOR_FLAG_CONSTANT,
    TENSOR_FLAG_LINEAR,
    OP_TYPE_MAP,
)
from tigris.emitters.binary.reader import read_binary_plan
from tigris.emitters.binary.writer import emit_binary
from tigris.operator_contract import OP_TYPE_BY_CODE
from tigris.pipeline import plan_model, reserve_weights
from tigris.runtime import Session, check_trace, trace_units
from tigris.graph.ir import Stage

if __package__:
    from .contract_cases import (
        chains, common, control_flow, convolution, elementwise, matmul, models, movement, qdq,
        reduction, tiling, tiling_2d,
    )
    from .contract_cases.common import Array, ContractCase, _model
else:
    from contract_cases import (
        chains, common, control_flow, convolution, elementwise, matmul, models, movement, qdq,
        reduction, tiling, tiling_2d,
    )
    from contract_cases.common import Array, ContractCase, _model


_DTYPE_BY_ONNX_CODE = {
    TensorProto.FLOAT: np.dtype("<f4"),
    TensorProto.UINT8: np.dtype("u1"),
    TensorProto.INT8: np.dtype("i1"),
    TensorProto.INT32: np.dtype("<i4"),
    TensorProto.INT64: np.dtype("<i8"),
    TensorProto.BOOL: np.dtype("bool"),
}
_INT8_LSB_TOLERANCE = 1


class _AccelDispatch(NamedTuple):
    """An accelerated runner target, the --dispatch it runs, and the prefix
    of the vendor symbols that dispatcher's adapter calls."""

    runner: str
    dispatch: str
    prefix: str


# Dispatchers an int8 plan also runs on, each to the s8 reference's bytes.
# The ANSI runner links ESP-NN without CONFIG_NN_OPTIMIZED.
_ACCEL_DISPATCHES = {
    "cmsis-nn": _AccelDispatch("tigris_contract_runner_accel", "cmsis-nn", "arm_"),
    "esp-nn": _AccelDispatch("tigris_contract_runner_accel", "esp-nn", "esp_nn_"),
    "esp-nn-ansi": _AccelDispatch("tigris_contract_runner_accel_ansi", "esp-nn", "esp_nn_"),
}
# Wrapped vendor symbols no plan reaches in the host build, each with the reason.
_VENDOR_UNREACHABLE = {
    # Only the MVE build of arm_fully_connected_s8_get_buffer_size asks for a
    # buffer; the portable C build returns 0, and the adapter computes the
    # kernel sums into that buffer only when it has one.
    "arm_vector_sum_s8": "the portable C FullyConnected takes no kernel-sum buffer",
}


def _to_runtime_layout(value: Array) -> Array:
    if value.ndim == 4:
        return np.ascontiguousarray(value.transpose(0, 2, 3, 1))
    if value.ndim == 3:
        return np.ascontiguousarray(value.transpose(0, 2, 1))
    return np.ascontiguousarray(value)


def _from_runtime_layout(value: Array, original_ndim: int) -> Array:
    if original_ndim == 4:
        return np.ascontiguousarray(value.transpose(0, 3, 1, 2))
    if original_ndim == 3:
        return np.ascontiguousarray(value.transpose(0, 2, 1))
    return value


def _compile_plan(
    model_path: Path,
    plan_path: Path,
    *,
    mem_budget: str,
    slow_budget: str | None,
    compression: str | None,
    xip: bool,
    force_one_op_stages: bool = False,
    int8_io: bool = False,
) -> dict:
    budgets = (mem_budget,) if slow_budget is None else (mem_budget, slow_budget)
    # Forced stages replace the planned ones, so their weights are reserved
    # for the stages emitted rather than for the discarded layout.
    graph = plan_model(model_path, budgets, compress=None if force_one_op_stages else compression,
                       int8_io=int8_io)
    if force_one_op_stages:
        stages: list[Stage] = []
        for index, op in enumerate(graph.ops):
            op.stage = index
            inputs = [
                name
                for name in op.inputs
                if name in graph.tensors and not graph.tensors[name].is_constant
            ]
            outputs = [
                name
                for name in op.outputs
                if name in graph.tensors and not graph.tensors[name].is_constant
            ]
            peak = sum(
                (graph.tensors[name].size_bytes + 31) & ~31
                for name in dict.fromkeys(inputs + outputs)
            )
            stages.append(
                Stage(
                    stage_id=index,
                    op_indices=[index],
                    input_tensors=inputs,
                    output_tensors=outputs,
                    peak_bytes=peak,
                )
            )
        graph.stages = stages
        if compression:
            reserve_weights(graph)
    validation = validate_memory_plan(graph)
    if not validation.feasible:
        details = "; ".join(issue.describe() for issue in validation.issues)
        raise AssertionError(f"compiler produced an infeasible graph: {details}")
    emit_binary(graph, plan_path, compress=compression, xip=xip)
    plan_bytes = plan_path.read_bytes()
    plan = read_binary_plan(plan_bytes)
    plan["_compiler_scheduled_peak"] = validation.scheduled_peak_bytes
    plan["_compiler_slow_budget"] = graph.budget.slow
    plan["_compiler_slow_peak"] = slow_pool_usage(graph).slow_peak_bytes
    # Generated applications size their slow arena from the plan alone.
    derived = slow_arena_required(read_binary_plan(plan_bytes, decompress_weights=False))
    if derived != plan["_compiler_slow_peak"]:
        raise AssertionError(
            f"slow arena derived from the plan ({derived}) differs from the "
            f"compiler's slow requirement ({plan['_compiler_slow_peak']})")
    plan["_graph"] = graph
    return plan


def _declared_dtype(tensor: dict) -> int:
    """The dtype the model states for this boundary, which the runner converts."""
    return tensor["iface_dtype"] or tensor["dtype"]


def _pack_inputs(plan: dict, inputs: dict[str, Array]) -> bytes:
    chunks: list[bytes] = []
    for tensor_index in plan["model_inputs"]:
        tensor = plan["tensors"][tensor_index]
        source = inputs[tensor["name"]]
        # The runner is handed the dtype the model declares and converts it,
        # so the plan is exercised through the interface the model states.
        declared = _declared_dtype(tensor)
        if declared == TensorProto.FLOAT:
            encoded = source.astype(np.float32, copy=False)
        elif declared == TensorProto.INT8:
            if source.dtype != np.int8:
                raise AssertionError(
                    f"int8 input {tensor['name']!r} is given as {source.dtype}")
            encoded = source
        elif declared == TensorProto.BOOL:
            encoded = source.astype(np.bool_, copy=False)
        elif declared in (TensorProto.INT32, TensorProto.INT64):
            encoded = source.astype(_DTYPE_BY_ONNX_CODE[declared], copy=False)
        elif declared == TensorProto.UINT8:
            # A camera hands over bytes; the runtime moves them onto the int8
            # tensor itself, so the harness passes them through untouched.
            encoded = source.astype(np.uint8, copy=False)
        else:
            raise AssertionError(
                f"unsupported contract input dtype {declared}"
            )
        if not tensor["flags"] & TENSOR_FLAG_LINEAR:
            encoded = _to_runtime_layout(encoded)
        expected = tensor["size_bytes"]
        if declared != tensor["dtype"]:
            expected = (
                tensor["size_bytes"] // np.dtype(_DTYPE_BY_ONNX_CODE[tensor["dtype"]]).itemsize
            ) * np.dtype(_DTYPE_BY_ONNX_CODE[declared]).itemsize
        if encoded.nbytes != expected:
            raise AssertionError(
                f"input {tensor['name']!r} has {encoded.nbytes} bytes, "
                f"the declared interface expects {expected}"
            )
        chunks.append(encoded.tobytes())
    return b"".join(chunks)


def _decode_outputs(
    plan: dict, raw: bytes, reference_outputs: list[Array]
) -> list[Array]:
    decoded: list[Array] = []
    offset = 0
    if len(plan["model_outputs"]) != len(reference_outputs):
        raise AssertionError("plan and ONNX Runtime output counts differ")
    for tensor_index, reference in zip(plan["model_outputs"], reference_outputs):
        tensor = plan["tensors"][tensor_index]
        declared = _declared_dtype(tensor)
        dtype = _DTYPE_BY_ONNX_CODE.get(declared)
        if dtype is None:
            raise AssertionError(
                f"unsupported contract output dtype {declared}"
            )
        stored = _DTYPE_BY_ONNX_CODE[tensor["dtype"]]
        size_bytes = (
            tensor["size_bytes"] // np.dtype(stored).itemsize
        ) * np.dtype(dtype).itemsize
        end = offset + size_bytes
        if end > len(raw):
            raise AssertionError("runtime output file is truncated")
        value = np.frombuffer(raw[offset:end], dtype=dtype).copy()
        value = value.reshape(tensor["shape"])
        # Schema 7 records the order a tensor is stored in, so the harness reads
        # it instead of inferring it from the shape of the producing graph. That
        # guess could not tell a model's own terminal Transpose from a layout
        # conversion the compiler inserted, since both are a Transpose writing a
        # model output.
        if not tensor["flags"] & TENSOR_FLAG_LINEAR:
            value = _from_runtime_layout(value, reference.ndim)
        decoded.append(value)
        offset = end
    if offset != len(raw):
        raise AssertionError("runtime output file has trailing bytes")
    return decoded


def _run(
    command: list[str], description: str, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(command, text=True, capture_output=True, check=False, env=env)
    if completed.returncode != 0:
        raise RuntimeError(
            f"{description} failed (exit {completed.returncode}):\n"
            f"{completed.stdout}{completed.stderr}"
        )
    return completed


_MEMORY_REPORT = re.compile(
    r"^TIGRIS_CONTRACT_MEMORY budget=(\d+) activation_limit=(\d+) "
    r"reserve=(\d+) required=(\d+) allocated=(\d+) peak=(\d+) "
    r"slow_peak=(\d+)$",
    re.MULTILINE,
)


class _MemoryReport(NamedTuple):
    budget: int
    activation_limit: int
    reserve: int
    required: int
    allocated: int
    peak: int
    slow_peak: int


def _memory_report(case_name: str, runtime_stdout: str) -> _MemoryReport:
    match = _MEMORY_REPORT.search(runtime_stdout)
    if match is None:
        raise AssertionError(f"{case_name}: runtime emitted no memory report")
    return _MemoryReport(*map(int, match.groups()))


def _assert_memory_contract(
    case_name: str, plan: dict, runtime_stdout: str | _MemoryReport, *, full_budget: bool = False
) -> None:
    (
        budget,
        activation_limit,
        reserve,
        required,
        allocated,
        measured_peak,
        slow_peak,
    ) = (runtime_stdout if isinstance(runtime_stdout, _MemoryReport)
         else _memory_report(case_name, runtime_stdout))
    scheduled_peak = int(plan["_compiler_scheduled_peak"])
    if budget != plan["budget"]:
        raise AssertionError(
            f"{case_name}: runtime budget {budget} != plan budget {plan['budget']}"
        )
    # A plan that states its scheduled peak is sized by it, an older one by its budget.
    stated = plan["scheduled_peak"]
    if stated is not None and stated != scheduled_peak:
        raise AssertionError(
            f"{case_name}: plan states scheduled peak {stated}, the compiler "
            f"validated {scheduled_peak}"
        )
    activations = budget if stated is None else stated
    if required != activations + reserve:
        raise AssertionError(
            f"{case_name}: required arena {required} != activations + reserve "
            f"({activations} + {reserve})"
        )
    expected_limit = budget if full_budget else scheduled_peak
    if activation_limit != expected_limit:
        raise AssertionError(
            f"{case_name}: runner activation limit {activation_limit} != "
            f"requested activation limit {expected_limit}"
        )
    if scheduled_peak > budget:
        raise AssertionError(
            f"{case_name}: compiler scheduled peak {scheduled_peak} exceeds "
            f"activation budget {budget}"
        )
    if allocated != expected_limit + reserve:
        raise AssertionError(
            f"{case_name}: allocated arena {allocated} != compiler core "
            f"allocation {expected_limit + reserve}"
        )
    if measured_peak > scheduled_peak + reserve:
        raise AssertionError(
            f"{case_name}: runtime peak {measured_peak} exceeds compiler "
            f"core estimate {scheduled_peak + reserve} "
            f"({scheduled_peak} activations + {reserve} reserve)"
        )
    # The slow pool is sized by the compiler and filled by the runtime, and
    # each derives what shares a buffer from the plan on its own. A case that
    # names a slow budget is the check that those two rules still agree.
    predicted = int(plan["_compiler_slow_peak"])
    if slow_peak > predicted:
        raise AssertionError(
            f"{case_name}: runtime slow peak {slow_peak} exceeds the compiler's "
            f"slow requirement {predicted}"
        )
    slow_budget = int(plan["_compiler_slow_budget"])
    if slow_budget > 0 and slow_peak > slow_budget:
        raise AssertionError(
            f"{case_name}: runtime slow peak {slow_peak} exceeds the slow "
            f"budget {slow_budget} the compiler accepted"
        )


def _assert_runs_in_slow_figure(
    case: ContractCase, plan: dict, runner: _ContractRunner, plan_path: Path,
    inputs_path: Path, outputs_path: Path, runtime_stdout: str,
) -> None:
    """The runtime's slow peak is the compiler's slow figure, and the plan runs
    in a slow arena of exactly that size, to the same bytes."""
    figure = int(plan["_compiler_slow_peak"])
    slow_peak = _memory_report(case.name, runtime_stdout).slow_peak
    if slow_peak != figure:
        raise AssertionError(
            f"{case.name}: runtime slow peak {slow_peak} != the compiler's slow "
            f"requirement {figure}")
    exact = outputs_path.with_name("exact-slow.bin")
    runner.run(case.name, plan, plan_path, inputs_path, exact,
               [str(plan["_compiler_scheduled_peak"]), f"--slow-arena={figure}"],
               "run in the compiler's slow figure", routing=case, full_budget=False)
    if exact.read_bytes() != outputs_path.read_bytes():
        raise AssertionError(f"{case.name}: the run in the compiler's slow figure changed the outputs")


def _assert_runs_without_slow(
    case_name: str, plan: dict, runner: _ContractRunner, plan_path: Path, inputs_path: Path,
    outputs_path: Path, limit: int, routing: _Routing | None,
) -> bool:
    """For a plan that keeps its model inputs and outputs in the fast pool, the
    compiler must state no slow memory, and the runtime must run it without a
    slow arena at the compiler's fast figure, to the same bytes. Inputs the
    caller allocates in slow instead are copied through it at the same fast
    figure, to the same bytes. Returns whether the plan is one."""
    if not io_in_fast(graph_from_plan(plan)):
        return False
    if plan["_compiler_slow_peak"] != 0:
        raise AssertionError(
            f"{case_name}: the compiler needs {plan['_compiler_slow_peak']} slow bytes "
            "for a plan that keeps its inputs and outputs in the fast pool")
    expected = outputs_path.read_bytes()
    alone = outputs_path.with_name("no-slow.bin")
    runner.run(case_name, plan, plan_path, inputs_path, alone, [str(limit), "--slow-arena=0"],
               "run without a slow arena", routing=routing,
               full_budget=limit != plan["_compiler_scheduled_peak"])
    if alone.read_bytes() != expected:
        raise AssertionError(f"{case_name}: the run without a slow arena changed the outputs")
    copied = outputs_path.with_name("inputs-in-slow.bin")
    completed = runner.run(case_name, plan, plan_path, inputs_path, copied,
                           [str(limit), "--inputs-in-slow"], "run with inputs in slow",
                           routing=routing)
    report = _memory_report(case_name, completed.stdout)
    if report.peak > plan["_compiler_scheduled_peak"] + report.reserve or report.slow_peak == 0:
        raise AssertionError(
            f"{case_name}: inputs in slow peaked at {report.peak} fast and "
            f"{report.slow_peak} slow bytes")
    if copied.read_bytes() != expected:
        raise AssertionError(f"{case_name}: the run with inputs in slow changed the outputs")
    return True


def _run_peak_fixtures(runners: tuple[_ContractRunner, ...], work_dir: Path) -> None:
    fixtures = Path(__file__).resolve().parents[1] / "tests/fixtures/tflite/ops"
    # concat_rank2 and lstm_unrolled hold reshape groups over a stage input
    # and outputs nothing reads, which the runtime once kept live to the end
    # of the stage and pushed later outputs past the compiler's slow figure.
    for name, minimum in (("concat_rows", 96), ("select_v2_broadcast", 128),
                          ("select_v2_rank", 128), ("float_space_to_depth", 192),
                          ("concat_rank2", 96), ("lstm_unrolled", 160)):
        model_path = fixtures / f"{name}.tflite"
        graph = plan_model(model_path, (minimum - 1,))
        assert not validate_memory_plan(graph).feasible, name
        with np.load(fixtures / f"{name}.npz") as samples:
            for budget in (minimum, 256):
                case_name = f"{name}_peak_{budget}"
                case_dir = work_dir / case_name
                case_dir.mkdir()
                plan_path = case_dir / "model.tgrs"
                plan = _compile_plan(
                    model_path, plan_path, mem_budget=str(budget),
                    slow_budget=None, compression=None, xip=False,
                )
                input_path = case_dir / "inputs.bin"
                output_path = case_dir / "outputs.bin"
                for sample in range(len(samples["input_0"])):
                    input_path.write_bytes(b"".join(
                        samples[f"input_{i}"][sample].tobytes()
                        for i in range(len(plan["model_inputs"]))
                    ))
                    expected = b"".join(
                        samples[f"output_{i}"][sample].tobytes()
                        for i in range(len(plan["model_outputs"]))
                    )
                    for runner in runners:
                        for full_budget in (False, True):
                            limit = budget if full_budget else plan["_compiler_scheduled_peak"]
                            runner.run(case_name, plan, plan_path, input_path, output_path,
                                       [str(limit)], f"sample {sample}",
                                       routing=_ROUTINGS.get(name), full_budget=full_budget)
                            assert output_path.read_bytes() == expected, case_name
                            _assert_runs_without_slow(case_name, plan, runner, plan_path,
                                                      input_path, output_path, limit,
                                                      _ROUTINGS.get(name))
                        runner.refuse_below_peak(case_name, plan, plan_path, input_path,
                                                 output_path)
                print(f"PASS {case_name}")


def _build_environment() -> dict[str, str]:
    """Environment for `cmake --build`: CMAKE_BUILD_PARALLEL_LEVEL jobs, 4 unless already set."""
    return {**os.environ,
            "CMAKE_BUILD_PARALLEL_LEVEL": os.environ.get("CMAKE_BUILD_PARALLEL_LEVEL", "4")}


def _build_runner(
    runtime: Path, build_dir: Path, *, alignment: int | None = None
) -> tuple[Path, Path]:
    """Build the default and rows-instrumented contract runners.

    The rows-instrumented runner compiles the runtime sources with
    TIGRIS_COUNT_KERNEL_ROWS so the gate can read kernel output-row counts. The
    default runner and the shipping library stay free of that test-only counter.
    """
    _run(
        [
            "cmake",
            "-S",
            str(runtime),
            "-B",
            str(build_dir),
            "-DCMAKE_BUILD_TYPE=Release",
            *([f"-DCMAKE_C_FLAGS=-DTIGRIS_TENSOR_ALIGN={alignment}"]
              if alignment is not None else []),
        ],
        "runtime configure",
    )
    _run(
        [
            "cmake",
            "--build",
            str(build_dir),
            "--target",
            "tigris_contract_runner",
            "tigris_contract_runner_rows",
        ],
        "runtime contract-runner build",
        _build_environment(),
    )
    return (
        build_dir / "tigris_contract_runner",
        build_dir / "tigris_contract_runner_rows",
    )


def _build_accel_runner(runtime: Path, build_dir: Path, *, alignment: int | None = None) -> Path:
    """Build the contract runners on the CMSIS-NN and ESP-NN dispatchers and
    return their build directory.

    The runtime fetches both vendors' sources at the commits it pins and links
    their portable C kernels, so the host runs the adapters' own code paths.
    FETCHCONTENT_SOURCE_DIR_CMSIS_NN and FETCHCONTENT_SOURCE_DIR_ESP_NN, when
    set in the environment, are passed to the configure, which builds from
    those checkouts without network access once it has checked that each is
    unmodified at its pinned commit.
    """
    sources = [f"-D{name}={os.environ[name]}" for name in
               ("FETCHCONTENT_SOURCE_DIR_CMSIS_NN", "FETCHCONTENT_SOURCE_DIR_ESP_NN")
               if os.environ.get(name)]
    _run(["cmake", "-S", str(runtime), "-B", str(build_dir), "-DCMAKE_BUILD_TYPE=Release",
          "-DTIGRIS_CONTRACT_ACCEL=ON", *sources,
          *([f"-DCMAKE_C_FLAGS=-DTIGRIS_TENSOR_ALIGN={alignment}"] if alignment is not None else [])],
         "accelerated runtime configure")
    _run(["cmake", "--build", str(build_dir), "--target",
          *sorted({dispatch.runner for dispatch in _ACCEL_DISPATCHES.values()})],
         "accelerated contract-runner build", _build_environment())
    return build_dir


def _is_int8_plan(plan: dict) -> bool:
    """Whether the runtime runs the plan on int8 kernels: some activation is int8."""
    return any(not tensor["flags"] & TENSOR_FLAG_CONSTANT
               and tensor["dtype"] == TensorProto.INT8 for tensor in plan["tensors"])


def _first_difference(plan: dict, expected: bytes, actual: bytes) -> str:
    """The first model output element, in stored order, whose bytes differ."""
    if len(actual) != len(expected):
        return f"{len(actual)} output bytes, the s8 run wrote {len(expected)}"
    offset = 0
    for number, tensor_index in enumerate(plan["model_outputs"]):
        tensor = plan["tensors"][tensor_index]
        dtype = _DTYPE_BY_ONNX_CODE[_declared_dtype(tensor)]
        size = tensor["size_bytes"] // _DTYPE_BY_ONNX_CODE[tensor["dtype"]].itemsize * dtype.itemsize
        count = size // dtype.itemsize
        # Compare each element's bytes, so float NaNs and bools compare as stored.
        mine = np.frombuffer(actual, np.uint8, size, offset).reshape(count, dtype.itemsize)
        theirs = np.frombuffer(expected, np.uint8, size, offset).reshape(count, dtype.itemsize)
        differing = np.flatnonzero((mine != theirs).any(axis=1))
        if differing.size:
            index = int(differing[0])
            value = np.frombuffer(actual, dtype, count, offset)[index]
            wanted = np.frombuffer(expected, dtype, count, offset)[index]
            return (f"output {number} first differs at element {index} ({value} vs s8 {wanted}), "
                    f"{differing.size} of {count} elements differ")
        offset += size
    return "no element differs"


_DISPATCH_REPORT = re.compile(r"^TIGRIS_CONTRACT_DISPATCH name=(\S+) scratch=(\d+)$", re.MULTILINE)
_VENDOR_REPORT = re.compile(r"^TIGRIS_CONTRACT_VENDOR((?: \w+=\d+)*)$", re.MULTILINE)
_VENDOR_PREPARE_REPORT = re.compile(r"^TIGRIS_CONTRACT_VENDOR_PREPARE((?: \w+=\d+)*)$",
                                    re.MULTILINE)
_FALLBACK_REPORT = re.compile(r"^TIGRIS_CONTRACT_FALLBACK((?: \w+=\d+)*)$", re.MULTILINE)
_SCRATCH_REPORT = re.compile(
    r"^TIGRIS_CONTRACT_SCRATCH requested=(\d+) quant_channels=(\d+) tile_routed=(\d+)$",
    re.MULTILINE)
# The executor's refusal of a fast arena below the plan's scheduled peak.
_REFUSED_BELOW_PEAK = "inference failed: arena too small for the plan after 0 stages"


class _Routing(NamedTuple):
    """Vendor calls every run of a plan makes, with the fields of ContractCase:
    per dispatcher the symbols each run calls at least once, and the
    dispatchers on which no run calls any vendor function."""

    expect_vendor: Mapping[str, Collection[str]] = {}
    expect_no_vendor: Collection[str] = ()


# The vendor functions each adapter entry point calls in one run, per dispatcher.
_CONV = {"cmsis-nn": {"arm_convolve_wrapper_s8", "arm_convolve_wrapper_s8_get_buffer_size"},
         **{name: {f"esp_nn_conv_s8_{variant}", f"esp_nn_get_conv_scratch_size_{variant}",
                   f"esp_nn_set_conv_scratch_buf_{variant}"}
            for name, variant in (("esp-nn", "opt"), ("esp-nn-ansi", "ansi"))}}
_DEPTHWISE = {"cmsis-nn": {"arm_depthwise_conv_wrapper_s8",
                           "arm_depthwise_conv_wrapper_s8_get_buffer_size"},
              **{name: {f"esp_nn_depthwise_conv_s8_{variant}",
                        f"esp_nn_get_depthwise_conv_scratch_size_{variant}"}
                 for name, variant in (("esp-nn", "opt"), ("esp-nn-ansi", "ansi"))}}
# Both ESP-NN variants run these kernels from the ANSI C sources.
_FULLY_CONNECTED = {"cmsis-nn": {"arm_fully_connected_s8", "arm_fully_connected_s8_get_buffer_size"},
                    **dict.fromkeys(("esp-nn", "esp-nn-ansi"), {"esp_nn_fully_connected_s8_ansi"})}
_FULLY_CONNECTED_PER_CHANNEL = {
    "cmsis-nn": {"arm_fully_connected_per_channel_s8", "arm_fully_connected_s8_get_buffer_size"},
    **dict.fromkeys(("esp-nn", "esp-nn-ansi"), {"esp_nn_fully_connected_per_ch_s8_ansi"})}
_AVERAGE_POOL = {"cmsis-nn": {"arm_avgpool_s8", "arm_avgpool_s8_get_buffer_size"},
                 **dict.fromkeys(("esp-nn", "esp-nn-ansi"), {"esp_nn_avg_pool_s8_ansi"})}
_MUL = {"cmsis-nn": {"arm_elementwise_mul_s8"},
        **dict.fromkeys(("esp-nn", "esp-nn-ansi"), {"esp_nn_mul_elementwise_s8_ansi"})}
# ESP-NN has no Max or Min adapter, and the policy sends every binary
# operator but Mul to s8 there.
_MAX = {"cmsis-nn": {"arm_maximum_s8"}}
_MIN = {"cmsis-nn": {"arm_minimum_s8"}}


def _calls(*entries: dict[str, set[str]]) -> _Routing:
    calls: dict[str, frozenset[str]] = {}
    for entry in entries:
        for dispatch, symbols in entry.items():
            calls[dispatch] = calls.get(dispatch, frozenset()) | frozenset(symbols)
    return _Routing(calls, tuple(name for name in _ACCEL_DISPATCHES if name not in calls))


_NO_VENDOR = _Routing({}, tuple(_ACCEL_DISPATCHES))

# Vendor calls of the cases whose plans reach an adapter entry point on a
# tiled stage, of one untiled case per entry point, and of the cases whose
# operators the routing policy (tigris_accel_try_s8_ref and
# tigris_accel_pre_route in the runtime) keeps on s8. Each entry follows from
# that policy; the gate sets them on the cases by name.
_ROUTINGS: dict[str, _Routing] = {
    # Untiled: the policy routes none of these to s8. The fixture's LSTM gates
    # are FullyConnected layers with per-channel weights, and its same-shape
    # Mul runs on the Mul adapters.
    "int8_conv": _calls(_CONV),
    "int8_depthwise_multiplier_untiled": _calls(_DEPTHWISE),
    "int8_matmul": _calls(_FULLY_CONNECTED),
    "lstm_unrolled": _calls(_FULLY_CONNECTED_PER_CHANNEL, _MUL),
    "int8_averagepool": _calls(_AVERAGE_POOL),
    "int8_channel_split": _calls(_MUL),
    "int8_elementwise_max_untiled": _calls(_MAX),
    "int8_elementwise_min_untiled": _calls(_MIN),
    # Height tiles: the operator decides as untiled, told that a tile is active.
    **dict.fromkeys(("int8_banded_mul", "int8_channel_split_bands"), _calls(_MUL)),
    **dict.fromkeys(("int8_banded_max", "int8_elementwise_max"), _calls(_MAX)),
    **dict.fromkeys(("int8_banded_min", "int8_elementwise_min"), _calls(_MIN)),
    "int8_depthwise_multiplier": _calls(_DEPTHWISE),
    # A height-tiled Conv chained into a 2D-tiled Depthwise.
    "int8_inferred_kernel_chain": _calls(_CONV, _DEPTHWISE),
    # 2D tiles and rolled chains: the Conv and Depthwise adapters take the
    # tile's width and the roll's offsets.
    **dict.fromkeys(("int8_2d_tiled_conv", "int8_cotiled_concat_2d", "int8_concat_axis2_bands",
                     "int8_concat_axis3_bands", "int8_add_broadcast_1x4x12x1_second_bands",
                     "int8_add_broadcast_1x4x12x12_constant_second_bands",
                     "int8_sub_broadcast_1x4x1x12_first_bands", "int8_linebuffer_conv_chain",
                     "int8_split_chain", "int8_hardswish", "qdq_conv_inner_float_section_tiled",
                     "qdq_conv_terminal_float_section_tiled", "int8_add_constant_channels_second",
                     "int8_sub_constant_scalar_first"), _calls(_CONV)),
    "int8_strided_depthwise_2d": _calls(_DEPTHWISE),
    # A global average pool walked by input rows or in channel groups: ESP-NN
    # sends GlobalAveragePool to s8 always, CMSIS-NN whenever it is tiled.
    **dict.fromkeys(("int8_tiled_global_reduction", "int8_grouped_global_reduction_0",
                     "int8_grouped_global_reduction_1"), _NO_VENDOR),
    # Row bands: no adapter takes a band of a matrix, so every row band runs on s8.
    "int8_row_band_matmul": _NO_VENDOR,
    # Tiled pools run on s8 on both dispatchers, and neither has a MaxPool adapter.
    **dict.fromkeys(("int8_banded_averagepool", "int8_banded_maxpool"), _NO_VENDOR),
    # ESP-NN encodes unit dilation only; CMSIS-NN runs a dilated Conv untiled only.
    "int8_banded_dilated_conv": _NO_VENDOR,
}


class _AccelRun(NamedTuple):
    outputs: bytes
    # The memory report with the allocation the reference runner makes, once
    # the vendor scratch above it is accounted for.
    memory: _MemoryReport
    calls: dict[str, int]
    prepare_calls: dict[str, int]


def _call_report(where: str, pattern: re.Pattern[str], stdout: str) -> dict[str, int]:
    """The `name=count` items of the one line `pattern` matches."""
    reports = pattern.findall(stdout)
    if len(reports) != 1:
        raise AssertionError(f"{where}: {len(reports)} lines match {pattern.pattern}, expected one")
    return {name: int(count) for name, count in (item.split("=") for item in reports[0].split())}


def _align16(value: int) -> int:
    return (value + 15) & ~15


class _AccelReplay:
    """Reruns reference runs of int8 plans on every accelerated dispatcher.

    Each rerun takes the same arguments and inputs, exits 0, touches no guard
    byte, reports the reference run's memory figures with its vendor scratch
    on top, returns the reference bytes, hands no operator its adapter takes
    to s8, and reports the vendor calls it made. A rerun of a plan, inputs
    and arguments already rerun is compared from the recorded result.

    Reruns execute on a pool of worker threads, on copies of the plan and
    inputs taken when the reference run is checked; their results are checked
    in the order of the reference runs, by the thread that calls `replay`,
    while later cases run, and `finish` checks the rest."""

    def __init__(self, work_dir: Path) -> None:
        self.work_dir = work_dir / "accel-replay"
        self.work_dir.mkdir()
        try:
            workers = len(os.sched_getaffinity(0))
        except AttributeError:
            workers = os.cpu_count() or 1
        self.pool = ThreadPoolExecutor(max_workers=workers)
        self.launched = 0
        self.done: dict[tuple, dict[str, Future[tuple[subprocess.CompletedProcess[str], bytes]]]] = {}
        self.checked: dict[tuple[tuple, str], _AccelRun] = {}
        self.pending: deque[tuple] = deque()
        self.refused: set[tuple] = set()
        self.runs: Counter[str] = Counter()
        self.cases: set[str] = set()
        # Per runner target, the wrapped vendor symbols its report lists.
        self.wrapped: dict[str, frozenset[str]] = {}
        # Per dispatcher, the cases whose runs called each vendor symbol,
        # while preparing or running, and the cases that ran a vendor kernel.
        self.callers: dict[str, dict[str, set[str]]] = {name: {} for name in _ACCEL_DISPATCHES}
        self.kernel_cases: dict[str, set[str]] = {name: set() for name in _ACCEL_DISPATCHES}

    @staticmethod
    def _key(build_dir: Path, plan_path: Path, inputs_path: Path, args: list[str]) -> tuple:
        return (build_dir, hashlib.sha256(plan_path.read_bytes()).digest(),
                hashlib.sha256(inputs_path.read_bytes()).digest(), tuple(args))

    def _snapshot(self, path: Path, digest: bytes) -> Path:
        """A copy of `path` named by its content, which later cases cannot overwrite."""
        copy = self.work_dir / f"{digest.hex()}{path.suffix}"
        if not copy.exists():
            copy.write_bytes(path.read_bytes())
        return copy

    @staticmethod
    def _launch(command: list[str], outputs: Path) -> tuple[subprocess.CompletedProcess[str], bytes]:
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        return completed, outputs.read_bytes() if outputs.exists() else b""

    def _vendor_calls(self, where: str, name: str, pattern: re.Pattern[str],
                      stdout: str) -> dict[str, int]:
        accel = _ACCEL_DISPATCHES[name]
        calls = _call_report(where, pattern, stdout)
        wrapped = self.wrapped.setdefault(accel.runner, frozenset(calls))
        prefixes = tuple({dispatch.prefix for dispatch in _ACCEL_DISPATCHES.values()})
        foreign = sorted(symbol for symbol in wrapped if not symbol.startswith(prefixes))
        unknown = sorted(set(_VENDOR_UNREACHABLE) - wrapped)
        if not wrapped or foreign or unknown:
            raise AssertionError(f"{where}: wrapped vendor symbols {sorted(wrapped)} name no "
                                 f"dispatcher {foreign} or miss allowlisted {unknown}")
        if set(calls) != wrapped:
            raise AssertionError(f"{where}: vendor call report lists {sorted(calls)}, "
                                 f"earlier runs {sorted(wrapped)}")
        stray = sorted(symbol for symbol, count in calls.items()
                       if count and not symbol.startswith(accel.prefix))
        if stray:
            raise AssertionError(f"{where}: called another dispatcher's vendor functions {stray}")
        return calls

    def _check(self, where: str, name: str, completed: subprocess.CompletedProcess[str],
               outputs: bytes, reference: _MemoryReport) -> _AccelRun:
        accel = _ACCEL_DISPATCHES[name]
        guards = [line for line in completed.stdout.splitlines()
                  if line.startswith("TIGRIS_CONTRACT_GUARD")]
        if guards:
            raise AssertionError(f"{where}: wrote past an arena: {'; '.join(guards)}")
        if completed.returncode != 0:
            raise AssertionError(f"{where}: run failed (exit {completed.returncode}):\n"
                                 f"{completed.stdout}{completed.stderr}")
        report = _memory_report(where, completed.stdout)
        dispatch = _DISPATCH_REPORT.search(completed.stdout)
        if dispatch is None or dispatch.group(1) != accel.dispatch:
            raise AssertionError(f"{where}: runner reports dispatch "
                                 f"{dispatch.group(1) if dispatch else None}")
        # The runner allocates the vendor scratch it reports above the
        # 16-aligned reference arena, as tigris_cmsis_nn_fast_arena_required does.
        scratch = int(dispatch.group(2))
        allocated = _align16(reference.allocated) + scratch if scratch else reference.allocated
        if report.allocated != allocated:
            raise AssertionError(
                f"{where}: allocated arena {report.allocated} != the reference run's "
                f"{reference.allocated} + vendor scratch {scratch} ({allocated})")
        # The scratch is the largest buffer a vendor call was given plus the
        # per-channel quantization a call read from the workspace, each
        # rounded to 16 bytes as the workspace partitions them. The executor
        # tiles a stage only when the arena cannot hold it whole, so an
        # operator the policy sent to s8 for its tile keeps the scratch its
        # whole-tensor route takes; such a run bounds the scratch from below.
        measured = _SCRATCH_REPORT.findall(completed.stdout)
        if len(measured) != 1:
            raise AssertionError(f"{where}: {len(measured)} scratch reports, expected one")
        requested, channels, tile_routed = (int(value) for value in measured[0])
        expected = _align16(requested) + _align16(channels * 8)
        if scratch < expected or (scratch != expected and not tile_routed):
            raise AssertionError(
                f"{where}: vendor scratch {scratch} != the largest buffer a call was given "
                f"{requested} + quantization for {channels} channels ({expected})")
        fallbacks = _call_report(where, _FALLBACK_REPORT, completed.stdout)
        declined = {op: count for op, count in fallbacks.items() if count}
        if not fallbacks or declined:
            raise AssertionError(f"{where}: the adapter handed operators it takes to s8 "
                                 f"{declined or 'or reported no operator types'}")
        calls = self._vendor_calls(where, name, _VENDOR_REPORT, completed.stdout)
        prepare_calls = self._vendor_calls(f"{where} preparing", name, _VENDOR_PREPARE_REPORT,
                                           completed.stdout)
        return _AccelRun(outputs, report._replace(allocated=reference.allocated),
                         calls, prepare_calls)

    def _check_routing(self, where: str, name: str, calls: dict[str, int],
                       routing: _Routing | ContractCase) -> None:
        wrapped = self.wrapped[_ACCEL_DISPATCHES[name].runner]
        unknown = sorted(set(routing.expect_vendor) - set(_ACCEL_DISPATCHES)
                         | set(routing.expect_no_vendor) - set(_ACCEL_DISPATCHES))
        misnamed = sorted(symbol for dispatch in set(routing.expect_vendor) & {name}
                          for symbol in routing.expect_vendor[dispatch]
                          if symbol not in wrapped
                          or not symbol.startswith(_ACCEL_DISPATCHES[dispatch].prefix))
        if unknown or misnamed:
            raise AssertionError(f"{where}: routing names unknown dispatchers {unknown} "
                                 f"or symbols {misnamed}")
        called = sorted(symbol for symbol, count in calls.items() if count)
        missing = sorted(symbol for symbol in routing.expect_vendor.get(name, ())
                         if not calls[symbol])
        if missing:
            raise AssertionError(f"{where}: expected vendor calls {missing} were not made; "
                                 f"called {called}")
        if name in routing.expect_no_vendor and called:
            raise AssertionError(f"{where}: expected no vendor call, called {called}")

    def replay(self, build_dir: Path, case_name: str, plan: dict, plan_path: Path,
               inputs_path: Path, outputs_path: Path, args: list[str], where: str,
               reference_stdout: str, routing: _Routing | ContractCase | None,
               full_budget: bool | None) -> None:
        """Check the reference run that wrote `outputs_path` on every
        accelerated dispatcher of the runners in `build_dir`, which have the
        reference runner's tensor alignment."""
        expected = outputs_path.read_bytes()
        reference = _memory_report(where, reference_stdout)
        key = self._key(build_dir, plan_path, inputs_path, args)
        runs = self.done.setdefault(key, {})
        for name, accel in _ACCEL_DISPATCHES.items():
            if name not in runs:
                self.launched += 1
                outputs = self.work_dir / f"{self.launched}-{name}.bin"
                command = [str(build_dir / accel.runner), str(self._snapshot(plan_path, key[1])),
                           str(self._snapshot(inputs_path, key[2])), str(outputs), *args,
                           f"--dispatch={accel.dispatch}"]
                runs[name] = self.pool.submit(self._launch, command, outputs)
            self.pending.append((key, name, f"{where} on {name}", case_name, plan, expected,
                                 reference, routing, full_budget))
        self._drain(wait=False)

    def _drain(self, *, wait: bool) -> None:
        """Check finished reruns in the order they were requested; `wait`
        checks them all."""
        while self.pending:
            key, name, label, case_name, plan, expected, reference, routing, full_budget = (
                self.pending[0])
            future = self.done[key][name]
            if not wait and not future.done():
                return
            self.pending.popleft()
            run = self.checked.get((key, name))
            if run is None:
                completed, outputs = future.result()
                run = self._check(label, name, completed, outputs, reference)
                self.checked[(key, name)] = run
                self.runs[name] += 1
            if run.memory != reference:
                differing = ", ".join(
                    f"{field} {mine} vs {theirs}" for field, mine, theirs in
                    zip(_MemoryReport._fields, run.memory, reference) if mine != theirs)
                raise AssertionError(f"{label}: memory figures differ from the reference "
                                     f"run: {differing}")
            if full_budget is not None:
                _assert_memory_contract(label, plan, run.memory, full_budget=full_budget)
            if run.outputs != expected:
                raise AssertionError(f"{label}: outputs differ from the s8 run: "
                                     f"{_first_difference(plan, expected, run.outputs)}")
            if routing is not None:
                self._check_routing(label, name, run.calls, routing)
            for calls in (run.calls, run.prepare_calls):
                for symbol, count in calls.items():
                    if count:
                        self.callers[name].setdefault(symbol, set()).add(case_name)
            if any(run.calls.values()):
                self.kernel_cases[name].add(case_name)
            self.cases.add(case_name)

    def close(self) -> None:
        """Stop the reruns not yet started."""
        self.pool.shutdown(wait=True, cancel_futures=True)

    def refuse(self, build_dir: Path, case_name: str, plan_path: Path, inputs_path: Path,
               outputs_path: Path, limit: int) -> None:
        """The run the reference runner refused below the scheduled peak is
        refused before any stage on every accelerated dispatcher."""
        key = self._key(build_dir, plan_path, inputs_path, [str(limit)])
        if key in self.refused:
            return
        for name, accel in _ACCEL_DISPATCHES.items():
            completed = subprocess.run(
                [str(build_dir / accel.runner), str(plan_path), str(inputs_path),
                 str(outputs_path), str(limit), f"--dispatch={accel.dispatch}"], text=True,
                capture_output=True, check=False)
            # With vendor scratch, a CMSIS-NN arena whose 16-aligned core falls
            # short is refused at preparation, which precedes the first stage.
            refused = _REFUSED_BELOW_PEAK in completed.stderr or (
                accel.dispatch == "cmsis-nn" and "CMSIS-NN preparation failed" in completed.stderr)
            if completed.returncode == 0 or not refused or "TIGRIS_CONTRACT_" in completed.stdout:
                raise AssertionError(
                    f"{case_name} on {name}: a fast arena one byte below the scheduled peak "
                    f"was not refused up front (exit {completed.returncode}):\n"
                    f"{completed.stdout}{completed.stderr}")
        self.refused.add(key)

    def finish(self) -> None:
        """Every wrapped vendor symbol was called by some case on its
        dispatcher, unless it is listed as unreachable."""
        self._drain(wait=True)
        if not self.runs or not self.wrapped:
            raise AssertionError("no int8 plan ran on the accelerated dispatchers")
        print(f"PASS accelerated_dispatch_parity cases={len(self.cases)} " + " ".join(
            f"{name}={self.runs[name]}" for name in _ACCEL_DISPATCHES))
        symbols, cases = [], []
        for name, accel in _ACCEL_DISPATCHES.items():
            wrapped = sorted(symbol for symbol in self.wrapped[accel.runner]
                             if symbol.startswith(accel.prefix))
            callers = self.callers[name]
            missing = [symbol for symbol in wrapped
                       if symbol not in callers and symbol not in _VENDOR_UNREACHABLE]
            stale = {symbol: sorted(callers[symbol]) for symbol in _VENDOR_UNREACHABLE
                     if symbol in callers}
            if missing or stale:
                raise AssertionError(
                    f"{name}: no case calls {missing}; listed as unreachable but called: {stale}")
            symbols.append(f"{name}={len(callers)}/{len(wrapped)}")
            cases.append(f"{name}={len(self.kernel_cases[name])}")
        print(f"PASS accelerated_dispatch_routing {' '.join(symbols)} symbols "
              f"(unreachable: {', '.join(sorted(_VENDOR_UNREACHABLE))}), "
              f"cases reaching a vendor function while running {' '.join(cases)}, "
              f"no adapter fallback")


class _ContractRunner(NamedTuple):
    """A contract runner on the reference dispatchers. Every run of an int8
    plan through it is replayed on the accelerated dispatchers of the runners
    in the build directory `accel`, built with the same tensor alignment."""

    path: Path
    accel: Path
    replay: _AccelReplay

    def run(self, case_name: str, plan: dict, plan_path: Path, inputs_path: Path,
            outputs_path: Path, args: list[str], label: str, *,
            routing: _Routing | ContractCase | None = None,
            full_budget: bool | None = None) -> subprocess.CompletedProcess[str]:
        """Run the plan; `full_budget` set checks the memory contract of every run.

        `routing` holds the vendor calls of the case's own plan; a plan the
        gate derives from it passes none."""
        where = f"{case_name} {label}"
        completed = _run([str(self.path), str(plan_path), str(inputs_path), str(outputs_path),
                          *args], where)
        if full_budget is not None:
            _assert_memory_contract(case_name, plan, completed.stdout, full_budget=full_budget)
        if _is_int8_plan(plan):
            self.replay.replay(self.accel, case_name, plan, plan_path, inputs_path, outputs_path, args,
                               where, completed.stdout, routing, full_budget)
        return completed

    def refuse_below_peak(self, case_name: str, plan: dict, plan_path: Path,
                          inputs_path: Path, outputs_path: Path) -> None:
        """One byte below the plan's scheduled peak, the run is refused before any stage."""
        limit = plan["scheduled_peak"] - 1
        completed = subprocess.run(
            [str(self.path), str(plan_path), str(inputs_path), str(outputs_path), str(limit)],
            text=True, capture_output=True, check=False)
        if completed.returncode == 0 or _REFUSED_BELOW_PEAK not in completed.stderr:
            raise AssertionError(
                f"{case_name}: a fast arena one byte below the scheduled peak was not refused "
                f"up front (exit {completed.returncode}):\n{completed.stdout}{completed.stderr}")
        if _is_int8_plan(plan):
            self.replay.refuse(self.accel, case_name, plan_path, inputs_path, outputs_path, limit)


_ROWS_REPORT = re.compile(
    r"^TIGRIS_CONTRACT_ROWS kernel_rows=(\d+) chain_tiles=(\d+) "
    r"chain_tile_h=(\d+) interior_clamps=(\d+)$",
    re.MULTILINE,
)


def _parse_rows(case: ContractCase, runtime_stdout: str) -> dict[str, int]:
    match = _ROWS_REPORT.search(runtime_stdout)
    if match is None:
        raise AssertionError(f"{case.name}: rows runner emitted no row report")
    keys = ("kernel_rows", "chain_tiles", "chain_tile_h", "interior_clamps")
    return {key: int(value) for key, value in zip(keys, match.groups())}


def _prepare_case(case: ContractCase, work_dir: Path) -> tuple[Path, dict, list[Array]]:
    """Save the case's models, compile and check its plan, and pack its inputs.

    Returns the case directory, the plan and the ONNX Runtime outputs, or the
    case's recorded outputs.
    """
    case_dir = work_dir / case.name
    case_dir.mkdir()
    onnx.save(case.compile_model, case_dir / "compile.onnx")
    onnx.save(case.reference_model, case_dir / "reference.onnx")
    plan = _compile_plan(
        case_dir / "compile.onnx",
        case_dir / "model.tgrs",
        mem_budget=case.mem_budget,
        slow_budget=case.slow_budget,
        compression=case.compression,
        xip=case.xip,
        force_one_op_stages=case.force_one_op_stages,
        int8_io=case.int8_io,
    )
    _assert_plan_mode(case, plan)
    reference_outputs = list(case.recorded_outputs) or _reference_session(
        case, case_dir / "reference.onnx").run(None, case.inputs)
    (case_dir / "inputs.bin").write_bytes(_pack_inputs(plan, case.inputs))
    return case_dir, plan, reference_outputs


def _run_metric_case(
    case: ContractCase, rows_runner: _ContractRunner, work_dir: Path
) -> None:
    """Prove the line-buffered roll matches ORT and computes fewer kernel rows.

    Compiles the (asserted line-buffered) plan once, then executes it through
    the rows-instrumented runner twice: normally (roll on) and with
    ``--no-linebuffer`` (the recompute path). Both runs must match ONNX Runtime,
    and the roll must reduce total kernel output-rows, approaching the unique
    output-row count. It prints the ratio of recompute to line-buffered kernel
    rows.
    """
    case_dir, plan, reference_outputs = _prepare_case(case, work_dir)
    plan_path, inputs_path = case_dir / "model.tgrs", case_dir / "inputs.bin"
    limit = str(plan["_compiler_scheduled_peak"])

    def _execute(extra_args: list[str], label: str) -> dict[str, int]:
        outputs_path = case_dir / f"outputs_{label}.bin"
        completed = rows_runner.run(
            case.name, plan, plan_path, inputs_path, outputs_path, [limit, *extra_args],
            f"rows-runner ({label})", routing=case,
            full_budget=False if label == "linebuffered" else None)
        actual_outputs = _decode_outputs(
            plan, outputs_path.read_bytes(), reference_outputs
        )
        _assert_output_parity(
            actual_outputs, reference_outputs, _output_scales(plan), case.exact
        )
        return _parse_rows(case, completed.stdout)

    rows_on = _execute([], "linebuffered")
    rows_off = _execute(["--no-linebuffer"], "recompute")

    # Model output height (ONNX NCHW reference); the runtime tiles the height.
    out_h = int(reference_outputs[0].shape[2])
    tiles = rows_on["chain_tiles"]
    tile_h = rows_on["chain_tile_h"]
    if tiles < 3:
        raise AssertionError(
            f"{case.name}: chain produced {tiles} tiles, need >= 3 so tile 0, "
            f"an interior tile, and a last tile all run"
        )
    if tile_h <= 0 or out_h % tile_h == 0:
        raise AssertionError(
            f"{case.name}: output height {out_h} is not partial against tile "
            f"height {tile_h}; the last tile must be partial"
        )
    if rows_off["chain_tiles"] != tiles or rows_off["chain_tile_h"] != tile_h:
        raise AssertionError(
            f"{case.name}: recompute run tiled differently "
            f"({rows_off['chain_tiles']}x{rows_off['chain_tile_h']}) than the "
            f"line-buffered run ({tiles}x{tile_h})"
        )
    if not (rows_on["kernel_rows"] > 0 and rows_off["kernel_rows"] > 0):
        raise AssertionError(f"{case.name}: row counter recorded no work")
    if rows_on["kernel_rows"] >= rows_off["kernel_rows"]:
        raise AssertionError(
            f"{case.name}: line-buffered kernel rows "
            f"{rows_on['kernel_rows']} did not drop below recompute "
            f"{rows_off['kernel_rows']}"
        )

    unique_rows = len(case.expected_operators) * out_h
    ratio = rows_off["kernel_rows"] / rows_on["kernel_rows"]
    print(
        f"PASS {case.name} recompute-reduction "
        f"rows_on={rows_on['kernel_rows']} rows_off={rows_off['kernel_rows']} "
        f"ratio={ratio:.3f}x unique_rows={unique_rows} "
        f"tiles={tiles} tile_h={tile_h} last_tile_h={out_h - (tiles - 1) * tile_h}"
    )


def _reference_session(case: ContractCase, path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    if case.reference_unoptimized:
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def _section_reference_outputs(
    case: ContractCase, runner: _ContractRunner, case_dir: Path, plan: dict,
    primary: bytes, reference_outputs: list[Array],
) -> list[Array]:
    """Check section entry codes and evaluate float outputs on those same codes."""
    graph = plan["_graph"]
    section_inputs = {op.outputs[0] for op in graph.ops if op.op_type == "Dequantize"
                      and graph.tensors[op.inputs[0]].dtype == TensorProto.INT8}
    entries = list(dict.fromkeys(
        node.input[0] for node in case.compile_model.graph.node
        if node.op_type == "DequantizeLinear" and node.output[0] in section_inputs))
    if not entries:
        return reference_outputs

    variant = onnx.shape_inference.infer_shapes(case.compile_model)
    infos = {item.name: item for item in
             (*variant.graph.input, *variant.graph.value_info, *variant.graph.output)}
    original_names = [item.name for item in variant.graph.output]
    added = [name for name in entries if name not in original_names]
    variant.graph.output.extend(infos[name] for name in added)
    variant_path, variant_plan_path = case_dir / "entries.onnx", case_dir / "entries.tgrs"
    onnx.save(variant, variant_path)
    variant_plan = _compile_plan(
        variant_path, variant_plan_path, mem_budget=case.mem_budget,
        slow_budget=case.slow_budget, compression=case.compression, xip=case.xip,
        force_one_op_stages=case.force_one_op_stages, int8_io=case.int8_io,
    )
    oracle = onnx.shape_inference.infer_shapes(case.reference_model)
    oracle_infos = {item.name: item for item in
                    (*oracle.graph.input, *oracle.graph.value_info, *oracle.graph.output)}
    oracle.graph.output.extend(oracle_infos[name] for name in added)
    oracle_path = case_dir / "reference-entries.onnx"
    onnx.save(oracle, oracle_path)
    expected = _reference_session(case, oracle_path).run(None, case.inputs)
    variant_inputs, variant_outputs = case_dir / "entries-inputs.bin", case_dir / "entries-outputs.bin"
    variant_inputs.write_bytes(_pack_inputs(variant_plan, case.inputs))
    runner.run(case.name, variant_plan, variant_plan_path, variant_inputs, variant_outputs, [],
               "section entry execution")
    raw = variant_outputs.read_bytes()
    assert raw[:len(primary)] == primary, f"{case.name}: adding entry outputs changed original outputs"
    actual = _decode_outputs(variant_plan, raw, expected)
    names = original_names + added
    actual_by_name = dict(zip(names, actual))
    expected_by_name = dict(zip(names, expected))
    _assert_output_parity([actual_by_name[name] for name in entries],
                          [expected_by_name[name] for name in entries])

    outputs = list(reference_outputs)
    extractor = onnx.utils.Extractor(onnx.shape_inference.infer_shapes(case.reference_model))
    producers = {name: node for node in case.reference_model.graph.node for name in node.output}
    model_inputs = {item.name for item in case.reference_model.graph.input}
    for index, tensor_index in enumerate(plan["model_outputs"]):
        if plan["tensors"][tensor_index]["dtype"] != TensorProto.FLOAT:
            continue
        name = original_names[index]
        pending, visited, boundaries = [name], set(), set()
        while pending:
            current = pending.pop()
            if current in visited:
                continue
            visited.add(current)
            if current in entries or current in model_inputs:
                boundaries.add(current)
            elif current in producers:
                pending.extend(producers[current].input)
        if not boundaries.intersection(entries):
            continue
        inputs = sorted(boundaries)
        section = extractor.extract_model(inputs, [name])
        section_path = case_dir / f"section-{index}.onnx"
        onnx.save(section, section_path)
        section_session = _reference_session(case, section_path)
        feeds = {item.name: (actual_by_name[item.name] if item.name in entries else case.inputs[item.name])
                 for item in section_session.get_inputs()}
        outputs[index] = section_session.run(None, feeds)[0]
    return outputs


def _run_case(
    case: ContractCase, runner: _ContractRunner, work_dir: Path, roomy_runner: _ContractRunner
) -> Path:
    case_dir, plan, reference_outputs = _prepare_case(case, work_dir)
    compile_path, plan_path = case_dir / "compile.onnx", case_dir / "model.tgrs"
    inputs_path, outputs_path = case_dir / "inputs.bin", case_dir / "outputs.bin"
    run_args = [
        str(plan["_compiler_scheduled_peak"]),
        # Run with exactly the workspace generated code would reserve.
        "--workspace-limits=" + ",".join(
            str(value) for value in executor_workspace_limits(plan)),
    ]
    accelerated = _is_int8_plan(plan)
    if (case.expect_vendor or case.expect_no_vendor) and not accelerated:
        raise AssertionError(f"{case.name}: states vendor calls for a plan that is not int8")
    completed = runner.run(case.name, plan, plan_path, inputs_path, outputs_path, run_args,
                           "runtime execution", routing=case, full_budget=False)
    without_slow = _assert_runs_without_slow(
        case.name, plan, runner, plan_path, inputs_path, outputs_path,
        plan["_compiler_scheduled_peak"], case)
    if case.expect_inputs_in_fast and not without_slow:
        raise AssertionError(f"{case.name}: the plan does not read its inputs in place")
    if case.exact_slow:
        _assert_runs_in_slow_figure(case, plan, runner, plan_path, inputs_path,
                                    outputs_path, completed.stdout)
    actual_outputs = _decode_outputs(
        plan, outputs_path.read_bytes(), reference_outputs
    )
    reference_outputs = _section_reference_outputs(
        case, runner, case_dir, plan, outputs_path.read_bytes(), reference_outputs)
    _assert_output_parity(
        actual_outputs, reference_outputs, _output_scales(plan), case.exact
    )
    # Extra caller capacity must not change the scheduled memory bound.
    for full_runner in (runner, roomy_runner):
        full_runner.run(case.name, plan, plan_path, inputs_path, outputs_path, [],
                        "full-budget execution", routing=case, full_budget=True)
        _assert_output_parity(
            _decode_outputs(plan, outputs_path.read_bytes(), reference_outputs),
            reference_outputs, _output_scales(plan),
        )
    _assert_trace(case.name, plan_path, inputs_path.read_bytes(), outputs_path.read_bytes())
    if case.exact_untiled:
        assert plan["version"] == SCHEMA_VERSION
        banded_types = {OP_TYPE_MAP[k] for k in (case.tiled_operators or case.expected_operators) if k != "Transpose"}
        banded_stages = [stage for stage in plan["stages"] if any(
            plan["ops"][i]["op_type"] in banded_types for i in stage["ops"])]
        assert banded_stages and all(stage["tile_plan_idx"] != 65535 and
            plan["tile_plans"][stage["tile_plan_idx"]]["num_tiles"] > 1 for stage in banded_stages)
        untiled_path = case_dir / "untiled.tgrs"
        untiled = _compile_plan(compile_path, untiled_path, mem_budget="64K", slow_budget="16K",
                                compression=None, xip=False, int8_io=case.int8_io)
        assert not any(tile["tileable"] for tile in untiled["tile_plans"])
        assert untiled["version"] == SCHEMA_VERSION
        untiled_outputs = case_dir / "untiled.bin"
        runner.run(case.name, untiled, untiled_path, inputs_path, untiled_outputs,
                   [str(untiled["_compiler_scheduled_peak"])], "untiled execution")
        assert outputs_path.read_bytes() == untiled_outputs.read_bytes()
    if case.exact_uncompressed:
        plain_path = case_dir / "uncompressed.tgrs"
        plain = _compile_plan(compile_path, plain_path, mem_budget=case.mem_budget,
                              slow_budget=case.slow_budget, compression=None, xip=case.xip,
                              force_one_op_stages=case.force_one_op_stages, int8_io=case.int8_io)
        plain_outputs = case_dir / "uncompressed.bin"
        runner.run(case.name, plain, plain_path, inputs_path, plain_outputs,
                   [str(plain["_compiler_scheduled_peak"])], "uncompressed execution")
        if outputs_path.read_bytes() != plain_outputs.read_bytes():
            raise AssertionError(f"{case.name}: outputs differ from the uncompressed plan")
    print(f"PASS {case.name} memory-contract" + (" no-slow" if without_slow else "")
          + (" accelerated" if accelerated else ""))
    return plan_path


def _build_host_library(runtime: Path, build_dir: Path) -> Path:
    """Build the host library, which compiles the runtime with execution tracing."""
    _run(["cmake", "-S", str(runtime), "-B", str(build_dir), "-DCMAKE_BUILD_TYPE=Release",
          "-DTIGRIS_BUILD_HOST=ON"], "host library configure")
    _run(["cmake", "--build", str(build_dir), "--target", "tigris_host"],
         "host library build", _build_environment())
    for name in ("libtigris_host.so", "libtigris_host.dylib", "tigris_host.dll"):
        matches = sorted(build_dir.rglob(name))
        if matches:
            return matches[0]
    raise AssertionError(f"host library not found in {build_dir}")


def _assert_trace(case_name: str, plan_path: Path, inputs: bytes, outputs: bytes) -> None:
    """Run the plan once more with tracing on. The traced run produces the same
    bytes, its events add up to the runtime's counters and peaks, and the tiles
    of every spatially tiled stage spill each output cell at least once."""
    with Session(plan_path) as session:
        arrays, offset = {}, 0
        for item in session.inputs:
            count = item["size_bytes"] // item["dtype"].itemsize
            arrays[item["name"]] = np.frombuffer(
                inputs, item["dtype"], count, offset).reshape(item["shape"])
            offset += item["size_bytes"]
        results, events, counters = session.trace(arrays)
        memory = session.memory
        produced = b"".join(results[item["name"]].tobytes() for item in session.outputs)
    if produced != outputs:
        raise AssertionError(f"{case_name}: traced run produced different outputs")
    problems = check_trace(events, counters)
    if problems:
        raise AssertionError(f"{case_name}: " + "; ".join(problems))
    for pool in ("fast", "slow"):
        peak = max((event[f"{pool}_used"] for event in events), default=0)
        if peak != memory[f"{pool}_peak_bytes"]:
            raise AssertionError(f"{case_name}: {pool} events peak at {peak}, the runtime "
                                 f"counted {memory[f'{pool}_peak_bytes']}")
    plan = read_binary_plan(plan_path.read_bytes(), decompress_weights=False)
    for unit in trace_units(plan, events):
        if unit["path"] not in {"tiled", "tiled_2d", "chain"}:
            continue
        spills: dict[int, list[dict]] = {}
        for event in unit["events"]:
            if event["kind"] == "spill" and event["row0"] is not None:
                spills.setdefault(event["tensor"], []).append(event)
        for tensor, tiles in spills.items():
            shape = plan["tensors"][tensor]["shape"]
            written = np.zeros((shape[1], shape[2] if len(shape) == 4 else 1), np.int32)
            for event in tiles:
                columns = slice(event["col0"], event["col1"]) if event["col0"] is not None else slice(None)
                written[event["row0"]:event["row1"], columns] += 1
            name = plan["tensors"][tensor]["name"]
            if (written == 0).any():
                raise AssertionError(f"{case_name}: stage {unit['stage']} never spills "
                                     f"{int((written == 0).sum())} cells of {name}")
            if (written > 1).any():
                print(f"NOTE {case_name}: stage {unit['stage']} spills {int((written > 1).sum())} "
                      f"of {written.size} cells of {name} more than once")


def _output_scales(plan: dict) -> list[float]:
    """Quantization step of each model output, or 0 where the plan is not quantized.

    A float interface over a quantized plan is compared after dequantization.
    Both sides are then multiples of this step, so the comparison stays in
    whole steps rather than in floats that cannot represent the step exactly.
    """
    scales: list[float] = []
    for tensor_index in plan["model_outputs"]:
        tensor = plan["tensors"][tensor_index]
        if _declared_dtype(tensor) == tensor["dtype"] or tensor["dtype"] == TensorProto.INT32:
            scales.append(0.0)
            continue
        quant = plan["quant_params"][tensor["quant_param_idx"]]
        scales.append(float(quant["scale"]))
    return scales


def _assert_output_parity(
    actual_outputs: list[Array],
    reference_outputs: list[Array],
    output_scales: list[float] | None = None,
    exact: bool = False,
) -> None:
    output_scales = output_scales or [0.0] * len(actual_outputs)
    for index, (actual, expected, scale) in enumerate(zip(
        actual_outputs, reference_outputs, output_scales
    )):
        # Bool and index outputs carry no quantization step to round.
        if exact or expected.dtype.kind == "b" or expected.dtype in (np.int32, np.int64):
            if (actual.dtype != expected.dtype or actual.shape != expected.shape
                    or actual.tobytes() != expected.tobytes()):
                mismatched = int(np.count_nonzero(actual != expected)) if actual.shape == expected.shape else -1
                raise AssertionError(
                    f"output {index} is not exact: {actual.dtype}{list(actual.shape)} vs "
                    f"{expected.dtype}{list(expected.shape)}, {mismatched} elements differ"
                )
        elif scale > 0.0:
            # The runtime requantizes inside its kernels with fixed-point
            # multipliers, which round ties away from zero or up, while this
            # ONNX Runtime QDQ reference rounds in float with ties to even.
            # Keep the same one-LSB acceptance bound used by benchmark
            # validation, while rejecting any larger contract drift.
            steps = np.abs(np.rint((actual - expected) / scale))
            worst = float(steps.max()) if steps.size else 0.0
            if worst > _INT8_LSB_TOLERANCE:
                raise AssertionError(
                    f"dequantized output differs by {worst:.0f} quantization "
                    f"steps, bound is {_INT8_LSB_TOLERANCE}"
                )
        elif np.issubdtype(expected.dtype, np.floating):
            np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
        else:
            # Widen first: unsigned bytes would wrap in the subtraction.
            np.testing.assert_allclose(
                actual.astype(np.int64), expected.astype(np.int64),
                rtol=0, atol=_INT8_LSB_TOLERANCE,
            )


def _assert_plan_mode(case: ContractCase, plan: dict) -> None:
    """Ensure each corpus case actually exercises the intended plan mode."""
    actual_operators = tuple(
        OP_TYPE_BY_CODE[op["op_type"]] for op in plan["ops"]
    )
    if Counter(actual_operators) != Counter(case.expected_operators):
        raise AssertionError(
            f"{case.name}: plan operators {actual_operators}, expected "
            f"{case.expected_operators}"
        )

    grouped = any(tile.get("input_channel_group", 0) > 0 for tile in plan["tile_plans"])
    assert grouped == case.expect_grouped, case.name

    standalone_tiled = any(
        tile["tileable"] and tile["num_tiles"] > 1
        for tile in plan["tile_plans"]
    )
    chained = any(stage["chain_len"] >= 2 for stage in plan["stages"])
    # Streamable chains encode their tile height on the chain head rather than
    # creating standalone tile-plan records.
    tiled = standalone_tiled or chained
    if tiled != case.expect_tiled:
        raise AssertionError(f"{case.name}: tiled={tiled}, expected {case.expect_tiled}")
    if chained != case.expect_chain:
        raise AssertionError(f"{case.name}: chained={chained}, expected {case.expect_chain}")

    # Read the head-stage line-buffer flag from the emitted plan (a recomputing
    # chain marks only its head, where chain_id equals the stage's own index).
    # This makes "is actually line-buffered" a checked precondition of the
    # differential rather than an assumption.
    line_buffered = any(
        stage["flags"] & STAGE_FLAG_LINE_BUFFERED
        for index, stage in enumerate(plan["stages"])
        if stage["chain_len"] >= 2 and stage["chain_id"] == index
    )
    if line_buffered != case.expect_line_buffered:
        raise AssertionError(
            f"{case.name}: line_buffered={line_buffered}, expected "
            f"{case.expect_line_buffered}"
        )

    # Both axes of a 2D case split, by the stated counts: num_tiles // ceil(H / tile_h) columns.
    if case.expect_2d:
        tile_plan = SimpleNamespace(**next(tile for tile in plan["tile_plans"]
                                          if tile["tileable"]))
        if tile_plan.axis != TILE_AXIS_HW:
            raise AssertionError(
                f"{case.name}: tile plan axis {tile_plan.axis}, expected "
                f"TILE_AXIS_HW ({TILE_AXIS_HW})"
            )
        tiles_h = math.ceil(tile_plan.original_extent / tile_plan.tile_extent)
        tiles_w = tile_plan.num_tiles // tiles_h
        if not (tiles_h > 1 and tiles_w > 1):
            raise AssertionError(
                f"{case.name}: expected multi-tile on both axes, got "
                f"tiles_h={tiles_h} tiles_w={tiles_w} "
                f"(num_tiles={tile_plan.num_tiles}, "
                f"tile_h={tile_plan.tile_extent}, tile_w={tile_plan.output_width_extent})"
            )

    if case.compression == "lz4":
        if plan["weight_blocks_compression"] != COMPRESS_LZ4:
            raise AssertionError(f"{case.name}: expected LZ4 weight blocks")
    elif plan["weight_blocks"]:
        raise AssertionError(f"{case.name}: unexpected compressed weight blocks")
    has_xip = bool(plan["flags"] & FLAG_XIP)
    if has_xip != case.xip:
        raise AssertionError(f"{case.name}: XIP flag={has_xip}, expected {case.xip}")


def _assert_compile_rejected(work_dir: Path) -> None:
    def value(name: str, shape: list[int]) -> onnx.ValueInfoProto:
        return helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)

    resize_scales = numpy_helper.from_array(
        np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32), "scales"
    )
    cases = [
        (_model("unsupported_acos", [helper.make_node("Acos", ["input"], ["output"])],
                [value("input", [1])], [value("output", [1])]),
         "Acos_0 (Acos)"),
        (_model("unsupported_softmax_axis",
                [helper.make_node("Softmax", ["input"], ["output"], axis=0)],
                [value("input", [2, 4])], [value("output", [2, 4])]),
         "Softmax axis must map to the runtime's final dimension"),
        (_model("unsupported_concat_axis",
                [helper.make_node("Concat", ["left", "right"], ["output"], axis=0)],
                [value("left", [1, 2, 3, 4]), value("right", [1, 2, 3, 4])],
                [value("output", [2, 2, 3, 4])]),
         "runtime does not concatenate on the batch axis"),
        # Nearest and bilinear both execute. Cubic resampling reads a
        # four-by-four neighborhood and has no kernel, so it is refused.
        # The nearest convention is valid, so the mode is the only cause.
        (_model("unsupported_resize_mode",
                [helper.make_node("Resize", ["input", "", "scales"], ["output"], mode="cubic",
                                  coordinate_transformation_mode="asymmetric", nearest_mode="floor")],
                [value("input", [1, 1, 2, 2])], [value("output", [1, 1, 4, 4])], [resize_scales]),
         "Resize mode=cubic is not encoded"),
        (helper.make_model(
            # The body reads w from the main graph instead of taking it as a loop variable.
            helper.make_graph(
                [
                    helper.make_node("Relu", ["input"], ["w"]),
                    helper.make_node(
                        "While",
                        ["zero", "input"],
                        ["count", "output"],
                        domain="tigris",
                        cond_branch=helper.make_graph(
                            [helper.make_node("Less", ["c_i", "c_three"], ["c_go"])], "condition",
                            [helper.make_tensor_value_info("c_i", TensorProto.INT32, []),
                             helper.make_tensor_value_info("c_v", TensorProto.FLOAT, [1, 4])],
                            [helper.make_tensor_value_info("c_go", TensorProto.BOOL, [])],
                            [numpy_helper.from_array(np.asarray(3, np.int32), "c_three")],
                        ),
                        body_branch=helper.make_graph(
                            [helper.make_node("Add", ["b_i", "b_one"], ["b_next"]),
                             helper.make_node("Tanh", ["w"], ["b_y"])], "body",
                            [helper.make_tensor_value_info("b_i", TensorProto.INT32, []),
                             helper.make_tensor_value_info("b_v", TensorProto.FLOAT, [1, 4])],
                            [helper.make_tensor_value_info("b_next", TensorProto.INT32, []),
                             helper.make_tensor_value_info("b_y", TensorProto.FLOAT, [1, 4])],
                            [numpy_helper.from_array(np.asarray(1, np.int32), "b_one")],
                        ),
                    ),
                ],
                "unsupported_outer_read",
                [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])],
                [helper.make_tensor_value_info("count", TensorProto.INT32, []),
                 helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])],
                [numpy_helper.from_array(np.asarray(0, np.int32), "zero")],
            ),
            opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("tigris", 1)],
            ir_version=9,
        ), "reads w from an enclosing graph in its body_branch"),
        # Rounding up adds a window that starts past the input, which no
        # end padding states.
        (_model("unsupported_pool_ceil",
                [helper.make_node("MaxPool", ["input"], ["output"], ceil_mode=1,
                                  kernel_shape=[1, 1], strides=[3, 3])],
                [value("input", [1, 1, 5, 5])], [value("output", [1, 1, 3, 3])]),
         "ceil_mode=1 is not encoded"),
    ]
    for index, (model, reason) in enumerate(cases):
        path = work_dir / f"rejected-{index}.onnx"
        onnx.save(model, path)
        try:
            _compile_plan(
                path,
                work_dir / f"rejected-{index}.tgrs",
                mem_budget="4K",
                slow_budget=None,
                compression=None,
                xip=False,
            )
        except ValueError as exc:
            if reason not in str(exc):
                raise AssertionError(
                    f"{model.graph.name} was refused for another reason: {exc}"
                ) from exc
            continue
        raise AssertionError(f"{model.graph.name} unexpectedly compiled")
    print("PASS compiler_rejections")


def _assert_runtime_rejects_incompatible_plan(
    runner: Path, plan_path: Path, work_dir: Path
) -> None:
    data = bytearray(plan_path.read_bytes())
    data[4:8] = (0xFFFFFFFF).to_bytes(4, "little")
    incompatible = work_dir / "incompatible.tgrs"
    input_path = work_dir / "empty-input.bin"
    output_path = work_dir / "unexpected-output.bin"
    incompatible.write_bytes(data)
    input_path.write_bytes(b"")
    completed = subprocess.run(
        [str(runner), str(incompatible), str(input_path), str(output_path)],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode == 0:
        raise AssertionError("runtime accepted an incompatible plan version")
    if output_path.exists():
        raise AssertionError("runtime wrote output for an incompatible plan")
    print("PASS incompatible_plan_rejection")


def _run_gate(runtime: Path, work_dir: Path) -> None:
    cases = [
        *[reduction._nonadjacent_mean_case(s) for s in ("tokens", "keep", "image", "negative")],
        *[movement._constant_table_case(k, q, v) for k in ("Gather", "GatherND") for q in (False, True) for v in (False, True)],
        reduction._linear_spatial_max_case(quantized=False),
        reduction._linear_spatial_max_case(quantized=True),
        *[movement._movement_case(kind, q, variant, tiled=True)
          for kind, variant in (("Gather", 0), ("Gather", 2), ("GatherND", 0), ("EmbeddingLookup", 0),
                                ("StridedSlice", 0), ("MirrorPad", 0), ("ReverseV2", 0), ("DynamicUpdateSlice", 0))
          for q in (False, True)],
        *[reduction._reduction_case(kind, 2, quantized=q, tiled=True)
          for kind in ("ReduceMean", "ReduceMax", "ReduceMin", "ReduceSum", "CumSum") for q in (False, True)],
        reduction._reduce_all_case(2, True, tiled=True),
        *[reduction._arg_case(kind, 2, q, True, tiled=True) for kind in ("ArgMax", "ArgMin") for q in (False, True)],
        *[movement._movement_case(kind, q, variant)
          for kind, variants in (("Gather", 3), ("GatherND", 1), ("StridedSlice", 3),
                                 ("MirrorPad", 2), ("ReverseV2", 1), ("EmbeddingLookup", 1),
                                 ("DynamicUpdateSlice", 2))
          for variant in range(variants) for q in (False, True)],
        *[movement._runtime_index_case(kind, q, variant)
          for kind, variants in (("Gather", 3), ("GatherND", 1), ("EmbeddingLookup", 1),
                                 ("DynamicUpdateSlice", 2))
          for variant in range(variants) for q in (False, True)],
        *[movement._runtime_index_case("Gather", q, variant, arg, cast)
          for arg in ("ArgMax", "ArgMin") for q in (False, True)
          for variant, cast in ((0, False), (1, False), (1, True), (2, False))],
        *[reduction._arg_case(kind, axis, q, keep) for kind in ("ArgMax", "ArgMin")
          for axis in range(3) for q in (False, True) for keep in (False, True)],
        *[reduction._reduction_case(kind, axis, quantized=q, keep=keep)
          for kind in ("ReduceMax", "ReduceMin", "ReduceSum") for axis in range(3)
          for q in (False, True) for keep in (False, True)],
        *[reduction._reduction_case("CumSum", axis, quantized=q, exclusive=ex, reverse=rev)
          for axis in range(3) for q in (False, True)
          for ex in (False, True) for rev in (False, True)],
        *[reduction._reduce_all_case(axis, keep) for axis in range(3) for keep in (False, True)],
        *[elementwise._bool_case(kind, quantized=q, tiled=tiled)
          for kind in ("Equal", "Less", "LessOrEqual", "Greater", "GreaterOrEqual", "And", "Or", "Not", "Where", "Cast", "Sum")
          for q in (False, True) if not (q and kind in {"And", "Or", "Not"})
          for tiled in (False, True)],
        *[elementwise._bool_case(kind, quantized=q, variant=v)
          for kind in ("Equal", "Less", "LessOrEqual", "Greater", "GreaterOrEqual", "And", "Or", "Where")
          for q in (False, True) if not (q and kind in {"And", "Or"})
          for v in (1, 2, 10, 11, *([12] if kind == "Where" else []))],
        *[elementwise._bool_case(kind, quantized=q, tiled=v != 4, variant=v)
          for kind in ("Equal", "Less", "LessOrEqual", "Greater", "GreaterOrEqual", "And", "Or", "Not", "Where", "Cast", "Sum")
          for q in (False, True) if not (q and kind in {"And", "Or", "Not"})
          for v in (4, 5, 6)],
        *[elementwise._bool_composition_case(q, tiled) for q in (False, True) for tiled in (False, True)],
        *[elementwise._activation_case(kind, quantized=quantized)
          for kind in ("LeakyRelu", "PRelu", "Elu", "LogSoftmax", "L2Normalization", "L2Pool")
          for quantized in (False, True) if not (kind == "L2Pool" and quantized)],
        *[elementwise._elementwise_case(kind) for kind in (
            "Abs", "Neg", "Exp", "Log", "Sqrt", "Rsqrt", "Square", "SquaredDifference",
            "Div", "Max", "Min", "Floor", "Ceil", "Round", "Sin", "Cos", "FloorDiv", "FloorMod",
        )],
        *[elementwise._elementwise_case(kind, quantized=True) for kind in (
            "Abs", "Rsqrt", "SquaredDifference", "Max", "Min", "Div",
        )],
        *[replace(elementwise._elementwise_case(kind, quantized=True),
                  name=f"int8_elementwise_{kind.lower()}_untiled", mem_budget="64K",
                  expect_tiled=False) for kind in ("Max", "Min")],
        elementwise._constant_add_case(),
        elementwise._add_relu_fusion_case(),
        elementwise._residual_case(),
        movement._layout_mixing_residual_case(square=True),
        movement._layout_mixing_residual_case(square=False),
        tiling._banded_attention_case(),
        movement._head_permutation_case(),
        movement._reshape_alias_case(spatial=False),
        movement._reshape_alias_case(spatial=True),
        movement._reshape_group_case(),
        elementwise._token_bias_case(),
        movement._flattened_token_head_case(),
        elementwise._constant_divisor_case(),
        elementwise._per_channel_constant_case(),
        movement._bilinear_upsample_case(),
        movement._bilinear_upsample_case(tiled=True),
        *[movement._resize_case(mode, coordinate, geometry, quantized)
          for mode, coordinates in (("linear", ("half_pixel", "asymmetric", "align_corners")),
                                    ("nearest", ("asymmetric", "align_corners", "tf_half_pixel_for_nn", "half_pixel")))
          for coordinate in coordinates for geometry in ("down", "mixed", "scales")
          for quantized in (False, True)],
        chains._split_chain_case(quantized=False),
        chains._split_chain_case(quantized=True),
        convolution._inferred_kernel_chain_case(quantized=False),
        convolution._inferred_kernel_chain_case(quantized=True),
        elementwise._hardswish_case(quantized=False),
        elementwise._hardswish_case(quantized=True),
        elementwise._constant_operand_case(op_type="Add", quantized=True, extent="channels", first=False),
        elementwise._constant_operand_case(op_type="Sub", quantized=True, extent="scalar", first=True),
        elementwise._constant_operand_case(op_type="Mul", quantized=True, extent="full", first=False),
        elementwise._constant_operand_case(op_type="Sub", quantized=False, extent="channels", first=True),
        elementwise._constant_operand_case(op_type="Div", quantized=False, extent="scalar", first=True),
        elementwise._constant_operand_case(op_type="Max", quantized=False, extent="channels", first=False),
        elementwise._broadcast_case(op_type="Add", quantized=False, other_shape=[1, 4, 12, 1], first=False),
        elementwise._broadcast_case(op_type="Add", quantized=True, other_shape=[1, 4, 12, 1], first=False),
        elementwise._broadcast_case(op_type="Sub", quantized=True, other_shape=[1, 1, 12, 12], first=True),
        elementwise._broadcast_case(op_type="Mul", quantized=False, other_shape=[1, 1, 12, 1], first=True),
        elementwise._broadcast_case(op_type="Add", quantized=True, other_shape=[1, 1, 12, 1], first=False,
                                    constant=True),
        movement._standalone_pad_case(quantized=False),
        movement._standalone_pad_case(quantized=True),
        convolution._depthwise_multiplier_case(quantized=False),
        convolution._depthwise_multiplier_case(quantized=True),
        replace(convolution._depthwise_multiplier_case(quantized=True),
                name="int8_depthwise_multiplier_untiled", mem_budget="64K", expect_tiled=False),
        movement._inner_axis_concat_case(axis=2, quantized=False),
        movement._inner_axis_concat_case(axis=2, quantized=True),
        movement._inner_axis_concat_case(axis=3, quantized=False),
        movement._inner_axis_concat_case(axis=3, quantized=True),
        movement._channel_split_case(quantized=False),
        movement._channel_split_case(quantized=True),
        models._squeeze_excitation_case(quantized=False),
        models._squeeze_excitation_case(quantized=True),
        qdq._qdq_bilinear_upsample_case(),
        qdq._quantized_boundary_case(),
        qdq._int8_interface_case(),
        qdq._quantized_batchnorm_case(),
        reduction._whole_map_average_pool_case(),
        elementwise._traced_shape_scale_case(),
        movement._unfolded_feature_map_case(),
        movement._split_case(),
        movement._prepended_token_case(),
        reduction._reduce_mean_case(quantized=False, keepdims=True),
        reduction._reduce_mean_case(quantized=False, keepdims=False),
        reduction._reduce_mean_case(quantized=True, keepdims=True),
        reduction._reduce_mean_case(quantized=True, keepdims=False),
        movement._output_transpose_case(),
        convolution._dilated_conv_case(),
        convolution._depthwise_conv_case(),
        elementwise._math_normalization_case(),
        convolution._conv1d_case(),
        qdq._qdq_conv1d_case(),
        elementwise._rank3_pointwise_case(),
        models._many_stage_case(),
        models._tcn_16k_case(),
        models._ds_cnn_256k_case(),
        models._fc_autoencoder_256k_case(),
        reduction._reduce_case(maximum=False),
        reduction._reduce_case(maximum=True),
        movement._inference_identity_case(),
        convolution._channel_bias_add_case(producer_has_bias=False),
        convolution._channel_bias_add_case(producer_has_bias=True),
        matmul._float_gemm_bias_add_case(),
        matmul._batched_matmul_case(),
        matmul._dynamic_matmul_case(batched=False),
        matmul._dynamic_matmul_case(batched=True),
        elementwise._softmax_axis_case(rank=3, last_axis=True),
        elementwise._softmax_axis_case(rank=3, last_axis=False),
        elementwise._softmax_axis_case(rank=4, last_axis=True),
        elementwise._softmax_axis_case(rank=4, last_axis=False),
        elementwise._legacy_softmax_case(shape=[1, 2, 3], axis=None,
                                         operators=("Transpose", "Reshape", "Softmax", "Reshape", "Transpose")),
        elementwise._legacy_softmax_case(shape=[1, 4, 3, 5], axis=2,
                                         operators=("Transpose", "Reshape", "Softmax", "Reshape", "Transpose")),
        elementwise._legacy_softmax_case(shape=[1, 4, 6], axis=2, operators=("Transpose", "Softmax", "Transpose")),
        tiling._tiled_softmax_case(),
        elementwise._subtract_case(),
        qdq._qdq_subtract_case(),
        reduction._global_max_pool_case(),
        elementwise._sub_constant_case(),
        elementwise._clip_as_relu_case(),
        elementwise._negate_case(),
        matmul._gemm_scaled_case(),
        tiling._tiled_softmax_rank4_case(),
        tiling._tiled_rank4_binary_case(op_type="Add"),
        tiling._tiled_rank4_binary_case(op_type="Sub"),
        tiling._tiled_rank4_binary_case(op_type="Mul"),
        chains._chained_normalization_case(),
        chains._chain_pointwise_before_spatial_case(),
        chains._chain_skip_out_of_a_stage_case(),
        tiling._row_band_alias_case(),
        tiling._row_band_onto_output_case(),
        chains._cotiled_gate_chain_case(),
        tiling._reshape_alias_release_case(),
        qdq._qdq_max_pool_rescale_case(),
        qdq._qdq_rescale_passthrough_case(op_type="Relu"),
        qdq._qdq_rescale_passthrough_case(op_type="Relu6"),
        qdq._qdq_rescale_passthrough_case(op_type="Reshape"),
        qdq._qdq_rescale_passthrough_case(op_type="Resize"),
        tiling._tiled_last_axis_softmax_case(rank=3),
        tiling._tiled_last_axis_softmax_case(rank=4),
        elementwise._layer_norm_case(),
        *[elementwise._gelu_case(mode, shape, budget, two_dimensional=hw)
          for mode in ("none", "tanh")
          for shape, budget, hw in (([16, 8], "128", False),
                                    ([1, 4, 64], "128", False),
                                    ([1, 4, 8, 8], "384", False),
                                    ([1, 8, 7, 19], "256", True))],
        elementwise._erf_case(),
        qdq._qdq_erf_case(),
        *[qdq._qdq_gelu_case(mode, shape, two_dimensional=hw)
          for mode in ("none", "tanh")
          for shape, hw in (([16, 8], False), ([1, 4, 64], False),
                            ([1, 4, 8, 8], False), ([1, 16, 5, 17], True))],
        qdq._qdq_layer_norm_case(),
        tiling._tiled_layer_norm_case(),
        tiling._tiled_erf_case(),
        tiling._tiled_matrix_rows_case(),
        *[elementwise._rank2_dense_constant_case(op_type) for op_type in ("Add", "Max", "Less", "Where")],
        *[elementwise._rank2_row_band_case(kind) for kind in (
            "Div", "Max", "Min", "SquaredDifference", "FloorDiv", "FloorMod")],
        *[elementwise._rank2_row_band_case(kind, constant=True) for kind in ("Max", "Min", "SquaredDifference")],
        tiling._tiled_attention_transpose_case(),
        models._normalized_classifier_case(),
        models._two_input_compressed_case(),
        movement._resize_concat_case(),
        reduction._tiled_pool_case(),
        chains._tiled_pool_chain_case(),
        reduction._tiled_ceil_pool_case(),
        *[common._state_tile_case(case) for case in (
            reduction._tiled_pool_case(), reduction._tiled_global_reduction_case(), movement._fixed_axis_transpose_case(),
            tiling._tiled_matrix_rows_case(), tiling._tiled_attention_transpose_case(),
            movement._reshape_band_case(False, False), movement._reshape_band_case(False, True),
            elementwise._gelu_case("none", [1, 8, 7, 19], "256", two_dimensional=True),
        )],
        common._state_tile_case(reduction._grouped_global_reduction_case(False, False)),
        *[common._unread_input_case(reduction._tiled_pool_case(), state) for state in (False, True)],
        *[reduction._grouped_global_reduction_case(quantized, width_only)
          for quantized in (False, True) for width_only in (False, True)],
        reduction._tiled_global_reduction_case(),
        reduction._qdq_tiled_global_reduction_case(),
        chains._tiled_chain_case(compression="lz4"),
        chains._tiled_chain_case(xip=True),
        qdq._qdq_case("Conv"),
        qdq._qdq_case("AveragePool"),
        qdq._qdq_add_relu_case(),
        qdq._qdq_gemm_bias_add_case(),
        *[qdq._qdq_float_section_case(kind, terminal, tiled=tiled)
          for kind in ("gemm", "conv") for terminal in (True, False)
          for tiled in ((False, True) if kind == "conv" else (False,))],
        matmul._matmul_case(),
        matmul._gemm_no_transpose_case(),
        qdq._qdq_matmul_case(),
        qdq._qdq_row_band_matmul_case(),
        *[qdq._qdq_banded_pool_case(kind) for kind in ("AveragePool", "MaxPool")],
        qdq._qdq_banded_dilated_conv_case(),
        *[qdq._qdq_banded_binary_case(kind) for kind in ("Mul", "Max", "Min")],
        qdq._quint8_activation_case(),
        qdq._qdq_input_ties_case(),
        qdq._qdq_quantize_ties_case(),
        qdq._qdq_conv_relu6_ceiling_case(),
        qdq._tflite_relu6_ceiling_case(),
        qdq._qdq_relu6_ceiling_ties_case(),
        convolution._convtranspose_case("convtranspose", 19, kernel=2, pads=[0, 0, 0, 0]),
        convolution._conv_then_convtranspose_case(),
        convolution._convtranspose_case("convtranspose_overlap", 29, kernel=4, pads=[1, 1, 1, 1]),
        convolution._convtranspose_case("convtranspose_output_padding", 37, kernel=3,
                                        pads=[1, 1, 1, 1], output_padding=[1, 1]),
        convolution._convtranspose_case("convtranspose_asymmetric_pad", 41, kernel=3,
                                        pads=[0, 1, 1, 0]),
        qdq._qdq_case("ConvTranspose"),
        qdq._qdq_convtranspose_per_channel_case(),
        chains._linebuffer_conv_chain_case(),
        chains._qdq_conv_chain_case(),
        tiling_2d._2d_tiled_conv_case(),
        tiling_2d._qdq_2d_tiled_conv_case(),
        tiling_2d._strided_depthwise_2d_case(quantized=False),
        tiling_2d._strided_depthwise_2d_case(quantized=True),
        tiling_2d._2d_tiled_conv_sigmoid_case(),
        tiling_2d._cotiled_concat_2d_case(),
        tiling_2d._qdq_cotiled_concat_2d_case(),
        tiling_2d._cotiled_add_2d_case(),
        tiling_2d._convtranspose_2d_tiled_case(),
        tiling_2d._qdq_convtranspose_2d_tiled_case(),
        tiling_2d._convtranspose_2d_partial_edge_case(),
        models._svdf_case(),
        models._lstm_case(),
        *[replace(case, name=case.name + "_lz4", compression="lz4", exact_uncompressed=True)
          for case in (models._svdf_case(), models._lstm_case())],
        # One operator per stage, so the SVDF reads its constants from the second stage's block.
        replace(models._svdf_case(projected=True), name="projected_svdf_relu_lz4", compression="lz4",
                exact_uncompressed=True, force_one_op_stages=True),
        models._detection_case(),
        control_flow._if_case(True),
        control_flow._if_case(False),
        control_flow._while_case(),
        control_flow._while_counter_case(),
        control_flow._int8_if_case(True),
        control_flow._int8_if_case(False),
        control_flow._if_tiled_case(True),
        control_flow._if_tiled_case(False),
        control_flow._nested_if_case(),
        control_flow._if_case(True, standard=True),
        control_flow._if_case(False, standard=True),
        control_flow._int8_if_case(True, standard=True),
        control_flow._int8_if_case(False, standard=True),
        control_flow._nested_if_case(standard=True),
    ]
    cases.extend(movement._reshape_band_case(quantized, merge)
                 for quantized in (False, True) for merge in (False, True))
    cases.extend(movement._reshape_band_case(quantized, merge, contiguous=True)
                 for quantized in (False, True) for merge in (False, True))
    cases.extend(reduction._reduction_case(kind, 2, quantized=q, tiled=True, leading=True,
                                           exclusive=kind == "CumSum", reverse=kind == "CumSum")
                 for q in (False, True)
                 for kind in ("ReduceMean", "ReduceMax", "ReduceMin", "ReduceSum", "CumSum"))
    cases.extend(movement._movement_case(kind, q, 2 if kind == "Gather" else 0, tiled=True, leading=True)
                 for q in (False, True)
                 for kind in ("Gather", "StridedSlice", "MirrorPad", "ReverseV2", "DynamicUpdateSlice"))
    cases.extend(reduction._arg_case(kind, 2, q, True, tiled=True, leading=True)
                 for q in (False, True) for kind in ("ArgMax", "ArgMin"))
    cases.append(reduction._reduce_all_case(2, True, tiled=True, leading=True))
    cases.extend(movement._leading_index_case(kind, q)
                 for q in (False, True) for kind in ("GatherND", "EmbeddingLookup"))
    tight_cases = [
        models._svdf_case(), models._lstm_case(),
        movement._channel_split_case(quantized=True),
        movement._inner_axis_concat_case(axis=2, quantized=True),
        movement._inner_axis_concat_case(axis=3, quantized=True),
        movement._standalone_pad_case(quantized=True),
        elementwise._broadcast_case(op_type="Add", quantized=True, other_shape=[1, 4, 12, 1], first=False),
        elementwise._broadcast_case(op_type="Sub", quantized=True, other_shape=[1, 4, 1, 12], first=True),
        elementwise._broadcast_case(op_type="Add", quantized=True, other_shape=[1, 4, 12, 12], first=False, constant=True),
    ]
    cases.extend(replace(case, name=case.name + "_bands", mem_budget="256",
                         slow_budget="64K", expect_tiled=True, exact_untiled=True,
                         tiled_operators=tuple(k for k in case.expected_operators if k not in {"Conv", "Mul", "Transpose"}))
                 for case in tight_cases)
    cases.extend(reduction._arg_case(kind, 2, q, False, tiled=True, leading=True)
                 for q in (False, True) for kind in ("ArgMax", "ArgMin"))
    cases.extend(replace(movement._runtime_index_case(kind, q, variant=2 if kind == "Gather" else 0,
                                                      tiled=True), exact_untiled=True,
                         tiled_operators=(kind,))
                 for q in (False, True) for kind in ("Gather", "GatherND", "DynamicUpdateSlice"))
    unrouted = set(_ROUTINGS) - {case.name for case in cases} - {"lstm_unrolled"}
    if unrouted:
        raise AssertionError(f"vendor routing stated for cases the gate does not run: {sorted(unrouted)}")
    cases = [replace(case, **_ROUTINGS[case.name]._asdict()) if case.name in _ROUTINGS else case
             for case in cases]
    covered_operators = {
        operator for case in cases for operator in case.expected_operators
    }
    supported_operators = KERNEL_CAPABILITIES["reference"].native_operators
    if covered_operators != supported_operators:
        raise AssertionError(
            "reference corpus coverage drift: "
            f"missing={sorted(supported_operators - covered_operators)}, "
            f"unexpected={sorted(covered_operators - supported_operators)}"
        )
    print(
        f"PASS reference_operator_coverage "
        f"{len(covered_operators)}/{len(supported_operators)}"
    )

    # The builds are independent, so they run at once.
    with ThreadPoolExecutor(max_workers=5) as builds:
        default_build = builds.submit(_build_runner, runtime, work_dir / "runtime-build")
        a16_build = builds.submit(_build_runner, runtime, work_dir / "runtime-build-a16",
                                  alignment=16)
        host_build = builds.submit(_build_host_library, runtime, work_dir / "host-build")
        accel_build = builds.submit(_build_accel_runner, runtime,
                                    work_dir / "runtime-build-accel")
        accel_a16_build = builds.submit(_build_accel_runner, runtime,
                                        work_dir / "runtime-build-accel-a16", alignment=16)
    default_path, rows_path = default_build.result()
    a16_path, _ = a16_build.result()
    os.environ["TIGRIS_HOST_LIBRARY"] = str(host_build.result())
    accel_path, accel_a16_path = accel_build.result(), accel_a16_build.result()
    replay = _AccelReplay(work_dir)
    try:
        runner, rows_runner, runner_a16 = (
            _ContractRunner(path, accel, replay) for path, accel in
            ((default_path, accel_path), (rows_path, accel_path), (a16_path, accel_a16_path)))
        _run_peak_fixtures((runner, runner_a16), work_dir)
        first_plan: Path | None = None
        for case in cases:
            if case.recompute_metric:
                _run_metric_case(case, rows_runner, work_dir)
            else:
                plan_path = _run_case(case, runner, work_dir, runner_a16)
                if first_plan is None:
                    first_plan = plan_path
        assert first_plan is not None, "gate needs at least one non-metric case"
        replay.finish()
    finally:
        replay.close()
    _assert_compile_rejected(work_dir)
    _assert_runtime_rejects_incompatible_plan(runner.path, first_plan, work_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runtime",
        type=Path,
        default=Path("../tigris-runtime"),
        help="Path to the sibling tigris-runtime checkout",
    )
    parser.add_argument(
        "--artifacts",
        type=Path,
        help="Retain generated ONNX, plan, input, and output files in this empty directory",
    )
    args = parser.parse_args()
    runtime = args.runtime.resolve()
    if not (runtime / "CMakeLists.txt").is_file():
        raise SystemExit(f"runtime checkout not found: {runtime}")

    if args.artifacts:
        work_dir = args.artifacts.resolve()
        work_dir.mkdir(parents=True, exist_ok=True)
        if any(work_dir.iterdir()):
            raise SystemExit(f"artifact directory must be empty: {work_dir}")
        _run_gate(runtime, work_dir)
        print(f"Retained contract artifacts in {work_dir}")
    else:
        with tempfile.TemporaryDirectory(prefix="tigris-crossrepo-") as temp:
            _run_gate(runtime, Path(temp))
    print("Cross-repository compiler/runtime contract gate passed.")


if __name__ == "__main__":
    main()
