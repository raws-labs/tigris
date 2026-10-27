#!/usr/bin/env python3
"""Check the runtime's operator tables, dispatch switches, wire constants,
record sizes and build-limit defaults against the compiler."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import tempfile
from pathlib import Path

from tigris.analysis.plan_graph import BUILD_LIMIT_DEFAULTS
from tigris.capabilities import KERNEL_CAPABILITIES
from tigris.emitters.binary import defs
from tigris.loaders.onnx.normalize import detection_scratch_bytes
from tigris.operator_contract import EXECUTOR_OPERATORS, OP_C_NAMES, OP_TYPE_BY_CODE, OP_TYPE_MAP

if __package__:
    from .generate_operator_contract import render, render_config
else:
    from generate_operator_contract import render, render_config


_ENUM_RE = re.compile(r"\b(TIGRIS_OP_[A-Z0-9_]+)\s*=\s*(\d+)\s*,")
_CASE_RE = re.compile(r"\bcase\s+(TIGRIS_OP_[A-Z0-9_]+)\s*:")
_ADAPTER_CASE_RE = re.compile(
    r"\bcase\s+(TIGRIS_OP_[A-Z0-9_]+)\s*:\s*"
    r"(?:(?:rc\s*=\s*)|return\s+)adapt_[a-z0-9_]+\s*\(",
)
_DEFAULT_DELEGATE_RE = re.compile(r"\bdefault\s*:\s*return\s+(tigris_dispatch_[a-z0-9_]+)\s*\(")


_DEFINE_RE = re.compile(r"^[ \t]*#[ \t]*define[ \t]+(TIGRIS_[A-Z0-9_]+)[ \t]+([^\n]*)$", re.M)
_INTEGER_RE = re.compile(r"\(?\s*(0[xX][0-9A-Fa-f]+|\d+)[uU]?\s*\)?\s*(?:/[*/].*)?")
_ENUM_VALUE_RE = re.compile(r"\b(TIGRIS_[A-Z0-9_]+)\s*=\s*(0[xX][0-9A-Fa-f]+|\d+)[uU]?\s*[,}]")
_LAYOUT_SIZE_RE = re.compile(
    r"\bTIGRIS_LAYOUT_ASSERT\(\s*\w+\s*,\s*sizeof\((\w+)\)\s*==\s*(\d+)\s*\)")

# Each runtime record type and the size of its defs.py layout.
_RECORD_SIZES = {
    "tigris_file_header_t": defs.HEADER_SIZE,
    "tigris_section_entry_t": defs.SECTION_ENTRY_SIZE,
    "tigris_tensor_t": defs.TENSOR_SIZE,
    "tigris_spatial_attrs_t": defs.SPATIAL_ATTRS_STRUCT.size,
    "tigris_op_t": defs.OP_SIZE,
    "tigris_stage_t": defs.STAGE_SIZE,
    "tigris_tile_plan_t": defs.TILE_PLAN_SIZE,
    "tigris_tile_plan_v10_t": defs.TILE_PLAN_V10_SIZE,
    "tigris_tile_plan_v11_t": defs.TILE_PLAN_V11_SIZE,
    "tigris_weight_entry_t": defs.WEIGHT_ENTRY_SIZE,
    "tigris_quant_param_t": defs.QUANT_PARAM_SIZE,
    "tigris_weight_block_t": defs.WEIGHT_BLOCK_SIZE,
    "tigris_op_attribute_t": defs.OP_ATTRIBUTE_SIZE,
    "tigris_state_entry_t": defs.STATE_ENTRY_SIZE,
    "tigris_subgraph_t": defs.SUBGRAPH_ENTRY_SIZE,
    "tigris_memory_t": defs.MEMORY_SIZE,
}

# (boxes, detections, candidates) at which TIGRIS_DETECTION_SCRATCH_BYTES is evaluated.
_DETECTION_ARGS = [(boxes, detections, detections + extra)
                   for boxes in (1, 10, 1917) for detections in (1, 10, 100) for extra in (0, 100)]


def _header_integers(header: str) -> tuple[dict[str, list[int]], set[str]]:
    """Each TIGRIS_ name's integer values in a header, from #define lines and
    enum entries, and the #define names whose value is not an integer literal."""
    values: dict[str, list[int]] = {}
    other: set[str] = set()
    for name, value in _DEFINE_RE.findall(header):
        match = _INTEGER_RE.fullmatch(value.strip())
        if match:
            values.setdefault(name, []).append(int(match[1], 0))
        else:
            other.add(name)
    for name, value in _ENUM_VALUE_RE.findall(header):
        values.setdefault(name, []).append(int(value, 0))
    return values, other


def _wire_constants(header: str) -> list[str]:
    """defs.py integers against the tigris.h constant of the same name,
    TIGRIS_ prefixed, with TENSOR_FLAG_ written TENSOR_ there."""
    errors = []
    values, other = _header_integers(header)
    for name, value in sorted(vars(defs).items()):
        if not name.isupper() or not isinstance(value, int) or isinstance(value, bool):
            continue
        c_name = "TIGRIS_" + name.replace("TENSOR_FLAG_", "TENSOR_", 1)
        if c_name in other:
            errors.append(f"tigris.h {c_name} is not an integer literal; defs.py {name} is {value}")
        elif c_name in values and set(values[c_name]) != {value}:
            errors.append(f"tigris.h {c_name} is {values[c_name]}, defs.py {name} is {value}")
    return errors


def _record_sizes(header: str) -> list[str]:
    errors = []
    asserted: dict[str, list[int]] = {}
    for record, size in _LAYOUT_SIZE_RE.findall(header):
        asserted.setdefault(record, []).append(int(size))
    for record, size in _RECORD_SIZES.items():
        if record not in asserted:
            errors.append(f"tigris.h has no TIGRIS_LAYOUT_ASSERT on sizeof({record})")
        elif set(asserted[record]) != {size}:
            errors.append(f"tigris.h asserts sizeof({record}) == {asserted[record]}, defs.py has {size}")
    return errors


def _build_limit_defaults(config: str) -> list[str]:
    errors = []
    values, other = _header_integers(config)
    for name, default in BUILD_LIMIT_DEFAULTS.items():
        if name in other or name not in values:
            errors.append(f"tigris_config.h has no integer default for {name}")
        elif set(values[name]) != {default}:
            errors.append(f"tigris_config.h {name} defaults to {values[name]}, the compiler to {default}")
    return errors


def _detection_scratch(include: Path) -> list[str]:
    """TIGRIS_DETECTION_SCRATCH_BYTES, compiled and run, against the compiler's formula."""
    calls = "".join(
        f'    printf("%zu\\n", (size_t)TIGRIS_DETECTION_SCRATCH_BYTES({b}u, {d}u, {c}u));\n'
        for b, d, c in _DETECTION_ARGS)
    source = f'#include <stdio.h>\n#include "tigris.h"\nint main(void)\n{{\n{calls}    return 0;\n}}\n'
    with tempfile.TemporaryDirectory() as tmp:
        program = Path(tmp) / "detection_scratch"
        (Path(tmp) / "detection_scratch.c").write_text(source)
        try:
            build = subprocess.run(
                [os.environ.get("CC", "cc"), "-std=c11", "-I", str(include),
                 str(Path(tmp) / "detection_scratch.c"), "-o", str(program)],
                capture_output=True, text=True, check=False)
            if build.returncode != 0:
                return [f"cannot build the TIGRIS_DETECTION_SCRATCH_BYTES probe: {build.stderr.strip()}"]
            run = subprocess.run([str(program)], capture_output=True, text=True, check=False)
        except OSError as exc:
            return [f"cannot build the TIGRIS_DETECTION_SCRATCH_BYTES probe: {exc}"]
    if run.returncode != 0:
        return [f"the TIGRIS_DETECTION_SCRATCH_BYTES probe exited with {run.returncode}"]
    runtime = [int(line) for line in run.stdout.split()]
    if len(runtime) != len(_DETECTION_ARGS):
        return [f"the TIGRIS_DETECTION_SCRATCH_BYTES probe printed {len(runtime)} values, expected {len(_DETECTION_ARGS)}"]
    return [f"TIGRIS_DETECTION_SCRATCH_BYTES{args} is {value}, the compiler's is "
            f"{detection_scratch_bytes(*args)}"
            for args, value in zip(_DETECTION_ARGS, runtime)
            if value != detection_scratch_bytes(*args)]


def _function_body(source: str, function: str) -> str:
    match = re.search(rf"\b{re.escape(function)}\s*\([^;]*?\)\s*\{{", source, re.S)
    if not match:
        raise ValueError(f"cannot find runtime function {function}")
    start = match.end() - 1
    depth = 0
    for index in range(start, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[start:index + 1]
    raise ValueError(f"unterminated runtime function {function}")


def _dispatched_cases(source: str, function: str, shared: str) -> set[str]:
    """Case labels of a dispatcher and of the switches in `shared` its default case returns into."""
    body = _function_body(source, function)
    names = set(_CASE_RE.findall(body))
    seen = {function}
    while (delegate := _DEFAULT_DELEGATE_RE.search(body)) and delegate.group(1) not in seen:
        seen.add(delegate.group(1))
        body = _function_body(shared, delegate.group(1))
        names |= set(_CASE_RE.findall(body))
    return names


def _opcodes(names: set[str], enum_values: dict[str, int]) -> set[int]:
    missing = names - enum_values.keys()
    if missing:
        raise ValueError(f"runtime cases use unknown enum names: {sorted(missing)}")
    return {enum_values[name] for name in names}


def validate(runtime: Path) -> list[str]:
    errors: list[str] = []
    for name, content in (("tigris_op_contract.h", render()),
                          ("tigris_op_config.h", render_config())):
        path = runtime / "src" / name
        try:
            actual = path.read_text()
        except OSError as exc:
            errors.append(f"cannot read runtime operator contract {name}: {exc}")
            continue
        if actual != content:
            errors.append(f"runtime operator contract {name} is stale")

    try:
        header = (runtime / "include/tigris.h").read_text()
        config = (runtime / "include/tigris_config.h").read_text()
        sources = {
            "reference": (runtime / "src/tigris_kernels.c").read_text(),
            "s8_ref": (runtime / "src/tigris_kernels_s8.c").read_text(),
            "esp-nn": (runtime / "src/tigris_kernels_esp_nn.c").read_text(),
            "cmsis-nn": (runtime / "src/tigris_kernels_cmsis_nn.c").read_text(),
        }
        executor = (runtime / "src/tigris_executor.c").read_text()
    except OSError as exc:
        return errors + [f"cannot read runtime capability source: {exc}"]

    errors += _wire_constants(header)
    errors += _record_sizes(header)
    errors += _build_limit_defaults(config)
    errors += _detection_scratch(runtime / "include")

    enum_values = {name: int(value) for name, value in _ENUM_RE.findall(header)}
    runtime_names = {value: name.removeprefix("TIGRIS_OP_")
                     for name, value in enum_values.items() if value != 255}
    if runtime_names != OP_C_NAMES:
        errors.append("runtime/compiler opcode names differ")

    # The dispatch switches are written by hand, so the table cannot vouch for them.
    functions = {
        "reference": "tigris_dispatch_kernel",
        "s8_ref": "tigris_dispatch_kernel_s8",
        "esp-nn": "tigris_dispatch_kernel_esp_nn",
        "cmsis-nn": "tigris_dispatch_kernel_cmsis_nn",
    }
    # The float and int8 dispatchers share a switch for dtype-generic operators.
    reference_sources = sources["reference"] + "\n" + sources["s8_ref"]
    for backend, function in functions.items():
        try:
            if backend in {"esp-nn", "cmsis-nn"}:
                body = _function_body(sources[backend], function)
                case_names = set(_ADAPTER_CASE_RE.findall(body))
            else:
                case_names = _dispatched_cases(sources[backend], function, reference_sources)
            actual = _opcodes(case_names, enum_values)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        expected = {
            OP_TYPE_MAP[operator]
            for operator in KERNEL_CAPABILITIES[backend].native_operators - EXECUTOR_OPERATORS
        }
        if actual != expected:
            errors.append(
                f"{backend} dispatch differs from compiler capabilities: "
                f"runtime-only={[OP_TYPE_BY_CODE[code] for code in sorted(actual - expected)]}, "
                f"compiler-only={[OP_TYPE_BY_CODE[code] for code in sorted(expected - actual)]}"
            )

    names = {value: name for name, value in enum_values.items()}
    for operator in sorted(EXECUTOR_OPERATORS):
        if names.get(OP_TYPE_MAP[operator], "?") not in executor:
            errors.append(f"the executor does not run {operator}")

    for backend in ("esp-nn", "cmsis-nn"):
        if "tigris_dispatch_kernel_s8" not in sources[backend]:
            errors.append(f"{backend} no longer contains the declared s8_ref fallback")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, required=True)
    args = parser.parse_args()
    errors = validate(args.runtime.resolve())
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print("Compiler capability contract matches runtime dispatch.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
