#!/usr/bin/env python3
"""Check the runtime's operator tables and dispatch switches against the compiler."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from tigris.capabilities import EXECUTOR_OPERATORS, KERNEL_CAPABILITIES
from tigris.operator_contract import OP_C_NAMES, OP_TYPE_MAP

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
        sources = {
            "reference": (runtime / "src/tigris_kernels.c").read_text(),
            "s8_ref": (runtime / "src/tigris_kernels_s8.c").read_text(),
            "esp-nn": (runtime / "src/tigris_kernels_esp_nn.c").read_text(),
            "cmsis-nn": (runtime / "src/tigris_kernels_cmsis_nn.c").read_text(),
        }
        executor = (runtime / "src/tigris_executor.c").read_text()
    except OSError as exc:
        return errors + [f"cannot read runtime capability source: {exc}"]

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
    code_to_operator = {value: key for key, value in OP_TYPE_MAP.items()}
    for backend, function in functions.items():
        try:
            body = _function_body(sources[backend], function)
            case_names = set(
                (_ADAPTER_CASE_RE if backend in {"esp-nn", "cmsis-nn"} else _CASE_RE)
                .findall(body)
            )
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
                f"runtime-only={[code_to_operator[code] for code in sorted(actual - expected)]}, "
                f"compiler-only={[code_to_operator[code] for code in sorted(expected - actual)]}"
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
