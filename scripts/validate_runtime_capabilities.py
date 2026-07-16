#!/usr/bin/env python3
"""Check the compiler capability contract against runtime dispatch switches."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from tigris.capabilities import KERNEL_CAPABILITIES
from tigris.emitters.binary.defs import OP_TYPE_MAP


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
    try:
        header = (runtime / "include/tigris.h").read_text()
        enum_values = {name: int(value) for name, value in _ENUM_RE.findall(header)}
        sources = {
            "reference": (runtime / "src/tigris_kernels.c").read_text(),
            "s8_ref": (runtime / "src/tigris_kernels_s8.c").read_text(),
            "esp-nn": (runtime / "src/tigris_kernels_esp_nn.c").read_text(),
            "cmsis-nn": (runtime / "src/tigris_kernels_cmsis_nn.c").read_text(),
        }
    except OSError as exc:
        return [f"cannot read runtime capability source: {exc}"]

    expected_schema_codes = set(OP_TYPE_MAP.values())
    runtime_schema_codes = set(enum_values.values()) - {255}
    if runtime_schema_codes != expected_schema_codes:
        errors.append(
            "runtime/compiler opcode enums differ: "
            f"runtime-only={sorted(runtime_schema_codes - expected_schema_codes)}, "
            f"compiler-only={sorted(expected_schema_codes - runtime_schema_codes)}"
        )

    functions = {
        "reference": "tigris_dispatch_kernel",
        "s8_ref": "tigris_dispatch_kernel_s8",
        "esp-nn": "tigris_dispatch_kernel_esp_nn",
        "cmsis-nn": "tigris_dispatch_kernel_cmsis_nn",
    }
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
            for operator in KERNEL_CAPABILITIES[backend].native_operators
        }
        if actual != expected:
            code_to_operator = {value: key for key, value in OP_TYPE_MAP.items()}
            errors.append(
                f"{backend} dispatch differs from compiler capabilities: "
                f"runtime-only={[code_to_operator[code] for code in sorted(actual - expected)]}, "
                f"compiler-only={[code_to_operator[code] for code in sorted(expected - actual)]}"
            )

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
