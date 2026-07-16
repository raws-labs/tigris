#!/usr/bin/env python3
"""Generate or verify the public operator/backend capability artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tigris import SCHEMA_VERSION
from tigris.capabilities import (
    CODEGEN_BACKENDS,
    CONDITIONAL_FALLBACKS,
    KERNEL_CAPABILITIES,
    OPERATOR_CONSTRAINTS,
    capability_rows,
    describe_codegen_route,
    resolve_kernel_backend,
)


DTYPES = ("float32", "int8")


def capability_matrix() -> dict[str, object]:
    """Return the deterministic, language-neutral public capability contract."""
    operators: list[dict[str, object]] = []
    for raw_row in capability_rows():
        operator = str(raw_row["operator"])
        routes = {
            backend: raw_row[backend]
            for backend in KERNEL_CAPABILITIES
        }
        conditional = {
            backend: conditions[operator]
            for backend, conditions in CONDITIONAL_FALLBACKS.items()
            if operator in conditions
        }
        row: dict[str, object] = {
            "operator": operator,
            "opcode": raw_row["opcode"],
            "routes": routes,
        }
        if conditional:
            row["conditional_fallbacks"] = conditional
        constraints = OPERATOR_CONSTRAINTS.get(operator)
        if constraints:
            row["constraints"] = list(constraints)
        operators.append(row)

    selections = []
    for backend in CODEGEN_BACKENDS:
        for dtype in DTYPES:
            selections.append({
                "codegen_backend": backend,
                "dtype": dtype,
                "kernel_backend": resolve_kernel_backend(backend, dtype),
                "description": describe_codegen_route(backend, dtype),
            })

    return {
        "artifact_format": 1,
        "schema_version": SCHEMA_VERSION,
        "source": "tigris.capabilities",
        "scope": (
            "post-normalization operator dispatch; model-specific shape, attribute, "
            "quantization, memory, and tiling validation still applies"
        ),
        "legend": {
            "native": "implemented by this kernel backend",
            "fallback:<backend>": "executed by the named explicit fallback backend",
            "unsupported": "no executable route; code generation rejects the plan",
        },
        "codegen_selections": selections,
        "kernel_backends": [
            {
                "name": capability.name,
                "dtype": capability.dtype,
                "fallback": capability.fallback,
            }
            for capability in KERNEL_CAPABILITIES.values()
        ],
        "operators": operators,
    }


def encoded_matrix() -> str:
    return json.dumps(capability_matrix(), indent=2, sort_keys=True) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--check", type=Path, metavar="PATH")
    group.add_argument("--output", type=Path, metavar="PATH")
    args = parser.parse_args()
    encoded = encoded_matrix()
    if args.check:
        if not args.check.is_file() or args.check.read_text() != encoded:
            parser.error(f"capability artifact is stale: {args.check}")
        print(f"Capability artifact is current: {args.check}")
    elif args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    else:
        print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
