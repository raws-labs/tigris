#!/usr/bin/env python3
"""Generate a small reference-backend C harness for CI syntax checks."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import onnx

from tigris.cli import _run_pipeline
from tigris.emitters.binary.writer import emit_binary_bytes
from tigris.emitters.codegen import generate_c
from tigris.fixtures import build_linear_3op


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        model_path = Path(tmp) / "linear_3op.onnx"
        onnx.save(build_linear_3op(), model_path)
        graph, _ = _run_pipeline(str(model_path), ("4K",))
        source = generate_c(emit_binary_bytes(graph), "reference")

    args.output.write_text(source)


if __name__ == "__main__":
    main()
