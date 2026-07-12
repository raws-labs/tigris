#!/usr/bin/env python3
"""Generate C test fixtures for tigris-runtime.

Usage (from tigris/tigris/ with venv activated):

    python scripts/gen_fixtures.py models/mobilenetv2.onnx \
        -o ../tigris-runtime/test/fixtures --compress
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import click
import numpy as np
import onnx
import onnxruntime as ort
from rich.console import Console

from tigris.cli import _run_pipeline
from tigris.emitters.binary.writer import emit_binary
from tigris.fixtures import (
    build_conv_relu_chain,
    build_ds_cnn,
    build_fc_autoencoder,
    build_linear_3op,
    build_tcn,
)

console = Console()

_INLINE = [
    ("linear_3op",      build_linear_3op,      "4K",   False),
    # The stage's true co-resident input/output peak is 53,888 bytes.  Keep the
    # binary contract fixture single-stage so it tests the wire format rather
    # than temporal partitioning behavior.
    ("conv_relu_chain", build_conv_relu_chain,  "64K",  False),
    ("ds_cnn",          build_ds_cnn,           "256K", True),
    ("fc_autoencoder",  build_fc_autoencoder,   "256K", True),
    ("tcn",             build_tcn,              "256K", True),
]


def _gen_reference(onnx_path: str, out_path: Path) -> None:
    sess = ort.InferenceSession(onnx_path)
    inp_meta = sess.get_inputs()[0]
    shape = [1 if isinstance(d, str) else d for d in inp_meta.shape]
    inp_data = np.ones(shape, dtype=np.float32)
    results = sess.run(None, {inp_meta.name: inp_data})
    ref_data = b"".join(r.astype(np.float32).tobytes() for r in results)
    out_path.write_bytes(ref_data)
    console.print(f"  {out_path} ({len(ref_data)} bytes, {len(ref_data)//4} floats)")


@click.command()
@click.argument("model", type=click.Path(exists=True))
@click.option("--output-dir", "-o", required=True, type=click.Path())
@click.option("--mem", "-m", default="256K")
@click.option("--compress/--no-compress", default=False)
def main(model: str, output_dir: str, mem: str, compress: bool):
    """Regenerate all C test fixtures."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for name, builder, budget, gen_ref in _INLINE:
        console.print(f"[bold]Generating {name} fixture...[/]")
        onnx_model = builder()
        with tempfile.TemporaryDirectory() as tmp:
            onnx_path = Path(tmp) / f"{name}.onnx"
            onnx.save(onnx_model, str(onnx_path))
            ag, _ = _run_pipeline(str(onnx_path), (budget,))
            plan_path = out / f"{name}.tgrs"
            emit_binary(ag, plan_path)
            console.print(f"  {plan_path} ({plan_path.stat().st_size} bytes)")
            if compress and ag.weight_data:
                lz4_path = out / f"{name}.lz4.tgrs"
                emit_binary(ag, lz4_path, compress="lz4")
                console.print(f"  {lz4_path} ({lz4_path.stat().st_size} bytes, LZ4)")
            if gen_ref:
                _gen_reference(str(onnx_path), out / f"{name}.reference.bin")

    console.print(f"[bold]Compiling MobileNetV2: {model}...[/]")
    ag, _ = _run_pipeline(model, (mem,))
    plan_path = out / "mobilenetv2.tgrs"
    emit_binary(ag, plan_path)
    console.print(f"  Plan: {plan_path} ({plan_path.stat().st_size} bytes)")
    if compress:
        lz4_path = out / "mobilenetv2.lz4.tgrs"
        emit_binary(ag, lz4_path, compress="lz4")
        console.print(f"  LZ4 plan: {lz4_path} ({lz4_path.stat().st_size} bytes)")
    _gen_reference(model, out / "mobilenetv2.reference.bin")

    console.print(f"\n[bold green]All fixtures written to {out}/[/]")


if __name__ == "__main__":
    main()
