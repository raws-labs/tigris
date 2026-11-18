#!/usr/bin/env python3
"""Generate small ExecuTorch Cortex-M programs (.pte) and their inputs.

Each model below is quantized with ExecuTorch's Cortex-M quantizer, lowered by
its Cortex-M pass manager for one CPU of each CMSIS-NN class (M0 scalar, M7
DSP, M55 MVE) with int8 model inputs and outputs, and written as
`<model>_<cpu>.pte` to tests/fixtures/pte/. `bands` has activations large
enough for export to write row bands. `resblock_align64` is the residual model
for M7 only, serialized with 64-byte constant tensor alignment, which pads the
extended header to 64 bytes. `ops`, for M7 only, holds the
other cortex_m operators of a convolutional network. Next to each,
`<model>_<cpu>.npz` holds seeded int8 inputs in NHWC order (`inputs`) and the input quantization
(`input_scale`, `input_zero_point`). With `--runner`, a host build of the
ExecuTorch runtime with the cortex_m operators and CMSIS-NN's portable C
kernels (src/tigris/frontends/pte_host_runner.cpp; usage: `runner model.pte
input.bin output.bin`, one output per input, raw bytes) runs the scalar and DSP
programs, whose outputs must agree, and records them as `output_bytes`, one
row per input. Those kernels run an MVE program wrong, so MVE programs record
outputs only with `--mve-runner`, a command of the same usage that runs the
runtime built with MVE kernels (the same runner source on an emulated
Cortex-M55, built and run as its header describes); their outputs must agree
with the other classes'. Needs torch and executorch from
the pytorch group:

    python scripts/gen_pte_fixtures.py [--runner PATH] [--mve-runner CMD] [model ...]
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
from executorch.backends.cortex_m.edge_compile_config import cortex_m_edge_compile_config
from executorch.backends.cortex_m.passes.cortex_m_pass_manager import CortexMPassManager
from executorch.backends.cortex_m.quantizer.quantizer import CortexMQuantizer
from executorch.backends.cortex_m.target_config import CortexM, CortexMTargetConfig
from executorch.exir import ExecutorchBackendConfig, to_edge
from executorch.exir.passes.quantize_io_pass import QuantizeInputs, QuantizeOutputs
from torch import nn
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

OUT = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "pte"
CPUS = {"m0": "M0", "m7": "M7", "m55": "M55"}
RUNS = ("m0", "m7")  # the classes the portable C kernels run bit for bit
MVE_RUN = "m55"
SAMPLES = 4
CALIBRATION = 16


class Chain(nn.Module):
    """Convolutions at stride 1 and 2, a 5x5 depthwise convolution, whose
    CMSIS-NN scratch differs in each class, and a pointwise convolution."""

    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 16, 5, padding=2, groups=16), nn.ReLU(),
            nn.Conv2d(16, 8, 1), nn.ReLU())

    def forward(self, x):
        return self.body(x)


class ResBlock(nn.Module):
    """A strided stem, a residual block, average pooling and a linear head."""

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.ReLU())
        self.a = nn.Sequential(nn.Conv2d(16, 16, 3, padding=1), nn.ReLU())
        self.b = nn.Conv2d(16, 16, 3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AvgPool2d(8)
        self.head = nn.Linear(16, 10)
        with torch.no_grad():
            self.head.bias.copy_(torch.linspace(-0.5, 0.5, 10))

    def forward(self, x):
        y = self.stem(x)
        y = self.relu(self.b(self.a(y)) + y)
        return self.head(torch.flatten(self.pool(y), 1))


class Ops(nn.Module):
    """The other cortex_m operators a program can hold: max pooling, a
    sigmoid table, an elementwise product, explicit padding, a transposed
    convolution and softmax."""

    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 8, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        self.c2 = nn.Conv2d(8, 8, 3, padding=1)
        self.up = nn.ConvTranspose2d(8, 8, 2, stride=2)
        self.pool2 = nn.AvgPool2d(20)
        self.head = nn.Linear(8, 10)

    def forward(self, x):
        y = self.pool1(torch.relu(self.c1(x)))
        y = self.c2(y)
        y = self.up(nn.functional.pad(y * torch.sigmoid(y), (1, 1, 1, 1)))
        return torch.softmax(self.head(torch.flatten(self.pool2(y), 1)), -1)


class Bands(nn.Module):
    """A network whose activations are large next to its bookkeeping, so row
    bands pay off: a strided stem, a depthwise and a pointwise convolution, a
    1x3 convolution, a residual block, a strided convolution, average pooling
    and a linear head with a bias."""

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 8, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(8, 8, 3, padding=1, groups=8), nn.ReLU(),
            nn.Conv2d(8, 16, 1), nn.ReLU(),
            nn.Conv2d(16, 16, (1, 3), padding=(0, 1)), nn.ReLU())
        self.a = nn.Sequential(nn.Conv2d(16, 16, 3, padding=1), nn.ReLU())
        self.b = nn.Conv2d(16, 16, 3, padding=1)
        self.relu = nn.ReLU()
        self.down = nn.Sequential(nn.Conv2d(16, 24, 3, stride=2, padding=1), nn.ReLU())
        self.pool = nn.AvgPool2d(12)
        self.head = nn.Linear(24, 10)
        with torch.no_grad():
            self.head.bias.copy_(torch.linspace(-0.5, 0.5, 10))

    def forward(self, x):
        y = self.stem(x)
        y = self.relu(self.b(self.a(y)) + y)
        return self.head(torch.flatten(self.pool(self.down(y)), 1))


# Each model's module, input shape, CPUs and constant tensor alignment.
MODELS = {
    "bands": (Bands, (1, 3, 48, 48), tuple(CPUS), None),
    "chain": (Chain, (1, 3, 16, 16), tuple(CPUS), None),
    "resblock": (ResBlock, (1, 3, 16, 16), tuple(CPUS), None),
    "resblock_align64": (ResBlock, (1, 3, 16, 16), ("m7",), 64),
    "ops": (Ops, (1, 3, 16, 16), ("m7",), None),
}


def _channels_last(t: torch.Tensor) -> torch.Tensor:
    return t.to(memory_format=torch.channels_last)


def _lower(name: str, cpu: str):
    """The program for `cpu` as .pte bytes, seeded int8 NHWC inputs, and the
    input scale and zero point."""
    build, shape, _, alignment = MODELS[name]
    torch.manual_seed(0)
    model = build().eval()
    gen = torch.Generator().manual_seed(1)
    calibration = [torch.randn(shape, generator=gen) for _ in range(CALIBRATION)]
    samples = [torch.randn(shape, generator=gen) for _ in range(SAMPLES)]
    target = CortexMTargetConfig(cpu=CortexM[CPUS[cpu]])
    example = (_channels_last(calibration[0]),)
    graph = torch.export.export(model, example).module(check_guards=False)
    graph = CortexMPassManager(None, target_config=target).transform_for_annotation(graph)
    graph = prepare_pt2e(graph, CortexMQuantizer())
    for x in calibration:
        graph(_channels_last(x))
    graph = convert_pt2e(graph)
    quantize = torch.ops.quantized_decomposed.quantize_per_tensor.default
    node = next(n for n in graph.graph.nodes if n.target == quantize)
    scale, zero_point = float(node.args[1]), int(node.args[2])
    edge = to_edge(torch.export.export(graph, example),
                   compile_config=cortex_m_edge_compile_config())
    edge = edge.transform(passes=[QuantizeInputs(edge, [0]), QuantizeOutputs(edge, [0])])
    edge._edge_programs["forward"] = CortexMPassManager(
        edge.exported_program(), target_config=target).transform()
    config = ExecutorchBackendConfig(constant_tensor_alignment=alignment)
    pte = edge.to_executorch(config).buffer
    inputs = np.stack([
        torch.clamp(torch.round(x / scale) + zero_point, -128, 127).to(torch.int8)
        .permute(0, 2, 3, 1).contiguous().numpy()[0] for x in samples])
    return pte, inputs, scale, zero_point


def _run(runner: list, pte: bytes, inputs: np.ndarray) -> np.ndarray:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "model.pte").write_bytes(pte)
        inputs.tofile(tmp / "input.bin")
        subprocess.run([*runner, str(tmp / "model.pte"), str(tmp / "input.bin"),
                        str(tmp / "output.bin")], check=True, capture_output=True)
        out = np.fromfile(tmp / "output.bin", np.int8)
    if out.size % len(inputs):
        raise RuntimeError(f"runner wrote {out.size} bytes for {len(inputs)} inputs")
    return out.reshape(len(inputs), -1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runner", type=Path, help="host runner for the reference outputs")
    parser.add_argument("--mve-runner", help="runner command for MVE programs (split on spaces)")
    parser.add_argument("models", nargs="*", default=sorted(MODELS))
    args = parser.parse_args()
    logging.disable(logging.WARNING)
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    runners = {cpu: [str(args.runner)] for cpu in RUNS if args.runner}
    if args.mve_runner:
        runners[MVE_RUN] = args.mve_runner.split()
    for name in args.models:
        reference = None
        for cpu in MODELS[name][2]:
            pte, inputs, scale, zero_point = _lower(name, cpu)
            stem = f"{name}_{cpu}"
            (OUT / f"{stem}.pte").write_bytes(pte)
            record = {"inputs": inputs, "input_scale": np.float32(scale),
                      "input_zero_point": np.int32(zero_point)}
            runner = runners.get(cpu)
            if runner:
                outputs = _run(runner, pte, inputs)
                if reference is not None and not np.array_equal(outputs, reference):
                    raise RuntimeError(f"{stem}: outputs differ from the other class")
                reference = record["output_bytes"] = outputs
            np.savez(OUT / f"{stem}.npz", **record)
            print(f"{stem}.pte: {len(pte)} bytes"
                  + (", outputs recorded" if "output_bytes" in record else ""))


if __name__ == "__main__":
    main()
