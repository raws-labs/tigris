# TiGrIS

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/tigris-ml)](https://pypi.org/project/tigris-ml/)
[![Docs](https://img.shields.io/badge/docs-tigris--ml.dev-green)](https://tigris-ml.dev/getting-started/quickstart/)

**Tiled Graph Inference Scheduler.** An ahead-of-time compiler that tiles ML models to fit embedded devices with hard memory budgets.

Give it an ONNX model and a memory budget. It partitions the compute graph into stages, tiles spatial operations, and emits a flat binary plan that the [tigris-runtime](https://github.com/raws-labs/tigris-runtime) executes with zero dynamic allocation.

## The problem

On an embedded device with a few hundred KB of SRAM, most interesting models simply don't fit. The usual answer is to shrink the model: quantize harder, prune, pick a smaller architecture, and hope the accuracy hit is acceptable.

TiGrIS takes the other approach. It keeps the model you trained and rearranges the *computation* so that only a small working set lives in SRAM at any moment. Weights and intermediate spills go to flash or PSRAM. What comes out is a binary plan that the runtime executes as a flat sequence of kernel calls, with no interpreter, no tensor allocator, and no dynamic memory at all.

## Quick start

```bash
pip install tigris-ml

# Will this model fit in 256KB SRAM + 16MB flash?
tigris analyze mobilenetv2.onnx -m 256K -f 16M
```

```text
warning: input axis 0 (batch_size) has no fixed size; using 1
  pass --input-shape NAME:1x3x224x224 to compile for another shape
╭──────────────────────── TiGrIS - mobilenetv2 ────────────────────────╮
│ Operators            65                                              │
│ Tensors              244 (66 activations)                            │
│ Peak memory (naive)  5.74 MiB                                        │
│ Largest tensor       1x96x112x112 (4.59 MiB)                         │
│ Dtype                float32                                         │
╰──────────────────────────────────────────────────────────────────────╯
╭──────────────────────────────── SRAM ────────────────────────────────╮
│ Budget              256.00 KiB                                       │
│ Scheduled peak      252.00 KiB (4.3% of naive peak)                  │
│ Stages              58                                               │
│ Spill / reload I/O  26.21 MiB / 27.55 MiB                            │
│                                                                      │
│ Need tiling         47 of 58 stages                                  │
│   tileable          11 (138 tiles, max halo 2)                       │
╰────────────────  PASS - tiling resolves all stages  ─────────────────╯
╭─────────────────────────────── Flash ────────────────────────────────╮
│ Budget            16.00 MiB                                          │
│ Weight data       13.30 MiB                                          │
│ Plan overhead      0.01 MiB                                          │
│ Plan (est.)       13.31 MiB                                          │
│ Plan INT8 (est.)   3.34 MiB                                          │
╰─────────────────────────  PASS - plan fits  ─────────────────────────╯
```

The naive peak is 5.74 MiB. TiGrIS schedules it into 256 KiB through temporal partitioning and spatial tiling. `analyze` runs on your laptop; no hardware required.

That model is a stock export with a free batch dimension. TiGrIS binds a dimension the model leaves open to 1 and says so; pass `--input-shape input:4x3x224x224` to compile for a different one.

## From ONNX to embedded

Three steps take a model from ONNX to a C file you can drop into your firmware project:

```bash
# 1. Analyze feasibility against a memory budget
tigris analyze model.onnx -m 256K -f 16M

# 2. Compile to a binary plan (weights read-in-place from flash)
tigris compile model.onnx -m 256K -f 16M --xip -o model.tgrs

# 3. Generate a backend-specific C harness for your target
tigris codegen model.tgrs --backend esp-nn -o model.c
```

The `.tgrs` plan is target-agnostic: the same file runs on an ESP32, a Cortex-M, or a POSIX host. The kernel backend is chosen at `codegen` time and decides which kernel library the generated C calls into. The [operator and backend matrix](https://tigris-ml.dev/runtime/operator-support/) shows which operators are native, which fall back, and which are rejected.

## What you get

`tigris compile` writes a single `.tgrs` file holding the operator schedule, tile parameters, quantization tables, and the weights.

`tigris codegen` produces a C harness that loads the plan and hands it to the runtime: buffer and arena declarations, a target entry point, and the glue for reaching the plan bytes. `--format app` emits a standalone program. `--format core` emits a source and header for firmware that already owns its entry point, arenas, and input source, so several generated cores can coexist in one binary. The [`codegen` reference](https://tigris-ml.dev/toolchain/codegen/) documents the flags.

Link the harness against [tigris-runtime](https://github.com/raws-labs/tigris-runtime) and your kernel library, and you have a working inference binary.

## Further reading

- [Getting started](https://tigris-ml.dev/getting-started/quickstart/): installation, first compile, deploying to ESP32
- [Core compatibility data](compatibility.json): exact compiler/runtime releases and plan schemas
- [Introducing TiGrIS](https://tigris-ml.dev/blog/introducing-tigris): design, benchmarks, how tiling works
- [CLI reference](https://tigris-ml.dev/toolchain/analyze/): every flag, every subcommand

## Maintainer

TiGrIS is maintained by RAWS Labs. For applied embedded-ML engineering or collaboration, see [raws.at](https://www.raws.at/).

## Development

```bash
git clone https://github.com/raws-labs/tigris
cd tigris
pip install -e ".[dev]"
pytest
```
