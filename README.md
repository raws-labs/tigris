# TiGrIS

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/tigris-ml)](https://pypi.org/project/tigris-ml/)
[![Docs](https://img.shields.io/badge/docs-tigris--ml.dev-green)](https://tigris-ml.dev/docs)

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
╭──────────────────────── TiGrIS - mobilenetv2 ────────────────────────╮
│ Operators            65                                              │
│ Peak memory (naive)  4.59 MiB                                        │
│ Largest tensor       1x96x112x112 (4.59 MiB)                         │
╰──────────────────────────────────────────────────────────────────────╯
╭──────────────────────────────── SRAM ────────────────────────────────╮
│ Budget              256.00 KiB                                       │
│ Scheduled peak      254.62 KiB (5.4% of naive peak)                  │
│ Stages              42                                               │
│ Need tiling         31 of 42 stages                                  │
╰────────────────  PASS - tiling resolves all stages  ─────────────────╯
```

The naive peak is 4.59 MiB. TiGrIS schedules it into 256 KiB through temporal partitioning and spatial tiling. `analyze` runs on your laptop; no hardware required.

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

The `.tgrs` plan is target-agnostic: it is the same file whether you run it on an ESP32, a Cortex-M, or a POSIX host for testing. The choice of kernel backend happens at `codegen` time and decides which kernel library the generated C calls into.

Several kernel backends are available (portable C99, ESP32 family, Cortex-M family); see [tigris-runtime](https://github.com/raws-labs/tigris-runtime) for the current list. Switching between them is a `--backend` flag, not a rewrite.

## What you get

`tigris compile` writes a single `.tgrs` file that contains the operator schedule, tile parameters, quantization tables, and the weights. This file goes on flash at deployment time.

`tigris codegen` produces a small C harness that locates the plan on flash at runtime and hands it to the runtime:

- declarations for the input/output buffers and the arena
- a target entry point (`app_main()` for ESP-IDF, `main()` for POSIX/Cortex-M examples) that sets up memory and calls the runtime
- backend-specific glue for finding the plan: partition mmap on ESP-IDF, an `extern` flash symbol on Cortex-M, a file path on POSIX

Link the harness against [tigris-runtime](https://github.com/raws-labs/tigris-runtime) and your chosen kernel library, flash the `.tgrs` alongside the firmware, and you have a working inference binary.

## Further reading

- [Getting started](https://tigris-ml.dev/docs): installation, first compile, deploying to ESP32
- [Introducing TiGrIS](https://tigris-ml.dev/blog/introducing-tigris): design, benchmarks, how tiling works
- [CLI reference](https://tigris-ml.dev/docs/cli): every flag, every subcommand

## Maintainer

TiGrIS is maintained by RAWS Labs. For applied embedded-ML engineering or collaboration, see [raws.at](https://www.raws.at/).

## Development

```bash
git clone https://github.com/raws-labs/tigris
cd tigris
pip install -e ".[dev]"
pytest
```
