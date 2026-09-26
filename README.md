# TiGrIS

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/tigris-ml)](https://pypi.org/project/tigris-ml/)
[![Docs](https://img.shields.io/badge/docs-tigris--ml.dev-green)](https://tigris-ml.dev/getting-started/quickstart/)

**Tiled Graph Inference Scheduler.** An ahead-of-time compiler that tiles ML models to fit embedded devices with hard memory budgets.

Give it an ONNX model and a memory budget. It partitions the compute graph into stages, tiles spatial operations, and emits a flat binary plan that the [tigris-runtime](https://github.com/raws-labs/tigris-runtime) executes with zero dynamic allocation.

## The problem

On an embedded device with a few hundred KB of SRAM, most interesting models simply don't fit. The usual answer is to shrink the model: quantize harder, prune, pick a smaller architecture, and hope the accuracy hit is acceptable.

TiGrIS takes the other approach. It keeps the model you trained and rearranges the *computation* so that only a small working set lives in SRAM at any moment. Weights and intermediate spills go to flash or PSRAM. What comes out is a binary plan that the runtime executes as a flat sequence of kernel calls, with no interpreter, no tensor allocator, and no dynamic memory at all.

## Installation

`pip install tigris-ml` selects a wheel for the host. Native wheels include the
portable reference runtime for Linux x86-64 and ARM64 with glibc >= 2.28,
macOS x86-64 and ARM64, and Windows x86-64. The Linux wheels use
`manylinux_2_28`; older glibc, musllinux, Windows ARM, and 32-bit ARM Linux
install the pure Python wheel instead. A source distribution is also available.

The pure wheel supports inspection, compilation, analysis, code generation, and zoo downloads.
`run` reports "Bundled host runtime is unavailable" without the native
library. Neither installation nor execution downloads a runtime separately.

On platforms without a native wheel, build the host library from the matching
[tigris-runtime release](https://github.com/raws-labs/tigris-runtime/releases):
run `cmake -S . -B build-host -DTIGRIS_BUILD_HOST=ON`, then
`cmake --build build-host --target tigris_host`, and set `TIGRIS_HOST_LIBRARY`
to the resulting shared library file. `tigris --version` and `tigris run --json`
report the runtime's version and origin; an invalid override fails without falling back.

## Quick start

```bash
pip install tigris-ml

# Will this model fit in 256KB SRAM + 16MB flash?
tigris analyze mobilenetv2.onnx -m 256K -f 16M
```

```text
warning: input axis 0 (batch_size) is unset; using 1 (--input-shape overrides)
╭──────────────────────── TiGrIS - mobilenetv2 ────────────────────────╮
│ Operators            65                                              │
│ Tensors              244 (66 activations)                            │
│ Peak memory (naive)  5.74 MiB                                        │
│ Largest tensor       1x96x112x112 (4.59 MiB)                         │
│ Dtype                float32                                         │
│ Input                input 1x3x224x224 float32                       │
│ Output               output 1x1000 float32                           │
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

## Precompiled models

The model zoo hosts precompiled plans with runtime requirements, input/output
conventions, evaluation results, and per-model licenses. Public downloads need no
account. List available builds before choosing a model:

```bash
tigris zoo list --category classification
tigris zoo list --runtime 0.9.1 --backend reference -m 256K
tigris zoo fetch MODEL -o downloaded-model
tigris inspect downloaded-model/model.tgrs
tigris codegen downloaded-model/model.tgrs --format core -o model.c
```

Filters are optional. The newest published matching build wins. Runtime ranges
include both endpoints; a null maximum means no known upper compatibility bound.
Tested releases are reported separately. Matching a range does not claim that
every release in it has been tested. `-m` limits the fast arena, a second
`-m` limits the slow arena, and `-f` limits plan bytes. These are not total
application RAM or flash limits. Unspecified resources remain unconstrained.
The runtime for deployment on a device is supplied separately.

Use `fetch --artifact ID` to pin a build. Withdrawn builds are excluded from
automatic selection but remain explicitly retrievable with a warning. Downloads
verify file hashes and never replace an existing destination. `manifest.json`
preserves the original build metadata. `download.json` records the current
runtime constraints, tested releases, and pinned catalog revision; use it for
dependency integration. Catalog updates do not change artifact publication dates
or rebuild models. `tigris zoo --offline ...` uses the HF cache;
`tigris zoo --catalog catalog.json ...` reads a local zoo snapshot.

## Inspect a model

```bash
tigris inspect model.onnx
tigris inspect model.tgrs -v
tigris inspect model.tgrs --json
```

`inspect` detects the format from file contents. It reads declared ONNX inputs,
outputs, operators, and initializers, or compiled plan interfaces, stages, tiling,
quantization, and memory records. `-v` adds details; `--json` emits complete
metadata with an `inspection_version` field and exact byte counts, without weight
values. Inspection is offline: it does not compile, execute, infer shapes, or load
external ONNX tensor data. Recorded plan memory is not measured runtime peak or
total application RAM. Plan schemas do not imply runtime version requirements.
Inspection does not validate kernel numerics. `run` checks the portable reference
path; neither command validates ESP-NN or CMSIS-NN numerics or on-device latency.

## Run a compiled model

Platform wheels include the reference C runtime. Run preprocessed tensors in
the stored axis order and declared interface dtype reported by `inspect`:

```bash
tigris run downloaded-model/model.tgrs --input downloaded-model/example-input.bin --output prediction.bin
```

Compare `prediction.bin` with `example-output.bin` numerically; the reference
comes from ONNX Runtime, so float results differ in the last bits.

```bash
tigris run model.tgrs --input input.npy --output prediction.npy
tigris --version
```

For multiple inputs, repeat `--input NAME=FILE`. Use `.npz` for multiple named
outputs. Raw `.bin` files contain contiguous little-endian elements; `.npy`
inputs retain their declared shape and dtype. Output files must not exist.
`--json` reports execution metadata and arena peaks. These peaks exclude the
plan, executor workspace, and Python process memory. Host execution uses
only the portable float32 reference and int8 reference kernels. It checks plan
execution on that path, not ESP-NN or CMSIS-NN numerics or on-device latency.
It does not preprocess photos or CSV files.

```python
import numpy as np
from tigris.runtime import Session

with Session("model.tgrs") as session:
    outputs = session.run({"input": np.load("input.npy", allow_pickle=False)})
    print(session.runtime_version, session.memory)
```

Compiler and runtime releases share a version identifying the tested pair.
Execution checks matching zoo dependency records when present and lets the C
loader validate the plan. Schema versions, accepted schema ranges, and the host
ABI retain their own compatibility meanings.
Downloading and generating deployment code remain independent of the host runtime.
See [Contributing](CONTRIBUTING.md) for native-library staging and wheel builds.

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
