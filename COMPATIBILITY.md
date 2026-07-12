# TiGrIS Core Compatibility

Core releases are paired: use a compiler and runtime from the same row. The
runtime validates the plan revision encoded in every `.tgrs` file before it
exposes plan data to the executor.

| Compiler | Runtime | Contract gate | Status |
| --- | --- | --- | --- |
| `v0.3.2` | [`v0.3.2`](https://github.com/raws-labs/tigris-runtime/releases/tag/v0.3.2) | `python scripts/crossrepo_contract.py --runtime ../tigris-runtime` | Supported |

The release gate differentially checks generated plans against ONNX Runtime
using the reference float and int8 dispatchers. It covers constant and residual
elementwise paths, dilation, depthwise convolution, standalone tiling,
streamable tiled chains, LZ4-compressed weights, XIP metadata, and quantized
Conv/AveragePool. It also checks intentional compiler rejection and an
incompatible-plan rejection by the runtime.

Accelerated CMSIS-NN and ESP-NN execution has separate target-specific routing
and parity tests; it is not represented by the host reference contract gate.
