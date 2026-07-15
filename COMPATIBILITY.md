# TiGrIS Core Compatibility

Core releases are paired: use a compiler and runtime from the same row. The
runtime validates the plan revision encoded in every `.tgrs` file before it
exposes plan data to the executor.

| Compiler | Runtime | Contract gate | Status |
| --- | --- | --- | --- |
| `v0.4.0` | [`v0.4.0`](https://github.com/raws-labs/tigris-runtime/releases/tag/v0.4.0) | `python scripts/crossrepo_contract.py --runtime ../tigris-runtime` | Supported |

The v0.4 runtime accepts plan schema revisions 2, 3, and 4. The v0.4 compiler
emits revision 4, so newly compiled plans require a v0.4 runtime even though
that runtime remains backward-compatible with older plans.

The release gate differentially checks generated plans against ONNX Runtime
using the reference float and int8 dispatchers. It covers constant and residual
elementwise paths, dilation, depthwise convolution, standalone tiling,
streamable tiled chains, LZ4-compressed weights, XIP metadata, and quantized
Conv/AveragePool. It also covers observable terminal Transpose output semantics,
intentional compiler rejection, and incompatible-plan rejection by the runtime.

Accelerated CMSIS-NN and ESP-NN execution has separate target-specific routing
and parity tests; it is not represented by the host reference contract gate.
