# TiGrIS Core Compatibility

Core releases are paired: use a compiler and runtime from the same row. The
runtime validates the plan revision encoded in every `.tgrs` file before it
exposes plan data to the executor.

The machine-readable source of this table is
[`compatibility.json`](compatibility.json). Every compiler release entry names
the exact compiler commit, runtime tag and commit, emitted plan schema, and
runtime schema acceptance set. Release automation and downstream consumers
must validate that manifest rather than infer compatibility from matching
version numbers.

| Compiler | Runtime | Plan schema | Contract gate | Status |
| --- | --- | ---: | --- | --- |
| `v0.4.0` (`d61c2bc`) | [`v0.4.0`](https://github.com/raws-labs/tigris-runtime/releases/tag/v0.4.0) (`73bd717`) | emits 4; accepts 2-4 | `python scripts/crossrepo_contract.py --runtime ../tigris-runtime` | Supported |

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

## Repository and schema ownership

The compiler and runtime remain separate repositories. Combining them would
couple the dependency-free C runtime's release and integration surface to the
Python toolchain without eliminating the need for a stable on-flash contract.

The compiler repository owns the canonical wire definitions and publishes a
generated, versioned schema artifact such as
[`tigris-plan-v4.json`](src/tigris/schema/tigris-plan-v4.json). The compiler's
writer, reader, sizing code, and artifact generator use the same definitions.
The runtime vendors its C representation so it has no build-time or run-time
compiler dependency. Cross-repository CI compiles real plans, executes them
with the sibling runtime, and checks the generated schema artifact and runtime
acceptance set on every integration change.

`develop` is the coordinated integration branch in both public repositories.
Same-named branches may be selected for schema changes, but a release is not a
supported pair until `compatibility.json` records exact tagged commits and the
cross-repository gate passes them. Benchmarks record exact commits separately;
they never treat a mutable branch name as provenance.
