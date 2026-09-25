# Changelog

All notable changes to the TiGrIS compiler (`tigris-ml`) are recorded here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Releases up to and including `v0.6.0` predate this file; their history is in the
Git tags and the GitHub releases page.

## [Unreleased]

## [0.9.0] - 2026-09-25

### Added

- Plan schema 9. Schema 8 declares every operator-attribute kind this
  generation needs, so new operators land without further bumps; schema 9 adds
  row bands for matrix pipelines. Plans need runtime v0.10.0 or later.
- `tigris zoo list` and `tigris zoo fetch` for precompiled plans. Listing filters
  by model, category, runtime release, backend and memory limits; fetching
  verifies every file hash and records the runtime constraints in
  `download.json`.
- Operators: `LayerNormalization`, `Erf`, `HardSwish`, `Sub`, `MaxPool` to a
  global maximum, `Split`, and bilinear `Resize` as `ResizeLinear` with the
  coordinate convention taken from the graph's opset.
- `Add` and `Mul` accept a second operand with one value per channel.
- A graph input behind `DequantizeLinear` and an output ahead of
  `QuantizeLinear` compile to int8 model boundaries, uint8 included.
- More stages tile: global reductions along their input, layout conversions and
  any transpose whose stored permutation allows it, rank-2 matrix pipelines along
  their rows, and `Resize` as the spatial op of a height-tiled stage.
- An `AveragePool` over the whole map becomes `GlobalAveragePool`.

### Changed

- A stage is costed by the bytes the executor holds for it, not by the
  graph-wide live set, so the partitioner no longer cuts stages it does not need
  to.
- A detected chain too large for the fast budget is split into chains that fit
  instead of leaving every stage standalone.
- `codegen` emits a static assertion per runtime table limit, and `compile`
  warns when a plan needs more than the default limits.

### Fixed

- `Gemm` ignored `alpha`, `beta` and `transA` and returned a wrong result.
- An int8 `Conv1D` was requantized with the raw output scale.
- A `Div` by a constant shared between operators inverted the constant once per
  consumer.
- `Squeeze` and `Unsqueeze` skipped the reshape order check, so an implied
  layout change went unhandled.
- A `Resize` in an opset-10 graph was read with the opset-11 half-pixel default.
- 2D tiles are sized as output tiles by the runtime's own byte count; strided
  stages previously got tiles the runtime refused.
- Chain geometry takes a convolution's kernel from its weight when
  `kernel_shape` is absent; such a chain was sized as 1x1.
- A stage output written before a spatial op no longer enters a tile.

## [0.8.0] - 2026-09-14

### Added

- Plan schema 7: a tensor records whether its axes are stored in the order the
  model states them rather than channels-last. Without it the order a boundary
  arrives in was not recoverable from the plan, since an output written by a
  terminal `Transpose` keeps ONNX order while every other output does not.
- Tensor layout is tracked through the graph and converted with an explicit
  `Transpose` where a producer and a consumer disagree. Model inputs and outputs
  keep the channels-last convention, so the interface is unchanged.
- `MatMul` compiles in every form: a constant weight at rank 2 or above
  collapses into the fully-connected kernel, and two activations reach a
  dedicated kernel on both the float and int8 paths.
- `Softmax` over the last axis, which is what an exporter writes by default. The
  axis a kernel can reduce follows from the tensor's layout rather than its rank.
- `Softmax` stages tile, so a tensor larger than the fast pool no longer has to
  fit whole.
- `Dropout` and `Identity` are removed, and `Squeeze` and `Unsqueeze` become
  `Reshape` where the element order survives the change.
- A float per-channel constant `Add` folds into its producer's bias instead of
  being rejected for broadcasting the `Add` kernel cannot do.

### Changed

- A model in the ONNX QOperator format is now named as such, with the re-export
  flag that fixes it, instead of being reported as an unnamed unsupported
  operator alongside a contradictory dtype complaint.
- Shapes left free by an exporter are resolved rather than rejected, and a
  `Reshape`'s shape initializer is no longer mistaken for a weight.

### Fixed

- `Softmax` on a graph whose axis maps to the runtime's final dimension is no
  longer rejected for the rank-based rule that only held for spatial tensors.

## [0.7.0] - 2026-09-13

### Added

- Combined memory-tier syntax: `-m 256K+4M` expands to separate fast and slow
  pools, as sugar for repeated `-m 256K -m 4M`.
- Two-dimensional (height and width) spatial tiling for stages that do not fit
  the fast pool along a single axis, with a 2D receptive-field model and a tile
  solver that sizes tiles to the runtime working set.
- Line-buffered recompute chains: consecutive tiled stages keep their
  intermediate tensors in the fast pool as tiles instead of allocating them in
  full, and are marked and serialized as such.
- `ConvTranspose` support (capability, int8 requantization, and a dedicated 2D
  tile solver over the expanded output extent).
- Co-tiled skip connections: a `Concat` or `Add` that joins a skip tensor at the
  same resolution as its stage is tiled together with that stage, so
  encoder/decoder models tile rather than falling back to a full-size allocation.
- Native accelerator kernels (CMSIS-NN, ESP-NN) now execute tiled and
  line-buffered stages directly instead of falling back to the reference kernel.

### Changed

- `plan`, `simulate`, and `compile` now fail closed when a second `-m` tier is
  supplied but is not greater than zero. This was previously a silent no-op.
  `analyze` continues to treat the slow tier as display-only and still exits 0.
- `compile` now enforces `-f` / `--flash`: it fails when the emitted plan exceeds
  the given flash budget, rather than only reporting the size.
- `compile` fails closed when a stage overflows the slow-memory pool, and the
  slow-pool peak is now computed from concurrent tensor liveness so a long-lived
  skip tensor is no longer undercounted.
- `compile` fails closed on operators that cannot be routed to a supported
  kernel, including `ConvTranspose` configurations outside the supported subset.
