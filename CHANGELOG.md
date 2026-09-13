# Changelog

All notable changes to the TiGrIS compiler (`tigris-ml`) are recorded here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Releases up to and including `v0.6.0` predate this file; their history is in the
Git tags and the GitHub releases page.

## [Unreleased]

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
