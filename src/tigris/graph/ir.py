"""Core dataclasses for the tigris graph IR."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from tigris import TILE_AXIS_NONE

DTYPE_BYTES: dict[int, int] = {
    1: 4,   # FLOAT
    2: 1,   # UINT8
    3: 1,   # INT8
    5: 2,   # INT16
    6: 4,   # INT32
    7: 8,   # INT64
    9: 1,   # BOOL
    10: 2,  # FLOAT16
    11: 8,  # DOUBLE
    12: 4,  # UINT32
    13: 8,  # UINT64
    16: 2,  # BFLOAT16
}

def _elem_size(dtype: int) -> int:
    return DTYPE_BYTES.get(dtype, 4)


@dataclass
class QuantParam:
    """Per-tensor or per-channel quantization parameters."""

    scale: np.ndarray       # float32, shape [1] or [num_channels]
    zero_point: np.ndarray  # int8 or int32, shape [1] or [num_channels]
    axis: int = -1          # channel axis for per-channel (-1 = per-tensor)


@dataclass
class TensorInfo:
    """Static metadata for a single tensor in the graph."""

    name: str
    shape: tuple[int, ...]
    dtype: int  # ONNX TensorProto.DataType enum value
    is_constant: bool = False  # weights / initializers
    quant: QuantParam | None = None

    @property
    def elem_size(self) -> int:
        return _elem_size(self.dtype)

    @property
    def num_elements(self) -> int:
        if not self.shape:
            return 1
        return int(np.prod(self.shape))

    @property
    def size_bytes(self) -> int:
        return self.num_elements * self.elem_size


@dataclass
class TensorLifetime:
    """Activation lifetime: when it is produced and when it is last consumed."""

    tensor_name: str
    birth_step: int   # step index of producing op (-1 for model inputs)
    death_step: int   # step index of last consuming op (len(ops) for model outputs)
    size_bytes: int


@dataclass
class OpNode:
    """A single operator in the execution graph."""

    name: str
    op_type: str
    inputs: list[str]   # tensor names
    outputs: list[str]  # tensor names
    attrs: dict[str, Any] = field(default_factory=dict)
    step: int = -1      # assigned after topological sort
    stage: int = -1     # assigned after partitioning


@dataclass
class MemorySnapshot:
    """Memory state at a single execution step."""

    step: int
    live_bytes: int
    live_tensors: list[str]


@dataclass
class TilePlan:
    """Tiling analysis result for a single stage."""

    tileable: bool
    axis: int = TILE_AXIS_NONE
    tile_height: int = 0
    num_tiles: int = 0
    halo: int = 0
    receptive_field: int = 1
    original_height: int = 0
    tiled_peak_bytes: int = 0
    overhead_bytes: int = 0
    untileable_ops: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class Stage:
    """A partition of the graph that fits within the SRAM budget."""

    stage_id: int
    op_indices: list[int]              # step indices of ops in this stage
    input_tensors: list[str] = field(default_factory=list)   # must be loaded from slow mem
    output_tensors: list[str] = field(default_factory=list)  # must be spilled after stage
    peak_bytes: int = 0
    warnings: list[str] = field(default_factory=list)
    tile_plan: TilePlan | None = None

    # Chain fields - set by detect_chains() in partition_spatial
    chain_id: int = 0xFFFF   # stage index of chain head, or 0xFFFF = standalone
    chain_len: int = 0       # number of stages in chain (0 = not in chain)
    chain_tile_h: int = 0    # output tile height for last stage (set on head only)


@dataclass
class AnalyzedGraph:
    """Central object enriched by each pipeline stage."""

    # Populated by loader
    model_name: str = ""
    ops: list[OpNode] = field(default_factory=list)
    tensors: dict[str, TensorInfo] = field(default_factory=dict)
    model_inputs: list[str] = field(default_factory=list)
    model_outputs: list[str] = field(default_factory=list)

    # Populated by loader - raw weight arrays keyed by initializer name
    weight_data: dict[str, np.ndarray] = field(default_factory=dict)

    # Populated by lifetime analysis
    lifetimes: dict[str, TensorLifetime] = field(default_factory=dict)

    # Populated by memory analysis
    timeline: list[MemorySnapshot] = field(default_factory=list)
    peak_memory_bytes: int = 0

    # Populated by partitioner
    stages: list[Stage] = field(default_factory=list)
    mem_budget: int = 0  # primary (fastest) memory pool size in bytes

    # Bytes deliberately held outside ``mem_budget`` by the deployment
    # harness.  ``mem_budget`` is consequently always the activation arena
    # represented in the binary plan.  The compiler currently uses this for
    # compressed-weight blocks; it also makes an explicit target scratch
    # reservation possible without making activation feasibility ambiguous.
    fast_memory_reserve_bytes: int = 0

    # Physical allocation alignment used by the deployment memory model.
    # 32 bytes is conservative for the currently supported Cortex-M, ESP32-S3,
    # and host SIMD runtimes; target runtimes that need less simply have spare
    # arena capacity.  The planner must never sum raw tensor byte counts when
    # the bump allocator rounds every live allocation up.
    tensor_alignment: int = 32

    # Quantization
    is_quantized: bool = False
