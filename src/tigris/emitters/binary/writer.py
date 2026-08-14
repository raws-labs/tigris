"""Binary plan emitter - serialization (pool helpers + emit)."""

import math
import struct
from pathlib import Path

import numpy as np

from tigris import (
    SCHEMA_VERSION,
    TILE_AXIS_HEIGHT_OR_LENGTH,
    TILE_AXIS_NONE,
)
from tigris.graph.ir import AnalyzedGraph, OpNode

from .defs import (
    ACT_NONE,
    ACT_RELU,
    ACT_RELU6,
    COMPRESS_LZ4,
    COMPRESS_NONE,
    FLAG_XIP,
    HEADER_SIZE,
    HEADER_STRUCT,
    INDEX_STRUCT,
    MAGIC,
    NO_QUANT_PARAM,
    NO_WEIGHT,
    OP_ATTR_TRANSPOSE_PERM,
    OP_ACTIVATION_STRUCT,
    OP_ATTRIBUTE_SECTION_HEADER_STRUCT,
    OP_ATTRIBUTE_STRUCT,
    OP_PREFIX_STRUCT,
    OP_TYPE_MAP,
    OP_WEIGHT_BIAS_STRUCT,
    PLAN_SECTION_ALIGNMENT,
    QUANT_DATA_STRUCT,
    QUANT_PARAM_STRUCT,
    QUANT_SECTION_HEADER_STRUCT,
    SEC_INDEX_POOL,
    SEC_OP_ATTRIBUTES,
    SEC_OPS,
    SEC_QUANT_PARAMS,
    SEC_SHAPE_POOL,
    SEC_STAGES,
    SEC_STRINGS,
    SEC_TENSORS,
    SEC_TILE_PLANS,
    SEC_WEIGHT_BLOCKS,
    SEC_WEIGHTS,
    SECTION_ENTRY_SIZE,
    SECTION_ENTRY_STRUCT,
    SPATIAL_ATTRS_STRUCT,
    STAGE_FLAG_LINE_BUFFERED,
    STAGE_SIZE,
    STAGE_STRUCT,
    STAGE_TILE_PLAN_INDEX_OFFSET,
    TENSOR_FLAG_CONSTANT,
    TENSOR_FLAG_MODEL_INPUT,
    TENSOR_FLAG_MODEL_OUTPUT,
    TENSOR_STRUCT,
    TILE_PLAN_STRUCT,
    WEIGHT_BLOCK_SECTION_HEADER_STRUCT,
    WEIGHT_BLOCK_STRUCT,
    WEIGHT_ENTRY_SIZE,
    WEIGHT_ENTRY_STRUCT,
)


# Weight blobs are padded so every weight/bias starts on this byte boundary.
# The optimized CMSIS-NN Cortex-M kernels read weights/bias with LDRD (8-byte)
# and wide SIMD loads, which fault on unaligned data; 16 also covers Helium MVE.
WEIGHT_ALIGN = PLAN_SECTION_ALIGNMENT

# These are execution limits of the current runtime, rather than wire-format
# limits. Keep them explicit here so the compiler fails before producing a plan
# the loader will inevitably reject.
RUNTIME_MAX_TENSORS = 512
RUNTIME_MAX_STAGE_INPUTS = 16
RUNTIME_MAX_STAGE_OUTPUTS = 16
RUNTIME_MAX_CHAIN_STAGES = 16


def _require_uint(value: int, bits: int, field: str) -> None:
    """Reject a value that cannot be represented by a plan field."""
    if not 0 <= value < (1 << bits):
        raise ValueError(f"{field}={value} exceeds uint{bits} plan-format limit")


# Helper: String Table


class _StringTable:
    """Deduplicating null-terminated string pool."""

    def __init__(self) -> None:
        self._map: dict[str, int] = {}
        self._buf = bytearray()

    def add(self, s: str) -> int:
        """Add string, return byte offset. Empty string -> offset 0."""
        if not s:
            # ensure we have an empty string at offset 0
            if 0 not in self._map.values() or not self._buf:
                if not self._buf:
                    self._map[""] = 0
                    self._buf.append(0)
            return self._map.get("", 0)
        if s in self._map:
            return self._map[s]
        offset = len(self._buf)
        self._map[s] = offset
        self._buf.extend(s.encode("utf-8"))
        self._buf.append(0)
        return offset

    def data(self) -> bytes:
        if not self._buf:
            self._buf.append(0)  # at least one null byte
        return bytes(self._buf)


# Helper: Index Pool


class _IndexPool:
    """Shared uint16 index array pool."""

    def __init__(self) -> None:
        self._items: list[int] = []

    def add(self, indices: list[int]) -> tuple[int, int]:
        """Add a list of indices, return (offset, count)."""
        offset = len(self._items)
        count = len(indices)
        _require_uint(offset, 16, "index-pool offset")
        _require_uint(count, 16, "index-pool count")
        for index in indices:
            _require_uint(index, 16, "index-pool value")
        self._items.extend(indices)
        return offset, count

    def data(self) -> bytes:
        return struct.pack(f"<{len(self._items)}H", *self._items) if self._items else b""


# Helper: Shape Pool


class _ShapePool:
    """Shared int32 shape dimension pool."""

    def __init__(self) -> None:
        self._items: list[int] = []

    def add(self, shape: tuple[int, ...]) -> tuple[int, int]:
        """Add shape dims, return (offset, ndim)."""
        offset = len(self._items)
        _require_uint(offset, 16, "shape-pool offset")
        _require_uint(len(shape), 8, "tensor rank")
        self._items.extend(int(d) for d in shape)
        return offset, len(shape)

    def data(self) -> bytes:
        return struct.pack(f"<{len(self._items)}i", *self._items) if self._items else b""


# Spatial attribute helpers


def _pack_spatial_attrs(
    op: OpNode, weight_data: dict[str, np.ndarray] | None = None
) -> bytes:
    """Pack the 18-byte spatial-attribute layout.

    Layout:
        kernel_h(u8) kernel_w(u8) stride_h(u8) stride_w(u8)
        pad_top(u16) pad_bottom(u16) pad_left(u16) pad_right(u16)
        dilation_h(u16) dilation_w(u16) group(u16 LE)
    Total: 18 bytes. pad/dilation are u16 (was u8) so deep dilated convs
    (dilation/pad up to 65535, e.g. TCN/WaveNet) no longer overflow.
    """
    attrs = op.attrs
    ks = attrs.get("kernel_shape", [])
    st = attrs.get("strides", [])
    pa = attrs.get("pads", [])
    di = attrs.get("dilations", [])
    gr = int(attrs.get("group", 1))

    is_1d = len(ks) == 1
    is_spatial = op.op_type in {
        "Conv",
        "Conv1D",
        "DepthwiseConv",
        "ConvTranspose",
        "MaxPool",
        "AveragePool",
    }
    if not ks and op.op_type in {
        "Conv",
        "Conv1D",
        "DepthwiseConv",
        "ConvTranspose",
    }:
        # ONNX Conv kernel_shape is optional: infer it from the first constant
        # weight operand rather than serializing a zero-sized kernel.
        for input_name in op.inputs:
            weights = (weight_data or {}).get(input_name)
            if weights is None:
                continue
            if weights.ndim >= 4:
                ks = [int(weights.shape[-2]), int(weights.shape[-1])]
            elif weights.ndim == 3:
                ks = [int(weights.shape[-1])]
            break
    if is_spatial and not ks:
        raise ValueError(
            f"Cannot emit spatial operator {op.name!r} ({op.op_type}) "
            "without a concrete kernel_shape"
        )

    kernel_h = int(ks[0]) if len(ks) >= 1 else 0
    kernel_w = int(ks[1]) if len(ks) >= 2 else (1 if is_1d else (kernel_h if ks else 0))
    # ONNX defaults both stride and dilation to 1 for spatial operators.
    # Zero is reserved for non-spatial descriptors in the binary format.
    stride_h = int(st[0]) if len(st) >= 1 else (1 if is_spatial else 0)
    stride_w = int(st[1]) if len(st) >= 2 else (
        1 if is_1d else (stride_h if st or is_spatial else 0)
    )
    dilation_h = int(di[0]) if len(di) >= 1 else (1 if is_spatial else 0)
    dilation_w = int(di[1]) if len(di) >= 2 else (
        1 if is_1d else (dilation_h if di or is_spatial else 0)
    )

    if is_1d and len(pa) == 2:
        # ONNX Conv1D pads: [begin, end]
        pad_top = int(pa[0])
        pad_bottom = int(pa[1])
        pad_left = 0
        pad_right = 0
    else:
        pad_top = int(pa[0]) if len(pa) >= 1 else 0
        pad_left = int(pa[1]) if len(pa) >= 2 else 0
        pad_bottom = int(pa[2]) if len(pa) >= 3 else 0
        pad_right = int(pa[3]) if len(pa) >= 4 else 0

    return SPATIAL_ATTRS_STRUCT.pack(
        kernel_h, kernel_w, stride_h, stride_w,
        pad_top, pad_bottom, pad_left, pad_right,
        dilation_h, dilation_w,
        gr,
    )


# Weight processing


def _build_weight_op_map(ag: AnalyzedGraph) -> dict[str, str]:
    """Map each serialized weight to the operator that determines its layout."""
    result: dict[str, str] = {}
    for op in ag.ops:
        if op.op_type in {"Add", "Mul"}:
            for input_name in op.inputs:
                if input_name in ag.weight_data:
                    result[input_name] = op.op_type
        elif len(op.inputs) >= 2 and op.inputs[1] in ag.weight_data:
            result[op.inputs[1]] = op.op_type
    return result


def _transpose_weight_nhwc(arr: np.ndarray, op_type: str | None) -> np.ndarray:
    """Transpose weight from NCHW convention to NHWC convention at emission time."""
    if op_type in {"Add", "Mul"}:
        if arr.ndim == 4:
            return arr.transpose(0, 2, 3, 1)
        if arr.ndim == 3:
            return arr.transpose(0, 2, 1)

    if arr.ndim == 4 and op_type == "DepthwiseConv":
        # ONNX depthwise: [C, 1, KH, KW] -> OHWI gives [C, KH, KW, 1]
        # Then reshape to [C, KH, KW] and transpose to [KH, KW, C] (HWC)
        arr = arr.transpose(0, 2, 3, 1)   # [C, 1, KH, KW] -> [C, KH, KW, 1]
        C, KH, KW, _ = arr.shape
        arr = arr.reshape(C, KH, KW)      # drop trailing 1
        arr = arr.transpose(1, 2, 0)      # CHW -> HWC: [KH, KW, C]
        return np.ascontiguousarray(arr)
    elif arr.ndim == 4 and op_type in ("Conv",):
        return arr.transpose(0, 2, 3, 1)  # OIHW -> OHWI
    elif arr.ndim == 4 and op_type == "ConvTranspose":
        return arr.transpose(1, 2, 3, 0)  # IOHW -> OHWI
    elif arr.ndim == 3 and op_type == "Conv1D":
        return arr.transpose(0, 2, 1)     # OIK -> OKI
    # 2D (FC) and 1D (bias) stay as-is
    return arr


def _build_fc_layout_permutations(
    ag: AnalyzedGraph,
) -> dict[str, tuple[int, ...]]:
    """Find FC inputs that flatten a spatial tensor before Gemm.

    Both Flatten and a flattening Reshape preserve ONNX's NCHW/NCL byte order.
    The runtime holds those tensors as NHWC/NLC, so the FC weight columns must
    be permuted to keep the model's observable result unchanged.

    Returns: weight_name -> pre-flatten NCHW shape (for column permutation).
    """
    result: dict[str, tuple[int, ...]] = {}
    # Build output->op map
    out_to_op: dict[str, OpNode] = {}
    for op in ag.ops:
        for o in op.outputs:
            out_to_op[o] = op

    for op in ag.ops:
        if op.op_type != "Gemm":
            continue
        if len(op.inputs) < 2:
            continue
        # Check if input[0] comes from a spatial-to-vector transform.
        data_input = op.inputs[0]
        producer = out_to_op.get(data_input)
        if producer is None or producer.op_type not in {"Flatten", "Reshape"}:
            continue
        spatial_input = producer.inputs[0]
        spatial_info = ag.tensors.get(spatial_input)
        vector_info = ag.tensors.get(data_input)
        if spatial_info is None or vector_info is None:
            continue
        shape = spatial_info.shape
        if len(shape) not in (3, 4):
            continue
        if len(vector_info.shape) != 2:
            continue
        if (
            vector_info.shape[0] != shape[0]
            or int(np.prod(vector_info.shape))
            != int(np.prod(shape))
        ):
            continue
        # For 4D [N,C,H,W] with H=W=1 (e.g. after GlobalAvgPool), no permutation needed
        if len(shape) == 4 and shape[2] == 1 and shape[3] == 1:
            continue
        # Store the pre-flatten shape for column permutation
        weight_name = op.inputs[1]
        if weight_name in ag.weight_data:
            result[weight_name] = shape

    return result


def _permute_fc_weight_for_nhwc(
    W: np.ndarray, pre_flatten_shape: tuple[int, ...]
) -> np.ndarray:
    """Permute FC weight columns to match NHWC flatten order.

    When Flatten is applied to an NCHW tensor, the flat order is C-major.
    When applied to NHWC, it's H-major (channels last). We need to rearrange
    the weight columns so FC(NHWC_flattened) == FC(NCHW_flattened).

    W shape: [OC, IC] where IC = product(spatial_dims).
    """
    if W.ndim != 2:
        return W

    shape = pre_flatten_shape
    if len(shape) == 4:
        N, C, H, Wd = shape
        IC = C * H * Wd
        if W.shape[1] != IC:
            return W
        # Build permutation: for each NHWC flat position, which NCHW column to read.
        # NHWC flat index: h*W*C + w*C + c -> needs NCHW column: c*H*W + h*W + w
        perm = np.zeros(IC, dtype=np.intp)
        for c in range(C):
            for h in range(H):
                for w in range(Wd):
                    nchw_idx = c * H * Wd + h * Wd + w
                    nhwc_idx = h * Wd * C + w * C + c
                    perm[nhwc_idx] = nchw_idx
        return W[:, perm]
    elif len(shape) == 3:
        N, C, L = shape
        IC = C * L
        if W.shape[1] != IC:
            return W
        # For each NLC flat position, which NCL column to read.
        # NLC flat index: l*C + c -> needs NCL column: c*L + l
        perm = np.zeros(IC, dtype=np.intp)
        for c in range(C):
            for li in range(L):
                ncl_idx = c * L + li
                nlc_idx = li * C + c
                perm[nlc_idx] = ncl_idx
        return W[:, perm]

    return W


def _build_weights(
    ag: AnalyzedGraph,
    strings: _StringTable,
) -> tuple[bytes, dict[str, int]]:
    """Build weight entries and contiguous weight blob.

    Returns (section_bytes, name->weight_index map).
    Section layout: weight_entry_t[num_weights] + contiguous weight blob.
    """
    if not ag.weight_data:
        return b"", {}

    weight_op_map = _build_weight_op_map(ag)
    fc_layout_shapes = _build_fc_layout_permutations(ag)

    weight_idx: dict[str, int] = {}
    entries_buf = bytearray()
    blob_buf = bytearray()

    for idx, (name, arr) in enumerate(ag.weight_data.items()):
        weight_idx[name] = idx
        op_type = weight_op_map.get(name)
        arr = _transpose_weight_nhwc(arr, op_type)
        # Preserve ONNX flatten/reshape order across the internal layout change.
        if name in fc_layout_shapes:
            arr = _permute_fc_weight_for_nhwc(arr, fc_layout_shapes[name])
        # Preserve int8/int32 dtype for quantized weights
        if arr.dtype in (np.int8, np.int32):
            raw = arr.tobytes()
        else:
            raw = arr.astype(np.float32).tobytes()
        name_off = strings.add(name)
        # Pad so this weight starts WEIGHT_ALIGN-aligned within the blob.
        # Combined with a WEIGHT_ALIGN-aligned blob base (emit assembler aligns
        # the section) every weight/bias pointer is aligned for the opt kernels.
        blob_buf.extend(b"\x00" * ((-len(blob_buf)) % WEIGHT_ALIGN))
        blob_offset = len(blob_buf)
        entries_buf.extend(WEIGHT_ENTRY_STRUCT.pack(
            name_off,
            blob_offset,
            len(raw),
        ))
        blob_buf.extend(raw)

    return bytes(entries_buf) + bytes(blob_buf), weight_idx


def _group_weights_by_stage(
    ag: AnalyzedGraph,
    weight_idx: dict[str, int],
) -> list[tuple[int, list[int]]]:
    """Group global weight indices by stage.

    Returns list of (stage_idx, sorted list of global weight indices)
    for stages that have weights. Each weight belongs to exactly one stage.
    """
    # Map each weight index to its stage
    widx_to_stage: dict[int, int] = {}
    for stage in ag.stages:
        for op_i in stage.op_indices:
            op = ag.ops[op_i]
            w_idx, b_idx = _resolve_weight_bias(op, weight_idx)
            if w_idx != NO_WEIGHT:
                if w_idx in widx_to_stage and widx_to_stage[w_idx] != stage.stage_id:
                    raise ValueError(f"Weight {w_idx} shared across stages")
                widx_to_stage[w_idx] = stage.stage_id
            if b_idx != NO_WEIGHT:
                if b_idx in widx_to_stage and widx_to_stage[b_idx] != stage.stage_id:
                    raise ValueError(f"Bias {b_idx} shared across stages")
                widx_to_stage[b_idx] = stage.stage_id

    # Build per-stage groups
    stage_groups: dict[int, list[int]] = {}
    for widx, sid in widx_to_stage.items():
        stage_groups.setdefault(sid, []).append(widx)
    for v in stage_groups.values():
        v.sort()

    return sorted(stage_groups.items())


def _build_weights_compressed(
    ag: AnalyzedGraph,
    strings: _StringTable,
    compress: str,
) -> tuple[bytes, bytes, dict[str, int]]:
    """Build compressed per-stage weight blocks.

    Returns (entries_only_bytes, weight_blocks_section_bytes, weight_idx map).
    entries_only_bytes = weight_entry_t[] with block-relative offsets.
    weight_blocks_section_bytes = header + block_entry_t[] + compressed blobs.
    """
    import lz4.block

    if not ag.weight_data:
        return b"", b"", {}

    weight_op_map = _build_weight_op_map(ag)
    fc_layout_shapes = _build_fc_layout_permutations(ag)

    # First pass: prepare all weights (same transform as _build_weights)
    weight_idx: dict[str, int] = {}
    weight_names = list(ag.weight_data.keys())
    prepared: list[tuple[bytes, int]]  = []  # (raw_bytes, name_str_off)

    for idx, name in enumerate(weight_names):
        arr = ag.weight_data[name]
        weight_idx[name] = idx
        op_type = weight_op_map.get(name)
        arr = _transpose_weight_nhwc(arr, op_type)
        if name in fc_layout_shapes:
            arr = _permute_fc_weight_for_nhwc(arr, fc_layout_shapes[name])
        if arr.dtype in (np.int8, np.int32):
            raw = arr.tobytes()
        else:
            raw = arr.astype(np.float32).tobytes()
        name_off = strings.add(name)
        prepared.append((raw, name_off))

    # Group weights by stage
    groups = _group_weights_by_stage(ag, weight_idx)

    # Build per-stage blocks and entries with block-relative offsets
    entries_buf = bytearray(len(prepared) * WEIGHT_ENTRY_SIZE)
    blocks: list[tuple[int, int, int, int, int, bytes]] = []  # stage_idx, first_widx, num_w, uncomp_sz, comp_sz, comp_data
    blob_parts: list[bytes] = []
    compress_type = COMPRESS_LZ4 if compress == "lz4" else COMPRESS_NONE

    for stage_idx, widx_list in groups:
        # Build contiguous raw block for this stage's weights
        block_buf = bytearray()
        first_widx = widx_list[0]
        for widx in widx_list:
            raw, name_off = prepared[widx]
            # Pad to WEIGHT_ALIGN so each weight is aligned after decompression
            # into the (WEIGHT_ALIGN-aligned) fast arena - same requirement as
            # the uncompressed path.
            block_buf.extend(b"\x00" * ((-len(block_buf)) % WEIGHT_ALIGN))
            # Write entry with block-relative offset
            WEIGHT_ENTRY_STRUCT.pack_into(
                entries_buf, widx * WEIGHT_ENTRY_SIZE,
                name_off, len(block_buf), len(raw),
            )
            block_buf.extend(raw)

        uncompressed_size = len(block_buf)
        if compress_type == COMPRESS_LZ4:
            compressed = lz4.block.compress(
                bytes(block_buf), store_size=False)
        else:
            compressed = bytes(block_buf)
        compressed_size = len(compressed)

        blocks.append((stage_idx, first_widx, len(widx_list),
                       uncompressed_size, compressed_size, compressed))
        blob_parts.append(compressed)

    # Handle weights not assigned to any stage (shouldn't happen but be safe)
    assigned = set()
    for _, widx_list in groups:
        assigned.update(widx_list)
    for widx in range(len(prepared)):
        if widx not in assigned:
            raw, name_off = prepared[widx]
            WEIGHT_ENTRY_STRUCT.pack_into(
                entries_buf, widx * WEIGHT_ENTRY_SIZE,
                name_off, 0, len(raw),
            )

    # Build SEC_WEIGHT_BLOCKS section:
    # header: num_blocks(u16) + compression_type(u16)
    # block entries: tigris_weight_block_t[num_blocks]
    # blob data
    num_blocks = len(blocks)
    sec_header = WEIGHT_BLOCK_SECTION_HEADER_STRUCT.pack(num_blocks, compress_type)
    block_entries = bytearray()
    current_blob_off = 0
    for stage_idx, first_widx, num_w, uncomp_sz, comp_sz, comp_data in blocks:
        block_entries.extend(WEIGHT_BLOCK_STRUCT.pack(
            stage_idx, first_widx, num_w, 0,  # pad
            current_blob_off, comp_sz, uncomp_sz,
        ))
        current_blob_off += comp_sz

    blob_data = b"".join(blob_parts)
    weight_blocks_section = sec_header + bytes(block_entries) + blob_data

    return bytes(entries_buf), weight_blocks_section, weight_idx


def compressed_weight_reserve_bytes(ag: AnalyzedGraph) -> int:
    """Return the peak fast-arena reservation for compressed weight blocks.

    This mirrors ``tigris_weight_decompression_overhead()`` in the runtime:
    a normal stage needs one aligned decompressed block, while a tiled chain
    holds its member-stage blocks simultaneously.  It deliberately builds the
    same block grouping as serialization so plan validation cannot approve an
    arena that the executor later overcommits.
    """
    if not ag.weight_data:
        return 0

    # ``none`` retains the exact block layout without doing needless LZ4 work.
    _, section, _ = _build_weights_compressed(ag, _StringTable(), "none")
    if len(section) < 4:
        return 0
    num_blocks, _ = WEIGHT_BLOCK_SECTION_HEADER_STRUCT.unpack_from(section, 0)
    block_size = WEIGHT_BLOCK_STRUCT.size
    block_bytes: dict[int, int] = {}
    for i in range(num_blocks):
        off = 4 + i * block_size
        stage_idx, _, _, _, _, _, uncompressed = WEIGHT_BLOCK_STRUCT.unpack_from(
            section, off
        )
        aligned = (uncompressed + ag.tensor_alignment - 1) & ~(
            ag.tensor_alignment - 1
        )
        block_bytes[stage_idx] = aligned

    reserve = max(block_bytes.values(), default=0)
    for stage in ag.stages:
        if stage.chain_len < 2 or stage.chain_id != stage.stage_id:
            continue
        chain = sum(
            block_bytes.get(stage_idx, 0)
            for stage_idx in range(stage.stage_id, stage.stage_id + stage.chain_len)
        )
        reserve = max(reserve, chain)
    return reserve


# Section builders


def _build_tensors(
    ag: AnalyzedGraph,
    strings: _StringTable,
    shapes: _ShapePool,
    quant_idx_map: dict[str, int] | None = None,
    preserve_layout_names: set[str] | None = None,
) -> tuple[bytes, dict[str, int]]:
    """Build tensor table. Returns (bytes, name->index map)."""
    buf = bytearray()
    tensor_idx: dict[str, int] = {}
    idx = 0
    no_qp = NO_QUANT_PARAM

    for name, info in ag.tensors.items():
        if info.is_constant:
            continue

        tensor_idx[name] = idx
        idx += 1

        name_off = strings.add(name)

        # Internal tensors use NHWC/NLC layout. A terminal Transpose is an
        # explicit model-output boundary, so its output keeps the observable
        # ONNX shape and element ordering instead.
        shape = info.shape
        if preserve_layout_names and name in preserve_layout_names:
            pass
        elif len(shape) == 4:
            shape = (shape[0], shape[2], shape[3], shape[1])
        elif len(shape) == 3:
            shape = (shape[0], shape[2], shape[1])

        shape_off, ndim = shapes.add(shape)

        flags = 0
        if info.is_constant:
            flags |= TENSOR_FLAG_CONSTANT
        if name in ag.model_inputs:
            flags |= TENSOR_FLAG_MODEL_INPUT
        if name in ag.model_outputs:
            flags |= TENSOR_FLAG_MODEL_OUTPUT

        qp_idx = quant_idx_map.get(name, no_qp) if quant_idx_map else no_qp

        # tigris_tensor_t: 16 bytes
        # name_str(u32) size_bytes(u32) shape_off(u16) ndim(u8) dtype(u8) flags(u8) quant_param_idx(u16) pad(1)
        buf.extend(TENSOR_STRUCT.pack(
            name_off,
            info.size_bytes,
            shape_off,
            ndim,
            info.dtype,
            flags,
            qp_idx,
        ))

    return bytes(buf), tensor_idx


def _serialized_axis_map(rank: int, preserve_layout: bool = False) -> list[int]:
    """Map an ONNX axis to its serialized tensor axis."""
    if preserve_layout:
        return list(range(rank))
    if rank == 4:
        return [0, 3, 1, 2]  # NCHW -> NHWC
    if rank == 3:
        return [0, 2, 1]     # NCL -> NLC
    return list(range(rank))


def _build_op_attributes(
    ag: AnalyzedGraph, tensor_idx: dict[str, int], preserve_layout_names: set[str]
) -> bytes:
    """Build optional, typed per-operator attributes.

    The section is intentionally omitted when no operator needs metadata.
    Its fixed-width records keep the loader zero-copy while the data pool
    carries variable-sized payloads such as a Transpose permutation.
    """
    records: list[tuple[int, int, bytes]] = []
    for op_index, op in enumerate(ag.ops):
        if op.op_type != "Transpose":
            continue
        input_name, output_name = op.inputs[0], op.outputs[0]
        if input_name not in tensor_idx or output_name not in tensor_idx:
            raise ValueError(f"Transpose '{op.name}' must use runtime tensors")
        input_info = ag.tensors[input_name]
        rank = len(input_info.shape)
        raw_perm = [int(axis) for axis in op.attrs["perm"]]
        input_axes = _serialized_axis_map(rank)
        output_axes = _serialized_axis_map(
            rank, output_name in preserve_layout_names
        )
        output_raw_by_serialized = [0] * rank
        for raw_axis, serialized_axis in enumerate(output_axes):
            output_raw_by_serialized[serialized_axis] = raw_axis
        serialized_perm = bytes(
            input_axes[raw_perm[output_raw_by_serialized[serialized_axis]]]
            for serialized_axis in range(rank)
        )
        records.append((op_index, OP_ATTR_TRANSPOSE_PERM, serialized_perm))

    if not records:
        return b""
    records.sort(key=lambda record: (record[0], record[1]))
    if len(records) > 0xFFFF:
        raise ValueError("operator-attribute count exceeds uint16 plan limit")

    entries = bytearray()
    data = bytearray()
    previous_key: tuple[int, int] | None = None
    for op_index, attr_type, payload in records:
        key = (op_index, attr_type)
        if key == previous_key:
            raise ValueError(
                f"duplicate operator attribute type {attr_type} for operator {op_index}"
            )
        previous_key = key
        _require_uint(op_index, 16, "operator-attribute operator index")
        if len(payload) > 0xFF:
            raise ValueError("operator-attribute payload exceeds uint8 length")
        entries.extend(OP_ATTRIBUTE_STRUCT.pack(
            op_index, attr_type, len(payload), len(data)
        ))
        data.extend(payload)
    return (
        OP_ATTRIBUTE_SECTION_HEADER_STRUCT.pack(len(records), 0)
        + bytes(entries)
        + bytes(data)
    )


def _resolve_weight_bias(op: OpNode, weight_idx: dict[str, int]) -> tuple[int, int]:
    """Map an op's constant inputs to weight/bias indices.

    Convention: for ops like Conv/Gemm, input[1] is the weight and input[2]
    is the optional bias. For all ops, the first constant input is the weight
    and the second (if any) is the bias.
    Returns (weight_idx, bias_idx) with NO_WEIGHT for missing entries.
    """
    w_idx = NO_WEIGHT
    b_idx = NO_WEIGHT

    # Walk the op's inputs and find those that are weights
    const_inputs = [inp for inp in op.inputs if inp in weight_idx]

    if len(const_inputs) >= 1:
        w_idx = weight_idx[const_inputs[0]]
    if len(const_inputs) >= 2:
        b_idx = weight_idx[const_inputs[1]]

    return w_idx, b_idx


def _round_half_away(x: float) -> int:
    """Round half away from zero (TFLite TfLiteRound / C std::round), NOT Python's
    banker's round-half-to-even. The compiler's quant params must round the same
    way as tflite/gemmlowp/CMSIS-NN and the runtime, or a per-channel requant
    multiplier (or a ReLU6 act_max) lands 1 off for any scale on a .5 boundary."""
    return math.floor(x + 0.5) if x >= 0.0 else math.ceil(x - 0.5)


def _compute_act_bounds(
    ag: AnalyzedGraph, op: OpNode, fused_act: int,
) -> tuple[int, int]:
    """Compute int8 activation bounds for fused activation.

    Returns (act_min, act_max) as signed int8 values.
    For float models without quant params, returns placeholder (-128, 127);
    the float kernels use the fused_act enum directly.
    """
    if fused_act == ACT_NONE:
        return -128, 127

    # Try to get output quant params
    out_name = op.outputs[0] if op.outputs else None
    out_info = ag.tensors.get(out_name) if out_name else None
    qp = out_info.quant if out_info else None

    if qp is None:
        # Float model - bounds are unused; kernels check the enum
        return -128, 127

    output_zp = int(qp.zero_point[0])
    output_scale = float(qp.scale[0])

    if fused_act == ACT_RELU:
        act_min = max(output_zp, -128)
        act_max = 127
    elif fused_act == ACT_RELU6:
        act_min = max(output_zp, -128)
        act_max = min(_round_half_away(6.0 / output_scale) + output_zp, 127)
    else:
        act_min, act_max = -128, 127

    return act_min, act_max


def _build_ops(
    ag: AnalyzedGraph,
    tensor_idx: dict[str, int],
    weight_idx: dict[str, int],
    strings: _StringTable,
    index_pool: _IndexPool,
) -> bytes:
    """Build op table. Returns bytes."""
    buf = bytearray()

    for op in ag.ops:
        name_off = strings.add(op.name)
        op_type = OP_TYPE_MAP[op.op_type]

        # Map input/output tensor names to indices (skip constants)
        inp_indices = [tensor_idx[n] for n in op.inputs if n in tensor_idx]
        out_indices = [tensor_idx[n] for n in op.outputs if n in tensor_idx]

        inp_off, inp_count = index_pool.add(inp_indices)
        out_off, out_count = index_pool.add(out_indices)
        _require_uint(inp_count, 8, f"operator {op.name!r} input count")
        _require_uint(out_count, 8, f"operator {op.name!r} output count")
        if not -1 <= op.stage <= 0xFFFF:
            raise ValueError(
                f"operator {op.name!r} stage {op.stage} exceeds the uint16 stage-table limit"
            )

        spatial = _pack_spatial_attrs(op, ag.weight_data)

        # Resolve weight/bias indices from op's constant inputs
        w_idx, b_idx = _resolve_weight_bias(op, weight_idx)

        # Determine fused activation
        fused_act_str = op.attrs.get("fused_activation")
        fused_act = {"Relu": ACT_RELU, "Relu6": ACT_RELU6}.get(fused_act_str, ACT_NONE)
        act_min, act_max = _compute_act_bounds(ag, op, fused_act)

        # tigris_op_t: 38 bytes (layout retained by schema v3)
        # name_str(u32) op_type(u8) num_inputs(u8) num_outputs(u8)
        # stage_hint(u8).  Schema v5 makes the stage table authoritative; this
        # legacy byte retains the low stage-index byte for canonical encoding
        # and v2-v4 compatibility without limiting v5 plans to 256 stages.
        # inputs_offset(u16) outputs_offset(u16)
        # spatial_attrs(18 bytes)
        # weight_idx(u16) bias_idx(u16)
        # fused_act(u8) act_min(i8) act_max(i8) _pad(u8)
        buf.extend(OP_PREFIX_STRUCT.pack(
            name_off,
            op_type,
            inp_count,
            out_count,
            max(op.stage, 0) & 0xFF,
            inp_off,
            out_off,
        ))
        buf.extend(spatial)
        buf.extend(OP_WEIGHT_BIAS_STRUCT.pack(w_idx, b_idx))
        buf.extend(OP_ACTIVATION_STRUCT.pack(fused_act, act_min, act_max))

    return bytes(buf)


def _build_stages(
    ag: AnalyzedGraph,
    tensor_idx: dict[str, int],
    index_pool: _IndexPool,
) -> bytes:
    """Build stage table. Returns bytes."""
    buf = bytearray()

    for stage in ag.stages:
        ops_off, ops_count = index_pool.add(stage.op_indices)

        inp_indices = [tensor_idx[n] for n in stage.input_tensors if n in tensor_idx]
        out_indices = [tensor_idx[n] for n in stage.output_tensors if n in tensor_idx]

        if len(inp_indices) > RUNTIME_MAX_STAGE_INPUTS:
            raise ValueError(
                f"stage {stage.stage_id} has {len(inp_indices)} inputs; current runtime limit is "
                f"{RUNTIME_MAX_STAGE_INPUTS}"
            )
        if len(out_indices) > RUNTIME_MAX_STAGE_OUTPUTS:
            raise ValueError(
                f"stage {stage.stage_id} has {len(out_indices)} outputs; current runtime limit is "
                f"{RUNTIME_MAX_STAGE_OUTPUTS}"
            )
        if stage.chain_len > RUNTIME_MAX_CHAIN_STAGES:
            raise ValueError(
                f"stage {stage.stage_id} chain length {stage.chain_len} exceeds current runtime "
                f"limit {RUNTIME_MAX_CHAIN_STAGES}"
            )
        _require_uint(stage.chain_id, 16, f"stage {stage.stage_id} chain id")
        _require_uint(stage.chain_len, 16, f"stage {stage.stage_id} chain length")
        _require_uint(stage.chain_tile_h, 16, f"stage {stage.stage_id} chain tile height")

        inp_off, inp_count = index_pool.add(inp_indices)
        out_off, out_count = index_pool.add(out_indices)

        # tile_plan_idx: index into tile_plan table, or 0xFFFF if none
        tile_idx = 0xFFFF
        # We'll set this after building tile plans; for now track stage_id -> tile_plan_idx
        # Actually, we build tile plans in order of stages that have them
        # Let the caller patch this

        # tigris_stage_t: 28 bytes
        # peak_bytes(u32)
        # ops_offset(u16) ops_count(u16)
        # inputs_offset(u16) inputs_count(u16)
        # outputs_offset(u16) outputs_count(u16)
        # tile_plan_idx(u16) pad(u16)
        # chain_id(u16) chain_len(u16)
        # chain_tile_h(u16) _reserved1(u16)
        reserved1 = STAGE_FLAG_LINE_BUFFERED if stage.line_buffered else 0
        buf.extend(STAGE_STRUCT.pack(
            stage.peak_bytes,
            ops_off, ops_count,
            inp_off, inp_count,
            out_off, out_count,
            tile_idx, 0,
            stage.chain_id,
            stage.chain_len,
            stage.chain_tile_h,
            reserved1,
        ))

    return bytes(buf)


def _build_tile_plans(ag: AnalyzedGraph) -> tuple[bytes, dict[int, int]]:
    """Build tile plan table. Returns (bytes, stage_id->tile_plan_index map)."""
    buf = bytearray()
    stage_to_tile: dict[int, int] = {}
    idx = 0

    for stage in ag.stages:
        tp = stage.tile_plan
        if tp is None:
            continue

        stage_to_tile[stage.stage_id] = idx
        idx += 1
        if tp.tileable and tp.axis != TILE_AXIS_HEIGHT_OR_LENGTH:
            raise ValueError(
                f"stage {stage.stage_id} has unsupported tile axis {tp.axis}"
            )
        if not tp.tileable and tp.axis != TILE_AXIS_NONE:
            raise ValueError(
                f"untileable stage {stage.stage_id} must use tile axis 0"
            )

        # tigris_tile_plan_t: 24 bytes
        # tileable(u8) axis(u8) tile_height(u16)
        # num_tiles(u16) halo(u16)
        # receptive_field(u16) original_height(u16)
        # tiled_peak_bytes(u32)
        # overhead_bytes(u32)
        # reserved(u32)
        buf.extend(TILE_PLAN_STRUCT.pack(
            1 if tp.tileable else 0,
            tp.axis,
            tp.tile_height,
            tp.num_tiles,
            tp.halo,
            tp.receptive_field,
            tp.original_height,
            tp.tiled_peak_bytes,
            tp.overhead_bytes,
            0,  # reserved
        ))

    return bytes(buf), stage_to_tile


# Quantization


def _compute_multiplier_shift(scale: float) -> tuple[int, int]:
    """Convert a float scale to (Q0.31 multiplier, shift) pair.

    Uses the TFLite/CMSIS-NN convention:
        output = (acc * multiplier) >> (31 - shift)
    where multiplier is in [0.5, 1.0) in Q0.31 format.
    """
    if scale == 0.0:
        return 0, 0

    mantissa, exp = math.frexp(scale)  # 0.5 <= mantissa < 1.0, scale = mantissa * 2^exp

    multiplier = _round_half_away(mantissa * (1 << 31))
    if multiplier == (1 << 31):
        multiplier //= 2
        exp += 1

    # Runtime computes (acc * multiplier) >> 31, then applies `shift`, i.e.
    # acc * multiplier * 2^(shift - 31). With scale = multiplier * 2^(exp - 31)
    # (frexp), the correct shift is exactly `exp` (matches TFLite
    # QuantizeMultiplier). A `- 1` here halves every requant and collapses
    # deep int8 graphs to the output zero-point.
    return multiplier, exp


def _compute_effective_scales(ag: AnalyzedGraph) -> dict[str, np.ndarray]:
    """Compute effective requantization scales for op output tensors.

    For quantized conv/depthwise/FC ops, the kernel requantizes the int32
    accumulator back to int8 using:
        effective_scale = input_scale * weight_scale / output_scale

    Returns a map from output tensor name to per-channel effective scales.
    Tensors not produced by these ops (model inputs, intermediates without
    weights) are not included - they keep their raw tensor scale.
    """
    effective: dict[str, np.ndarray] = {}
    weight_ops = {"Conv", "ConvInteger", "DepthwiseConv", "MatMul", "Gemm",
                  "QLinearConv", "QLinearMatMul"}

    for op in ag.ops:
        if op.op_type not in weight_ops:
            continue

        # Output tensor
        out_name = op.outputs[0]
        out_info = ag.tensors.get(out_name)
        if not out_info or not out_info.quant:
            continue
        out_scale = out_info.quant.scale

        # Input tensor (first non-constant input)
        in_name = None
        for inp in op.inputs:
            inp_info = ag.tensors.get(inp)
            if inp_info and not inp_info.is_constant:
                in_name = inp
                break
        if not in_name:
            continue
        in_info = ag.tensors.get(in_name)
        if not in_info or not in_info.quant:
            continue
        in_scale = float(in_info.quant.scale[0])  # input is always per-tensor

        # Weight tensor (first constant input)
        w_name = None
        for inp in op.inputs:
            inp_info = ag.tensors.get(inp)
            if inp_info and inp_info.is_constant and inp_info.quant:
                w_name = inp
                break
        if not w_name:
            continue

        w_info = ag.tensors.get(w_name)
        if not w_info or not w_info.quant:
            continue
        w_scale = w_info.quant.scale  # may be per-channel

        # effective_scale[c] = in_scale * w_scale[c] / out_scale (per output
        # channel). The output ACTIVATION scale is per-tensor (len 1), but
        # per-channel weight quant has one scale per output channel, so the
        # number of effective scales is the per-channel WEIGHT count - using
        # len(out_scale) collapsed per-channel conv to per-tensor (reusing
        # channel 0's multiplier for all channels, which diverges from TFLite).
        num_w_ch = len(w_scale)
        num_ch = max(len(out_scale), num_w_ch)
        eff = np.zeros(num_ch, dtype=np.float64)
        for c in range(num_ch):
            ws = float(w_scale[c]) if c < num_w_ch else float(w_scale[0])
            os = float(out_scale[c]) if c < len(out_scale) else float(out_scale[0])
            eff[c] = in_scale * ws / os if os != 0 else 0.0

        effective[out_name] = eff

    return effective


def _build_quant_params(
    ag: AnalyzedGraph,
) -> tuple[bytes, dict[str, int]]:
    """Build quant params section.

    Returns (section_bytes, tensor_name->quant_param_idx map).

    Section layout:
        uint16_t num_quant_params
        uint16_t quant_data_pages (v3: 64K-element pages in data blob)
        tigris_quant_param_t[num_quant_params]
        int32_t[] quant_data  (multiplier/shift arrays for all params)
    """
    if not ag.is_quantized:
        return b"", {}

    # Pre-compute effective scales for op output tensors
    effective_scales = _compute_effective_scales(ag)

    quant_idx_map: dict[str, int] = {}
    entries_buf = bytearray()
    data_buf = bytearray()  # int32 elements
    data_offset = 0  # in int32 elements
    page_size = 1 << 16
    idx = 0

    for name, info in ag.tensors.items():
        if info.quant is None:
            continue

        qp = info.quant
        quant_idx_map[name] = idx
        idx += 1

        # Use effective scale for op outputs (requantization), raw scale otherwise
        eff = effective_scales.get(name)

        # Per-channel count comes from the effective scales (per output channel
        # for per-channel weight quant), not the per-tensor activation scale -
        # otherwise per-channel conv requant is stored as per-tensor.
        num_channels = len(eff) if eff is not None else len(qp.scale)
        # One v3 entry selects one 64K-element page for both arrays, so the
        # multiplier and shift arrays must fit together in that page.
        if num_channels > page_size // 2:
            raise ValueError(
                f"quantization channels for {name!r} exceed the v3 per-param "
                f"limit ({num_channels} > {page_size // 2})")

        # v3 retains the compact 16-byte entry and uses its former padding as
        # a page number. Keep each multiplier/shift pair inside one 64K-element
        # page so v2-sized offsets remain valid within that page.
        needed = 2 * num_channels
        if data_offset % page_size + needed > page_size:
            next_page = ((data_offset + page_size - 1) // page_size) * page_size
            data_buf.extend(b"\x00" * ((next_page - data_offset) * 4))
            data_offset = next_page

        # Store multiplier/shift for both per-tensor (1) and per-channel (>1)
        multiplier_off = data_offset
        shift_off = data_offset + num_channels
        for c in range(num_channels):
            scale = float(eff[c]) if eff is not None else float(qp.scale[c])
            m, s = _compute_multiplier_shift(scale)
            data_buf.extend(QUANT_DATA_STRUCT.pack(m))
        for c in range(num_channels):
            scale = float(eff[c]) if eff is not None else float(qp.scale[c])
            _, s = _compute_multiplier_shift(scale)
            data_buf.extend(QUANT_DATA_STRUCT.pack(s))
        data_offset += 2 * num_channels

        # tigris_quant_param_t: 16 bytes
        # scale(f32) zero_point(i32) num_channels(u16) multiplier_off(u16) shift_off(u16) pad(u16)
        entries_buf.extend(QUANT_PARAM_STRUCT.pack(
            float(qp.scale[0]),
            int(qp.zero_point[0]),
            num_channels,
            multiplier_off % page_size,
            shift_off % page_size,
            multiplier_off // page_size,  # v3 quant-data page
        ))

    if not quant_idx_map:
        return b"", {}

    if len(quant_idx_map) > 0xFFFF:
        raise ValueError("quantization parameter count exceeds uint16 plan limit")
    pages = (data_offset + page_size - 1) // page_size
    if pages > 0xFFFF:
        raise ValueError("quantization data exceeds v3 page-count limit")
    # v3 section header: num_quant_params(u16) + quant_data_pages(u16).
    header = QUANT_SECTION_HEADER_STRUCT.pack(len(quant_idx_map), pages)
    return header + bytes(entries_buf) + bytes(data_buf), quant_idx_map


def _patch_stage_tile_indices(stage_data: bytearray, ag: AnalyzedGraph, stage_to_tile: dict[int, int]) -> bytes:
    """Patch tile_plan_idx in stage entries."""
    for i, stage in enumerate(ag.stages):
        tile_idx = stage_to_tile.get(stage.stage_id, 0xFFFF)
        offset = i * STAGE_SIZE + STAGE_TILE_PLAN_INDEX_OFFSET
        INDEX_STRUCT.pack_into(stage_data, offset, tile_idx)
    return bytes(stage_data)


# Plan assembly


def emit_binary(ag: AnalyzedGraph, path: Path, compress: str | None = None, xip: bool = False) -> None:
    """Write an execution plan as a binary file to *path*."""
    data = emit_binary_bytes(ag, compress=compress, xip=xip)
    with open(path, "wb") as f:
        f.write(data)


def emit_binary_bytes(ag: AnalyzedGraph, compress: str | None = None, xip: bool = False) -> bytes:
    """Build the complete binary plan and return as bytes.

    Args:
        ag: Analyzed graph to serialize.
        compress: Weight compression method. ``None`` or ``"lz4"``.
        xip: If True, set FLAG_XIP in the header (weights read from flash).
    """
    from tigris.analysis.validation import (
        validate_execution_dtype,
        validate_memory_plan,
        validate_operator_support,
    )

    if not 0 <= ag.mem_budget <= 0xFFFFFFFF:
        raise ValueError("Fast-memory budget exceeds the uint32 plan-format limit")
    if not 0 <= ag.peak_memory_bytes <= 0xFFFFFFFF:
        raise ValueError("Peak activation memory exceeds the uint32 plan-format limit")

    dtype_validation = validate_execution_dtype(ag)
    if not dtype_validation.supported:
        raise ValueError(
            "Cannot emit a plan with unsupported tensor dtypes: "
            + dtype_validation.describe()
        )

    operator_validation = validate_operator_support(ag)
    if not operator_validation.supported:
        raise ValueError(
            "Cannot emit a plan with unsupported operators: "
            + operator_validation.describe()
        )

    if ag.fast_memory_reserve_bytes < 0:
        raise ValueError("Fast-memory reservation must not be negative")

    compressed_reserve = compressed_weight_reserve_bytes(ag) if compress else 0
    reserved_activation_budget = ag.fast_memory_reserve_bytes > 0
    if reserved_activation_budget and compressed_reserve > ag.fast_memory_reserve_bytes:
        raise ValueError(
            "Cannot emit a compressed plan: compressed-weight reservation "
            f"({compressed_reserve:,} bytes) exceeds the declared reservation "
            f"({ag.fast_memory_reserve_bytes:,} bytes)"
        )
    if (
        not reserved_activation_budget
        and ag.mem_budget > 0
        and compressed_reserve > ag.mem_budget
    ):
        raise ValueError(
            "Cannot emit a compressed plan: compressed-weight reservation "
            f"({compressed_reserve:,} bytes) exceeds the fast-memory budget "
            f"({ag.mem_budget:,} bytes)"
        )

    if ag.mem_budget > 0:
        validation = validate_memory_plan(
            ag,
            fast_reserve_bytes=0 if reserved_activation_budget else compressed_reserve,
        )
        if not validation.feasible:
            details = "; ".join(issue.describe() for issue in validation.issues)
            raise ValueError(f"Cannot emit an infeasible memory plan: {details}")

    # The serialized plan budget is the activation arena.  The compile command
    # already partitions with the compressed-weight reservation removed.  Keep
    # the defensive direct-emitter behaviour for callers that still provide a
    # total arena budget instead.
    serialized_budget = (
        ag.mem_budget if reserved_activation_budget else ag.mem_budget - compressed_reserve
    )

    # Reject counts before any fixed-width table field is packed. The runtime
    # has a deliberately bounded tensor working set; the other limits are
    # direct consequences of the current wire layout.
    runtime_tensor_count = sum(1 for info in ag.tensors.values() if not info.is_constant)
    if runtime_tensor_count > RUNTIME_MAX_TENSORS:
        raise ValueError(
            f"plan has {runtime_tensor_count} tensors; current runtime loader limit is "
            f"{RUNTIME_MAX_TENSORS}"
        )
    _require_uint(len(ag.weight_data), 16, "weight count")
    _require_uint(len(ag.stages), 16, "stage count")

    strings = _StringTable()
    shapes = _ShapePool()
    index_pool = _IndexPool()

    # Add model name to strings first
    model_name_off = strings.add(ag.model_name)

    # Build weights section (must happen before ops so weight_idx map is ready)
    weight_blocks_data = b""
    if compress and ag.weight_data:
        weight_entries_only, weight_blocks_data, weight_idx = \
            _build_weights_compressed(ag, strings, compress)
        weight_data = weight_entries_only  # entries only, no blob
    else:
        weight_data, weight_idx = _build_weights(ag, strings)

    # Build quant params (must happen before tensors to get quant_idx_map)
    quant_data, quant_idx_map = _build_quant_params(ag)

    # Add model I/O to index pool
    preserve_output_layouts = {
        op.outputs[0]
        for op in ag.ops
        if op.op_type == "Transpose" and len(op.outputs) == 1 and
        op.outputs[0] in ag.model_outputs
    }
    tensor_data, tensor_idx = _build_tensors(
        ag, strings, shapes, quant_idx_map or None, preserve_output_layouts
    )

    model_inp_indices = [tensor_idx[n] for n in ag.model_inputs if n in tensor_idx]
    model_out_indices = [tensor_idx[n] for n in ag.model_outputs if n in tensor_idx]
    model_io_off, model_io_count = index_pool.add(model_inp_indices + model_out_indices)

    op_data = _build_ops(ag, tensor_idx, weight_idx, strings, index_pool)
    op_attributes_data = _build_op_attributes(
        ag, tensor_idx, preserve_output_layouts
    )
    stage_data = bytearray(_build_stages(ag, tensor_idx, index_pool))
    tile_data, stage_to_tile = _build_tile_plans(ag)
    stage_data_final = _patch_stage_tile_indices(stage_data, ag, stage_to_tile)

    index_data = index_pool.data()
    shape_data = shapes.data()
    string_data = strings.data()

    # Counts
    num_tensors = len(tensor_idx)
    num_ops = len(ag.ops)
    num_stages = len(ag.stages)
    num_tile_plans = len(stage_to_tile)
    num_weights = len(weight_idx)

    for value, field in (
        (num_tensors, "tensor count"),
        (num_ops, "operator count"),
        (num_stages, "stage count"),
        (num_tile_plans, "tile-plan count"),
        (num_weights, "weight count"),
    ):
        _require_uint(value, 16, field)
    _require_uint(model_io_off, 16, "model I/O index-pool offset")
    _require_uint(len(model_inp_indices), 8, "model input count")
    _require_uint(len(model_out_indices), 8, "model output count")

    # Section directory (one entry per section + sentinel)
    section_parts = [
        (SEC_TENSORS, tensor_data),
        (SEC_OPS, op_data),
        (SEC_STAGES, stage_data_final),
        (SEC_TILE_PLANS, tile_data),
        (SEC_INDEX_POOL, index_data),
        (SEC_SHAPE_POOL, shape_data),
        (SEC_STRINGS, string_data),
    ]
    if num_weights > 0:
        section_parts.append((SEC_WEIGHTS, weight_data))
    if len(quant_data) > 0:
        section_parts.append((SEC_QUANT_PARAMS, quant_data))
    if len(weight_blocks_data) > 0:
        section_parts.append((SEC_WEIGHT_BLOCKS, weight_blocks_data))
    if op_attributes_data:
        section_parts.append((SEC_OP_ATTRIBUTES, op_attributes_data))

    section_dir_size = (len(section_parts) + 1) * SECTION_ENTRY_SIZE  # +1 for sentinel
    body_start = HEADER_SIZE + section_dir_size

    # Align every section start to WEIGHT_ALIGN. The loader reinterprets several
    # sections zero-copy as int32 arrays (SEC_SHAPE_POOL, the SEC_QUANT_PARAMS
    # multiplier/shift pool) and uint16 (SEC_INDEX_POOL); a section landing at a
    # non-4-byte offset would make those casts unaligned -> UB, and a hard fault
    # on strict-alignment cores (Cortex-M0/M33, -mno-unaligned-access). The file
    # base is WEIGHT_ALIGN-aligned (bin2c / mmap), so aligning each section start
    # to WEIGHT_ALIGN makes every section pointer aligned. SEC_WEIGHTS needs the
    # weight BLOB aligned too: the loader computes
    #   weight_blob = file_base + section_offset + num_weights*WEIGHT_ENTRY_SIZE
    # so pad additionally past the entry table.
    sections = []
    placed_parts = []
    current = body_start
    for sec_id, data in section_parts:
        pad = (-current) % WEIGHT_ALIGN
        if sec_id == SEC_WEIGHTS:
            blob_pos = current + pad + num_weights * WEIGHT_ENTRY_SIZE
            pad += (-blob_pos) % WEIGHT_ALIGN
        current += pad
        sections.append((sec_id, current))
        placed_parts.append(b"\x00" * pad + data)
        current += len(data)
    file_size = current

    # Build section directory
    sec_dir = bytearray()
    for sec_type, sec_off in sections:
        sec_dir.extend(SECTION_ENTRY_STRUCT.pack(sec_type, sec_off))
    sec_dir.extend(SECTION_ENTRY_STRUCT.pack(0, 0))  # sentinel

    # Build header (48 bytes)
    # magic(4) version(u32)
    # file_size(u32) section_dir_offset(u32)
    # num_tensors(u16) num_ops(u16) num_stages(u16) num_tile_plans(u16)
    # budget(u32) peak(u32)
    # model_name_str(u32)
    # model_io_offset(u16) num_model_inputs(u8) num_model_outputs(u8)
    # num_weights(u16) num_quant_params(u16) flags(u32)
    num_qp = len(quant_idx_map) if quant_idx_map else 0
    header_flags = 0
    if xip:
        header_flags |= FLAG_XIP
    header = HEADER_STRUCT.pack(
        MAGIC,
        SCHEMA_VERSION,
        file_size,
        HEADER_SIZE,  # section_dir starts right after header
        num_tensors,
        num_ops,
        num_stages,
        num_tile_plans,
        serialized_budget,
        ag.peak_memory_bytes,
        model_name_off,
        model_io_off,
        len(model_inp_indices),
        len(model_out_indices),
        num_weights,
        num_qp,
        header_flags,
    )
    if len(header) != HEADER_SIZE:
        raise ValueError(f"Header is {len(header)} bytes, expected {HEADER_SIZE}")

    # Assemble
    out = bytearray()
    out.extend(header)
    out.extend(sec_dir)
    for data in placed_parts:
        out.extend(data)

    if len(out) != file_size:
        raise ValueError(f"Output is {len(out)} bytes, expected {file_size}")
    return bytes(out)
