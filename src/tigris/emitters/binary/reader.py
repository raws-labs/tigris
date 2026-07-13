"""Binary plan reader - deserialization for test validation."""

import struct

from tigris import SCHEMA_VERSION

from .defs import (
    HEADER_SIZE,
    MAGIC,
    OP_SIZE,
    QUANT_PARAM_SIZE,
    SECTION_ENTRY_SIZE,
    SEC_INDEX_POOL,
    SEC_OPS,
    SEC_QUANT_PARAMS,
    SEC_SHAPE_POOL,
    SEC_STAGES,
    SEC_STRINGS,
    SEC_TENSORS,
    SEC_TILE_PLANS,
    SEC_WEIGHT_BLOCKS,
    SEC_WEIGHTS,
    STAGE_SIZE,
    TENSOR_SIZE,
    TILE_PLAN_SIZE,
    WEIGHT_BLOCK_SIZE,
    WEIGHT_ENTRY_SIZE,
)


def read_binary_plan(data: bytes) -> dict:
    """Parse a binary plan file into a dict for test validation.

    Returns a dict with keys: magic, version, file_size, num_tensors, num_ops,
    num_stages, num_tile_plans, budget, peak, model_name, tensors, ops, stages,
    tile_plans, model_inputs, model_outputs.
    """
    if len(data) < HEADER_SIZE:
        raise ValueError(f"File too small: {len(data)} < {HEADER_SIZE}")

    # Parse header
    (
        magic, version,
        file_size, section_dir_off,
        num_tensors, num_ops, num_stages, num_tile_plans,
        budget, peak,
        model_name_str,
        model_io_off, num_model_inputs, num_model_outputs,
        num_weights, num_quant_params, flags,
    ) = struct.unpack_from("<4sI II HHHH II I HBB HH I", data, 0)

    if magic != MAGIC:
        raise ValueError(f"Bad magic: {magic!r}")
    if version not in {2, SCHEMA_VERSION}:
        raise ValueError(
            f"Unsupported schema version: {version} (expected 2 or {SCHEMA_VERSION})"
        )
    if file_size != len(data):
        raise ValueError(f"File size mismatch: header says {file_size}, got {len(data)}")

    # Parse section directory
    sections: dict[int, int] = {}
    off = section_dir_off
    found_sentinel = False
    while off + SECTION_ENTRY_SIZE <= len(data):
        sec_type, sec_off = struct.unpack_from("<II", data, off)
        off += SECTION_ENTRY_SIZE
        if sec_type == 0:
            found_sentinel = True
            break
        if sec_type > SEC_WEIGHT_BLOCKS:
            raise ValueError(f"Unknown section type: {sec_type}")
        if sec_type in sections:
            raise ValueError(f"Duplicate section type: {sec_type}")
        if sec_off > len(data):
            raise ValueError(
                f"Section {sec_type} offset {sec_off} exceeds file size {len(data)}"
            )
        sections[sec_type] = sec_off

    if not found_sentinel:
        raise ValueError("Section directory has no sentinel")

    required_sections = {
        SEC_TENSORS,
        SEC_OPS,
        SEC_INDEX_POOL,
        SEC_SHAPE_POOL,
        SEC_STRINGS,
    }
    if num_stages:
        required_sections.add(SEC_STAGES)
    if num_tile_plans:
        required_sections.add(SEC_TILE_PLANS)
    if num_weights:
        required_sections.add(SEC_WEIGHTS)
    if num_quant_params:
        required_sections.add(SEC_QUANT_PARAMS)
    missing = sorted(
        sec_type
        for sec_type in required_sections
        if not sections.get(sec_type)
    )
    if missing:
        raise ValueError(f"Missing required section type(s): {missing}")

    def _read_string(str_off: int) -> str:
        base = sections[SEC_STRINGS]
        pos = base + str_off
        end = data.index(0, pos)
        return data[pos:end].decode("utf-8")

    def _read_index_pool(pool_off: int, count: int) -> list[int]:
        base = sections[SEC_INDEX_POOL]
        pos = base + pool_off * 2
        return list(struct.unpack_from(f"<{count}H", data, pos))

    def _read_shape(shape_off: int, ndim: int) -> list[int]:
        base = sections[SEC_SHAPE_POOL]
        pos = base + shape_off * 4
        return list(struct.unpack_from(f"<{ndim}i", data, pos))

    model_name = _read_string(model_name_str)

    # Parse tensors
    tensors = []
    t_base = sections[SEC_TENSORS]
    for i in range(num_tensors):
        pos = t_base + i * TENSOR_SIZE
        name_off, size_bytes, shape_off, ndim, dtype, t_flags, qp_idx = struct.unpack_from(
            "<IIHBBBHx", data, pos
        )
        tensors.append({
            "name": _read_string(name_off),
            "size_bytes": size_bytes,
            "shape": _read_shape(shape_off, ndim),
            "ndim": ndim,
            "dtype": dtype,
            "flags": t_flags,
            "quant_param_idx": qp_idx,
        })

    # Parse ops
    ops = []
    o_base = sections[SEC_OPS]
    for i in range(num_ops):
        pos = o_base + i * OP_SIZE
        name_off, op_type, num_inp, num_out, stage, inp_off, out_off = struct.unpack_from(
            "<IBBBB HH", data, pos
        )
        spatial_pos = pos + 12  # after the first 12 bytes
        (kh, kw, sh, sw, pt, pb, pl, pr, dh, dw, group) = struct.unpack_from(
            "<4B7H", data, spatial_pos  # 18-byte spatial (pad/dilation u16)
        )
        weight_idx_val, bias_idx_val = struct.unpack_from("<HH", data, pos + 30)
        fused_act, act_min, act_max = struct.unpack_from("<Bbbx", data, pos + 34)
        ops.append({
            "name": _read_string(name_off),
            "op_type": op_type,
            "inputs": _read_index_pool(inp_off, num_inp),
            "outputs": _read_index_pool(out_off, num_out),
            "stage": stage,
            "spatial": {
                "kernel_h": kh, "kernel_w": kw,
                "stride_h": sh, "stride_w": sw,
                "pad_top": pt, "pad_bottom": pb,
                "pad_left": pl, "pad_right": pr,
                "dilation_h": dh, "dilation_w": dw,
                "group": group,
            },
            "weight_idx": weight_idx_val,
            "bias_idx": bias_idx_val,
            "fused_act": fused_act,
            "act_min": act_min,
            "act_max": act_max,
        })

    # Parse stages
    stages = []
    s_base = sections.get(SEC_STAGES, 0)
    for i in range(num_stages):
        pos = s_base + i * STAGE_SIZE
        (
            peak_bytes,
            ops_off, ops_count,
            inp_off, inp_count,
            out_off, out_count,
            tile_idx, _pad,
            chain_id, chain_len,
            chain_tile_h, _reserved1,
        ) = struct.unpack_from("<I HHHHHH HH HHHH", data, pos)
        stages.append({
            "peak_bytes": peak_bytes,
            "ops": _read_index_pool(ops_off, ops_count),
            "inputs": _read_index_pool(inp_off, inp_count),
            "outputs": _read_index_pool(out_off, out_count),
            "tile_plan_idx": tile_idx,
            "chain_id": chain_id,
            "chain_len": chain_len,
            "chain_tile_h": chain_tile_h,
        })

    # Parse tile plans
    tile_plans = []
    tp_base = sections.get(SEC_TILE_PLANS, 0)
    for i in range(num_tile_plans):
        pos = tp_base + i * TILE_PLAN_SIZE
        (
            tileable, _pad, tile_height,
            n_tiles, halo,
            rf, orig_h,
            tiled_peak, overhead, _reserved,
        ) = struct.unpack_from("<BBH HH HH I I I", data, pos)
        tile_plans.append({
            "tileable": bool(tileable),
            "tile_height": tile_height,
            "num_tiles": n_tiles,
            "halo": halo,
            "receptive_field": rf,
            "original_height": orig_h,
            "tiled_peak_bytes": tiled_peak,
            "overhead_bytes": overhead,
        })

    # Parse weights
    weights = []
    w_base = sections.get(SEC_WEIGHTS, 0)
    if w_base and num_weights > 0:
        entries_size = num_weights * WEIGHT_ENTRY_SIZE
        blob_base = w_base + entries_size
        for i in range(num_weights):
            pos = w_base + i * WEIGHT_ENTRY_SIZE
            w_name_off, w_offset, w_size = struct.unpack_from("<III", data, pos)
            weights.append({
                "name": _read_string(w_name_off),
                "offset": w_offset,
                "size_bytes": w_size,
                "blob_base": blob_base,
            })

    # Parse quant params (optional)
    quant_params = []
    qp_base = sections.get(SEC_QUANT_PARAMS, 0)
    if qp_base:
        nqp, qd_field = struct.unpack_from("<HH", data, qp_base)
        entries_start = qp_base + 4
        data_start = entries_start + nqp * QUANT_PARAM_SIZE
        if version == 2:
            qd_len = qd_field
        else:
            quant_end = min(
                (offset for offset in sections.values() if offset > qp_base),
                default=len(data),
            )
            qd_bytes = quant_end - data_start
            if qd_bytes < 0 or qd_bytes % 4:
                raise ValueError("Malformed v3 quant-data section")
            qd_len = qd_bytes // 4
        for i in range(nqp):
            pos = entries_start + i * QUANT_PARAM_SIZE
            scale, zp, num_ch, mult_off, shift_off, page = struct.unpack_from(
                "<fiHHHH", data, pos
            )
            qp_entry = {
                "scale": scale,
                "zero_point": zp,
                "num_channels": num_ch,
                "multiplier_off": mult_off,
                "shift_off": shift_off,
            }
            if num_ch > 1:
                # Read per-channel multipliers and shifts
                page_base = 0 if version == 2 else page * (1 << 16)
                m_pos = data_start + (page_base + mult_off) * 4
                s_pos = data_start + (page_base + shift_off) * 4
                qp_entry["multipliers"] = list(struct.unpack_from(f"<{num_ch}i", data, m_pos))
                qp_entry["shifts"] = list(struct.unpack_from(f"<{num_ch}i", data, s_pos))
            quant_params.append(qp_entry)

    # Parse weight blocks (optional - compressed plans)
    weight_blocks = []
    weight_blocks_compression = 0
    wb_base = sections.get(SEC_WEIGHT_BLOCKS, 0)
    if wb_base:
        num_blocks, compression_type = struct.unpack_from("<HH", data, wb_base)
        weight_blocks_compression = compression_type
        entries_start = wb_base + 4
        blobs_start = entries_start + num_blocks * WEIGHT_BLOCK_SIZE
        for i in range(num_blocks):
            pos = entries_start + i * WEIGHT_BLOCK_SIZE
            (stage_idx, first_widx, num_w, _pad,
             blob_off, comp_sz, uncomp_sz) = struct.unpack_from(
                "<HHHH III", data, pos)
            block_entry = {
                "stage_idx": stage_idx,
                "first_weight_idx": first_widx,
                "num_weights": num_w,
                "blob_offset": blob_off,
                "compressed_size": comp_sz,
                "uncompressed_size": uncomp_sz,
            }
            # Decompress for validation
            comp_data = data[blobs_start + blob_off : blobs_start + blob_off + comp_sz]
            if compression_type == 1:  # LZ4
                import lz4.block
                decompressed = lz4.block.decompress(
                    comp_data, uncompressed_size=uncomp_sz)
                assert len(decompressed) == uncomp_sz
                block_entry["decompressed"] = decompressed
            else:
                block_entry["decompressed"] = comp_data
            weight_blocks.append(block_entry)

    # Resolve model I/O from index pool
    all_io = _read_index_pool(model_io_off, num_model_inputs + num_model_outputs)
    model_inputs_idx = all_io[:num_model_inputs]
    model_outputs_idx = all_io[num_model_inputs:]

    return {
        "magic": magic,
        "version": version,
        "file_size": file_size,
        "num_tensors": num_tensors,
        "num_ops": num_ops,
        "num_stages": num_stages,
        "num_tile_plans": num_tile_plans,
        "num_weights": num_weights,
        "num_quant_params": num_quant_params,
        "flags": flags,
        "budget": budget,
        "peak": peak,
        "model_name": model_name,
        "tensors": tensors,
        "ops": ops,
        "stages": stages,
        "tile_plans": tile_plans,
        "weights": weights,
        "quant_params": quant_params,
        "model_inputs": model_inputs_idx,
        "model_outputs": model_outputs_idx,
        "weight_blocks": weight_blocks,
        "weight_blocks_compression": weight_blocks_compression,
    }
