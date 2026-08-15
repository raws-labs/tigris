"""Compiler serializes the HW tile axis and tile_width, and refuses to emit
a plan when even a 1x1 2D core tile does not fit the fast-memory budget.

Byte-level, mirrors test_linebuffer_plan.py: builds a real ONNX model, runs
the shared CLI pipeline, and inspects the emitted binary tile-plan record.
"""

import struct

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

from tigris import TILE_AXIS_HW
from tigris.analysis.validation import validate_memory_plan
from tigris.cli import _run_pipeline
from tigris.emitters.binary.defs import (
    HEADER_STRUCT,
    SEC_TILE_PLANS,
    SECTION_ENTRY_STRUCT,
)
from tigris.emitters.binary.writer import emit_binary_bytes

TILE_PLAN_STRUCT = struct.Struct("<BBHHHHHIII")  # matches tigris_tile_plan_t (24 bytes)


def _build_high_res_conv(h: int, w: int, c: int) -> onnx.ModelProto:
    """A single 3x3 stride-1 pad-1 Conv on a [1,c,h,w] float32 activation.

    Same channel count in and out so the stage's live-tensor peak is exactly
    two copies of the [1,c,h,w] tensor, matching the proportional tile model
    used by solve_2d_tile.
    """
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, c, h, w])
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, c, h, w])
    w0 = helper.make_tensor(
        "w0", TensorProto.FLOAT, [c, c, 3, 3],
        np.zeros((c, c, 3, 3), dtype=np.float32).flatten().tolist(),
    )
    b0 = helper.make_tensor("b0", TensorProto.FLOAT, [c], np.zeros(c, dtype=np.float32).tolist())
    conv0 = helper.make_node(
        "Conv", ["input", "w0", "b0"], ["output"], name="conv0",
        kernel_shape=[3, 3], strides=[1, 1], pads=[1, 1, 1, 1],
    )
    graph = helper.make_graph([conv0], "high_res_conv", [X], [Y], initializer=[w0, b0])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model


def compile_high_res_conv_to_bytes(h: int, w: int, c: int, budget: str, tmp_path) -> bytes:
    """Build the graph, run the shared pipeline, and emit the binary plan bytes.

    Follows the crossrepo _compile_plan pattern: validate feasibility before
    emitting so an infeasible plan raises with a descriptive reason instead
    of being silently written out.
    """
    model_path = tmp_path / "high_res_conv.onnx"
    onnx.save(_build_high_res_conv(h, w, c), str(model_path))
    ag, _ = _run_pipeline(str(model_path), (budget,))
    validation = validate_memory_plan(ag)
    if not validation.feasible:
        details = "; ".join(issue.describe() for issue in validation.issues)
        raise AssertionError(f"compiler produced an infeasible graph: {details}")
    return emit_binary_bytes(ag)


def decode_first_tile_plan(plan_bytes: bytes):
    """Locate the tile-plan section and unpack the first record."""
    header = HEADER_STRUCT.unpack_from(plan_bytes, 0)
    section_dir_off = header[3]

    sections: dict[int, int] = {}
    off = section_dir_off
    while off + SECTION_ENTRY_STRUCT.size <= len(plan_bytes):
        sec_type, sec_off = SECTION_ENTRY_STRUCT.unpack_from(plan_bytes, off)
        off += SECTION_ENTRY_STRUCT.size
        if sec_type == 0:
            break
        sections[sec_type] = sec_off

    tp_base = sections[SEC_TILE_PLANS]
    (
        tileable, axis, tile_height,
        num_tiles, halo,
        receptive_field, original_height,
        tiled_peak_bytes, overhead_bytes, reserved,
    ) = TILE_PLAN_STRUCT.unpack_from(plan_bytes, tp_base)

    class _TilePlan:
        pass

    tp = _TilePlan()
    tp.tileable = tileable
    tp.axis = axis
    tp.tile_height = tile_height
    tp.tile_width = reserved & 0xFFFF
    tp.num_tiles = num_tiles
    tp.halo = halo
    tp.receptive_field = receptive_field
    tp.original_height = original_height
    tp.tiled_peak_bytes = tiled_peak_bytes
    tp.overhead_bytes = overhead_bytes
    return tp


def test_2d_plan_encodes_hw_axis_and_tile_width(tmp_path):
    plan_bytes = compile_high_res_conv_to_bytes(h=256, w=256, c=256, budget="24K", tmp_path=tmp_path)
    tp = decode_first_tile_plan(plan_bytes)
    assert tp.tileable == 1
    assert tp.axis == TILE_AXIS_HW
    assert tp.tile_height >= 1 and tp.tile_width >= 1


def test_even_2d_min_tile_infeasible_fails_closed(tmp_path):
    # Budget so small even a 1x1 2D core will not fit -> compile refuses.
    with pytest.raises(Exception) as exc:
        compile_high_res_conv_to_bytes(h=64, w=64, c=4096, budget="1K", tmp_path=tmp_path)
    assert "2D tile" in str(exc.value)
