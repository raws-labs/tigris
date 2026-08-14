"""Compiler marks recomputing chain heads as line_buffered and serializes
the flag into the head stage's _reserved1 field.

Compile-side only: this test verifies the compiler model (Stage.line_buffered)
and the emitted binary bytes, not any runtime behavior.
"""

from tigris.cli import _run_pipeline
from tigris.emitters.binary.defs import (
    HEADER_STRUCT,
    SEC_STAGES,
    SECTION_ENTRY_STRUCT,
    STAGE_FLAG_LINE_BUFFERED,
    STAGE_STRUCT,
)
from tigris.emitters.binary.writer import emit_binary_bytes
from tigris.fixtures import build_ds_cnn, build_linear_3op


def _chain_heads(ag):
    return [s for s in ag.stages if s.chain_len >= 2 and s.chain_id == s.stage_id]


def _stage_reserved1(plan_bytes: bytes, stage_index: int) -> int:
    """Decode the _reserved1 field (offset 26 in the 28-byte record) of the
    stage at ``stage_index`` directly from the emitted binary plan bytes."""
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

    s_base = sections[SEC_STAGES]
    pos = s_base + stage_index * STAGE_STRUCT.size
    fields = STAGE_STRUCT.unpack_from(plan_bytes, pos)
    return fields[-1]  # _reserved1 is the last packed field


def test_recomputing_chain_head_is_line_buffered(tmp_path):
    import onnx
    m = tmp_path / "dscnn.onnx"
    onnx.save(build_ds_cnn(), str(m))
    ag, _ = _run_pipeline(str(m), ("60K",))

    heads = _chain_heads(ag)
    assert heads, "expected a chain to form for DS-CNN at 60K"
    assert all(h.line_buffered for h in heads)
    # non-head / non-chain stages must not be flagged
    assert all(
        not s.line_buffered for s in ag.stages
        if not (s.chain_len >= 2 and s.chain_id == s.stage_id)
    )

    plan_bytes = emit_binary_bytes(ag)
    for stage in ag.stages:
        reserved1 = _stage_reserved1(plan_bytes, stage.stage_id)
        if stage.line_buffered:
            assert reserved1 & STAGE_FLAG_LINE_BUFFERED
        else:
            assert not (reserved1 & STAGE_FLAG_LINE_BUFFERED)


def test_no_chain_plan_has_reserved1_clear(tmp_path):
    """A plan with no chain should have the flag bit clear everywhere."""
    import onnx
    m = tmp_path / "linear3op.onnx"
    onnx.save(build_linear_3op(), str(m))
    ag, _ = _run_pipeline(str(m), ("60K",))

    assert not _chain_heads(ag), "expected no chain for a pointwise-only model"
    assert all(not s.line_buffered for s in ag.stages)

    plan_bytes = emit_binary_bytes(ag)
    for stage in ag.stages:
        reserved1 = _stage_reserved1(plan_bytes, stage.stage_id)
        assert reserved1 == 0
