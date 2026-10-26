"""Compiler marks recomputing chain heads as line_buffered and serializes
the flag into the head stage's flags field.

Compile-side only: this test verifies the compiler model (Stage.line_buffered)
and the emitted binary bytes, not any runtime behavior.
"""

from tigris.cli import _run_pipeline
from tigris.emitters.binary.defs import STAGE_FLAG_LINE_BUFFERED
from tigris.emitters.binary.reader import read_binary_plan
from tigris.emitters.binary.writer import emit_binary_bytes
from scripts.contract_cases.fixtures import build_ds_cnn


def _chain_heads(ag):
    return [s for s in ag.stages if s.chain_len >= 2 and s.chain_id == s.stage_id]


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

    stages = read_binary_plan(emit_binary_bytes(ag))["stages"]
    for stage in ag.stages:
        flags = stages[stage.stage_id]["flags"]
        if stage.line_buffered:
            assert flags & STAGE_FLAG_LINE_BUFFERED
        else:
            assert not (flags & STAGE_FLAG_LINE_BUFFERED)

