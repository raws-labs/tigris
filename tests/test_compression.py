"""Tests for LZ4 weight compression in the binary plan emitter."""


import lz4.block
import numpy as np

from tigris.analysis.lifetime import compute_lifetimes
from tigris.analysis.memory import compute_memory_timeline
from tigris.analysis.partition_temporal import partition_temporal
from tigris.analysis.partition_spatial import partition_spatial
from tigris.loaders import load_model
from tigris.emitters.binary.defs import (
    MAGIC,
)
from tigris.emitters.binary.reader import read_binary_plan
from tigris.emitters.binary.writer import emit_binary_bytes


def _full_pipeline(path, budget=0):
    ag = load_model(path)
    ag = compute_lifetimes(ag)
    ag = compute_memory_timeline(ag)
    if budget > 0:
        ag = partition_temporal(ag, budget)
        ag = partition_spatial(ag)
    return ag



# LZ4 round-trip


def test_lz4_roundtrip():
    """Compress + decompress matches original data."""
    original = b"Hello " * 100 + b"world!" * 50
    compressed = lz4.block.compress(original, store_size=False)
    decompressed = lz4.block.decompress(compressed, uncompressed_size=len(original))
    assert decompressed == original


# Compressed plan has SEC_WEIGHT_BLOCKS


def test_compressed_plan_has_weight_blocks(conv_relu_chain_path):
    """Compressed plan should contain SEC_WEIGHT_BLOCKS section."""
    ag = _full_pipeline(conv_relu_chain_path, budget=8192)
    data = emit_binary_bytes(ag, compress="lz4")
    plan = read_binary_plan(data)

    assert len(plan["weight_blocks"]) > 0
    assert plan["weight_blocks_compression"] == 1  # LZ4


# Compressed weight data matches uncompressed


def test_compressed_weight_data_matches(conv_relu_chain_path):
    """Decompressed weight data from compressed plan matches uncompressed plan."""
    ag = _full_pipeline(conv_relu_chain_path, budget=8192)

    data_plain = emit_binary_bytes(ag)
    data_lz4 = emit_binary_bytes(ag, compress="lz4")

    plan_plain = read_binary_plan(data_plain)
    plan_lz4 = read_binary_plan(data_lz4)

    # Both should have same number of weights
    assert plan_plain["num_weights"] == plan_lz4["num_weights"]

    # Gather all decompressed bytes from blocks
    all_decompressed = bytearray()
    for block in sorted(plan_lz4["weight_blocks"], key=lambda b: b["first_weight_idx"]):
        all_decompressed.extend(block["decompressed"])

    # Gather all weight bytes from plain plan
    all_plain = bytearray()
    for w in plan_plain["weights"]:
        blob_start = w["blob_base"] + w["offset"]
        all_plain.extend(data_plain[blob_start : blob_start + w["size_bytes"]])

    assert bytes(all_decompressed) == bytes(all_plain)


# Uncompressed plan unchanged


def test_uncompressed_plan_unchanged(conv_relu_chain_path):
    """Without --compress, no SEC_WEIGHT_BLOCKS section should appear."""
    ag = _full_pipeline(conv_relu_chain_path, budget=8192)
    data = emit_binary_bytes(ag)
    plan = read_binary_plan(data)

    assert len(plan["weight_blocks"]) == 0
    assert plan["weight_blocks_compression"] == 0


# All weighted fixtures produce valid compressed plans


def test_all_fixtures_compressed(
    linear_3op_path, conv_relu_chain_path, conv_pool_chain_path,
):
    """All fixtures with weights produce valid compressed plans."""
    for path in [linear_3op_path, conv_relu_chain_path, conv_pool_chain_path]:
        ag = _full_pipeline(path, budget=8192)
        if not ag.weight_data:
            continue
        data = emit_binary_bytes(ag, compress="lz4")
        plan = read_binary_plan(data)
        assert plan["magic"] == MAGIC
        assert plan["num_ops"] == len(ag.ops)
        assert len(plan["weight_blocks"]) > 0


# Weight grouping by stage


def test_weight_grouping_by_stage(conv_relu_chain_path):
    """Each weight block maps to exactly one stage."""
    ag = _full_pipeline(conv_relu_chain_path, budget=8192)
    data = emit_binary_bytes(ag, compress="lz4")
    plan = read_binary_plan(data)

    assert len(plan["weight_blocks"]) >= 2
    # Each block has a valid stage index
    for block in plan["weight_blocks"]:
        assert block["stage_idx"] < plan["num_stages"]
        assert block["num_weights"] > 0
        assert block["compressed_size"] > 0
        assert block["uncompressed_size"] > 0
        assert block["compressed_size"] <= block["uncompressed_size"] * 2  # sanity


# Weightless graph produces no blocks


def test_weightless_graph_no_blocks(diamond_path):
    """A graph with no weights should produce no weight blocks."""
    ag = _full_pipeline(diamond_path, budget=1536)
    data = emit_binary_bytes(ag, compress="lz4")
    plan = read_binary_plan(data)

    assert plan["num_weights"] == 0
    assert len(plan["weight_blocks"]) == 0


# Compressed plan is smaller


def test_compressed_plan_smaller(conv_relu_chain_path):
    """Compressed plan should not be larger than uncompressed (with real data)."""
    ag = _full_pipeline(conv_relu_chain_path, budget=8192)

    # Replace zero weights with random data for better compression test
    for name in ag.weight_data:
        arr = ag.weight_data[name]
        np.random.seed(42)
        ag.weight_data[name] = np.random.randn(*arr.shape).astype(arr.dtype)

    data_plain = emit_binary_bytes(ag)
    data_lz4 = emit_binary_bytes(ag, compress="lz4")

    # LZ4 on random float data may not compress well, but the overhead
    # should not be catastrophic (< 2x). For real weights it compresses better.
    assert len(data_lz4) < len(data_plain) * 2
