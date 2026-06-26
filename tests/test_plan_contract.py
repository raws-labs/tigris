"""Binary format contract tests.

Verify that the Python writer produces .tgrs files with known
field values. The same values are checked by test_plan_contract.c
in tigris-runtime.
"""

import struct
import tempfile
from pathlib import Path

import onnx

from tigris import SCHEMA_VERSION
from tigris.cli import _run_pipeline
from tigris.fixtures import build_linear_3op, build_conv_relu_chain, build_ds_cnn
from tigris.emitters.binary.defs import MAGIC, HEADER_SIZE
from tigris.emitters.binary.writer import emit_binary_bytes


# Helpers


def _compile_fixture(builder, budget_str):
    """Build ONNX model, run pipeline, return binary plan bytes."""
    model = builder()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.onnx"
        onnx.save(model, str(path))
        ag, _ = _run_pipeline(str(path), (budget_str,))
        return emit_binary_bytes(ag)


def _parse_header(data):
    """Parse the 48-byte file header into a dict."""
    assert len(data) >= HEADER_SIZE
    h = {}
    h["magic"] = data[:4]
    h["version"] = struct.unpack_from("<H", data, 4)[0]
    h["file_size"] = struct.unpack_from("<I", data, 8)[0]
    h["num_tensors"] = struct.unpack_from("<H", data, 16)[0]
    h["num_ops"] = struct.unpack_from("<H", data, 18)[0]
    h["num_stages"] = struct.unpack_from("<H", data, 20)[0]
    h["num_tile_plans"] = struct.unpack_from("<H", data, 22)[0]
    h["budget"] = struct.unpack_from("<I", data, 24)[0]
    h["peak"] = struct.unpack_from("<I", data, 28)[0]
    h["num_model_inputs"] = struct.unpack_from("<B", data, 38)[0]
    h["num_model_outputs"] = struct.unpack_from("<B", data, 39)[0]
    h["num_weights"] = struct.unpack_from("<H", data, 40)[0]
    return h


# Contract values
# These must match the assertions in test_plan_contract.c exactly.
# If the compiler changes output, update BOTH files.

CONTRACT = {
    "linear_3op": {
        "builder": build_linear_3op,
        "budget_str": "4K",
        "version": 1,
        "num_tensors": 4,
        "num_ops": 3,
        "num_stages": 1,
        "num_tile_plans": 0,
        "budget": 4096,
        "num_model_inputs": 1,
        "num_model_outputs": 1,
        "num_weights": 2,
    },
    "conv_relu_chain": {
        "builder": build_conv_relu_chain,
        # 64K: the real working-set peak is 53,888 B (conv1 holds its [1,8,30,30]
        # input co-resident with its [1,8,28,28] output - see test_memory). 32K
        # would (correctly) force multi-stage slow-overflow; 64K keeps this a
        # clean single-stage plan, which is what this format contract checks.
        "budget_str": "64K",
        "version": 1,
        "num_tensors": 3,
        "num_ops": 2,        # Relu fused into first Conv
        "num_stages": 1,
        "num_tile_plans": 0,
        "budget": 65536,
        "num_model_inputs": 1,
        "num_model_outputs": 1,
        "num_weights": 4,
    },
    "ds_cnn": {
        "builder": build_ds_cnn,
        "budget_str": "256K",
        "version": 1,
        "num_tensors": 13,
        "num_ops": 12,
        "num_stages": 1,
        "num_tile_plans": 0,
        "budget": 262144,
        "num_model_inputs": 1,
        "num_model_outputs": 1,
        "num_weights": 20,
    },
}


# Tests


def test_header_size():
    """File header must be exactly 48 bytes."""
    assert HEADER_SIZE == 48


def test_schema_version():
    """Schema version constant must be 1."""
    assert SCHEMA_VERSION == 1


def test_linear_3op_contract():
    """Verify linear_3op plan header matches contract values."""
    spec = CONTRACT["linear_3op"]
    data = _compile_fixture(spec["builder"], spec["budget_str"])
    h = _parse_header(data)

    assert h["magic"] == MAGIC
    assert h["version"] == spec["version"]
    assert h["file_size"] == len(data)
    assert h["num_tensors"] == spec["num_tensors"]
    assert h["num_ops"] == spec["num_ops"]
    assert h["num_stages"] == spec["num_stages"]
    assert h["num_tile_plans"] == spec["num_tile_plans"]
    assert h["budget"] == spec["budget"]
    assert h["num_model_inputs"] == spec["num_model_inputs"]
    assert h["num_model_outputs"] == spec["num_model_outputs"]
    assert h["num_weights"] == spec["num_weights"]
    assert h["peak"] > 0


def test_conv_relu_chain_contract():
    """Verify conv_relu_chain plan header matches contract values."""
    spec = CONTRACT["conv_relu_chain"]
    data = _compile_fixture(spec["builder"], spec["budget_str"])
    h = _parse_header(data)

    assert h["magic"] == MAGIC
    assert h["version"] == spec["version"]
    assert h["file_size"] == len(data)
    assert h["num_tensors"] == spec["num_tensors"]
    assert h["num_ops"] == spec["num_ops"]
    assert h["num_stages"] == spec["num_stages"]
    assert h["num_tile_plans"] == spec["num_tile_plans"]
    assert h["budget"] == spec["budget"]
    assert h["num_model_inputs"] == spec["num_model_inputs"]
    assert h["num_model_outputs"] == spec["num_model_outputs"]
    assert h["num_weights"] == spec["num_weights"]
    assert h["peak"] > 0


def test_ds_cnn_contract():
    """Verify ds_cnn plan header matches contract values."""
    spec = CONTRACT["ds_cnn"]
    data = _compile_fixture(spec["builder"], spec["budget_str"])
    h = _parse_header(data)

    assert h["magic"] == MAGIC
    assert h["version"] == spec["version"]
    assert h["file_size"] == len(data)
    assert h["num_tensors"] == spec["num_tensors"]
    assert h["num_ops"] == spec["num_ops"]
    assert h["num_stages"] == spec["num_stages"]
    assert h["num_tile_plans"] == spec["num_tile_plans"]
    assert h["budget"] == spec["budget"]
    assert h["num_model_inputs"] == spec["num_model_inputs"]
    assert h["num_model_outputs"] == spec["num_model_outputs"]
    assert h["num_weights"] == spec["num_weights"]
    assert h["peak"] > 0


def test_file_sizes_deterministic():
    """Verify that plan generation is deterministic (same input -> same bytes)."""
    for name, spec in CONTRACT.items():
        data1 = _compile_fixture(spec["builder"], spec["budget_str"])
        data2 = _compile_fixture(spec["builder"], spec["budget_str"])
        assert data1 == data2, f"{name}: non-deterministic plan output"
