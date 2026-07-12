"""Audited kernel capability source and documentation rows."""

from tigris.capabilities import (
    KERNEL_CAPABILITIES,
    capability_rows,
    describe_codegen_route,
    effective_operators,
    operator_route,
)
from tigris.emitters.binary.defs import OP_TYPE_MAP


def test_capability_source_is_schema_consistent():
    assert set(KERNEL_CAPABILITIES) == {
        "reference",
        "s8_ref",
        "esp-nn",
        "cmsis-nn",
    }

    for name, capability in KERNEL_CAPABILITIES.items():
        assert capability.name == name
        assert capability.native_operators <= OP_TYPE_MAP.keys()
        if capability.fallback is not None:
            assert capability.fallback in KERNEL_CAPABILITIES
            assert KERNEL_CAPABILITIES[capability.fallback].dtype == capability.dtype


def test_accelerated_fallbacks_are_explicit():
    assert KERNEL_CAPABILITIES["esp-nn"].fallback == "s8_ref"
    assert KERNEL_CAPABILITIES["cmsis-nn"].fallback == "s8_ref"

    assert operator_route("esp-nn", "AveragePool") == "esp-nn"
    assert operator_route("cmsis-nn", "AveragePool") == "cmsis-nn"
    assert operator_route("reference", "AveragePool") is None
    assert operator_route("s8_ref", "AveragePool") == "s8_ref"
    assert operator_route("esp-nn", "Relu") == "s8_ref"
    assert operator_route("cmsis-nn", "MaxPool") == "s8_ref"
    assert operator_route("esp-nn", "MatMul") is None
    assert "MatMul" not in effective_operators("cmsis-nn")


def test_capability_rows_are_stable_and_docs_ready():
    rows = capability_rows()

    assert len(rows) == len(OP_TYPE_MAP)
    assert [row["opcode"] for row in rows] == sorted(OP_TYPE_MAP.values())

    by_operator = {row["operator"]: row for row in rows}
    assert by_operator["AveragePool"] == {
        "operator": "AveragePool",
        "opcode": OP_TYPE_MAP["AveragePool"],
        "reference": "unsupported",
        "s8_ref": "native",
        "esp-nn": "native",
        "cmsis-nn": "native",
    }
    assert by_operator["Relu"]["esp-nn"] == "fallback:s8_ref"
    assert by_operator["GlobalAveragePool"]["esp-nn"] == "fallback:s8_ref"
    assert by_operator["GlobalAveragePool"]["cmsis-nn"] == "native"
    assert all(
        by_operator["MatMul"][backend] == "unsupported"
        for backend in KERNEL_CAPABILITIES
    )


def test_codegen_route_descriptions_do_not_imply_float_acceleration():
    assert describe_codegen_route("reference", "int8") == "s8_ref"
    assert describe_codegen_route("esp-nn", "int8") == "esp-nn -> s8_ref fallback"
    assert describe_codegen_route("cmsis-nn", "int8") == (
        "cmsis-nn -> s8_ref fallback"
    )
    assert describe_codegen_route("esp-nn", "float32") == (
        "reference (explicit float32 fallback; esp-nn acceleration is int8-only)"
    )
