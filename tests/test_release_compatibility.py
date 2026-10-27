"""Machine-readable coordinated-release and schema-package contract."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tigris import SCHEMA_VERSION
from scripts.generate_capability_matrix import encoded_matrix
from scripts.generate_schema_package import encoded_package
from scripts.validate_compatibility import validate
from scripts.validate_runtime_capabilities import validate as validate_capabilities


ROOT = Path(__file__).resolve().parents[1]
_RUNTIME_SOURCE = os.environ.get("TIGRIS_RUNTIME_SOURCE")
RUNTIME = Path(_RUNTIME_SOURCE).resolve() if _RUNTIME_SOURCE else ROOT.parent / "tigris-runtime"
# Derived, so a schema bump does not quietly leave these tests checking the
# previous version's artifact.
SCHEMA_ARTIFACT = f"schema/tigris-plan-v{SCHEMA_VERSION}.json"


def test_schema_package_matches_compiler_definitions():
    assert (ROOT / "src/tigris" / SCHEMA_ARTIFACT).read_text() == encoded_package()


def test_capability_artifact_matches_compiler_definitions():
    artifact = ROOT / "src/tigris/schema/operator-capabilities-v1.json"
    assert artifact.read_text() == encoded_matrix()


def test_capability_audit_imports_from_an_uninstalled_source_checkout():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from tigris.capabilities import KERNEL_CAPABILITIES; "
            "assert 'reference' in KERNEL_CAPABILITIES",
        ],
        cwd=ROOT,
        env={"PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_release_manifest_matches_current_runtime_when_available():
    runtime = RUNTIME if (RUNTIME / "include/tigris.h").is_file() else None
    if runtime is None and _RUNTIME_SOURCE:
        pytest.fail(f"TIGRIS_RUNTIME_SOURCE has no runtime headers: {RUNTIME}")
    assert validate(ROOT, runtime) == []


def test_capability_contract_matches_current_runtime_when_available():
    if not (RUNTIME / "include/tigris.h").is_file():
        if _RUNTIME_SOURCE:
            pytest.fail(f"TIGRIS_RUNTIME_SOURCE has no runtime headers: {RUNTIME}")
        pytest.skip("tigris-runtime checkout not found; set TIGRIS_RUNTIME_SOURCE")
    assert validate_capabilities(RUNTIME) == []


def _runtime_copy(tmp_path):
    if not (RUNTIME / "include/tigris.h").is_file():
        if _RUNTIME_SOURCE:
            pytest.fail(f"TIGRIS_RUNTIME_SOURCE has no runtime headers: {RUNTIME}")
        pytest.skip("tigris-runtime checkout not found; set TIGRIS_RUNTIME_SOURCE")
    copy = tmp_path / "runtime"
    for part in ("include", "src"):
        shutil.copytree(RUNTIME / part, copy / part)
    return copy


def _edit(path, pattern, replacement=""):
    text, count = re.subn(pattern, replacement, path.read_text())
    assert count, f"{pattern} not found in {path.name}"
    path.write_text(text)


def _dispatch_error(backend, operator):
    return (
        f"{backend} dispatch differs from compiler capabilities: "
        f"runtime-only=[], compiler-only=['{operator}']"
    )


@pytest.mark.parametrize(
    ("operator", "label"),
    [("Pad", "PAD"), ("Gather", "GATHER"), ("Split", "SPLIT")],
)
def test_capability_contract_rejects_reference_dispatch_without_an_operator(
    tmp_path, operator, label,
):
    runtime = _runtime_copy(tmp_path)
    removed = 0
    for name in ("tigris_kernels.c", "tigris_kernels_s8.c"):
        source = runtime / "src" / name
        text, count = re.subn(rf"\bcase\s+TIGRIS_OP_{label}\s*:", "", source.read_text())
        source.write_text(text)
        removed += count
    assert removed

    assert validate_capabilities(runtime) == [
        _dispatch_error("reference", operator),
        _dispatch_error("s8_ref", operator),
    ]


def test_capability_contract_rejects_int8_dispatch_without_its_own_case(tmp_path):
    runtime = _runtime_copy(tmp_path)
    _edit(runtime / "src/tigris_kernels_s8.c", r"\bcase\s+TIGRIS_OP_PAD\s*:")

    assert validate_capabilities(runtime) == [_dispatch_error("s8_ref", "Pad")]


@pytest.mark.parametrize(
    ("backend", "name", "function"),
    [
        ("reference", "tigris_kernels.c", "tigris_dispatch_kernel"),
        ("s8_ref", "tigris_kernels_s8.c", "tigris_dispatch_kernel_s8"),
    ],
)
def test_capability_contract_counts_a_shared_switch_only_when_the_dispatcher_reaches_it(
    tmp_path, backend, name, function,
):
    runtime = _runtime_copy(tmp_path)
    source = runtime / "src" / name
    dispatcher = re.search(
        rf"\b{function}\s*\([^;]*?\)\s*\{{.*?\n\}}", source.read_text(), re.S,
    ).group(0)
    if not re.search(r"\bdefault\s*:\s*return\s+tigris_dispatch_", dispatcher):
        pytest.skip(f"{function} has no default case returning into a shared switch")
    _edit(
        source,
        r"\bdefault\s*:\s*return\s+tigris_dispatch_[a-z0-9_]+\s*\([^;]*\);",
        "default: return -1;",
    )

    errors = validate_capabilities(runtime)

    assert len(errors) == 1
    assert errors[0].startswith(f"{backend} dispatch differs from compiler capabilities: ")
    assert "runtime-only=[]" in errors[0]
    assert "'Gather'" in errors[0]


@pytest.mark.parametrize("header,old,new,error", [
    ("tigris_config.h", "#define TIGRIS_MAX_CHAIN_STAGES 16u", "#define TIGRIS_MAX_CHAIN_STAGES 32u",
     "tigris_config.h TIGRIS_MAX_CHAIN_STAGES defaults to [32], the compiler to 16"),
    ("tigris.h", "#define TIGRIS_SEC_MEMORY         14", "#define TIGRIS_SEC_MEMORY         15",
     "tigris.h TIGRIS_SEC_MEMORY is [15], defs.py SEC_MEMORY is 14"),
    ("tigris.h", "sizeof(tigris_weight_block_t) == 20", "sizeof(tigris_weight_block_t) == 24",
     "tigris.h asserts sizeof(tigris_weight_block_t) == [24], defs.py has 20"),
    ("tigris.h", "2u * (size_t)(candidates)", "3u * (size_t)(candidates)",
     "TIGRIS_DETECTION_SCRATCH_BYTES(1, 1, 1) is 57, the compiler's is 53"),
], ids=["build-limit-default", "wire-constant", "record-size", "detection-scratch"])
def test_capability_contract_names_a_drifted_runtime_constant(tmp_path, header, old, new, error):
    if not (RUNTIME / "include/tigris.h").is_file():
        if _RUNTIME_SOURCE:
            pytest.fail(f"TIGRIS_RUNTIME_SOURCE has no runtime headers: {RUNTIME}")
        pytest.skip("tigris-runtime checkout not found; set TIGRIS_RUNTIME_SOURCE")
    for part in ("include", "src"):
        shutil.copytree(RUNTIME / part, tmp_path / part)
    path = tmp_path / "include" / header
    text = path.read_text()
    assert text.count(old) == 1
    path.write_text(text.replace(old, new))
    assert error in validate_capabilities(tmp_path)


def _document():
    return json.loads((ROOT / "compatibility.json").read_text())


def _validated_copy(tmp_path, document):
    (tmp_path / "compatibility.json").write_text(json.dumps(document))
    for relative in (
        f"src/tigris/{SCHEMA_ARTIFACT}",
        "src/tigris/emitters/binary/defs.py",
    ):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / relative).read_bytes())
    return validate(tmp_path)


def test_release_manifest_copy_validates_unchanged(tmp_path):
    assert _validated_copy(tmp_path, _document()) == []


def test_release_manifest_rejects_incompatible_pair(tmp_path):
    document = _document()
    pair = next(
        release
        for release in document["releases"]
        if "compiler" in release and "runtime" in release
    )
    pair["runtime"]["accepts_schemas"] = [2, 3]

    errors = _validated_copy(tmp_path, document)

    assert any("incompatible compiler and runtime" in error for error in errors)


def test_release_manifest_accepts_an_independent_component_release():
    document = _document()
    assert any(
        "compiler" not in release or "runtime" not in release
        for release in document["releases"]
    ), "manifest no longer exercises an independently released component"


def test_release_manifest_rejects_an_entry_naming_no_component(tmp_path):
    document = _document()
    document["releases"].append({"status": "supported"})

    errors = _validated_copy(tmp_path, document)

    assert any(
        "must name a compiler or a runtime object" in error for error in errors
    )


def test_release_manifest_rejects_a_repeated_runtime_release(tmp_path):
    document = _document()
    entry = next(release for release in document["releases"] if "runtime" in release)
    document["releases"].append(json.loads(json.dumps(entry)))

    errors = _validated_copy(tmp_path, document)

    assert any("duplicate runtime release" in error for error in errors)


def test_release_manifest_rejects_a_compiler_schema_no_runtime_accepts(tmp_path):
    document = _document()
    for release in document["releases"]:
        if "runtime" in release:
            release["runtime"]["accepts_schemas"] = [2]

    errors = _validated_copy(tmp_path, document)

    assert any(
        "no supported runtime release accepts compiler schema" in error
        for error in errors
    )
