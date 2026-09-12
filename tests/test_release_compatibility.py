"""Machine-readable coordinated-release and schema-package contract."""

from __future__ import annotations

import json
import subprocess
import sys
from importlib.resources import files
from pathlib import Path

from scripts.generate_capability_matrix import capability_matrix
from scripts.generate_schema_package import schema_package
from scripts.validate_compatibility import validate
from scripts.validate_runtime_capabilities import validate as validate_capabilities


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT.parent / "tigris-runtime"


def test_schema_package_matches_compiler_definitions():
    artifact = json.loads(
        (ROOT / "src/tigris/schema/tigris-plan-v5.json").read_text()
    )
    assert artifact == schema_package()


def test_schema_package_is_installed_package_data():
    artifact = files("tigris").joinpath("schema/tigris-plan-v5.json")
    assert json.loads(artifact.read_text()) == schema_package()


def test_capability_artifact_matches_compiler_definitions():
    artifact = json.loads(
        (ROOT / "src/tigris/schema/operator-capabilities-v1.json").read_text()
    )
    assert artifact == capability_matrix()


def test_capability_artifact_is_installed_package_data():
    artifact = files("tigris").joinpath(
        "schema/operator-capabilities-v1.json"
    )
    assert json.loads(artifact.read_text()) == capability_matrix()


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
    assert validate(ROOT, runtime) == []


def test_capability_contract_matches_current_runtime_when_available():
    if (RUNTIME / "include/tigris.h").is_file():
        assert validate_capabilities(RUNTIME) == []


def _document():
    return json.loads((ROOT / "compatibility.json").read_text())


def _validated_copy(tmp_path, document):
    (tmp_path / "compatibility.json").write_text(json.dumps(document))
    for relative in (
        "src/tigris/schema/tigris-plan-v5.json",
        "src/tigris/emitters/binary/defs.py",
    ):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / relative).read_bytes())
    return validate(tmp_path)


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
