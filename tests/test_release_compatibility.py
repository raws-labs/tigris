"""Machine-readable coordinated-release and schema-package contract."""

from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path

from scripts.generate_schema_package import schema_package
from scripts.validate_compatibility import validate


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT.parent / "tigris-runtime"


def test_schema_package_matches_compiler_definitions():
    artifact = json.loads(
        (ROOT / "src/tigris/schema/tigris-plan-v4.json").read_text()
    )
    assert artifact == schema_package()


def test_schema_package_is_installed_package_data():
    artifact = files("tigris").joinpath("schema/tigris-plan-v4.json")
    assert json.loads(artifact.read_text()) == schema_package()


def test_release_manifest_matches_current_runtime_when_available():
    runtime = RUNTIME if (RUNTIME / "include/tigris.h").is_file() else None
    assert validate(ROOT, runtime) == []


def test_release_manifest_rejects_incompatible_pair(tmp_path):
    document = json.loads((ROOT / "compatibility.json").read_text())
    document["releases"][0]["runtime"]["accepts_schemas"] = [2, 3]
    (tmp_path / "compatibility.json").write_text(json.dumps(document))
    for relative in (
        "src/tigris/schema/tigris-plan-v4.json",
        "src/tigris/emitters/binary/defs.py",
    ):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / relative).read_bytes())

    errors = validate(tmp_path)

    assert any("incompatible compiler and runtime" in error for error in errors)
