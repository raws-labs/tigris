#!/usr/bin/env python3
"""Validate coordinated compiler/runtime release and schema metadata."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from tigris import SCHEMA_VERSION, SUPPORTED_SCHEMA_VERSIONS


_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
_TAG_RE = re.compile(r"v\d+\.\d+\.\d+")
_RUNTIME_SCHEMA_RE = re.compile(
    r"^#define\s+TIGRIS_SCHEMA_VERSION(?:_V\d+)?\s+(\d+)\s*$",
    re.MULTILINE,
)


def _positive_schema_list(value: object, label: str, errors: list[str]) -> list[int]:
    if not isinstance(value, list) or not value:
        errors.append(f"{label} must be a non-empty list")
        return []
    if any(not isinstance(item, int) or item < 1 for item in value):
        errors.append(f"{label} must contain positive integers")
        return []
    result = list(value)
    if result != sorted(set(result)):
        errors.append(f"{label} must be sorted and duplicate-free")
    return result


def validate(root: Path, runtime: Path | None = None) -> list[str]:
    errors: list[str] = []
    manifest_path = root / "compatibility.json"
    try:
        document = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return [f"cannot read {manifest_path}: {exc}"]

    if document.get("format_version") != 1:
        errors.append("format_version must be 1")
    policy = document.get("policy")
    if not isinstance(policy, dict):
        return errors + ["policy must be an object"]
    expected_policy = {
        "repository_model": "separate-repositories",
        "schema_owner": "raws-labs/tigris",
        "schema_versioning": "wire-format",
        "capability_growth": "same-schema with fail-closed runtime validation",
        "runtime_dependency": "vendored-wire-contract",
        "integration_branch": "develop",
    }
    for key, expected in expected_policy.items():
        if policy.get(key) != expected:
            errors.append(f"policy.{key} must be {expected!r}")
    for key in ("schema_source", "schema_artifact"):
        value = policy.get(key)
        if not isinstance(value, str) or not (root / value).is_file():
            errors.append(f"policy.{key} must name a tracked file")

    integration = document.get("integration")
    if not isinstance(integration, dict):
        return errors + ["integration must be an object"]
    emitted = integration.get("compiler_emits_schema")
    accepted = _positive_schema_list(
        integration.get("runtime_accepts_schemas"),
        "integration.runtime_accepts_schemas",
        errors,
    )
    if accepted != list(SUPPORTED_SCHEMA_VERSIONS):
        errors.append(
            "integration.runtime_accepts_schemas must match the compiler reader's "
            f"supported schemas {list(SUPPORTED_SCHEMA_VERSIONS)}"
        )
    if emitted != SCHEMA_VERSION:
        errors.append(
            "integration.compiler_emits_schema must match tigris.SCHEMA_VERSION"
        )
    if isinstance(emitted, int) and emitted not in accepted:
        errors.append("integration runtime does not accept the compiler schema")
    for key in ("compiler_branch", "runtime_branch"):
        if integration.get(key) != policy.get("integration_branch"):
            errors.append(f"integration.{key} must follow the integration branch")
    if integration.get("contract_gate") != (
        "python scripts/crossrepo_contract.py --runtime ../tigris-runtime"
    ):
        errors.append("integration.contract_gate must name the canonical gate")

    artifact_path = root / str(policy.get("schema_artifact", ""))
    try:
        artifact = json.loads(artifact_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"cannot read schema artifact: {exc}")
    else:
        if artifact.get("schema_version") != emitted:
            errors.append("schema artifact and integration schema disagree")

    releases = document.get("releases")
    if not isinstance(releases, list) or not releases:
        errors.append("releases must be a non-empty list")
        return errors
    compiler_tags: set[str] = set()
    for index, release in enumerate(releases):
        label = f"releases[{index}]"
        if not isinstance(release, dict):
            errors.append(f"{label} must be an object")
            continue
        if release.get("status") not in {"supported", "retired"}:
            errors.append(f"{label}.status must be supported or retired")
        compiler = release.get("compiler")
        runtime_release = release.get("runtime")
        if not isinstance(compiler, dict) or not isinstance(runtime_release, dict):
            errors.append(f"{label} must name compiler and runtime objects")
            continue
        compiler_tag = compiler.get("tag")
        runtime_tag = runtime_release.get("tag")
        if not isinstance(compiler_tag, str) or not _TAG_RE.fullmatch(compiler_tag):
            errors.append(f"{label}.compiler.tag is not a release tag")
        elif compiler_tag in compiler_tags:
            errors.append(f"duplicate compiler release {compiler_tag}")
        else:
            compiler_tags.add(compiler_tag)
        if not isinstance(runtime_tag, str) or not _TAG_RE.fullmatch(runtime_tag):
            errors.append(f"{label}.runtime.tag is not a release tag")
        for component, data in (("compiler", compiler), ("runtime", runtime_release)):
            commit = data.get("commit")
            if not isinstance(commit, str) or not _COMMIT_RE.fullmatch(commit):
                errors.append(f"{label}.{component}.commit must be a full Git SHA")
        release_emitted = compiler.get("emits_schema")
        release_accepted = _positive_schema_list(
            runtime_release.get("accepts_schemas"),
            f"{label}.runtime.accepts_schemas",
            errors,
        )
        if not isinstance(release_emitted, int) or release_emitted < 1:
            errors.append(f"{label}.compiler.emits_schema must be positive")
        elif release_emitted not in release_accepted:
            errors.append(f"{label} pairs an incompatible compiler and runtime")

    if runtime is not None:
        header = runtime / "include" / "tigris.h"
        try:
            runtime_schemas = sorted(
                {int(value) for value in _RUNTIME_SCHEMA_RE.findall(header.read_text())}
            )
        except OSError as exc:
            errors.append(f"cannot read runtime schema header: {exc}")
        else:
            if runtime_schemas != accepted:
                errors.append(
                    "runtime header accepts schemas "
                    f"{runtime_schemas}, manifest declares {accepted}"
                )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--runtime", type=Path)
    args = parser.parse_args()
    errors = validate(args.root.resolve(), args.runtime.resolve() if args.runtime else None)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print("Compiler/runtime compatibility manifest is valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
