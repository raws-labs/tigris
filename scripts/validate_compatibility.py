#!/usr/bin/env python3
"""Validate coordinated compiler/runtime release and schema metadata."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

from tigris import SCHEMA_VERSION, SUPPORTED_SCHEMA_VERSIONS


_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
_TAG_RE = re.compile(r"v\d+\.\d+\.\d+")
_RUNTIME_SCHEMA_RE = re.compile(
    r"^#define\s+(TIGRIS_SCHEMA_VERSION\w*)\s+(\w+)\s*$",
    re.MULTILINE,
)
_COMPILER_SCHEMA_RE = re.compile(r"^SCHEMA_VERSION\s*=\s*(\d+)\s*$", re.MULTILINE)


def runtime_schemas(header: str) -> list[int]:
    """TIGRIS_SCHEMA_VERSION_MIN through TIGRIS_SCHEMA_VERSION of a runtime
    header, following macro aliases."""
    defines = dict(_RUNTIME_SCHEMA_RE.findall(header))

    def resolve(name: str) -> int:
        for _ in range(len(defines)):
            if name.isdigit():
                break
            name = defines[name]
        return int(name)

    return list(
        range(resolve("TIGRIS_SCHEMA_VERSION_MIN"), resolve("TIGRIS_SCHEMA_VERSION") + 1)
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
        "release_cadence": "paired-same-version",
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
    # Releases pair since v0.11.0, but an entry may name one component (earlier
    # releases, the runtime-only v0.11.1); schemas, not versions, are checked.
    seen_tags: dict[str, set[str]] = {"compiler": set(), "runtime": set()}
    emitted_schemas: set[int] = set()
    accepted_schema_sets: list[list[int]] = []
    for index, release in enumerate(releases):
        label = f"releases[{index}]"
        if not isinstance(release, dict):
            errors.append(f"{label} must be an object")
            continue
        status = release.get("status")
        if status not in {"supported", "retired"}:
            errors.append(f"{label}.status must be supported or retired")
        components = {
            name: release[name]
            for name in ("compiler", "runtime")
            if release.get(name) is not None
        }
        if not components:
            errors.append(f"{label} must name a compiler or a runtime object")
            continue
        if any(not isinstance(data, dict) for data in components.values()):
            errors.append(f"{label} compiler and runtime must be objects")
            continue
        for name, data in components.items():
            tag = data.get("tag")
            if not isinstance(tag, str) or not _TAG_RE.fullmatch(tag):
                errors.append(f"{label}.{name}.tag is not a release tag")
            elif tag in seen_tags[name]:
                errors.append(f"duplicate {name} release {tag}")
            else:
                seen_tags[name].add(tag)
            commit = data.get("commit")
            if not isinstance(commit, str) or not _COMMIT_RE.fullmatch(commit):
                errors.append(f"{label}.{name}.commit must be a full Git SHA")
        release_emitted = None
        if "compiler" in components:
            release_emitted = components["compiler"].get("emits_schema")
            if not isinstance(release_emitted, int) or release_emitted < 1:
                errors.append(f"{label}.compiler.emits_schema must be positive")
                release_emitted = None
            elif status == "supported":
                emitted_schemas.add(release_emitted)
        release_accepted: list[int] = []
        if "runtime" in components:
            release_accepted = _positive_schema_list(
                components["runtime"].get("accepts_schemas"),
                f"{label}.runtime.accepts_schemas",
                errors,
            )
            if release_accepted and status == "supported":
                accepted_schema_sets.append(release_accepted)
        if (
            release_emitted is not None
            and release_accepted
            and release_emitted not in release_accepted
        ):
            errors.append(f"{label} pairs an incompatible compiler and runtime")

    for schema in sorted(emitted_schemas):
        if not any(schema in accepted for accepted in accepted_schema_sets):
            errors.append(
                f"no supported runtime release accepts compiler schema {schema}"
            )

    if runtime is not None:
        header = runtime / "include" / "tigris.h"
        try:
            header_schemas = runtime_schemas(header.read_text())
        except (OSError, KeyError, ValueError) as exc:
            errors.append(f"cannot read runtime schema range: {exc!r}")
        else:
            if header_schemas != accepted:
                errors.append(
                    "runtime header accepts schemas "
                    f"{header_schemas}, manifest declares {accepted}"
                )
    return errors


def _git(repo: Path, *args: str) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=False
    )
    return result.stdout if result.returncode == 0 else None


def record(root: Path, runtime: Path, version: str, status: str) -> int:
    """Add the release entry for compiler and runtime tag vVERSION, newest first.
    Both sides are read at their tags, so the runtime checkout only needs the tag."""
    tag = f"v{version}"
    manifest = root / "compatibility.json"
    original = manifest.read_text()
    document = json.loads(original)
    if any(
        (release.get("compiler") or {}).get("tag") == tag
        and (release.get("runtime") or {}).get("tag") == tag
        for release in document["releases"]
    ):
        print(f"compatibility.json already records {tag}.")
        return 0
    commits = {}
    for name, repo in (("compiler", root), ("runtime", runtime)):
        commit = _git(repo, "rev-parse", "--verify", "--quiet", f"{tag}^{{commit}}")
        if commit is None:
            print(f"ERROR: {name} checkout {repo} has no tag {tag}")
            return 1
        commits[name] = commit.strip()
    emitted = _COMPILER_SCHEMA_RE.search(
        _git(root, "show", f"{tag}:src/tigris/__init__.py") or ""
    )
    try:
        accepted = runtime_schemas(_git(runtime, "show", f"{tag}:include/tigris.h") or "")
    except (KeyError, ValueError) as exc:
        print(f"ERROR: cannot read the runtime schema range at {tag}: {exc!r}")
        return 1
    if emitted is None:
        print(f"ERROR: cannot read the compiler SCHEMA_VERSION at {tag}")
        return 1
    document["releases"].insert(0, {
        "status": status,
        "compiler": {"tag": tag, "commit": commits["compiler"], "emits_schema": int(emitted[1])},
        "runtime": {"tag": tag, "commit": commits["runtime"], "accepts_schemas": accepted},
    })
    manifest.write_text(json.dumps(document, indent=2) + "\n")
    errors = validate(root)
    if errors:
        manifest.write_text(original)
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print(f"Recorded {tag} in compatibility.json.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--runtime", type=Path)
    parser.add_argument(
        "--record", metavar="X.Y.Z", help="add the release entry for the vX.Y.Z tags"
    )
    parser.add_argument("--status", choices=("supported", "retired"), default="supported")
    args = parser.parse_args(argv)
    if args.record is not None:
        if not re.fullmatch(r"\d+\.\d+\.\d+", args.record):
            parser.error("--record takes a version X.Y.Z")
        if args.runtime is None:
            parser.error("--record needs --runtime")
        return record(args.root.resolve(), args.runtime.resolve(), args.record, args.status)
    errors = validate(args.root.resolve(), args.runtime.resolve() if args.runtime else None)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print("Compiler/runtime compatibility manifest is valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
