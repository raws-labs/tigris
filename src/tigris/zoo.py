"""Catalog selection and verified model-zoo downloads."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

from packaging.version import InvalidVersion, Version

REPOSITORY = "raws-labs/tigris-zoo"
_ID = re.compile(r"[a-z0-9][a-z0-9._-]*\Z")
_SHA = re.compile(r"[0-9a-f]{64}\Z")


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def version(value: str) -> Version:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", value):
        raise ValueError("runtime versions must be release versions such as 0.9.1")
    try:
        return Version(value)
    except InvalidVersion as exc:
        raise ValueError(str(exc)) from exc


def _identifier(value) -> None:
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise ValueError(f"invalid catalog identifier: {value!r}")


def _integer(value, label: str, minimum: int = 0) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")


def timestamp(value: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError("published_at must be a UTC timestamp ending in Z")
    try:
        return datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ValueError("invalid published_at timestamp") from exc


def validate_runtime(runtime: dict) -> None:
    if not isinstance(runtime, dict) or set(runtime) != {"min", "max"}:
        raise ValueError("runtime requires min and max (null when no upper bound is known)")
    minimum = version(runtime["min"])
    if runtime["max"] is not None and minimum > version(runtime["max"]):
        raise ValueError("runtime range must have min <= max")


def runtime_matches(runtime: dict, requested: Version) -> bool:
    return version(runtime["min"]) <= requested and (
        runtime["max"] is None or requested <= version(runtime["max"])
    )


def validate_artifact(artifact: dict, *, catalog: bool = False) -> dict:
    """Validate the shared manifest fields before using any paths or filters."""
    try:
        if type(artifact["format_version"]) is not int or artifact["format_version"] != 1:
            raise ValueError("unsupported artifact format_version")
        for field in ("id", "model", "category", "quantization"):
            _identifier(artifact[field])
        timestamp(artifact["published_at"])
        _integer(artifact["schema"], "schema", 1)
        validate_runtime(artifact["runtime"])
        if not isinstance(artifact["backends"], list) or not artifact["backends"]:
            raise ValueError("backends must be a nonempty list")
        for backend in artifact["backends"]:
            _identifier(backend)
        for field in ("fast_bytes", "slow_bytes", "flash_bytes"):
            _integer(artifact["memory"][field], field)
        for field in ("inputs", "outputs"):
            if not isinstance(artifact[field], list) or not artifact[field]:
                raise ValueError(f"{field} must be a nonempty list")
            names = set()
            for tensor in artifact[field]:
                if not isinstance(tensor["name"], str) or not tensor["name"] or tensor["name"] in names:
                    raise ValueError("tensor names must be nonempty and unique")
                names.add(tensor["name"])
                _identifier(tensor["dtype"])
                if not isinstance(tensor["shape"], list) or not tensor["shape"]:
                    raise ValueError("tensor shape must be a nonempty list")
                for dim in tensor["shape"]:
                    _integer(dim, "tensor dimension", 1)
        if not isinstance(artifact["license"], str) or not artifact["license"]:
            raise ValueError("license must be specified")
        for field in ("compiler", "source", "evaluation"):
            if not isinstance(artifact[field], dict) or not artifact[field]:
                raise ValueError(f"{field} must be a nonempty object")
        files = artifact["files"]
        if not isinstance(files, list) or not files:
            raise ValueError("files must be a nonempty list")
        paths = set()
        for item in files:
            # Artifact files are flat, portable names, never relative paths.
            _identifier(item["path"])
            if item["path"] in ("manifest.json", "download.json") or item["path"] in paths:
                raise ValueError("duplicate or reserved artifact filename")
            paths.add(item["path"])
            _integer(item["size"], "file size")
            if not isinstance(item["sha256"], str) or not _SHA.fullmatch(item["sha256"]):
                raise ValueError("invalid file SHA-256")
        if "model.tgrs" not in paths:
            raise ValueError("artifact must include model.tgrs")
        if catalog:
            tested = artifact["tested_runtime_versions"]
            if not isinstance(tested, list) or len(set(tested)) != len(tested):
                raise ValueError("tested_runtime_versions must be a list of unique releases")
            for tested_version in tested:
                version(tested_version)
            note = artifact["compatibility_note"]
            if not isinstance(note, str) or not note.strip():
                raise ValueError("compatibility_note must explain the runtime constraints")
            if not isinstance(artifact["manifest_sha256"], str) or not _SHA.fullmatch(artifact["manifest_sha256"]):
                raise ValueError("invalid manifest SHA-256")
            withdrawn = artifact.get("withdrawn")
            if withdrawn is not None and (not isinstance(withdrawn, str) or not withdrawn.strip()):
                raise ValueError("withdrawn must be null or a reason")
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError(f"malformed artifact metadata: {exc}") from exc
    return artifact


def validate_catalog(data: dict) -> list[dict]:
    if not isinstance(data, dict) or type(data.get("format_version")) is not int or data["format_version"] != 1:
        raise ValueError("unsupported catalog format_version")
    artifacts = data.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("catalog artifacts must be a list")
    ids = set()
    for artifact in artifacts:
        validate_artifact(artifact, catalog=True)
        if artifact["id"] in ids:
            raise ValueError(f"duplicate artifact ID: {artifact['id']}")
        ids.add(artifact["id"])
    return artifacts


def select(artifacts: list[dict], *, model=None, category=None, runtime=None,
           backend=None, quantization=None, fast=None, slow=None, flash=None,
           artifact_id=None) -> list[dict]:
    requested = version(runtime) if runtime is not None else None
    matches = []
    for item in artifacts:
        if item.get("withdrawn") and artifact_id is None:
            continue
        if any(wanted is not None and item[field] != wanted for field, wanted in (
            ("id", artifact_id), ("model", model), ("category", category), ("quantization", quantization),
        )):
            continue
        if requested is not None and not runtime_matches(item["runtime"], requested):
            continue
        if backend is not None and backend not in item["backends"]:
            continue
        if any(limit is not None and item["memory"][field] > limit for field, limit in (
            ("fast_bytes", fast), ("slow_bytes", slow), ("flash_bytes", flash),
        )):
            continue
        matches.append(item)
    return sorted(matches, key=lambda item: (timestamp(item["published_at"]), item["id"]), reverse=True)


def artifact_path(artifact: dict) -> str:
    return f"models/{artifact['model']}/artifacts/{artifact['id']}"


def manifest_fields(artifact: dict) -> dict:
    """Immutable fields shared by the build manifest and catalog entry."""
    mutable = {"runtime", "tested_runtime_versions", "compatibility_note", "manifest_sha256", "withdrawn"}
    return {key: value for key, value in artifact.items() if key not in mutable}


class Zoo:
    def __init__(self, *, catalog: Path | None = None, repository: str = REPOSITORY,
                 revision: str = "main", offline: bool = False, cache_dir: Path | None = None):
        self.local = catalog.resolve().parent if catalog else None
        self.repository = repository
        self.revision = revision
        self.offline = offline
        self.cache_dir = cache_dir
        path = catalog if catalog else self._download("catalog.json")
        if not catalog:
            # Hub cache snapshots resolve branch names to immutable commits.
            self.revision = path.parent.name
            if not re.fullmatch(r"[0-9a-f]{40}", self.revision):
                raise ValueError("Hub did not return a pinned catalog snapshot")
        self.catalog_sha256 = digest(path)
        self.artifacts = validate_catalog(json.loads(path.read_text(encoding="utf-8")))

    def _download(self, filename: str) -> Path:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import EntryNotFoundError, LocalEntryNotFoundError
        from huggingface_hub.utils import logging as hub_logging

        # The CLI reports failures; SDK warnings can suggest unnecessary login.
        verbosity = hub_logging.get_verbosity()
        hub_logging.set_verbosity_error()
        try:
            return Path(hf_hub_download(
                repo_id=self.repository, filename=filename, revision=self.revision,
                token=False, local_files_only=self.offline, cache_dir=self.cache_dir,
            ))
        except LocalEntryNotFoundError as exc:
            raise ValueError("Zoo file unavailable; check the connection or populate the cache before using --offline") from exc
        except EntryNotFoundError as exc:
            raise ValueError(f"Zoo file is not published at this revision: {filename}") from exc
        finally:
            hub_logging.set_verbosity(verbosity)

    def _file(self, filename: str) -> Path:
        if self.local:
            path = (self.local / filename).resolve()
            if not path.is_relative_to(self.local):
                raise ValueError("artifact path escapes the local catalog")
            return path
        return self._download(filename)

    def fetch(self, artifact: dict, destination: Path) -> Path:
        destination = destination.absolute()
        if destination.exists() or destination.is_symlink():
            raise ValueError(f"destination already exists: {destination}")
        base = artifact_path(artifact)
        manifest_path = self._file(f"{base}/manifest.json")
        if digest(manifest_path) != artifact["manifest_sha256"]:
            raise ValueError("manifest checksum mismatch")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        validate_artifact(manifest)
        if manifest_fields(manifest) != manifest_fields(artifact):
            raise ValueError("catalog and manifest disagree")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".tigris-zoo-", dir=destination.parent) as temporary:
            stage = Path(temporary) / "artifact"
            stage.mkdir()
            shutil.copyfile(manifest_path, stage / "manifest.json")
            for item in artifact["files"]:
                source = self._file(f"{base}/{item['path']}")
                target = stage / item["path"]
                shutil.copyfile(source, target)
                if target.stat().st_size != item["size"] or digest(target) != item["sha256"]:
                    raise ValueError(f"checksum or size mismatch: {item['path']}")
            receipt = {
                "artifact_id": artifact["id"], "repository": None if self.local else self.repository,
                "revision": None if self.local else self.revision,
                "catalog_sha256": self.catalog_sha256, "withdrawn": artifact.get("withdrawn"),
                "runtime": artifact["runtime"],
                "tested_runtime_versions": artifact["tested_runtime_versions"],
                "compatibility_note": artifact["compatibility_note"],
            }
            (stage / "download.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
            if destination.exists() or destination.is_symlink():
                raise ValueError(f"destination already exists: {destination}")
            stage.rename(destination)
        return destination / "model.tgrs"
