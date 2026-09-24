import copy
import hashlib
import json

import pytest
from click.testing import CliRunner

from tigris.cli import cli
from tigris.zoo import Zoo, artifact_path, digest, select, validate_catalog


def artifact(artifact_id="example-a", **changes):
    item = {
        "format_version": 1, "id": artifact_id, "model": "example",
        "category": "classification", "quantization": "float32",
        "published_at": "2026-01-01T00:00:00Z", "schema": 7,
        "runtime": {"min": "0.9.1", "max": "0.9.3"},
        "backends": ["reference"],
        "memory": {"fast_bytes": 1024, "slow_bytes": 0, "flash_bytes": 4},
        "inputs": [{"name": "input", "shape": [1, 4], "dtype": "float32"}],
        "outputs": [{"name": "output", "shape": [1, 2], "dtype": "float32"}],
        "license": "apache-2.0", "compiler": {"version": "0.8.0"},
        "source": {"model": "example"}, "evaluation": {"test_cases": 1},
        "files": [{"path": "model.tgrs", "size": 4,
                   "sha256": hashlib.sha256(b"plan").hexdigest()}],
    }
    item.update(changes)
    return item


def snapshot(root, items):
    entries = []
    for item in items:
        item = copy.deepcopy(item)
        withdrawn = item.pop("withdrawn", None)
        directory = root / artifact_path(item)
        directory.mkdir(parents=True)
        manifest = directory / "manifest.json"
        manifest.write_text(json.dumps(item))
        (directory / "model.tgrs").write_bytes(b"plan")
        entries.append(dict(item, manifest_sha256=digest(manifest), withdrawn=withdrawn,
                            tested_runtime_versions=["0.9.1"], compatibility_note="Release requirements."))
    catalog = root / "catalog.json"
    catalog.write_text(json.dumps({"format_version": 1, "artifacts": entries}))
    return catalog


def test_filters_precede_recency_and_do_not_require_runtime():
    older = artifact()
    newer = artifact("example-b", published_at="2026-02-01T00:00:00Z",
                     runtime={"min": "0.10.0", "max": "0.11.0"})
    assert select([older, newer])[0]["id"] == "example-b"
    assert select([older, newer], runtime="0.9.1")[0]["id"] == "example-a"
    assert select([older, newer], runtime="0.9.0") == []
    assert select([older], runtime="0.10.0") == []
    assert select([older], runtime="0.9.3") == [older]
    assert select([older], runtime="0.9.4") == []
    assert select([newer], runtime="0.10.0") == [newer]
    assert select([newer], runtime="0.11.0") == [newer]
    assert select([older], category="detection") == []
    assert select([older], backend="cmsis-nn") == []
    assert select([older], quantization="int8") == []
    assert select([older], fast=1023) == []
    assert select([older], fast=1024, slow=0, flash=4) == [older]
    assert select([older], flash=3) == []


def test_single_supported_release_is_a_valid_range(tmp_path):
    catalog = snapshot(tmp_path, [artifact(runtime={"min": "0.9.1", "max": "0.9.1"})])
    source = Zoo(catalog=catalog)
    assert select(source.artifacts, runtime="0.9.1") == source.artifacts
    assert select(source.artifacts, runtime="0.9.0") == []
    assert select(source.artifacts, runtime="0.9.2") == []


def test_unknown_upper_bound_does_not_filter_out_untested_releases(tmp_path):
    catalog = snapshot(tmp_path, [artifact(runtime={"min": "0.9.1", "max": None})])
    source = Zoo(catalog=catalog)
    assert select(source.artifacts, runtime="0.9.0") == []
    assert select(source.artifacts, runtime="0.10.0") == source.artifacts
    assert source.artifacts[0]["tested_runtime_versions"] == ["0.9.1"]
    result = CliRunner().invoke(cli, ["zoo", "--catalog", str(catalog), "list", "--runtime", "0.10.0"])
    assert result.exit_code == 0, result.output
    assert "no known upper bound" in result.output
    assert "tested runtimes: 0.9.1" in result.output


def test_download_uses_current_compatibility_without_rewriting_manifest(tmp_path):
    catalog = snapshot(tmp_path / "source", [artifact()])
    data = json.loads(catalog.read_text())
    item = data["artifacts"][0]
    manifest = catalog.parent / artifact_path(item) / "manifest.json"
    original_manifest = manifest.read_bytes()
    item["runtime"]["max"] = None
    item["tested_runtime_versions"].append("0.10.0")
    item["compatibility_note"] = "No known incompatibility above the minimum."
    catalog.write_text(json.dumps(data))
    source = Zoo(catalog=catalog)
    destination = tmp_path / "download"
    source.fetch(select(source.artifacts, runtime="0.10.0")[0], destination)
    assert (destination / "manifest.json").read_bytes() == original_manifest
    receipt = json.loads((destination / "download.json").read_text())
    assert receipt["runtime"] == {"min": "0.9.1", "max": None}
    assert receipt["tested_runtime_versions"] == ["0.9.1", "0.10.0"]
    assert receipt["compatibility_note"] == item["compatibility_note"]
    assert source.artifacts[0]["published_at"] == "2026-01-01T00:00:00Z"


def test_ties_withdrawals_and_exact_ids():
    a, b = artifact(), artifact("example-b")
    assert select([a, b])[0] == b
    b["withdrawn"] = "Incorrect output on affected inputs"
    assert select([a, b]) == [a]
    assert select([a, b], artifact_id=b["id"]) == [b]
    assert select([b], artifact_id=b["id"], backend="cmsis-nn") == []


@pytest.mark.parametrize("mutation", [
    lambda a: a.update(id="../escape"),
    lambda a: a.update(model="/absolute"),
    lambda a: a["files"][0].update(path="../outside"),
    lambda a: a["files"][0].update(path="a\\b"),
    lambda a: a["files"][0].update(path="manifest.json"),
    lambda a: a["files"].append(a["files"][0].copy()),
    lambda a: a["files"][0].update(size=True),
    lambda a: a["memory"].update(fast_bytes=-1),
    lambda a: a["runtime"].update(max="0.9.0"),
    lambda a: a.update(runtime={"min": "0.9.1", "max_exclusive": "0.9.2"}),
    lambda a: a["runtime"].update(min="0.9.1rc1"),
    lambda a: a.update(published_at="2026-01-01"),
    lambda a: a.update(schema=True),
    lambda a: a.update(manifest_sha256="invalid"),
    lambda a: a.update(withdrawn=True),
    lambda a: a.update(inputs=[]),
    lambda a: a.update(tested_runtime_versions=["bad"]),
    lambda a: a.update(tested_runtime_versions=["0.9.1", "0.9.1"]),
    lambda a: a.update(compatibility_note=""),
])
def test_invalid_catalog_is_rejected(tmp_path, mutation):
    catalog = snapshot(tmp_path, [artifact()])
    data = json.loads(catalog.read_text())
    mutation(data["artifacts"][0])
    with pytest.raises(ValueError):
        validate_catalog(data)


def test_duplicate_ids_are_rejected(tmp_path):
    catalog = snapshot(tmp_path, [artifact()])
    data = json.loads(catalog.read_text())
    data["artifacts"] *= 2
    with pytest.raises(ValueError, match="duplicate artifact ID"):
        validate_catalog(data)


def test_fetch_preserves_manifest_receipt_and_existing_files(tmp_path):
    catalog = snapshot(tmp_path / "source", [artifact()])
    source = Zoo(catalog=catalog)
    destination = tmp_path / "download"
    assert source.fetch(source.artifacts[0], destination).read_bytes() == b"plan"
    assert json.loads((destination / "manifest.json").read_text())["runtime"]["min"] == "0.9.1"
    assert json.loads((destination / "download.json").read_text())["catalog_sha256"] == digest(catalog)
    with pytest.raises(ValueError, match="already exists"):
        source.fetch(source.artifacts[0], destination)
    assert (destination / "model.tgrs").read_bytes() == b"plan"


@pytest.mark.parametrize("damage", ["plan", "manifest", "metadata", "missing", "symlink"])
def test_failed_download_exposes_no_partial_result(tmp_path, damage):
    catalog = snapshot(tmp_path / "source", [artifact()])
    source = Zoo(catalog=catalog)
    item = source.artifacts[0]
    directory = catalog.parent / artifact_path(item)
    if damage == "plan":
        (directory / "model.tgrs").write_bytes(b"bad!")
    elif damage == "manifest":
        (directory / "manifest.json").write_text("{}")
    elif damage == "metadata":
        item["memory"]["fast_bytes"] += 1
    elif damage == "missing":
        (directory / "model.tgrs").unlink()
    else:
        outside = tmp_path / "outside"
        outside.write_bytes(b"plan")
        (directory / "model.tgrs").unlink()
        (directory / "model.tgrs").symlink_to(outside)
    destination = tmp_path / "download"
    with pytest.raises((ValueError, OSError)):
        source.fetch(item, destination)
    assert not destination.exists()
    assert not list(tmp_path.glob(".tigris-zoo-*"))


def test_hub_requests_are_anonymous_pinned_and_offline_aware(tmp_path, monkeypatch):
    from huggingface_hub.utils import logging as hub_logging

    original_verbosity = hub_logging.get_verbosity()
    revision = "a" * 40
    catalog = snapshot(tmp_path / "snapshots" / revision, [artifact()])
    calls = []

    def download(**kwargs):
        assert hub_logging.get_verbosity() == hub_logging.ERROR
        calls.append(kwargs)
        return str(catalog.parent / kwargs["filename"])

    monkeypatch.setattr("huggingface_hub.hf_hub_download", download)
    source = Zoo(offline=True, cache_dir=tmp_path / "cache")
    source.fetch(source.artifacts[0], tmp_path / "result")
    assert calls[0]["revision"] == "main"
    assert all(call["revision"] == revision for call in calls[1:])
    assert all(call["token"] is False and call["local_files_only"] for call in calls)
    receipt = json.loads((tmp_path / "result/download.json").read_text())
    assert receipt["revision"] == revision
    assert hub_logging.get_verbosity() == original_verbosity


def test_hub_errors_are_reported_without_requesting_credentials(monkeypatch):
    from huggingface_hub.errors import LocalEntryNotFoundError
    from huggingface_hub.utils import logging as hub_logging

    original_verbosity = hub_logging.get_verbosity()

    def download(**kwargs):
        raise LocalEntryNotFoundError("no cached file")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", download)
    result = CliRunner().invoke(cli, ["zoo", "--offline", "list"])
    assert result.exit_code == 1
    assert "populate the cache" in result.output
    assert "HF_TOKEN" not in result.output
    assert hub_logging.get_verbosity() == original_verbosity


def test_cli_filtering_and_download(tmp_path):
    catalog = snapshot(tmp_path / "source", [artifact(), artifact("example-b", withdrawn="bad output")])
    runner = CliRunner()
    prefix = ["zoo", "--catalog", str(catalog)]
    result = runner.invoke(cli, prefix + ["list", "--runtime", "0.9.1", "-m", "1K+0", "--json"])
    assert result.exit_code == 0, result.output
    assert [item["id"] for item in json.loads(result.output)] == ["example-a"]
    output = tmp_path / "download"
    result = runner.invoke(cli, prefix + ["fetch", "--artifact", "example-b", "-o", str(output)])
    assert result.exit_code == 0, result.output
    assert "WARNING: withdrawn: bad output" in result.output
    assert "runtime >= 0.9.1, <= 0.9.3" in result.output
    assert (output / "model.tgrs").read_bytes() == b"plan"
    for args in (["fetch"], ["fetch", "example", "--artifact", "example-a"],
                 ["fetch", "example", "--runtime", "0.10.0"],
                 ["list", "-m", "-1"], ["list", "--runtime", "broken"],
                 ["list", "-m", "1K+2K+3K"]):
        assert runner.invoke(cli, prefix + args).exit_code != 0
