"""Native archive integrity and extraction rules, and the runtime build commands."""

import hashlib
import io
import json
import re
import subprocess
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from packaging.tags import sys_tags

from scripts.runtime_bundle import stage
from scripts import runtime_bundle


def archive(*, library_hash=None, extra=None, link=False, platform=None):
    library = b"test library payload"
    manifest = {"version": "1.2.3", "abi": runtime_bundle.HOST_ABI, "library": "libtigris_host.so",
                "platform": platform or next(sys_tags()).platform,
                "sha256": library_hash or hashlib.sha256(library).hexdigest()}
    files = {"manifest.json": json.dumps(manifest).encode(), "LICENSE": b"license",
             "tigris_host.h": b"header", "libtigris_host.so": library}
    if extra:
        files[extra] = b"unexpected"
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as stream:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            if link and name == "libtigris_host.so":
                info.type = tarfile.SYMTYPE
                info.linkname = "../../library"
            stream.addfile(info, io.BytesIO(data))
    data = buffer.getvalue()
    return data, hashlib.sha256(data).hexdigest()


def test_stage_valid_archive(tmp_path):
    data, checksum = archive()
    result = stage(data, checksum, tmp_path / "native", release="v1.2.3")
    assert result["version"] == "1.2.3"
    assert (tmp_path / "native" / "libtigris_host.so").read_bytes() == b"test library payload"
    assert {item.name for item in (tmp_path / "native").iterdir()} == {"manifest.json", "LICENSE", "libtigris_host.so"}


@pytest.mark.parametrize("change", [{"library_hash": "0" * 64}, {"extra": "../escape"},
                                    {"link": True}, {"platform": "unsupported_platform"}])
def test_invalid_archive_keeps_previous_bundle(tmp_path, change):
    target = tmp_path / "native"
    target.mkdir()
    (target / "existing").write_text("keep")
    data, checksum = archive(**change)
    with pytest.raises(ValueError):
        stage(data, checksum, target)
    assert (target / "existing").read_text() == "keep"


def test_archive_checksum_and_release_pin(tmp_path):
    data, checksum = archive()
    with pytest.raises(ValueError, match="checksum"):
        stage(data, "0" * 64, tmp_path / "native")
    with pytest.raises(ValueError, match="version"):
        stage(data, checksum, tmp_path / "native", release="v1.2.4")


def test_source_build_does_not_require_release_pin(tmp_path, monkeypatch, capsys):
    (tmp_path / "runtime-host.json").write_text('{"release": null, "artifacts": {}}')
    monkeypatch.setattr(runtime_bundle, "ROOT", tmp_path)
    monkeypatch.setattr("sys.argv", ["runtime_bundle.py", "--source", str(tmp_path)])
    sources = []

    def build(source):
        sources.append(source)
        return {"version": "1.2.3", "platform": "linux_x86_64"}

    monkeypatch.setattr(runtime_bundle, "build_source", build)
    runtime_bundle.main()
    assert sources == [tmp_path]
    assert "Staged runtime 1.2.3" in capsys.readouterr().out


@pytest.mark.parametrize("pin, tag, version", [
    (None, "v1.2.3", "1.2.3"),
    ("v1.2.2", "v1.2.3", "1.2.3"),
    ("v1.2.3", "v1.2.3", "1.2.4.dev1"),
])
def test_release_requires_matching_pin_tag_and_scm(tmp_path, monkeypatch, pin, tag, version):
    (tmp_path / "runtime-host.json").write_text(json.dumps({"release": pin, "artifacts": {}}))
    monkeypatch.setattr(runtime_bundle, "ROOT", tmp_path)
    monkeypatch.setitem(sys.modules, "setuptools_scm", SimpleNamespace(get_version=lambda **kwargs: version))
    monkeypatch.setattr("sys.argv", ["runtime_bundle.py", "--expect-release", tag])
    with pytest.raises(SystemExit) as failure:
        runtime_bundle.main()
    assert failure.value.code == 2


def test_pin_records_release_and_archive_checksums(tmp_path, monkeypatch):
    (tmp_path / "runtime-host.json").write_text(json.dumps(
        {"release": "v0.1.0", "artifacts": {"plat_a": {"sha256": "0" * 64}, "plat_b": {"sha256": "0" * 64}}}))
    monkeypatch.setattr(runtime_bundle, "ROOT", tmp_path)
    requested = []

    def fake_urlopen(url, timeout):
        requested.append(url)
        name = url.rsplit("/", 1)[1].removesuffix(".sha256")
        return io.BytesIO(f"{hashlib.sha256(name.encode()).hexdigest()}  {name}\n".encode())

    monkeypatch.setattr(runtime_bundle.urllib.request, "urlopen", fake_urlopen)
    runtime_bundle.pin_release("v1.2.3")
    pin = json.loads((tmp_path / "runtime-host.json").read_text())
    assert pin["release"] == "v1.2.3"
    assert pin["artifacts"]["plat_a"]["sha256"] == hashlib.sha256(b"tigris-host-1.2.3-plat_a.tar.gz").hexdigest()
    assert set(pin["artifacts"]) == {"plat_a", "plat_b"}
    assert all("/releases/download/v1.2.3/" in url for url in requested)


def test_pin_refuses_a_malformed_release():
    with pytest.raises(SystemExit):
        runtime_bundle.pin_release("0.11.0")


def _parallel_level(monkeypatch, level):
    if level is None:
        monkeypatch.delenv("CMAKE_BUILD_PARALLEL_LEVEL", raising=False)
    else:
        monkeypatch.setenv("CMAKE_BUILD_PARALLEL_LEVEL", level)


def _build_jobs(command, env):
    """The job count a `cmake --build` command runs with, or None for another command."""
    if command[:2] != ["cmake", "--build"]:
        return None
    if "--parallel" in command:
        return command[command.index("--parallel") + 1]
    return (env or {}).get("CMAKE_BUILD_PARALLEL_LEVEL", "unset")


@pytest.mark.parametrize("level, jobs", [(None, "4"), ("2", "2")])
def test_source_build_stays_out_of_the_checkout(tmp_path, monkeypatch, level, jobs):
    _parallel_level(monkeypatch, level)
    source = tmp_path / "runtime"
    source.mkdir()
    configured, built, staged = [], [], []

    def run(command, check, env=None):
        if command[:2] == ["cmake", "-S"]:
            configured.append(command)
        if _build_jobs(command, env) is not None:
            built.append(_build_jobs(command, env))
        if "host_archive" in command:
            platform = next(arg for arg in configured[0] if arg.startswith("-DTIGRIS_HOST_PLATFORM="))
            artifacts = Path(command[2]) / "host-artifacts"
            (artifacts / "stage").mkdir(parents=True)
            (artifacts / "stage" / "manifest.json").write_text('{"version": "1.2.3"}')
            name = f"tigris-host-1.2.3-{platform.split('=', 1)[1]}.tar.gz"
            (artifacts / name).write_bytes(b"archive")
            (artifacts / f"{name}.sha256").write_text(f"{'a' * 64}  {name}\n")

    def stage(data, checksum, destination, platform):
        staged.append((data, checksum))
        return {"version": "1.2.3", "platform": platform}

    monkeypatch.setattr(runtime_bundle.subprocess, "run", run)
    monkeypatch.setattr(runtime_bundle, "stage", stage)
    runtime_bundle.build_source(source)

    build = Path(configured[0][configured[0].index("-B") + 1])
    assert staged == [(b"archive", "a" * 64)]
    assert built == [jobs, jobs]
    assert not build.is_relative_to(source) and not build.exists()
    assert list(source.iterdir()) == []


@pytest.mark.parametrize("level, jobs", [(None, "4"), ("2", "2")])
def test_contract_gate_builds_take_the_parallel_level(tmp_path, monkeypatch, level, jobs):
    from scripts import crossrepo_contract

    _parallel_level(monkeypatch, level)
    built = []

    def run(command, text, capture_output, check, env=None):
        if _build_jobs(command, env) is not None:
            built.append(_build_jobs(command, env))
            (Path(command[2]) / "libtigris_host.so").write_bytes(b"")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(crossrepo_contract.subprocess, "run", run)
    (tmp_path / "runner").mkdir()
    (tmp_path / "host").mkdir()
    crossrepo_contract._build_runner(tmp_path / "runtime", tmp_path / "runner")
    crossrepo_contract._build_host_library(tmp_path / "runtime", tmp_path / "host")
    assert built == [jobs, jobs]


def test_pure_wheel_job_reads_the_staged_host_library():
    workflow = (runtime_bundle.ROOT / ".github" / "workflows" / "wheels.yml").read_text()
    staged = runtime_bundle.DESTINATION.relative_to(runtime_bundle.ROOT).as_posix()
    assert re.findall(r"TIGRIS_TEST_HOST_LIBRARY: (.*)", workflow) == [
        "${{ github.workspace }}/" + staged + "/libtigris_host.so"]
