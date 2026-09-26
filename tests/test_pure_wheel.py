"""Installed pure-wheel behavior without a host library."""

import os
import json
from pathlib import Path

import pytest
import numpy as np
from click.testing import CliRunner

import tigris
from tigris.cli import cli

pytestmark = pytest.mark.skipif(os.environ.get("TIGRIS_REQUIRE_PURE") != "1",
                                reason="Requires an installed pure wheel")


def test_compiler_without_host_runtime(linear_3op_path, tmp_path, monkeypatch):
    monkeypatch.delenv("TIGRIS_HOST_LIBRARY", raising=False)
    package = Path(tigris.__file__).resolve().parent
    assert "site-packages" in str(package)
    assert not (package / "native").exists()
    runner = CliRunner()
    plan = tmp_path / "model.tgrs"
    harness = tmp_path / "model.c"
    commands = [
        ["analyze", str(linear_3op_path), "-m", "64K"],
        ["compile", str(linear_3op_path), "-m", "64K", "-o", str(plan)],
        ["codegen", str(plan), "--backend", "reference", "-o", str(harness)],
    ]
    for command in commands:
        result = runner.invoke(cli, command)
        assert result.exit_code == 0, result.output
    assert plan.stat().st_size > 0
    assert "tigris" in harness.read_text()

    for model in (linear_3op_path, plan):
        result = runner.invoke(cli, ["inspect", str(model)])
        assert result.exit_code == 0, result.output
    result = runner.invoke(cli, ["run", str(plan), "--input", "input.bin",
                                 "--output", str(tmp_path / "output.bin")])
    assert result.exit_code == 1, result.output
    assert "Bundled host runtime is unavailable" in result.output
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0, result.output
    assert "unavailable" in result.output


def test_pure_wheel_runs_with_override(linear_3op_path, tmp_path, monkeypatch):
    assert not (Path(tigris.__file__).resolve().parent / "native").exists()
    library = Path(os.environ["TIGRIS_TEST_HOST_LIBRARY"]).resolve(strict=True)
    monkeypatch.setenv("TIGRIS_HOST_LIBRARY", str(library))
    runner = CliRunner()
    plan = tmp_path / "model.tgrs"
    result = runner.invoke(cli, ["compile", str(linear_3op_path), "-m", "64K", "-o", str(plan)])
    assert result.exit_code == 0, result.output
    x = np.arange(-32, 32, dtype=np.float32).reshape(1, 64)
    np.save(tmp_path / "input.npy", x)
    result = runner.invoke(cli, ["run", str(plan), "--input", str(tmp_path / "input.npy"),
                                 "--output", str(tmp_path / "result.npy"), "--json"])
    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report["runtime_source"] == f"TIGRIS_HOST_LIBRARY={library}"
    np.testing.assert_array_equal(np.load(tmp_path / "result.npy"), np.maximum(x, 0))
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0, result.output
    assert report["runtime_version"] in result.output
    assert report["runtime_source"] in result.output
