"""Installed pure-wheel behavior without a host library."""

import os
from pathlib import Path

import pytest
from click.testing import CliRunner

import tigris
from tigris.cli import cli

pytestmark = pytest.mark.skipif(os.environ.get("TIGRIS_REQUIRE_PURE") != "1",
                                reason="Requires an installed pure wheel")


def test_compiler_without_host_runtime(linear_3op_path, tmp_path):
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
