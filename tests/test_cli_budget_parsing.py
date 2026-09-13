import pytest
from click import BadParameter
from click.testing import CliRunner

from tigris.cli import _expand_mem, _parse_size, cli


def test_parse_size_suffixes():
    assert _parse_size("256K") == 262144
    assert _parse_size("4M") == 4 * 1024 * 1024
    assert _parse_size("1024") == 1024


def test_expand_single_plus_token():
    assert _expand_mem(None, None, ("256K+4M",)) == ("256K", "4M")


def test_expand_mixes_with_repeated_flags():
    assert _expand_mem(None, None, ("256K+4M", "8M")) == ("256K", "4M", "8M")


def test_expand_passthrough_without_plus():
    assert _expand_mem(None, None, ("256K", "4M")) == ("256K", "4M")


@pytest.mark.parametrize("bad", ["256K+", "+4M", "256K++4M", "+"])
def test_expand_rejects_empty_parts(bad):
    with pytest.raises(BadParameter):
        _expand_mem(None, None, (bad,))


def test_combined_and_repeated_mem_are_equivalent(conv_relu_chain_path, tmp_path):
    # conv_relu_chain_path is the shared fixture from tests/conftest.py
    combined = CliRunner().invoke(
        cli, ["compile", str(conv_relu_chain_path), "-m", "64K+8M",
               "-o", str(tmp_path / "a.tgrs")])
    repeated = CliRunner().invoke(
        cli, ["compile", str(conv_relu_chain_path), "-m", "64K", "-m", "8M",
               "-o", str(tmp_path / "b.tgrs")])
    assert combined.exit_code == repeated.exit_code
    assert (tmp_path / "a.tgrs").exists() == (tmp_path / "b.tgrs").exists()
