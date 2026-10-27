import pytest
from click import BadParameter
from click.testing import CliRunner

from tigris.cli import _memory_sizes, cli
from tigris.pipeline import parse_size


_MALFORMED = "expected bytes or a number with a K, KB, M or MB suffix, for example 256K"
_NONPOSITIVE = "must be greater than zero"


@pytest.mark.parametrize("text, size", [
    ("1024", 1024), ("256K", 262144), ("256kb", 262144), ("4M", 4 * 1024 * 1024),
    (" 4M ", 4 * 1024 * 1024), ("1.5MB", 1536 * 1024),
])
def test_parse_size_suffixes(text, size):
    assert parse_size(text) == size


@pytest.mark.parametrize("text, reason", [
    ("64Q", _MALFORMED), ("4X", _MALFORMED), ("5G", _MALFORMED), ("", _MALFORMED),
    ("1.5", _MALFORMED), ("infK", _MALFORMED), ("nanM", _MALFORMED),
    ("-5", _NONPOSITIVE), ("0", _NONPOSITIVE), ("-1K", _NONPOSITIVE), ("0.0001K", _NONPOSITIVE),
])
def test_a_malformed_memory_size_raises_value_error(text, reason):
    with pytest.raises(ValueError, match="invalid memory size") as error:
        parse_size(text)
    assert reason in str(error.value)


@pytest.mark.parametrize("command", ["analyze", "compile"])
@pytest.mark.parametrize("sizes, reason", [
    (["-m", "64Q"], _MALFORMED),
    (["-m", "64K", "-f", "4X"], _MALFORMED),
    (["-m", "5G"], _MALFORMED),
    (["-m", "-5"], _NONPOSITIVE),
    (["-m", "0"], _NONPOSITIVE),
    (["-m", "-1K"], _NONPOSITIVE),
    (["-m", "64K", "-m", "0"], _NONPOSITIVE),
    (["-m", "64K+0"], _NONPOSITIVE),
    (["-m", "64K", "-f", "0"], _NONPOSITIVE),
    (["-m", "64K", "-f", "4M+1M"], _MALFORMED),
])
def test_a_malformed_memory_size_is_a_usage_error(linear_3op_path, tmp_path, command, sizes, reason):
    output = tmp_path / "should-not-exist.tgrs"
    args = [command, str(linear_3op_path), *sizes]
    if command == "compile":
        args += ["-o", str(output)]

    result = CliRunner().invoke(cli, args)

    assert result.exit_code == 2, result.output
    assert isinstance(result.exception, SystemExit)
    assert "invalid memory size" in result.output and reason in result.output
    assert not output.exists()


def test_memory_sizes_single_plus_token():
    assert _memory_sizes(None, None, ("256K+4M",)) == ("256K", "4M")


def test_memory_sizes_rejects_a_third_pool():
    with pytest.raises(BadParameter, match="3 memory pools"):
        _memory_sizes(None, None, ("256K+4M", "8M"))


def test_memory_sizes_passthrough_without_plus():
    assert _memory_sizes(None, None, ("256K", "4M")) == ("256K", "4M")


@pytest.mark.parametrize("bad", ["256K+", "+4M", "256K++4M", "+"])
def test_memory_sizes_rejects_empty_parts(bad):
    with pytest.raises(BadParameter):
        _memory_sizes(None, None, (bad,))


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
