"""Execute a compiled plan on the current host."""

import json
import zipfile
from pathlib import Path

import click
import numpy as np

from tigris.cli import cli, text
from tigris.runtime import Session
from tigris.utils import fmt_bytes


def read_inputs(session, input_files) -> dict:
    """Input arrays from .bin or .npy files, one per model input; a single
    input may omit its NAME=."""
    paths = {}
    for item in input_files:
        if "=" in item:
            name, filename = item.split("=", 1)
        elif len(session.inputs) == 1:
            name, filename = session.inputs[0]["name"], item
        else:
            raise ValueError("Multiple-input models require NAME=FILE for each input")
        if name in paths:
            raise ValueError(f"Duplicate input: {name}")
        paths[name] = Path(filename)
    if set(paths) != {info["name"] for info in session.inputs}:
        raise ValueError("Input names must match the complete model interface")
    inputs = {}
    for info in session.inputs:
        path = paths[info["name"]]
        if path.suffix == ".npy":
            value = np.load(path, allow_pickle=False)
        elif path.suffix == ".bin":
            if path.stat().st_size != info["size_bytes"]:
                raise ValueError(f"Input {info['name']!r} requires {info['size_bytes']} bytes")
            value = np.fromfile(path, dtype=info["dtype"].newbyteorder("<")).reshape(info["shape"])
        else:
            raise ValueError("Inputs must end in .bin or .npy")
        inputs[info["name"]] = value
    return inputs


@cli.command("run")
@click.argument("model", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--input", "input_files", multiple=True, required=True,
              help="Input .bin or .npy tensor; use NAME=FILE for multiple inputs.")
@click.option("--output", "output", required=True, type=click.Path(path_type=Path),
              help="Output .bin/.npy for one tensor, or .npz for named outputs.")
@click.option("--json", "as_json", is_flag=True, help="Print execution metadata as JSON.")
def run(model: Path, input_files: tuple[str, ...], output: Path, as_json: bool):
    """Run a model or plan on the host reference runtime.

    Inputs take the axis order and dtype inspect shows; .bin files hold
    little-endian, contiguous elements. This checks the portable reference
    kernels, not ESP-NN or CMSIS-NN numerics or device latency. Set
    TIGRIS_HOST_LIBRARY to use another runtime library.
    """
    try:
        if output.exists():
            raise ValueError(f"Output already exists: {output}")
        if output.suffix not in {".bin", ".npy", ".npz"}:
            raise ValueError("Output must end in .bin, .npy, or .npz")
        with Session(model) as session:
            inputs = read_inputs(session, input_files)
            if len(session.outputs) > 1 and output.suffix != ".npz":
                raise ValueError("Multiple-output models require an .npz output")
            outputs = session.run(inputs)
            with output.open("xb") as stream:
                if output.suffix == ".npz":
                    with zipfile.ZipFile(stream, "w") as archive:
                        for name, value in outputs.items():
                            with archive.open(f"{name}.npy", "w", force_zip64=True) as member:
                                np.save(member, value, allow_pickle=False)
                elif output.suffix == ".npy":
                    np.save(stream, next(iter(outputs.values())), allow_pickle=False)
                else:
                    value = next(iter(outputs.values()))
                    stream.write(value.astype(value.dtype.newbyteorder("<"), copy=False).tobytes())
            report = {"runtime_version": session.runtime_version, "runtime_source": session.runtime_source,
                      "backend": "reference",
                      "output": str(output), "memory": session.memory,
                      "tensors": [{"name": name, "dtype": str(value.dtype), "shape": list(value.shape)}
                                  for name, value in outputs.items()]}
    except (OSError, ValueError, KeyError) as exc:
        raise click.ClickException(str(exc)) from exc
    if as_json:
        click.echo(json.dumps(report, indent=2))
    else:
        memory = report["memory"]
        text.echo(text.bold(model.name) + f"   runtime {report['runtime_version']}, "
                  f"{report['runtime_source']}, host reference backend")
        for line in text.columns([
                ["fast arena peak", fmt_bytes(memory["fast_peak_bytes"]),
                 text.dim("measured; plan and workspace excluded")],
                ["slow arena peak", fmt_bytes(memory["slow_peak_bytes"]), ""]], "<>"):
            text.echo(line)
        text.gap()
        shapes = ", ".join(f"{'x'.join(map(str, item['shape'])) or 'scalar'} {item['dtype']}"
                           for item in report["tensors"])
        text.echo(f"wrote {output}   {shapes}")
