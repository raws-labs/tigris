"""Run a plan, a TFLite model or an ExecuTorch program on the host."""

import json
import os
import re
import sys
import tempfile
import zipfile
from importlib.metadata import version
from pathlib import Path

import click
import numpy as np

from tigris.cli import cli, text
from tigris.cli.export import refuse_options
from tigris.emitters.binary.defs import MAGIC
from tigris.frontends import is_onnx, pt2, tflite
from tigris.frontends.pte_schema import is_pte
from tigris.utils import fmt_bytes

# What TFLite Micro prints when the arena it was given is too small.
_ARENA_SHORT = re.compile(r"too small|Failed to (allocate|resize)")


def _kind(path: Path, data: bytes) -> str:
    """The engine for the file by its identifier; a model that needs
    compiling first gets a pointer to compile."""
    if data.startswith(MAGIC):
        return "plan"
    if tflite.is_tflite(data):
        return "tflite"
    if is_pte(data):
        return "pte"
    if is_onnx(data):
        raise ValueError(f"{path.name} is an ONNX model; compile it with tigris compile and run "
                         f"the plan")
    if pt2.is_pt2(path):
        raise ValueError(f"{path.name} is an exported PyTorch program; compile it with tigris "
                         f"compile and run the plan")
    raise ValueError(f"{path.name} is not a plan, a TFLite model or an ExecuTorch program")


def input_paths(names: list[str], input_files) -> dict[str, Path]:
    """The file of each model input, by name; a single input may omit its
    NAME=."""
    paths = {}
    for item in input_files:
        if "=" in item:
            name, filename = item.split("=", 1)
        elif len(names) == 1:
            name, filename = names[0], item
        else:
            raise ValueError("Multiple-input models require NAME=FILE for each input")
        if name in paths:
            raise ValueError(f"Duplicate input: {name}")
        paths[name] = Path(filename)
    if set(paths) != set(names):
        raise ValueError("Input names must match the complete model interface")
    return paths


def _load_npy(path: Path) -> np.ndarray:
    """The array in the .npy file at `path`."""
    try:
        value = np.load(path, allow_pickle=False)
    except (EOFError, zipfile.BadZipFile):
        raise ValueError(f"{path.name} is not a .npy file") from None
    if not isinstance(value, np.ndarray):
        value.close()
        raise ValueError(f"{path.name} is not a .npy file")
    return value


def read_inputs(interface: list[dict], input_files) -> dict:
    """Input arrays from .bin or .npy files, one per model input, each input
    described by its name, dtype, shape and size in bytes."""
    paths = input_paths([info["name"] for info in interface], input_files)
    inputs = {}
    for info in interface:
        path = paths[info["name"]]
        if path.suffix == ".npy":
            value = _load_npy(path)
        elif path.suffix == ".bin":
            if path.stat().st_size != info["size_bytes"]:
                raise ValueError(f"Input {info['name']!r} requires {info['size_bytes']} bytes")
            value = np.fromfile(path, dtype=info["dtype"].newbyteorder("<")).reshape(info["shape"])
        else:
            raise ValueError("Inputs must end in .bin or .npy")
        inputs[info["name"]] = value
    return inputs


def _check_output(output: Path) -> None:
    if output.exists():
        raise ValueError(f"Output already exists: {output}")
    if output.suffix not in {".bin", ".npy", ".npz"}:
        raise ValueError("Output must end in .bin, .npy, or .npz")


def _check_outputs(output: Path, count: int) -> None:
    if count > 1 and output.suffix != ".npz":
        raise ValueError("Multiple-output models require an .npz output")


def _write(output: Path, arrays: dict[str, np.ndarray], raw: dict[str, bytes]) -> None:
    """Each output as .npy in an .npz, or the one output as .npy or as its
    raw bytes in .bin."""
    with output.open("xb") as stream:
        if output.suffix == ".npz":
            with zipfile.ZipFile(stream, "w") as archive:
                for name, value in arrays.items():
                    with archive.open(f"{name}.npy", "w", force_zip64=True) as member:
                        np.save(member, value, allow_pickle=False)
        elif output.suffix == ".npy":
            np.save(stream, next(iter(arrays.values())), allow_pickle=False)
        else:
            stream.write(next(iter(raw.values())))


def _le_bytes(value: np.ndarray) -> bytes:
    return value.astype(value.dtype.newbyteorder("<"), copy=False).tobytes()


def _shapes(tensors: list[dict]) -> str:
    """Each output's shape and dtype; a .pte tensor's shape in memory order
    with its dim order, as inspect shows it."""
    from tigris.frontends.pte_host import memory_shape

    shapes = []
    for item in tensors:
        shape, order = memory_shape(item["shape"], item.get("dim_order", range(len(item["shape"]))))
        shapes.append(f"{shape} {item['dtype']}{order}")
    return ("; " if any("," in shape for shape in shapes) else ", ").join(shapes)


def _tensors(arrays: dict[str, np.ndarray]) -> list[dict]:
    return [{"name": name, "dtype": str(value.dtype), "shape": list(value.shape)}
            for name, value in arrays.items()]


def _run_plan(path: Path, input_files, output: Path) -> tuple[dict, str, list]:
    from tigris.runtime import Session

    with Session(path) as session:
        inputs = read_inputs(session.inputs, input_files)
        _check_outputs(output, len(session.outputs))
        outputs = session.run(inputs)
        _write(output, outputs, {name: _le_bytes(value) for name, value in outputs.items()})
        report = {"runtime_version": session.runtime_version,
                  "runtime_source": session.runtime_source, "backend": "reference",
                  "output": str(output), "memory": session.memory,
                  "tensors": _tensors(outputs)}
    memory = report["memory"]
    title = (f"runtime {report['runtime_version']}, {report['runtime_source']}, "
             f"host reference backend")
    rows = [["fast arena peak", fmt_bytes(memory["fast_peak_bytes"]),
             text.dim("measured; plan and workspace excluded")],
            ["slow arena peak", fmt_bytes(memory["slow_peak_bytes"]), ""]]
    return report, title, rows


def _captured(call):
    """The result of `call` and the text TFLite Micro printed from C to
    stderr meanwhile, which stays off the terminal."""
    sys.stderr.flush()
    saved = os.dup(2)
    with tempfile.TemporaryFile() as sink:
        os.dup2(sink.fileno(), 2)
        try:
            result, error = call(), None
        except (RuntimeError, ValueError) as exc:
            result, error = None, exc
        finally:
            os.dup2(saved, 2)
            os.close(saved)
        sink.seek(0)
        printed = sink.read().decode("utf-8", "replace")
    return result, error, printed


def _reason(error, printed: str) -> str:
    """The first line TFLite Micro printed, which names the cause, or the
    exception's own text."""
    lines = [line.strip() for line in printed.splitlines() if line.strip()]
    return lines[0] if lines else str(error)


def _run_tflite(path: Path, data: bytes, input_files, output: Path) -> tuple[dict, str, list]:
    try:
        from tflite_micro.python.tflite_micro import runtime
    except ImportError:
        raise click.ClickException("running a TFLite model needs tflite-micro, which has wheels "
                                   "for Linux x86_64: pip install tflite-micro") from None
    from tigris.frontends.tflite_export import host_arena

    graph = tflite.describe(data)["graph"]
    try:
        arena = host_arena(data)
    except ValueError as error:
        raise ValueError(f"tigris run cannot size TFLite Micro's arena for {path.name}: "
                         f"{error}") from None
    interpreter, error, printed = _captured(
        lambda: runtime.Interpreter.from_bytes(data, arena_size=arena))
    if error is not None:
        reason = _reason(error, printed)
        if _ARENA_SHORT.search(printed):
            raise ValueError(f"TFLite Micro reports the arena too small for {path.name}: "
                             f"{reason}; tigris run sized it at {arena} bytes from its model "
                             f"of TFLite Micro's arena")
        raise ValueError(f"TFLite Micro cannot run {path.name}: {reason}")
    interface = []
    for index, item in enumerate(graph["inputs"]):
        details = interpreter.get_input_details(index)
        dtype, shape = np.dtype(details["dtype"]), tuple(int(d) for d in details["shape"])
        interface.append({"name": item["name"], "dtype": dtype, "shape": shape,
                          "size_bytes": int(np.prod(shape, dtype=np.int64)) * dtype.itemsize})
    names = [item["name"] for item in graph["outputs"]]
    if len(set(names)) != len(names) or len({i["name"] for i in interface}) != len(interface):
        raise ValueError("Model interface contains duplicate names")
    inputs = read_inputs(interface, input_files)
    _check_outputs(output, len(names))
    for index, info in enumerate(interface):
        value = np.asarray(inputs[info["name"]])
        if value.dtype != info["dtype"] or value.shape != info["shape"]:
            raise ValueError(f"Input {info['name']!r} requires {info['dtype']} {info['shape']}; "
                             f"got {value.dtype} {value.shape}")
        interpreter.set_input(np.ascontiguousarray(value), index)
    _, error, printed = _captured(interpreter.invoke)
    if error is not None:
        raise ValueError(f"TFLite Micro stopped on {path.name}: {_reason(error, printed)}")
    outputs = {name: np.array(interpreter.get_output(index)) for index, name in enumerate(names)}
    _write(output, outputs, {name: _le_bytes(value) for name, value in outputs.items()})
    report = {"runtime": "tflite-micro", "runtime_version": version("tflite-micro"),
              "backend": "reference", "output": str(output), "tensors": _tensors(outputs)}
    return report, f"TFLite Micro {report['runtime_version']}, reference kernels", []


def _from_home(path: Path) -> str:
    """`path` with the home directory it lies in written as ~."""
    try:
        return str(Path("~") / path.relative_to(Path.home()))
    except ValueError:
        return str(path)


def _pte_runner(executorch: str | None):
    """The host runner of the ExecuTorch checkout at `executorch`, built
    and cached on the first run against it."""
    from tigris.frontends import pte_host as ph

    if executorch is None:
        raise click.ClickException("running an ExecuTorch program needs --executorch, an "
                                   "ExecuTorch 1.6 checkout")
    try:
        tree = ph.checkout(Path(executorch))
    except ValueError as error:
        raise click.BadParameter(str(error), param_hint="'--executorch'") from None
    runner = ph.cached_runner(tree)
    if runner is not None:
        return runner
    try:
        ph.check_python(sys.executable)
    except ValueError as error:
        raise click.ClickException(str(error)) from None
    changes = " with uncommitted changes" if tree.changes else ""
    text.echo(f"building the ExecuTorch host runner at {tree.commit[:12]}{changes}, once; it is "
              f"kept in {_from_home(ph.cache_root())}", err=True)
    try:
        return ph.build_runner(tree, python=sys.executable)
    except ph.BuildError as error:
        lines = "\n".join("  " + line for line in error.lines)
        raise click.ClickException(f"the host runner build stopped at its {error.step} "
                                   f"step:\n{lines}\nfull log: {error.log}") from None


def _pte_input(spec, name: str, path: Path) -> bytes:
    """The bytes of the method input in memory: a .bin file as it is, a
    .npy array laid out as pte_host.npy_bytes reads it."""
    from tigris.frontends import pte_host as ph

    if path.suffix == ".npy":
        data = ph.npy_bytes(path.name, _load_npy(path), spec)
    elif path.suffix == ".bin":
        data = path.read_bytes()
    else:
        raise ValueError("Inputs must end in .bin or .npy")
    if len(data) != spec.nbytes:
        raise ValueError(f"{path.name} holds {len(data)} bytes; the method input takes "
                         f"{spec.nbytes} ({spec.describe()})")
    return data


def _run_pte(path: Path, data: bytes, input_files, output: Path,
             executorch: str | None) -> tuple[dict, str, list]:
    from tigris.frontends import pte, pte_cortex_m as cm, pte_host as ph

    try:
        program = pte.read(data, layout=False)
    except ValueError as error:
        raise click.ClickException(f"{path.name} fails to parse: {error}") from None
    try:
        plan = ph.forward_plan(program)
        try:
            classes = cm.infer_class(plan, cm.known_calls(plan))
        except pte.Refusal:
            raise
        except ValueError as error:
            raise ValueError(f"is malformed: {error}") from None
        spec = ph.method_input(plan)
        specs = ph.method_outputs(plan)
    except ValueError as error:
        raise click.ClickException(f"{path.name} {error}") from None
    if cm.SCALAR not in classes and cm.DSP not in classes:
        raise click.ClickException(f"{path.name} was lowered for {cm.describe(classes)}; the "
                                   f"host runs programs lowered for scalar or DSP kernels")
    names = [str(k) for k in range(len(specs))]
    paths = input_paths(["0"], input_files)
    _check_outputs(output, len(specs))
    for name, item in zip(names, specs):
        if item.dtype is None:
            raise ValueError(f"{path.name} returns output {name} in a scalar type NumPy has "
                             f"no dtype for")
    feed = _pte_input(spec, "0", paths["0"])
    runner = _pte_runner(executorch)
    with tempfile.TemporaryDirectory(prefix="tigris-run-") as work:
        try:
            (produced,) = ph.run([str(runner.path)], path.resolve(), [feed], Path(work))
        except ph.RunError as error:
            raise click.ClickException(f"the host runner stopped on {path.name}: "
                                       f"{error}") from None
    if len(produced) != sum(item.nbytes for item in specs):
        raise click.ClickException(f"the host runner stopped on {path.name}: wrote "
                                   f"{len(produced)} output bytes for "
                                   f"{sum(item.nbytes for item in specs)}")
    raw, arrays, offset = {}, {}, 0
    for name, item in zip(names, specs):
        raw[name] = produced[offset:offset + item.nbytes]
        arrays[name] = item.array(raw[name])
        offset += item.nbytes
    _write(output, arrays, raw)
    changes = " with uncommitted changes" if runner.changes else ""
    report = {"runtime": "executorch", "runtime_version": runner.executorch,
              "runtime_commit": runner.commit, "runtime_changes": runner.changes or None,
              "runtime_source": str(runner.path), "backend": "cmsis-nn",
              "cmsis_nn_version": runner.cmsis_nn, "output": str(output),
              "tensors": [dict(entry, dim_order=list(item.dim_order))
                          for entry, item in zip(_tensors(arrays), specs)]}
    title = (f"ExecuTorch {runner.executorch} at {runner.commit[:12]}{changes}, "
             f"CMSIS-NN {runner.cmsis_nn} portable kernels")
    return report, title, []


@cli.command("run")
@click.argument("model", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--input", "input_files", multiple=True, required=True,
              help="Input .bin or .npy tensor; use NAME=FILE for multiple inputs.")
@click.option("--output", "output", required=True, type=click.Path(path_type=Path),
              help="Output .bin/.npy for one tensor, or .npz for named outputs.")
@click.option("--executorch", "executorch", metavar="PATH", default=None,
              type=click.Path(exists=True, file_okay=False),
              help="ExecuTorch 1.6 checkout with its submodules, for .pte programs; the host "
                   "runner is built from it once and cached.")
@click.option("--json", "as_json", is_flag=True, help="Print execution metadata as JSON.")
def run(model: Path, input_files: tuple[str, ...], output: Path, executorch: str | None,
        as_json: bool):
    """Run a plan, a TFLite model or an ExecuTorch program on the host.

    A plan (.tgrs) runs on the TiGrIS host runtime, a TFLite model on TFLite
    Micro's reference kernels, an ExecuTorch Cortex-M program (.pte) on
    ExecuTorch with CMSIS-NN's portable kernels. A model and its export
    produce the same output bytes.

    Inputs take the axis order and dtype inspect shows; .bin files hold
    little-endian, contiguous elements. Set TIGRIS_HOST_LIBRARY to use another
    runtime library.
    """
    try:
        data = model.read_bytes()
        kind = _kind(model, data)
        if kind != "pte":
            refuse_options(click.get_current_context(), {"executorch": "--executorch"},
                           "ExecuTorch programs")
        _check_output(output)
        if kind == "plan":
            report, title, rows = _run_plan(model, input_files, output)
        elif kind == "tflite":
            report, title, rows = _run_tflite(model, data, input_files, output)
        else:
            report, title, rows = _run_pte(model, data, input_files, output, executorch)
    except (OSError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    if as_json:
        click.echo(json.dumps(report, indent=2))
        return
    text.echo(text.bold(model.name) + "   " + title)
    for line in text.columns(rows, "<>"):
        text.echo(line)
    text.gap()
    text.echo(f"wrote {output}   {_shapes(report['tensors'])}")
