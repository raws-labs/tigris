"""Exported PyTorch programs, as `torch.export.save` writes them.

torch.onnx converts the program to ONNX, which needs torch and onnxscript.
ExecuTorch's planned arena for the same program needs executorch.
"""

import functools
import logging
import zipfile
from contextlib import contextmanager
from pathlib import Path

import onnx

__all__ = ["EXECUTORCH_CPU", "executorch_arena", "is_pt2", "to_onnx"]

EXECUTORCH_CPU = "cortex-m7"
_ZIP_MAGIC = b"PK\x03\x04"


def is_pt2(path: str | Path) -> bool:
    """Whether `path` is a PT2 archive: a zip with an archive_format entry reading pt2."""
    try:
        with Path(path).open("rb") as stream:
            if stream.read(4) != _ZIP_MAGIC:
                return False
        with zipfile.ZipFile(path) as archive:
            return any(name.endswith("/archive_format") and archive.read(name) == b"pt2"
                       for name in archive.namelist())
    except (OSError, zipfile.BadZipFile):
        return False


def to_onnx(path: str | Path) -> onnx.ModelProto:
    """The program in the PT2 archive `path`, exported with torch.onnx."""
    path = Path(path).resolve()
    stat = path.stat()
    return onnx.load_model_from_string(_exported(str(path), stat.st_mtime_ns, stat.st_size))


# A command loads its model several times, and each export takes seconds.
@functools.lru_cache(maxsize=4)
def _exported(path: str, mtime_ns: int, size: int) -> bytes:
    program = _program(path)
    import torch

    try:
        with _quiet():
            exported = torch.onnx.export(program, dynamo=True, verbose=False)
    except Exception as exc:
        raise ValueError(f"cannot export {Path(path).name} to ONNX: {_first_line(exc)}") from exc
    return exported.model_proto.SerializeToString()


def _program(path: str):
    name = Path(path).name
    try:
        import onnxscript  # noqa: F401
        import torch
    except ImportError as exc:
        raise ValueError(f"reading {name} needs torch and onnxscript: "
                         "pip install torch onnxscript") from exc
    # Registers the quantize and dequantize operators a converted PT2E program calls.
    with _quiet():
        import torch.ao.quantization.fx._decomposed  # noqa: F401

    try:
        with _quiet():
            return torch.export.load(path)
    except Exception as exc:
        raise ValueError(f"cannot read {name}: {_first_line(exc)}") from exc


def executorch_arena(path: str | Path) -> tuple[int | None, str | None]:
    """ExecuTorch's planned arena for the program in `path`, as its Cortex-M
    lowering plans it for EXECUTORCH_CPU, CMSIS-NN scratch included: (bytes,
    None), or (None, why it was not measured)."""
    try:
        with _quiet():
            from executorch.backends.cortex_m.edge_compile_config import (
                cortex_m_edge_compile_config)
            from executorch.backends.cortex_m.passes.cortex_m_pass_manager import (
                CortexMPassManager)
            from executorch.backends.cortex_m.target_config import CortexM, CortexMTargetConfig
            from executorch.exir import ExecutorchBackendConfig, to_edge
    except ImportError:
        return None, "executorch is not installed"
    # Loaded again: lowering may change the program it is given.
    program = _program(str(Path(path).resolve()))
    try:
        with _quiet():
            edge = to_edge(program, compile_config=cortex_m_edge_compile_config())
            # As ExecuTorch's Arm AOT compiler replaces the method with its lowering.
            edge._edge_programs["forward"] = CortexMPassManager(
                edge.exported_program(),
                target_config=CortexMTargetConfig(cpu=CortexM.M7)).transform()
            plan = edge.to_executorch(ExecutorchBackendConfig()).executorch_program
    except Exception as exc:
        return None, f"its Cortex-M lowering failed: {_first_line(exc)}"
    return sum(plan.execution_plan[0].non_const_buffer_sizes), None


@contextmanager
def _quiet():
    """Holds back the progress and deprecation messages torch and ExecuTorch
    log; some of their loggers set their own level, so the threshold is global."""
    previous = logging.root.manager.disable
    logging.disable(logging.WARNING)
    try:
        yield
    finally:
        logging.disable(previous)


def _first_line(exc: Exception) -> str:
    lines = str(exc).strip().splitlines()
    return lines[0] if lines else type(exc).__name__
