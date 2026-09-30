"""Model formats the compiler reads, each expressed as ONNX.

ONNX is the one interchange format: every other format converts to a QDQ ONNX
graph that states its semantics exactly, or refuses what it cannot state, and
the ONNX loader and its normalization passes do the rest.
"""

from pathlib import Path

import onnx

from tigris.frontends import tflite

__all__ = ["SUFFIXES", "load_onnx"]

SUFFIXES = (".onnx", ".tflite")


def load_onnx(path: str | Path) -> onnx.ModelProto:
    """Read a model file of any supported format as an ONNX model."""
    path = Path(path)
    with path.open("rb") as stream:
        head = stream.read(8)
    if tflite.is_tflite(head):
        return tflite.to_onnx(path.read_bytes(), path.stem)
    return onnx.load(str(path))
