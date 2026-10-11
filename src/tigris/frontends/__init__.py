"""Model formats the compiler reads, each expressed as ONNX.

ONNX is the one interchange format: every other format converts to a QDQ ONNX
graph that states its semantics exactly, or refuses what it cannot state, and
the ONNX loader and its normalization passes do the rest.
"""

from pathlib import Path

import onnx
from google.protobuf.message import DecodeError

from tigris.frontends import tflite

__all__ = ["SUFFIXES", "is_onnx", "load_onnx", "not_a_model"]

SUFFIXES = (".onnx", ".tflite")


def load_onnx(path: str | Path) -> onnx.ModelProto:
    """Read a model file of any supported format as an ONNX model."""
    path = Path(path)
    with path.open("rb") as stream:
        head = stream.read(8)
    if tflite.is_tflite(head):
        return tflite.to_onnx(path.read_bytes(), path.stem)
    try:
        model = onnx.load(str(path))
    except DecodeError:
        model = None
    if not _is_model(model):
        raise ValueError(not_a_model(path))
    return model


def is_onnx(data: bytes) -> bool:
    try:
        return _is_model(onnx.load_model_from_string(data))
    except DecodeError:
        return False


def _is_model(model: onnx.ModelProto | None) -> bool:
    # An empty or unrelated protobuf parses as a ModelProto without a graph.
    return model is not None and model.HasField("graph") and model.ir_version > 0


def not_a_model(path: str | Path) -> str:
    return f"{Path(path).name} is not an ONNX model, a TFLite model or a plan"
