"""ONNX loader: loads and normalizes an ONNX model into TiGrIS IR."""

from tigris.loaders.onnx.loader import load_model as _load_raw
from tigris.loaders.onnx.normalize import normalize


def load_model(path, input_shapes=None):
    ag = _load_raw(path, input_shapes)
    return normalize(ag)
