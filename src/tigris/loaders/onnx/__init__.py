"""ONNX loader: loads and normalizes an ONNX model into TiGrIS IR."""

from tigris.loaders.onnx.loader import load_model as _load_raw
from tigris.loaders.onnx.normalize import normalize


def load_model(path, input_shapes=None):
    ag = _load_raw(path, input_shapes)
    return _normalized(ag)


def _normalized(ag):
    """The graph and every subgraph it runs, each normalized on its own."""
    ag.subgraphs = [_normalized(sub) for sub in ag.subgraphs]
    return normalize(ag)
