"""Input adapters for loading models into the TiGrIS IR."""

from pathlib import Path

from tigris.graph.ir import AnalyzedGraph

__all__ = ["load_model"]

_EXTENSION_MAP = {
    ".onnx": "tigris.loaders.onnx",
}


def load_model(
    path: str | Path,
    input_shapes: dict[str, tuple[int, ...]] | None = None,
) -> AnalyzedGraph:
    """Load a model file and return an AnalyzedGraph.

    Dispatches to the appropriate loader based on file extension.
    ``input_shapes`` maps a model input name to the full shape to compile for,
    overriding whatever the file declares.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    module_name = _EXTENSION_MAP.get(suffix)
    if module_name is None:
        supported = ", ".join(sorted(_EXTENSION_MAP.keys()))
        raise ValueError(
            f"Unsupported model format '{suffix}'. Supported: {supported}"
        )

    import importlib
    module = importlib.import_module(module_name)
    return module.load_model(path, input_shapes)
