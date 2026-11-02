"""QDQ ONNX graph building for the TFLite importer, TensorFlow's split of
SAME padding into explicit begin and end pads, and the float32 to int8
quantization a plan states."""

import numpy as np
from onnx import helper, numpy_helper


class GraphBuilder:
    """Accumulates nodes and initializers for one ONNX graph."""

    def __init__(self):
        self.nodes = []
        self.initializers = []
        self._names = set()

    def unique(self, name: str) -> str:
        candidate, counter = name, 1
        while candidate in self._names:
            counter += 1
            candidate = f"{name}_{counter}"
        self._names.add(candidate)
        return candidate

    def constant(self, array, name: str) -> str:
        name = self.unique(name)
        self.initializers.append(numpy_helper.from_array(np.asarray(array), name))
        return name

    def node(self, op_type: str, inputs: list[str], name: str, **attributes) -> str:
        output = self.unique(name)
        self.nodes.append(helper.make_node(op_type, inputs, [output], **attributes))
        return output

    def multi_node(self, op_type: str, inputs: list[str], names: list[str], **attributes) -> list[str]:
        outputs = [self.unique(name) for name in names]
        self.nodes.append(helper.make_node(op_type, inputs, outputs, **attributes))
        return outputs

    def dequantized_constant(self, values, scale, zero_point, name: str, *, axis=None,
                             zero_dtype=np.int8) -> str:
        """An integer constant behind a DequantizeLinear, per-channel when
        `scale` holds more than one value."""
        scale = np.atleast_1d(np.asarray(scale, dtype=np.float32))
        zero_point = np.atleast_1d(np.asarray(zero_point)).astype(zero_dtype)
        per_channel = scale.size > 1
        data = self.constant(values, name)
        s = self.constant(scale if per_channel else scale[0], name + "_scale")
        z = self.constant(zero_point if per_channel else zero_point.reshape(()), name + "_zero_point")
        attributes = {"axis": axis} if per_channel and axis is not None else {}
        return self.node("DequantizeLinear", [data, s, z], name + "_dq", **attributes)

    def requantized(self, source: str, scale: float, zero_point: int, name: str) -> str:
        """A QuantizeLinear/DequantizeLinear pair: the int8 tensor `name`,
        read back as the float the next operator consumes."""
        s = self.constant(np.float32(scale), name + "_scale")
        z = self.constant(np.array(zero_point, dtype=np.int8), name + "_zero_point")
        quantized = self.node("QuantizeLinear", [source, s, z], name + "_q")
        return self.node("DequantizeLinear", [quantized, s, z], name)

    def fused_activation(self, source: str, kind: str, name: str) -> str:
        """Relu or Relu6 after an operator, or the source for none."""
        if kind == "none":
            return source
        if kind == "relu":
            return self.node("Relu", [source], name + "_relu")
        if kind == "relu6":
            low = self.constant(np.float32(0.0), name + "_relu6_min")
            high = self.constant(np.float32(6.0), name + "_relu6_max")
            return self.node("Clip", [source, low, high], name + "_relu6")
        raise ValueError(f"unsupported fused activation {kind}")


def same_padding(input_size: int, output_size: int, kernel: int, stride: int) -> tuple[int, int]:
    """Explicit (begin, end) padding that reproduces a SAME-style output size,
    with the extra row or column at the end as TensorFlow places it."""
    total = max((output_size - 1) * stride + kernel - input_size, 0)
    return total // 2, total - total // 2


def quantize_int8(values, scale, zero_point, ties_even: bool) -> np.ndarray:
    """Float values in an int8 encoding as the plan's rule computes it: a
    float32 division by the scale, rounded with ties to even (ONNX
    QuantizeLinear) or away from zero (TFLite QUANTIZE), plus the zero point,
    clamped; NaN gives -128, as the runtime and ONNX Runtime give.
    ``scale`` and ``zero_point`` broadcast against ``values``."""
    ratio = (np.asarray(values, np.float32) / np.asarray(scale, np.float32)).astype(np.float64)
    rounded = np.rint(ratio) if ties_even else np.sign(ratio) * np.floor(np.abs(ratio) + 0.5)
    level = np.where(np.isnan(rounded), -128.0, rounded + np.asarray(zero_point, np.float64))
    return np.clip(level, -128, 127).astype(np.int8)
