"""Operator dtype slots and auxiliary tensor placement."""

from dataclasses import dataclass

from tigris.emitters.binary.defs import OP_TYPE_MAP


DATA = 0
DATA_OR_BOOL = 255
# Data, or int16 on a tensor that holds state between runs.
STATE = 254
# Data, or int32 counters and indices; an operator's int32 slots go together.
DATA_OR_INT32 = 253
# Bool, or int32 a CAST turns into data.
BOOL_OR_INT32 = 252
# Data, int32, or float32 that control flow carries across an int8 plan's
# subgraph boundary.
CONTROL_FLOW_VALUE = 251


@dataclass(frozen=True)
class DTypeSignature:
    # The final slot repeats for variadic operators; semantic validation checks arity.
    inputs: tuple[int, ...] = (DATA, DATA, DATA)
    outputs: tuple[int, ...] = (DATA,)


@dataclass(frozen=True)
class AuxiliaryDType:
    terminal_only: bool
    requires_source: bool = False
    state_only: bool = False


AUXILIARY_DTYPES = {6: AuxiliaryDType(False, requires_source=True), 9: AuxiliaryDType(False),
                    5: AuxiliaryDType(False, state_only=True)}
OP_DTYPE_SIGNATURES = {kind: DTypeSignature() for kind in OP_TYPE_MAP}
OP_DTYPE_SIGNATURES.update({
    **{kind: DTypeSignature(inputs=(DATA, 6, 6))
       for kind in ("Gather", "GatherND", "EmbeddingLookup")},
    "DynamicUpdateSlice": DTypeSignature(inputs=(DATA, DATA, 6)),
    "ArgMax": DTypeSignature(outputs=(6,)),
    "ArgMin": DTypeSignature(outputs=(6,)),
    **{kind: DTypeSignature(inputs=(DATA_OR_INT32,) * 3, outputs=(9,))
       for kind in ("Equal", "Less", "LessOrEqual", "Greater", "GreaterOrEqual")},
    **{kind: DTypeSignature(inputs=(DATA_OR_INT32,) * 3, outputs=(DATA_OR_INT32,))
       for kind in ("Add", "Sub", "Mul")},
    "And": DTypeSignature(inputs=(9, 9, 9), outputs=(9,)),
    "Or": DTypeSignature(inputs=(9, 9, 9), outputs=(9,)),
    "Not": DTypeSignature(inputs=(9, 9, 9), outputs=(9,)),
    "Where": DTypeSignature(inputs=(9, DATA, DATA)),
    "Cast": DTypeSignature(inputs=(BOOL_OR_INT32,) * 3),
    "ReduceAll": DTypeSignature(inputs=(9, 9, 9), outputs=(9,)),
    # Constant operands sit between the input and the state; slots skip them.
    # The output role admits int16 only on the next state, a state tensor.
    "Svdf": DTypeSignature(inputs=(DATA, STATE, STATE), outputs=(STATE,)),
    "Lstm": DTypeSignature(inputs=(DATA, STATE, STATE), outputs=(STATE,)),
    "If": DTypeSignature(inputs=(9, CONTROL_FLOW_VALUE, CONTROL_FLOW_VALUE), outputs=(CONTROL_FLOW_VALUE,)),
    # Float32 results in a float32 or an int8 plan, which no operator reads.
    "DetectionPostProcess": DTypeSignature(outputs=(1,)),
    "While": DTypeSignature(inputs=(CONTROL_FLOW_VALUE,) * 3, outputs=(CONTROL_FLOW_VALUE,)),
    "Quantize": DTypeSignature(inputs=(1, 1, 1)),
    "Dequantize": DTypeSignature(outputs=(1,)),
    **{kind: DTypeSignature(inputs=(DATA_OR_BOOL, DATA_OR_BOOL, DATA_OR_BOOL), outputs=(DATA_OR_BOOL,))
       for kind in ("Transpose", "Reshape", "Flatten")},
})


_FLOAT_WRITERS = {"Dequantize", "If", "While"}
_FLOAT_READERS = {"Quantize", "If", "While"}


def check_dtype_signatures(tensors, operators, model_inputs, model_outputs, state=()):
    """Return data tensors by dtype and errors for (name, dtype, constant, quantized) records.
    `state` names the tensors that carry state between runs."""
    state = set(state)
    by_name = {name: (dtype, constant, quantized) for name, dtype, constant, quantized in tensors}
    data = {}
    issues = []
    # Float32 written where an operator's slot states float32 stands apart
    # from the data dtype and must not be read.
    stated = {name for kind, _, outputs in operators
              if OP_DTYPE_SIGNATURES.get(kind, DTypeSignature()).outputs == (1,)
              for name in outputs}
    for name in stated:
        if any(name in inputs and kind not in _FLOAT_READERS for kind, inputs, _ in operators):
            issues.append(f"float32 result {name} must not be read by another operator")
    # In an int8 plan, float32 crosses a control-flow boundary: written by a
    # Dequantize, control flow or as a graph input, read by a Quantize,
    # control flow or as a graph output.
    if any(dtype == 3 and not constant for dtype, constant, _ in by_name.values()):
        for name, (dtype, constant, _) in by_name.items():
            if dtype != 1 or constant or name in stated:
                continue
            producers = {kind for kind, _, outputs in operators if name in outputs}
            readers = {kind for kind, inputs, _ in operators if name in inputs}
            if ((producers <= _FLOAT_WRITERS and (producers or name in model_inputs))
                    and readers <= _FLOAT_READERS):
                stated.add(name)
    for name, (dtype, constant, quantized) in by_name.items():
        if constant or (name in stated and dtype == 1):
            continue
        policy = AUXILIARY_DTYPES.get(dtype)
        if policy is None:
            data.setdefault(dtype, []).append(name)
            continue
        if policy.state_only:
            if name not in state:
                issues.append(f"ONNX dtype {dtype} tensor {name} must hold state between runs")
            continue
        if quantized:
            issues.append(f"auxiliary tensor {name} cannot carry quantization")
        if policy.requires_source and sum(name in outputs for _, _, outputs in operators) != (0 if name in model_inputs else 1):
            issues.append(f"auxiliary tensor {name} requires one producer or a model input declaration")
        if policy.terminal_only and (
                name not in model_outputs or name in model_inputs
                or any(name in inputs for _, inputs, _ in operators)
                or sum(name in outputs for _, _, outputs in operators) != 1):
            issues.append(f"unsupported activation tensor dtype {dtype}: {name} must be a terminal model output")
    for kind, inputs, outputs in operators:
        signature = OP_DTYPE_SIGNATURES.get(kind)
        if signature is None:
            continue  # Operator support validation reports unknown operators.
        for direction, names, slots in (("input", inputs, signature.inputs),
                                        ("output", outputs, signature.outputs)):
            for position, name in enumerate(names):
                tensor = by_name.get(name)
                if tensor is None or tensor[1]:
                    continue
                dtype = tensor[0]
                expected = slots[min(position, len(slots) - 1)]
                if expected == STATE:
                    if dtype not in {1, 3} and not (dtype == 5 and name in state):
                        issues.append(f"{kind} {direction} {position} ({name}) requires data "
                                      f"or int16 state, got ONNX dtype {dtype}")
                    continue
                if expected in (DATA_OR_INT32, CONTROL_FLOW_VALUE):
                    if dtype not in {1, 3, 6}:
                        issues.append(f"{kind} {direction} {position} ({name}) requires data or "
                                      f"int32, got ONNX dtype {dtype}")
                    continue
                if expected == BOOL_OR_INT32:
                    if dtype not in {6, 9}:
                        issues.append(f"{kind} {direction} {position} ({name}) requires bool or "
                                      f"int32, got ONNX dtype {dtype}")
                    continue
                if expected == DATA_OR_BOOL:
                    if dtype not in {1, 3, 9} or dtype != by_name[inputs[0]][0]:
                        issues.append(f"{kind} must preserve its data or bool dtype")
                    continue
                if (expected == DATA and dtype not in {1, 3}) or (expected != DATA and dtype != expected):
                    label = "data (float32 or int8)" if expected == DATA else f"ONNX dtype {expected}"
                    issues.append(f"{kind} {direction} {position} ({name}) requires {label}, got ONNX dtype {dtype}")
        # An operator computing on int32 does so on all its operands.
        if kind in ("Add", "Sub", "Mul", "Equal", "Less", "LessOrEqual", "Greater", "GreaterOrEqual"):
            slots = [n for n in (*inputs, *(outputs if kind in ("Add", "Sub", "Mul") else ()))
                     if n in by_name and not by_name[n][1]]
            if len({by_name[n][0] == 6 for n in slots}) > 1:
                issues.append(f"{kind} mixes int32 with other operands")
    return data, tuple(issues)
