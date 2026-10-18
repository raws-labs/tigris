"""Operator dtype slots and auxiliary tensor placement."""

from dataclasses import dataclass

from tigris.operator_contract import (
    DTypeSignature, OP_DTYPE_SIGNATURES as OP_DTYPE_SIGNATURES,
    DATA, DATA_OR_BOOL, STATE, DATA_OR_INT32, BOOL_OR_INT32,
    CONTROL_FLOW_VALUE, FLOAT_OR_DATA,
)



@dataclass(frozen=True)
class AuxiliaryDType:
    terminal_only: bool
    requires_source: bool = False
    state_only: bool = False


AUXILIARY_DTYPES = {6: AuxiliaryDType(False, requires_source=True), 9: AuxiliaryDType(False),
                    5: AuxiliaryDType(False, state_only=True)}


_FLOAT_WRITERS = {"Dequantize", "If", "While"}
_FLOAT_READERS = {"Quantize", "If", "While"}


def check_dtype_signatures(tensors, operators, model_inputs, model_outputs, state=()):
    """Return data tensors by dtype and errors for (name, dtype, constant, quantized) records.
    `state` names the tensors that carry state between runs."""
    state = set(state)
    by_name = {name: (dtype, constant, quantized) for name, dtype, constant, quantized in tensors}
    data = {}
    issues = []
    stated = {name for kind, _, outputs in operators
              if OP_DTYPE_SIGNATURES.get(kind, DTypeSignature()).outputs == (1,)
              for name in outputs}
    sections = {name for kind, _, outputs in operators if kind in _FLOAT_WRITERS
                for name in outputs if name in by_name and by_name[name][0] == 1}
    section_ops = set()
    changed = True
    while changed:
        changed = False
        for index, (kind, inputs, outputs) in enumerate(operators):
            if index in section_ops or kind in {"Quantize", "DetectionPostProcess"}:
                continue
            float_inputs = [n for n in inputs if n in by_name and by_name[n][0] == 1
                            and not by_name[n][1]]
            if (float_inputs and all(n in sections for n in float_inputs)
                    and all(n is None or (n in by_name and (by_name[n][0] == 1
                                or by_name[n][0] in AUXILIARY_DTYPES)) for n in inputs)
                    and all(n in by_name and (by_name[n][0] == 1
                            or by_name[n][0] in AUXILIARY_DTYPES) for n in outputs)):
                section_ops.add(index)
                sections.update(n for n in outputs if by_name[n][0] == 1)
                changed = True
    for name in stated | sections:
        if any(name in inputs and kind not in _FLOAT_READERS and index not in section_ops
               for index, (kind, inputs, _) in enumerate(operators)):
            issues.append(f"float32 result {name} must not be read by another operator")
    stated.update(sections)
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
                if expected == FLOAT_OR_DATA:
                    if dtype not in {1, 3}:
                        issues.append(f"{kind} {direction} {position} ({name}) requires float32 or "
                                      f"int8, got ONNX dtype {dtype}")
                    continue
                if expected == BOOL_OR_INT32:
                    if dtype not in {6, 9}:
                        issues.append(f"{kind} {direction} {position} ({name}) requires bool or "
                                      f"int32, got ONNX dtype {dtype}")
                    continue
                if expected == DATA_OR_BOOL:
                    source = by_name[inputs[0]][0]
                    if dtype not in {1, 3, 6, 9}:
                        issues.append(f"{kind} {direction} {position} ({name}) requires float32, "
                                      f"int8, int32 or bool, got ONNX dtype {dtype}")
                    elif dtype != source:
                        issues.append(f"{kind} reads {_dtype_name(source)} and writes "
                                      f"{_dtype_name(dtype)}; it has to keep the dtype")
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


_NAMES = {1: "float32", 3: "int8", 6: "int32", 9: "bool"}


def _dtype_name(dtype: int) -> str:
    return _NAMES.get(dtype, f"ONNX dtype {dtype}")
