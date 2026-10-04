"""Operator dtype slots and auxiliary tensor placement."""

from dataclasses import dataclass

from tigris.emitters.binary.defs import OP_TYPE_MAP


DATA = 0
DATA_OR_BOOL = 255


@dataclass(frozen=True)
class DTypeSignature:
    # The final slot repeats for variadic operators; semantic validation checks arity.
    inputs: tuple[int, ...] = (DATA, DATA, DATA)
    outputs: tuple[int, ...] = (DATA,)


@dataclass(frozen=True)
class AuxiliaryDType:
    terminal_only: bool


AUXILIARY_DTYPES = {6: AuxiliaryDType(True), 9: AuxiliaryDType(False)}
OP_DTYPE_SIGNATURES = {kind: DTypeSignature() for kind in OP_TYPE_MAP}
OP_DTYPE_SIGNATURES.update({
    "ArgMax": DTypeSignature(outputs=(6,)),
    "ArgMin": DTypeSignature(outputs=(6,)),
    "Equal": DTypeSignature(outputs=(9,)),
    "Less": DTypeSignature(outputs=(9,)),
    "LessOrEqual": DTypeSignature(outputs=(9,)),
    "Greater": DTypeSignature(outputs=(9,)),
    "GreaterOrEqual": DTypeSignature(outputs=(9,)),
    "And": DTypeSignature(inputs=(9, 9, 9), outputs=(9,)),
    "Or": DTypeSignature(inputs=(9, 9, 9), outputs=(9,)),
    "Not": DTypeSignature(inputs=(9, 9, 9), outputs=(9,)),
    "Where": DTypeSignature(inputs=(9, DATA, DATA)),
    "Cast": DTypeSignature(inputs=(9, 9, 9)),
    "ReduceAll": DTypeSignature(inputs=(9, 9, 9), outputs=(9,)),
    **{kind: DTypeSignature(inputs=(DATA_OR_BOOL, DATA_OR_BOOL, DATA_OR_BOOL), outputs=(DATA_OR_BOOL,))
       for kind in ("Transpose", "Reshape", "Flatten")},
})


def check_dtype_signatures(tensors, operators, model_inputs, model_outputs):
    """Return data tensors by dtype and errors for (name, dtype, constant, quantized) records."""
    by_name = {name: (dtype, constant, quantized) for name, dtype, constant, quantized in tensors}
    data = {}
    issues = []
    for name, (dtype, constant, quantized) in by_name.items():
        if constant:
            continue
        policy = AUXILIARY_DTYPES.get(dtype)
        if policy is None:
            data.setdefault(dtype, []).append(name)
            continue
        if quantized:
            issues.append(f"auxiliary tensor {name} cannot carry quantization")
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
                if expected == DATA_OR_BOOL:
                    if dtype not in {1, 3, 9} or dtype != by_name[inputs[0]][0]:
                        issues.append(f"{kind} must preserve its data or bool dtype")
                    continue
                if (expected == DATA and dtype not in {1, 3}) or (expected != DATA and dtype != expected):
                    label = "data (float32 or int8)" if expected == DATA else f"ONNX dtype {expected}"
                    issues.append(f"{kind} {direction} {position} ({name}) requires {label}, got ONNX dtype {dtype}")
    return data, tuple(issues)
