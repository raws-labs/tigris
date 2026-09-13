def fmt_bytes(b: int, *, unit_ref: int | None = None) -> str:
    """Format a byte count as a human-readable string.

    If unit_ref is given, the unit is chosen based on unit_ref's magnitude
    so that multiple values can be displayed in the same unit.
    """
    ref = unit_ref if unit_ref is not None else b
    if ref >= 1024 * 1024:
        return f"{b / (1024 * 1024):.2f} MiB"
    if ref >= 1024:
        return f"{b / 1024:.2f} KiB"
    return f"{b} B"


_DTYPE_NAMES = {1: "float32", 2: "uint8", 3: "int8", 6: "int32", 7: "int64"}


def describe_interface(ag) -> list[tuple[str, str]]:
    """One (label, description) per model input and output.

    The plan executes on one dtype, which is not always the dtype the model
    declares at its boundary: an ONNX graph states float32 and quantizes inside
    itself. The description names what the caller hands over and reads back,
    and the encoding the plan stores it in where the two differ.
    """
    rows: list[tuple[str, str]] = []
    for label, names, declared_dtypes in (
        ("Input", ag.model_inputs, ag.model_input_dtypes),
        ("Output", ag.model_outputs, ag.model_output_dtypes),
    ):
        for index, name in enumerate(names):
            info = ag.tensors.get(name)
            if info is None:
                continue
            shape = "x".join(str(dim) for dim in info.shape) or "scalar"
            declared = (
                declared_dtypes[index]
                if index < len(declared_dtypes)
                else info.dtype
            )
            text = f"{name} {shape} {_DTYPE_NAMES.get(declared, declared)}"
            if declared != info.dtype and info.quant is not None:
                text += (
                    f", stored as {_DTYPE_NAMES.get(info.dtype, info.dtype)}"
                    f" at scale {float(info.quant.scale[0]):.6g}"
                    f", zero point {int(info.quant.zero_point[0])}"
                )
            rows.append((label, text))
    return rows
