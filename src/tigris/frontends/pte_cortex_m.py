"""The cortex_m operators of an ExecuTorch 1.6.0 program and the CMSIS-NN
class its kernels were sized for.

Each kernel call names an operator and lists value indices: the arguments of
the operator's out variant, then the out tensor, then the returned value (the
out tensor again). `kernel_calls` checks every call against the operator
table below, the argument kinds of each position and the length of each fixed
list, and refuses an operator or signature it does not know with the operator
named.

ExecuTorch sizes the scratch tensor of a convolution, depthwise convolution
or average pool with CMSIS-NN v8.0.0's buffer-size function for one class of
CPU: SCALAR (M0, M0+, M3, M23), DSP (M4, M7, M33, M35P) or MVE (M55, M85).
The program does not record the CPU, but the stored scratch sizes and the
linear layers' argument pattern (MVE kernels take a precomputed kernel sum in
place of the bias) narrow it down. The buffer-size functions here are those
of CMSIS-NN v8.0.0 for int8 activations and weights built with GCC, where the
1x1 fast path needs no buffer and the MVE depthwise-as-convolution path starts
above one output channel.
"""

from dataclasses import dataclass
from typing import NamedTuple

from tigris.frontends import pte

SCALAR, DSP, MVE = "SCALAR", "DSP", "MVE"
CLASSES = (SCALAR, DSP, MVE)
CPU_CLASSES = {
    "M0": SCALAR, "M0PLUS": SCALAR, "M3": SCALAR, "M23": SCALAR,
    "M4": DSP, "M7": DSP, "M33": DSP, "M35P": DSP,
    "M55": MVE, "M85": MVE,
}
# How the report names each class and its cores.
CLASS_NAMES = {SCALAR: "scalar", DSP: "DSP", MVE: "MVE"}
CLASS_CORES = {SCALAR: "Cortex-M0, M0+, M3, M23", DSP: "Cortex-M4, M7, M33, M35P",
               MVE: "Cortex-M55, M85"}
NAMESPACE = "cortex_m::"

# The out variant of each operator, from backends/cortex_m/ops/operators.yaml
# of ExecuTorch 1.6.0. "int[2]" is a list that must hold two values.
_SCHEMAS = {
    "quantized_conv2d": (
        "Tensor input, Tensor weight, Tensor? bias, int[2] stride, int[2] padding, "
        "int[2] dilation, int input_offset, int output_offset, "
        "Tensor requantize_multipliers, Tensor requantize_shifts, int activation_min, "
        "int activation_max, Tensor scratch"),
    "quantized_depthwise_conv2d": (
        "Tensor input, Tensor weight, Tensor? bias, int[2] stride, int[2] padding, "
        "int[2] dilation, int depth_multiplier, int input_offset, int output_offset, "
        "Tensor requantize_multipliers, Tensor requantize_shifts, int activation_min, "
        "int activation_max, Tensor scratch"),
    "quantized_transpose_conv2d": (
        "Tensor input, Tensor weight, Tensor? bias, int[2] stride, int[2] padding, "
        "int[2] output_padding, int[2] dilation, int input_offset, int output_offset, "
        "Tensor requantize_multipliers, Tensor requantize_shifts, int activation_min, "
        "int activation_max, Tensor scratch, Tensor output_scratch"),
    "quantized_add": (
        "Tensor self, int self_zero_point, int self_multiplier, int self_shift, "
        "Tensor other, int other_zero_point, int other_multiplier, int other_shift, "
        "int output_zero_point, int output_multiplier, int output_shift, "
        "int activation_min, int activation_max"),
    "quantized_mul": (
        "Tensor self, int self_zero_point, Tensor other, int other_zero_point, "
        "int output_zero_point, int output_multiplier, int output_shift"),
    "quantized_avg_pool2d": (
        "Tensor input, int[2] kernel_size, int[2] stride, int[2] padding, bool ceil_mode, "
        "int zero_point, int multiplier, int shift, Tensor scratch"),
    "quantized_max_pool2d": (
        "Tensor input, int[2] kernel_size, int[2] stride, int[2] padding, int[2] dilation, "
        "bool ceil_mode, int input_zero_point, int output_zero_point, int activation_min, "
        "int activation_max"),
    "quantized_linear": (
        "Tensor input, Tensor weights, Tensor? bias, Tensor? kernel_sum, int input_offset, "
        "int filter_offset, int output_offset, int[] requantize_multipliers, "
        "int[] requantize_shifts, int activation_max, int activation_min"),
    "quantized_activation": "Tensor input, Tensor lut",
    "pad": "Tensor input, int[4] pre_pad, int[4] post_pad, int pad_value",
    "softmax": (
        "Tensor input, int dim, int input_zero_point, int output_zero_point, "
        "int input_multiplier, int input_shift, int diff_min"),
}
_KINDS = {"Tensor": (pte.Tensor,), "Tensor?": (pte.Tensor, pte.Null), "int": (pte.Int,),
          "bool": (pte.Bool,)}
_SCRATCH = ("scratch", "output_scratch")
# Operators that convert between float and int8 at the model's edges.
_FLOAT_IO = ("quantize_per_tensor", "dequantize_per_tensor")


@dataclass(frozen=True)
class Signature:
    """An operator's arguments in kernel-call order: `names` and `kinds` (a
    tuple of accepted record types each) per position, and the required
    length of each fixed-length list by name. The last two positions are
    "out" and the returned value."""
    name: str
    names: tuple[str, ...]
    kinds: tuple[tuple[type, ...], ...]
    lengths: dict

    def index(self, name: str) -> int:
        return self.names.index(name)

    @property
    def scratch(self) -> tuple[int, ...]:
        return tuple(i for i, n in enumerate(self.names) if n in _SCRATCH)

    @property
    def inputs(self) -> tuple[int, ...]:
        """Positions of the tensors the operator reads."""
        return tuple(i for i, (n, k) in enumerate(zip(self.names, self.kinds))
                     if pte.Tensor in k and n not in _SCRATCH and i < len(self.names) - 2)


def _signature(op: str, schema: str) -> Signature:
    names, kinds, lengths = [], [], {}
    for argument in schema.split(", "):
        kind, name = argument.split(" ")
        if kind.startswith("int["):
            if kind != "int[]":
                lengths[name] = int(kind[4:-1])
            kinds.append((pte.IntList,))
        else:
            kinds.append(_KINDS[kind])
        names.append(name)
    return Signature(NAMESPACE + op, (*names, "out", "return"),
                     (*kinds, (pte.Tensor,), (pte.Tensor,)), lengths)


SIGNATURES = {NAMESPACE + op: _signature(op, schema) for op, schema in _SCHEMAS.items()}


@dataclass
class Call:
    """A kernel call: its position in the chain, its signature and its
    argument values in signature order."""
    index: int
    signature: Signature
    args: list  # value indices
    values: list  # the records at those indices

    def value(self, name: str):
        return self.values[self.signature.index(name)]

    def ints(self, name: str, plan: pte.ExecutionPlan) -> list[int]:
        return [plan.values[i].int_val for i in self.value(name).items]


def signature(plan: pte.ExecutionPlan, call: pte.KernelCall) -> Signature:
    """The signature of `call`; raises ValueError naming the operator when it
    is not a known cortex_m operator or its arguments do not match."""
    operators = plan.operators or []
    if not 0 <= call.op_index < len(operators):
        raise ValueError(f"a kernel call names operator {call.op_index}, which does not exist")
    op = operators[call.op_index]
    name = op.name or ""
    if name.endswith("_nhwc"):
        raise pte.Refusal(f"uses the explicit NHWC layout ({name}), which tigris export does "
                          f"not rewrite")
    if name.removeprefix(NAMESPACE) in _FLOAT_IO:
        raise pte.Refusal(f"calls {name}; tigris export rewrites programs that take and return "
                          f"int8 tensors (ExecuTorch's QuantizeInputs and QuantizeOutputs "
                          f"passes)")
    if name not in SIGNATURES:
        raise pte.Refusal(f"calls operator {name}; tigris export rewrites programs of the "
                          f"cortex_m operators {', '.join(_SCHEMAS)}")
    if op.overload != "out":
        raise pte.Refusal(f"calls operator {name}.{op.overload}; tigris export rewrites "
                          f"programs that call the out variant of each operator")
    sig = SIGNATURES[name]
    args = call.args or []
    if len(args) != len(sig.names):
        raise ValueError(f"{name} takes {len(sig.names)} arguments, the program passes "
                         f"{len(args)}")
    values = plan.values or []
    for position, (arg, arg_name, kinds) in enumerate(zip(args, sig.names, sig.kinds)):
        if not 0 <= arg < len(values):
            raise ValueError(f"{name} argument {arg_name} is value {arg}, which does not exist")
        value = values[arg]
        if not isinstance(value, kinds):
            raise ValueError(f"{name} argument {arg_name} is a {type(value).__name__}, "
                             f"expected {' or '.join(k.__name__ for k in kinds)}")
        if isinstance(value, pte.IntList):
            items = value.items or []
            if any(not 0 <= i < len(values) or not isinstance(values[i], pte.Int)
                   for i in items):
                raise ValueError(f"{name} argument {arg_name} lists a value that is not an Int")
            if arg_name in sig.lengths and len(items) != sig.lengths[arg_name]:
                raise ValueError(f"{name} argument {arg_name} holds {len(items)} values, "
                                 f"expected {sig.lengths[arg_name]}")
    if args[-1] != args[-2]:
        raise ValueError(f"{name} returns a value other than its out tensor")
    return sig


def kernel_calls(plan: pte.ExecutionPlan) -> list[Call]:
    """The plan's instructions as kernel calls with their signatures; raises
    ValueError for a plan of several chains, for any instruction that is not a
    kernel call, and for any call `signature` refuses."""
    chains = plan.chains or []
    if len(chains) != 1:
        raise pte.Refusal(f"runs {len(chains)} chains; tigris export rewrites a method of one "
                          f"chain")
    calls = []
    for index, instruction in enumerate(chains[0].instructions or []):
        if not isinstance(instruction, pte.KernelCall):
            kind = type(instruction).__name__ if instruction is not None else "an empty instruction"
            raise pte.Refusal(f"has a {kind} at instruction {index}; tigris export rewrites "
                              f"methods of kernel calls")
        sig = signature(plan, instruction)
        calls.append(Call(index, sig, list(instruction.args),
                          [plan.values[a] for a in instruction.args]))
    return calls


def known_calls(plan: pte.ExecutionPlan) -> list[Call]:
    """The kernel calls, in every chain, of the out variant of an operator in
    SIGNATURES, with their signatures; other instructions are left out, so
    the runtime that executes the plan decides about them. Raises ValueError
    as `signature` does for a call it accepts by name."""
    operators = plan.operators or []
    calls = []
    for chain in plan.chains or []:
        for index, instruction in enumerate(chain.instructions or []):
            if not isinstance(instruction, pte.KernelCall):
                continue
            op = (operators[instruction.op_index]
                  if 0 <= instruction.op_index < len(operators) else None)
            if op is not None and (op.name not in SIGNATURES or op.overload != "out"):
                continue
            sig = signature(plan, instruction)
            calls.append(Call(index, sig, list(instruction.args),
                              [plan.values[a] for a in instruction.args]))
    return calls


class Dims(NamedTuple):
    n: int
    h: int
    w: int
    c: int


def check_geometry(what: str, inp: Dims, filt: Dims, out: Dims, padding: tuple,
                   stride: tuple, dilation: tuple) -> None:
    """Raises ValueError naming `what` and the argument when a dimension is
    below 1, a padding below 0, or a stride or dilation below 1."""
    for name, dims in (("input", inp), ("weight", filt), ("output", out)):
        if min(dims) < 1:
            raise ValueError(f"{what} has {name} dimensions {list(dims)}, each must be at "
                             f"least 1")
    for name, pair, least in (("padding", padding, 0), ("stride", stride, 1),
                              ("dilation", dilation, 1)):
        if min(pair) < least:
            raise ValueError(f"{what} has {name} {list(pair)}, each must be at least {least}")


def _cdiv(a: int, b: int) -> int:
    """C integer division, truncating toward zero."""
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b > 0) else -q


def _is_1x1(padding: tuple, dilation: tuple, inp: Dims, filt: Dims) -> bool:
    return (padding == (0, 0) and filt.w == 1 and filt.h == 1 and dilation == (1, 1)
            and inp.c == filt.c)


def _is_1_x_n(stride: tuple, dilation: tuple, inp: Dims, filt: Dims) -> bool:
    return (inp.h == 1 and dilation[1] == 1 and filt.h == 1 and (stride[1] * inp.c) % 4 == 0
            and inp.c == filt.c)


def _conv_im2col(cls: str, inp: Dims, filt: Dims) -> int:
    """arm_convolve_s8_get_buffer_size."""
    cols = inp.c * filt.w * filt.h
    if cls == MVE:
        return 4 * ((cols + 15) // 16) * 16
    return 2 * (-(-cols // 4) * 4) * 2


def _conv_1_x_n_mve(padding: tuple, stride: tuple, inp: Dims, filt: Dims, out: Dims) -> int:
    pad_x, stride_x = padding[1], stride[1]
    total_pad = (out.w - 1) * stride_x + filt.w - inp.w
    asym_pad = total_pad - 2 * _cdiv(total_pad, 2)
    right = max(1, _cdiv(pad_x + asym_pad + stride_x - 1, stride_x)) if pad_x + asym_pad else 0
    left = max(1, _cdiv(pad_x + stride_x - 1, stride_x)) if pad_x else 0
    middle = max(out.w - (right + left), 0)
    if right + middle + left != out.w:
        return _conv_im2col(MVE, inp, filt)
    pad_left = pad_x * inp.c
    pad_right = right * inp.c if asym_pad else pad_left
    elem_left = filt.w * inp.c
    return max(elem_left + pad_left, elem_left - inp.c + pad_right)


def conv_scratch(cls: str, inp: Dims, filt: Dims, out: Dims, padding: tuple, stride: tuple,
                 dilation: tuple) -> int:
    """arm_convolve_wrapper_s8_get_buffer_size for `cls`; dimensions are NHWC,
    the filter's (out channels, height, width, in channels), and padding,
    stride and dilation are (height, width). Raises ValueError for geometry
    `check_geometry` refuses."""
    padding, stride, dilation = tuple(padding), tuple(stride), tuple(dilation)
    check_geometry("the convolution", inp, filt, out, padding, stride, dilation)
    if _is_1x1(padding, dilation, inp, filt):
        return 0
    if _is_1_x_n(stride, dilation, inp, filt) and cls == MVE:
        return _conv_1_x_n_mve(padding, stride, inp, filt, out)
    return _conv_im2col(cls, inp, filt)


# Depthwise channels the MVE kernel processes per block (CH_IN_BLOCK_MVE).
_MVE_DEPTHWISE_BLOCK = 124
# Output channels above which MVE runs a one-input-channel depthwise as a convolution.
_MVE_DEPTHWISE_AS_CONV = 1


def depthwise_scratch(cls: str, inp: Dims, filt: Dims, out: Dims, padding: tuple,
                      stride: tuple, dilation: tuple) -> int:
    """arm_depthwise_conv_wrapper_s8_get_buffer_size for `cls`; the filter is
    (out channels, height, width, 1). Raises ValueError for geometry
    `check_geometry` refuses."""
    padding, stride, dilation = tuple(padding), tuple(stride), tuple(dilation)
    check_geometry("the depthwise convolution", inp, filt, out, padding, stride, dilation)
    size = 0
    if inp.c == out.c and inp.n == 1 and dilation == (1, 1):
        if cls == MVE:
            size = 4 * _MVE_DEPTHWISE_BLOCK * filt.w * filt.h
        elif cls == DSP and not (filt.w == 3 and filt.h == 3 and padding[0] <= 1
                                 and padding[1] <= 1):
            size = inp.c * filt.w * filt.h * 2
    if cls == MVE and inp.c == 1 and out.c > _MVE_DEPTHWISE_AS_CONV:
        as_conv = Dims(filt.c, filt.h, filt.w, filt.n)
        size = max(size, conv_scratch(MVE, inp, as_conv, out, padding, stride, dilation)
                   + filt.n * filt.h * filt.w * filt.c)
    return size


def conv_kernel(cls: str, inp: Dims, filt: Dims, padding: tuple, stride: tuple,
                dilation: tuple) -> str:
    """The kernel arm_convolve_wrapper_s8 runs for `cls`: "1x1", "1xN" (MVE
    only; the other classes run it as "im2col") or "im2col"."""
    padding, stride, dilation = tuple(padding), tuple(stride), tuple(dilation)
    if _is_1x1(padding, dilation, inp, filt):
        return "1x1"
    if cls == MVE and _is_1_x_n(stride, dilation, inp, filt):
        return "1xN"
    return "im2col"


def depthwise_kernel(cls: str, inp: Dims, filt: Dims, out: Dims, padding: tuple,
                     stride: tuple, dilation: tuple) -> str:
    """The kernel arm_depthwise_conv_wrapper_s8 runs for `cls`; for a
    depthwise convolution MVE runs as a convolution, that convolution's
    kernel after "as conv"."""
    padding, stride, dilation = tuple(padding), tuple(stride), tuple(dilation)
    if cls == MVE and inp.c == 1 and out.c > _MVE_DEPTHWISE_AS_CONV:
        as_conv = Dims(filt.c, filt.h, filt.w, filt.n)
        return "as conv " + conv_kernel(MVE, inp, as_conv, padding, stride, dilation)
    if out.c == inp.c and inp.n == 1 and dilation == (1, 1):
        if cls != MVE and filt.w == 3 and filt.h == 3 and padding[0] <= 1 and padding[1] <= 1:
            return "3x3"
        return "opt"
    return "generic"


def avgpool_scratch(cls: str, out_width: int, channels: int) -> int:
    """arm_avgpool_s8_get_buffer_size for `cls`, which does not depend on
    `out_width`."""
    return channels * 4 if cls == DSP else 0


def _nhwc(tensor: pte.Tensor, what: str) -> Dims:
    from tigris.frontends.pte_host import MethodTensor

    sizes, order = tensor.sizes or [], tensor.dim_order or []
    if len(sizes) != 4 or list(order) != [0, 2, 3, 1]:
        shown = MethodTensor(tensor.scalar_type, tuple(sizes), tuple(order or range(len(sizes))))
        raise ValueError(f"the {what} is {shown.describe()}, expected a channels-last 4-D "
                         f"tensor (dim order 0,2,3,1)")
    n, c, h, w = sizes
    return Dims(n, h, w, c)


def required_scratch(call: Call, plan: pte.ExecutionPlan) -> dict[str, int] | None:
    """The scratch bytes each class needs for `call`, or None for an operator
    whose scratch does not depend on the class here. Raises ValueError naming
    the instruction for geometry `check_geometry` refuses, and for a linear
    layer whose weights are not 2-D or whose output has no sizes."""
    op = call.signature.name.removeprefix(NAMESPACE)
    what = f"instruction {call.index} ({call.signature.name})"
    if op == "quantized_linear":
        weight = call.value("weights").sizes or []
        if len(weight) != 2:
            raise ValueError(f"the {op} weight has sizes {weight}, expected two dimensions")
        if not call.value("out").sizes:
            raise ValueError(f"the {op} output has sizes {call.value('out').sizes or []}, "
                             f"expected at least one dimension")
        return None
    if op == "quantized_avg_pool2d":
        inp = _nhwc(call.value("input"), f"{op} input")
        out = _nhwc(call.value("out"), f"{op} output")
        kernel, stride, padding = (tuple(call.ints(n, plan))
                                   for n in ("kernel_size", "stride", "padding"))
        if min(kernel) < 1:
            raise ValueError(f"{what} has kernel_size {list(kernel)}, each must be at least 1")
        check_geometry(what, inp, Dims(1, *kernel, 1), out, padding, stride, (1, 1))
        return {cls: avgpool_scratch(cls, out.w, inp.c) for cls in CLASSES}
    if op not in ("quantized_conv2d", "quantized_depthwise_conv2d"):
        return None
    inp = _nhwc(call.value("input"), f"{op} input")
    out = _nhwc(call.value("out"), f"{op} output")
    weight = call.value("weight").sizes or []
    if len(weight) != 4:
        raise ValueError(f"the {op} weight has sizes {weight}, expected four")
    geometry = [tuple(call.ints(name, plan)) for name in ("padding", "stride", "dilation")]
    if op == "quantized_conv2d":
        filt = Dims(*weight)  # OHWI
        check_geometry(what, inp, filt, out, *geometry)
        return {cls: conv_scratch(cls, inp, filt, out, *geometry) for cls in CLASSES}
    filt = Dims(weight[3], weight[1], weight[2], 1)  # stored as 1HWO
    if weight[0] != 1:
        raise ValueError(f"{what} has weight sizes {weight}, expected one in the first")
    check_geometry(what, inp, filt, out, *geometry)
    return {cls: depthwise_scratch(cls, inp, filt, out, *geometry) for cls in CLASSES}


def infer_class(plan: pte.ExecutionPlan, calls: list[Call] | None = None) -> tuple[str, ...]:
    """The CMSIS-NN classes, in CLASSES order, whose kernels every operator of
    `plan` was sized for: each scratch tensor holds the bytes that class
    needs, and each linear layer takes a kernel sum (MVE) or not (SCALAR,
    DSP). `calls` are the calls that decide it, `kernel_calls(plan)` unless
    given. Several classes come back when the program does not tell them
    apart; raises ValueError when none fits, naming the first operator that
    rules out the last one."""
    candidates = list(CLASSES)
    for call in kernel_calls(plan) if calls is None else calls:
        op = call.signature.name
        if op == NAMESPACE + "quantized_linear":
            required_scratch(call, plan)
            mve = isinstance(call.value("kernel_sum"), pte.Tensor)
            if mve and isinstance(call.value("bias"), pte.Tensor):
                raise pte.Refusal(f"fits no CMSIS-NN class: instruction {call.index} ({op}) "
                                  f"takes both a bias and a kernel sum")
            fits = {cls: (cls == MVE) == mve for cls in CLASSES}
            reason = "takes a kernel sum" if mve else "takes no kernel sum"
        else:
            need = required_scratch(call, plan)
            if need is None:
                continue
            (index,) = call.signature.scratch
            stored = call.values[index].nbytes
            fits = {cls: need[cls] == stored for cls in CLASSES}
            reason = (f"stores {stored} scratch bytes where SCALAR, DSP and MVE need "
                      f"{need[SCALAR]}, {need[DSP]} and {need[MVE]}")
        kept = [cls for cls in candidates if fits[cls]]
        if not kept:
            raise pte.Refusal(f"fits no CMSIS-NN class: instruction {call.index} ({op}) "
                              f"{reason}, which rules out {' and '.join(candidates)}")
        candidates = kept
    return tuple(candidates)


def describe(classes: tuple[str, ...]) -> str:
    """The kernels of `classes` as the report names them: one class with its
    cores, several by name."""
    if len(classes) == 1:
        return f"{CLASS_NAMES[classes[0]]} kernels ({CLASS_CORES[classes[0]]})"
    return " or ".join(CLASS_NAMES[cls] for cls in classes) + " kernels"


def cpu_class(cpu: str) -> str:
    """The CMSIS-NN class of a Cortex-M CPU named like "M7", "cortex-m7" or
    "M0+"."""
    key = cpu.strip().upper().removeprefix("CORTEX-").replace("+", "PLUS")
    if key not in CPU_CLASSES:
        raise ValueError(f"unknown Cortex-M CPU {cpu!r}; expected one of "
                         f"{', '.join(c.replace('PLUS', '+') for c in CPU_CLASSES)}")
    return CPU_CLASSES[key]


def check_cpu(classes: tuple[str, ...], cpu: str | None) -> str:
    """The class the program is rewritten for: the one `classes` holds, or
    the class of `cpu`, which must be among `classes`. Raises ValueError for a
    mismatch and when `classes` holds several and no CPU picks one."""
    if cpu is None:
        if len(classes) != 1:
            raise ValueError(f"the program fits CMSIS-NN classes {', '.join(classes)}; "
                             f"name the CPU to pick one")
        return classes[0]
    cls = cpu_class(cpu)
    if cls not in classes:
        raise pte.Refusal(f"was lowered for {describe(classes)}; --cpu {cpu} runs "
                          f"{CLASS_NAMES[cls]} kernels")
    return cls
