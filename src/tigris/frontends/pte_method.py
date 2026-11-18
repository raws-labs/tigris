"""The memory ExecuTorch 1.6.0 takes to load a method on a 32-bit Arm target.

`Method::load` and `Method::init` (runtime/executor/method.cpp) allocate the
method's bookkeeping from the method allocator, a bump allocator that aligns
each allocation from where the last one ended (runtime/core/memory_allocator.h).
`method_allocator_bytes` replays those allocations in order with the sizes and
alignments of the runtime's types under the 32-bit Arm procedure call
standard; the planned buffers and the program's constant data are not part of
it.

Resolving each kernel takes a TensorMeta and a dim order list per argument
from the temp allocator when its capacity is at least 28 bytes per argument,
and from the method allocator otherwise; the temp allocator is reset after
each kernel, so `temp_allocator_min_bytes` is enough for every one of them.
"""

from tigris.frontends import pte

# (size, alignment) of the runtime types allocated, on a 32-bit Arm target.
_EVALUE = (16, 8)
_TENSOR_IMPL = (32, 4)
_TENSOR = (4, 4)  # a Tensor holds a TensorImpl pointer
_BOXED_LIST = (12, 4)  # BoxedEvalueList<T>
_ARRAY_REF = (8, 4)
_OPTIONAL_TENSOR = (8, 4)
_POINTER = (4, 4)
_INT64 = (8, 8)
_DIM_ORDER = (1, 1)  # DimOrderType
_STRIDE = (4, 4)  # StridesType
_CHAIN = (16, 4)
_INSTRUCTION_ARGS = (8, 4)  # Span<EValue*>
_OP_FUNCTION = (4, 4)
_TENSOR_META = (12, 4)
_BACKEND_DELEGATE_ALIGN = 8
_PLATFORM_ALLOCATOR = (28, 4)  # the temp allocator made when none is given
_DIM_LIMIT = 16  # kTensorDimensionLimit
_STATIC = 0  # TensorShapeDynamism::STATIC
# A TensorMeta and a dim order list of the rank limit, per kernel argument.
_META_PER_ARG = _TENSOR_META[0] + _DIM_ORDER[0] * _DIM_LIMIT


class _Bump:
    def __init__(self):
        self.used = 0

    def take(self, size: int, alignment: int) -> None:
        self.used = -(-self.used // alignment) * alignment + size

    def array(self, count: int, item: tuple[int, int]) -> None:
        self.take(count * item[0], item[1])


def temp_allocator_min_bytes(plan: pte.ExecutionPlan) -> int:
    """The smallest temp allocator that resolves every kernel of `plan`:
    a TensorMeta and a dim order list of the rank limit per argument of the
    kernel call with the most arguments."""
    return _META_PER_ARG * max((len(i.args or []) for c in plan.chains or ()
                                for i in c.instructions or () if isinstance(i, pte.KernelCall)),
                               default=0)


def method_allocator_bytes(plan: pte.ExecutionPlan, temp_bytes: int | None = None,
                           temp_allocator: bool = True) -> int:
    """Bytes of the method allocator that loading `plan` uses. `temp_bytes` is
    the temp allocator's capacity, None for one of at least
    `temp_allocator_min_bytes`; with `temp_allocator` False the runner passes
    none and the runtime takes an empty one from the method allocator. Raises
    ValueError for a plan with delegates, external constants or a dynamic
    tensor, whose allocations are not replayed, and for one the runtime refuses
    to load: an argument naming no value or a tensor above the rank limit."""
    for delegate in plan.delegates or ():
        raise pte.Refusal(f"delegates to {delegate.id}; tigris export rewrites programs that "
                          f"run on cortex_m operators")
    values = plan.values or []
    for chain in plan.chains or []:
        for index, instruction in enumerate(chain.instructions or []):
            if not isinstance(instruction, pte.KernelCall):
                continue
            for arg in instruction.args or []:
                if not 0 <= arg < len(values):
                    raise ValueError(f"instruction {index} passes value {arg}, which does "
                                     f"not exist")
    a = _Bump()
    if not temp_allocator:
        a.array(1, _PLATFORM_ALLOCATOR)
        temp_bytes = 0
    elif temp_bytes is None:
        temp_bytes = temp_allocator_min_bytes(plan)
    a.array(len(values), _EVALUE)  # one EValue per value
    if plan.inputs:
        a.take(len(plan.inputs), 1)  # one bool per input: whether it is set
    for value in values:
        if isinstance(value, pte.IntList):
            count = len(value.items or [])
            a.array(count, _POINTER)  # the boxed list: an EValue pointer per item
            a.array(count, _INT64)  # and the unboxed values
            a.array(1, _BOXED_LIST)
        elif isinstance(value, (pte.BoolList, pte.DoubleList, pte.String)):
            a.array(1, _ARRAY_REF)
        elif isinstance(value, pte.Tensor):
            info = value.extra_tensor_info
            if info is not None and info.location != 0:
                raise pte.Refusal("has a tensor stored outside the program; tigris export "
                                  "rewrites programs that hold all their tensors")
            if value.shape_dynamism != _STATIC:
                raise pte.Refusal("has a tensor with a dynamic shape; tigris export rewrites "
                                  "programs with static shapes")
            rank = len(value.sizes or [])
            if rank > _DIM_LIMIT:
                raise pte.Refusal(f"has a tensor of rank {rank}, above the runtime's limit "
                                  f"of {_DIM_LIMIT}")
            a.array(rank, _STRIDE)
            a.array(1, _TENSOR_IMPL)
        elif isinstance(value, pte.TensorList):
            count = len(value.items or [])
            a.array(count, _TENSOR)
            a.array(count, _POINTER)
            a.array(1, _BOXED_LIST)
        elif isinstance(value, pte.OptionalTensorList):
            count = len(value.items or [])
            a.array(count, _POINTER)
            a.array(count, _OPTIONAL_TENSOR)
            a.array(1, _BOXED_LIST)
    a.take(0, _BACKEND_DELEGATE_ALIGN)  # the empty list of delegates
    chains = plan.chains or []
    a.array(len(chains), _CHAIN)
    for chain in chains:
        instructions = chain.instructions or []
        a.array(len(instructions), _OP_FUNCTION)
        a.array(len(instructions), _INSTRUCTION_ARGS)
        for instruction in instructions:
            if not isinstance(instruction, pte.KernelCall):
                continue
            args = instruction.args or []
            a.array(len(args), _POINTER)  # an EValue pointer per argument
            if temp_bytes == 0 or temp_bytes < _META_PER_ARG * len(args):
                a.array(len(args), _TENSOR_META)  # resolved from the method allocator
                for arg in args:
                    if isinstance(values[arg], pte.Tensor):
                        a.array(len(values[arg].sizes or []), _DIM_ORDER)
    return a.used
