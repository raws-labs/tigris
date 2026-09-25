"""Tensor lifetime computation - birth and death step for each activation."""

import numpy as np

from tigris.graph.ir import AnalyzedGraph, Layout, TensorLifetime, serialized_shape

# Operators that give a tensor a new shape without moving a byte.
_RESHAPE_OPS = frozenset({"Reshape", "Flatten"})


def _states_its_own_order(info) -> bool:
    """Whether the tensor is stored in the axis order it declares.

    A linear tensor is, by definition. So is anything at rank 2 or below,
    where the two orders cannot differ.
    """
    return info.layout is Layout.LINEAR or len(info.shape) <= 2


def _same_quantization(src, dst) -> bool:
    """Whether the two read their bytes as the same values.

    A plan may give a reshape's two tensors different quantization, and then
    it has to re-encode rather than pass through: the bytes are only the same
    bytes if they mean the same thing.
    """
    a, b = src.quant, dst.quant
    if a is None or b is None:
        return a is None and b is None
    return bool(np.array_equal(a.scale, b.scale)
                and np.array_equal(a.zero_point, b.zero_point))


def _same_stored_bytes(src, dst) -> bool:
    """Whether a reshape between these two leaves the stored bytes alone.

    Two tensors that each state their own axis order hold their elements in
    the order the shape lists them, so regrouping the shape moves nothing.

    Two spatial tensors hold their channel axis last and their spatial axes in
    order ahead of it, so regrouping only the spatial axes moves nothing
    either: that is what an unfold is, and it is the shape a vision
    transformer uses to turn a feature map into patches. The batch and the
    channel extent have to survive, because they are the ends of the stored
    order and a reshape that changes either is moving bytes.
    """
    if src.size_bytes != dst.size_bytes:
        return False
    if not _same_quantization(src, dst):
        return False
    src_own = _states_its_own_order(src)
    dst_own = _states_its_own_order(dst)
    if src_own and dst_own:
        return True
    if src_own or dst_own:
        return False
    a = serialized_shape(src.shape, src.layout)
    b = serialized_shape(dst.shape, dst.layout)
    return bool(a) and bool(b) and a[0] == b[0] and a[-1] == b[-1]


def pure_reinterpretations(ag: AnalyzedGraph) -> dict[str, str]:
    """Map each reshape output to the tensor whose bytes it shares.

    A reshape between two tensors that each state their own axis order, with
    the same byte count, produces the same bytes in the same order: only the
    shape the consumer reads differs. The executor gives the two the same
    buffer instead of allocating and copying, so the model counts them as one
    allocation. The chain is followed to its source, since a reshape pair
    around a lowered matrix product is two of these back to back.

    Mirrors is_pure_reinterpretation in the executor, which derives the same
    fact from the plan rather than being told it.
    """
    alias: dict[str, str] = {}
    for op in ag.ops:
        if op.op_type not in _RESHAPE_OPS or len(op.outputs) != 1:
            continue
        sources = [
            name for name in op.inputs
            if name and (info := ag.tensors.get(name)) is not None
            and not info.is_constant
        ]
        if len(sources) != 1:
            continue
        src = ag.tensors[sources[0]]
        dst = ag.tensors.get(op.outputs[0])
        if dst is None or dst.is_constant:
            continue
        if not _same_stored_bytes(src, dst):
            continue
        root = sources[0]
        while root in alias:
            root = alias[root]
        alias[op.outputs[0]] = root
    return alias


def compute_lifetimes(ag: AnalyzedGraph) -> AnalyzedGraph:
    """Walk the sorted ops and record birth/death steps for activation tensors.

    - Model inputs:  birth_step = -1
    - Model outputs: death_step = len(ops) (kept alive beyond last op)
    - Constants (weights/initializers) are skipped entirely.
    """
    num_ops = len(ag.ops)

    birth: dict[str, int] = {}
    death: dict[str, int] = {}

    # Model inputs are born before execution starts
    for name in ag.model_inputs:
        birth[name] = -1

    # Walk ops in execution order
    for op in ag.ops:
        # Outputs of this op are born at this step
        for out_name in op.outputs:
            if out_name == "":
                continue
            info = ag.tensors.get(out_name)
            if info is None or info.is_constant:
                continue
            birth[out_name] = op.step

        # Inputs consumed at this step - update last-use
        for inp_name in op.inputs:
            if inp_name == "":
                continue
            info = ag.tensors.get(inp_name)
            if info is None or info.is_constant:
                continue
            death[inp_name] = op.step

    # Model outputs must survive until after the last op
    for name in ag.model_outputs:
        death[name] = num_ops

    # A reshape that moves no byte shares its input's allocation, so the two
    # are one lifetime: the storage has to survive until the last of them is
    # read, and it is counted once rather than twice.
    alias = pure_reinterpretations(ag)
    for name, root in alias.items():
        if name in death:
            death[root] = max(death.get(root, death[name]), death[name])

    # Build lifetime records
    ag.lifetimes = {}
    for name in birth:
        info = ag.tensors.get(name)
        if info is None or info.is_constant:
            continue
        # An alias owns no storage of its own: its bytes are the root's, whose
        # lifetime was extended above to cover it. It keeps its record, since
        # it is still a tensor the graph names and stages carry.
        size = 0 if name in alias else info.size_bytes
        d = death.get(name, birth[name])  # unused tensor dies at birth
        ag.lifetimes[name] = TensorLifetime(
            tensor_name=name,
            birth_step=birth[name],
            death_step=d,
            size_bytes=size,
        )

    return ag
