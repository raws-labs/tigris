"""Names and uses shared by the normalization passes."""

from collections import Counter
from collections.abc import Collection
from dataclasses import replace

from tigris.graph.ir import AnalyzedGraph, OpNode


def _fresh_name(ag: AnalyzedGraph, base: str, taken: Collection[str] = ()) -> str:
    """A tensor name the graph does not already carry and ``taken`` does not hold."""
    name = base
    suffix = 0
    while name in ag.tensors or name in ag.weight_data or name in taken:
        suffix += 1
        name = f"{base}_{suffix}"
    return name


class _Uses:
    """Which operator writes each name and which operand positions read it.

    It holds the operators as they stood when it was built, so a pass that
    rewires as it goes keeps reading the graph it started from; only
    ``sole_reader`` checks the model outputs as they are when asked. An
    omitted operand ("") is no read. ``reads`` counts each name once per
    operand position and once per model output, and the folds that copy a
    shared constant keep it current.
    """

    def __init__(self, ag: AnalyzedGraph):
        self._ag = ag
        self._ops = ag.ops
        self.producers: dict[str, int] = {}
        self._readers: dict[str, list[tuple[int, int]]] = {}
        for index, op in enumerate(ag.ops):
            for position, name in enumerate(op.inputs):
                if name:
                    self._readers.setdefault(name, []).append((index, position))
            for name in op.outputs:
                self.producers[name] = index
        self.reads = Counter({name: len(readers) for name, readers in self._readers.items()})
        self.reads.update(ag.model_outputs)

    def producer(self, name: str) -> OpNode | None:
        """The operator ``producers`` points at for ``name``, or None."""
        index = self.producers.get(name)
        return None if index is None else self._ops[index]

    def readers(self, name: str) -> list[tuple[int, int]]:
        """(operator index, operand position) for every position that reads ``name``."""
        return self._readers.get(name, [])

    def reader_count(self, name: str) -> int:
        return len(self._readers.get(name, ()))

    def reading_ops(self, name: str) -> set[int]:
        """The operators that read ``name``, each once however many operands it gives it."""
        return {index for index, _ in self._readers.get(name, ())}

    def sole_reader(self, name: str) -> int | None:
        """The operator reading ``name`` if exactly one operand position does and it is no model output."""
        readers = self._readers.get(name, ())
        if len(readers) != 1 or name in self._ag.model_outputs:
            return None
        return readers[0][0]


def _own_constant(ag: AnalyzedGraph, op: OpNode, position: int, reads: Counter) -> str:
    """The constant ``op`` reads at ``position``, under a name nothing else reads.

    A fold rewrites the constant it returns. One that ``reads`` counts more than
    once is copied under a fresh name and ``op`` reads the copy, so every other
    reader keeps the original. ``reads`` is kept current.
    """
    name = op.inputs[position]
    if reads[name] <= 1:
        return name
    copy = _fresh_name(ag, name)
    ag.weight_data[copy] = ag.weight_data[name].copy()
    if name in ag.tensors:
        ag.tensors[copy] = replace(ag.tensors[name], name=copy)
    op.inputs[position] = copy
    reads[name] -= 1
    reads[copy] += 1
    return copy


def _forget_unread(ag: AnalyzedGraph, names) -> None:
    """Drop each constant in ``names`` that no operator reads and the model does not return."""
    reads = _Uses(ag).reads
    for name in names:
        if name in ag.weight_data and not reads[name]:
            del ag.weight_data[name]
            ag.tensors.pop(name, None)
