import dataclasses

import pytest

from tigris.graph.ir import AnalyzedGraph, MemoryBudget


def test_memory_budget_defaults_properties_and_frozen():
    ag = AnalyzedGraph()
    assert ag.budget == MemoryBudget()
    assert ag.mem_budget == 0
    assert ag.fast_memory_reserve_bytes == 0

    ag = AnalyzedGraph(budget=MemoryBudget(fast=64, slow=4096, flash=1024, fast_reserve=8))
    assert ag.mem_budget == 64
    assert ag.fast_memory_reserve_bytes == 8

    with pytest.raises(dataclasses.FrozenInstanceError):
        ag.budget.fast = 128  # type: ignore[misc]
