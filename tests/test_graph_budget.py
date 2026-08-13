import dataclasses

from tigris.graph.ir import AnalyzedGraph, MemoryBudget


def test_default_budget_is_all_zero():
    ag = AnalyzedGraph()
    assert ag.budget == MemoryBudget()
    assert ag.mem_budget == 0
    assert ag.fast_memory_reserve_bytes == 0


def test_mem_budget_property_reads_fast_tier():
    ag = AnalyzedGraph(budget=MemoryBudget(fast=64, slow=4096, flash=1024, fast_reserve=8))
    assert ag.mem_budget == 64
    assert ag.fast_memory_reserve_bytes == 8


def test_total_fast_pool_is_fast_plus_reserve():
    ag = AnalyzedGraph(budget=MemoryBudget(fast=200, fast_reserve=56))
    assert ag.budget.fast + ag.budget.fast_reserve == 256


def test_memory_budget_is_frozen():
    b = MemoryBudget(fast=64)
    try:
        b.fast = 128  # type: ignore[misc]
    except dataclasses.FrozenInstanceError:
        return
    raise AssertionError("MemoryBudget must be frozen")
