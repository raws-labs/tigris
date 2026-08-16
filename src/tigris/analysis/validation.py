"""Fail-closed validation of compiler deployment plans."""

from dataclasses import dataclass

from tigris.analysis.lifetime import compute_lifetimes
from tigris.analysis.partition_spatial import (
    _back_propagate_tile_heights,
    _chain_fast_bytes,
    _get_stage_spatial_params,
)
from tigris.capabilities import KERNEL_CAPABILITIES, effective_operators
from tigris.emitters.binary.defs import OP_TYPE_MAP
from tigris.graph.ir import AnalyzedGraph, Stage


_FLOAT32 = 1
_INT8 = 3


@dataclass(frozen=True)
class ExecutionDTypeValidation:
    """Result of validating the runtime's graph-wide dispatcher dtype."""

    dtype: str | None
    issues: tuple[str, ...]

    @property
    def supported(self) -> bool:
        return not self.issues

    def describe(self) -> str:
        return "; ".join(self.issues)


def validate_execution_dtype(ag: AnalyzedGraph) -> ExecutionDTypeValidation:
    """Require one supported dtype across all non-constant tensors.

    Runtime dispatch is selected once for the whole plan.  A plan containing
    both float32 and int8 activations cannot safely be routed through either
    dispatcher, even if every individual opcode is otherwise supported.
    """
    by_dtype: dict[int, list[str]] = {}
    for name, tensor in ag.tensors.items():
        if not tensor.is_constant:
            by_dtype.setdefault(tensor.dtype, []).append(name)

    if not by_dtype:
        return ExecutionDTypeValidation(dtype=None, issues=())

    labels = {_FLOAT32: "float32", _INT8: "int8"}
    unsupported = sorted(dtype for dtype in by_dtype if dtype not in labels)
    if unsupported:
        details = ", ".join(
            f"ONNX dtype {dtype} ({', '.join(sorted(by_dtype[dtype]))})"
            for dtype in unsupported
        )
        return ExecutionDTypeValidation(
            dtype=None,
            issues=(f"unsupported activation tensor dtype(s): {details}",),
        )

    if len(by_dtype) != 1:
        details = ", ".join(
            f"{labels[dtype]} ({', '.join(sorted(names))})"
            for dtype, names in sorted(by_dtype.items())
        )
        return ExecutionDTypeValidation(
            dtype=None,
            issues=(
                "mixed activation dtypes cannot use a graph-wide runtime "
                f"dispatcher: {details}",
            ),
        )

    tensor_dtype = next(iter(by_dtype))
    expected_dtype = _INT8 if ag.is_quantized else _FLOAT32
    if tensor_dtype != expected_dtype:
        actual = labels[tensor_dtype]
        expected = labels[expected_dtype]
        return ExecutionDTypeValidation(
            dtype=None,
            issues=(
                f"graph quantization metadata selects {expected}, but all "
                f"activation tensors are {actual}",
            ),
        )

    return ExecutionDTypeValidation(dtype=labels[tensor_dtype], issues=())


@dataclass(frozen=True)
class UnsupportedOperatorIssue:
    """A normalized operator that cannot be represented by the plan schema."""

    op_name: str
    op_type: str
    reason: str = ""

    def describe(self) -> str:
        if self.reason:
            return f"{self.op_name} ({self.op_type}: {self.reason})"
        return f"{self.op_name} ({self.op_type})"


@dataclass(frozen=True)
class OperatorSupportValidation:
    """Result of validating normalized operators against the plan schema."""

    issues: tuple[UnsupportedOperatorIssue, ...]

    @property
    def supported(self) -> bool:
        return not self.issues

    def describe(self) -> str:
        return ", ".join(issue.describe() for issue in self.issues)


def _routed_operators() -> frozenset[str]:
    """Operators reachable through some runtime dispatcher, any backend.

    Wire-encodability (OP_TYPE_MAP) and runtime routing (capabilities) are
    two separate contracts. An op can be added to the binary schema before a
    kernel exists for it; without this check the compiler would accept such
    an op and only fail once the plan reaches a device.
    """
    return frozenset().union(
        *(effective_operators(backend) for backend in KERNEL_CAPABILITIES)
    )


def validate_operator_support(ag: AnalyzedGraph) -> OperatorSupportValidation:
    """Return operators or attributes not representable by the plan/runtime."""
    issues: list[UnsupportedOperatorIssue] = []
    routed_operators = _routed_operators()
    for op in ag.ops:
        if op.op_type not in OP_TYPE_MAP:
            issues.append(
                UnsupportedOperatorIssue(op_name=op.name, op_type=op.op_type)
            )
            continue

        if op.op_type not in routed_operators:
            issues.append(
                UnsupportedOperatorIssue(
                    op_name=op.name,
                    op_type=op.op_type,
                    reason=(
                        f"{op.op_type} is wire-encodable but has no runtime kernel"
                    ),
                )
            )
            continue

        reasons: list[str] = []
        auto_pad = op.attrs.get("auto_pad", "NOTSET")
        if op.op_type in {
            "Conv",
            "Conv1D",
            "DepthwiseConv",
            "ConvTranspose",
            "MaxPool",
            "AveragePool",
        } and auto_pad not in ("", "NOTSET"):
            reasons.append(f"auto_pad={auto_pad!r} requires explicit pads")

        if op.op_type == "ConvTranspose":
            group = int(op.attrs.get("group", 1))
            if group != 1:
                reasons.append(f"group={group} is not implemented (group=1 only)")
            dilations = [int(value) for value in op.attrs.get("dilations", [1, 1])]
            if any(value != 1 for value in dilations):
                reasons.append("ConvTranspose dilation is not implemented")

        if op.op_type in {"MaxPool", "AveragePool"}:
            if int(op.attrs.get("ceil_mode", 0)) != 0:
                reasons.append("ceil_mode=1 is not encoded")
            dilations = [int(value) for value in op.attrs.get("dilations", [1, 1])]
            if any(value != 1 for value in dilations):
                reasons.append("pooling dilation is not implemented")

        if op.op_type == "AveragePool" and int(
            op.attrs.get("count_include_pad", 0)
        ) != 0:
            reasons.append("count_include_pad=1 is not encoded")

        if op.op_type == "MaxPool":
            if int(op.attrs.get("storage_order", 0)) != 0:
                reasons.append("storage_order=1 is not encoded")
            if len(op.outputs) != 1:
                reasons.append("MaxPool indices output is not implemented")

        if op.op_type == "Softmax":
            input_tensor = (
                ag.tensors.get(op.inputs[0]) if len(op.inputs) == 1 else None
            )
            if input_tensor is None or not input_tensor.shape:
                reasons.append("runtime requires one concrete, non-scalar input")
            else:
                rank = len(input_tensor.shape)
                axis = int(op.attrs.get("axis", -1))
                normalized_axis = axis + rank if axis < 0 else axis
                # Rank-3/4 activations are converted from NCL/NCHW to NLC/NHWC;
                # their channel axis becomes the runtime's final dimension.
                runtime_final_axis = 1 if rank in {3, 4} else rank - 1
                if normalized_axis != runtime_final_axis:
                    reasons.append(
                        "Softmax axis must map to the runtime's final dimension "
                        f"(axis {runtime_final_axis} for rank {rank})"
                    )

        if op.op_type == "Concat":
            tensors = [ag.tensors.get(name) for name in op.inputs]
            normalized_axis = op.attrs.get("kernel_shape", [])
            if (
                not tensors
                or any(tensor is None or len(tensor.shape) != 4 for tensor in tensors)
                or normalized_axis != [3]
            ):
                reasons.append("runtime supports rank-4 channel-axis Concat only")

        if op.op_type == "Resize":
            if op.attrs.get("mode", "nearest") != "nearest":
                reasons.append("Resize mode must be 'nearest'")
            if op.attrs.get("coordinate_transformation_mode", "half_pixel") != (
                "asymmetric"
            ):
                reasons.append(
                    "Resize coordinate_transformation_mode must be 'asymmetric'"
                )
            if op.attrs.get("nearest_mode", "round_prefer_floor") != "floor":
                reasons.append("Resize nearest_mode must be 'floor'")
            if "axes" in op.attrs:
                reasons.append("Resize axes is not encoded")

            input_tensor = (
                ag.tensors.get(op.inputs[0]) if len(op.inputs) == 1 else None
            )
            output_tensor = (
                ag.tensors.get(op.outputs[0]) if len(op.outputs) == 1 else None
            )
            if (
                input_tensor is None
                or output_tensor is None
                or len(input_tensor.shape) != 4
                or len(output_tensor.shape) != 4
            ):
                reasons.append("runtime supports rank-4 Resize only")
            else:
                in_n, in_c, in_h, in_w = input_tensor.shape
                out_n, out_c, out_h, out_w = output_tensor.shape
                if (
                    in_h <= 0
                    or in_w <= 0
                    or in_n != out_n
                    or in_c != out_c
                    or out_h % in_h != 0
                    or out_w % in_w != 0
                ):
                    reasons.append(
                        "Resize requires unchanged N/C and integer H/W upscaling"
                    )

        if op.op_type in {"Add", "Mul"}:
            dynamic_inputs = [
                ag.tensors[name]
                for name in op.inputs
                if name in ag.tensors and not ag.tensors[name].is_constant
            ]
            constant_names = [
                name
                for name in op.inputs
                if name in ag.weight_data
                or (name in ag.tensors and ag.tensors[name].is_constant)
            ]
            output = ag.tensors.get(op.outputs[0]) if len(op.outputs) == 1 else None

            if len(dynamic_inputs) not in {1, 2} or (
                len(dynamic_inputs) + len(constant_names) != 2
            ):
                reasons.append("runtime requires exactly two tensor operands")
            elif len(op.outputs) != 1 or output is None:
                reasons.append("runtime requires one concrete output tensor")
            else:
                reference_shape = dynamic_inputs[0].shape
                if output.shape != reference_shape:
                    reasons.append("output shape must match the first operand exactly")
                if any(
                    tensor.shape != reference_shape
                    for tensor in dynamic_inputs[1:]
                ):
                    reasons.append("dynamic operand broadcasting is not implemented")

                if constant_names:
                    if ag.is_quantized:
                        reasons.append(
                            "quantized constant operands lack shape/quant metadata"
                        )
                    else:
                        stage = next(
                            (
                                candidate
                                for candidate in ag.stages
                                if candidate.stage_id == op.stage
                            ),
                            None,
                        )
                        tiled_execution = stage is not None and (
                            stage.chain_id != 0xFFFF
                            or (
                                stage.tile_plan is not None
                                and stage.tile_plan.tileable
                            )
                        )
                        for name in constant_names:
                            constant = ag.weight_data.get(name)
                            if constant is None:
                                reasons.append(f"constant operand {name!r} has no data")
                                continue
                            if str(constant.dtype) in {"int8", "int32"}:
                                reasons.append(
                                    f"constant operand {name!r} is not float data"
                                )
                            full_shape_constant = (
                                tuple(constant.shape) == reference_shape
                            )
                            if constant.size != 1 and not full_shape_constant:
                                reasons.append(
                                    f"constant operand {name!r} requires unsupported "
                                    "broadcasting"
                                )
                            elif full_shape_constant and tiled_execution:
                                reasons.append(
                                    f"constant operand {name!r} cannot be offset "
                                    "for tiled execution"
                                )

        if reasons:
            issues.append(
                UnsupportedOperatorIssue(
                    op_name=op.name,
                    op_type=op.op_type,
                    reason=", ".join(reasons),
                )
            )

    return OperatorSupportValidation(issues=tuple(issues))


@dataclass(frozen=True)
class MemoryPlanIssue:
    """A stage or chain that cannot execute within the fast-memory budget."""

    stage_id: int
    required_bytes: int
    budget: int
    reason: str

    def describe(self) -> str:
        return (
            f"stage {self.stage_id} requires {self.required_bytes:,} bytes "
            f"but the fast-memory budget is {self.budget:,} bytes ({self.reason})"
        )


@dataclass(frozen=True)
class MemoryPlanValidation:
    """Result of validating every runtime execution unit in a plan."""

    scheduled_peak_bytes: int
    issues: tuple[MemoryPlanIssue, ...]

    @property
    def feasible(self) -> bool:
        return not self.issues


def validate_memory_plan(
    ag: AnalyzedGraph, *, fast_reserve_bytes: int = 0
) -> MemoryPlanValidation:
    """Validate that every stage or chain fits the fast-memory budget.

    A zero budget means no deployment constraint was requested.  Structural
    analysis and binary-format tests may use that mode; the deployment CLI
    separately requires a positive budget before emitting a plan.
    """
    if fast_reserve_bytes < 0:
        raise ValueError("fast_reserve_bytes must not be negative")

    budget = ag.mem_budget
    if budget <= 0:
        return MemoryPlanValidation(scheduled_peak_bytes=0, issues=())

    issues: list[MemoryPlanIssue] = []
    scheduled_peak = 0

    for stage in ag.stages:
        # Non-head chain members execute as part of their head's memory unit.
        if stage.chain_id != 0xFFFF and stage.chain_id != stage.stage_id:
            continue

        activation_required, reason, invalid = _execution_unit_requirement(ag, stage)
        required = activation_required + fast_reserve_bytes
        scheduled_peak = max(scheduled_peak, required)
        if fast_reserve_bytes:
            reason = f"{reason}; {fast_reserve_bytes:,} bytes reserved"
        if invalid or required > budget:
            issues.append(
                MemoryPlanIssue(
                    stage_id=stage.stage_id,
                    required_bytes=required,
                    budget=budget,
                    reason=reason,
                )
            )

    if not ag.stages:
        issues.append(
            MemoryPlanIssue(
                stage_id=0,
                required_bytes=ag.peak_memory_bytes,
                budget=budget,
                reason="no executable stages were produced",
            )
        )

    return MemoryPlanValidation(
        scheduled_peak_bytes=scheduled_peak,
        issues=tuple(issues),
    )


def _execution_unit_requirement(
    ag: AnalyzedGraph, stage: Stage
) -> tuple[int, str, bool]:
    """Return ``(required_bytes, reason, invalid)`` for an execution unit."""
    if stage.chain_id != 0xFFFF:
        chain_stages = sorted(
            (candidate for candidate in ag.stages if candidate.chain_id == stage.chain_id),
            key=lambda candidate: candidate.stage_id,
        )
        if (
            stage.chain_id != stage.stage_id
            or stage.chain_tile_h <= 0
            or len(chain_stages) != stage.chain_len
        ):
            return stage.peak_bytes, "invalid chain metadata", True

        params = [_get_stage_spatial_params(ag, candidate) for candidate in chain_stages]
        heights = _back_propagate_tile_heights(params, stage.chain_tile_h)
        required = _chain_fast_bytes(ag, chain_stages, heights)
        return required, f"chain of {len(chain_stages)} stages", False

    tile_plan = stage.tile_plan
    if tile_plan is not None:
        if not tile_plan.tileable:
            detail = ", ".join(tile_plan.untileable_ops)
            reason = f"untileable operators: {detail}" if detail else "stage is not tileable"
            return stage.peak_bytes, reason, True
        if tile_plan.tiled_peak_bytes <= 0:
            return stage.peak_bytes, "tile solver produced no positive working set", True
        if tile_plan.min_2d_tile_infeasible:
            return tile_plan.tiled_peak_bytes, "minimum 2D tile", False
        return tile_plan.tiled_peak_bytes, "minimum spatial tile", False

    return stage.peak_bytes, "untiled stage", False


@dataclass(frozen=True)
class SlowMemoryUsage:
    slow_peak_bytes: int
    slow_budget: int
    overflow_stage_ids: tuple[int, ...]

    @property
    def fits(self) -> bool:
        return not self.overflow_stage_ids

    def describe(self) -> str:
        from tigris.utils import fmt_bytes
        return (
            f"{len(self.overflow_stage_ids)} stage(s) overflow slow memory "
            f"({fmt_bytes(self.slow_peak_bytes)} needed, "
            f"{fmt_bytes(self.slow_budget)} available)"
        )


def slow_pool_usage(ag: AnalyzedGraph) -> SlowMemoryUsage:
    """Slow/PSRAM residency for tiled stages. Empty overflow_stage_ids == fits.

    A stage needs tiling when its peak exceeds the TOTAL fast pool
    (fast + reserve), so compressed and uncompressed compiles gate identically
    and match analyze (where reserve is 0).

    The slow-resident set is every tensor that crosses a stage boundary (a
    stage's input or output tensor). A tensor born before a tiled stage and
    not yet consumed by the time that stage runs stays slow-resident
    throughout it, so the peak is the max over every op-step belonging to a
    tiled stage of the total bytes of tensors slow-resident at that step,
    not just the current stage's own input+output (which under-counts a
    long-lived skip that spans several stages).

    Liveness uses the closed-closed window birth_step <= step <= death_step,
    matching _live_bytes_by_step and stage.peak_bytes (partition_temporal.py)
    so a tensor that is both consumed and produced at the same op-step is
    counted at that step by both the fast-pool and slow-pool models. This is
    the conservative (upper-bounding) choice for a fail-closed budget check.
    """
    slow_budget = ag.budget.slow
    if slow_budget <= 0 or not ag.stages:
        return SlowMemoryUsage(0, slow_budget, ())
    fast_total = ag.budget.fast + ag.budget.fast_reserve
    ag = compute_lifetimes(ag)

    # Slow-resident set: every tensor that crosses a stage boundary.
    slow_names: set[str] = set()
    for s in ag.stages:
        slow_names.update(s.input_tensors)
        slow_names.update(s.output_tensors)
    slow_lifetimes = [ag.lifetimes[n] for n in slow_names if n in ag.lifetimes]

    def live_bytes(step: int) -> int:
        return sum(
            lt.size_bytes
            for lt in slow_lifetimes
            if lt.birth_step <= step <= lt.death_step
        )

    peak = 0
    overflow: list[int] = []
    for s in ag.stages:
        if s.peak_bytes <= fast_total:
            continue
        stage_peak = max((live_bytes(t) for t in s.op_indices), default=0)
        peak = max(peak, stage_peak)
        if stage_peak > slow_budget:
            overflow.append(s.stage_id)
    return SlowMemoryUsage(peak, slow_budget, tuple(overflow))


@dataclass(frozen=True)
class BudgetValidation:
    fast: MemoryPlanValidation
    slow: SlowMemoryUsage

    @property
    def feasible(self) -> bool:
        return self.fast.feasible and self.slow.fits


def validate_budget(ag: AnalyzedGraph) -> BudgetValidation:
    return BudgetValidation(fast=validate_memory_plan(ag), slow=slow_pool_usage(ag))
