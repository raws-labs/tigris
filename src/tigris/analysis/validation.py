"""Fail-closed validation of compiler deployment plans."""

from dataclasses import dataclass

from tigris.analysis.partition_spatial import (
    _back_propagate_tile_heights,
    _chain_fast_bytes,
    _get_stage_spatial_params,
)
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


def validate_operator_support(ag: AnalyzedGraph) -> OperatorSupportValidation:
    """Return operators or attributes not representable by the plan/runtime."""
    issues: list[UnsupportedOperatorIssue] = []
    for op in ag.ops:
        if op.op_type not in OP_TYPE_MAP:
            issues.append(
                UnsupportedOperatorIssue(op_name=op.name, op_type=op.op_type)
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
        return tile_plan.tiled_peak_bytes, "minimum spatial tile", False

    return stage.peak_bytes, "untiled stage", False
