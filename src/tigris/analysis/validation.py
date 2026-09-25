"""Fail-closed validation of compiler deployment plans."""

from dataclasses import dataclass

from tigris import TILE_AXIS_HEIGHT_OR_LENGTH
from tigris.analysis.lifetime import compute_lifetimes, pure_reinterpretations
from tigris.analysis.partition_spatial import (
    _back_propagate_tile_heights,
    _chain_fast_bytes,
    _get_stage_spatial_params,
    _row_view,
    _stage_is_row_tiled,
)
from tigris.capabilities import KERNEL_CAPABILITIES, effective_operators
from tigris.emitters.binary.defs import OP_TYPE_MAP
from tigris.graph.ir import (
    AnalyzedGraph,
    Layout,
    Stage,
    serialized_axis_map,
)


_FLOAT32 = 1
_INT8 = 3

# ONNX expresses static quantization two ways. QDQ keeps float operators and
# brackets them with QuantizeLinear/DequantizeLinear pairs, which normalize.py
# folds into the operator. QOperator replaces the operator itself with a fused
# integer one, and carries the scales as operator inputs, so there is nothing
# to fold and the graph never becomes quantized as far as the rest of the
# compiler is concerned. Naming them is what turns "unsupported operator" into
# an instruction the user can act on.
QOPERATOR_OP_TYPES = frozenset({
    "QLinearConv",
    "QLinearMatMul",
    "QGemm",
    "QLinearAdd",
    "QLinearMul",
    "QLinearAveragePool",
    "QLinearGlobalAveragePool",
    "QLinearConcat",
    "QLinearLeakyRelu",
    "QLinearSigmoid",
    "QLinearSoftmax",
    "ConvInteger",
    "MatMulInteger",
    "DynamicQuantizeLinear",
    "DynamicQuantizeMatMul",
    "DynamicQuantizeLSTM",
})

_QOPERATOR_ADVICE = (
    "ONNX QOperator format; TiGrIS ingests QDQ. Re-export with "
    "quant_format=QuantFormat.QDQ"
)


def _qoperator_op_types(ag: AnalyzedGraph) -> list[str]:
    """QOperator operator types present, in first-seen order."""
    seen: list[str] = []
    for op in ag.ops:
        if op.op_type in QOPERATOR_OP_TYPES and op.op_type not in seen:
            seen.append(op.op_type)
    return seen


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
        # A QOperator graph always lands here: its operators carry the scales,
        # so nothing folds and is_quantized stays False while the tensors are
        # int8. That is the format, not a second defect, and
        # validate_operator_support already fails the model with the advice
        # attached. Reporting it twice, in contradictory terms, is what sent a
        # reader looking for a dtype problem that does not exist.
        if _qoperator_op_types(ag):
            return ExecutionDTypeValidation(dtype=None, issues=())
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
                UnsupportedOperatorIssue(
                    op_name=op.name,
                    op_type=op.op_type,
                    reason=(
                        _QOPERATOR_ADVICE
                        if op.op_type in QOPERATOR_OP_TYPES
                        else ""
                    ),
                )
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

        # Without padding every window lies inside the input, so counting the
        # padded positions changes nothing: PyTorch states the flag by default.
        if (
            op.op_type == "AveragePool"
            and int(op.attrs.get("count_include_pad", 0)) != 0
            and any(int(pad) != 0 for pad in op.attrs.get("pads", []))
        ):
            reasons.append("count_include_pad=1 with padding is not encoded")

        if op.op_type == "MaxPool":
            if int(op.attrs.get("storage_order", 0)) != 0:
                reasons.append("storage_order=1 is not encoded")
            if len(op.outputs) != 1:
                reasons.append("MaxPool indices output is not implemented")

        if op.op_type in ("Softmax", "LayerNormalization"):
            input_tensor = (
                ag.tensors.get(op.inputs[0]) if op.inputs else None
            )
            if input_tensor is None or not input_tensor.shape:
                reasons.append("runtime requires one concrete, non-scalar input")
            else:
                rank = len(input_tensor.shape)
                axis = int(op.attrs.get("axis", -1))
                normalized_axis = axis + rank if axis < 0 else axis
                # The kernel reduces along the final stored dimension. Spatial
                # storage puts the channel axis there, the model's own order
                # puts the last ONNX axis there, so which axis is reducible is
                # a property of the tensor rather than of its rank.
                if rank in {3, 4} and input_tensor.layout is Layout.SPATIAL:
                    runtime_final_axis = 1
                else:
                    runtime_final_axis = rank - 1
                if normalized_axis != runtime_final_axis:
                    reasons.append(
                        f"{op.op_type} axis must map to the runtime's final "
                        f"dimension (axis {runtime_final_axis} for rank {rank} "
                        f"in {input_tensor.layout.value} layout)"
                    )

        if op.op_type == "LayerNormalization":
            # Scale is a weight and bias an optional one; both are indexed by
            # position along the normalized axis, so neither may be an
            # activation the plan would have to route.
            if not 2 <= len(op.inputs) <= 3:
                reasons.append(
                    "runtime requires a scale and an optional bias, both "
                    "constant"
                )
            elif any(name not in ag.weight_data for name in op.inputs[1:]):
                reasons.append("scale and bias must be constants")
            if len(op.outputs) != 1:
                reasons.append(
                    "the mean and inverse-deviation outputs are not implemented"
                )
            if int(op.attrs.get("stash_type", 1)) != 1:
                reasons.append("stash_type other than float32 is not encoded")

        if op.op_type == "Gemm":
            if int(op.attrs.get("transA", 0)) != 0:
                reasons.append(
                    "transA transposes an activation, which the plan has no "
                    "field for and no constant can absorb")
            for name in ("alpha", "beta"):
                if float(op.attrs.get(name, 1.0)) != 1.0:
                    reasons.append(
                        f"{name} is not 1 and could not be folded into a "
                        "constant, and the plan has no field for it")

        if op.op_type == "Sub":
            dynamic = [
                name for name in op.inputs
                if name in ag.tensors and not ag.tensors[name].is_constant
            ]
            if len(dynamic) != 2:
                reasons.append(
                    "Sub does not commute and the plan does not record which "
                    "side a constant was on; a constant subtrahend is rewritten "
                    "as an added negation, a constant minuend is not expressible"
                )

        if op.op_type == "Concat":
            tensors = [ag.tensors.get(name) for name in op.inputs]
            result = ag.tensors.get(op.outputs[0]) if op.outputs else None
            normalized_axis = op.attrs.get("kernel_shape", [])
            rank = len(result.shape) if result is not None else 0
            if not tensors or result is None or any(
                tensor is None or len(tensor.shape) != rank for tensor in tensors
            ):
                reasons.append("Concat operands are not runtime tensors")
            elif rank == 4:
                if normalized_axis != [3]:
                    reasons.append(
                        "runtime concatenates a rank-4 tensor on its channel "
                        "axis only"
                    )
            elif any(
                name in ag.weight_data
                for name in op.inputs[1:]
            ):
                reasons.append(
                    "a constant Concat operand is only expressible as the "
                    "leading part"
                )
            elif op.inputs[0] in ag.weight_data and (
                ag.tensors[op.outputs[0]].quant is not None
            ):
                reasons.append(
                    "a quantized Concat carries no scale for a constant part"
                )
            elif rank == 3:
                # The runtime walks the positions ahead of the last stored
                # axis and copies one run per input, so that is the only axis
                # it can cut.
                if normalized_axis != [2]:
                    reasons.append(
                        "runtime concatenates a rank-3 tensor on its last "
                        "stored axis only"
                    )
            else:
                reasons.append("runtime concatenates rank-3 and rank-4 only")

        if op.op_type == "Split":
            source = ag.tensors.get(op.inputs[0])
            parts = [ag.tensors.get(name) for name in op.outputs]
            axis = int(op.attrs.get("axis", 0))
            if source is None or any(part is None for part in parts):
                reasons.append("Split operands are not runtime tensors")
            else:
                rank = len(source.shape)
                stored = serialized_axis_map(rank, source.layout)
                # Only the outermost stored axis leaves every part a
                # contiguous run of the input; anything else interleaves.
                if rank == 0 or stored[axis % rank] != 0:
                    reasons.append(
                        "runtime splits only along the outermost stored axis"
                    )
                elif any(
                    len(part.shape) != rank
                    or tuple(part.shape[1:]) != tuple(source.shape[1:])
                    for part in parts
                ):
                    reasons.append(
                        "every Split part keeps the shape it was cut from"
                    )

        if op.op_type in {"Resize", "ResizeLinear"}:
            linear = op.op_type == "ResizeLinear"
            if linear:
                # Both conventions execute: half-pixel places a sample at
                # (o + 0.5) / scale - 0.5, asymmetric at o / scale. The
                # loader has already resolved which one the opset means.
                if op.attrs.get("coordinate_transformation_mode",
                                "half_pixel") not in {
                    "half_pixel", "asymmetric"
                }:
                    reasons.append(
                        "bilinear Resize coordinate_transformation_mode must "
                        "be 'half_pixel' or 'asymmetric'"
                    )
            else:
                if op.attrs.get("mode", "nearest") != "nearest":
                    reasons.append("Resize mode must be 'nearest'")
                if op.attrs.get("coordinate_transformation_mode",
                                "half_pixel") != "asymmetric":
                    reasons.append(
                        "Resize coordinate_transformation_mode must be "
                        "'asymmetric'"
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
                per_channel = [
                    tensor
                    for tensor in dynamic_inputs[1:]
                    if tensor.shape != reference_shape
                ]
                # The runtime takes a per-channel operand at the full
                # operand's rank; a constant may be written shorter, a
                # dynamic tensor may not.
                if any(
                    len(tensor.shape) != len(reference_shape)
                    or not _is_per_channel_constant(
                        tuple(tensor.shape), dynamic_inputs[0])
                    for tensor in per_channel
                ):
                    reasons.append(
                        "a dynamic second operand must have the first "
                        "operand's shape or one value per channel"
                    )
                elif per_channel:
                    stage = next(
                        (candidate for candidate in ag.stages
                         if candidate.stage_id == op.stage),
                        None,
                    )
                    # One row high, a per-channel operand is loaded whole
                    # for every band of a rank-4 height stripe. No other tile
                    # executor has that rule.
                    tiled = stage is not None and stage.tile_plan is not None \
                        and stage.tile_plan.tileable
                    if stage is not None and (
                        stage.chain_id != 0xFFFF
                        or (tiled and (
                            stage.tile_plan.tile_width != 0
                            or stage.tile_plan.axis != TILE_AXIS_HEIGHT_OR_LENGTH
                            or len(dynamic_inputs[0].shape) != 4))
                    ):
                        reasons.append(
                            "a per-channel operand is banded only by a "
                            "rank-4 height stripe"
                        )

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
                            per_channel = _is_per_channel_constant(
                                tuple(constant.shape), dynamic_inputs[0]
                            )
                            if (
                                constant.size != 1
                                and not full_shape_constant
                                and not per_channel
                            ):
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


def _is_per_channel_constant(
    constant_shape: tuple[int, ...], operand
) -> bool:
    """Whether a constant carries one value per channel of the operand.

    Every tensor reaches the runtime with its channels innermost, so a
    constant of that length repeats on its own wherever a tile starts and
    needs no shape of its own in the plan. The constant has to say so in the
    model's own axis order: one extent on the channel axis and one everywhere
    else, which is how an exporter writes an input normalization.
    """
    shape = tuple(int(dim) for dim in operand.shape)
    if not shape or not constant_shape:
        return False
    axis = len(shape) - 1 if operand.layout is Layout.LINEAR else 1
    if axis >= len(shape):
        return False
    channels = shape[axis]
    if channels <= 1:
        return False
    padded = (1,) * (len(shape) - len(constant_shape)) + tuple(constant_shape)
    if len(padded) != len(shape):
        return False
    return all(
        dim == (channels if index == axis else 1)
        for index, dim in enumerate(padded)
    )


def _runs_whole(ag: AnalyzedGraph, stage: Stage, fast_total: int) -> bool:
    """Whether the runtime executes this stage in one pass rather than in tiles.

    A stage reaches the tiled path only when it carries a usable tile plan and
    its own boundary tensors do not fit the fast pool together, which is the
    test the executor makes before it picks a path.
    """
    if stage.chain_len >= 2:
        return False
    plan = stage.tile_plan
    if plan is None or not plan.tileable:
        return True
    names = dict.fromkeys([*stage.input_tensors, *stage.output_tensors])
    total_io = sum(
        ag.tensors[n].size_bytes for n in names if n in ag.tensors
    )
    return total_io <= fast_total


def untiled_stage_spill(
    ag: AnalyzedGraph, stage: Stage, fast_total: int
) -> int:
    """Bytes a stage run in one pass pushes into slow memory on its own.

    The executor allocates each operator's output in the fast pool, compacts
    when that fails, and falls back to slow when it still does not fit. What
    lands in slow stays there for the rest of the stage: nothing compacts the
    slow pool until the stage is over. So a stage whose working set exceeds
    the fast pool leaves a trail of intermediates in slow that no boundary
    tensor accounts for, and a budget check that counts only boundary tensors
    reads far too low.

    This walks the stage the way the executor does, holding live bytes rather
    than a bump pointer because compaction is what makes those equivalent.
    """
    alias = pure_reinterpretations(ag)
    align = ag.tensor_alignment

    def sized(name: str) -> int:
        info = ag.tensors.get(name)
        if info is None:
            return 0
        return (info.size_bytes + align - 1) & ~(align - 1)

    ops = [ag.ops[i] for i in stage.op_indices]
    outputs = set(stage.output_tensors)
    last_use: dict[str, int] = {}
    for position, op in enumerate(ops):
        for name in op.inputs:
            if name in ag.tensors and not ag.tensors[name].is_constant:
                last_use[name] = position

    live: dict[str, int] = {}
    for name in dict.fromkeys(stage.input_tensors):
        live[name] = sized(name)
    placed = set(live)
    spilled = 0

    for position, op in enumerate(ops):
        for name in op.outputs:
            if name in placed or name not in ag.tensors:
                continue
            if alias.get(name) is not None:
                placed.add(name)
                continue
            size = sized(name)
            if sum(live.values()) + size <= fast_total:
                live[name] = size
            elif name not in outputs:
                # A stage output that lands in slow is already counted as a
                # tensor crossing the boundary; only the intermediates the
                # stage leaves behind are extra.
                spilled += size
            placed.add(name)
        for name in op.inputs:
            if last_use.get(name) == position and name not in outputs:
                live.pop(name, None)
    return spilled


def row_band_aliases(ag: AnalyzedGraph, fast_total: int) -> dict[str, str]:
    """Stage outputs the runtime writes over their own input, output -> input.

    A row-banded stage gathers a band out of its input before any op runs and
    scatters the result back to the same rows, so one buffer can carry both
    tensors. The runtime does that whenever the two agree on size and on row
    view, nothing after the stage reads the input, and the input is not a
    model input, whose buffer belongs to the caller. Mirrored here so the slow
    pool is not sized for a pair that never coexists.

    The rule is deliberately the narrower of the two: every condition the
    runtime tests is tested here, plus the same total-I/O threshold that sends
    a stage down the banded path at all. Claiming an alias the runtime does
    not make would under-size the pool.
    """
    last_reader: dict[str, int] = {}
    for stage in ag.stages:
        for name in stage.input_tensors:
            last_reader[name] = stage.stage_id
    model_inputs = set(ag.model_inputs)

    aliases: dict[str, str] = {}
    for stage in ag.stages:
        if len(stage.input_tensors) != 1 or len(stage.output_tensors) != 1:
            continue
        src, dst = stage.input_tensors[0], stage.output_tensors[0]
        if src in model_inputs or last_reader.get(src) != stage.stage_id:
            continue
        plan = stage.tile_plan
        if plan is None or not plan.tileable or plan.axis != TILE_AXIS_HEIGHT_OR_LENGTH:
            continue
        stage_ops = [ag.ops[i] for i in stage.op_indices]
        if not _stage_is_row_tiled(ag, stage, stage_ops):
            continue
        in_info, out_info = ag.tensors.get(src), ag.tensors.get(dst)
        if in_info is None or out_info is None:
            continue
        if in_info.size_bytes != out_info.size_bytes:
            continue
        if _row_view(in_info) != _row_view(out_info):
            continue
        total_io = in_info.size_bytes + out_info.size_bytes
        if total_io <= fast_total:
            continue
        aliases[dst] = src
    return aliases


def slow_pool_usage(ag: AnalyzedGraph) -> SlowMemoryUsage:
    """Slow/PSRAM residency. Empty overflow_stage_ids == fits.

    Every stage spills its outputs to slow memory and loads its inputs back
    from there, tiled or not, so residency is measured over every stage rather
    than only the ones that need tiling.

    The slow-resident set is every tensor that crosses a stage boundary (a
    stage's input or output tensor). PSRAM is freed at STAGE granularity, not
    at per-op granularity, so residency is measured per tiled stage over that
    stage's whole op-step INTERVAL, never sampled at individual op steps.

    For a tiled stage S, let its op-step interval be
    [first, last] = [min(op_indices), max(op_indices)]. A boundary tensor is
    slow-resident during S iff its lifetime interval overlaps that interval:
    birth_step <= last and death_step >= first. The stage's residency is the
    sum of those tensors' sizes; the peak is the max over tiled stages, and a
    stage overflows when its sum exceeds slow_budget.

    Interval overlap counts a stage's own inputs AND outputs concurrently
    (both always overlap S) as well as a long-lived skip that spans S (its
    interval still overlaps). A per-op-step sample under-counts a MULTI-OP
    stage whose input dies at an early op step and whose output is born at a
    later op step to max(input, output): no single sampled step sees both,
    even though both occupy slow memory for the whole stage. Interval overlap
    upper-bounds true concurrent residency, the conservative choice for a
    fail-closed budget check.
    """
    slow_budget = ag.budget.slow
    if slow_budget <= 0 or not ag.stages:
        return SlowMemoryUsage(0, slow_budget, ())
    fast_total = ag.budget.fast + ag.budget.fast_reserve
    ag = compute_lifetimes(ag)

    # Stages execute one group at a time: a chain runs as a unit, everything
    # else on its own. Slow residency is held for the whole group, so a chain
    # is measured over its whole op interval rather than stage by stage.
    groups: list[list[Stage]] = []
    index = 0
    while index < len(ag.stages):
        stage = ag.stages[index]
        span = stage.chain_len if (
            stage.chain_len >= 2 and stage.chain_id == stage.stage_id
        ) else 1
        groups.append(ag.stages[index:index + span])
        index += span

    # A chain streams every tensor it produces short of its last stage, so
    # those never reach slow memory at all.
    streamed: set[str] = set()
    for group in groups:
        for stage in group[:-1]:
            streamed.update(stage.output_tensors)

    # Slow-resident set: every tensor that crosses a stage boundary.
    slow_names: set[str] = set()
    for s in ag.stages:
        slow_names.update(s.input_tensors)
        slow_names.update(s.output_tensors)
    slow_names -= streamed

    # A stage that writes its output over its own input holds one buffer, not
    # two, so the pair counts once over the union of their lifetimes.
    aliases = row_band_aliases(ag, fast_total)
    buffers: list[tuple[int, int, int]] = []  # birth, death, size
    merged: set[str] = set()
    for dst, src in aliases.items():
        if dst not in ag.lifetimes or src not in ag.lifetimes:
            continue
        a, b = ag.lifetimes[src], ag.lifetimes[dst]
        buffers.append((
            min(a.birth_step, b.birth_step),
            max(a.death_step, b.death_step),
            max(a.size_bytes, b.size_bytes),
        ))
        merged.add(dst)
        merged.add(src)
    for n in slow_names:
        if n in ag.lifetimes and n not in merged:
            lt = ag.lifetimes[n]
            buffers.append((lt.birth_step, lt.death_step, lt.size_bytes))

    def interval_bytes(first: int, last: int) -> int:
        return sum(
            size
            for birth, death, size in buffers
            if birth <= last and death >= first
        )

    peak = 0
    overflow: list[int] = []
    for group in groups:
        steps = [i for s in group for i in s.op_indices]
        if not steps:
            continue
        group_peak = interval_bytes(min(steps), max(steps))
        if len(group) == 1 and _runs_whole(ag, group[0], fast_total):
            group_peak += untiled_stage_spill(ag, group[0], fast_total)
        peak = max(peak, group_peak)
        if group_peak > slow_budget:
            overflow.append(group[0].stage_id)
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
