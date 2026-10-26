"""C code generator: emits a deployment harness from a .tgrs plan.

Backend selection implies the target platform:
- reference  -> POSIX (float reference or s8_ref, selected by plan dtype)
- esp-nn     -> ESP-IDF (ESP-NN int8 with s8_ref fallback; float reference)
- cmsis-nn   -> Cortex-M (CMSIS-NN int8 with s8_ref fallback; float reference)
"""

from __future__ import annotations

import functools
import re

from tigris.capabilities import (
    CODEGEN_BACKENDS,
    DTypeMode,
    describe_codegen_route,
    effective_operators,
    resolve_kernel_backend,
)
from tigris.dtypes import check_dtype_signatures
from tigris.emitters.binary.defs import (
    FLAG_XIP, OP_ATTR_SUBGRAPHS, OP_C_NAMES, OP_TYPE_MAP, TENSOR_FLAG_CONSTANT,
    TENSOR_FLAG_STATE,
)
from tigris.analysis.memory import TENSOR_ALIGN, align_up
from tigris.analysis.plan_graph import fast_activation_bytes, slow_arena_required
from tigris.emitters.binary.reader import read_binary_plan
from tigris.operator_contract import OP_TYPE_BY_CODE, operators_with


BACKENDS = CODEGEN_BACKENDS
_CMSIS_TENSOR_ALIGN = 16


# Headers with TIGRIS_RUNTIME_VERSION_MAJOR declare tigris_input_alloc(), which generated code calls.
_RUNTIME_CHECK = (
    "#ifndef TIGRIS_RUNTIME_VERSION_MAJOR\n"
    '#error "these runtime headers predate tigris_input_alloc(), which this generated code '
    'calls; build with the runtime of the same TiGrIS release as the compiler"\n'
    "#endif\n"
)


def _weight_decompression_overhead(plan: dict, alignment: int) -> int:
    """Mirror the runtime's grouping at the selected allocation alignment."""
    blocks = [
        block
        for block in plan.get("weight_blocks", [])
        if block["compressed_size"] > 0
    ]
    if not blocks:
        return 0

    def block_size(block: dict) -> int:
        return align_up(block["uncompressed_size"], alignment)

    max_required = max(block_size(block) for block in blocks)
    stages = plan.get("stages", [])
    for stage_idx, stage in enumerate(stages):
        chain_len = stage["chain_len"]
        if (
            chain_len < 2
            or stage["chain_id"] != stage_idx
            or chain_len > len(stages) - stage_idx
        ):
            continue

        chain_required = 0
        for member_idx in range(stage_idx, stage_idx + chain_len):
            block = next(
                (item for item in blocks if item["stage_idx"] == member_idx),
                None,
            )
            if block is not None:
                chain_required += block_size(block)
        max_required = max(max_required, chain_required)

    return max_required


def _cmsis_weight_decompression_overhead(plan: dict) -> int:
    """Return the exact compressed-weight reserve for CMSIS static codegen."""
    return _weight_decompression_overhead(plan, _CMSIS_TENSOR_ALIGN)


# The operators the executor reserves per-op tiling state for in a chain
# (tigris_height_spatial_op in the runtime).
_HEIGHT_SPATIAL_OPS = operators_with("spatial")


def _chains(plan: dict) -> list[list[dict]]:
    """The stages of each chain, from its head."""
    stages = plan.get("stages", [])
    chains = []
    for stage_idx, stage in enumerate(stages):
        chain_len = stage["chain_len"]
        if chain_len < 2 or stage["chain_id"] != stage_idx:
            continue
        if stage_idx + chain_len > len(stages):
            raise ValueError("Plan contains a chain beyond the stage table")
        chains.append(stages[stage_idx : stage_idx + chain_len])
    return chains


def _spatial_ops(plan: dict, stage: dict) -> int:
    ops = plan.get("ops", [])
    return sum(OP_TYPE_BY_CODE.get(ops[op_idx]["op_type"]) in _HEIGHT_SPATIAL_OPS
               for op_idx in stage["ops"])


def _subgraph_depth(plan: dict) -> int:
    """How many control-flow operators run one inside another at most."""
    graphs = plan.get("subgraphs", [])
    if not graphs:
        return 0
    runs = {attr["op_index"]: attr["data"] for attr in plan.get("op_attributes", [])
            if attr["type"] == OP_ATTR_SUBGRAPHS}
    stages = plan["stages"]

    @functools.cache
    def height(graph: int) -> int:
        first, count = graphs[graph]["first_stage"], graphs[graph]["num_stages"]
        children = [int.from_bytes(runs[op][slot:slot + 2], "little")
                    for stage in stages[first:first + count] for op in stage["ops"]
                    if op in runs for slot in (0, 2)]
        return max((height(child) + 1 for child in children), default=0)

    return height(0)


def _build_limits_needed(plan: dict) -> tuple[int, int, int, int, int]:
    """The build limits the plan asks for, in tigris_config.h's order."""
    stages = plan.get("stages", [])
    chains = _chains(plan)
    return (
        plan["num_tensors"],
        max((len(stage["inputs"]) for stage in stages), default=0),
        max((len(stage["outputs"]) for stage in stages), default=0),
        max((len(chain) for chain in chains), default=0),
        max((_spatial_ops(plan, stage) for chain in chains for stage in chain), default=0),
    )


def _executor_workspace_limits(plan: dict) -> tuple[int, int, int, int, int, int]:
    """The arguments of TIGRIS_EXECUTOR_WORKSPACE_BYTES_FOR_LIMITS for the
    plan: its tensors, the most inputs and outputs of a stage, the longest
    chain, the most spatial operators in one chain and the nesting depth."""
    tensors, inputs, outputs, chain, _ = _build_limits_needed(plan)
    chain_spatial = max((sum(_spatial_ops(plan, stage) for stage in chain)
                         for chain in _chains(plan)), default=0)
    return tensors, inputs, outputs, chain, chain_spatial, _subgraph_depth(plan)


def _executor_workspace_declaration(plan: dict) -> str:
    args = ", ".join(f"{value}u" for value in _executor_workspace_limits(plan))
    return f"""\
#define TIGRIS_GENERATED_EXECUTOR_WORKSPACE_BYTES \\
    TIGRIS_EXECUTOR_WORKSPACE_BYTES_FOR_LIMITS( \\
        {args})
static uint8_t executor_workspace[TIGRIS_GENERATED_EXECUTOR_WORKSPACE_BYTES];
"""


def _build_limit_assertions(plan: dict) -> str:
    """Require the build to allow what this plan needs.

    The workspace above is sized from the plan, but the loader refuses a plan
    that asks more than the build it runs in was compiled to carry, and those
    limits are compiled into the runtime rather than this file.
    Defining them here would change only this translation unit and leave the
    loader refusing at run time, so this says what the build must set and
    fails while there is still a compiler to say it to.
    """
    tensors, inputs, outputs, chain_stages, spatial_ops = _build_limits_needed(plan)
    needed = (
        ("TIGRIS_MAX_TENSORS", tensors),
        ("TIGRIS_MAX_STAGE_INPUTS", inputs),
        ("TIGRIS_MAX_STAGE_OUTPUTS", outputs),
        ("TIGRIS_MAX_CHAIN_STAGES", chain_stages),
        ("TIGRIS_MAX_SPATIAL_OPS_PER_STAGE", spatial_ops),
    )
    lines = [
        f'_Static_assert({name} >= {value}u,\n'
        f'    "this plan needs {name} >= {value}; rebuild the runtime with '
        f'-D{name}={value}");'
        for name, value in needed
        if value > 0
    ]
    return "\n".join(lines) + "\n" if lines else ""


def _plan_dtype(plan: dict) -> DTypeMode:
    """Resolve the graph-wide runtime dtype from serialized tensors."""
    constants = {a["op_index"]: a["data"][0] for a in plan.get("op_attributes", [])
                 if a["type"] == 9 and a["data"]}
    operators = []
    for index, op in enumerate(plan.get("ops", [])):
        inputs = list(op["inputs"])
        if index in constants:
            inputs.insert(constants[index], None)
        elif OP_TYPE_BY_CODE.get(op["op_type"]) == "DynamicUpdateSlice" and op["weight_idx"] != 65535:
            inputs.insert(1, None)
        operators.append((OP_TYPE_BY_CODE.get(op["op_type"], ""), inputs, op["outputs"]))
    by_dtype, issues = check_dtype_signatures(
        [(index, tensor["dtype"], bool(tensor.get("flags", 0) & 1),
          tensor.get("quant_param_idx", 65535) != 65535)
         for index, tensor in enumerate(plan.get("tensors", []))],
        operators,
        plan.get("model_inputs", []), plan.get("model_outputs", []),
        [index for index, tensor in enumerate(plan.get("tensors", []))
         if tensor.get("flags", 0) & TENSOR_FLAG_STATE])
    if issues:
        raise ValueError("Plan dtype signature mismatch: " + "; ".join(issues))
    tensor_dtypes = set(by_dtype)
    num_quant_params = plan.get("num_quant_params", 0)

    if tensor_dtypes <= {1} and num_quant_params == 0:
        return "float32"
    if tensor_dtypes == {3} and num_quant_params > 0:
        return "int8"

    labels = {1: "float32", 3: "int8"}
    found = ", ".join(
        labels.get(dtype, f"ONNX dtype {dtype}") for dtype in sorted(tensor_dtypes)
    ) or "none"
    raise ValueError(
        "Plan cannot select one graph-wide runtime dispatcher: activation "
        f"dtypes are {found}, quantization parameter count is {num_quant_params}"
    )


def generate_c(
    plan_data: bytes,
    backend: str,
    output_format: str = "app",
    core_header: str = "tigris_codegen_core.h",
    core_name: str = "tigris_codegen",
) -> str:
    """Generate C deployment code for the given plan and backend.

    ``app`` is the self-contained example program historically produced by the
    command.  ``core`` is deliberately platform-neutral: an application owns
    flash placement, arena placement, input contents, and its entry point,
    while the generated source owns plan loading, backend preparation, and
    dispatch.  That lets the same backend output be embedded in bare-metal,
    RTOS, and benchmark applications without a target-specific codegen mode.
    """
    if backend not in BACKENDS:
        raise ValueError(f"Unknown backend: {backend!r}. Choose from: {', '.join(BACKENDS)}")
    if output_format not in {"app", "core"}:
        raise ValueError("Unknown output format: choose from: app, core")
    if output_format == "core":
        _validate_core_name(core_name)

    plan = read_binary_plan(plan_data)
    if not plan["stages"]:
        raise ValueError("Plan has no executable stages; compile it with a memory budget")
    xip = bool(plan["flags"] & FLAG_XIP)
    dtype = _plan_dtype(plan)
    is_quantized = dtype == "int8"
    kernel_backend = resolve_kernel_backend(backend, dtype)
    supported = effective_operators(kernel_backend)

    unsupported = []
    for op in plan["ops"]:
        opcode = op["op_type"]
        op_type = OP_TYPE_BY_CODE.get(opcode)
        float_op = any(plan["tensors"][i]["dtype"] == 1 for i in op["outputs"])
        op_supported = effective_operators("reference") if float_op and opcode not in {
            OP_TYPE_MAP["Dequantize"], OP_TYPE_MAP["If"], OP_TYPE_MAP["While"],
            OP_TYPE_MAP["DetectionPostProcess"]} else supported
        if op_type is None or op_type not in op_supported:
            label = op_type if op_type is not None else f"opcode {opcode}"
            unsupported.append(f"{op['name']} ({label})" if op["name"] else label)

    if unsupported:
        route = describe_codegen_route(backend, dtype)
        raise ValueError(
            f"Backend {backend!r} cannot execute this {dtype} plan. "
            f"Kernel route: {route}. Unsupported operators: "
            + ", ".join(unsupported)
        )

    header_comment = _header_comment(plan, backend, xip, dtype)

    if output_format == "core":
        return header_comment + _generate_core(
            plan, backend, is_quantized, core_header, core_name
        )
    if backend == "esp-nn":
        app = _generate_esp(plan, is_quantized)
    elif backend == "cmsis-nn":
        app = _generate_cmsis(plan, is_quantized)
    else:
        app = _generate_posix(plan, is_quantized)
    return header_comment + _with_state(app, plan)


# The one inference each example application runs.
_APP_RUN = re.compile(
    r"tigris_exec_error_t eerr = tigris_run_with_workspace_buffer\(\n"
    r"(?P<indent>[ ]+)&plan, &mem, (?P<dispatch>\w+), NULL, &stats,\n"
    r"[ ]+executor_workspace, sizeof\(executor_workspace\)\);")


def _with_state(app: str, plan: dict) -> str:
    """An example application for a plan that keeps variables: a static state
    buffer, prepared with the initial values before the run that uses it."""
    size = plan["state"]["bytes"]
    if not size:
        return app
    runs = list(_APP_RUN.finditer(app))
    if len(runs) != 1:
        raise ValueError("generated application has no single inference to give state")
    indent, dispatch = runs[0]["indent"], runs[0]["dispatch"]
    outer = indent[:-4]
    block = (
        f"static uint8_t generated_state[{size}];\n"
        f"{outer}tigris_exec_error_t eerr = tigris_state_init(\n"
        f"{indent}&plan, generated_state, sizeof(generated_state));\n"
        f"{outer}if (eerr == TIGRIS_EXEC_OK)\n"
        f"{indent}eerr = tigris_run_with_state(\n"
        f"{indent}    &plan, &mem, {dispatch}, NULL, &stats,\n"
        f"{indent}    executor_workspace, sizeof(executor_workspace),\n"
        f"{indent}    generated_state, sizeof(generated_state));")
    return app[:runs[0].start()] + block + app[runs[0].end():]


def _validate_core_name(core_name: str) -> None:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", core_name):
        raise ValueError(
            "Core name must be an ASCII C identifier "
            "([A-Za-z_][A-Za-z0-9_]*)"
        )


# Operators the runtime runs as int32 arithmetic when their first input is int32.
_INT32_OPERATORS = frozenset(OP_TYPE_MAP[name] for name in (
    "Add", "Sub", "Mul", "Equal", "Less", "LessOrEqual", "Greater", "GreaterOrEqual", "Cast"))
# Operators with typed dispatch of their own rather than a float32 kernel.
_TYPED_OPERATORS = frozenset(OP_TYPE_MAP[name] for name in (
    "Quantize", "Dequantize", "If", "While", "DetectionPostProcess"))
_CONTROL_FLOW = frozenset((OP_TYPE_MAP["If"], OP_TYPE_MAP["While"]))


def kernel_macros(plan: dict) -> list[str]:
    """The runtime kernel macros a plan needs, routed as the runtime routes it:
    int32 arithmetic, float32 kernels in a float32 plan and for float32
    operands of an int8 plan, int8 kernels otherwise. Control flow runs in the
    executor and needs none."""
    tensors = plan["tensors"]
    int8_plan = any(t["dtype"] == 3 and not t.get("flags", 0) & TENSOR_FLAG_CONSTANT
                    for t in tensors)
    macros = set()
    for op in plan["ops"]:
        code = op["op_type"]
        if code in _CONTROL_FLOW:
            continue
        inputs = op["inputs"]
        if code in _INT32_OPERATORS and inputs and tensors[inputs[0]]["dtype"] == 6:
            macros.add("TIGRIS_KERNEL_I32")
        elif not int8_plan or (code not in _TYPED_OPERATORS
                               and any(tensors[i]["dtype"] == 1 for i in inputs)):
            macros.add(f"TIGRIS_KERNEL_F32_{OP_C_NAMES[code]}")
        else:
            macros.add(f"TIGRIS_KERNEL_S8_{OP_C_NAMES[code]}")
    return sorted(macros)


def generate_kernel_config(plan_data: bytes, guard: str = "TIGRIS_PLAN_KERNELS_H") -> str:
    """A header that makes the runtime link only the kernels this plan uses."""
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", guard):
        raise ValueError("Header guard must be an ASCII C identifier")
    lines = "".join(f"#define {macro} 1\n" for macro in kernel_macros(read_binary_plan(plan_data)))
    return f"""\
/* Operator kernels this plan uses. Build the TiGrIS runtime with
 * TIGRIS_OP_CONFIG set to this file (the CMake cache variable of that name, or
 * -DTIGRIS_OP_CONFIG='"path"') so it links only these; a plan that needs any
 * other kernel then fails to load. */
#ifndef {guard}
#define {guard}

#define TIGRIS_KERNEL_DEFAULT 0
{lines}
#endif
"""


def generate_core_header(plan_data: bytes, core_name: str = "tigris_codegen") -> str:
    """Return the model-specific public header for ``codegen --format core``."""
    _validate_core_name(core_name)
    plan = read_binary_plan(plan_data)
    macro_prefix = core_name.upper()
    guard = f"{macro_prefix}_CORE_H"
    weight_reserve = _weight_decompression_overhead(plan, TENSOR_ALIGN)
    fast_arena_required = fast_activation_bytes(plan) + weight_reserve
    slow_arena_bytes = slow_arena_required(plan)
    workspace_limits = _executor_workspace_limits(plan)
    workspace_limits_args = ", ".join(f"{value}u" for value in workspace_limits)
    if fast_arena_required > 0xFFFFFFFF:
        raise ValueError("Core fast-arena requirement exceeds uint32")
    if slow_arena_bytes > 0xFFFFFFFF:
        raise ValueError("Core slow-arena requirement exceeds uint32")
    return f"""\
/* Auto-generated-code API.  This header is target-neutral. */
#ifndef {guard}
#define {guard}

#include <stdint.h>

#include "tigris.h"
#include "tigris_executor.h"
#include "tigris_iface.h"
#include "tigris_loader.h"
#include "tigris_mem.h"

/* Model-specific compile-time requirements for static embedding. The plan
 * cost model uses 32-byte allocations, conservatively covering supported
 * runtimes whose TIGRIS_TENSOR_ALIGN is at most this value. Backend-specific
 * scratch/workspace is prepared separately and is not included here. */
#define {macro_prefix}_TENSOR_CAPACITY {plan['num_tensors']}u
#define {macro_prefix}_PLAN_TENSOR_ALIGNMENT_BYTES {TENSOR_ALIGN}u
#define {macro_prefix}_PLAN_BUDGET_BYTES {plan['budget']}u
#define {macro_prefix}_WEIGHT_DECOMPRESSION_RESERVE_BYTES {weight_reserve}u
#define {macro_prefix}_CORE_FAST_ARENA_BYTES {fast_arena_required}u
/* Slow memory the plan needs; for 0 pass a NULL slow arena of size 0. */
#define {macro_prefix}_CORE_SLOW_ARENA_BYTES {slow_arena_bytes}u
#define {macro_prefix}_EXECUTOR_WORKSPACE_BYTES \\
    TIGRIS_EXECUTOR_WORKSPACE_BYTES_FOR_LIMITS({workspace_limits_args})
/* Variables the model keeps across runs; 0 when it keeps none. A plan with
 * state runs through {core_name}_run_with_state on a buffer of this size,
 * prepared once with tigris_state_init(), which also resets it. */
#define {macro_prefix}_STATE_BYTES {plan['state']['bytes']}u

typedef void (*{core_name}_input_init_fn)(
    void *data, uint32_t size_bytes, uint16_t tensor_index, void *user_ctx);

/* Load and validate the serialized plan supplied by the embedding app. */
tigris_error_t {core_name}_load_plan(
    const uint8_t *plan_data, uint32_t plan_len, tigris_plan_t *out_plan);

/* Set up runtime memory, prepare the selected backend, and place inputs with
 * tigris_input_alloc(). Call once before using reset/run. ``init_input`` may
 * be NULL. Returns TIGRIS_ERR_TENSOR_SLOTS when tensor_capacity is below the
 * plan's tensor count, and TIGRIS_ERR_MEM when an arena is too small. */
tigris_error_t {core_name}_init(
    const tigris_plan_t *plan, tigris_mem_t *mem,
    void **tensor_ptrs, uint16_t tensor_capacity,
    void *fast_arena, uint32_t fast_arena_size,
    void *slow_arena, uint32_t slow_arena_size,
    {core_name}_input_init_fn init_input, void *user_ctx);

/* Reset activations and inputs for another inference. The backend reservation
 * created by init is retained, so CMSIS-NN scratch cannot alias activations.
 * Read the outputs first: reset drops every tensor, and a plan that reads
 * its inputs in place leaves its outputs in the fast arena, where the next
 * inputs go. */
tigris_error_t {core_name}_reset(
    const tigris_plan_t *plan, tigris_mem_t *mem,
    {core_name}_input_init_fn init_input, void *user_ctx);

/* The backend dispatcher is generated from --backend. */
tigris_kernel_fn {core_name}_dispatch(void);

/* Run using the generated dispatcher and caller-owned executor workspace. */
tigris_exec_error_t {core_name}_run(
    const tigris_plan_t *plan, tigris_mem_t *mem, tigris_exec_stats_t *stats,
    tigris_executor_workspace_t *workspace);

/* Plan-sized alternative: no generic executor limits are reserved. */
tigris_exec_error_t {core_name}_run_with_workspace_buffer(
    const tigris_plan_t *plan, tigris_mem_t *mem, tigris_exec_stats_t *stats,
    void *workspace, size_t workspace_size);

/* Plan-sized run that reads and updates the state buffer; required for a
 * plan with state, valid for any plan. */
tigris_exec_error_t {core_name}_run_with_state(
    const tigris_plan_t *plan, tigris_mem_t *mem, tigris_exec_stats_t *stats,
    void *workspace, size_t workspace_size, void *state, size_t state_size);

#endif
"""


def _generate_core(
    plan: dict, backend: str, is_quantized: bool, core_header: str,
    core_name: str,
) -> str:
    """Generate embeddable backend glue with no platform or entry-point policy."""
    if backend == "cmsis-nn":
        dispatch = "tigris_dispatch_kernel_cmsis_nn" if is_quantized else "tigris_dispatch_kernel"
        kernel_include = '#include "tigris_kernels_cmsis_nn.h"' if is_quantized else '#include "tigris_kernels.h"'
        teardown = """\
    /* CMSIS-NN reserves scratch from the previous fast arena.  Release that
     * reservation before reinitializing the arena so repeated reset() calls
     * cannot make scratch alias live activations.  The first reset has no
     * reservation; deinit intentionally returns -1 and is harmless. */
    (void)tigris_cmsis_nn_deinit(mem);
""" if is_quantized else ""
        prepare = """\
    uint32_t cmsis_fast_required =
        tigris_cmsis_nn_fast_arena_required(plan);
    if (cmsis_fast_required == UINT32_MAX ||
        fast_arena_size < cmsis_fast_required)
        return TIGRIS_ERR_MEM;
    if (tigris_cmsis_nn_prepare(plan, mem) != 0)
        return TIGRIS_ERR_MEM;
""" if is_quantized else ""
    elif backend == "esp-nn":
        dispatch = "tigris_dispatch_kernel_esp_nn" if is_quantized else "tigris_dispatch_kernel"
        kernel_include = '#include "tigris_kernels_esp_nn.h"' if is_quantized else '#include "tigris_kernels.h"'
        teardown = ""
        prepare = """\
    if (tigris_esp_nn_prepare(plan, mem) != 0)
        return TIGRIS_ERR_MEM;
""" if is_quantized else ""
    else:
        dispatch = "tigris_dispatch_kernel_s8" if is_quantized else "tigris_dispatch_kernel"
        kernel_include = '#include "tigris_kernels_s8.h"' if is_quantized else '#include "tigris_kernels.h"'
        teardown = ""
        prepare = ""

    return f'''\
#include <string.h>

#include "{core_header}"
{_RUNTIME_CHECK}{kernel_include}

static tigris_error_t {core_name}_allocate_inputs(
    const tigris_plan_t *plan, tigris_mem_t *mem,
    {core_name}_input_init_fn init_input, void *user_ctx)
{{
    for (uint8_t i = 0; i < plan->header->num_model_inputs; i++) {{
        uint16_t tidx = plan->model_inputs[i];
        tigris_error_t err = tigris_input_alloc(plan, mem, tidx);
        if (err != TIGRIS_OK)
            return err;
        if (init_input)
            init_input(mem->tensor_ptrs[tidx], plan->tensors[tidx].size_bytes, tidx, user_ctx);
    }}
    return TIGRIS_OK;
}}

/* tigris_mem_init() fails only on a NULL argument or an arena too small to align. */
static tigris_error_t {core_name}_mem_error(tigris_mem_error_t err)
{{
    return err == TIGRIS_MEM_ERR_OOM ? TIGRIS_ERR_MEM : TIGRIS_ERR_NULL;
}}

tigris_error_t {core_name}_load_plan(
    const uint8_t *plan_data, uint32_t plan_len, tigris_plan_t *out_plan)
{{
    return tigris_plan_load(plan_data, plan_len, out_plan);
}}

tigris_error_t {core_name}_init(
    const tigris_plan_t *plan, tigris_mem_t *mem,
    void **tensor_ptrs, uint16_t tensor_capacity,
    void *fast_arena, uint32_t fast_arena_size,
    void *slow_arena, uint32_t slow_arena_size,
    {core_name}_input_init_fn init_input, void *user_ctx)
{{
    if (plan->header->num_tensors > tensor_capacity)
        return TIGRIS_ERR_TENSOR_SLOTS;

{teardown}
    memset(tensor_ptrs, 0, (size_t)tensor_capacity * sizeof(*tensor_ptrs));
    tigris_mem_error_t err = tigris_mem_init(
        mem, tensor_ptrs, plan->header->num_tensors,
        fast_arena, fast_arena_size, slow_arena, slow_arena_size);
    if (err != TIGRIS_MEM_OK)
        return {core_name}_mem_error(err);
{prepare}
    return {core_name}_allocate_inputs(plan, mem, init_input, user_ctx);
}}

tigris_error_t {core_name}_reset(
    const tigris_plan_t *plan, tigris_mem_t *mem,
    {core_name}_input_init_fn init_input, void *user_ctx)
{{
    if (!plan || !plan->header || !mem || !mem->tensor_ptrs ||
        !mem->fast_base || (!mem->slow_base && mem->slow_size))
        return TIGRIS_ERR_NULL;

    /* Preserve the backend's reservation. CMSIS-NN reduces fast_size in init;
     * reinitializing with that reduced size keeps scratch disjoint from fresh
     * activation allocations on every inference. */
    tigris_mem_error_t err = tigris_mem_init(
        mem, mem->tensor_ptrs, plan->header->num_tensors,
        mem->fast_base, mem->fast_size, mem->slow_base, mem->slow_size);
    if (err != TIGRIS_MEM_OK)
        return {core_name}_mem_error(err);
    return {core_name}_allocate_inputs(plan, mem, init_input, user_ctx);
}}

tigris_kernel_fn {core_name}_dispatch(void)
{{
    return {dispatch};
}}

tigris_exec_error_t {core_name}_run(
    const tigris_plan_t *plan, tigris_mem_t *mem, tigris_exec_stats_t *stats,
    tigris_executor_workspace_t *workspace)
{{
    return tigris_run_with_workspace(
        plan, mem, {dispatch}, NULL, stats, workspace);
}}

tigris_exec_error_t {core_name}_run_with_workspace_buffer(
    const tigris_plan_t *plan, tigris_mem_t *mem, tigris_exec_stats_t *stats,
    void *workspace, size_t workspace_size)
{{
    return tigris_run_with_workspace_buffer(
        plan, mem, {dispatch}, NULL, stats, workspace, workspace_size);
}}

tigris_exec_error_t {core_name}_run_with_state(
    const tigris_plan_t *plan, tigris_mem_t *mem, tigris_exec_stats_t *stats,
    void *workspace, size_t workspace_size, void *state, size_t state_size)
{{
    return tigris_run_with_state(
        plan, mem, {dispatch}, NULL, stats, workspace, workspace_size,
        state, state_size);
}}
'''


def _header_comment(
    plan: dict, backend: str, xip: bool, dtype: DTypeMode
) -> str:
    route = describe_codegen_route(backend, dtype)
    return f"""\
/*
 * Auto-generated by: tigris codegen --backend {backend}
 *
 * Model:   {plan['model_name']}
 * Ops:     {plan['num_ops']}
 * Stages:  {plan['num_stages']}
 * Weights: {plan['num_weights']}
 * Budget:  {plan['budget']} bytes
 * XIP:     {'yes' if xip else 'no'}
 * Dtype:   {dtype}
 * Kernels: {route}
 */

"""


def _generate_posix(plan: dict, is_quantized: bool) -> str:
    dispatch = "tigris_dispatch_kernel_s8" if is_quantized else "tigris_dispatch_kernel"
    kernel_include = '#include "tigris_kernels_s8.h"' if is_quantized else '#include "tigris_kernels.h"'
    budget = plan["budget"] or 65536
    slow_size = slow_arena_required(plan)
    if slow_size > 0xFFFFFFFF:
        raise ValueError("Slow arena exceeds the uint32 runtime size limit")
    workspace_declaration = (
        _build_limit_assertions(plan) + _executor_workspace_declaration(plan)
    )
    slow_buffer = "allocate_aligned(slow_size)" if slow_size else "NULL"
    slow_missing = " || !slow_buf" if slow_size else ""

    return f"""\
#define _POSIX_C_SOURCE 200112L

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "tigris.h"
#include "tigris_iface.h"
#include "tigris_loader.h"
#include "tigris_mem.h"
#include "tigris_executor.h"
{_RUNTIME_CHECK}{kernel_include}

{workspace_declaration}

static void *allocate_aligned(uint32_t size)
{{
    void *ptr = NULL;
    size_t alignment = TIGRIS_TENSOR_ALIGN;
    if (alignment < sizeof(void *)) alignment = sizeof(void *);
    if (size == 0 || posix_memalign(&ptr, alignment, size) != 0)
        return NULL;
    return ptr;
}}

static uint8_t *load_file(const char *path, uint32_t *out_len)
{{
    FILE *f = fopen(path, "rb");
    if (!f) {{ perror(path); return NULL; }}
    if (fseek(f, 0, SEEK_END) != 0) {{ fclose(f); return NULL; }}
    long sz = ftell(f);
    if (sz < 0 || (unsigned long)sz > UINT32_MAX ||
        fseek(f, 0, SEEK_SET) != 0) {{
        fclose(f);
        return NULL;
    }}
    uint8_t *buf = allocate_aligned((uint32_t)sz);
    if (!buf || fread(buf, 1, (size_t)sz, f) != (size_t)sz) {{
        free(buf);
        fclose(f);
        return NULL;
    }}
    fclose(f);
    *out_len = (uint32_t)sz;
    return buf;
}}

int main(int argc, char **argv)
{{
    if (argc < 2) {{
        fprintf(stderr, "Usage: %s <plan.tgrs>\\n", argv[0]);
        return 1;
    }}

    /* 1. Load plan from file */
    uint32_t plan_len = 0;
    uint8_t *plan_buf = load_file(argv[1], &plan_len);
    if (!plan_buf) return 1;

    tigris_plan_t plan;
    tigris_error_t err = tigris_plan_load(plan_buf, plan_len, &plan);
    if (err != TIGRIS_OK) {{
        fprintf(stderr, "Plan load failed: %s\\n", tigris_error_str(err));
        free(plan_buf);
        return 1;
    }}

    printf("Model:   %s\\n", tigris_model_name(&plan));
    printf("Ops:     %u  Stages: %u  Budget: %u bytes\\n",
           plan.header->num_ops, plan.header->num_stages, plan.header->budget);

    /* 2. Allocate buffers */
    uint32_t fast_size = tigris_fast_arena_required(&plan);
    if (fast_size == 0) fast_size = {budget};
    if (fast_size == UINT32_MAX) {{
        fprintf(stderr, "Invalid core fast-arena requirement\\n");
        free(plan_buf);
        return 1;
    }}

    uint32_t slow_size = {slow_size}u;

    void *fast_buf = allocate_aligned(fast_size);
    void *slow_buf = {slow_buffer};
    uint16_t num_t = plan.header->num_tensors;
    void **tensor_ptrs = calloc(num_t, sizeof(void *));
    if (!fast_buf{slow_missing} || !tensor_ptrs) {{
        fprintf(stderr, "Allocation failed\\n");
        free(tensor_ptrs); free(slow_buf); free(fast_buf); free(plan_buf);
        return 1;
    }}

    /* 3. Init memory manager */
    tigris_mem_t mem;
    tigris_mem_error_t merr = tigris_mem_init(
        &mem, tensor_ptrs, num_t, fast_buf, fast_size, slow_buf, slow_size);
    if (merr != TIGRIS_MEM_OK) {{
        fprintf(stderr, "Memory init failed: %s\\n", tigris_mem_error_str(merr));
        free(tensor_ptrs); free(slow_buf); free(fast_buf); free(plan_buf);
        return 1;
    }}

    /* 4. Allocate and zero-fill model inputs */
    for (uint8_t i = 0; i < plan.header->num_model_inputs; i++) {{
        uint16_t tidx = plan.model_inputs[i];
        tigris_error_t ierr = tigris_input_alloc(&plan, &mem, tidx);
        if (ierr != TIGRIS_OK) {{
            fprintf(stderr, "Input allocation failed for tensor %u: %s\\n",
                    tidx, tigris_error_str(ierr));
            free(tensor_ptrs); free(slow_buf); free(fast_buf); free(plan_buf);
            return 1;
        }}
        memset(mem.tensor_ptrs[tidx], 0, plan.tensors[tidx].size_bytes);
    }}

    /* 5. Run inference */
    tigris_exec_stats_t stats;
    tigris_exec_error_t eerr = tigris_run_with_workspace_buffer(
        &plan, &mem, {dispatch}, NULL, &stats,
        executor_workspace, sizeof(executor_workspace));
    if (eerr != TIGRIS_EXEC_OK) {{
        fprintf(stderr, "Inference failed: %s\\n", tigris_exec_error_str(eerr));
        free(tensor_ptrs); free(slow_buf); free(fast_buf); free(plan_buf);
        return 1;
    }}

    printf("OK  normal=%u tiled=%u chain=%u\\n",
           stats.stages_normal, stats.stages_tiled, stats.stages_chain);

    /* 6. Print outputs in the dtype the model declares, which the runtime
     *    converts to from whatever the plan executes on. */
    for (uint8_t i = 0; i < plan.header->num_model_outputs; i++) {{
        uint16_t tidx = plan.model_outputs[i];
        const tigris_tensor_t *t = &plan.tensors[tidx];
        uint32_t iface_bytes = tigris_iface_bytes(&plan, tidx);
        void *staging;
        if (iface_bytes == 0u) continue;
        staging = malloc(iface_bytes);
        if (!staging) continue;
        if (tigris_output_read(&plan, &mem, tidx, staging, iface_bytes) != TIGRIS_OK) {{
            free(staging);
            continue;
        }}
        printf("Output '%s': %u bytes\\n", tigris_tensor_name(&plan, t), iface_bytes);
        if (t->dtype == 6u) {{
            uint32_t n = t->size_bytes / (uint32_t)sizeof(int32_t);
            for (uint32_t j = 0; j < n && j < 10u; j++) {{
                long long value = t->iface_dtype == 7u
                    ? (long long)((const int64_t *)staging)[j]
                    : (long long)((const int32_t *)staging)[j];
                printf("  [%u] %lld\\n", j, value);
            }}
        }} else if (t->iface_dtype == 0u && t->dtype == 3u) {{
            const int8_t *out = (const int8_t *)staging;
            uint32_t n = iface_bytes;
            uint32_t show = n < 10 ? n : 10;
            for (uint32_t j = 0; j < show; j++)
                printf("  [%u] % d\\n", j, (int)out[j]);
            if (n > show) printf("  ... (%u more)\\n", n - show);
        }} else if (t->iface_dtype == 2u || t->dtype == 9u) {{
            const uint8_t *out = (const uint8_t *)staging;
            for (uint32_t j = 0; j < iface_bytes && j < 10u; j++)
                printf("  [%u] %u\\n", j, (unsigned)out[j]);
        }} else {{
            const float *out = (const float *)staging;
            uint32_t n = iface_bytes / (uint32_t)sizeof(float);
            uint32_t show = n < 10 ? n : 10;
            for (uint32_t j = 0; j < show; j++)
                printf("  [%u] %.6f\\n", j, out[j]);
            if (n > show) printf("  ... (%u more)\\n", n - show);
        }}
        free(staging);
    }}

    free(tensor_ptrs); free(slow_buf); free(fast_buf); free(plan_buf);
    return 0;
}}
"""


def _generate_esp(plan: dict, is_quantized: bool) -> str:
    budget = plan["budget"] or 65536
    slow_size = slow_arena_required(plan)
    if slow_size > 0xFFFFFFFF:
        raise ValueError("Slow arena exceeds the uint32 runtime size limit")
    workspace_declaration = (
        _build_limit_assertions(plan) + _executor_workspace_declaration(plan)
    )
    psram_buffer = "heap_caps_malloc(slow_size, MALLOC_CAP_SPIRAM)" if slow_size else "NULL"
    sram_buffer = ("heap_caps_malloc(slow_size,\n        MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT)"
                   if slow_size else "NULL")
    slow_missing = " || !slow_buf" if slow_size else ""

    if is_quantized:
        dispatch = "tigris_dispatch_kernel_esp_nn"
        kernel_includes = """\
#include "tigris_kernels_esp_nn.h"
#include "tigris_kernels_s8.h"
"""
        prepare_block = """\

    /* ESP-NN scratch buffer setup */
    if (tigris_esp_nn_prepare(&plan, &mem) != 0) {
        ESP_LOGE(TAG, "ESP-NN preparation failed");
        goto cleanup;
    }
"""
    else:
        dispatch = "tigris_dispatch_kernel"
        kernel_includes = '#include "tigris_kernels.h"'
        prepare_block = ""

    return f"""\
#include <stdio.h>
#include <string.h>
#include <inttypes.h>

#include "esp_partition.h"
#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "esp_log.h"

#include "tigris.h"
#include "tigris_iface.h"
#include "tigris_loader.h"
#include "tigris_mem.h"
#include "tigris_executor.h"
{_RUNTIME_CHECK}{kernel_includes}

static const char *TAG = "tigris";
{workspace_declaration}

void app_main(void)
{{
    /* 1. Memory-map the plan partition */
    const esp_partition_t *part = esp_partition_find_first(
        ESP_PARTITION_TYPE_DATA, 0x40, "plan");
    if (!part) {{
        ESP_LOGE(TAG, "partition 'plan' not found");
        return;
    }}

    const void *mapped_ptr = NULL;
    esp_partition_mmap_handle_t mmap_handle;
    esp_err_t err = esp_partition_mmap(
        part, 0, part->size, ESP_PARTITION_MMAP_DATA, &mapped_ptr, &mmap_handle);
    if (err != ESP_OK) {{
        ESP_LOGE(TAG, "mmap failed: %s", esp_err_to_name(err));
        return;
    }}

    /* 2. Load the plan (zero-copy from flash) */
    if (part->size < sizeof(tigris_file_header_t)) {{
        ESP_LOGE(TAG, "plan partition is smaller than the file header");
        esp_partition_munmap(mmap_handle);
        return;
    }}
    const tigris_file_header_t *raw_hdr = (const tigris_file_header_t *)mapped_ptr;
    uint32_t plan_size = raw_hdr->file_size;
    if ((size_t)plan_size > part->size) {{
        ESP_LOGE(TAG, "plan file size exceeds the mapped partition");
        esp_partition_munmap(mmap_handle);
        return;
    }}

    tigris_plan_t plan;
    tigris_error_t perr = tigris_plan_load(
        (const uint8_t *)mapped_ptr, plan_size, &plan);
    if (perr != TIGRIS_OK) {{
        ESP_LOGE(TAG, "plan load failed: %s", tigris_error_str(perr));
        esp_partition_munmap(mmap_handle);
        return;
    }}

    printf("Model: %s  Ops: %u  Stages: %u\\n",
           tigris_model_name(&plan), plan.header->num_ops, plan.header->num_stages);

    /* 3. Allocate buffers */
    uint32_t fast_size = tigris_fast_arena_required(&plan);
    if (fast_size == 0) fast_size = {budget};
    if (fast_size == UINT32_MAX) {{
        ESP_LOGE(TAG, "invalid core fast-arena requirement");
        esp_partition_munmap(mmap_handle);
        return;
    }}

    uint32_t slow_size = {slow_size}u;

    void *fast_buf = heap_caps_malloc(fast_size,
        MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
#if CONFIG_SPIRAM
    void *slow_buf = {psram_buffer};
    uint16_t num_t = plan.header->num_tensors;
    void **tensor_ptrs = heap_caps_calloc(num_t, sizeof(void *), MALLOC_CAP_SPIRAM);
#else
    void *slow_buf = {sram_buffer};
    uint16_t num_t = plan.header->num_tensors;
    void **tensor_ptrs = heap_caps_calloc(num_t, sizeof(void *),
        MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
#endif
    if (!fast_buf{slow_missing} || !tensor_ptrs) {{
        ESP_LOGE(TAG, "allocation failed");
        goto cleanup;
    }}

    printf("Fast: %lu  Slow: %lu\\n",
           (unsigned long)fast_size, (unsigned long)slow_size);

    /* 4. Init memory manager */
    tigris_mem_t mem;
    tigris_mem_error_t merr = tigris_mem_init(
        &mem, tensor_ptrs, num_t, fast_buf, fast_size, slow_buf, slow_size);
    if (merr != TIGRIS_MEM_OK) {{
        ESP_LOGE(TAG, "memory init failed: %s", tigris_mem_error_str(merr));
        goto cleanup;
    }}
{prepare_block}
    /* 5. Allocate and zero-fill model inputs */
    for (uint8_t i = 0; i < plan.header->num_model_inputs; i++) {{
        uint16_t tidx = plan.model_inputs[i];
        tigris_error_t ierr = tigris_input_alloc(&plan, &mem, tidx);
        if (ierr != TIGRIS_OK) {{
            ESP_LOGE(TAG, "input allocation failed for tensor %u: %s",
                     tidx, tigris_error_str(ierr));
            goto cleanup;
        }}
        memset(mem.tensor_ptrs[tidx], 0, plan.tensors[tidx].size_bytes);
    }}

    /* 6. Run inference */
    int64_t t0 = esp_timer_get_time();
    tigris_exec_stats_t stats;
    tigris_exec_error_t eerr = tigris_run_with_workspace_buffer(
        &plan, &mem, {dispatch}, NULL, &stats,
        executor_workspace, sizeof(executor_workspace));
    int64_t t1 = esp_timer_get_time();

    if (eerr != TIGRIS_EXEC_OK) {{
        ESP_LOGE(TAG, "inference failed: %s", tigris_exec_error_str(eerr));
        goto cleanup;
    }}

    float elapsed_ms = (float)(t1 - t0) / 1000.0f;
    printf("Inference OK  %.1f ms  normal=%u tiled=%u chain=%u\\n",
           elapsed_ms, stats.stages_normal, stats.stages_tiled, stats.stages_chain);

    /* 7. Print outputs */
    for (uint8_t i = 0; i < plan.header->num_model_outputs; i++) {{
        uint16_t tidx = plan.model_outputs[i];
        const tigris_tensor_t *t = &plan.tensors[tidx];
        void *ptr = mem.tensor_ptrs[tidx];
        if (!ptr) continue;

        printf("Output '%s': %u bytes\\n", tigris_tensor_name(&plan, t), t->size_bytes);
        if (t->dtype == 9u) {{
            const uint8_t *values = (const uint8_t *)ptr;
            for (uint32_t j = 0; j < t->size_bytes && j < 10u; j++)
                printf("  [%u] %u\\n", j, (unsigned)values[j]);
            continue;
        }}
        if (t->dtype == 6u) {{
            const int32_t *indices = (const int32_t *)ptr;
            uint32_t count = t->size_bytes / (uint32_t)sizeof(int32_t);
            for (uint32_t j = 0; j < count && j < 10u; j++)
                printf("  [%u] %ld\\n", j, (long)indices[j]);
            continue;
        }}
        uint32_t n = t->size_bytes / (t->dtype == 1u ? sizeof(float) : 1u);
        uint32_t show = n < 10 ? n : 10;
        for (uint32_t j = 0; j < show; j++) {{
            if (t->dtype == 1u)
                printf("  [%u] %.6f\\n", j, ((const float *)ptr)[j]);
            else
                printf("  [%u] % d\\n", j, (int)((const int8_t *)ptr)[j]);
        }}
        if (n > show) printf("  ... (%u more)\\n", n - show);
    }}

cleanup:
    heap_caps_free(tensor_ptrs);
    heap_caps_free(slow_buf);
    heap_caps_free(fast_buf);
    esp_partition_munmap(mmap_handle);
}}
"""


def _generate_cmsis(plan: dict, is_quantized: bool) -> str:
    dispatch = "tigris_dispatch_kernel_cmsis_nn" if is_quantized else "tigris_dispatch_kernel"
    kernel_include = '#include "tigris_kernels_cmsis_nn.h"' if is_quantized else '#include "tigris_kernels.h"'
    weight_overhead = _cmsis_weight_decompression_overhead(plan)
    core_fast_arena_size = align_up(
        (fast_activation_bytes(plan) or 65536) + weight_overhead, 16)
    if core_fast_arena_size > 0xFFFFFFFF:
        raise ValueError(
            "CMSIS-NN static fast arena exceeds the uint32 runtime size limit"
        )
    slow_arena_size = slow_arena_required(plan)
    workspace_declaration = (
        _build_limit_assertions(plan) + _executor_workspace_declaration(plan)
    )
    if slow_arena_size > 0xFFFFFFFF:
        raise ValueError(
            "CMSIS-NN static slow arena exceeds the uint32 runtime size limit"
        )
    slow_declaration = (
        f"static uint8_t slow_arena[{slow_arena_size}] __attribute__((aligned(16)));\n"
        if slow_arena_size else "")
    slow_arguments = "slow_arena, sizeof(slow_arena)" if slow_arena_size else "NULL, 0u"

    if is_quantized:
        scratch_declaration = f"""\
/* Override at build time for larger plans. The runtime query below validates
 * this allowance against the linked CMSIS-NN library before initialization. */
#ifndef TIGRIS_CMSIS_NN_SCRATCH_BYTES
#define TIGRIS_CMSIS_NN_SCRATCH_BYTES 4096u
#endif
#if TIGRIS_CMSIS_NN_SCRATCH_BYTES > UINT32_MAX - {core_fast_arena_size}u
#error "CMSIS-NN static fast arena exceeds uint32"
#endif
"""
        fast_arena_expr = (
            f"{core_fast_arena_size}u + TIGRIS_CMSIS_NN_SCRATCH_BYTES"
        )
        prepare_block = """\

    uint32_t cmsis_fast_required =
        tigris_cmsis_nn_fast_arena_required(&plan);
    if (cmsis_fast_required == UINT32_MAX) {
        printf("CMSIS-NN fast arena requirement is invalid or unrepresentable.\\n");
        return 1;
    }
    if (cmsis_fast_required > sizeof(fast_arena)) {
        printf("CMSIS-NN fast arena needs %lu bytes; generated capacity is %lu. "
               "Increase TIGRIS_CMSIS_NN_SCRATCH_BYTES.\\n",
               (unsigned long)cmsis_fast_required,
               (unsigned long)sizeof(fast_arena));
        return 1;
    }
    if (tigris_cmsis_nn_prepare(&plan, &mem) != 0) {
        printf("CMSIS-NN preparation failed\\n");
        return 1;
    }
"""
    else:
        scratch_declaration = ""
        fast_arena_expr = f"{core_fast_arena_size}u"
        prepare_block = ""

    return f"""\
/*
 * Cortex-M deployment harness.
 *
 * The .tgrs plan must be linked into flash. Use your linker script or
 * objcopy to place it at a known address, then reference it here:
 *
 *   extern const uint8_t _binary_model_tgrs_start[];
 *   extern const uint8_t _binary_model_tgrs_end[];
 */

#include <stdio.h>
#include <string.h>

#include "tigris.h"
#include "tigris_iface.h"
#include "tigris_loader.h"
#include "tigris_mem.h"
#include "tigris_executor.h"
{_RUNTIME_CHECK}{kernel_include}

{scratch_declaration}

/* Plan binary linked into flash: symbol provided by linker */
extern const uint8_t _binary_model_tgrs_start[];
extern const uint8_t _binary_model_tgrs_end[];

/* Static buffers. Fast memory preserves the full core arena below CMSIS scratch. */
static uint8_t fast_arena[{fast_arena_expr}] __attribute__((aligned(16)));
{slow_declaration}static void *tensor_ptrs[{plan['num_tensors']}];
{workspace_declaration}

int main(void)
{{
    /* 1. Load plan from flash */
    uint32_t plan_len = (uint32_t)(_binary_model_tgrs_end - _binary_model_tgrs_start);

    tigris_plan_t plan;
    tigris_error_t err = tigris_plan_load(_binary_model_tgrs_start, plan_len, &plan);
    if (err != TIGRIS_OK) {{
        printf("Plan load failed: %s\\n", tigris_error_str(err));
        return 1;
    }}

    printf("Model: %s  Ops: %u  Stages: %u\\n",
           tigris_model_name(&plan), plan.header->num_ops, plan.header->num_stages);

    /* 2. Validate the generated static compressed-weight reservation. */
    uint32_t weight_overhead = tigris_weight_decompression_overhead(&plan);
    if (weight_overhead > {weight_overhead}u) {{
        printf("Compressed-weight arena requirement exceeds generated reserve\\n");
        return 1;
    }}

    tigris_mem_t mem;
    memset(tensor_ptrs, 0, sizeof(tensor_ptrs));
    tigris_mem_error_t merr = tigris_mem_init(
        &mem, tensor_ptrs, {plan['num_tensors']},
        fast_arena, sizeof(fast_arena), {slow_arguments});
    if (merr != TIGRIS_MEM_OK) {{
        printf("Memory init failed: %s\\n", tigris_mem_error_str(merr));
        return 1;
    }}
{prepare_block}

    /* 3. Allocate and zero-fill model inputs */
    for (uint8_t i = 0; i < plan.header->num_model_inputs; i++) {{
        uint16_t tidx = plan.model_inputs[i];
        tigris_error_t ierr = tigris_input_alloc(&plan, &mem, tidx);
        if (ierr != TIGRIS_OK) {{
            printf("Input allocation failed for tensor %u: %s\\n",
                   tidx, tigris_error_str(ierr));
            return 1;
        }}
        memset(mem.tensor_ptrs[tidx], 0, plan.tensors[tidx].size_bytes);
    }}

    /* 4. Run inference */
    tigris_exec_stats_t stats;
    tigris_exec_error_t eerr = tigris_run_with_workspace_buffer(
        &plan, &mem, {dispatch}, NULL, &stats,
        executor_workspace, sizeof(executor_workspace));
    if (eerr != TIGRIS_EXEC_OK) {{
        printf("Inference failed: %s\\n", tigris_exec_error_str(eerr));
        return 1;
    }}

    printf("OK  normal=%u tiled=%u chain=%u\\n",
           stats.stages_normal, stats.stages_tiled, stats.stages_chain);

    return 0;
}}
"""
