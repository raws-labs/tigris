"""C code generator — emits a deployment harness from a .tgrs plan.

Backend selection implies the target platform:
- reference  → POSIX (float reference or s8_ref, selected by plan dtype)
- esp-nn     → ESP-IDF (ESP-NN int8 with s8_ref fallback; float reference)
- cmsis-nn   → Cortex-M (CMSIS-NN int8 with s8_ref fallback; float reference)
"""

from __future__ import annotations

import re

from tigris.capabilities import (
    CODEGEN_BACKENDS,
    DTypeMode,
    OP_TYPE_BY_CODE,
    describe_codegen_route,
    effective_operators,
    resolve_kernel_backend,
)
from tigris.emitters.binary.defs import FLAG_XIP
from tigris.emitters.binary.reader import read_binary_plan


BACKENDS = CODEGEN_BACKENDS
_CMSIS_TENSOR_ALIGN = 16
_PLAN_TENSOR_ALIGN = 32


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


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
        return _align_up(block["uncompressed_size"], alignment)

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


def _plan_dtype(plan: dict) -> DTypeMode:
    """Resolve the graph-wide runtime dtype from serialized tensors."""
    tensor_dtypes = {tensor["dtype"] for tensor in plan.get("tensors", [])}
    num_quant_params = plan.get("num_quant_params", 0)

    if tensor_dtypes == {1} and num_quant_params == 0:
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
        if op_type is None or op_type not in supported:
            label = op_type if op_type is not None else f"opcode {opcode}"
            unsupported.append(f"{op['name']} ({label})")

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
        return header_comment + _generate_esp(plan, is_quantized)
    elif backend == "cmsis-nn":
        return header_comment + _generate_cmsis(plan, is_quantized)
    else:
        return header_comment + _generate_posix(plan, is_quantized)


def _validate_core_name(core_name: str) -> None:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", core_name):
        raise ValueError(
            "Core name must be an ASCII C identifier "
            "([A-Za-z_][A-Za-z0-9_]*)"
        )


def generate_core_header(plan_data: bytes, core_name: str = "tigris_codegen") -> str:
    """Return the model-specific public header for ``codegen --format core``."""
    _validate_core_name(core_name)
    plan = read_binary_plan(plan_data)
    macro_prefix = core_name.upper()
    guard = f"{macro_prefix}_CORE_H"
    weight_reserve = _weight_decompression_overhead(plan, _PLAN_TENSOR_ALIGN)
    fast_arena_required = plan["budget"] + weight_reserve
    if fast_arena_required > 0xFFFFFFFF:
        raise ValueError("Core fast-arena requirement exceeds uint32")
    return f"""\
/* Auto-generated-code API.  This header is target-neutral. */
#ifndef {guard}
#define {guard}

#include <stdint.h>

#include "tigris.h"
#include "tigris_executor.h"
#include "tigris_loader.h"
#include "tigris_mem.h"

/* Model-specific compile-time requirements for static embedding. The plan
 * cost model uses 32-byte allocations, conservatively covering supported
 * runtimes whose TIGRIS_TENSOR_ALIGN is at most this value. Backend-specific
 * scratch/workspace is prepared separately and is not included here. */
#define {macro_prefix}_TENSOR_CAPACITY {plan['num_tensors']}u
#define {macro_prefix}_PLAN_TENSOR_ALIGNMENT_BYTES {_PLAN_TENSOR_ALIGN}u
#define {macro_prefix}_PLAN_BUDGET_BYTES {plan['budget']}u
#define {macro_prefix}_WEIGHT_DECOMPRESSION_RESERVE_BYTES {weight_reserve}u
#define {macro_prefix}_CORE_FAST_ARENA_BYTES {fast_arena_required}u

typedef void (*{core_name}_input_init_fn)(
    void *data, uint32_t size_bytes, uint16_t tensor_index, void *user_ctx);

/* Load and validate the serialized plan supplied by the embedding app. */
tigris_error_t {core_name}_load_plan(
    const uint8_t *plan_data, uint32_t plan_len, tigris_plan_t *out_plan);

/* Set up runtime memory, prepare the selected backend, and allocate inputs.
 * Call once before using reset/run. ``init_input`` may be NULL. */
tigris_mem_error_t {core_name}_init(
    const tigris_plan_t *plan, tigris_mem_t *mem,
    void **tensor_ptrs, uint16_t tensor_capacity,
    void *fast_arena, uint32_t fast_arena_size,
    void *slow_arena, uint32_t slow_arena_size,
    {core_name}_input_init_fn init_input, void *user_ctx);

/* Reset activations and inputs for another inference. The backend reservation
 * created by init is retained, so CMSIS-NN scratch cannot alias activations. */
tigris_mem_error_t {core_name}_reset(
    const tigris_plan_t *plan, tigris_mem_t *mem,
    {core_name}_input_init_fn init_input, void *user_ctx);

/* The backend dispatcher is generated from --backend. */
tigris_kernel_fn {core_name}_dispatch(void);

/* Run using the generated dispatcher. */
tigris_exec_error_t {core_name}_run(
    const tigris_plan_t *plan, tigris_mem_t *mem, tigris_exec_stats_t *stats);

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
        return TIGRIS_MEM_ERR_OOM;
    if (tigris_cmsis_nn_prepare(plan, mem) != 0)
        return TIGRIS_MEM_ERR_OOM;
""" if is_quantized else ""
    elif backend == "esp-nn":
        dispatch = "tigris_dispatch_kernel_esp_nn" if is_quantized else "tigris_dispatch_kernel"
        kernel_include = '#include "tigris_kernels_esp_nn.h"' if is_quantized else '#include "tigris_kernels.h"'
        teardown = ""
        prepare = """\
    if (tigris_esp_nn_prepare(plan, mem) != 0)
        return TIGRIS_MEM_ERR_OOM;
""" if is_quantized else ""
    else:
        dispatch = "tigris_dispatch_kernel_s8" if is_quantized else "tigris_dispatch_kernel"
        kernel_include = '#include "tigris_kernels_s8.h"' if is_quantized else '#include "tigris_kernels.h"'
        teardown = ""
        prepare = ""

    return f'''\
#include <string.h>

#include "{core_header}"
{kernel_include}

static tigris_mem_error_t {core_name}_allocate_inputs(
    const tigris_plan_t *plan, tigris_mem_t *mem,
    {core_name}_input_init_fn init_input, void *user_ctx)
{{
    for (uint8_t i = 0; i < plan->header->num_model_inputs; i++) {{
        uint16_t tidx = plan->model_inputs[i];
        uint32_t size = plan->tensors[tidx].size_bytes;
        tigris_mem_error_t err = tigris_mem_alloc_slow(mem, tidx, size);
        if (err != TIGRIS_MEM_OK)
            return err;
        if (init_input)
            init_input(mem->tensor_ptrs[tidx], size, tidx, user_ctx);
    }}
    return TIGRIS_MEM_OK;
}}

tigris_error_t {core_name}_load_plan(
    const uint8_t *plan_data, uint32_t plan_len, tigris_plan_t *out_plan)
{{
    return tigris_plan_load(plan_data, plan_len, out_plan);
}}

tigris_mem_error_t {core_name}_init(
    const tigris_plan_t *plan, tigris_mem_t *mem,
    void **tensor_ptrs, uint16_t tensor_capacity,
    void *fast_arena, uint32_t fast_arena_size,
    void *slow_arena, uint32_t slow_arena_size,
    {core_name}_input_init_fn init_input, void *user_ctx)
{{
    if (plan->header->num_tensors > tensor_capacity)
        return TIGRIS_MEM_ERR_BAD_INDEX;

{teardown}
    memset(tensor_ptrs, 0, (size_t)tensor_capacity * sizeof(*tensor_ptrs));
    tigris_mem_error_t err = tigris_mem_init(
        mem, tensor_ptrs, plan->header->num_tensors,
        fast_arena, fast_arena_size, slow_arena, slow_arena_size);
    if (err != TIGRIS_MEM_OK)
        return err;
{prepare}
    return {core_name}_allocate_inputs(plan, mem, init_input, user_ctx);
}}

tigris_mem_error_t {core_name}_reset(
    const tigris_plan_t *plan, tigris_mem_t *mem,
    {core_name}_input_init_fn init_input, void *user_ctx)
{{
    if (!plan || !plan->header || !mem || !mem->tensor_ptrs ||
        !mem->fast_base || !mem->slow_base)
        return TIGRIS_MEM_ERR_NULL;

    /* Preserve the backend's reservation. CMSIS-NN reduces fast_size in init;
     * reinitializing with that reduced size keeps scratch disjoint from fresh
     * activation allocations on every inference. */
    tigris_mem_error_t err = tigris_mem_init(
        mem, mem->tensor_ptrs, plan->header->num_tensors,
        mem->fast_base, mem->fast_size, mem->slow_base, mem->slow_size);
    if (err != TIGRIS_MEM_OK)
        return err;
    return {core_name}_allocate_inputs(plan, mem, init_input, user_ctx);
}}

tigris_kernel_fn {core_name}_dispatch(void)
{{
    return {dispatch};
}}

tigris_exec_error_t {core_name}_run(
    const tigris_plan_t *plan, tigris_mem_t *mem, tigris_exec_stats_t *stats)
{{
    return tigris_run(plan, mem, {dispatch}, NULL, stats);
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

    return f"""\
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "tigris.h"
#include "tigris_loader.h"
#include "tigris_mem.h"
#include "tigris_executor.h"
{kernel_include}

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
    uint8_t *buf = malloc((size_t)sz);
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

    if (plan.header->peak > UINT32_MAX / 4u) {{
        fprintf(stderr, "Slow-memory arena requirement exceeds uint32\\n");
        free(plan_buf);
        return 1;
    }}
    uint32_t slow_size = plan.header->peak * 4u;
    if (slow_size < 256 * 1024) slow_size = 256 * 1024;

    void *fast_buf = malloc(fast_size);
    void *slow_buf = malloc(slow_size);
    uint16_t num_t = plan.header->num_tensors;
    void **tensor_ptrs = calloc(num_t, sizeof(void *));
    if (!fast_buf || !slow_buf || !tensor_ptrs) {{
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
        merr = tigris_mem_alloc_slow(&mem, tidx, plan.tensors[tidx].size_bytes);
        if (merr != TIGRIS_MEM_OK) {{
            fprintf(stderr, "Input allocation failed for tensor %u: %s\\n",
                    tidx, tigris_mem_error_str(merr));
            free(tensor_ptrs); free(slow_buf); free(fast_buf); free(plan_buf);
            return 1;
        }}
        memset(mem.tensor_ptrs[tidx], 0, plan.tensors[tidx].size_bytes);
    }}

    /* 5. Run inference */
    tigris_exec_stats_t stats;
    tigris_exec_error_t eerr = tigris_run(&plan, &mem, {dispatch}, NULL, &stats);
    if (eerr != TIGRIS_EXEC_OK) {{
        fprintf(stderr, "Inference failed: %s\\n", tigris_exec_error_str(eerr));
        free(tensor_ptrs); free(slow_buf); free(fast_buf); free(plan_buf);
        return 1;
    }}

    printf("OK  normal=%u tiled=%u chain=%u\\n",
           stats.stages_normal, stats.stages_tiled, stats.stages_chain);

    /* 6. Print outputs */
    for (uint8_t i = 0; i < plan.header->num_model_outputs; i++) {{
        uint16_t tidx = plan.model_outputs[i];
        const tigris_tensor_t *t = &plan.tensors[tidx];
        void *ptr = mem.tensor_ptrs[tidx];
        if (!ptr) continue;
        printf("Output '%s': %u bytes\\n", tigris_tensor_name(&plan, t), t->size_bytes);
        {"int8_t" if is_quantized else "float"} *out = ({"int8_t" if is_quantized else "float"} *)ptr;
        uint32_t n = t->size_bytes / {"1" if is_quantized else "sizeof(float)"};
        uint32_t show = n < 10 ? n : 10;
        for (uint32_t j = 0; j < show; j++)
            printf("  [%u] {"% d" if is_quantized else "%.6f"}\\n", j, {"(int)" if is_quantized else ""}out[j]);
        if (n > show) printf("  ... (%u more)\\n", n - show);
    }}

    free(tensor_ptrs); free(slow_buf); free(fast_buf); free(plan_buf);
    return 0;
}}
"""


def _generate_esp(plan: dict, is_quantized: bool) -> str:
    budget = plan["budget"] or 65536

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
#include "tigris_loader.h"
#include "tigris_mem.h"
#include "tigris_executor.h"
{kernel_includes}

static const char *TAG = "tigris";

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
    uint32_t fast_size = plan.header->budget;
    if (fast_size == 0) fast_size = {budget};
    uint32_t weight_overhead = tigris_weight_decompression_overhead(&plan);
    if (weight_overhead == UINT32_MAX || fast_size > UINT32_MAX - weight_overhead) {{
        ESP_LOGE(TAG, "invalid compressed-weight arena requirement");
        esp_partition_munmap(mmap_handle);
        return;
    }}
    fast_size += weight_overhead;

#if CONFIG_SPIRAM
    uint32_t slow_size = heap_caps_get_largest_free_block(MALLOC_CAP_SPIRAM);
    if (slow_size > 64 * 1024) slow_size -= 16 * 1024;
#else
    uint32_t slow_size = heap_caps_get_largest_free_block(
        MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    if (slow_size > 32 * 1024) slow_size -= 16 * 1024;
#endif
    if (slow_size < 64 * 1024) slow_size = 64 * 1024;

    void *fast_buf = heap_caps_malloc(fast_size,
        MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
#if CONFIG_SPIRAM
    void *slow_buf = heap_caps_malloc(slow_size, MALLOC_CAP_SPIRAM);
    uint16_t num_t = plan.header->num_tensors;
    void **tensor_ptrs = heap_caps_calloc(num_t, sizeof(void *), MALLOC_CAP_SPIRAM);
#else
    void *slow_buf = heap_caps_malloc(slow_size,
        MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    uint16_t num_t = plan.header->num_tensors;
    void **tensor_ptrs = heap_caps_calloc(num_t, sizeof(void *),
        MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
#endif
    if (!fast_buf || !slow_buf || !tensor_ptrs) {{
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
        merr = tigris_mem_alloc_slow(&mem, tidx, plan.tensors[tidx].size_bytes);
        if (merr != TIGRIS_MEM_OK) {{
            ESP_LOGE(TAG, "input allocation failed for tensor %u: %s",
                     tidx, tigris_mem_error_str(merr));
            goto cleanup;
        }}
        memset(mem.tensor_ptrs[tidx], 0, plan.tensors[tidx].size_bytes);
    }}

    /* 6. Run inference */
    int64_t t0 = esp_timer_get_time();
    tigris_exec_stats_t stats;
    tigris_exec_error_t eerr = tigris_run(&plan, &mem, {dispatch}, NULL, &stats);
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
        {"int8_t" if is_quantized else "float"} *out = ({"int8_t" if is_quantized else "float"} *)ptr;
        uint32_t n = t->size_bytes / {"1" if is_quantized else "sizeof(float)"};
        uint32_t show = n < 10 ? n : 10;
        for (uint32_t j = 0; j < show; j++)
            printf("  [%u] {"% d" if is_quantized else "%.6f"}\\n", j, {"(int)" if is_quantized else ""}out[j]);
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
    budget = plan["budget"] or 65536
    weight_overhead = _cmsis_weight_decompression_overhead(plan)
    core_fast_arena_size = _align_up(budget + weight_overhead, 16)
    if core_fast_arena_size > 0xFFFFFFFF:
        raise ValueError(
            "CMSIS-NN static fast arena exceeds the uint32 runtime size limit"
        )
    slow_arena_size = max(budget * 4, 256 * 1024)
    if slow_arena_size > 0xFFFFFFFF:
        raise ValueError(
            "CMSIS-NN static slow arena exceeds the uint32 runtime size limit"
        )

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
#include "tigris_loader.h"
#include "tigris_mem.h"
#include "tigris_executor.h"
{kernel_include}

{scratch_declaration}

/* Plan binary linked into flash — symbol provided by linker */
extern const uint8_t _binary_model_tgrs_start[];
extern const uint8_t _binary_model_tgrs_end[];

/* Static buffers. Fast memory preserves the full core arena below CMSIS scratch. */
static uint8_t fast_arena[{fast_arena_expr}] __attribute__((aligned(16)));
static uint8_t slow_arena[{slow_arena_size}] __attribute__((aligned(16)));
static void *tensor_ptrs[{plan['num_tensors']}];

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
        fast_arena, sizeof(fast_arena), slow_arena, sizeof(slow_arena));
    if (merr != TIGRIS_MEM_OK) {{
        printf("Memory init failed: %s\\n", tigris_mem_error_str(merr));
        return 1;
    }}
{prepare_block}

    /* 3. Allocate and zero-fill model inputs */
    for (uint8_t i = 0; i < plan.header->num_model_inputs; i++) {{
        uint16_t tidx = plan.model_inputs[i];
        merr = tigris_mem_alloc_slow(&mem, tidx, plan.tensors[tidx].size_bytes);
        if (merr != TIGRIS_MEM_OK) {{
            printf("Input allocation failed for tensor %u: %s\\n",
                   tidx, tigris_mem_error_str(merr));
            return 1;
        }}
        memset(mem.tensor_ptrs[tidx], 0, plan.tensors[tidx].size_bytes);
    }}

    /* 4. Run inference */
    tigris_exec_stats_t stats;
    tigris_exec_error_t eerr = tigris_run(&plan, &mem, {dispatch}, NULL, &stats);
    if (eerr != TIGRIS_EXEC_OK) {{
        printf("Inference failed: %s\\n", tigris_exec_error_str(eerr));
        return 1;
    }}

    printf("OK  normal=%u tiled=%u chain=%u\\n",
           stats.stages_normal, stats.stages_tiled, stats.stages_chain);

    return 0;
}}
"""
