"""Compile generated deployment harnesses as C99 with target API stubs."""

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from tigris.analysis.lifetime import compute_lifetimes
from tigris.analysis.memory import compute_memory_timeline
from tigris.analysis.partition_spatial import partition_spatial
from tigris.analysis.partition_temporal import partition_temporal
from tigris.emitters.binary.writer import emit_binary_bytes
from tigris.emitters.codegen import generate_c, generate_core_header
from tigris.loaders import load_model


_ROOT = Path(__file__).resolve().parents[1]
# With TIGRIS_RUNTIME_SOURCE set or under CI, a missing compiler or header fails.
_RUNTIME_SOURCE = os.environ.get("TIGRIS_RUNTIME_SOURCE")
_RUNTIME_INCLUDE = next(
    (
        candidate / "include"
        for candidate in (
            (Path(_RUNTIME_SOURCE).resolve(),)
            if _RUNTIME_SOURCE
            else (
                _ROOT / "runtime-source",
                _ROOT / "tigris-runtime",
                _ROOT.parent / "tigris-runtime",
            )
        )
        if (candidate / "include").is_dir()
    ),
    None,
)
_CC = shutil.which("cc")


@pytest.fixture(autouse=True)
def require_toolchain():
    if _CC is None or _RUNTIME_INCLUDE is None:
        message = "C compiler and tigris-runtime headers are required"
        if _RUNTIME_SOURCE or os.environ.get("CI"):
            pytest.fail(message)
        pytest.skip(message)


def _plan_bytes(path, *, compress=None):
    graph = load_model(path)
    graph = compute_lifetimes(graph)
    graph = compute_memory_timeline(graph)
    graph = partition_temporal(graph, 4096)
    graph = partition_spatial(graph)
    return emit_binary_bytes(graph, compress=compress)


def _write_esp_stubs(include_dir: Path) -> None:
    (include_dir / "esp_partition.h").write_text(
        """\
#ifndef ESP_PARTITION_H
#define ESP_PARTITION_H
#include <stddef.h>
typedef int esp_err_t;
typedef int esp_partition_mmap_handle_t;
typedef struct { size_t size; } esp_partition_t;
#define ESP_OK 0
#define ESP_PARTITION_TYPE_DATA 0
#define ESP_PARTITION_MMAP_DATA 0
const esp_partition_t *esp_partition_find_first(int, int, const char *);
esp_err_t esp_partition_mmap(const esp_partition_t *, size_t, size_t, int,
                             const void **, esp_partition_mmap_handle_t *);
void esp_partition_munmap(esp_partition_mmap_handle_t);
#endif
"""
    )
    (include_dir / "esp_heap_caps.h").write_text(
        """\
#ifndef ESP_HEAP_CAPS_H
#define ESP_HEAP_CAPS_H
#include <stddef.h>
#include <stdint.h>
#define MALLOC_CAP_SPIRAM 1
#define MALLOC_CAP_INTERNAL 2
#define MALLOC_CAP_8BIT 4
uint32_t heap_caps_get_largest_free_block(uint32_t);
void *heap_caps_malloc(size_t, uint32_t);
void *heap_caps_calloc(size_t, size_t, uint32_t);
void heap_caps_free(void *);
#endif
"""
    )
    (include_dir / "esp_timer.h").write_text(
        """\
#ifndef ESP_TIMER_H
#define ESP_TIMER_H
#include <stdint.h>
int64_t esp_timer_get_time(void);
#endif
"""
    )
    (include_dir / "esp_log.h").write_text(
        """\
#ifndef ESP_LOG_H
#define ESP_LOG_H
#define ESP_LOGE(tag, ...) ((void)(tag))
#endif
"""
    )


@pytest.mark.parametrize(
    ("backend", "quantized"),
    [
        pytest.param("reference", False, id="reference-float32"),
        pytest.param("reference", True, id="reference-int8"),
        pytest.param("esp-nn", True, id="esp-nn-int8"),
        pytest.param("cmsis-nn", True, id="cmsis-nn-int8-compressed"),
    ],
)
def test_generated_harness_is_valid_c99(
    linear_3op_path, qdq_conv_path, tmp_path, backend, quantized
):
    if backend == "cmsis-nn":
        data = _plan_bytes(qdq_conv_path, compress="lz4")
    elif quantized:
        data = _plan_bytes(qdq_conv_path)
    else:
        data = _plan_bytes(linear_3op_path)

    mode = "int8" if quantized else "float32"
    source_path = tmp_path / f"generated-{backend}-{mode}.c"
    source_path.write_text(generate_c(data, backend))

    command = [
        _CC,
        "-std=c99",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-fsyntax-only",
        f"-I{_RUNTIME_INCLUDE}",
    ]
    if backend == "esp-nn":
        stub_dir = tmp_path / "esp-stubs"
        stub_dir.mkdir()
        _write_esp_stubs(stub_dir)
        command.extend([f"-I{stub_dir}", "-DTIGRIS_HAS_ESP_NN"])
    elif backend == "cmsis-nn":
        command.append("-DTIGRIS_HAS_CMSIS_NN")
    command.append(str(source_path))

    result = subprocess.run(command, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("backend", ["reference", "cmsis-nn", "esp-nn"])
def test_generated_core_is_valid_c99(qdq_conv_path, tmp_path, backend):
    source_path = tmp_path / "generated-core.c"
    header_path = tmp_path / "generated_core.h"
    source_path.write_text(
        generate_c(
            _plan_bytes(qdq_conv_path), backend, output_format="core",
            core_header=header_path.name,
        )
    )
    plan_data = _plan_bytes(qdq_conv_path)
    header_path.write_text(generate_core_header(plan_data))

    command = [
        _CC, "-std=c99", "-Wall", "-Wextra", "-Werror", "-fsyntax-only",
        f"-I{_RUNTIME_INCLUDE}", f"-I{tmp_path}", str(source_path),
    ]
    if backend == "cmsis-nn":
        command.append("-DTIGRIS_HAS_CMSIS_NN")
    elif backend == "esp-nn":
        command.append("-DTIGRIS_HAS_ESP_NN")

    result = subprocess.run(command, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr


def test_named_generated_cores_link_together(qdq_conv_path, tmp_path):
    plan_data = _plan_bytes(qdq_conv_path)
    cores = [("tigris_codegen", "default"), ("audio_codegen", "audio")]
    objects = []
    for core_name, stem in cores:
        source_path = tmp_path / f"{stem}.c"
        header_path = tmp_path / f"{stem}.h"
        object_path = tmp_path / f"{stem}.o"
        source_path.write_text(
            generate_c(
                plan_data, "reference", output_format="core",
                core_header=header_path.name, core_name=core_name,
            )
        )
        header_path.write_text(generate_core_header(plan_data, core_name))
        result = subprocess.run(
            [
                _CC, "-std=c99", "-c", f"-I{_RUNTIME_INCLUDE}",
                f"-I{tmp_path}", str(source_path), "-o", str(object_path),
            ],
            text=True, capture_output=True, check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        objects.append(object_path)

    result = subprocess.run(
        [_CC, "-r", *(str(path) for path in objects), "-o", str(tmp_path / "cores.o")],
        text=True, capture_output=True, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("kind", ["Equal", "Not", "Where", "Cast", "Sum"])
@pytest.mark.parametrize("quantized", [False, True])
def test_bool_harness_and_core_are_valid_c99(tmp_path, kind, quantized):
    import onnx
    from scripts.crossrepo_contract import _bool_case
    from tigris.cli import _run_pipeline

    case = _bool_case(kind, quantized)
    path = tmp_path / "bool.onnx"
    onnx.save(case.compile_model, path)
    graph, _ = _run_pipeline(str(path), ("4K",))
    data = emit_binary_bytes(graph)
    header = tmp_path / "generated_core.h"
    header.write_text(generate_core_header(data))
    for fmt in ("app", "core"):
        source = tmp_path / f"bool_{fmt}.c"
        source.write_text(generate_c(data, "reference", output_format=fmt, core_header=header.name))
        result = subprocess.run([_CC, "-std=c99", "-Wall", "-Wextra", "-Werror", "-fsyntax-only",
                                 f"-I{_RUNTIME_INCLUDE}", f"-I{tmp_path}", str(source)],
                                text=True, capture_output=True, check=False)
        assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("backend", ["reference", "esp-nn", "cmsis-nn"])
def test_stateful_harness_and_core_are_valid_c99(tmp_path, backend):
    """A plan keeping a variable gets a state buffer, prepared before the run
    that reads and updates it."""
    data = (Path(__file__).parent / "schema_compat" / "schema-v10-state.tgrs").read_bytes()
    header = tmp_path / "generated_core.h"
    header.write_text(generate_core_header(data))
    assert "_STATE_BYTES 48u" in header.read_text()
    command = [_CC, "-std=c99", "-Wall", "-Wextra", "-Werror", "-fsyntax-only",
               f"-I{_RUNTIME_INCLUDE}", f"-I{tmp_path}"]
    if backend == "esp-nn":
        stub_dir = tmp_path / "esp-stubs"
        stub_dir.mkdir()
        _write_esp_stubs(stub_dir)
        command.append(f"-I{stub_dir}")
    for fmt in ("app", "core"):
        code = generate_c(data, backend, output_format=fmt, core_header=header.name)
        assert ("tigris_state_init" in code) == (fmt == "app")
        source = tmp_path / f"state_{fmt}.c"
        source.write_text(code)
        result = subprocess.run([*command, str(source)], text=True, capture_output=True,
                                check=False)
        assert result.returncode == 0, result.stdout + result.stderr


def _syntax_check(source: Path, include: Path, tmp_path: Path, backend: str):
    command = [_CC, "-std=c99", "-Wall", "-Wextra", "-Werror", "-fsyntax-only",
               f"-I{include}", f"-I{tmp_path}"]
    if backend == "esp-nn":
        stub_dir = tmp_path / "esp-stubs"
        if not stub_dir.is_dir():
            stub_dir.mkdir()
            _write_esp_stubs(stub_dir)
        command.extend([f"-I{stub_dir}", "-DTIGRIS_HAS_ESP_NN"])
    elif backend == "cmsis-nn":
        command.append("-DTIGRIS_HAS_CMSIS_NN")
    return subprocess.run([*command, str(source)], text=True, capture_output=True, check=False)


@pytest.mark.parametrize("fmt", ["app", "core"])
@pytest.mark.parametrize("backend", ["reference", "esp-nn", "cmsis-nn"])
def test_generated_code_stops_on_older_runtime_headers(
    qdq_conv_path, conv_relu_chain_path, tmp_path, backend, fmt
):
    """Generated code places its inputs with tigris_input_alloc(), so it builds
    only against runtime headers that state their version, whether the plan
    reads its inputs in place or is staged."""
    older = tmp_path / "older-include"
    shutil.copytree(_RUNTIME_INCLUDE, older)
    header = older / "tigris.h"
    text = header.read_text()
    stripped = re.sub(r"^#define TIGRIS_RUNTIME_VERSION_\w+ \d+\n", "", text, flags=re.MULTILINE)
    assert stripped.count("\n") == text.count("\n") - 3
    header.write_text(stripped)

    for path in (qdq_conv_path, conv_relu_chain_path):
        data = _plan_bytes(path)
        core_header = tmp_path / "generated_core.h"
        core_header.write_text(generate_core_header(data))
        source = tmp_path / f"generated-{fmt}.c"
        source.write_text(generate_c(data, backend, output_format=fmt,
                                     core_header=core_header.name))
        assert "tigris_input_alloc(" in source.read_text()

        current = _syntax_check(source, _RUNTIME_INCLUDE, tmp_path, backend)
        assert current.returncode == 0, current.stdout + current.stderr
        result = _syntax_check(source, older, tmp_path, backend)
        assert result.returncode != 0
        assert "these runtime headers predate tigris_input_alloc()" in result.stderr


_CORE_DRIVER = r"""
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "generated_core.h"

static _Alignas(32) uint8_t plan_buf[1 << 20];
static _Alignas(32) uint8_t fast_arena[TIGRIS_CODEGEN_CORE_FAST_ARENA_BYTES];
static _Alignas(32) uint8_t slow_arena[TIGRIS_CODEGEN_CORE_SLOW_ARENA_BYTES + 1u];
static _Alignas(32) uint8_t workspace[TIGRIS_CODEGEN_EXECUTOR_WORKSPACE_BYTES];
static void *tensor_ptrs[TIGRIS_CODEGEN_TENSOR_CAPACITY];
static uint8_t outputs[3][1 << 16];

static void fill(void *data, uint32_t size, uint16_t index, void *ctx)
{
    (void)ctx;
    for (uint32_t i = 0; i < size; i++)
        ((uint8_t *)data)[i] = (uint8_t)(7u * i + index);
}

int main(int argc, char **argv)
{
    FILE *f = fopen(argv[1], "rb");
    size_t len = f ? fread(plan_buf, 1, sizeof(plan_buf), f) : 0;
    if (f) fclose(f);
    uint32_t fast_size = sizeof(fast_arena) - (uint32_t)atoi(argv[2]);
    tigris_plan_t plan;
    tigris_mem_t mem;
    tigris_exec_stats_t stats;
    if (tigris_codegen_load_plan(plan_buf, (uint32_t)len, &plan) != TIGRIS_OK)
        return 2;
    tigris_error_t err = tigris_codegen_init(
        &plan, &mem, tensor_ptrs, TIGRIS_CODEGEN_TENSOR_CAPACITY, fast_arena, fast_size,
        TIGRIS_CODEGEN_CORE_SLOW_ARENA_BYTES ? slow_arena : NULL,
        TIGRIS_CODEGEN_CORE_SLOW_ARENA_BYTES, fill, NULL);
    if (err != TIGRIS_OK) {
        printf("init: %s\n", tigris_error_str(err));
        return 3;
    }
    for (int run = 0; run < 3; run++) {
        if (run > 0 && tigris_codegen_reset(&plan, &mem, fill, NULL) != TIGRIS_OK)
            return 4;
        void *placed[8];
        uint8_t inputs = plan.header->num_model_inputs;
        if (inputs > 8u)
            return 8;
        for (uint8_t i = 0; i < inputs; i++)
            placed[i] = tensor_ptrs[plan.model_inputs[i]];
        uint32_t entry = mem.fast_used;
        tigris_exec_error_t ran = tigris_codegen_run_with_workspace_buffer(
            &plan, &mem, &stats, workspace, sizeof(workspace));
        if (ran != TIGRIS_EXEC_OK) {
            int kept = 1;
            for (uint8_t i = 0; i < inputs; i++)
                kept &= tensor_ptrs[plan.model_inputs[i]] == placed[i];
            printf("run: %s after %u stages; fast used %u before, %u after, peak %u; inputs %s\n",
                   tigris_exec_error_str(ran),
                   (unsigned)(stats.stages_normal + stats.stages_tiled + stats.stages_chain),
                   (unsigned)entry, (unsigned)mem.fast_used, (unsigned)mem.fast_peak,
                   kept ? "kept" : "moved");
            return 5;
        }
        uint32_t offset = 0;
        for (uint8_t i = 0; i < plan.header->num_model_outputs; i++) {
            uint16_t tidx = plan.model_outputs[i];
            uint32_t bytes = tigris_iface_bytes(&plan, tidx);
            if (offset + bytes > sizeof(outputs[run]) ||
                tigris_output_read(&plan, &mem, tidx, outputs[run] + offset, bytes) != TIGRIS_OK)
                return 6;
            offset += bytes;
        }
        if (run > 0 && memcmp(outputs[run], outputs[0], offset) != 0)
            return 7;
    }
    printf("core: 3 identical runs, fast peak %u of %u\n", (unsigned)mem.fast_peak,
           (unsigned)fast_size);
    return 0;
}
"""


@pytest.fixture(scope="module")
def sanitized_runtime(tmp_path_factory):
    """The runtime sources compiled once with address and undefined-behavior sanitizers."""
    if _CC is None or _RUNTIME_INCLUDE is None:
        pytest.skip("C compiler and tigris-runtime headers are required")
    build = tmp_path_factory.mktemp("sanitized-runtime")
    objects = []
    for source in sorted((_RUNTIME_INCLUDE.parent / "src").glob("*.c")):
        target = build / f"{source.stem}.o"
        result = subprocess.run(
            [_CC, "-std=c11", "-O1", "-g", "-fsanitize=address,undefined",
             "-fno-sanitize-recover=all", "-fno-omit-frame-pointer", f"-I{_RUNTIME_INCLUDE}",
             "-c", str(source), "-o", str(target)],
            text=True, capture_output=True, check=False)
        assert result.returncode == 0, result.stdout + result.stderr
        objects.append(str(target))
    return objects


def _sanitized_binary(objects, sources, include: Path, output: Path) -> Path:
    result = subprocess.run(
        [_CC, "-std=c11", "-O1", "-g", "-fsanitize=address,undefined",
         "-fno-sanitize-recover=all", "-fno-omit-frame-pointer", f"-I{_RUNTIME_INCLUDE}",
         f"-I{include}", *map(str, sources), *objects, "-lm", "-o", str(output)],
        text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    return output


@pytest.mark.parametrize("plan_kind", ["compressed-one-stage", "staged"])
def test_generated_code_runs_on_the_host(
    sanitized_runtime, conv_relu_chain_path, tmp_path, plan_kind
):
    """Generated app and core code place their inputs with tigris_input_alloc()
    and run under sanitizers at the arena the plan states: a compressed
    one-stage plan reads its inputs in place above the weights, a staged plan
    reads them from slow. The core runs three times on one manager with the
    same result, and an arena one allocation unit short is refused."""
    if plan_kind == "compressed-one-stage":
        from click.testing import CliRunner
        from tigris.cli import cli

        model = Path(__file__).parent / "fixtures" / "tflite" / "ops" / "conv_bias.tflite"
        result = CliRunner().invoke(cli, ["compile", str(model), "-m", "64K", "-c", "lz4",
                                          "-o", str(tmp_path / "compiled.tgrs")])
        assert result.exit_code == 0, result.output
        data = (tmp_path / "compiled.tgrs").read_bytes()
    else:
        data = _plan_bytes(conv_relu_chain_path)
    plan_path = tmp_path / "model.tgrs"
    plan_path.write_bytes(data)
    (tmp_path / "generated_core.h").write_text(generate_core_header(data))
    env = {**os.environ, "ASAN_OPTIONS": "detect_leaks=1", "UBSAN_OPTIONS": "halt_on_error=1"}

    app = tmp_path / "app.c"
    app.write_text(generate_c(data, "reference", output_format="app"))
    binary = _sanitized_binary(sanitized_runtime, [app], tmp_path, tmp_path / "app")
    result = subprocess.run([str(binary), str(plan_path)], text=True, capture_output=True,
                            check=False, env=env)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK  normal=" in result.stdout

    core = tmp_path / "core.c"
    core.write_text(generate_c(data, "reference", output_format="core",
                               core_header="generated_core.h"))
    driver = tmp_path / "driver.c"
    driver.write_text(_CORE_DRIVER)
    binary = _sanitized_binary(sanitized_runtime, [core, driver], tmp_path, tmp_path / "core")
    result = subprocess.run([str(binary), str(plan_path), "0"], text=True,
                            capture_output=True, check=False, env=env)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "core: 3 identical runs" in result.stdout
    short = subprocess.run([str(binary), str(plan_path), "32"], text=True,
                           capture_output=True, check=False, env=env)
    assert short.returncode == 5, short.stdout + short.stderr
    # Refused before the first stage: nothing allocated, the inputs where they were placed.
    refused = re.search(r"run: arena too small for the plan after 0 stages; "
                        r"fast used (\d+) before, (\d+) after, peak (\d+); inputs kept", short.stdout)
    assert refused, short.stdout
    assert len(set(refused.groups())) == 1, short.stdout


_TILE_FACTS_DUMP = r"""
#include "tigris.h"
#include "tigris_loader.h"
#include <stdio.h>
#include <stdlib.h>

/* Per plan its load result, then what the runtime derives for each tile record,
 * and the band kind and mode its own order finds when it reads the records as schema 10. */
int main(int argc, char **argv)
{
    for (int a = 1; a < argc; a++) {
        FILE *file = fopen(argv[a], "rb");
        if (!file) return 2;
        fseek(file, 0, SEEK_END);
        long size = ftell(file);
        fseek(file, 0, SEEK_SET);
        uint8_t *buf = aligned_alloc(64, (size_t)(size + 63) / 64 * 64);
        if (!buf || fread(buf, 1, (size_t)size, file) != (size_t)size) return 3;
        fclose(file);
        tigris_plan_t plan;
        tigris_error_t err = tigris_plan_load(buf, (uint32_t)size, &plan);
        printf("plan %d %d\n", a - 1, (int)err);
        tigris_tile_plan_v10_t *choices = NULL;
        tigris_file_header_t header;
        tigris_plan_t older;
        if (err == TIGRIS_OK) {
            choices = calloc(plan.header->num_tile_plans + 1u, sizeof(*choices));
            if (!choices) return 4;
            for (uint16_t t = 0; t < plan.header->num_tile_plans; t++)
                choices[t] = plan.tile_plans_v11[t].choices;
            header = *plan.header;
            header.version = TIGRIS_SCHEMA_VERSION_V10;
            older = plan;
            older.header = &header;
            older.tile_plans_v10 = choices;
            older.tile_plans_v11 = NULL;
        }
        for (uint16_t s = 0; err == TIGRIS_OK && s < plan.header->num_stages; s++) {
            if (plan.stages[s].tile_plan_idx == TIGRIS_NO_TILE_PLAN) continue;
            tigris_tile_facts_t d;
            int ok = tigris_stage_tile_facts(&plan, &plan.stages[s], &d);
            tigris_stage_band_t band;
            tigris_band_kind_t kind = tigris_stage_band_kind(&older, &plan.stages[s], &band);
            if (!plan.tile_plans_v11[plan.stages[s].tile_plan_idx].choices.tileable)
                kind = TIGRIS_BAND_KIND_NONE;
            printf("stage %u %d %u %u %u %u %u %u %u %d %d\n", (unsigned)s, ok, (unsigned)d.tile_extent,
                   (unsigned)d.num_tiles, (unsigned)d.original_extent, (unsigned)d.halo,
                   (unsigned)d.band_axis, (unsigned)TIGRIS_TILE_BAND_KIND(d.band),
                   (unsigned)TIGRIS_TILE_BAND_MODE(d.band), (int)kind,
                   kind == TIGRIS_BAND_KIND_LEADING ? (int)band.mode : 0);
        }
        free(choices);
        free(buf);
    }
    return 0;
}
"""


def _rank2_dense_constant_model(directory: Path, op_type: str) -> Path:
    """y = op(x, k) with x [256, 64] and k a constant of the same shape."""
    import numpy as np
    from onnx import helper, numpy_helper
    from conftest import save_onnx, vi

    constant = np.linspace(1.0, 2.0, 256 * 64, dtype=np.float32).reshape(256, 64)
    return save_onnx(directory, f"{op_type.lower()}_dense_constant",
                     [helper.make_node(op_type, ["x", "k"], ["y"])], [vi("x", [256, 64])],
                     [vi("y", [256, 64])], [numpy_helper.from_array(constant, "k")])


def test_runtime_derives_every_stated_tile_fact(sanitized_runtime, tmp_path):
    """Every tile record of the operator fixtures at 256 B, 1 KiB and 4 KiB, and
    of a rank-2 Add and Max with a full-shape constant at 16 KiB, states what
    the runtime derives from the plan with the rules it bands the stage by, and
    reads back as the compiler's tile plan: the band, the tile count, the
    extent, the halo, the band axis and the band kind and mode, and the loader
    accepts the plan. The stated kind and mode are the ones the runtime's own
    order finds for the stage when it reads the plan's records as schema 10."""
    from tigris.analysis.plan_graph import graph_from_plan
    from tigris.analysis.validation import validate_memory_plan
    from tigris.cli import _run_pipeline
    from tigris.emitters.binary.reader import read_binary_plan

    facts = ("tile_extent", "num_tiles", "original_extent", "halo", "band_axis", "band_kind", "band_mode")
    fields = ("tileable", "axis", "band_on_columns", "band_in_output_rows", "tile_height",
              "tile_width", "num_tiles", "original_height", "halo", "band_axis",
              "input_channel_group", "input_spatial_interval", "tiled_peak_bytes",
              "band_kind", "band_mode")
    expected, paths = [], []
    fixtures = Path(__file__).parent / "fixtures" / "tflite" / "ops"
    budgets = {model: ("256", "1K", "4K") for model in sorted(fixtures.glob("*.tflite"))}
    for op_type in ("Add", "Max"):
        budgets[_rank2_dense_constant_model(tmp_path, op_type)] = (("16K", "4M"),)
    for model, model_budgets in budgets.items():
        for budget in model_budgets:
            graph, _ = _run_pipeline(str(model), budget if isinstance(budget, tuple) else (budget,),
                                     report_bindings=False)
            if not validate_memory_plan(graph).feasible or not any(
                    stage.tile_plan for stage in graph.stages):
                continue
            data = emit_binary_bytes(graph)
            plan = read_binary_plan(data)
            stated, kinds = {}, {}
            for index, stage in enumerate(plan["stages"]):
                if stage["tile_plan_idx"] != 0xFFFF:
                    record = plan["tile_plans"][stage["tile_plan_idx"]]
                    stated[index] = tuple(record[field] for field in facts)
                    kinds[index] = (record["band_kind"], record["band_mode"])
            stages = graph_from_plan(plan).stages
            assert len(stages) == len(graph.stages)
            for compiled, read in zip(graph.stages, stages):
                if compiled.tile_plan is not None and compiled.tile_plan.tileable:
                    assert ([getattr(compiled.tile_plan, field) for field in fields] ==
                            [getattr(read.tile_plan, field) for field in fields]), (
                        model.stem, budget, compiled.stage_id)
            paths.append(tmp_path / f"{model.stem}-{budget[0] if isinstance(budget, tuple) else budget}.tgrs")
            paths[-1].write_bytes(data)
            expected.append((stated, kinds))
    assert len(paths) > 300

    source = tmp_path / "facts.c"
    source.write_text(_TILE_FACTS_DUMP)
    binary = _sanitized_binary(sanitized_runtime, [source], tmp_path, tmp_path / "facts")
    result = subprocess.run([str(binary), *map(str, paths)], text=True, capture_output=True,
                            check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    derived: list[tuple[dict, dict]] = []
    for line in result.stdout.splitlines():
        kind, *values = line.split()
        if kind == "plan":
            assert values[1] == "0", (paths[int(values[0])].name, line)
            derived.append(({}, {}))
        else:
            assert values[1] == "1", (paths[len(derived) - 1].name, line)
            numbers = tuple(int(value) for value in values[2:])
            derived[-1][0][int(values[0])] = numbers[:7]
            derived[-1][1][int(values[0])] = numbers[7:]
    differences = [(path.name, index, record, got[0].get(index))
                   for path, (want, _), got in zip(paths, expected, derived)
                   for index, record in want.items() if got[0].get(index) != record]
    assert len(derived) == len(paths) and not differences, differences[:10]
    # The solver's kind is the runtime's own choice, so stating it changes no stage's run.
    kinds = [(path.name, index, kind, got[1].get(index))
             for path, (_, want), got in zip(paths, expected, derived)
             for index, kind in want.items() if got[1].get(index) != kind]
    assert not kinds, kinds[:10]
    assert {kind for _, want in expected for kind, _ in want.values()} == set(range(1, 8))
    # A dense constant keeps the rank-2 stage off the row band: it is a leading band on axis 0.
    from tigris.emitters.binary.defs import TILE_BAND_KIND_LEADING, TILE_BAND_MODE_AXIS_BASE
    dense = [set(want.values()) for path, (_, want) in zip(paths, expected) if "dense_constant" in path.name]
    assert dense == [{(TILE_BAND_KIND_LEADING, TILE_BAND_MODE_AXIS_BASE)}] * 2
