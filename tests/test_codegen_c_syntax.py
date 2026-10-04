"""Compile generated deployment harnesses as C99 with target API stubs."""

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
_RUNTIME_INCLUDE = next(
    (
        candidate
        for candidate in (
            _ROOT / "tigris-runtime" / "include",
            _ROOT.parent / "tigris-runtime" / "include",
        )
        if candidate.is_dir()
    ),
    None,
)
_CC = shutil.which("cc")

pytestmark = pytest.mark.skipif(
    _CC is None or _RUNTIME_INCLUDE is None,
    reason="C compiler and tigris-runtime headers are required",
)


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
