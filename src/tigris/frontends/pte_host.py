"""ExecuTorch programs on the host, behind `tigris run`: the runner built
once per ExecuTorch checkout from pte_host_runner.cpp and kept in the user's
cache, the method's input and output tensors, and one runner call."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tigris.frontends import pte
from tigris.frontends.pte_schema import SCALAR_TYPES

RUNNER_SOURCE = Path(__file__).with_name("pte_host_runner.cpp")
RUNNER_NAME = "pte_host_runner"
# The ExecuTorch release series whose runtime the runner source is written for.
EXECUTORCH_SERIES = "1.6"
# The method the runner executes.
METHOD = "forward"
# Submodules the runtime build reads.
SUBMODULES = ("backends/xnnpack/third-party/FXdiv", "third-party/flatbuffers",
              "third-party/flatcc", "third-party/gflags", "third-party/json")
# The runtime with the cortex_m operators on CMSIS-NN, whose kernels compile to
# their portable C paths on the host.
CMAKE_OPTIONS = (
    "-DCMAKE_BUILD_TYPE=Release",
    "-DEXECUTORCH_BUILD_CORTEX_M=ON",
    "-DEXECUTORCH_BUILD_PORTABLE_OPS=OFF",
    "-DEXECUTORCH_BUILD_EXECUTOR_RUNNER=OFF",
    "-DEXECUTORCH_BUILD_CPUINFO=OFF",
    "-DEXECUTORCH_BUILD_PTHREADPOOL=OFF",
    "-DEXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR=OFF",
    "-DEXECUTORCH_ENABLE_PROGRAM_VERIFICATION=ON",
    "-DEXECUTORCH_ENABLE_LOGGING=ON",
)
CMAKE_TARGETS = ("executorch", "executorch_core", "cortex_m_ops_lib", "cortex_m_kernels",
                 "cmsis-nn", "kernels_util_all_deps")
# Libraries whose static operator registration must be linked whole, then the
# rest in link order, relative to the build directory.
WHOLE_ARCHIVES = ("libexecutorch.a", "backends/cortex_m/libcortex_m_ops_lib.a")
ARCHIVES = ("libexecutorch_core.a", "backends/cortex_m/libcortex_m_kernels.a",
            "backends/cortex_m/cmsis_nn-build/libcmsis-nn.a",
            "kernels/portable/cpu/util/libkernels_util_all_deps.a", "libexecutorch_core.a")
# Lines of error messages shown when a build step fails.
ERROR_LINES = 8
# Modules ExecuTorch's code generator imports during the runner build.
CODEGEN_MODULES = ("torchgen", "yaml")
# The byte the runner fills the planned memory with before each inference.
POISON = 0xA5
# NumPy dtype of each ExecuTorch scalar type an input file can hold.
NUMPY_DTYPES = {0: "uint8", 1: "int8", 2: "int16", 3: "int32", 4: "int64", 5: "float16",
                6: "float32", 7: "float64", 11: "bool", 12: "int8", 13: "uint8", 27: "uint16",
                28: "uint32", 29: "uint64"}


class BuildError(Exception):
    """A runner build step failed; `lines` are the last lines of its output
    that name an error, `log` the file that holds all of it."""

    def __init__(self, step: str, lines: list[str], log: Path):
        super().__init__(step)
        self.step, self.lines, self.log = step, lines, log


@dataclass(frozen=True)
class Checkout:
    """An ExecuTorch source tree; `changes` is the hash of its uncommitted
    changes to tracked files, empty for a clean tree."""
    root: Path
    version: str
    commit: str
    changes: str = ""


@dataclass(frozen=True)
class Runner:
    path: Path
    executorch: str
    commit: str
    cmsis_nn: str
    changes: str = ""


@dataclass(frozen=True)
class MethodTensor:
    """A method input or output: its scalar type, sizes and dim order."""
    scalar_type: int
    sizes: tuple
    dim_order: tuple

    @property
    def nbytes(self) -> int:
        return int(np.prod(self.sizes, dtype=np.int64)) * SCALAR_TYPES[self.scalar_type][1]

    @property
    def dtype(self) -> str | None:
        return NUMPY_DTYPES.get(self.scalar_type)

    @property
    def ordered_sizes(self) -> tuple:
        """The sizes in dim order, outermost first: the shape of the tensor's
        bytes in memory."""
        return tuple(self.sizes[d] for d in self.dim_order)

    def describe(self) -> str:
        """The shape in memory order, the dtype and a dim order other than the
        identity, as inspect shows a tensor: 1x48x48x3 int8, dim order 0,2,3,1."""
        kind = self.dtype or SCALAR_TYPES.get(self.scalar_type, (str(self.scalar_type),))[0]
        shape, order = memory_shape(self.sizes, self.dim_order)
        return f"{shape} {kind.lower()}{order}"

    def array(self, data: bytes) -> np.ndarray:
        """The tensor's bytes in memory as an array in its sizes, as PyTorch
        shows the tensor."""
        inverse = np.argsort(self.dim_order)
        return np.frombuffer(data, self.dtype).reshape(self.ordered_sizes).transpose(inverse)


def checkout(path: Path) -> Checkout:
    """The ExecuTorch source tree at `path`, with its release and commit;
    raises ValueError naming what the runner build needs of it."""
    # The build includes <executorch/...> from the parent of the real directory.
    real = path.resolve()
    if real.name != "executorch":
        raise ValueError(f"{path} is not a directory named executorch, the name ExecuTorch "
                         f"builds from")
    version_file = path / "version.txt"
    if not version_file.is_file() or not (path / "backends" / "cortex_m").is_dir():
        raise ValueError(f"{path} is not an ExecuTorch source tree with backends/cortex_m")
    version = version_file.read_text().strip()
    if version != EXECUTORCH_SERIES and not version.startswith(EXECUTORCH_SERIES + "."):
        raise ValueError(f"{path} is not an ExecuTorch {EXECUTORCH_SERIES} checkout; it holds "
                         f"ExecuTorch {version}")
    empty = [m for m in SUBMODULES if not (path / m).is_dir() or not any((path / m).iterdir())]
    if empty:
        state = "is empty" if len(empty) == 1 else "are empty"
        raise ValueError(f"{path} is not a checkout with its submodules ({', '.join(empty)} "
                         f"{state}); run git submodule update --init --recursive in it")
    try:
        done = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                              capture_output=True, text=True)
    except OSError:
        done = None
    if done is None or done.returncode:
        raise ValueError(f"{path} is not a git checkout; the runner is cached by the ExecuTorch "
                         f"commit it was built from")
    try:
        changes = local_changes(path)
    except subprocess.CalledProcessError as error:
        stderr = error.stderr.decode(errors="replace").strip()
        raise ValueError(f"{path}: git {error.cmd[3]} fails: {stderr}") from None
    return Checkout(path, version, done.stdout.strip(), changes)


def local_changes(path: Path) -> str:
    """The hash of the uncommitted changes to tracked files in the checkout at
    `path`, its submodules included, or "" for a clean tree."""
    status = subprocess.run(["git", "-C", str(path), "status", "--porcelain",
                             "--untracked-files=no"], capture_output=True, check=True)
    if not status.stdout.strip():
        return ""
    diff = subprocess.run(["git", "-C", str(path), "diff", "HEAD", "--binary",
                           "--submodule=diff"], capture_output=True, check=True)
    return hashlib.sha256(status.stdout + diff.stdout).hexdigest()


def cache_root() -> Path:
    """Where built runners are kept: the user's cache directory."""
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local"
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Caches"
    else:
        base = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(base) / "tigris" / "pte-runner"


def runner_key(commit: str, source: bytes, changes: str = "") -> str:
    """The cache key of a runner: the ExecuTorch commit and its uncommitted
    changes, the runner source and the build options."""
    parts = [commit, hashlib.sha256(source).hexdigest(), CMAKE_OPTIONS, CMAKE_TARGETS,
             WHOLE_ARCHIVES, ARCHIVES]
    if changes:
        parts.append(changes)
    recipe = json.dumps(parts)
    return f"{commit[:12]}-{hashlib.sha256(recipe.encode()).hexdigest()[:16]}"


def cached_runner(tree: Checkout, cache: Path | None = None) -> Runner | None:
    """The runner built earlier for `tree` and this runner source, or None."""
    key = runner_key(tree.commit, RUNNER_SOURCE.read_bytes(), tree.changes)
    directory = (cache or cache_root()) / key
    try:
        info = json.loads((directory / "runner.json").read_text())
    except (OSError, ValueError):
        return None
    path = directory / RUNNER_NAME
    if not path.is_file():
        return None
    try:
        return Runner(path, info["executorch"], info["commit"], info["cmsis_nn"],
                      info.get("changes", ""))
    except (KeyError, TypeError):
        return None


def missing_modules(python: str, modules: Sequence[str] = CODEGEN_MODULES) -> list[str]:
    """The `modules` that `python` cannot import, checked in a child process
    so that this one imports none of them; raises OSError when `python` does
    not start."""
    code = ("import importlib, sys\n"
            "for name in sys.argv[1:]:\n"
            "    try:\n"
            "        importlib.import_module(name)\n"
            "    except Exception:\n"
            "        print(name)\n")
    done = subprocess.run([python, "-c", code, *modules], capture_output=True, text=True,
                          stdin=subprocess.DEVNULL)
    if done.returncode:
        return list(modules)
    return done.stdout.split()


def check_python(python: str) -> None:
    """Raise ValueError unless `python` runs ExecuTorch's code generator."""
    try:
        missing = missing_modules(python)
    except OSError as error:
        raise ValueError(f"{python} does not start: {error.strerror or error}") from None
    if missing:
        raise ValueError(f"{python} cannot import {' or '.join(missing)}; the runner build runs "
                         f"ExecuTorch's code generator, which needs a Python with torch and "
                         f"PyYAML")


def error_lines(output: str, count: int = ERROR_LINES) -> list[str]:
    """The first `count` lines of a build's error messages, each message a line
    naming an error and the indented lines after it, or the last `count`
    lines of the output when no line names one."""
    lines = output.splitlines()
    shown = []
    k = 0
    while k < len(lines) and len(shown) < count:
        line = lines[k]
        k += 1
        if "error" not in line.lower() or line.startswith("ninja: build stopped"):
            continue
        shown.append(line.rstrip())
        while k < len(lines) and (not lines[k].strip() or lines[k][0].isspace()):
            if lines[k].strip():
                shown.append(lines[k].rstrip())
            k += 1
    if shown:
        return shown[:count]
    return [line.rstrip() for line in lines if line.strip()][-count:]


def _step(name: str, command: list[str], log: Path, cwd: Path | None = None) -> str:
    with log.open("a") as out:
        out.write(f"$ {' '.join(command)}\n")
    try:
        done = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, cwd=cwd, stdin=subprocess.DEVNULL)
    except OSError as error:
        raise BuildError(name, [str(error)], log) from None
    with log.open("a") as out:
        out.write(done.stdout)
    if done.returncode:
        raise BuildError(name, error_lines(done.stdout), log)
    return done.stdout


def _cache_entry(build: Path, name: str) -> str | None:
    """A value of CMakeCache.txt in `build`."""
    for line in (build / "CMakeCache.txt").read_text().splitlines():
        key, _, value = line.partition("=")
        if key.split(":", 1)[0] == name:
            return value
    return None


def build_runner(tree: Checkout, cache: Path | None = None,
                 python: str | None = None) -> Runner:
    """Build the runner against `tree` into the cache and return it, with
    `python` running ExecuTorch's code generator; raises ValueError before
    any step when that Python cannot run it, and BuildError for a failed
    step, keeping that step's log in the cache."""
    python = python or sys.executable
    check_python(python)
    cache = cache or cache_root()
    source = RUNNER_SOURCE.read_bytes()
    key = runner_key(tree.commit, source, tree.changes)
    cache.mkdir(parents=True, exist_ok=True)
    work = cache / f".build-{key}-{os.getpid()}"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir()
    log = work / "build.log"
    build = work / "executorch"
    try:
        generator = ["-G", "Ninja"] if shutil.which("ninja") else []
        _step("configure", ["cmake", "-S", str(tree.root), "-B", str(build), *generator,
                            *CMAKE_OPTIONS, f"-DPYTHON_EXECUTABLE={python}"],
              log)
        jobs = [] if os.environ.get("CMAKE_BUILD_PARALLEL_LEVEL") else [str(os.cpu_count() or 1)]
        _step("build", ["cmake", "--build", str(build), "--target", *CMAKE_TARGETS,
                        "--parallel", *jobs], log)
        compiler = _cache_entry(build, "CMAKE_CXX_COMPILER") or "c++"
        output = work / RUNNER_NAME
        _step("link", [compiler, "-std=c++17", "-O2", f"-I{tree.root.resolve().parent}",
                       f"-I{tree.root / 'runtime' / 'core' / 'portable_type' / 'c10'}",
                       "-DC10_USING_CUSTOM_GENERATED_MACROS", str(RUNNER_SOURCE),
                       "-Wl,--whole-archive", *(str(build / a) for a in WHOLE_ARCHIVES),
                       "-Wl,--no-whole-archive", *(str(build / a) for a in ARCHIVES),
                       "-lpthread", "-ldl", "-o", str(output)], log)
        info = {"executorch": tree.version, "commit": tree.commit,
                "cmsis_nn": _cache_entry(build, "CMSIS_NN_VERSION") or "unknown",
                "source_sha256": hashlib.sha256(source).hexdigest(), "changes": tree.changes}
        entry = work / "entry"
        entry.mkdir()
        shutil.copy2(output, entry / RUNNER_NAME)
        (entry / "runner.json").write_text(json.dumps(info, indent=1) + "\n")
        try:
            os.replace(entry, cache / key)
        except OSError:  # built meanwhile by another run, or a stale entry
            if cached_runner(tree, cache) is None:
                shutil.rmtree(cache / key, ignore_errors=True)
                os.replace(entry, cache / key)
    except BuildError as error:
        kept = cache / f"build-{key}.log"
        shutil.copy2(log, kept)
        error.log = kept
        raise
    finally:
        shutil.rmtree(work, ignore_errors=True)
    runner = cached_runner(tree, cache)
    assert runner is not None
    return runner


def forward_plan(program: pte.Program) -> pte.ExecutionPlan:
    """The method the runner executes; raises ValueError when the program has
    none or it takes other than one tensor input."""
    plans = [p for p in program.execution_plan or () if p.name == METHOD]
    if not plans:
        raise ValueError(f"has no method {METHOD}, the method the host runs")
    plan = plans[0]
    inputs = plan.inputs or []
    if len(inputs) != 1 or not isinstance(plan.values[inputs[0]], pte.Tensor):
        raise ValueError(f"method {METHOD} takes {len(inputs)} inputs; the host runs methods "
                         f"that take one tensor")
    return plan


def _tensor(plan: pte.ExecutionPlan, index: int, what: str) -> MethodTensor:
    tensor = plan.values[index]
    if not isinstance(tensor, pte.Tensor):
        raise ValueError(f"method {METHOD} returns {what}, which is not a tensor; the host runs "
                         f"methods that return tensors")
    if tensor.scalar_type not in SCALAR_TYPES:
        raise ValueError(f"method {METHOD} has {what} of unknown scalar type "
                         f"{tensor.scalar_type}")
    sizes = tuple(tensor.sizes or ())
    order = tuple(tensor.dim_order or range(len(sizes)))
    if sorted(order) != list(range(len(sizes))):
        raise ValueError(f"method {METHOD} has {what} with dim order "
                         f"{','.join(str(d) for d in order)} for {len(sizes)} dimensions")
    return MethodTensor(tensor.scalar_type, sizes, order)


def method_input(plan: pte.ExecutionPlan) -> MethodTensor:
    return _tensor(plan, plan.inputs[0], "an input")


def method_outputs(plan: pte.ExecutionPlan) -> list[MethodTensor]:
    return [_tensor(plan, index, f"output {k}") for k, index in enumerate(plan.outputs or ())]


def _shape(shape: Sequence[int]) -> str:
    return "x".join(str(s) for s in shape) or "scalar"


def memory_shape(sizes: Sequence[int], dim_order: Sequence[int]) -> tuple[str, str]:
    """A tensor's shape as inspect shows it: the sizes in memory order
    (as given when the dim order is no permutation of them), and the
    ", dim order 0,2,3,1" clause, empty for the identity order."""
    order = list(dim_order)
    if order == list(range(len(sizes))):
        return _shape(sizes), ""
    if sorted(order) == list(range(len(sizes))):
        sizes = [sizes[d] for d in order]
    return _shape(sizes), ", dim order " + ",".join(str(d) for d in order)


def _squeezed(shape: Sequence[int]) -> tuple:
    return tuple(s for s in shape if s != 1)


def npy_bytes(name: str, array: np.ndarray, spec: MethodTensor) -> bytes:
    """The input bytes of a .npy array: one with the method input's sizes,
    as PyTorch shows the tensor, laid out in its dim order; one with the
    sizes in dim order, or with one dimension, as it is. Dimensions of size
    1 are ignored. Raises ValueError for other shapes and for a shape that
    reads both ways with different bytes."""
    if spec.dtype is None or array.dtype != np.dtype(spec.dtype):
        raise ValueError(f"{name} holds {array.dtype} values; the method input is "
                         f"{spec.describe()}")
    shape = _squeezed(array.shape)
    raw = np.ascontiguousarray(array).tobytes()
    if array.ndim <= 1:
        return raw
    as_sizes = shape == _squeezed(spec.sizes)
    as_ordered = shape == _squeezed(spec.ordered_sizes)
    if not as_sizes and not as_ordered:
        layouts = f"shape {_shape(spec.ordered_sizes)}"
        if spec.ordered_sizes != spec.sizes:
            layouts += f", shape {_shape(spec.sizes)} as PyTorch shows the tensor,"
        raise ValueError(f"{name} has shape {_shape(array.shape)}; the method input is "
                         f"{spec.describe()}; a .npy input has {layouts} or one dimension")
    if not as_sizes:
        return raw
    laid_out = np.ascontiguousarray(
        array.reshape(spec.sizes).transpose(spec.dim_order)).tobytes()
    if as_ordered and laid_out != raw:
        raise ValueError(f"{name} has shape {_shape(array.shape)}, which matches the method "
                         f"input ({spec.describe()}) both in memory order and as PyTorch "
                         f"shows the tensor; a .npy input with one dimension or a .bin file "
                         f"gives the bytes in memory order")
    return laid_out


class RunError(Exception):
    """The runner stopped on a program; `lines` are its last error lines."""

    def __init__(self, lines: list[str]):
        super().__init__("; ".join(lines))
        self.lines = lines


def run(runner: list[str], program: Path, inputs: list[bytes], work: Path,
        poison: int = POISON) -> list[bytes]:
    """The output bytes of `program` for each of `inputs`, from one runner
    call that runs them in order, its planned memory filled with `poison`
    before each."""
    feed = work / f"{program.stem}.in"
    out = work / f"{program.stem}.{poison:02x}.out"
    after = work / f"{program.stem}.{poison:02x}.after"
    feed.write_bytes(b"".join(inputs))
    done = subprocess.run([*runner, str(program), str(feed), str(out), str(after),
                           f"0x{poison:02x}"], capture_output=True, text=True,
                          stdin=subprocess.DEVNULL)
    if done.returncode:
        raise RunError(error_lines(done.stderr, 3) or [f"exit status {done.returncode}"])
    data = out.read_bytes()
    if len(data) % len(inputs):
        raise RunError([f"wrote {len(data)} output bytes for {len(inputs)} inputs"])
    size = len(data) // len(inputs)
    return [data[k * size:(k + 1) * size] for k in range(len(inputs))]
