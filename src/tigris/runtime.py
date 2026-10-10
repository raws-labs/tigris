"""Host inference through a bundled or explicitly selected C library."""

import builtins
import ctypes as ct
import hashlib
import json
import os
import threading
from pathlib import Path

import numpy as np
from packaging.version import Version


class RuntimeError(ValueError):
    """A host runtime or model execution error."""


# The host library interface this package speaks; the bundled manifest records it.
HOST_ABI = 2


class TraceEvent(ct.Structure):
    """One runtime execution event, as include/tigris_trace.h lays it out."""

    _fields_ = [("kind", ct.c_uint8), ("pool", ct.c_uint8), ("path", ct.c_uint8), ("_pad", ct.c_uint8),
                ("stage", ct.c_uint16), ("tensor", ct.c_uint16), ("op", ct.c_uint16), ("_pad2", ct.c_uint16),
                ("offset", ct.c_uint32), ("src_offset", ct.c_uint32), ("bytes", ct.c_uint32),
                ("row0", ct.c_int32), ("row1", ct.c_int32), ("col0", ct.c_int32), ("col1", ct.c_int32),
                ("fast_used", ct.c_uint32), ("slow_used", ct.c_uint32)]


TRACE_KINDS = {1: "stage_begin", 2: "stage_end", 3: "tile_begin", 4: "load", 5: "spill", 6: "alloc",
               7: "reset", 8: "move", 9: "weights", 10: "op", 11: "copy"}
TRACE_PATHS = {0: "normal", 1: "tiled", 2: "tiled_2d", 3: "by_input", 4: "transpose", 5: "rows",
               6: "reshape", 7: "chain", 8: "control"}
_NONE_U16 = 0xFFFF


def _library():
    override = os.environ.get("TIGRIS_HOST_LIBRARY")
    manifest = None
    source = "bundled"
    if override is not None:
        try:
            path = Path(override).expanduser().resolve(strict=True)
            if not override or not path.is_file():
                raise OSError("Expected a shared library file")
            lib = ct.CDLL(str(path))
        except (OSError, builtins.RuntimeError) as exc:
            raise RuntimeError(f"Cannot load TIGRIS_HOST_LIBRARY={override!r}: {exc}") from exc
        source = f"TIGRIS_HOST_LIBRARY={path}"
    else:
        directory = Path(__file__).with_name("native")
        try:
            manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
            name = manifest["library"]
            if name not in {"libtigris_host.so", "libtigris_host.dylib", "tigris_host.dll"}:
                raise RuntimeError("Invalid bundled library name")
            path = directory / name
            if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["sha256"]:
                raise RuntimeError("Bundled runtime checksum mismatch")
            lib = ct.CDLL(str(path))
        except (OSError, KeyError, json.JSONDecodeError) as exc:
            raise RuntimeError("Bundled host runtime is unavailable. Install a supported platform wheel "
                               "or set TIGRIS_HOST_LIBRARY.") from exc
    signatures = {
        "abi": (ct.c_uint32, []), "version": (ct.c_char_p, []),
        "create": (ct.c_char_p, [ct.c_void_p, ct.c_uint32, ct.c_uint32, ct.POINTER(ct.c_void_p)]),
        "destroy": (None, [ct.c_void_p]),
        "tensor_count": (ct.c_uint32, [ct.c_void_p, ct.c_int]),
        "tensor_name": (ct.c_char_p, [ct.c_void_p, ct.c_int, ct.c_uint32]),
        "tensor_bytes": (ct.c_uint32, [ct.c_void_p, ct.c_int, ct.c_uint32]),
        "tensor_dtype": (ct.c_uint32, [ct.c_void_p, ct.c_int, ct.c_uint32]),
        "tensor_rank": (ct.c_uint32, [ct.c_void_p, ct.c_int, ct.c_uint32]),
        "tensor_dim": (ct.c_int32, [ct.c_void_p, ct.c_int, ct.c_uint32, ct.c_uint32]),
        "run": (ct.c_char_p, [ct.c_void_p, ct.POINTER(ct.c_void_p), ct.POINTER(ct.c_uint32), ct.c_uint32,
                              ct.POINTER(ct.c_void_p), ct.POINTER(ct.c_uint32), ct.c_uint32]),
        "metric": (ct.c_uint64, [ct.c_void_p, ct.c_uint32]),
        "reset_state": (ct.c_char_p, [ct.c_void_p]),
        "trace_buffer": (None, [ct.c_void_p, ct.c_void_p, ct.c_uint32]),
        "trace_count": (ct.c_uint32, [ct.c_void_p]),
    }
    try:
        lib.tigris_host_abi.restype = ct.c_uint32
        lib.tigris_host_abi.argtypes = []
        if lib.tigris_host_abi() != HOST_ABI or (manifest is not None and manifest["abi"] != HOST_ABI):
            raise RuntimeError(f"Unsupported host library ABI ({source})")
        for name, (result, arguments) in signatures.items():
            function = getattr(lib, f"tigris_host_{name}")
            function.restype, function.argtypes = result, arguments
        version = lib.tigris_host_version().decode("ascii")
        if manifest is not None and version != manifest["version"]:
            raise RuntimeError("Bundled runtime version differs from its manifest")
    except (AttributeError, KeyError, UnicodeDecodeError) as exc:
        raise RuntimeError(f"Invalid host library interface ({source})") from exc
    return lib, {"version": version, "source": source}


def runtime_info() -> dict[str, str]:
    """Return the loaded host runtime's version and origin."""
    return _library()[1]


def runtime_version() -> str:
    """Return the loaded host runtime's version."""
    return runtime_info()["version"]


class Session:
    """Execute a compiled plan with arrays in stored axis order and interface dtype.

    Use as a context manager to release the plan, arenas, and executor workspace.
    Calls on a session are serialized; different sessions have independent state.
    """

    def __init__(self, model: str | Path, *, slow_capacity: int = 0):
        if not isinstance(slow_capacity, int) or not 0 <= slow_capacity <= 0xFFFFFFFF:
            raise RuntimeError("Slow capacity must fit uint32 bytes")
        self._lock = threading.RLock()
        self._handle = ct.c_void_p()
        self._lib, info = _library()
        self.runtime_version = info["version"]
        self.runtime_source = info["source"]
        path = Path(model)
        data = path.read_bytes()
        if len(data) > 0xFFFFFFFF:
            raise RuntimeError("Plan exceeds the runtime file-size limit")
        self._check_dependencies(path, data)
        error = self._lib.tigris_host_create(data, len(data), slow_capacity, ct.byref(self._handle))
        self._check(error)
        try:
            self.inputs = self._interface(0)
            self.outputs = self._interface(1)
        except Exception:
            self.close()
            raise

    def _check_dependencies(self, path, data):
        manifest_path = path.with_name("manifest.json")
        if not manifest_path.exists():
            return
        from tigris.zoo import runtime_matches, validate_runtime

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        files = manifest.get("files", [])
        entry = next((item for item in files if item.get("path") == path.name), None)
        if entry is None:
            return
        if entry.get("sha256") != hashlib.sha256(data).hexdigest():
            raise RuntimeError("Plan checksum differs from the zoo manifest")
        dependency = manifest["runtime"]
        receipt_path = path.with_name("download.json")
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if receipt.get("artifact_id") != manifest.get("id"):
                raise RuntimeError("Zoo download record belongs to another artifact")
            dependency = receipt["runtime"]
        validate_runtime(dependency)
        if not runtime_matches(dependency, Version(self.runtime_version)):
            raise RuntimeError(f"Model requires runtime {dependency['min']} through "
                               f"{dependency['max'] or 'no known upper limit'}; "
                               f"installed runtime is {self.runtime_version}")

    @staticmethod
    def _check(error):
        if error:
            raise RuntimeError(error.decode("utf-8", "replace"))

    def _interface(self, output):
        result = []
        for index in range(self._lib.tigris_host_tensor_count(self._handle, output)):
            args = (self._handle, output, index)
            code = self._lib.tigris_host_tensor_dtype(*args)
            dtype = {1: np.dtype("float32"), 2: np.dtype("uint8"), 3: np.dtype("int8"),
                     6: np.dtype("int32"), 7: np.dtype("int64"), 9: np.dtype("bool")}.get(code)
            if dtype is None:
                raise RuntimeError(f"Unsupported interface dtype {code}")
            shape = tuple(self._lib.tigris_host_tensor_dim(*args, axis)
                          for axis in range(self._lib.tigris_host_tensor_rank(*args)))
            result.append({"name": self._lib.tigris_host_tensor_name(*args).decode("utf-8"),
                           "dtype": dtype, "shape": shape,
                           "size_bytes": self._lib.tigris_host_tensor_bytes(*args)})
        if len({item["name"] for item in result}) != len(result):
            raise RuntimeError("Model interface contains duplicate names")
        return result

    def run(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        with self._lock:
            if not self._handle:
                raise RuntimeError("Session is closed")
            if set(inputs) != {item["name"] for item in self.inputs}:
                raise RuntimeError("Input names must match the complete model interface")
            arrays = []
            for info in self.inputs:
                value = np.asarray(inputs[info["name"]])
                if value.dtype != info["dtype"] or value.shape != info["shape"]:
                    raise RuntimeError(f"Input {info['name']!r} requires {info['dtype']} {info['shape']}; "
                                       f"got {value.dtype} {value.shape}")
                arrays.append(np.require(value, requirements=["C", "A"]))
            outputs = [np.empty(info["shape"], dtype=info["dtype"]) for info in self.outputs]
            def pointers(values):
                return (ct.c_void_p * len(values))(*(value.ctypes.data for value in values))

            def sizes(values):
                return (ct.c_uint32 * len(values))(*(value.nbytes for value in values))
            self._check(self._lib.tigris_host_run(self._handle, pointers(arrays), sizes(arrays), len(arrays),
                                                 pointers(outputs), sizes(outputs), len(outputs)))
            return {info["name"]: value for info, value in zip(self.outputs, outputs)}

    def trace(self, inputs: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], list[dict], dict[str, int]]:
        """Run once and return the outputs, every execution event the runtime
        reported in order, and the run's counters. A buffer too small for the
        events is enlarged and the run repeated from the same state."""
        with self._lock:
            if not self._handle:
                raise RuntimeError("Session is closed")
            capacity = 1 << 14
            while True:
                buffer = (TraceEvent * capacity)()
                self._lib.tigris_host_trace_buffer(self._handle, buffer, capacity)
                try:
                    outputs = self.run(inputs)
                finally:
                    count = self._lib.tigris_host_trace_count(self._handle)
                    self._lib.tigris_host_trace_buffer(self._handle, None, 0)
                if count <= capacity:
                    break
                capacity = count
                self.reset_state()
            events = []
            for event in buffer[:count]:
                item = {"kind": TRACE_KINDS.get(event.kind, f"unknown({event.kind})")}
                for name in ("stage", "tensor", "op"):
                    value = getattr(event, name)
                    item[name] = None if value == _NONE_U16 else value
                item["pool"] = {0: "fast", 1: "slow"}.get(event.pool)
                if item["kind"] == "stage_begin":
                    item["path"] = TRACE_PATHS.get(event.path, f"unknown({event.path})")
                for name in ("offset", "src_offset", "bytes", "fast_used", "slow_used"):
                    item[name] = getattr(event, name)
                for name in ("row0", "row1", "col0", "col1"):
                    value = getattr(event, name)
                    item[name] = None if value < 0 else value
                events.append(item)
            return outputs, events, self.counters

    @property
    def counters(self) -> dict[str, int]:
        """The last run's data movement as the runtime counted it."""
        with self._lock:
            if not self._handle:
                raise RuntimeError("Session is closed")
            names = {"load_bytes": 5, "spill_bytes": 6, "weight_bytes": 7, "copy_bytes": 8,
                     "compactions": 9, "tiles": 10, "tensor_align": 11}
            return {name: self._lib.tigris_host_metric(self._handle, index) for name, index in names.items()}

    def reset_state(self):
        """Return every variable the plan keeps across runs to its initial
        value, as when the session was opened."""
        with self._lock:
            if not self._handle:
                raise RuntimeError("Session is closed")
            self._check(self._lib.tigris_host_reset_state(self._handle))

    @property
    def memory(self) -> dict[str, int]:
        with self._lock:
            if not self._handle:
                raise RuntimeError("Session is closed")
            names = ("fast_capacity_bytes", "slow_capacity_bytes", "fast_peak_bytes",
                     "slow_peak_bytes", "executor_workspace_bytes")
            return {name: self._lib.tigris_host_metric(self._handle, index) for index, name in enumerate(names)}

    def close(self):
        with self._lock:
            if self._handle:
                self._lib.tigris_host_destroy(self._handle)
                self._handle = ct.c_void_p()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        if getattr(self, "_handle", None):
            self.close()
