"""Picks the TFLite Micro goldens that match the running C library.

Float kernels that call the platform's expm1f, tanhf or erff round as its
libm does, and glibc 2.39 rounds these differently from a correctly rounded
libm such as glibc 2.43. The cases in CASES therefore keep two sets:
`<case>.npz` recorded with glibc 2.39 and `<case>_cr.npz` recorded with a
correctly rounded libm. The running libm is told apart by its results on fixed
probe inputs, never by its version.
"""

import ctypes
import ctypes.util
import struct

# Per function: (input, glibc 2.39 result as float32 bits, correctly rounded result).
PROBES = {
    "expm1f": [(-4.453125, 0xBF7D0506, 0xBF7D0505), (-3.515625, 0xBF7863AA, 0xBF7863AB),
               (-2.765625, 0xBF6FE360, 0xBF6FE361)],
    "tanhf": [(-1.625, 0xBF6CE37E, 0xBF6CE37D), (-1.375, 0xBF613C52, 0xBF613C53),
              (-0.953125, 0xBF3DBEEE, 0xBF3DBEED)],
    "erff": [(-1.703125, 0xBF7BE678, 0xBF7BE677), (-0.96875, 0xBF544E42, 0xBF544E43),
             (-0.171875, 0xBE44A809, 0xBE44A808)],
}

# The cases whose TFLite Micro outputs differ between the two, and the functions they call.
CASES = {
    "float_elu": ("expm1f",), "float_tanh": ("tanhf",), "float_if_through": ("tanhf",),
    "float_lstm": ("tanhf",), "float_lstm_time_major_clip": ("tanhf",),
    "float_lstm_unrolled": ("tanhf",),
}

SUFFIXES = {"legacy": "", "correctly rounded": "_cr"}


def _bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def behavior(function: str) -> str:
    """How the running libm rounds `function`; an error names the probe that fits no set."""
    call = getattr(ctypes.CDLL(ctypes.util.find_library("m")), function)
    call.argtypes, call.restype = [ctypes.c_float], ctypes.c_float
    kinds = []
    for argument, legacy, rounded in PROBES[function]:
        result = _bits(call(argument))
        kinds.append({legacy: "legacy", rounded: "correctly rounded"}.get(result))
        if kinds[-1] is None or kinds[-1] != kinds[0]:
            raise RuntimeError(
                f"{function}({argument!r}) returned 0x{result:08x}, "
                f"where glibc 2.39 gives 0x{legacy:08x} and correct rounding 0x{rounded:08x}: "
                + ("an unknown libm" if kinds[-1] is None else f"unlike its {kinds[0]} first probe"))
    return kinds[0]


def golden_suffix(case: str) -> str:
    """The file name suffix of the goldens of `case` that match the running libm."""
    kinds = {function: behavior(function) for function in CASES.get(case, ())}
    if len(set(kinds.values())) > 1:
        raise RuntimeError(f"{case}: no goldens for a libm whose functions round as {kinds}")
    return SUFFIXES[next(iter(kinds.values()))] if kinds else ""
