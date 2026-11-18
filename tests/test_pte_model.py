"""The cortex_m operator table, CMSIS-NN class inference and the replay of
ExecuTorch's method allocator."""

import importlib
import itertools
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tigris.frontends import pte
from tigris.frontends import pte_cortex_m as cm
from tigris.frontends import pte_method as pm
from tigris.frontends.pte_cortex_m import DSP, MVE, SCALAR, Dims

FIXTURES = Path(__file__).parent / "fixtures" / "pte"
PTES = sorted(FIXTURES.glob("*.pte"))
_CPU_CLASS = {"m0": SCALAR, "m7": DSP, "m55": MVE}


def _require(module: str):
    """Skip without `module`; fail under TIGRIS_REQUIRE_PYTORCH."""
    try:
        return importlib.import_module(module)
    except ImportError:
        if os.environ.get("TIGRIS_REQUIRE_PYTORCH"):
            pytest.fail(f"{module} is not installed")
        pytest.skip(f"{module} is not installed")


def _plan(name: str) -> pte.ExecutionPlan:
    return pte.read((FIXTURES / name).read_bytes()).plan()


def _call(plan: pte.ExecutionPlan, op: str) -> pte.KernelCall:
    return next(i for i in plan.chains[0].instructions
                if plan.operators[i.op_index].name == cm.NAMESPACE + op)


@pytest.mark.parametrize("path", PTES, ids=lambda p: p.stem)
def test_every_fixture_call_has_a_signature(path):
    plan = pte.read(path.read_bytes()).plan()
    calls = cm.kernel_calls(plan)
    assert [c.index for c in calls] == list(range(len(plan.chains[0].instructions)))


def test_fixtures_hold_every_operator_of_the_table():
    """Each signature in the table is one a lowered program was checked against."""
    seen = {c.signature.name for path in PTES
            for c in cm.kernel_calls(pte.read(path.read_bytes()).plan())}
    assert seen == set(cm.SIGNATURES)


def test_signature_roles():
    conv = cm.SIGNATURES["cortex_m::quantized_conv2d"]
    assert len(conv.names) == 15 and conv.names[-2:] == ("out", "return")
    assert conv.scratch == (12,) and conv.inputs == (0, 1, 2, 8, 9)
    assert conv.lengths == {"stride": 2, "padding": 2, "dilation": 2}
    assert len(cm.SIGNATURES["cortex_m::quantized_depthwise_conv2d"].names) == 16
    assert cm.SIGNATURES["cortex_m::quantized_transpose_conv2d"].scratch == (13, 14)
    assert cm.SIGNATURES["cortex_m::quantized_add"].inputs == (0, 4)
    assert cm.SIGNATURES["cortex_m::pad"].lengths == {"pre_pad": 4, "post_pad": 4}


def _refusal(plan: pte.ExecutionPlan, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        cm.kernel_calls(plan)


def test_unknown_and_nhwc_operators_are_refused():
    plan = _plan("chain_m7.pte")
    conv = plan.operators[_call(plan, "quantized_conv2d").op_index]
    conv.name = "cortex_m::quantized_conv2d_nhwc"
    _refusal(plan, r"uses the explicit NHWC layout \(cortex_m::quantized_conv2d_nhwc\), which "
             "tigris export does not rewrite")
    conv.name = "aten::convolution"
    _refusal(plan, "calls operator aten::convolution; tigris export rewrites programs of the "
             "cortex_m operators quantized_conv2d, ")
    conv.name = "cortex_m::quantize_per_tensor"
    _refusal(plan, "calls cortex_m::quantize_per_tensor; tigris export rewrites programs that "
             "take and return int8 tensors")
    conv.name = "cortex_m::dequantize_per_tensor"
    _refusal(plan, "rewrites programs that take and return int8 tensors")
    conv.name = "cortex_m::minimum"
    _refusal(plan, "calls operator cortex_m::minimum; tigris export rewrites programs of the "
             "cortex_m operators quantized_conv2d, quantized_depthwise_conv2d, "
             "quantized_transpose_conv2d, quantized_add, quantized_mul, quantized_avg_pool2d, "
             "quantized_max_pool2d, quantized_linear, quantized_activation, pad, softmax$")
    conv.name, conv.overload = "cortex_m::quantized_conv2d", "default"
    _refusal(plan, r"calls operator cortex_m::quantized_conv2d\.default; tigris export "
             "rewrites programs that call the out variant of each operator$")


def test_signature_mismatches_are_refused():
    plan = _plan("chain_m7.pte")
    call = _call(plan, "quantized_conv2d")
    args = list(call.args)
    call.args = args[:-1]
    _refusal(plan, "quantized_conv2d takes 15 arguments, the program passes 14")
    call.args = [args[0], args[6]] + args[2:]
    _refusal(plan, "argument weight is a Int, expected Tensor")
    call.args = args[:3] + [args[6]] + args[4:]
    _refusal(plan, "argument stride is a Int, expected IntList")
    call.args = args[:2] + [args[3]] + args[3:]
    _refusal(plan, "argument bias is a IntList, expected Tensor or Null")
    call.args = args[:-1] + [args[0]]
    _refusal(plan, "returns a value other than its out tensor")
    plan.values.append(pte.IntList(items=plan.values[args[4]].items * 2))
    call.args = args[:4] + [len(plan.values) - 1] + args[5:]
    _refusal(plan, "argument padding holds 4 values, expected 2")
    plan.values[-1].items = [args[0], args[0]]
    _refusal(plan, "argument padding lists a value that is not an Int")


def test_bias_may_be_absent():
    plan = _plan("chain_m7.pte")
    call = _call(plan, "quantized_conv2d")
    plan.values.append(pte.Null())
    call.args[2] = len(plan.values) - 1
    assert cm.kernel_calls(plan)


def test_other_instructions_are_refused():
    plan = _plan("chain_m7.pte")
    plan.chains[0].instructions.append(pte.MoveCall(0, 1))
    _refusal(plan, "has a MoveCall at instruction 4; tigris export rewrites methods of kernel "
             "calls")
    plan = _plan("chain_m7.pte")
    plan.chains.append(plan.chains[0])
    _refusal(plan, "runs 2 chains; tigris export rewrites a method of one chain")
    plan = _plan("chain_m7.pte")
    plan.chains[0].instructions[0].op_index = 9
    _refusal(plan, "names operator 9, which does not exist")


@pytest.mark.parametrize("path", [p for p in PTES if p.stem != "ops_m7"], ids=lambda p: p.stem)
def test_class_is_inferred(path):
    expected = _CPU_CLASS[path.stem.rsplit("_", 1)[1]]
    assert cm.infer_class(pte.read(path.read_bytes()).plan()) == (expected,)


def test_ops_program_is_dsp():
    """The 3x3 convolutions with padding store the im2col buffer of a non-MVE
    class and the average pool the DSP accumulator row."""
    assert cm.infer_class(_plan("ops_m7.pte")) == (DSP,)


def test_ambiguous_program_returns_several_classes():
    """Without its average pool, the residual program's 3x3 convolutions and
    biased linear layer fit both SCALAR and DSP kernels."""
    plan = _plan("resblock_m7.pte")
    pool = _call(plan, "quantized_avg_pool2d")
    plan.chains[0].instructions.remove(pool)
    assert cm.infer_class(plan) == (SCALAR, DSP)
    plan = _plan("resblock_m55.pte")
    plan.chains[0].instructions.remove(_call(plan, "quantized_avg_pool2d"))
    assert cm.infer_class(plan) == (MVE,)


def test_class_mismatch_is_refused():
    plan = _plan("chain_m7.pte")
    call = _call(plan, "quantized_depthwise_conv2d")
    plan.values[call.args[13]].sizes = [799]
    with pytest.raises(ValueError, match=r"instruction 2 \(cortex_m::quantized_depthwise_conv2d\) "
                       r"stores 799 scratch bytes where SCALAR, DSP and MVE need 0, 800 and "
                       r"12400, which rules out SCALAR and DSP"):
        cm.infer_class(plan)
    plan = _plan("resblock_m7.pte")
    call = _call(plan, "quantized_linear")
    call.args[3] = call.args[2]
    with pytest.raises(ValueError, match="takes both a bias and a kernel sum"):
        cm.infer_class(plan)
    plan = _plan("resblock_m55.pte")
    call = _call(plan, "quantized_linear")
    call.args[2], call.args[3] = call.args[3], call.args[2]
    with pytest.raises(ValueError, match=r"quantized_linear\) takes no kernel sum, which "
                       "rules out MVE"):
        cm.infer_class(plan)


def _set_ints(plan: pte.ExecutionPlan, call: pte.KernelCall, position: int, ints) -> None:
    for value in ints:
        plan.values.append(pte.Int(int_val=value))
    plan.values.append(pte.IntList(items=list(range(len(plan.values) - len(ints),
                                                     len(plan.values)))))
    call.args[position] = len(plan.values) - 1


@pytest.mark.parametrize("position, ints, match", [
    (3, (1, 0), r"stride \[1, 0\], each must be at least 1"),
    (4, (0, -1), r"padding \[0, -1\], each must be at least 0"),
    (5, (0, 1), r"dilation \[0, 1\], each must be at least 1"),
])
def test_bad_convolution_geometry_is_refused(position, ints, match):
    plan = _plan("chain_m7.pte")
    _set_ints(plan, _call(plan, "quantized_conv2d"), position, ints)
    with pytest.raises(ValueError, match=r"instruction 0 \(cortex_m::quantized_conv2d\) has "
                       + match):
        cm.infer_class(plan)


def test_bad_sizes_are_refused():
    plan = _plan("chain_m7.pte")
    call = _call(plan, "quantized_conv2d")
    plan.values[call.args[0]].sizes = [1, 3, -16, 16]
    with pytest.raises(ValueError, match=r"instruction 0 \(cortex_m::quantized_conv2d\) has "
                       r"input dimensions \[1, -16, 16, 3\], each must be at least 1"):
        cm.infer_class(plan)
    plan = _plan("chain_m7.pte")
    call = _call(plan, "quantized_depthwise_conv2d")
    weight = plan.values[call.args[1]]
    weight.sizes = [1, 0, *weight.sizes[2:]]
    with pytest.raises(ValueError, match="has weight dimensions .*, each must be at least 1"):
        cm.infer_class(plan)
    weight.sizes = [2, 5, *weight.sizes[2:]]
    with pytest.raises(ValueError, match="expected one in the first"):
        cm.infer_class(plan)
    plan = _plan("resblock_m7.pte")
    _set_ints(plan, _call(plan, "quantized_avg_pool2d"), 1, (0, 4))
    with pytest.raises(ValueError, match=r"quantized_avg_pool2d\) has kernel_size \[0, 4\], "
                       "each must be at least 1"):
        cm.infer_class(plan)
    plan = _plan("resblock_m7.pte")
    _set_ints(plan, _call(plan, "quantized_avg_pool2d"), 2, (1, 0))
    with pytest.raises(ValueError, match=r"quantized_avg_pool2d\) has stride \[1, 0\]"):
        cm.infer_class(plan)


def test_scratch_functions_refuse_bad_geometry():
    row = (Dims(1, 1, 8, 4), Dims(8, 1, 3, 4), Dims(1, 1, 8, 8), (0, 1), (1, 0), (1, 1))
    for cls in cm.CLASSES:
        with pytest.raises(ValueError, match=r"the convolution has stride \[1, 0\]"):
            cm.conv_scratch(cls, *row)
        with pytest.raises(ValueError, match="the depthwise convolution has stride"):
            cm.depthwise_scratch(cls, *row)


def test_cpu_is_checked_against_the_class():
    assert cm.check_cpu((DSP,), None) == DSP
    assert cm.check_cpu((DSP,), "cortex-m7") == DSP
    assert cm.check_cpu((SCALAR, DSP), "M0+") == SCALAR
    assert cm.check_cpu((SCALAR, DSP), "m33") == DSP
    with pytest.raises(ValueError, match=r"^was lowered for DSP kernels \(Cortex-M4, M7, M33, "
                       r"M35P\); --cpu M55 runs MVE kernels$"):
        cm.check_cpu((DSP,), "M55")
    with pytest.raises(ValueError, match="fits CMSIS-NN classes SCALAR, DSP; name the CPU"):
        cm.check_cpu((SCALAR, DSP), None)
    with pytest.raises(ValueError, match="unknown Cortex-M CPU 'M5'"):
        cm.check_cpu((DSP,), "M5")
    assert {cm.cpu_class(c) for c in cm.CPU_CLASSES} == set(cm.CLASSES)


def test_scratch_functions_on_known_layers():
    """Values CMSIS-NN v8.0.0 gives for layers of the fixtures and for its
    special paths."""
    conv = (Dims(1, 16, 16, 3), Dims(8, 3, 3, 3), Dims(1, 16, 16, 8), (1, 1), (1, 1), (1, 1))
    assert [cm.conv_scratch(c, *conv) for c in cm.CLASSES] == [112, 112, 128]
    pointwise = (Dims(1, 8, 8, 16), Dims(8, 1, 1, 16), Dims(1, 8, 8, 8), (0, 0), (1, 1), (1, 1))
    assert [cm.conv_scratch(c, *pointwise) for c in cm.CLASSES] == [0, 0, 0]
    # 1xN: one input row, stride x channels a multiple of four.
    row = (Dims(1, 1, 10, 4), Dims(8, 1, 3, 4), Dims(1, 1, 10, 8), (0, 1), (1, 1), (1, 1))
    assert [cm.conv_scratch(c, *row) for c in cm.CLASSES] == [48, 48, 16]
    # A single output column padded on both sides falls back to the im2col buffer.
    column = (Dims(1, 1, 1, 4), Dims(8, 1, 3, 4), Dims(1, 1, 1, 8), (0, 1), (1, 1), (1, 1))
    assert [cm.conv_scratch(c, *column) for c in cm.CLASSES] == [48, 48, 64]
    dw5 = (Dims(1, 8, 8, 16), Dims(16, 5, 5, 1), Dims(1, 8, 8, 16), (2, 2), (1, 1), (1, 1))
    assert [cm.depthwise_scratch(c, *dw5) for c in cm.CLASSES] == [0, 800, 12400]
    dw3 = (Dims(1, 8, 8, 16), Dims(16, 3, 3, 1), Dims(1, 8, 8, 16), (1, 1), (1, 1), (1, 1))
    assert [cm.depthwise_scratch(c, *dw3) for c in cm.CLASSES] == [0, 0, 4464]
    # MVE runs a one-input-channel depthwise with several outputs as a convolution.
    dw_in1 = (Dims(1, 8, 8, 1), Dims(8, 3, 3, 1), Dims(1, 8, 8, 8), (1, 1), (1, 1), (1, 1))
    assert [cm.depthwise_scratch(c, *dw_in1) for c in cm.CLASSES] == [0, 0, 64 + 72]
    assert [cm.avgpool_scratch(c, 1, 16) for c in cm.CLASSES] == [0, 64, 0]


def _out(size: int, kernel: int, stride: int, pad: int, dilation: int) -> int:
    return (size + 2 * pad - dilation * (kernel - 1) - 1) // stride + 1


def _grid():
    """Convolution, depthwise and pooling cases over the branches of the
    buffer-size functions: (kind, input, filter, output, padding, stride,
    dilation)."""
    sizes = ((1, 7), (1, 10), (5, 5), (8, 6))
    kernels = ((1, 1), (1, 3), (1, 5), (3, 1), (3, 3), (5, 5))
    for (h, w), (kh, kw), c, stride, pad, dil in itertools.product(
            sizes, kernels, (1, 3, 4, 8, 17), ((1, 1), (2, 2), (1, 2), (2, 1)),
            ((0, 0), (1, 1), (0, 2), (2, 1)), ((1, 1), (2, 2), (1, 2))):
        oh = _out(h, kh, stride[0], pad[0], dil[0])
        ow = _out(w, kw, stride[1], pad[1], dil[1])
        if oh <= 0 or ow <= 0:
            continue
        for groups in (1, 2, 4):
            if c % groups:
                continue
            for oc in sorted({groups, 8}):
                yield "conv", Dims(1, h, w, c), Dims(oc, kh, kw, c // groups), \
                    Dims(1, oh, ow, oc), pad, stride, dil
        for mult in (1, 2, 8):
            for n in (1, 2):
                yield "dw", Dims(n, h, w, c), Dims(c * mult, kh, kw, 1), \
                    Dims(n, oh, ow, c * mult), pad, stride, dil
    for ow, c in itertools.product((1, 4, 9), (1, 8, 17)):
        yield "pool", Dims(1, 1, 1, c), Dims(1, 1, 1, 1), Dims(1, 1, ow, c), (0, 0), (1, 1), \
            (1, 1)


def _port(kind, inp, filt, out, pad, stride, dil) -> list[int]:
    if kind == "conv":
        return [cm.conv_scratch(c, inp, filt, out, pad, stride, dil) for c in cm.CLASSES]
    if kind == "dw":
        return [cm.depthwise_scratch(c, inp, filt, out, pad, stride, dil) for c in cm.CLASSES]
    return [cm.avgpool_scratch(c, out.w, inp.c) for c in cm.CLASSES]


def _branch(kind, inp, filt, out, pad, stride, dil) -> str:
    if kind == "pool":
        return "pool"
    if kind == "conv":
        if cm._is_1x1(pad, dil, inp, filt):
            return "conv 1x1"
        return "conv 1xN" if cm._is_1_x_n(stride, dil, inp, filt) else "conv im2col"
    if inp.c == 1 and out.c > 1:
        return "depthwise as conv"
    if inp.c != out.c or inp.n != 1 or dil != (1, 1):
        return "depthwise generic"
    return "depthwise 3x3" if filt[1:3] == (3, 3) and max(pad) <= 1 else "depthwise opt"


def test_grid_reaches_every_branch():
    cases = list(_grid())
    branches = {_branch(*case) for case in cases}
    assert branches == {"pool", "conv 1x1", "conv 1xN", "conv im2col", "depthwise as conv",
                        "depthwise generic", "depthwise 3x3", "depthwise opt"}
    # Grouped convolutions of the 1x1 and 1xN shapes, which take the im2col branch.
    grouped = [(inp, filt, pad, stride, dil) for kind, inp, filt, _, pad, stride, dil in cases
               if kind == "conv" and filt.c != inp.c]
    assert any(cm._is_1x1(pad, dil, inp, filt._replace(c=inp.c))
               for inp, filt, pad, stride, dil in grouped)
    assert any(cm._is_1_x_n(stride, dil, inp, filt._replace(c=inp.c))
               for inp, filt, pad, stride, dil in grouped)
    results = [_port(*case) for case in cases]
    assert any(r[0] != r[1] for r in results) and any(r[1] != r[2] for r in results)


def test_scratch_functions_agree_with_executorch():
    """ExecuTorch's binding of CMSIS-NN's own buffer-size functions, which
    sized the scratch tensors of every program, gives the same bytes."""
    _require("executorch")
    try:
        from executorch.backends.cortex_m.library import cmsis_nn
        cmsis_nn.resolve_backend(cmsis_nn.CortexM.M7)
    except (ImportError, ModuleNotFoundError) as error:
        if os.environ.get("TIGRIS_REQUIRE_PYTORCH"):
            pytest.fail(f"the CMSIS-NN binding is not built: {error}")
        pytest.skip(f"the CMSIS-NN binding is not built: {error}")
    backends = [getattr(cmsis_nn.Backend, c) for c in cm.CLASSES]
    a8w8 = cmsis_nn.DataType.A8W8
    assert [cmsis_nn.resolve_backend(getattr(cmsis_nn.CortexM, cpu)).name
            for cpu in cm.CPU_CLASSES] == list(cm.CPU_CLASSES.values())
    for case in _grid():
        kind, inp, filt, out, pad, stride, dil = case
        if kind == "pool":
            theirs = [cmsis_nn.avgpool_buffer_size(b, a8w8, dim_dst_width=out.w, ch_src=inp.c)
                      for b in backends]
        else:
            geometry = dict(input_nhwc=list(inp), filter_nhwc=list(filt), output_nhwc=list(out),
                            padding_hw=list(pad), stride_hw=list(stride), dilation_hw=list(dil))
            if kind == "conv":
                theirs = [cmsis_nn.convolve_wrapper_buffer_size(b, a8w8, **geometry)
                          for b in backends]
            else:
                theirs = [cmsis_nn.depthwise_conv_wrapper_buffer_size(
                    b, a8w8, ch_mult=out.c // inp.c, **geometry) for b in backends]
        assert _port(*case) == theirs, case


_HARNESS = r"""
#include <stdio.h>
#include "arm_nnfunctions.h"

int main(void) {
    char kind[8];
    int v[18];
    while (scanf("%7s", kind) == 1) {
        for (int i = 0; i < 18; i++) if (scanf("%d", &v[i]) != 1) return 1;
        cmsis_nn_dims in = {v[0], v[1], v[2], v[3]}, f = {v[4], v[5], v[6], v[7]},
                      o = {v[8], v[9], v[10], v[11]};
        cmsis_nn_tile pad = {v[13], v[12]}, stride = {v[15], v[14]}, dil = {v[17], v[16]};
        cmsis_nn_activation act = {-128, 127};
        if (kind[0] == 'c') {
            cmsis_nn_conv_params p = {0, 0, stride, pad, dil, act};
            printf("%d %d %d\n", (int)arm_convolve_wrapper_s8_get_buffer_size(&p, &in, &f, &o),
                   (int)arm_convolve_wrapper_s8_get_buffer_size_dsp(&p, &in, &f, &o),
                   (int)arm_convolve_wrapper_s8_get_buffer_size_mve(&p, &in, &f, &o));
        } else if (kind[0] == 'd') {
            cmsis_nn_dw_conv_params p = {0, 0, o.c / in.c, stride, pad, dil, act};
            printf("%d %d %d\n",
                   (int)arm_depthwise_conv_wrapper_s8_get_buffer_size(&p, &in, &f, &o),
                   (int)arm_depthwise_conv_wrapper_s8_get_buffer_size_dsp(&p, &in, &f, &o),
                   (int)arm_depthwise_conv_wrapper_s8_get_buffer_size_mve(&p, &in, &f, &o));
        } else {
            printf("%d %d %d\n", (int)arm_avgpool_s8_get_buffer_size(o.w, in.c),
                   (int)arm_avgpool_s8_get_buffer_size_dsp(o.w, in.c),
                   (int)arm_avgpool_s8_get_buffer_size_mve(o.w, in.c));
        }
    }
    return 0;
}
"""
_SOURCES = ("ConvolutionFunctions/arm_convolve_get_buffer_sizes_s8.c",
            "ConvolutionFunctions/arm_depthwise_conv_get_buffer_sizes_s8.c",
            "PoolingFunctions/arm_avgpool_get_buffer_sizes_s8.c")


def test_scratch_functions_agree_with_cmsis_nn_source(tmp_path):
    """CMSIS-NN v8.0.0's C functions, compiled with GCC for the host from the
    checkout TIGRIS_CMSIS_NN_SOURCE names: a plain build gives SCALAR from the
    generic function and DSP and MVE from the class-suffixed ones, the way
    ExecuTorch's binding calls them; a build with ARM_MATH_DSP gives DSP from
    the generic function, the way a DSP target selects it."""
    source = os.environ.get("TIGRIS_CMSIS_NN_SOURCE")
    compiler = shutil.which("gcc") or shutil.which("cc")
    if not source or not compiler:
        pytest.skip("needs TIGRIS_CMSIS_NN_SOURCE (a CMSIS-NN v8.0.0 checkout) and a C compiler")
    root = Path(source)
    harness = tmp_path / "harness.c"
    harness.write_text(_HARNESS)
    cases = list(_grid())
    lines = "".join(
        f"{kind} {' '.join(map(str, (*inp, *filt, *out, *pad, *stride, *dil)))}\n"
        for kind, inp, filt, out, pad, stride, dil in cases)
    results = {}
    for name, defines in (("plain", []), ("dsp", ["-DARM_MATH_DSP"])):
        binary = tmp_path / name
        subprocess.run([compiler, "-std=c99", "-O1", "-w", *defines, f"-I{root / 'Include'}",
                        "-o", str(binary), str(harness),
                        *(str(root / "Source" / s) for s in _SOURCES)],
                       check=True, capture_output=True)
        out = subprocess.run([str(binary)], input=lines, capture_output=True, text=True,
                             check=True).stdout.split("\n")
        results[name] = [list(map(int, row.split())) for row in out if row]
    assert len(results["plain"]) == len(results["dsp"]) == len(cases)
    for case, plain, dsp in zip(cases, results["plain"], results["dsp"]):
        assert _port(*case) == plain, case
        assert dsp[0] == plain[1], case


# Method allocator bytes measured on an emulated Cortex-M7 (QEMU mps2-an500)
# running the ExecuTorch 1.6.0 runtime built with arm-none-eabi-gcc, per temp
# allocator: one of 64 KiB, none, an empty one, one of the minimum and one a
# byte short of it.
_MEASURED = {
    "bands_m0": (4212, 6165, 6133, 4212, 4424),
    "bands_m7": (4212, 6165, 6133, 4212, 4424),
    "bands_m55": (4220, 6174, 6142, 4220, 4432),
    "chain_m0": (1996, 2840, 2808, 1996, 2208),
    "chain_m7": (1996, 2840, 2808, 1996, 2208),
    "chain_m55": (1996, 2840, 2808, 1996, 2208),
    "resblock_m0": (2504, 3645, 3613, 2504, 3300),
    "resblock_m7": (2504, 3645, 3613, 2504, 3300),
    "resblock_m55": (2512, 3654, 3622, 2512, 3308),
    "resblock_align64_m7": (2504, 3645, 3613, 2504, 3300),
    "ops_m7": (3204, 4722, 4690, 3204, 3432),
}
_TEMP_MIN = {"bands": 448, "chain": 448, "resblock": 420, "ops": 476}


def _replay(plan: pte.ExecutionPlan) -> tuple[int, ...]:
    least = pm.temp_allocator_min_bytes(plan)
    return (pm.method_allocator_bytes(plan, 64 * 1024),
            pm.method_allocator_bytes(plan, temp_allocator=False),
            pm.method_allocator_bytes(plan, 0),
            pm.method_allocator_bytes(plan, least),
            pm.method_allocator_bytes(plan, least - 1))


@pytest.mark.parametrize("path", PTES, ids=lambda p: p.stem)
def test_method_allocator_matches_the_device(path):
    plan = pte.read(path.read_bytes()).plan()
    assert pm.temp_allocator_min_bytes(plan) == _TEMP_MIN[path.stem.split("_")[0]]
    assert _replay(plan) == _MEASURED[path.stem]
    assert pm.method_allocator_bytes(plan) == _MEASURED[path.stem][0]


def test_method_allocator_grows_with_the_program():
    """One more kernel call with a fresh output tensor costs that tensor's
    EValue, strides and TensorImpl, the call's slots and its argument list."""
    plan = _plan("chain_m7.pte")
    before = pm.method_allocator_bytes(plan)
    call = plan.chains[0].instructions[-1]
    out = plan.values[call.args[-2]]
    plan.values.append(pte.Tensor(**{**out.__dict__, "sizes": list(out.sizes)}))
    plan.chains[0].instructions.append(pte.KernelCall(call.op_index, call.args[:-2] + [
        len(plan.values) - 1] * 2))
    assert pm.method_allocator_bytes(plan) - before == 16 + 4 * 4 + 32 + 4 + 8 + 4 * 15


def test_method_allocator_refuses_what_the_runtime_does_not_load():
    for arg in (-1, 10**6):
        plan = _plan("chain_m7.pte")
        plan.chains[0].instructions[0].args[0] = arg
        with pytest.raises(ValueError, match=f"instruction 0 passes value {arg}, which does "
                           "not exist"):
            pm.method_allocator_bytes(plan)
    plan = _plan("chain_m7.pte")
    next(v for v in plan.values if isinstance(v, pte.Tensor)).sizes = [1] * 17
    with pytest.raises(ValueError, match="tensor of rank 17, above the runtime's limit of 16"):
        pm.method_allocator_bytes(plan)


def test_method_allocator_refuses_what_it_does_not_replay():
    plan = _plan("chain_m7.pte")
    plan.delegates = [pte.BackendDelegate(id="x")]
    with pytest.raises(ValueError, match="delegates"):
        pm.method_allocator_bytes(plan)
    plan = _plan("chain_m7.pte")
    plan.values[plan.inputs[0]].shape_dynamism = 1
    with pytest.raises(ValueError, match="dynamic shape"):
        pm.method_allocator_bytes(plan)
    plan = _plan("chain_m7.pte")
    tensor = next(v for v in plan.values if isinstance(v, pte.Tensor) and v.data_buffer_idx)
    tensor.extra_tensor_info = pte.ExtraTensorInfo(location=1)
    with pytest.raises(ValueError, match="outside the program"):
        pm.method_allocator_bytes(plan)


# Measured like _MEASURED on four ExecuTorch 1.6.0 programs lowered for the
# Cortex-M7 with int8 inputs: DS-CNN, ResNet-8, MobileNetV2-0.35 at 96x96 and
# MobileNetV2 at 224x224.
_PROGRAMS = {
    "dscnn": (4860, 7065, 7033, 4860, 5920),
    "r8": (5352, 8085, 8053, 5352, 7740),
    "mbv2": (24324, 37233, 37201, 24324, 27928),
    "m224": (24404, 37313, 37281, 24404, 28008),
}


@pytest.mark.parametrize("name", _PROGRAMS)
def test_method_allocator_on_larger_programs(name):
    """Needs TIGRIS_PTE_PROGRAMS, a directory with <name>/model.pte for each."""
    root = os.environ.get("TIGRIS_PTE_PROGRAMS")
    if not root:
        pytest.skip("TIGRIS_PTE_PROGRAMS is not set")
    plan = pte.read((Path(root) / name / "model.pte").read_bytes()).plan()
    assert _replay(plan) == _PROGRAMS[name]
    assert cm.infer_class(plan) == (DSP,)


def test_modules_need_neither_torch_nor_executorch():
    code = ("import sys; from tigris.frontends import pte, pte_cortex_m, pte_method; "
            f"p = pte.read(open({str(PTES[0])!r}, 'rb').read()).plan(); "
            "pte_cortex_m.infer_class(p); pte_method.method_allocator_bytes(p); "
            "print(sorted(m for m in ('torch', 'executorch', 'flatbuffers') if m in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"
