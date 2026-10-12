"""Non-adjacent means preserve each output's reduction sequence."""
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest

from tigris.analysis.validation import validate_budget
from tigris.cli import _run_pipeline
from tigris.emitters.binary.reader import read_binary_plan
from tigris.emitters.binary.writer import emit_binary_bytes
from tigris.frontends import tflite
from tigris.runtime import Session

FIXTURES = Path(__file__).parent / "fixtures/tflite/ops"


@pytest.mark.parametrize("name", [f"{prefix}mean_nonadjacent_{suffix}"
                                  for prefix in ("", "float_")
                                  for suffix in ("tokens", "keep", "image", "negative")])
def test_nonadjacent_mean_matches_micro_and_onnx(name, tmp_path):
    source = FIXTURES / f"{name}.tflite"
    golden = np.load(source.with_suffix('.npz'))
    assert tflite.unsupported(source.read_bytes()) == []
    model = tflite.to_onnx(source.read_bytes(), name)
    onnx.checker.check_model(model)
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    oracle = ort.InferenceSession(model.SerializeToString(), options, providers=['CPUExecutionProvider'])
    previous = None
    tight = ('2K' if name.endswith('keep') else '1536') if name.startswith('float_') and not name.endswith('tokens') else '512'
    for budget in (tight, '1M'):
        ag, _ = _run_pipeline(str(source), (budget, '8M'), report_bindings=False)
        assert validate_budget(ag).feasible
        assert any(op.op_type == 'ReduceMean' for op in ag.ops)
        binary = emit_binary_bytes(ag)
        if budget == tight:
            assert any(t['tileable'] and t['num_tiles'] > 1 for t in read_binary_plan(binary)['tile_plans'])
        plan = tmp_path / f'{budget}.tgrs'
        plan.write_bytes(binary)
        values = []
        with Session(plan, slow_capacity=8 * 1024 * 1024) as session:
            for x, target in zip(golden['input_0'], golden['output_0']):
                result = session.run({session.inputs[0]['name']: x})[session.outputs[0]['name']]
                np.testing.assert_array_equal(result, target)
                values.append(result)
                # The imported ONNX graph lists its feature maps channels-first.
                native = x.transpose(0, x.ndim - 1, *range(1, x.ndim - 1))
                expected = oracle.run(None, {oracle.get_inputs()[0].name: native})[0]
                if expected.ndim in (3, 4):
                    expected = expected.transpose(0, *range(2, expected.ndim), 1)
                if result.dtype == np.float32:
                    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)
        if previous is not None:
            np.testing.assert_array_equal(values, previous)
        previous = values
