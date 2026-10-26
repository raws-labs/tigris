"""Gather reads constant tables without placing them in the activation arena."""
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper, numpy_helper

from tigris.analysis.validation import validate_budget
from tigris.cli import _run_pipeline
from tigris.emitters.binary.reader import read_binary_plan
from tigris.emitters.binary.writer import emit_binary_bytes
from tigris.frontends import tflite
from tigris.runtime import RuntimeError, Session

from conftest import save_onnx

FIXTURES = Path(__file__).parent / "fixtures/tflite/ops"


@pytest.mark.parametrize('kind,index_type', [('Gather', TensorProto.INT32),
                                           ('Gather', TensorProto.INT64), ('GatherND', TensorProto.INT64)])
@pytest.mark.usefixtures("require_host_library")
def test_constant_gather_matches_onnx_with_bounded_bands(kind, index_type, tmp_path):
    shape = (32, 16) if kind == 'Gather' else (6, 6, 16)
    data = np.random.default_rng(15).normal(size=shape).astype(np.float32)
    indices = np.arange(31).reshape(1, 31) % 32 if kind == 'Gather' else np.arange(62).reshape(1, 31, 2) % 6
    indices = indices.astype(np.int32 if index_type == TensorProto.INT32 else np.int64)
    indices.flat[0] = -1
    out_shape = [1, 31, 16]
    model = helper.make_model(helper.make_graph([helper.make_node(kind, ['table', 'indices'], ['output'])],
        'table', [helper.make_tensor_value_info('indices', index_type, indices.shape)],
        [helper.make_tensor_value_info('output', TensorProto.FLOAT, out_shape)],
        [numpy_helper.from_array(data, 'table')]), opset_imports=[helper.make_opsetid('', 17)], ir_version=8)
    source = tmp_path / 'table.onnx'
    onnx.save(model, source)
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    expected = ort.InferenceSession(model.SerializeToString(), options, providers=['CPUExecutionProvider']).run(None, {'indices': indices})[0]
    previous = None
    for budget in ('256', '1M'):
        ag, _ = _run_pipeline(str(source), (budget, '8M'), report_bindings=False)
        assert validate_budget(ag).feasible
        binary = emit_binary_bytes(ag)
        plan = tmp_path / f'{budget}.tgrs'
        plan.write_bytes(binary)
        decoded = read_binary_plan(binary)
        assert all(t['name'] != 'table' for t in decoded['tensors'])
        if budget == '256':
            assert any(t['tileable'] and t['num_tiles'] > 1 for t in decoded['tile_plans'])
        with Session(plan, slow_capacity=8 * 1024 * 1024) as session:
            result = session.run({'indices': indices})['output']
            if result.shape != expected.shape:
                result = result.transpose(0, 2, 1)
            np.testing.assert_array_equal(result, expected)
            oversized = indices.astype(np.int64)
            oversized.flat[0] = 1 << 40
            with pytest.raises(RuntimeError):
                session.run({'indices': oversized})
        if previous is not None:
            np.testing.assert_array_equal(result, previous)
        previous = result


@pytest.mark.parametrize('name', ['embedding_table', 'gather_nd_table', 'float_embedding_table',
                                  'float_gather_nd_table', 'float_embedding_table_cast', 'float_embedding_table_int64'])
@pytest.mark.usefixtures("require_host_library")
def test_constant_table_matches_micro(name, tmp_path):
    source = FIXTURES / f'{name}.tflite'
    assert not tflite.unsupported(source.read_bytes())
    golden = np.load(source.with_suffix('.npz'))
    for budget in ('256', '1M'):
        ag, _ = _run_pipeline(str(source), (budget, '8M'), report_bindings=False)
        assert validate_budget(ag).feasible
        binary = emit_binary_bytes(ag)
        plan = tmp_path / f'{budget}.tgrs'
        plan.write_bytes(binary)
        if budget == '256':
            assert any(t['tileable'] and t['num_tiles'] > 1 for t in read_binary_plan(binary)['tile_plans'])
        with Session(plan, slow_capacity=8 * 1024 * 1024) as session:
            for x, expected in zip(golden['input_0'], golden['output_0']):
                actual = session.run({session.inputs[0]['name']: x})[session.outputs[0]['name']]
                np.testing.assert_array_equal(actual, expected)


def test_int64_tflite_input_is_narrowed_only_for_an_index_chain(monkeypatch):
    data = (FIXTURES / 'float_embedding_table.tflite').read_bytes()
    original = tflite._read
    limits = []

    def edited(data):
        model, graphs = original(data)
        entering = graphs[0][1][graphs[0][2][0]]
        entering.type = 'INT64'
        entering.maximum = limits
        return model, graphs

    monkeypatch.setattr(tflite, '_read', edited)
    assert not tflite.unsupported(data)
    assert tflite.to_onnx(data, 'indices').graph.input[0].type.tensor_type.elem_type == TensorProto.INT64
    limits.append(float(1 << 40))
    assert any('INT64' in reason for reason in tflite.unsupported(data))


@pytest.mark.parametrize('kind', ['Gather', 'GatherND'])
@pytest.mark.usefixtures("require_host_library")
def test_constant_indices_and_data_feed_a_runtime_consumer(kind, tmp_path):
    shape = (6, 16) if kind == 'Gather' else (6, 6, 16)
    table = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) * 0.125
    indices = np.array([1, 3], np.int64) if kind == 'Gather' else np.array([[1, 2], [3, 5]], np.int64)
    source = save_onnx(tmp_path, "constant_indices",
                       [helper.make_node(kind, ['table', 'indices'], ['selected']),
                        helper.make_node('Add', ['selected', 'input'], ['output'])],
                       [helper.make_tensor_value_info('input', TensorProto.FLOAT, [2, 16])],
                       [helper.make_tensor_value_info('output', TensorProto.FLOAT, [2, 16])],
                       [numpy_helper.from_array(table, 'table'), numpy_helper.from_array(indices, 'indices')],
                       opset=17, ir_version=8, filename="constants.onnx")
    values = np.arange(32, dtype=np.float32).reshape(2, 16) * 0.25
    expected = table[indices] if kind == 'Gather' else table[tuple(indices.T)]
    expected = expected + values
    ag, _ = _run_pipeline(str(source), ('256', '8M'), report_bindings=False)
    assert validate_budget(ag).feasible
    plan = tmp_path / 'constants.tgrs'
    plan.write_bytes(emit_binary_bytes(ag))
    with Session(plan, slow_capacity=8 * 1024 * 1024) as session:
        np.testing.assert_array_equal(session.run({'input': values})['output'], expected)
