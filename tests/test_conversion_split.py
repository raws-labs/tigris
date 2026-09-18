"""A stage that cannot tile because it fuses a layout conversion is re-cut."""

from types import SimpleNamespace

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from tigris.analysis.lifetime import compute_lifetimes
from tigris.analysis.memory import compute_memory_timeline
from tigris.analysis.partition_spatial import (
    _row_view,
    _transpose_band_extents,
    conversion_cut_points,
    partition_spatial,
)
from tigris.graph.ir import Layout
from tigris.analysis.partition_temporal import partition_temporal
from tigris.loaders import load_model


def _planned(tmp_path, nodes, shapes, budget, name="split", initializers=()):
    graph = helper.make_graph(
        nodes,
        name,
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, shapes[0])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, shapes[1])],
        initializer=list(initializers),
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)]
    )
    path = tmp_path / f"{name}.onnx"
    onnx.save(model, str(path))

    ag = load_model(path)
    ag = compute_lifetimes(ag)
    ag = compute_memory_timeline(ag, capture_live_tensors=False)
    ag = partition_temporal(ag, budget)
    return partition_spatial(ag)


def _stage_op_types(ag):
    return [[ag.ops[i].op_type for i in st.op_indices] for st in ag.stages]


# A rank-3 last-axis Softmax: the normalizer converts to linear order, reduces,
# and converts back. 1 x 4096 x 8 floats is 128 KiB per tensor.
_SOFTMAX_SHAPE = [1, 4096, 8]
_SOFTMAX_NODES = [
    helper.make_node("Softmax", ["x"], ["y"], axis=-1, name="sm1")
]


def test_an_oversized_conversion_stage_is_split_so_its_interior_tiles(tmp_path):
    ag = _planned(
        tmp_path, _SOFTMAX_NODES, (_SOFTMAX_SHAPE, _SOFTMAX_SHAPE), 32_000
    )

    assert _stage_op_types(ag) == [["Transpose"], ["Softmax"], ["Transpose"]]
    interior = ag.stages[1].tile_plan
    assert interior is not None and interior.tileable
    assert interior.num_tiles > 1


def test_a_conversion_that_fits_is_left_fused(tmp_path):
    """Splitting a stage that fits would push its interior out to slow memory."""
    ag = _planned(
        tmp_path, _SOFTMAX_NODES, (_SOFTMAX_SHAPE, _SOFTMAX_SHAPE), 4_000_000
    )

    assert _stage_op_types(ag) == [["Transpose", "Softmax", "Transpose"]]
    assert ag.stages[0].tile_plan is None


def test_an_untileable_stage_without_a_conversion_is_left_alone(tmp_path):
    """The fallback only answers the disagreement a conversion creates.

    Resize changes the height a stripe would carry and has no tiled execution
    path at all, which makes it the untileable operator that is neither a
    reduction nor a conversion.
    """
    scales = numpy_helper.from_array(
        np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32), "scales"
    )
    nodes = [
        helper.make_node("Relu", ["x"], ["h"], name="relu1"),
        helper.make_node(
            "Resize", ["h", "", "scales"], ["y"], name="resize1",
            mode="nearest", coordinate_transformation_mode="asymmetric",
            nearest_mode="floor",
        ),
    ]
    ag = _planned(
        tmp_path, nodes, ([1, 8, 32, 32], [1, 8, 64, 64]), 32_000,
        initializers=(scales,),
    )

    resize = next(st for st in ag.stages if "Resize" in
                  [ag.ops[i].op_type for i in st.op_indices])
    assert not resize.tile_plan.tileable
    assert conversion_cut_points(ag) == frozenset()


def test_an_isolated_conversion_tiles_along_its_longer_axis(tmp_path):
    """Splitting is only useful if the conversion stages tile afterwards."""
    ag = _planned(
        tmp_path, _SOFTMAX_NODES, (_SOFTMAX_SHAPE, _SOFTMAX_SHAPE), 32_000
    )

    assert _stage_op_types(ag) == [["Transpose"], ["Softmax"], ["Transpose"]]
    for stage in ag.stages:
        assert stage.tile_plan is not None and stage.tile_plan.tileable
        assert stage.tile_plan.num_tiles > 1

    # Both conversions band the 4096 axis, whichever side of the permutation
    # it sits on.
    assert ag.stages[0].tile_plan.original_height == 4096
    assert ag.stages[2].tile_plan.original_height == 4096


def test_a_conversion_whose_narrow_slice_does_not_fit_stays_untileable(tmp_path):
    """The smallest band is one column of the short axis on each side."""
    ag = _planned(
        tmp_path, _SOFTMAX_NODES, (_SOFTMAX_SHAPE, _SOFTMAX_SHAPE), 40
    )

    conversions = [
        stage for stage in ag.stages
        if [ag.ops[i].op_type for i in stage.op_indices] == ["Transpose"]
    ]
    assert conversions
    assert all(not st.tile_plan.tileable for st in conversions)


_RANK4_SHAPE = [1, 6, 16, 16]
_RANK4_NODES = [
    helper.make_node("Softmax", ["x"], ["y"], axis=-1, name="sm4")
]


def test_a_rank4_conversion_tiles_on_the_collapsed_matrix(tmp_path):
    """The two spatial axes keep their order, so H*W reads as one extent."""
    ag = _planned(
        tmp_path, _RANK4_NODES, (_RANK4_SHAPE, _RANK4_SHAPE), 8_000,
        name="rank4",
    )

    assert _stage_op_types(ag) == [["Transpose"], ["Softmax"], ["Transpose"]]
    for stage in ag.stages:
        assert stage.tile_plan is not None and stage.tile_plan.tileable
        assert stage.tile_plan.num_tiles > 1

    # 16 x 16 spatial against 6 channels: the spatial pair is the long axis.
    assert ag.stages[0].tile_plan.original_height == 256
    assert ag.stages[2].tile_plan.original_height == 256


def test_a_matrix_pipeline_bands_along_its_rows(tmp_path):
    """Rank 2 has no axis the stripe contract names, but rows are independent."""
    rows, width, hidden = 256, 32, 64
    inits = [
        numpy_helper.from_array(
            np.zeros((width, hidden), dtype=np.float32), "w1"),
        numpy_helper.from_array(
            np.zeros((hidden, width), dtype=np.float32), "w2"),
    ]
    nodes = [
        helper.make_node("MatMul", ["x", "w1"], ["h"]),
        helper.make_node("Relu", ["h"], ["a"]),
        helper.make_node("MatMul", ["a", "w2"], ["y"]),
    ]
    ag = _planned(
        tmp_path, nodes, ([1, rows, width], [1, rows, width]), 8_000,
        name="rows", initializers=inits,
    )

    banded = [
        st for st in ag.stages
        if st.tile_plan is not None and st.tile_plan.tileable
        and st.tile_plan.original_height == rows
    ]
    assert banded, _stage_op_types(ag)
    for stage in banded:
        assert stage.tile_plan.num_tiles > 1
        assert stage.tile_plan.halo == 0

    # The Reshape pair around a lowered product joins the band: it drops a unit
    # leading axis without moving a byte.
    reshapes = [
        st for st in ag.stages
        if _stage_op_types(ag)[st.stage_id] == ["Reshape"]
    ]
    assert reshapes
    assert all(st.tile_plan is None or st.tile_plan.tileable
               for st in reshapes)


def test_a_transpose_the_model_asks_for_is_banded_too(tmp_path):
    """What decides the band is the permutation, not why the transpose exists."""
    tokens, width = 256, 16
    weight = numpy_helper.from_array(
        np.zeros((width, width), dtype=np.float32), "wk")
    nodes = [
        helper.make_node("MatMul", ["x", "wk"], ["keys"]),
        helper.make_node("Transpose", ["keys"], ["y"], perm=[0, 2, 1]),
    ]
    ag = _planned(
        tmp_path, nodes, ([1, tokens, width], [1, width, tokens]), 8_000,
        name="attn", initializers=(weight,),
    )

    transposes = [
        st for st in ag.stages
        if _stage_op_types(ag)[st.stage_id] == ["Transpose"]
    ]
    assert transposes
    for stage in transposes:
        assert stage.tile_plan is not None and stage.tile_plan.tileable
        assert stage.tile_plan.original_height == tokens
        assert stage.tile_plan.num_tiles > 1


def test_band_extents_come_from_the_permutation_not_the_layout():
    """The executor's transpose_extents reads the stored permutation alone.

    Splitting the tensor by its layout instead agrees only when the
    permutation happens to run the way that layout implies. Where it runs the
    other way the two disagree, and the plan then states a band geometry the
    runtime does not perform.
    """
    # A LINEAR tensor serializes in the model's own order, so stored is the
    # shape as written. Both permutations below are ones the runtime bands.
    linear = SimpleNamespace(shape=[1, 8, 6, 6], layout=Layout.LINEAR)

    # The spatial pair is the row axis: 8 * 6 rows of 6.
    assert _transpose_band_extents(linear, (0, 3, 1, 2)) == (1, 48, 6)
    # The other way round: 8 rows of 6 * 6. This is also what the layout
    # alone would have said, which is why the disagreement stayed hidden.
    assert _transpose_band_extents(linear, (0, 2, 3, 1)) == (1, 8, 36)
    # A permutation the runtime does not band gets no extents at all.
    assert _transpose_band_extents(linear, (0, 1, 2, 3)) is None

def test_a_row_pipeline_too_tall_for_the_plan_format_declines(tmp_path):
    """The band has to be stateable in the plan, not just computable.

    tile_height, num_tiles and original_height are uint16 fields. A solver
    that bands a single spatial axis never reaches that on a shape which fits
    an embedded budget; one that bands a matrix's rows does.
    """
    rows, width, hidden = 70_000, 8, 8
    inits = [
        numpy_helper.from_array(
            np.zeros((width, hidden), dtype=np.float32), "w1"),
    ]
    nodes = [helper.make_node("MatMul", ["x", "w1"], ["y"])]
    ag = _planned(
        tmp_path, nodes, ([1, rows, width], [1, rows, hidden]), 8_000,
        name="tallrows", initializers=inits,
    )

    banded = [
        st for st in ag.stages
        if st.tile_plan is not None and st.tile_plan.tileable
        and st.tile_plan.original_height > 0xFFFF
    ]
    assert not banded
    declined = [
        st for st in ag.stages
        if st.tile_plan is not None and not st.tile_plan.tileable
        and any("a tile plan can state" in w for w in st.tile_plan.warnings)
    ]
    assert declined, [st.tile_plan for st in ag.stages]


def test_a_conversion_too_wide_for_the_plan_format_declines(tmp_path):
    """A 256x256 plane collapses to 65,536 rows, one past the field."""
    shape = [1, 4, 256, 256]
    nodes = [helper.make_node("Softmax", ["x"], ["y"], axis=-1, name="smbig")]
    ag = _planned(tmp_path, nodes, (shape, shape), 32_000, name="bigconv")

    conversions = [
        st for st in ag.stages
        if _stage_op_types(ag)[st.stage_id] == ["Transpose"]
    ]
    assert conversions
    for stage in conversions:
        assert not stage.tile_plan.tileable
        assert any("a tile plan can state" in w
                   for w in stage.tile_plan.warnings)


def test_a_batched_matrix_region_bands_over_its_rows(tmp_path):
    """The attention shape: the head axis is batch the band spans, not cuts.

    The last two axes are the matrix and everything before them is batch,
    which is what a matrix product already means by its shape. A band of query
    rows therefore means the same thing on a rank-2 projection and on a rank-4
    attention tensor, and the second operand of the product is read whole the
    way a weight is.
    """
    heads, tokens, width = 4, 64, 16
    nodes = [
        helper.make_node("Transpose", ["x"], ["kt"], perm=[0, 1, 3, 2]),
        helper.make_node("MatMul", ["x", "kt"], ["s"]),
        helper.make_node("Softmax", ["s"], ["y"], axis=-1),
    ]
    ag = _planned(
        tmp_path, nodes,
        ([1, heads, tokens, width], [1, heads, tokens, tokens]),
        65_536, name="attnband")

    banded = [
        st for st in ag.stages
        if st.tile_plan is not None and st.tile_plan.tileable
        and st.tile_plan.original_height == tokens
        and st.tile_plan.num_tiles > 1
    ]
    assert banded, _stage_op_types(ag)
    for stage in banded:
        assert stage.tile_plan.halo == 0


def test_the_row_view_reads_the_trailing_pair_as_the_matrix():
    linear4 = SimpleNamespace(shape=[1, 4, 64, 16], layout=Layout.LINEAR)
    assert _row_view(linear4) == (4, 64, 16)

    linear3 = SimpleNamespace(shape=[2, 64, 16], layout=Layout.LINEAR)
    assert _row_view(linear3) == (2, 64, 16)

    rank2 = SimpleNamespace(shape=[64, 16], layout=Layout.LINEAR)
    assert _row_view(rank2) == (1, 64, 16)

    # A spatial tensor holds its channels last, so its trailing pair is not a
    # matrix and it presents no row view at all.
    spatial = SimpleNamespace(shape=[1, 4, 64, 16], layout=Layout.SPATIAL)
    assert _row_view(spatial) is None
