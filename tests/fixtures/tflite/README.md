`kws_ref_model.tflite` is the MLPerf Tiny keyword-spotting reference model
(github.com/mlcommons/tiny, commit 4addd0fa08d216e20637637874e084895f289da4,
benchmark/training/keyword_spotting/trained_models), Apache License 2.0.

`kws_ref_model_tflm.npz` holds eight int8 inputs drawn around the model's input
zero point and the int8 outputs TFLite Micro's reference kernels produce for
them, so the frontend can be checked bit for bit without TFLite Micro installed.

`ops/` holds one-operator models and their reference outputs, written by
`scripts/gen_tflite_fixtures.py` (needs tensorflow and tflite-micro). Each
`.npz` holds seeded inputs and the outputs TFLite Micro produces for them.
The `float_` models are converted without quantization.

Where TFLite Micro disagrees with TFLite's reference kernels, the reference
kernels' outputs are recorded instead. Two cases do: `relu6`, because TFLite
Micro's int8 RELU6 ignores the output quantization, and `space_to_batch`,
because its SPACE_TO_BATCH_ND leaves the padded positions unwritten, holding
whatever the arena held before. TFLite Micro is recorded unchecked where
TFLite has no reference kernel (CEIL, ELU, int8 CUMSUM) and for `sum_channels`
and `sum_spatial`, whose int8 SUM reference kernel does not requantize its
result in this release; TFLite Micro matches the exact sum there.

`div` keeps its numerators off 0 and -1 after the zero point, where TFLite's
int8 arithmetic shifts a 32-bit value by 32 or more; the runtime's own tests
cover that range. `squeeze_op` and `expand_dims_op` rewrite the converter's
RESHAPE into the SQUEEZE and EXPAND_DIMS operators, which the converter never
emits itself. The converter keeps int8 ELU, CUMSUM and DYNAMIC_UPDATE_SLICE in
float between DEQUANTIZEs and a QUANTIZE, so `elu`, `cumsum`,
`cumsum_exclusive_reverse` and `dynamic_update_slice` are rewritten to run the
int8 operator itself: the CUMSUM cases with the input zero point at 0, and the
update with the operand's quantization, since that kernel copies raw bytes.
`float_l2_pool` recodes an AVERAGE_POOL_2D as L2_POOL_2D, and the
`embedding_lookup` cases recode a GATHER on axis 0 as EMBEDDING_LOOKUP; the
converter emits neither.
