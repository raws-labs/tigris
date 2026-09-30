`kws_ref_model.tflite` is the MLPerf Tiny keyword-spotting reference model
(github.com/mlcommons/tiny, commit 4addd0fa08d216e20637637874e084895f289da4,
benchmark/training/keyword_spotting/trained_models), Apache License 2.0.

`kws_ref_model_tflm.npz` holds eight int8 inputs drawn around the model's input
zero point and the int8 outputs TFLite Micro's reference kernels produce for
them, so the frontend can be checked bit for bit without TFLite Micro installed.

`ops/` holds one-operator models and their reference outputs, written by
`scripts/gen_tflite_fixtures.py` (needs tensorflow and tflite-micro). Each
`.npz` holds seeded inputs and the outputs TFLite Micro produces for them.
Where TFLite Micro disagrees with TFLite's reference kernels, the reference
kernels' outputs are recorded instead; `relu6` is the one such case, because
TFLite Micro's int8 RELU6 ignores the output quantization.
The `float_` models are converted without quantization.
`div` keeps its numerators off 0 and -1 after the zero point, where TFLite's
int8 arithmetic shifts a 32-bit value by 32 or more; the runtime's own tests
cover that range.
`squeeze_op` and `expand_dims_op` rewrite the converter's RESHAPE into the SQUEEZE
and EXPAND_DIMS operators, which the converter never emits itself.
