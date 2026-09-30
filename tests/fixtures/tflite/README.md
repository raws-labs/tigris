`kws_ref_model.tflite` is the MLPerf Tiny keyword-spotting reference model
(github.com/mlcommons/tiny, commit 4addd0fa08d216e20637637874e084895f289da4,
benchmark/training/keyword_spotting/trained_models), Apache License 2.0.

`kws_ref_model_tflm.npz` holds eight int8 inputs drawn around the model's input
zero point and the int8 outputs TFLite Micro's reference kernels produce for
them, so the frontend can be checked bit for bit without TFLite Micro installed.
