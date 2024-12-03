from optimum.exporters.onnx import main_export

main_export(
    "./tagger_output",
    opset=15,
    task="token-classification",
    output="./tagger_onnx",
)

main_export(
    "./decoder_output/best_checkpoint",
    opset=15,
    no_post_process=True,
    task="text2text-generation-with-past",
    output="./decoder_onnx",
)

