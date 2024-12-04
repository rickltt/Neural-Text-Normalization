import onnx
import os
from onnxruntime.quantization import quantize_dynamic, QuantType

def quantize(model_path, model_name):
    model_fp32 = os.path.join(model_path,f"{model_name}.onnx")
    model_quant = os.path.join(model_path,f"{model_name}_quant.onnx")

    quantized_model = quantize_dynamic(
        model_input=model_fp32,  # 输入模型路径
        model_output=model_quant,  # 输出量化模型路径
        weight_type=QuantType.QUInt8  # 量化类型
    )
    os.remove(model_fp32)

if __name__ == '__main__':
    quantize("./cpp/onnx/tagger_onnx","model")
    quantize("./cpp/onnx/decoder_onnx","encoder_model")
    quantize("./cpp/onnx/decoder_onnx","decoder_model")
    quantize("./cpp/onnx/decoder_onnx","decoder_with_past_model")

