from pprint import pprint
import onnxruntime
def printinfo(onnx_session):
    print("----------------- 输入部分 -----------------")
    input_tensors = onnx_session.get_inputs()  # 该 API 会返回列表
    for input_tensor in input_tensors:         # 因为可能有多个输入，所以为列表

        input_info = {
            "name" : input_tensor.name,
            "type" : input_tensor.type,
            "shape": input_tensor.shape,
        }
        pprint(input_info)

    print("----------------- 输出部分 -----------------")
    output_tensors = onnx_session.get_outputs()  # 该 API 会返回列表
    for output_tensor in output_tensors:         # 因为可能有多个输出，所以为列表

        output_info = {
            "name" : output_tensor.name,
            "type" : output_tensor.type,
            "shape": output_tensor.shape,
        }
        pprint(output_info)
ort_session_encoder = onnxruntime.InferenceSession("./decoder_onnx/encoder_model.onnx")
ort_session_decoder = onnxruntime.InferenceSession("./decoder_onnx/decoder_model.onnx")
ort_session_past = onnxruntime.InferenceSession("./decoder_onnx/decoder_with_past_model.onnx")
printinfo(ort_session_encoder)
printinfo(ort_session_decoder)

printinfo(ort_session_past)