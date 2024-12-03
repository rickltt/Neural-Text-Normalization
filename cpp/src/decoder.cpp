#include "decoder.h"


DecoderOnnx::~DecoderOnnx()
{
}

DecoderOnnx::DecoderOnnx(const char *model_path, int nNumThread)
{
  init_tokenizer(model_path);
  LOG(INFO) << "Init Decoder Tokenizer!";

  LoadModel(model_path, nNumThread);
  LOG(INFO) << "Load Decoder Finished!";
}

void DecoderOnnx::LoadModel(const std::string& model_path, int nNumThread)
{
	sessionOptions.SetIntraOpNumThreads(nNumThread);
	sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
  sessionOptions.DisableMemPattern();
  sessionOptions.DisableCpuMemArena();

	std::string EncoderPath = model_path + "/" + "encoder_model_quant.onnx";
  std::string DncoderPath = model_path + "/" + "decoder_model_quant.onnx";
  std::string PastPath = model_path + "/" + "decoder_with_past_model_quant.onnx";

	encoder_session = std::make_unique<Ort::Session>(env, EncoderPath.c_str(), sessionOptions);
  decoder_session = std::make_unique<Ort::Session>(env, DncoderPath.c_str(), sessionOptions);
  past_session = std::make_unique<Ort::Session>(env, PastPath.c_str(), sessionOptions);
}

void DecoderOnnx:: init_tokenizer(std::string model_path){

    std::string tokenizer_path = model_path + "/spiece.model";
    const auto status = tokenizer.Load(tokenizer_path);
    if (!status.ok()) {
       std::cerr << status.ToString() << std::endl;
    }

    for (size_t i=0; i < tokenizer.GetPieceSize();i++) {
      id_to_token.insert(std::make_pair(i, tokenizer.IdToPiece(i)));
      token_to_id.insert(std::make_pair(tokenizer.IdToPiece(i),i));
    }

    std::string special_tokens_config_path = model_path + "/special_tokens_map.json";
    std::ifstream tokenizer_f(special_tokens_config_path);
    nlohmann::json tokenizer_config = nlohmann::json::parse(tokenizer_f);

    eos_token = tokenizer_config["eos_token"]["content"];
    unk_token = tokenizer_config["unk_token"]["content"];
    pad_token = tokenizer_config["pad_token"]["content"];

    std::string generation_config_path = model_path + "/generation_config.json";
    std::ifstream generation_f(generation_config_path);
    nlohmann::json generation_config = nlohmann::json::parse(generation_f);
    decoder_start_token_id = generation_config["decoder_start_token_id"];
    eos_token_id = generation_config["eos_token_id"];
    pad_token_id = generation_config["pad_token_id"];

}

std::string DecoderOnnx::inference(std::string text){

    int64_t token_index;
    std::vector<int64_t> final_ids;
    std::vector<std::string> pieces;
    std::vector<int64_t> encoder_input_ids;
    tokenizer.Encode(text, &pieces);
    int count = 0;
    for (const std::string &token : pieces)
    {
      if (count == MAX_LENGTH)
      {
        break;
      }
      if (tokenizer.IsUnknown(token_to_id[token]))
      {
        encoder_input_ids.push_back(token_to_id[unk_token]);
      }
      else
      {
        encoder_input_ids.push_back(token_to_id[token]);
      }
      count++;
    }

    encoder_input_ids.push_back(eos_token_id);

    for (auto &id : encoder_input_ids)
    {
      std::cout << id << " ";
    }
    std::cout << "\n";

    for (auto &id : encoder_input_ids)
    {
      std::cout << id_to_token[id] << " ";
    }
    std::cout << "\n";

    std::vector<int64_t> encoder_attention_mask(encoder_input_ids.size(),1);

    std::array<int64_t, 2> encoder_input_ids_shape{ 1,(int64_t)encoder_input_ids.size()};
    Ort::Value encoder_input_ids_tensor = Ort::Value::CreateTensor<int64_t>(memoryInfo, encoder_input_ids.data(), encoder_input_ids.size(),
                                                    encoder_input_ids_shape.data(), encoder_input_ids_shape.size());

    std::array<int64_t, 2> encoder_attention_mask_shape{ 1,(int64_t)encoder_attention_mask.size()};
    Ort::Value encoder_attention_mask_tensor = Ort::Value::CreateTensor<int64_t>(memoryInfo, encoder_attention_mask.data(), encoder_attention_mask.size(),
                                                    encoder_attention_mask_shape.data(), encoder_attention_mask_shape.size());

    std::vector<Ort::Value> encoder_input_onnx;
    encoder_input_onnx.emplace_back(std::move(encoder_input_ids_tensor));
    encoder_input_onnx.emplace_back(std::move(encoder_attention_mask_tensor));

    auto encoder_outputTensor = encoder_session->Run(Ort::RunOptions(nullptr),
                                                     encoder_input_names.data(),
                                                     encoder_input_onnx.data(),
                                                     encoder_input_names.size(),
                                                     encoder_output_names.data(),
                                                     encoder_output_names.size());

    float *output_data = encoder_outputTensor[0].GetTensorMutableData<float>();
    size_t output_size = encoder_outputTensor[0].GetTensorTypeAndShapeInfo().GetElementCount();
    LOG(INFO) << "output_size:" << output_size;

    std::vector<int64_t> decoder_input_ids = {decoder_start_token_id};
    final_ids.push_back(decoder_start_token_id);
    std::array<int64_t, 2> decoder_input_ids_shape{1, (int64_t)decoder_input_ids.size()};
    Ort::Value decoder_input_ids_tensor = Ort::Value::CreateTensor<int64_t>(memoryInfo, decoder_input_ids.data(), decoder_input_ids.size(),
                                                                            decoder_input_ids_shape.data(), decoder_input_ids_shape.size());
    Ort::Value decoder_attention_mask_tensor = Ort::Value::CreateTensor<int64_t>(memoryInfo, encoder_attention_mask.data(), encoder_attention_mask.size(),
                                                                                 encoder_attention_mask_shape.data(), encoder_attention_mask_shape.size());

    std::vector<Ort::Value> decoder_input_onnx;
    decoder_input_onnx.emplace_back(std::move(decoder_attention_mask_tensor));
    decoder_input_onnx.emplace_back(std::move(decoder_input_ids_tensor));
    decoder_input_onnx.emplace_back(std::move(encoder_outputTensor[0]));

    auto decoder_outputTensor = decoder_session->Run(Ort::RunOptions(nullptr),
                                                     decoder_input_names.data(),
                                                     decoder_input_onnx.data(),
                                                     decoder_input_names.size(),
                                                     decoder_output_names.data(),
                                                     decoder_output_names.size());

    float *decoder_output_data = decoder_outputTensor[0].GetTensorMutableData<float>();
    size_t decoder_output_size = decoder_outputTensor[0].GetTensorTypeAndShapeInfo().GetElementCount();
    std::vector<float> logits(decoder_output_data, decoder_output_data + decoder_output_size);
    token_index = argmax(logits.begin(), logits.end());
    // token_index = sample_top_p_with_penalty(logits, temperature, final_ids, penalty);
    final_ids.push_back(token_index);

    LOG(INFO) << "logits size: " << logits.size();
    LOG(INFO) << "next token: " << token_index;

    std::vector<int64_t> past_input_ids = {token_index};
    std::array<int64_t, 2> past_input_ids_shape{1, (int64_t)past_input_ids.size()};
    Ort::Value past_input_ids_tensor = Ort::Value::CreateTensor<int64_t>(memoryInfo, past_input_ids.data(), past_input_ids.size(),
                                                                            past_input_ids_shape.data(), past_input_ids_shape.size());
    Ort::Value past_attention_mask_tensor = Ort::Value::CreateTensor<int64_t>(memoryInfo, encoder_attention_mask.data(), encoder_attention_mask.size(),
                                                                                 encoder_attention_mask_shape.data(), encoder_attention_mask_shape.size());

    std::vector<Ort::Value> past_input_onnx;
    past_input_onnx.emplace_back(std::move(past_attention_mask_tensor));
    past_input_onnx.emplace_back(std::move(past_input_ids_tensor));
    past_input_onnx.emplace_back(std::move(decoder_input_onnx[2]));

    for (size_t i = 1; i < decoder_outputTensor.size(); i++)
    {
      past_input_onnx.emplace_back(std::move(decoder_outputTensor[i]));
    }

    int64_t *current_input_id = past_input_onnx[1].GetTensorMutableData<int64_t>();

    for (size_t i = 0; i < MAX_LENGTH; i += 1)
    {
        auto past_outputTensor = past_session->Run(Ort::RunOptions(nullptr),
                                                          past_input_names.data(),
                                                          past_input_onnx.data(),
                                                          past_input_names.size(),
                                                          past_output_names.data(),
                                                          past_output_names.size());

        float *past_output_data = past_outputTensor[0].GetTensorMutableData<float>();
        size_t past_output_size = past_outputTensor[0].GetTensorTypeAndShapeInfo().GetElementCount();

        std::vector<float> past_logits(past_output_data, past_output_data + past_output_size);

        token_index = argmax(past_logits.begin(), past_logits.end());
        // token_index = sample_top_p_with_penalty(past_logits, temperature, final_ids, penalty);
        final_ids.push_back(token_index);
        LOG(INFO) << "next token:" << token_index << ": " << id_to_token[token_index];
        current_input_id[0] = token_index;

        for (size_t j = 1; j < past_outputTensor.size(); j += 2)
        {
          past_input_onnx[2*j+1] = std::move(past_outputTensor[j]);
          past_input_onnx[2*j+2] = std::move(past_outputTensor[j + 1]);
        }

        if (token_index == eos_token_id)
        {
          break;
        }

    }
    std::string result;
    for (size_t j = 1; j < final_ids.size() - 1; j += 1)
    {
      std::string tmp = id_to_token[final_ids[j]];
      std::cout << tmp << " ";
      if (tmp == "<unk>")
      {
        continue;
      }
      result += tmp;
    }
    std::cout << "\n";

    replaceSubstring(result, "\u2581", " ");

    LOG(INFO) << result;
    return trim(result);

}
