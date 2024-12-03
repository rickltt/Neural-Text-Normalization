#ifndef DECODER_H
#define DECODER_H

#define MAX_LENGTH 32
#include <iostream>
#include <numeric>
#include <fstream>
#include <map>
#include <vector>
#include <cmath>
#include <random>
#include <regex>
#include <unordered_set>
#include <algorithm>
#include <codecvt>
#include <queue>
#include <utility>
#include <glog/logging.h>
#include <sentencepiece_processor.h>
#include <onnxruntime_cxx_api.h>
#include "nlohmann/json.hpp"
#include "utils.h"

class DecoderOnnx
{
public:
    DecoderOnnx(const char *model_path, int nNumThread);
    ~DecoderOnnx();
    std::string inference(std::string text);
    void LoadModel(const std::string &model_path, int nNumThread);
    void init_tokenizer(std::string model_path);

private:

    int decoder_start_token_id;
    int eos_token_id;
    int pad_token_id;

    std::string eos_token;
    std::string pad_token;
    std::string unk_token;

    float temperature = 0.7;
    float penalty = 1.5;

    Ort::SessionOptions sessionOptions = Ort::SessionOptions();
    std::unique_ptr<Ort::Session> encoder_session;
    std::unique_ptr<Ort::Session> decoder_session;
    std::unique_ptr<Ort::Session> past_session;

    Ort::Env env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "Decoder");
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);


    sentencepiece::SentencePieceProcessor tokenizer;
    std::map<int, std::string> id_to_token;
    std::map<std::string, int> token_to_id;

    std::vector<const char *> encoder_input_names{"input_ids", "attention_mask"};
    std::vector<const char *> encoder_output_names{"last_hidden_state"};

    std::vector<const char *> decoder_input_names{"encoder_attention_mask", "input_ids", "encoder_hidden_states"};
    std::vector<const char *> decoder_output_names{
        "logits",
        "present.0.decoder.key",
        "present.0.decoder.value",
        "present.0.encoder.key",
        "present.0.encoder.value",
        "present.1.decoder.key",
        "present.1.decoder.value",
        "present.1.encoder.key",
        "present.1.encoder.value",
        "present.2.decoder.key",
        "present.2.decoder.value",
        "present.2.encoder.key",
        "present.2.encoder.value",
        "present.3.decoder.key",
        "present.3.decoder.value",
        "present.3.encoder.key",
        "present.3.encoder.value",
        "present.4.decoder.key",
        "present.4.decoder.value",
        "present.4.encoder.key",
        "present.4.encoder.value",
        "present.5.decoder.key",
        "present.5.decoder.value",
        "present.5.encoder.key",
        "present.5.encoder.value",

    };

    std::vector<const char *> past_input_names{
        "encoder_attention_mask",
        "input_ids",
        "encoder_hidden_states",
        "past_key_values.0.decoder.key",
        "past_key_values.0.decoder.value",
        "past_key_values.0.encoder.key",
        "past_key_values.0.encoder.value",
        "past_key_values.1.decoder.key",
        "past_key_values.1.decoder.value",
        "past_key_values.1.encoder.key",
        "past_key_values.1.encoder.value",
        "past_key_values.2.decoder.key",
        "past_key_values.2.decoder.value",
        "past_key_values.2.encoder.key",
        "past_key_values.2.encoder.value",
        "past_key_values.3.decoder.key",
        "past_key_values.3.decoder.value",
        "past_key_values.3.encoder.key",
        "past_key_values.3.encoder.value",
        "past_key_values.4.decoder.key",
        "past_key_values.4.decoder.value",
        "past_key_values.4.encoder.key",
        "past_key_values.4.encoder.value",
        "past_key_values.5.decoder.key",
        "past_key_values.5.decoder.value",
        "past_key_values.5.encoder.key",
        "past_key_values.5.encoder.value",
    };

    std::vector<const char *> past_output_names{
        "logits",
        "present.0.decoder.key",
        "present.0.decoder.value",
        "present.1.decoder.key",
        "present.1.decoder.value",
        "present.2.decoder.key",
        "present.2.decoder.value",
        "present.3.decoder.key",
        "present.3.decoder.value",
        "present.4.decoder.key",
        "present.4.decoder.value",
        "present.5.decoder.key",
        "present.5.decoder.value",
    };
};

#endif