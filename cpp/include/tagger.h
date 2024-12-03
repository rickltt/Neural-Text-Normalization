#ifndef TAGGER_H
#define TAGGER_H

#define NUM_LABEL 11
#define LABEL_LIST "DATE", "CARDINAL", "DECIMAL","MEASURE", "MONEY", "ORDINAL", "TIME", "DIGIT", "FRACTION", "TELEPHONE", "ADDRESS"
#include <iostream>
#include <numeric>
#include <fstream>
#include <map>
#include <algorithm>
#include <glog/logging.h>
#include "onnxruntime_cxx_api.h"
#include "nlohmann/json.hpp"
#include "wordpiece.h"
#include "utils.h"

class TaggerOnnx
{
public:
    TaggerOnnx(const char *model_path, int nNumThread);
    ~TaggerOnnx();
    std::vector<std::string> inference(std::string text);
    void init_tokenizer(std::string model_path);
    void load_model(const std::string &model_path, int nNumThread);

    // tokenizer
    WordPieceTokenizer *tokenizer;
private:

    Ort::Session *m_session;
    Ort::SessionOptions sessionOptions = Ort::SessionOptions();
    Ort::Env env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "TestTagger");
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    

    std::vector<std::string> labels;
    std::map<std::string, int64_t> label2id;
    std::map<int64_t, std::string> id2label;

    std::vector<const char *> input_names{"input_ids", "attention_mask", "token_type_ids"};
    std::vector<const char *> output_names{"logits"};
};

#endif