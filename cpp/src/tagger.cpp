#include "tagger.h"

TaggerOnnx::TaggerOnnx(const char *model_path, int nNumThread)
{
    labels.push_back("O");
    const char* label_list[NUM_LABEL] = { LABEL_LIST };
    for (int i = 0; i < NUM_LABEL; ++i){
        std::string label_name = label_list[i];
        labels.push_back("B-"+label_name);
        labels.push_back("I-"+label_name);
    }
    for (int i = 0; i < labels.size(); ++i) {
        label2id.insert(std::pair<std::string, int64_t>(labels[i], i));
        id2label.insert(std::pair<int64_t, std::string>(i,labels[i]));
    }
    init_tokenizer(model_path);
    load_model(model_path, nNumThread);
}

TaggerOnnx::~TaggerOnnx()
{

    if (m_session)
    {
        delete m_session;
        m_session = nullptr;
    }

    if(tokenizer){
        delete tokenizer;
        tokenizer = nullptr;
    }
}

void TaggerOnnx::load_model(const std::string &model_path, int nNumThread)
{
    sessionOptions.SetIntraOpNumThreads(nNumThread);
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    sessionOptions.DisableMemPattern();
    sessionOptions.DisableCpuMemArena();

    std::string strModelPath = model_path + "/" + "model_quant.onnx";

    m_session = new Ort::Session(env, strModelPath.c_str(), sessionOptions);
    LOG(INFO) << "Load Tagger Model!";
}


void TaggerOnnx::init_tokenizer(std::string model_path)
{

    tokenizer = new WordPieceTokenizer(model_path);
    LOG(INFO) << "Load Tagger Vocab";
}

std::vector<std::string> TaggerOnnx::inference(std::string text)
{
    std::vector<std::string> tokens = tokenizer->tokenize(text, false);

    std::vector<int64_t> input_ids = tokenizer->encode(text);

    for(const auto & id: input_ids){
        std::cout << id << " ";
    }
    std::cout << "\n";

    std::vector<int64_t> attention_mask(input_ids.size(),1);
    std::vector<int64_t> token_type_ids(input_ids.size(),0);


    std::array<int64_t, 2> input_ids_shape{1, (int64_t)input_ids.size()};
    Ort::Value input_ids_tensor = Ort::Value::CreateTensor<int64_t>(memoryInfo, input_ids.data(), input_ids.size(),
                                                                    input_ids_shape.data(), input_ids_shape.size());

    std::array<int64_t, 2> attention_mask_shape{1, (int64_t)attention_mask.size()};
    Ort::Value attention_mask_tensor = Ort::Value::CreateTensor<int64_t>(memoryInfo, attention_mask.data(), attention_mask.size(),
                                                                         attention_mask_shape.data(), attention_mask_shape.size());

    std::array<int64_t, 2> token_type_ids_shape{1, (int64_t)token_type_ids.size()};
    Ort::Value token_type_ids_tensor = Ort::Value::CreateTensor<int64_t>(memoryInfo, token_type_ids.data(), token_type_ids.size(),
                                                                         token_type_ids_shape.data(), token_type_ids_shape.size());
    std::vector<Ort::Value> input_onnx;
    input_onnx.emplace_back(std::move(input_ids_tensor));
    input_onnx.emplace_back(std::move(attention_mask_tensor));
    input_onnx.emplace_back(std::move(token_type_ids_tensor));

    auto outputTensor = m_session->Run(Ort::RunOptions(), input_names.data(), input_onnx.data(), input_names.size(), output_names.data(), output_names.size());
    std::vector<int64_t> outputShape = outputTensor[0].GetTensorTypeAndShapeInfo().GetShape();
    float *logits = outputTensor[0].GetTensorMutableData<float>();

    int64_t outputCount = std::accumulate(outputShape.begin(), outputShape.end(), 1, std::multiplies<int64_t>());
    std::vector<int64_t> predictions;
    int num_labels = 2 * NUM_LABEL + 1;
    for (size_t i = 0; i < outputCount; i += num_labels)
    {
      int64_t index = argmax(logits + i, logits + i + num_labels - 1);
      predictions.push_back(index);
    }

    predictions.assign(predictions.begin()+1, predictions.begin() + std::count(attention_mask.begin(), attention_mask.end(), 1) -1 );
    
    std::vector<std::string> bioTags;
    // for(auto &pred: predictions){
    //     bioTags.push_back(id2label[pred]);
    //     LOG(INFO) << pred << " " << id2label[pred];
    // }
    for (size_t i = 0; i < predictions.size(); ++i){
        bioTags.push_back(id2label[predictions[i]]);
        LOG(INFO) << tokens[i] << " " << id2label[predictions[i]];
    }
    // 转换 BIO 到实体
    std::vector<Entity> entities = bioToEntities(bioTags, tokens);

    std::vector<std::string> result;
    nlohmann::json json_result;
    // 输出结果
    for (const auto& entity : entities) {
        std::vector<std::string> words = tokenizer->split(entity.word);
        std::string word;
        std::string pre_word;
        // 使用范围 for 循环遍历
        for (const auto& w : words) {
            if (!pre_word.empty() && std::ispunct(static_cast<unsigned char>(pre_word.back()))){
                word += w;
            }else{
                if (word.empty()){
                    word += w;
                }else{
                    word += " " + w;
                }
            }
            pre_word = w;
        }
        // json_result["word"] = entity.word;
        json_result["word"] = word;
        json_result["label"] = entity.label;
        json_result["start"] = entity.start;
        json_result["end"] = entity.end;
        // json_result[entity.label] = entity.word;
        std::cout << "实体: " << entity.word << ", 标签: " << entity.label << "(" << entity.start << "," << entity.end << ")" << std::endl;
        result.push_back(json_result.dump());
        json_result.clear();
    }
    return result;

}