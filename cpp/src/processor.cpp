#include "processor.h"


Processor::Processor(const char* tagger_path, const char* decoder_path, int nNumThread)
{

    std::string TaggerPath(tagger_path);
    std::string DecoderPath(decoder_path);

    tagger = new TaggerOnnx(TaggerPath.c_str(), nNumThread);
    decoder = new DecoderOnnx(DecoderPath.c_str(), nNumThread);

    LOG(INFO) << "Load Processor";
}


Processor::~Processor()
{
    if (tagger)
    {
        delete tagger;
        tagger = nullptr;
    }

    if (decoder)
    {
        delete decoder;
        decoder = nullptr;
    }

}


std::string Processor::inference(std::string text)
{
    // std::string tagger_reuslt = tagger->inference(text);
    // LOG(INFO) << tagger_reuslt;

    // nlohmann::json tagger_json = nlohmann::json::parse(tagger_reuslt);
    // std::string decoder_input;
    // std::string decoder_output;
    // for (auto& obj : tagger_json.items()) {
        
    //     LOG(INFO) << "key: " << obj.key() << ", value: " << obj.value().template get<std::string>();

    //     decoder_input = "Normalize " + obj.key() + ": \n" + obj.value().template get<std::string>();
    //     LOG(INFO) << decoder_input;
    //     decoder_output = decoder->inference(decoder_input);
    //     LOG(INFO) << decoder_output;
    // }

    nlohmann::json json_result;

    LOG(INFO) << text;
    std::vector<std::string> tokens = tagger->tokenizer->tokenize(text, true);

    for(const auto & token: tokens){
        std::cout << token << " ";
    }
    std::cout << "\n";

    std::string decoder_input;
    std::string decoder_output;

    std::vector<std::string> tagger_reuslt = tagger->inference(text);
    std::vector<std::string> decoder_reuslt;
    for (const auto &obj: tagger_reuslt){
        nlohmann::json tagger_json = nlohmann::json::parse(obj);
        LOG(INFO) << "word: " << tagger_json["word"] << ", label: " << tagger_json["label"];

        std::string word = tagger_json["label"].template get<std::string>();
        std::string label = tagger_json["word"].template get<std::string>();
        // int start = tagger_json["start"];
        // int end = tagger_json["end"];
        std::transform(word.begin(), word.end(), word.begin(), [](unsigned char c) { return std::tolower(c); });
        decoder_input = "normalize " + word + ": " + label;
        LOG(INFO) << decoder_input;
        decoder_output = decoder->inference(decoder_input);
        LOG(INFO) << decoder_output;
        tagger_json["decoder_output"] = decoder_output;
        decoder_reuslt.push_back(tagger_json.dump());
    }

    // 使用匿名函数（lambda）进行升序排序
    std::sort(decoder_reuslt.begin(), decoder_reuslt.end(), [](std::string a, std::string b) {
        return nlohmann::json::parse(a)["start"] < nlohmann::json::parse(b)["start"];  // 按升序排列
    });

    int old_size = tokens.size();
    int idx_change = 0;
    nlohmann::json details;;
    
    for (const auto &obj: decoder_reuslt){
        nlohmann::json decoder_json = nlohmann::json::parse(obj);
        LOG(INFO) << "word: " << decoder_json["word"] << ", decoder_output: " << decoder_json["decoder_output"];
        std::string word = decoder_json["word"].template get<std::string>();
        std::string decoder_output = decoder_json["decoder_output"].template get<std::string>();
        details[word] = decoder_output;
        int start = decoder_json["start"].template get<int>() + idx_change;
        int end = decoder_json["end"].template get<int>() + idx_change;
        tokens.erase(tokens.begin() + start, tokens.begin() + end);
        tokens.insert(tokens.begin() + start, decoder_json["decoder_output"]);
        idx_change = tokens.size() - old_size;
    }

    for(const auto & token: tokens){
        std::cout << token << " ";
    }
    std::cout << "\n";

    std::string final_result;

    for(const auto & token: tokens){

        if (token.find("##") != std::string::npos) {
            final_result += token.substr(2);
        }
        else if (std::ispunct(static_cast<unsigned char>(token[0]))){
            final_result += token;
        }
        else{
            if (final_result.empty()){
                final_result += token;
            }else{
                final_result += " " + token;
            }
        }
    }
    json_result["input"] = text;
    json_result["output"] = final_result;
    json_result["details"] = details;
    return json_result.dump(4); 
}