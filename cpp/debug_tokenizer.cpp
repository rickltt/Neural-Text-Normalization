#include "wordpiece.h"

int main(){

    WordPieceTokenizer tokenizer("../onnx/tagger_onnx");
    // std::string text = "Today is 25 December 2014, I want to have exciting dinner 425 times.";
    std::string text = "   This no-cost   digital event features real-world strategies for app modernization, cloud compliance, and leveraging AI.   ";
    // std::vector<std::string> output_tokens = tokenizer.wordpiece_tokenize(text);
    
    // for(const auto& token: output_tokens){
    //     LOG(INFO) << token;
    // }

    std::vector<std::string> tokens = tokenizer.tokenize(text, true);

    for(const auto& token: tokens){
        LOG(INFO) << token;
    }

    // std::vector<std::string> tokens = tokenizer.convert_ids_to_tokens(token_ids);

    // for(const auto& token: tokens){
    //     LOG(INFO) << token;
    // }

    // std::vector<int64_t> ids = tokenizer.convert_tokens_to_ids(tokens);

    // for(const auto& id: ids){
    //     LOG(INFO) << id;
    // }


    return 0;
}