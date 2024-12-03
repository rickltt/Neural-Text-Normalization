#ifndef WORDPIECE_H
#define WORDPIECE_H

#include <iostream>
#include <string>
#include <cctype> 
#include <algorithm>
#include <vector>
#include <fstream>  
#include <glog/logging.h>
#include "nlohmann/json.hpp"

class WordPieceTokenizer
{
private:

    nlohmann::json id_to_token;
    nlohmann::json token_to_id;
    size_t max_input_chars_per_word;
    std::string unk_token;
    std::string cls_token;
    std::string sep_token;

    int max_length;
    int special_token_count = 2; // cls_token, sep_token

public:
    WordPieceTokenizer(const std::string& config_path);
    ~WordPieceTokenizer();
    std::vector<std::string> run_split_on_func(const std::string& text);
    int get_word_index(const std::string& word);
    std::string to_lower(const std::string& input_text);
    std::vector<std::string> split(const std::string& input_text);
    std::vector<std::string> tokenize(const std::string& input_text, bool do_case);
    std::vector<int64_t> encode(const std::string& input_text);
    std::vector<std::string> wordpiece_tokenize(const std::string& input_text, bool do_case);
    std::vector<int64_t> convert_tokens_to_ids(const std::vector<std::string>& tokens);
    std::vector<std::string> convert_ids_to_tokens(const std::vector<int64_t>& token_ids);
};

#endif