#include "wordpiece.h"



WordPieceTokenizer::~WordPieceTokenizer()
{
}

WordPieceTokenizer::WordPieceTokenizer(const std::string& model_path)
{
    std::string config_path = model_path + "/tokenizer.json";
    std::ifstream config_f(config_path);
    nlohmann::json vocab = nlohmann::json::parse(config_f);
    token_to_id = vocab["model"]["vocab"];

    for (const auto &el: token_to_id.items()){
        // std::cout << el.key() << " : " << el.value() << "\n";
        std::string k = el.key();
        size_t v = el.value();
        id_to_token[v] = k;
    }

    max_input_chars_per_word = vocab["model"]["max_input_chars_per_word"];

    std::string map_path = model_path + "/special_tokens_map.json";
    std::ifstream map_f(map_path);
    nlohmann::json special_tokens_map = nlohmann::json::parse(map_f);
    unk_token = special_tokens_map["unk_token"]["content"];
    cls_token = special_tokens_map["cls_token"]["content"];
    sep_token = special_tokens_map["sep_token"]["content"];


    std::string tokenizer_config_path = model_path + "/tokenizer_config.json";
    std::ifstream tokenizer_config_f(tokenizer_config_path);
    nlohmann::json tokenize_config = nlohmann::json::parse(tokenizer_config_f);
    max_length = tokenize_config["model_max_length"];

    LOG(INFO) << "Init WordPieceTokenizer!";

}

// 将输入的字符串 input 按空格分割成单词
std::vector<std::string> WordPieceTokenizer::split(const std::string& input_text) {
    std::stringstream stream(input_text);
    std::vector<std::string> words;
    std::string word;
    while (stream >> word) {
        words.push_back(word);
    }
    return words;
}

int WordPieceTokenizer::get_word_index(const std::string& word)
{
    if(token_to_id.contains(word)) {
        //cout << "Found word. Id: " << vocab[word] << endl;
        return token_to_id[word];
    }else{
        //cout << "Not found" << endl;
        return -1;
    }
}

std::string WordPieceTokenizer::to_lower(const std::string& input_text)
{
    std::string lower_text = input_text;
    std::transform(lower_text.begin(), lower_text.end(), lower_text.begin(), [](unsigned char c) {
            return std::tolower(c);
        });
    return lower_text;
}



std::vector<std::string> WordPieceTokenizer::wordpiece_tokenize(const std::string& input_text, bool do_case)
{

    std::vector<std::string> tokens = split(input_text);
    std::vector<std::string> output_tokens;
    for(size_t i = 0; i < tokens.size(); i++) {
        auto& tok = tokens[i];
        if(tok.length() > max_input_chars_per_word) {
            output_tokens.push_back(unk_token);
            continue;
        }

        bool is_bad = false;
        size_t start = 0;
        std::vector<std::string> sub_tokens;
        size_t idx;
        while(start < tok.length()) {
            size_t end = tok.length();
            std::string cur_substr;
            while(start < end) {
                std::string substr = tok.substr(start, end-start);
                if(start > 0) {
                    substr = "##" + substr;
                }

                // size_t idx = get_word_index(substr);
                if (do_case){
                    idx = get_word_index(to_lower(substr));
                }else{
                    std::transform(substr.begin(), substr.end(), substr.begin(), [](unsigned char c) {
                        return std::tolower(c);
                    });
                    idx = get_word_index(substr);
                }
                if(idx != -1) {
                    cur_substr = substr;
                    break;
                }
                end--;
            }

            if(cur_substr.empty()) {
                is_bad = true;
                break;
            }
            sub_tokens.push_back(cur_substr);
            start = end;
        }

        if(is_bad) {
            output_tokens.push_back(unk_token);
        }else{
            output_tokens.insert(output_tokens.end(), sub_tokens.begin(), sub_tokens.end());
        }
    }
    return output_tokens;
}

std::vector<std::string> WordPieceTokenizer::run_split_on_func(const std::string& text)
{
    size_t i = 0;
    bool start_new_word = true;
    std::vector<std::vector<unsigned char>> output;

    while(i < text.length()) {
        unsigned char c = text[i];
        if (std::ispunct(static_cast<unsigned char>(c))) {
            std::vector<unsigned char> s;
            s.push_back(c);
            output.push_back(s);
            start_new_word = true;
        }else{
            if(start_new_word) {
                std::vector<unsigned char> empty_str;
                output.push_back(empty_str);
            }
            start_new_word = false;
            output.back().push_back(c);
        }
        i++;
    }

    std::vector<std::string> out_str;
    for (size_t i = 0; i < output.size(); i++) {
        std::string s(output[i].begin(), output[i].end());
        out_str.push_back(s);
    }
    return out_str;
}


std::vector<std::string> WordPieceTokenizer::tokenize(const std::string& input_text, bool do_case)
{
    std::vector<std::string> tokens = split(input_text);
    std::vector<std::string> basic_tokenized;
    for(size_t i = 0; i < tokens.size(); i++) {
        auto splitted_by_punc = run_split_on_func(tokens[i]);
        basic_tokenized.insert(basic_tokenized.end(), splitted_by_punc.begin(), splitted_by_punc.end());
    }

    std::vector<std::string> wordpiece_tokenized;
    for(std::string token: basic_tokenized) {

        // std::transform(token.begin(), token.end(), token.begin(), [](unsigned char c) {
        //     return std::tolower(c);
        // });
        auto splitted_by_wordpiece = wordpiece_tokenize(token, do_case);
        
        wordpiece_tokenized.insert(wordpiece_tokenized.end(), splitted_by_wordpiece.begin(), splitted_by_wordpiece.end());
    }

    return wordpiece_tokenized;
}


std::vector<int64_t> WordPieceTokenizer::encode(const std::string& input_text)
{

    std::vector<std::string> _tokens = tokenize(input_text, false);

    std::vector<std::string> tokens;
    int count = 0;
    for (const std::string &token : _tokens)
    {
      if (count == max_length - special_token_count )
      {
        break;
      }
        tokens.push_back(token);
      count++;
    }

    std::vector<int64_t> token_ids;

    token_ids.push_back(get_word_index(cls_token));
    std::vector<int64_t> seq_ids = convert_tokens_to_ids(tokens);
    token_ids.insert(token_ids.end(), seq_ids.begin(), seq_ids.end());
    token_ids.push_back(get_word_index(sep_token));
    return token_ids;
}

std::vector<int64_t> WordPieceTokenizer::convert_tokens_to_ids(const std::vector<std::string>& tokens)
{
    std::vector<int64_t> token_ids;
    for(size_t i = 0; i < tokens.size(); i++) {
        token_ids.push_back(get_word_index(tokens[i]));
    }
    return token_ids;
}

std::vector<std::string> WordPieceTokenizer::convert_ids_to_tokens(const std::vector<int64_t>& token_ids)
{
    std::vector<std::string> tokens;
    for(size_t i = 0; i < token_ids.size(); i++) {
        tokens.push_back(id_to_token[token_ids[i]]);
    }
    return tokens;
}