#include "utils.h"

std::vector<Entity> bioToEntities(const std::vector<std::string>& bioTags, const std::vector<std::string>& words) {
    std::vector<Entity> entities;
    Entity currentEntity;
    
    for (size_t i = 0; i < bioTags.size(); ++i) {
        const std::string& tag = bioTags[i];

        if (tag.substr(0, 2) == "B-") {  // 开始实体
            if (!currentEntity.word.empty()) {
                entities.push_back(currentEntity);
            }
            currentEntity.word = words[i];
            currentEntity.label = tag.substr(2);
            currentEntity.start = i;
        } else if (tag.substr(0, 2) == "I-") {  // 中间实体
            if (!currentEntity.word.empty()) {
                if (words[i].find("##") != std::string::npos){
                    std::string word = words[i].substr(2);
                    currentEntity.word += word;
                }
                else if (std::ispunct(static_cast<unsigned char>(words[i][0]))){
                    currentEntity.word += words[i];
                }else{
                    currentEntity.word += " " + words[i];
                }
            }
        } else {  // O 或不属于当前实体
            if (!currentEntity.word.empty()) {
                currentEntity.end = i;
                entities.push_back(currentEntity);
                currentEntity = Entity();  // 重置当前实体
            }
        }
    }
    
    // 添加最后的实体
    if (!currentEntity.word.empty()) {
        entities.push_back(currentEntity);
    }

    return entities;
}


std::string ltrim(std::string str)
{
    return regex_replace(str, std::regex("^\\s+"), std::string(""));
}

std::string rtrim(std::string str)
{
    return regex_replace(str, std::regex("\\s+$"), std::string(""));
}

std::string trim(std::string str)
{
    return ltrim(rtrim(str));

}

// 函数：替换字符串中的子字符串为空字符串
void replaceSubstring(std::string &str, const std::string &toReplace, const std::string &to)
{
  std::size_t pos = str.find(toReplace);
  // 循环查找并替换所有出现的子字符串
  while (pos != std::string::npos)
  {
    str.replace(pos, toReplace.length(), to); // 替换该子字符串为空字符串
    pos = str.find(toReplace, pos);           // 继续查找下一个出现的位置
  }
}