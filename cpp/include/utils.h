#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <vector>
#include <string>
#include <regex>
#include <algorithm>


struct Entity {
    std::string word;
    std::string label;
    int start;
    int end;
};

template<typename T>
void printVector(std::vector<T> &v)
{

    for (typename std::vector<T>::iterator it = v.begin(); it != v.end(); it++)
    {
        std::cout << *it << " ";
    }
    std::cout << std::endl;
}

template <class ForwardIterator>
inline size_t argmax(ForwardIterator first, ForwardIterator last)
{
    return std::distance(first, std::max_element(first, last));
}

std::vector<Entity> bioToEntities(const std::vector<std::string>& bioTags, const std::vector<std::string>& words);


std::string ltrim(std::string str);
std::string rtrim(std::string str);
std::string trim(std::string str);

void replaceSubstring(std::string &str, const std::string &toReplace, const std::string &to);

#endif