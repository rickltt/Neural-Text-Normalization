#ifndef PROCESSOR_H
#define PROCESSOR_H

#include "decoder.h"
#include "tagger.h"


class Processor
{
public:
    Processor(const char* tagger_path, const char* decoder_path, int nNumThread);
    ~Processor();
    std::string inference(std::string text);

private:

    TaggerOnnx* tagger;
    DecoderOnnx* decoder;

};

#endif