#include <iostream>
#include "tagger.h"
#include "processor.h"
using namespace std;


int main()
{
  // string text = "Today is 29 December 2024, I want to have dinner 425 times.";
  // string text = "The total cost at the start of construction in 2009 was estimated to be €340,000,000.";
  // string text = "In 2013, the total housing area in Kazakhstan amounted to 336.1 million square meters.";
  string text = "In June 2011, Liquid Robotics received $22M investor funding from VantagePoint Capital Partners and Schlumberger, ltd.";
  // TaggerOnnx tagger("../onnx/tagger_onnx",1);
  // tagger.inference(text);
  Processor processor("../onnx/tagger_onnx","../onnx/decoder_onnx", 1);

  std::string result = processor.inference(text);
  std::cout << "result: " << result << std::endl;
  return 0;

}