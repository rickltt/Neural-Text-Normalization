cmake_minimum_required(VERSION 3.18)

project(TextNormalization)

set(ONNXRUNTIME_ROOTDIR ${CMAKE_SOURCE_DIR}/onnxruntime-linux-x64-1.19.0)
include_directories(${ONNXRUNTIME_ROOTDIR}/include)
link_directories(${ONNXRUNTIME_ROOTDIR}/lib)

set(SENTENCEPIECE_ROOTDIR ${CMAKE_SOURCE_DIR}/sentencepiece-0.2.0/build/root)
include_directories(${SENTENCEPIECE_ROOTDIR}/include)
link_directories(${SENTENCEPIECE_ROOTDIR}/lib)


find_package(glog 0.7.1 REQUIRED)

include_directories(include)

add_library(text_norm SHARED src/wordpiece.cpp src/utils.cpp src/tagger.cpp src/decoder.cpp src/processor.cpp)
target_link_libraries(text_norm PUBLIC sentencepiece onnxruntime glog::glog)

add_executable(debug src/debug.cpp)

target_link_libraries(debug text_norm)


add_executable(tokenizer src/debug_tokenizer.cpp)

target_link_libraries(tokenizer text_norm)
