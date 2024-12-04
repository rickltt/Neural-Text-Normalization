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

FILE(GLOB_RECURSE SOURCE_FILES ${CMAKE_SOURCE_DIR}/src/*.cpp)

SET(ALL_SRCS ${SOURCE_FILES})

add_library(text_norm SHARED ${ALL_SRCS})
target_link_libraries(text_norm PUBLIC sentencepiece onnxruntime glog::glog)

add_executable(debug debug.cpp)

target_link_libraries(debug text_norm)


add_executable(tokenizer debug_tokenizer.cpp)

target_link_libraries(tokenizer text_norm)
