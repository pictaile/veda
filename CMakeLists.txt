cmake_minimum_required(VERSION 3.20)
project(veda)

set(CMAKE_CXX_STANDARD 23)



add_executable(veda src/main.cpp
        src/libs/tensor/Tensor.cpp
        src/libs/tensor/Tensor.h
        src/libs/weights_loader/WeightsLoader.cpp
        src/libs/weights_loader/WeightsLoader.h
        src/libs/byte_pair_encoder/BytePairEncoding.cpp
        src/libs/byte_pair_encoder/BytePairEncoding.h
        src/libs/sampler/Sampler.cpp
        src/libs/sampler/Sampler.h
        src/core/facade/Facade.cpp
        src/core/facade/Facade.h)

target_include_directories(veda PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/core
        ${CMAKE_CURRENT_SOURCE_DIR}/src/libs/tensor
        ${CMAKE_CURRENT_SOURCE_DIR}/src/libs/weights_loader
        ${CMAKE_CURRENT_SOURCE_DIR}/src/libs/byte_pair_encoder
        ${CMAKE_CURRENT_SOURCE_DIR}/src/libs/sampler
        ${CMAKE_CURRENT_SOURCE_DIR}/src/core/facade
)
