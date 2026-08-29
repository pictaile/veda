cmake_minimum_required(VERSION 3.20)
project(veda)

set(CMAKE_CXX_STANDARD 23)



add_executable(veda src/main.cpp
        src/libs/tensor/Shape.cpp
        src/libs/tensor/Shape.h
        src/libs/tensor/Storage.h
        src/libs/tensor/Strides.cpp
        src/libs/tensor/Strides.h
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

enable_testing()

# Кожен тест — окремий бінарник з власним main(); харнес у tests/TestSupport.h.
function(veda_add_test name)
    add_executable(${name} tests/${name}.cpp ${ARGN})
    target_include_directories(${name} PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/tests
            ${CMAKE_CURRENT_SOURCE_DIR}/src/libs/tensor
    )
    add_test(NAME ${name} COMMAND ${name})
endfunction()

veda_add_test(ShapeTest src/libs/tensor/Shape.cpp)
veda_add_test(TensorTest src/libs/tensor/Shape.cpp src/libs/tensor/Strides.cpp src/libs/tensor/Tensor.cpp)
veda_add_test(StridesTest src/libs/tensor/Shape.cpp src/libs/tensor/Strides.cpp src/libs/tensor/Tensor.cpp)
veda_add_test(ContiguityTest src/libs/tensor/Shape.cpp src/libs/tensor/Strides.cpp)
veda_add_test(ReshapeTest src/libs/tensor/Shape.cpp src/libs/tensor/Strides.cpp src/libs/tensor/Tensor.cpp)
