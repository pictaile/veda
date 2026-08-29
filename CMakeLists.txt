cmake_minimum_required(VERSION 3.20)
project(veda)

set(CMAKE_CXX_STANDARD 23)



add_executable(veda src/main.cpp
        src/libs/tensor/BFloat16.h
        src/libs/tensor/DType.cpp
        src/libs/tensor/DType.h
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
        src/libs/ops/Activations.cpp
        src/libs/ops/Activations.h
        src/libs/ops/Broadcast.cpp
        src/libs/ops/Broadcast.h
        src/libs/ops/Elementwise.cpp
        src/libs/ops/Elementwise.h
        src/libs/ops/Matmul.cpp
        src/libs/ops/Matmul.h
        src/libs/ops/Normalize.cpp
        src/libs/ops/Normalize.h
        src/libs/ops/Softmax.cpp
        src/libs/ops/Softmax.h
        src/libs/ops/Walk.h
        src/core/facade/Facade.cpp
        src/core/facade/Facade.h)

target_include_directories(veda PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/core
        ${CMAKE_CURRENT_SOURCE_DIR}/src/libs/tensor
        ${CMAKE_CURRENT_SOURCE_DIR}/src/libs/ops
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
            ${CMAKE_CURRENT_SOURCE_DIR}/src/libs/ops
    )
    add_test(NAME ${name} COMMAND ${name})
endfunction()

set(VEDA_CORE_SOURCES
        src/libs/tensor/Shape.cpp
        src/libs/tensor/Strides.cpp
        src/libs/tensor/Tensor.cpp
        src/libs/tensor/DType.cpp)

set(VEDA_OPS_SOURCES
        src/libs/ops/Activations.cpp
        src/libs/ops/Normalize.cpp
        src/libs/ops/Broadcast.cpp
        src/libs/ops/Elementwise.cpp
        src/libs/ops/Matmul.cpp
        src/libs/ops/Softmax.cpp)

veda_add_test(ShapeTest src/libs/tensor/Shape.cpp)
veda_add_test(TensorTest ${VEDA_CORE_SOURCES})
veda_add_test(StridesTest ${VEDA_CORE_SOURCES})
veda_add_test(ContiguityTest ${VEDA_CORE_SOURCES})
veda_add_test(ReshapeTest ${VEDA_CORE_SOURCES})
veda_add_test(TransposeTest ${VEDA_CORE_SOURCES})
veda_add_test(SliceTest ${VEDA_CORE_SOURCES})
veda_add_test(FloatFormatsTest)
veda_add_test(DTypeTest ${VEDA_CORE_SOURCES})
veda_add_test(BFloat16Test ${VEDA_CORE_SOURCES})
veda_add_test(BroadcastingTest ${VEDA_CORE_SOURCES})
veda_add_test(ElementwiseTest ${VEDA_CORE_SOURCES} ${VEDA_OPS_SOURCES})
veda_add_test(MatmulIdeaTest ${VEDA_CORE_SOURCES})
veda_add_test(MatmulTest ${VEDA_CORE_SOURCES} ${VEDA_OPS_SOURCES})
veda_add_test(MatmulNtTest ${VEDA_CORE_SOURCES} ${VEDA_OPS_SOURCES})
veda_add_test(BatchedMatmulTest ${VEDA_CORE_SOURCES} ${VEDA_OPS_SOURCES})
veda_add_test(SoftmaxIdeaTest)
veda_add_test(SoftmaxTest ${VEDA_CORE_SOURCES} ${VEDA_OPS_SOURCES})
veda_add_test(ActivationsTest ${VEDA_CORE_SOURCES} ${VEDA_OPS_SOURCES})
veda_add_test(RmsNormalizeTest ${VEDA_CORE_SOURCES} ${VEDA_OPS_SOURCES})
