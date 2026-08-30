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
        src/libs/io/Json.cpp
        src/libs/io/Json.h
        src/libs/io/ModelConfig.cpp
        src/libs/io/ModelConfig.h
        src/libs/io/Safetensors.cpp
        src/libs/io/Safetensors.h
        src/libs/nn/Embedding.cpp
        src/libs/nn/Embedding.h
        src/libs/nn/Linear.cpp
        src/libs/nn/Linear.h
        src/libs/nn/RMSNorm.cpp
        src/libs/nn/RMSNorm.h
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
        ${CMAKE_CURRENT_SOURCE_DIR}/src/libs/io
        ${CMAKE_CURRENT_SOURCE_DIR}/src/libs/nn
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
            ${CMAKE_CURRENT_SOURCE_DIR}/src/libs/testing
            ${CMAKE_CURRENT_SOURCE_DIR}/src/libs/io
            ${CMAKE_CURRENT_SOURCE_DIR}/src/libs/nn
    )
    add_test(NAME ${name} COMMAND ${name})
endfunction()

set(VEDA_CORE_SOURCES
        src/libs/tensor/Shape.cpp
        src/libs/tensor/Strides.cpp
        src/libs/tensor/Tensor.cpp
        src/libs/tensor/DType.cpp)

# Development tooling: linked into tests only, never into the veda binary (E3 acceptance criterion).
set(VEDA_TESTING_SOURCES
        src/libs/testing/Compare.cpp
        src/libs/testing/TensorFile.cpp)

set(VEDA_IO_SOURCES
        src/libs/io/Json.cpp
        src/libs/io/ModelConfig.cpp
        src/libs/io/Safetensors.cpp)

set(VEDA_NN_SOURCES
        src/libs/nn/Embedding.cpp
        src/libs/nn/Linear.cpp
        src/libs/nn/RMSNorm.cpp)

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
veda_add_test(ToleranceTest ${VEDA_CORE_SOURCES} ${VEDA_OPS_SOURCES})
veda_add_test(JsonScopeTest)
veda_add_test(CompareTest ${VEDA_CORE_SOURCES} ${VEDA_OPS_SOURCES} ${VEDA_TESTING_SOURCES})
veda_add_test(TensorFileTest ${VEDA_CORE_SOURCES} ${VEDA_OPS_SOURCES} ${VEDA_TESTING_SOURCES})

# The cross-language check: the golden-file format read and written from both sides. A helper
# binary stands in for the C++ side; the script drives it. Skipped when no interpreter is found —
# building and running veda never requires Python (E3 acceptance criterion).
add_executable(tensor_file_tool tools/tensor_file_tool.cpp
        src/libs/testing/TensorFile.cpp
        ${VEDA_CORE_SOURCES})
target_include_directories(tensor_file_tool PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/libs/tensor
        ${CMAKE_CURRENT_SOURCE_DIR}/src/libs/ops
        ${CMAKE_CURRENT_SOURCE_DIR}/src/libs/testing)

find_package(Python3 COMPONENTS Interpreter)
if (Python3_Interpreter_FOUND)
    add_test(NAME CrossLanguageRoundTrip
            COMMAND ${Python3_EXECUTABLE}
                    ${CMAKE_CURRENT_SOURCE_DIR}/tests/cross_language_roundtrip.py
                    $<TARGET_FILE:tensor_file_tool>)
else ()
    message(STATUS "python3 not found: skipping the cross-language format test")
endif ()
veda_add_test(JsonTest ${VEDA_CORE_SOURCES} ${VEDA_IO_SOURCES})
veda_add_test(SafetensorsTest ${VEDA_CORE_SOURCES} ${VEDA_IO_SOURCES})
veda_add_test(ModelConfigTest ${VEDA_CORE_SOURCES} ${VEDA_IO_SOURCES})
veda_add_test(EmbeddingIdeaTest ${VEDA_CORE_SOURCES} ${VEDA_OPS_SOURCES})

# The E4 milestone: load a real model's config and weights and print what was read.
add_executable(veda_inspect tools/inspect_model.cpp ${VEDA_CORE_SOURCES} ${VEDA_IO_SOURCES})
target_include_directories(veda_inspect PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/libs/tensor
        ${CMAKE_CURRENT_SOURCE_DIR}/src/libs/io)
veda_add_test(EmbeddingTest ${VEDA_CORE_SOURCES} ${VEDA_OPS_SOURCES} ${VEDA_NN_SOURCES})
veda_add_test(LinearTest ${VEDA_CORE_SOURCES} ${VEDA_OPS_SOURCES} ${VEDA_NN_SOURCES})
veda_add_test(ProjectionIdeaTest ${VEDA_CORE_SOURCES} ${VEDA_OPS_SOURCES})
veda_add_test(RMSNormLayerTest ${VEDA_CORE_SOURCES} ${VEDA_OPS_SOURCES} ${VEDA_NN_SOURCES})
