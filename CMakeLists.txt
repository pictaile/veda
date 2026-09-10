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
        src/libs/autograd/GradientCheck.cpp
        src/libs/autograd/GradientCheck.h
        src/libs/autograd/NonLinear.cpp
        src/libs/autograd/Loss.cpp
        src/libs/autograd/Loss.h
        src/libs/autograd/Variable.cpp
        src/libs/autograd/Variable.h
        src/libs/autograd/Views.cpp
        src/libs/optim/AdamW.cpp
        src/libs/optim/AdamW.h
        src/libs/optim/Clip.cpp
        src/libs/optim/Clip.h
        src/libs/data/Corpus.cpp
        src/libs/data/Corpus.h
        src/libs/train/Model.cpp
        src/libs/train/Model.h
        src/libs/tokenizer/ByteLevel.cpp
        src/libs/tokenizer/ByteLevel.h
        src/libs/binding/Binding.cpp
        src/libs/binding/Binding.h
        src/libs/io/Json.cpp
        src/libs/io/Json.h
        src/libs/generate/Generator.cpp
        src/libs/generate/Generator.h
        src/libs/profile/Profile.cpp
        src/libs/profile/Profile.h
        src/libs/generate/Sampling.cpp
        src/libs/generate/Sampling.h
        src/libs/tokenizer/Tokenizer.cpp
        src/libs/tokenizer/Tokenizer.h
        src/libs/tokenizer/Vocabulary.cpp
        src/libs/tokenizer/Vocabulary.h
        src/libs/io/ModelConfig.cpp
        src/libs/io/ModelConfig.h
        src/libs/io/Safetensors.cpp
        src/libs/io/Safetensors.h
        src/libs/model/Attention.cpp
        src/libs/model/Attention.h
        src/libs/model/FeedForward.cpp
        src/libs/model/FeedForward.h
        src/libs/model/KVCache.cpp
        src/libs/model/KVCache.h
        src/libs/model/LMHead.cpp
        src/libs/model/LMHead.h
        src/libs/model/Rope.cpp
        src/libs/model/Rope.h
        src/libs/model/Transformer.cpp
        src/libs/model/Transformer.h
        src/libs/model/TransformerBlock.cpp
        src/libs/model/TransformerBlock.h
        src/libs/model/ScaledDotProductAttention.cpp
        src/libs/model/ScaledDotProductAttention.h
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
        ${CMAKE_CURRENT_SOURCE_DIR}/src/libs/model
        ${CMAKE_CURRENT_SOURCE_DIR}/src/libs/binding
        ${CMAKE_CURRENT_SOURCE_DIR}/src/libs/tokenizer
        ${CMAKE_CURRENT_SOURCE_DIR}/src/libs/generate
        ${CMAKE_CURRENT_SOURCE_DIR}/src/libs/profile
        ${CMAKE_CURRENT_SOURCE_DIR}/src/libs/autograd
        ${CMAKE_CURRENT_SOURCE_DIR}/src/libs/optim
        ${CMAKE_CURRENT_SOURCE_DIR}/src/libs/data
        ${CMAKE_CURRENT_SOURCE_DIR}/src/libs/train
        ${CMAKE_CURRENT_SOURCE_DIR}/src/libs/weights_loader
        ${CMAKE_CURRENT_SOURCE_DIR}/src/libs/byte_pair_encoder
        ${CMAKE_CURRENT_SOURCE_DIR}/src/libs/sampler
        ${CMAKE_CURRENT_SOURCE_DIR}/src/core/facade
)

find_package(Threads REQUIRED)

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
            ${CMAKE_CURRENT_SOURCE_DIR}/src/libs/model
            ${CMAKE_CURRENT_SOURCE_DIR}/src/libs/binding
            ${CMAKE_CURRENT_SOURCE_DIR}/src/libs/tokenizer
            ${CMAKE_CURRENT_SOURCE_DIR}/src/libs/generate
            ${CMAKE_CURRENT_SOURCE_DIR}/src/libs/profile
            ${CMAKE_CURRENT_SOURCE_DIR}/src/libs/autograd
            ${CMAKE_CURRENT_SOURCE_DIR}/src/libs/optim
            ${CMAKE_CURRENT_SOURCE_DIR}/src/libs/data
            ${CMAKE_CURRENT_SOURCE_DIR}/src/libs/train
    )
    target_link_libraries(${name} PRIVATE Threads::Threads)
    add_test(NAME ${name} COMMAND ${name})
endfunction()

set(VEDA_CORE_SOURCES
        src/libs/profile/Profile.cpp
        src/libs/tensor/Shape.cpp
        src/libs/tensor/Strides.cpp
        src/libs/tensor/Tensor.cpp
        src/libs/tensor/DType.cpp)

# Development tooling: linked into tests only, never into the veda binary (E3 acceptance criterion).
set(VEDA_TESTING_SOURCES
        src/libs/testing/Compare.cpp
        src/libs/testing/TensorFile.cpp)

# Samplers depend on core only; the generation loop also needs the model, so tests that exercise
# sampling alone link the smaller list.
set(VEDA_SAMPLING_SOURCES
        src/libs/generate/Sampling.cpp)

set(VEDA_GENERATE_SOURCES
        ${VEDA_SAMPLING_SOURCES}
        src/libs/generate/Generator.cpp)

set(VEDA_TOKENIZER_SOURCES
        src/libs/tokenizer/Tokenizer.cpp
        src/libs/tokenizer/ByteLevel.cpp
        src/libs/tokenizer/Vocabulary.cpp)

# Training (Phase 1.5): the tape over the ops E2 built.
set(VEDA_AUTOGRAD_SOURCES
        src/libs/autograd/GradientCheck.cpp
        src/libs/autograd/NonLinear.cpp
        src/libs/autograd/Loss.cpp
        src/libs/autograd/Variable.cpp
        src/libs/autograd/Views.cpp)

set(VEDA_TRAIN_SOURCES
        src/libs/train/Model.cpp)

set(VEDA_DATA_SOURCES
        src/libs/data/Corpus.cpp)

set(VEDA_OPTIM_SOURCES
        src/libs/optim/AdamW.cpp
        src/libs/optim/Clip.cpp)

set(VEDA_BINDING_SOURCES
        src/libs/binding/Binding.cpp)

set(VEDA_IO_SOURCES
        src/libs/io/Json.cpp
        src/libs/io/ModelConfig.cpp
        src/libs/io/Safetensors.cpp)

set(VEDA_MODEL_SOURCES
        src/libs/model/Attention.cpp
        src/libs/model/FeedForward.cpp
        src/libs/model/KVCache.cpp
        src/libs/model/LMHead.cpp
        src/libs/model/Rope.cpp
        src/libs/model/Transformer.cpp
        src/libs/model/TransformerBlock.cpp
        src/libs/model/ScaledDotProductAttention.cpp)

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
        ${CMAKE_CURRENT_SOURCE_DIR}/src/libs/profile
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

# The E10 milestone: load a model, run token ids through it, print the next token.
add_executable(veda_run tools/run_model.cpp
        ${VEDA_CORE_SOURCES} ${VEDA_OPS_SOURCES} ${VEDA_NN_SOURCES}
        ${VEDA_MODEL_SOURCES} ${VEDA_IO_SOURCES} ${VEDA_BINDING_SOURCES}
        ${VEDA_TOKENIZER_SOURCES} ${VEDA_GENERATE_SOURCES})
target_link_libraries(veda_run PRIVATE Threads::Threads)
target_include_directories(veda_run PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/libs/profile
        ${CMAKE_CURRENT_SOURCE_DIR}/src/libs/tensor
        ${CMAKE_CURRENT_SOURCE_DIR}/src/libs/ops
        ${CMAKE_CURRENT_SOURCE_DIR}/src/libs/nn
        ${CMAKE_CURRENT_SOURCE_DIR}/src/libs/model
        ${CMAKE_CURRENT_SOURCE_DIR}/src/libs/io
        ${CMAKE_CURRENT_SOURCE_DIR}/src/libs/binding
        ${CMAKE_CURRENT_SOURCE_DIR}/src/libs/tokenizer
        ${CMAKE_CURRENT_SOURCE_DIR}/src/libs/generate)

# The E13 milestone: where the time goes, and what an optimisation bought.
add_executable(veda_bench tools/bench.cpp
        ${VEDA_CORE_SOURCES} ${VEDA_OPS_SOURCES} ${VEDA_NN_SOURCES}
        ${VEDA_MODEL_SOURCES} ${VEDA_IO_SOURCES} ${VEDA_BINDING_SOURCES}
        ${VEDA_GENERATE_SOURCES})
target_include_directories(veda_bench PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/libs/profile
        ${CMAKE_CURRENT_SOURCE_DIR}/src/libs/tensor
        ${CMAKE_CURRENT_SOURCE_DIR}/src/libs/ops
        ${CMAKE_CURRENT_SOURCE_DIR}/src/libs/nn
        ${CMAKE_CURRENT_SOURCE_DIR}/src/libs/model
        ${CMAKE_CURRENT_SOURCE_DIR}/src/libs/io
        ${CMAKE_CURRENT_SOURCE_DIR}/src/libs/binding
        ${CMAKE_CURRENT_SOURCE_DIR}/src/libs/generate)
target_link_libraries(veda_bench PRIVATE Threads::Threads)

# The E4 milestone: load a real model's config and weights and print what was read.
add_executable(veda_inspect tools/inspect_model.cpp ${VEDA_CORE_SOURCES} ${VEDA_IO_SOURCES})
target_include_directories(veda_inspect PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/libs/profile
        ${CMAKE_CURRENT_SOURCE_DIR}/src/libs/tensor
        ${CMAKE_CURRENT_SOURCE_DIR}/src/libs/io)
veda_add_test(EmbeddingTest ${VEDA_CORE_SOURCES} ${VEDA_OPS_SOURCES} ${VEDA_NN_SOURCES})
veda_add_test(LinearTest ${VEDA_CORE_SOURCES} ${VEDA_OPS_SOURCES} ${VEDA_NN_SOURCES})
veda_add_test(ProjectionIdeaTest ${VEDA_CORE_SOURCES} ${VEDA_OPS_SOURCES})
veda_add_test(RMSNormLayerTest ${VEDA_CORE_SOURCES} ${VEDA_OPS_SOURCES} ${VEDA_NN_SOURCES})
veda_add_test(AttentionIdeaTest ${VEDA_CORE_SOURCES} ${VEDA_OPS_SOURCES})
veda_add_test(HeadSplitTest ${VEDA_CORE_SOURCES} ${VEDA_OPS_SOURCES} ${VEDA_NN_SOURCES} ${VEDA_MODEL_SOURCES})
veda_add_test(ScalingTest ${VEDA_CORE_SOURCES} ${VEDA_OPS_SOURCES} ${VEDA_NN_SOURCES} ${VEDA_MODEL_SOURCES})
veda_add_test(CausalMaskTest ${VEDA_CORE_SOURCES} ${VEDA_OPS_SOURCES} ${VEDA_NN_SOURCES} ${VEDA_MODEL_SOURCES})
veda_add_test(AttentionOutputTest ${VEDA_CORE_SOURCES} ${VEDA_OPS_SOURCES} ${VEDA_NN_SOURCES} ${VEDA_MODEL_SOURCES})
veda_add_test(AttentionLayerTest ${VEDA_CORE_SOURCES} ${VEDA_OPS_SOURCES} ${VEDA_NN_SOURCES} ${VEDA_MODEL_SOURCES})
veda_add_test(PositionIdeaTest ${VEDA_CORE_SOURCES} ${VEDA_OPS_SOURCES} ${VEDA_NN_SOURCES} ${VEDA_MODEL_SOURCES})
veda_add_test(RopeTest ${VEDA_CORE_SOURCES} ${VEDA_OPS_SOURCES} ${VEDA_NN_SOURCES} ${VEDA_MODEL_SOURCES})
veda_add_test(QkNormTest ${VEDA_CORE_SOURCES} ${VEDA_OPS_SOURCES} ${VEDA_NN_SOURCES} ${VEDA_MODEL_SOURCES} ${VEDA_TESTING_SOURCES})
veda_add_test(GqaTest ${VEDA_CORE_SOURCES} ${VEDA_OPS_SOURCES} ${VEDA_NN_SOURCES} ${VEDA_MODEL_SOURCES})
veda_add_test(FeedForwardTest ${VEDA_CORE_SOURCES} ${VEDA_OPS_SOURCES} ${VEDA_NN_SOURCES} ${VEDA_MODEL_SOURCES})
veda_add_test(TransformerBlockTest ${VEDA_CORE_SOURCES} ${VEDA_OPS_SOURCES} ${VEDA_NN_SOURCES} ${VEDA_MODEL_SOURCES})
veda_add_test(BindingTest ${VEDA_CORE_SOURCES} ${VEDA_OPS_SOURCES} ${VEDA_NN_SOURCES} ${VEDA_MODEL_SOURCES} ${VEDA_IO_SOURCES} ${VEDA_BINDING_SOURCES})
veda_add_test(TransformerTest ${VEDA_CORE_SOURCES} ${VEDA_OPS_SOURCES} ${VEDA_NN_SOURCES} ${VEDA_MODEL_SOURCES} ${VEDA_IO_SOURCES} ${VEDA_BINDING_SOURCES} ${VEDA_TESTING_SOURCES})
veda_add_test(LMHeadTest ${VEDA_CORE_SOURCES} ${VEDA_OPS_SOURCES} ${VEDA_NN_SOURCES} ${VEDA_MODEL_SOURCES} ${VEDA_IO_SOURCES} ${VEDA_BINDING_SOURCES} ${VEDA_TESTING_SOURCES})
veda_add_test(VocabularyTest ${VEDA_IO_SOURCES} ${VEDA_CORE_SOURCES} ${VEDA_TOKENIZER_SOURCES})
veda_add_test(EncodeTest ${VEDA_IO_SOURCES} ${VEDA_CORE_SOURCES} ${VEDA_TOKENIZER_SOURCES})
veda_add_test(DecodeTest ${VEDA_IO_SOURCES} ${VEDA_CORE_SOURCES} ${VEDA_TOKENIZER_SOURCES})
veda_add_test(SamplerTest ${VEDA_CORE_SOURCES} ${VEDA_OPS_SOURCES} ${VEDA_SAMPLING_SOURCES})
veda_add_test(SamplingTest ${VEDA_CORE_SOURCES} ${VEDA_OPS_SOURCES} ${VEDA_SAMPLING_SOURCES})
veda_add_test(GeneratorTest ${VEDA_CORE_SOURCES} ${VEDA_OPS_SOURCES} ${VEDA_NN_SOURCES} ${VEDA_MODEL_SOURCES} ${VEDA_IO_SOURCES} ${VEDA_BINDING_SOURCES} ${VEDA_GENERATE_SOURCES})
veda_add_test(KVCacheTest ${VEDA_CORE_SOURCES} ${VEDA_OPS_SOURCES} ${VEDA_NN_SOURCES} ${VEDA_MODEL_SOURCES} ${VEDA_IO_SOURCES} ${VEDA_BINDING_SOURCES} ${VEDA_GENERATE_SOURCES})
veda_add_test(PerformanceTest ${VEDA_CORE_SOURCES} ${VEDA_OPS_SOURCES})
veda_add_test(AutogradTest ${VEDA_CORE_SOURCES} ${VEDA_OPS_SOURCES} ${VEDA_AUTOGRAD_SOURCES})
veda_add_test(AutogradMatmulTest ${VEDA_CORE_SOURCES} ${VEDA_OPS_SOURCES} ${VEDA_AUTOGRAD_SOURCES})
veda_add_test(CrossEntropyTest ${VEDA_CORE_SOURCES} ${VEDA_OPS_SOURCES} ${VEDA_AUTOGRAD_SOURCES})
veda_add_test(AdamWTest ${VEDA_CORE_SOURCES} ${VEDA_OPS_SOURCES} ${VEDA_AUTOGRAD_SOURCES} ${VEDA_OPTIM_SOURCES})
veda_add_test(GradientClipTest ${VEDA_CORE_SOURCES} ${VEDA_OPS_SOURCES} ${VEDA_AUTOGRAD_SOURCES} ${VEDA_OPTIM_SOURCES})
veda_add_test(TrainingStepTest ${VEDA_CORE_SOURCES} ${VEDA_OPS_SOURCES} ${VEDA_AUTOGRAD_SOURCES} ${VEDA_OPTIM_SOURCES})
veda_add_test(AutogradViewsTest ${VEDA_CORE_SOURCES} ${VEDA_OPS_SOURCES} ${VEDA_AUTOGRAD_SOURCES} src/libs/model/Rope.cpp)
veda_add_test(TinyModelTest ${VEDA_CORE_SOURCES} ${VEDA_OPS_SOURCES} ${VEDA_IO_SOURCES} ${VEDA_AUTOGRAD_SOURCES} ${VEDA_OPTIM_SOURCES} ${VEDA_TRAIN_SOURCES} src/libs/model/Rope.cpp)
veda_add_test(CorpusTest ${VEDA_CORE_SOURCES} ${VEDA_OPS_SOURCES} ${VEDA_IO_SOURCES} ${VEDA_TOKENIZER_SOURCES} ${VEDA_DATA_SOURCES})
veda_add_test(AutogradNonLinearTest ${VEDA_CORE_SOURCES} ${VEDA_OPS_SOURCES} ${VEDA_AUTOGRAD_SOURCES})
