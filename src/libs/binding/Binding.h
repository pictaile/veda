#ifndef VEDA_BINDING_H
#define VEDA_BINDING_H

#include "ModelConfig.h"
#include "Safetensors.h"
#include "Transformer.h"

// The only module allowed to know both a file's naming convention and the model's structure.
//
// io produces a name-to-tensor map and knows nothing about transformers; model has objects with
// weight-shaped holes and knows nothing about files. Something has to say that
// model.layers.7.self_attn.q_proj.weight is the query projection of block 7, and the architecture
// puts that somewhere sideways in the layering (§1) on purpose: it is what keeps "do not couple
// file parsing with Transformer classes" enforceable rather than aspirational.
namespace veda::binding
{

// Builds every layer from views into the loader's buffer (AD2) — binding a 2.4 GB model allocates
// nothing but the layer objects.
//
// Names are built from the config and asked for, never scanned from the file: asking is what makes
// a missing tensor an error rather than a default. Both directions are checked — every name the
// model needs must exist, and every tensor in the file must have been taken. The second half is the
// one people skip, and it is what catches a misspelled name whose real weight was silently ignored.
//
// Shapes are validated against the config, so a file and a config that disagree say so at startup
// rather than as a puzzling matmul error inside layer 7.
//
// Throws std::runtime_error for a missing name, a shape mismatch, or an unconsumed tensor.
model::Transformer bind(const io::WeightFile& weights, const io::ModelConfig& config);

} // namespace veda::binding

#endif //VEDA_BINDING_H
