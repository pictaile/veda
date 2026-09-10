#ifndef VEDA_BYTE_LEVEL_H
#define VEDA_BYTE_LEVEL_H

#include "Tokenizer.h"

namespace veda::tokenizer
{

// A tokenizer that needs no training: the 256 byte values, and no merges at all.
//
// E5 built a BPE encoder, not a BPE trainer — learning merges from a corpus is a different program,
// and training a small model from scratch (E16, target A) does not need one. Bytes are enough: the
// vocabulary is exactly 256 entries, every string in every language encodes, and nothing is ever
// out of vocabulary.
//
// What it costs is sequence length. ASCII is one token per character and Cyrillic is two, so a
// Ukrainian sentence is twice the length of its English translation and the model has to learn
// correspondingly longer dependencies (risk R2). That is a fair price for a first working run.
//
// The ids are the byte values themselves: id 97 is 'a'. That makes a corpus readable in a debugger
// and keeps the mapping to the printable-code-point form of E5 exactly as it was.
Tokenizer byte_tokenizer();

} // namespace veda::tokenizer

#endif //VEDA_BYTE_LEVEL_H
