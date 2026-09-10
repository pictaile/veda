#!/usr/bin/env python3
"""Dumps reference activations from the real model, for veda to compare against.

    python3 tools/dump_reference.py --model <path> --prompt "The capital of France is" --out reference/

Why this exists (architecture AD7): hand-computed 2x2 tests catch algebra errors. They do not catch
a transposed weight, a swapped head layout or a wrong RoPE base — those compute without error and
return confident nonsense. From E6 onward, "is this right?" is answered by comparing intermediate
tensors against these files.

Development tooling only. It requires torch and transformers; Veda itself requires neither, and no
CMake target references this script. The dumps are large and are not committed — regenerate them.

What is written, per prompt:

    embeddings.bin              after the embedding lookup
    layer_00_output.bin ...     after each transformer block   <- the backbone of any bisection
    final_norm.bin              after the final norm
    logits_last.bin             the last position only  ([1, T, V] would be ~600 MB)
"""

import argparse
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from veda_tensor import write_torch


def main():
    parser = argparse.ArgumentParser(description="dump Qwen3 reference activations")
    parser.add_argument("--model", required=True, help="path to the model directory")
    parser.add_argument("--prompt", required=True, help="the prompt to run, fixed per dump")
    parser.add_argument("--out", required=True, help="directory to write the .bin files into")
    arguments = parser.parse_args()

    os.makedirs(arguments.out, exist_ok=True)

    # Determinism: a fixed prompt, no sampling, evaluation mode, a fixed seed. The same model and
    # prompt must produce byte-identical files, or a comparison against them means nothing.
    torch.manual_seed(0)

    tokenizer = AutoTokenizer.from_pretrained(arguments.model)
    model = AutoModelForCausalLM.from_pretrained(arguments.model, torch_dtype=torch.float32)
    model.eval()

    captured = {}

    def capture(name):
        def hook(_module, _inputs, output):
            # A block returns a tuple; the hidden state is its first element.
            captured[name] = output[0] if isinstance(output, tuple) else output

        return hook

    # Forward hooks are the least invasive capture mechanism: the model is not modified, and they
    # fire in execution order.
    handles = [model.model.embed_tokens.register_forward_hook(capture("embeddings"))]
    for layer_index, block in enumerate(model.model.layers):
        handles.append(block.register_forward_hook(capture(f"layer_{layer_index:02d}_output")))
    handles.append(model.model.norm.register_forward_hook(capture("final_norm")))

    tokens = tokenizer(arguments.prompt, return_tensors="pt")
    with torch.inference_mode():
        outputs = model(**tokens)

    for handle in handles:
        handle.remove()

    # The dump is fp32 regardless of how the weights are stored: the comparison target is Veda's
    # compute dtype (AD1), not the file's.
    captured["logits_last"] = outputs.logits[:, -1, :]

    for name, tensor in captured.items():
        path = os.path.join(arguments.out, f"{name}.bin")
        write_torch(path, tensor)
        print(f"writing {path}  {tuple(tensor.shape)}")

    print(f"\n{len(captured)} tensors written for prompt: {arguments.prompt!r}")
    print(f"tokens: {tokens['input_ids'].tolist()[0]}")


if __name__ == "__main__":
    main()
