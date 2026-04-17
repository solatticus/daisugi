"""Export trained model for GGUF conversion.

Hard-snaps all ternary weights, saves as a HuggingFace model directory.
The output is a standard safetensors checkpoint with exact ternary values
({-scale, 0, +scale} per block of 256) that llama.cpp's converter handles:

    python convert_hf_to_gguf.py ./output/hf --outtype tq2_0

Stock llama.cpp loads the resulting GGUF — no fork needed.
"""

import os
import torch
from ternary import TernaryLinear, hard_snap, BLOCK_SIZE


def pack(model, tokenizer, output_dir):
    """Hard-snap weights and save as HuggingFace model."""

    print("pack: snapping weights to exact ternary")
    hard_snap(model)

    # verify
    bad = 0
    for name, mod in model.named_modules():
        if not isinstance(mod, TernaryLinear):
            continue
        w = mod.weight.data.float()
        numel = w.numel()
        remainder = numel % BLOCK_SIZE
        if remainder:
            flat = torch.nn.functional.pad(w.reshape(-1), (0, BLOCK_SIZE - remainder))
        else:
            flat = w.reshape(-1)
        g = flat.view(-1, BLOCK_SIZE)
        scale = g.abs().amax(dim=1, keepdim=True)
        normed = g / scale.clamp(min=1e-8)
        rounded = normed.round().clamp(-1, 1)
        drift = (normed - rounded).abs().max().item()
        if drift > 1e-4:
            print(f"  {name}: max drift {drift:.6f}")
            bad += 1

    if bad:
        print(f"pack: WARNING — {bad} layers with drift after snap")

    # restore nn.Linear for save_pretrained compatibility
    _restore_linear(model)

    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)

    param_bytes = sum(p.nelement() * p.element_size() for p in model.parameters())
    print(f"pack: saved HF model to {output_dir} ({param_bytes / 1e9:.2f} GB)")
    print(f"pack: convert with:")
    print(f"  python convert_hf_to_gguf.py {output_dir} --outtype tq2_0")


def _restore_linear(model):
    """Replace TernaryLinear back to nn.Linear for save_pretrained.

    The weight data is already hard-snapped — this just changes the module type
    so HuggingFace serialization works without custom class registration.
    """
    for name, parent in _parents_with(model, TernaryLinear):
        for attr, child in list(parent.named_children()):
            if not isinstance(child, TernaryLinear):
                continue
            linear = torch.nn.Linear(
                child.in_features, child.out_features,
                bias=child.bias is not None,
                dtype=child.weight.dtype,
                device=child.weight.device,
            )
            linear.weight.data = child.weight.data
            if child.bias is not None:
                linear.bias.data = child.bias.data
            setattr(parent, attr, linear)


def _parents_with(model, cls):
    for name, mod in model.named_modules():
        if any(isinstance(c, cls) for c in mod.children()):
            yield name, mod
