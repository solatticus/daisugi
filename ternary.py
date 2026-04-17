"""Ternary quantization-aware training core.

Group-wise ternary quantization with straight-through estimation for QAT
fine-tuning. Each weight snaps to {-1, 0, +1} * per-block scale during
forward. Gradients flow through the snap via STE (identity).

Weight format: w_i = s_b * t_i,  t_i in {-1, 0, +1}
Block size:    256 (matches GGML TQ2_0: one FP16 scale per 256 weights)
Scale:         absmax per block (matches TQ2_0 quantize_row_tq2_0_ref)
Effective:     ~2.0625 bits/weight in TQ2_0 packing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

BLOCK_SIZE = 256  # QK_K — matches GGML TQ2_0 block size


class TernarySnap(Function):
    """Snap weights to ternary with STE backward.

    Forward: group weights into blocks of BLOCK_SIZE, compute per-block absmax
    scale, normalize to [-1, 1], round to {-1, 0, +1} at threshold 0.5,
    rescale. Matches GGML TQ2_0 quantization exactly.

    Backward: identity — grad passes through unchanged. Nothing saved to ctx
    because STE needs no forward state.
    """

    @staticmethod
    def forward(ctx, weight, threshold):
        shape = weight.shape
        numel = weight.numel()
        remainder = numel % BLOCK_SIZE

        if remainder:
            w = F.pad(weight.reshape(-1), (0, BLOCK_SIZE - remainder))
        else:
            w = weight.reshape(-1)

        g = w.view(-1, BLOCK_SIZE)                                  # (B, 256)
        scale = g.abs().amax(dim=1, keepdim=True)                   # (B, 1)
        normed = g / scale.clamp(min=1e-8)                          # (B, 256)
        ternary = normed.round().clamp(-1, 1)                       # {-1,0,+1}
        out = (ternary * scale).reshape(-1)[:numel].view(shape)

        return out

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


def ternary_snap(weight, threshold=0.5):
    return TernarySnap.apply(weight, threshold)


class TernaryLinear(nn.Module):
    """Drop-in Linear with ternary QAT.

    Latent weights live in param dtype for optimizer. Forward pass produces
    ternary-quantized effective weights. No extra buffers allocated per call —
    all intermediates are transient and hit the CUDA allocator cache after the
    first step.
    """

    __constants__ = ["in_features", "out_features"]

    def __init__(self, in_features, out_features, bias=True,
                 dtype=torch.bfloat16, threshold=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.threshold = threshold
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=dtype))
        self.bias = (nn.Parameter(torch.zeros(out_features, dtype=dtype))
                     if bias else None)

    def forward(self, x):
        return F.linear(x, ternary_snap(self.weight, self.threshold), self.bias)

    @classmethod
    def from_linear(cls, linear, threshold=0.5):
        """Transplant weights from an existing nn.Linear."""
        tl = cls(linear.in_features, linear.out_features,
                 bias=linear.bias is not None,
                 dtype=linear.weight.dtype, threshold=threshold)
        tl.weight.data.copy_(linear.weight.data)
        if linear.bias is not None:
            tl.bias.data.copy_(linear.bias.data)
        return tl


def ternarize_model(model, threshold=0.5, skip=("norm",)):
    """Replace nn.Linear layers with TernaryLinear throughout the model.

    Normalization layers (RMSNorm, LayerNorm) are skipped — they stay full
    precision for numerical stability. Their parameter count is negligible.

    Returns the number of layers replaced.
    """
    skip_lower = tuple(s.lower() for s in skip)
    replaced = 0

    for name, parent in _linear_parents(model):
        for attr, child in list(parent.named_children()):
            if not isinstance(child, nn.Linear):
                continue
            full = f"{name}.{attr}" if name else attr
            if any(s in full.lower() for s in skip_lower):
                continue
            setattr(parent, attr, TernaryLinear.from_linear(child, threshold))
            replaced += 1

    return replaced


def hard_snap(model):
    """Final hard-snap: overwrite latent weights with exact ternary values.

    After training, latent weights are close to ternary but not exact (last
    optimizer step may have drifted them). This snaps every TernaryLinear's
    weight.data in-place to {-scale, 0, +scale} per block of BLOCK_SIZE.

    Call this before save_pretrained() so the saved HF model has exact ternary
    weights ready for convert_hf_to_gguf.py --outtype tq2_0.
    """
    for name, mod in model.named_modules():
        if not isinstance(mod, TernaryLinear):
            continue

        w = mod.weight.data
        shape = w.shape
        numel = w.numel()
        remainder = numel % BLOCK_SIZE

        if remainder:
            flat = F.pad(w.float().reshape(-1), (0, BLOCK_SIZE - remainder))
        else:
            flat = w.float().reshape(-1)

        g = flat.view(-1, BLOCK_SIZE)
        scale = g.abs().amax(dim=1, keepdim=True)
        normed = g / scale.clamp(min=1e-8)
        snapped = normed.round().clamp(-1, 1) * scale

        mod.weight.data = snapped.reshape(-1)[:numel].view(shape).to(w.dtype)


def verify_ternary(state):
    """Check every extracted weight is exactly {-1, 0, +1}. Returns bad count."""
    bad = 0
    for name, entry in state.items():
        t = entry["ternary"]
        mask = (t == -1) | (t == 0) | (t == 1)
        n = (~mask).sum().item()
        if n:
            bad += n
            print(f"  {name}: {n} non-ternary values")
    return bad


# --- internals ---

def _linear_parents(model):
    """Yield (name, module) for every module that has at least one Linear child."""
    for name, mod in model.named_modules():
        has_linear = any(isinstance(c, nn.Linear) for c in mod.children())
        if has_linear:
            yield name, mod
