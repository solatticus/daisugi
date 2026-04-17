# Daisugi

https://en.wikipedia.org/wiki/Daisugi

Prism dropped their [Ternary Bonsai](https://huggingface.co/collections/prism-ml/ternary-bonsai) models today. 8B at 1.75 GB, benchmarks within 5% of full precision. **Incredible**.

Apple Silicon only as far as I can tell as of this writing.

Read the [whitepaper](https://github.com/PrismML-Eng/Bonsai-demo/blob/main/ternary-bonsai-8b-whitepaper.pdf) (more like benchmarks though). The weight format is seemingly simple: every weight is -1, 0, or +1 times a shared scale per block of 256.

The training recipe is two existing papers combined:

  * [BitNet](https://arxiv.org/abs/2310.11453) — 1-bit training with straight-through estimators
  * [BitNet b1.58](https://arxiv.org/abs/2402.17764) — extended to ternary {-1, 0, +1}

Full-precision latent weights for the optimizer, ternary snap on every forward pass. The model learns to be good at your task *within the constraint*.

The pre-trained model is a block of wood (in my head at least). Daisugi is the method. What comes out is small, hard, and purpose-built for your domain. ~2 bits per weight, runs on hardware that couldn't touch the original.

My use case is a 12GB 4070 Ti. It runs a model that deeply classifies content that I send to myself via inbox post (+agent moniker email address, whitelist of course). I train on my dataset with ~5 epochs (still playing with this) on a 5090:

## Recipe

```
         config.toml                       JSONL
              |                              |
              v                              v
forge.py ---- QAT fine-tune (ternary snap every forward pass)
              |
              v
pack.py ----- hard-snap weights, save as HuggingFace model
              |
              v
convert_hf_to_gguf.py --outtype tq2_0  (basic llama.cpp)
              |
              v
llama-server -m model.gguf -ngl 99         (GPU)
```

## Usage

```bash
python forge.py                     # uses config.toml
python forge.py --config my.toml    # custom config
```

Training data is JSONL, one example per line:
```json
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

## Config

```toml
[model]
base = "Qwen/Qwen3-8B"           # any HuggingFace causal LM
device = "cuda"                   # cuda, mps
skip = ["norm"]                   # MoE: ["norm", "gate"]

[training]
optimizer = "adafactor"           # adafactor (big models) or adamw (small)
epochs = 3
batch_size = 1
grad_accum = 8
lr = 2e-5
max_seq_len = 2048

[data]
train = "./data/train.jsonl"

[output]
dir = "./output"
```

## What's in the box

| File | Lines | Job |
|------|-------|-----|
| `ternary.py` | 173 | The math. TernarySnap autograd function, STE, block-256 absmax scale, model surgery. |
| `forge.py` | 170 | Training loop. Raw PyTorch, gradient checkpointing, warmup, logging. |
| `data.py` | 59 | JSONL chat dataset with tokenizer chat template. |
| `pack.py` | 85 | Hard-snap trained weights, save as HF model for GGUF conversion. |

Block size and scale computation match GGML's `TQ2_0` exactly — `quantize_row_tq2_0_ref` in llama.cpp uses absmax over 256 weights, round, clamp. No train/deploy mismatch.

## Requirements

- PyTorch
- transformers
- safetensors

## License

MIT
