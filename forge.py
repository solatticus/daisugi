"""Daisugi forge — ternary QAT fine-tuning.

Load a pre-trained model, replace linear layers with ternary-quantized
versions, fine-tune on domain data. Every forward pass sees ternary weights.
Export as HuggingFace model, then convert to GGUF TQ2_0 via llama.cpp.

Usage:
    python forge.py                     # uses config.toml
    python forge.py --config my.toml    # custom config
"""

import argparse
import os
import time

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

from ternary import ternarize_model
from data import ChatDataset
from pack import pack


def load_config(path):
    with open(path, "rb") as f:
        return tomllib.load(f)


def build_optimizer(model, cfg):
    name = cfg["training"].get("optimizer", "adafactor")
    lr = cfg["training"]["lr"]

    if name == "adafactor":
        from transformers import Adafactor
        return Adafactor(
            model.parameters(),
            lr=lr,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False,
        )

    if name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=cfg["training"].get("weight_decay", 0.01),
        )

    raise ValueError(f"unknown optimizer: {name}")


def forge(cfg):
    mcfg = cfg["model"]
    tcfg = cfg["training"]
    dcfg = cfg["data"]
    ocfg = cfg["output"]

    os.makedirs(ocfg["dir"], exist_ok=True)

    # --- model ---
    device = mcfg.get("device", "cuda")
    attn_impl = mcfg.get("attn", "flash_attention_2" if device == "cuda" else "sdpa")
    skip = mcfg.get("skip", ["norm"])

    print(f"forge: loading {mcfg['base']} → {device} ({attn_impl})")
    model = AutoModelForCausalLM.from_pretrained(
        mcfg["base"],
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
    )
    tokenizer = AutoTokenizer.from_pretrained(mcfg["base"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    count = ternarize_model(model, threshold=mcfg.get("threshold", 0.5), skip=skip)
    print(f"forge: ternarized {count} linear layers")

    model.gradient_checkpointing_enable()
    model.to(device)

    if tcfg.get("compile"):
        print("forge: compiling model")
        model = torch.compile(model)

    # --- data ---
    dataset = ChatDataset(dcfg["train"], tokenizer, tcfg.get("max_seq_len", 2048))
    loader = DataLoader(
        dataset,
        batch_size=tcfg.get("batch_size", 1),
        shuffle=True,
        pin_memory=True,
        num_workers=2,
        drop_last=True,
    )

    # --- optimizer ---
    optimizer = build_optimizer(model, cfg)
    grad_accum = tcfg.get("grad_accum", 1)
    max_norm = tcfg.get("max_grad_norm", 1.0)
    log_every = ocfg.get("log_every", 10)

    # --- warmup ---
    warmup_steps = tcfg.get("warmup_steps", 0)

    def warmup_lr(step):
        if warmup_steps and step < warmup_steps:
            return (step + 1) / warmup_steps
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_lr)

    # --- train ---
    model.train()
    global_step = 0

    for epoch in range(tcfg.get("epochs", 3)):
        epoch_loss = 0.0
        micro_steps = 0
        t0 = time.monotonic()

        for batch in loader:
            ids = batch["input_ids"].to(device, non_blocking=True)
            mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            with torch.amp.autocast(device, dtype=torch.bfloat16):
                loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
                loss = loss / grad_accum

            loss.backward()
            epoch_loss += loss.item() * grad_accum
            micro_steps += 1

            if micro_steps % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % log_every == 0:
                    avg = epoch_loss / micro_steps
                    elapsed = time.monotonic() - t0
                    lr_now = scheduler.get_last_lr()[0]
                    print(f"  epoch {epoch + 1}  step {global_step}  "
                          f"loss {avg:.4f}  lr {lr_now:.2e}  "
                          f"{elapsed:.0f}s")

        avg = epoch_loss / max(micro_steps, 1)
        print(f"forge: epoch {epoch + 1} done — avg loss {avg:.4f}")

        ckpt = os.path.join(ocfg["dir"], f"epoch_{epoch + 1}.pt")
        torch.save(model.state_dict(), ckpt)
        print(f"forge: saved {ckpt}")

    # --- export ---
    hf_dir = os.path.join(ocfg["dir"], "hf")
    print("forge: snapping and exporting HF model")
    pack(model, tokenizer, hf_dir)
    print("forge: done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Daisugi — ternary QAT forge")
    parser.add_argument("--config", default="config.toml")
    args = parser.parse_args()
    forge(load_config(args.config))
