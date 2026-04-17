"""Training data loader.

Reads JSONL where each line is {"messages": [...]}, tokenizes with the model's
chat template, and produces causal-LM training batches. The JSONL is prepared
offline from cortex intake corpus — this module does not know about cortex
internals, only the chat message format.
"""

import json
import torch
from torch.utils.data import Dataset


class ChatDataset(Dataset):

    def __init__(self, path, tokenizer, max_seq_len=2048):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.examples = []

        with open(path) as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    print(f"data: skipping malformed line {i + 1}")
                    continue
                msgs = obj.get("messages")
                if not msgs:
                    print(f"data: no messages on line {i + 1}")
                    continue
                self.examples.append(msgs)

        print(f"data: loaded {len(self.examples)} examples from {path}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        text = self.tokenizer.apply_chat_template(
            self.examples[idx], tokenize=False, add_generation_prompt=False)

        enc = self.tokenizer(
            text,
            max_length=self.max_seq_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        ids = enc["input_ids"].squeeze(0)
        mask = enc["attention_mask"].squeeze(0)
        labels = ids.clone()
        labels[mask == 0] = -100

        return {"input_ids": ids, "attention_mask": mask, "labels": labels}
