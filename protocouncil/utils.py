import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import yaml


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path, **overrides):
    cfg = yaml.safe_load(Path(path).read_text())
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    return cfg


def autocast(precision, device):
    dtype = {'bf16': torch.bfloat16, 'fp16': torch.float16}.get(precision)
    return torch.autocast(device_type=device.type, dtype=dtype or torch.float32, enabled=dtype is not None)


def warmup_cosine(optimizer, steps_per_epoch, epochs, warmup_epochs=5):
    """Linear warm-up over ``warmup_epochs`` followed by cosine decay, stepped every iteration."""
    warmup, total = warmup_epochs * steps_per_epoch, epochs * steps_per_epoch

    def multiplier(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1., (step - warmup) / max(1, total - warmup))))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def append_jsonl(path, row):
    with open(path, 'a') as f:
        f.write(json.dumps(row) + '\n')


def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
