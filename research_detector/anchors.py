"""Semantic-anchor bank for the research detector (60 anchors x 256 dims, 10 per Grade Group 0-5).

The paper's detectors used a bank of projected PANDA tile prototypes from a pathology teacher. That bank is derived
from PANDA (CC BY-NC-SA 4.0) and Prov-GigaPath features (research-use license) and is not redistributed here; pass
it with ``--anchor-bank <file.npz>`` if you have it. The default is the matched random bank used as a control in our
experiments (unit Gaussian vectors with the same grade/state counts, seed 92121), which is regenerated exactly.
"""
from pathlib import Path

import numpy as np

GRADES = np.repeat(np.arange(6), 10)
STATES = np.where(GRADES == 0, 0, np.where(GRADES >= 3, 2, 1))


def random_bank(seed=92121, dim=256):
    x = np.random.default_rng(seed).standard_normal((len(GRADES), dim), dtype=np.float32)
    x = x / np.linalg.norm(x, axis=1, keepdims=True).clip(1e-8)
    return dict(anchors=x.astype(np.float32), state_labels=STATES.astype(np.int64), grade_labels=GRADES.astype(np.int64),
                anchor_ids=np.asarray([f'random-{i:02d}' for i in range(len(GRADES))]))


def resolve_bank(spec, out_dir):
    """'random' -> write the matched random bank into ``out_dir``; otherwise an existing .npz path."""
    if spec == 'random':
        path = Path(out_dir) / 'random_anchor_bank.npz'
        np.savez_compressed(path, **random_bank())
        return path.resolve()
    path = Path(spec)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.resolve()
