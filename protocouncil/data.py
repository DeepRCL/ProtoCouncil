"""Core-level micro-ultrasound data from a CSV manifest.

One row per biopsy core. Required columns:

    core_id, patient_id, center      identifiers (center is used to stratify patient folds)
    image                            path to the B-mode frame (grayscale .png/.jpg/.tif or 2-D .npy)
    prostate_mask, needle_mask       paths to binary masks with the same size as the image
    grade_group                      ISUP Grade Group 0-5 (0 = benign); empty for unlabeled cores
    involvement                      cancer involvement of the core as a fraction in [0, 1]; empty if unknown
    age, psa, psad                   clinical values (empty -> training-split mean)
    base_apex                        core location: -1 base, 0 mid, 1 apex
    mid_lateral                      core location: 1 medial, -1 lateral
Optional: ``fold`` (integer patient fold). Without it, folds are drawn by ``assign_folds``.

Relative paths are resolved against the manifest's directory. Images are expected in the same orientation as
the training data (transducer at the top row of the array).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image as PILImage
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2 as T
from torchvision.transforms.functional import InterpolationMode

REQUIRED = ('core_id', 'patient_id', 'center', 'image', 'prostate_mask', 'needle_mask', 'grade_group',
            'involvement', 'age', 'psa', 'psad', 'base_apex', 'mid_lateral')
CONTINUOUS_PROMPTS = ('age', 'psa', 'psad')
LOCATION_PROMPTS = ('base_apex', 'mid_lateral')


def read_manifest(path):
    path = Path(path)
    df = pd.read_csv(path, dtype={'core_id': str, 'patient_id': str, 'center': str})
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f'manifest {path} is missing columns {missing}')
    if not df.core_id.is_unique:
        raise ValueError('core_id must be unique')
    for col in ('image', 'prostate_mask', 'needle_mask'):
        df[col] = [str(p) if Path(p).is_absolute() else str(path.parent / p) for p in df[col]]
    gg = df.grade_group
    if not gg.dropna().between(0, 5).all() or not (gg.dropna() == gg.dropna().round()).all():
        raise ValueError('grade_group must be an integer in 0..5 (or empty)')
    inv = df.involvement.dropna()
    if not inv.between(0, 1).all():
        raise ValueError('involvement must be a fraction in [0, 1]; divide percentages by 100')
    return df


def assign_folds(df, n_folds=5, seed=0):
    """Patient-level folds stratified by center (shuffled StratifiedKFold over unique patients)."""
    patients = df.drop_duplicates('patient_id')[['patient_id', 'center']].reset_index(drop=True)
    fold_of = {}
    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for k, (_, idx) in enumerate(kfold.split(patients.patient_id, patients.center)):
        fold_of.update({p: k for p in patients.patient_id.iloc[idx]})
    out = df.copy()
    out['fold'] = out.patient_id.map(fold_of).astype(int)
    return out


def fold_split(df, fold, exclude_benign_from_positive_patients=True):
    """Training and validation tables for one fold. Training drops benign cores of patients with any cancer core."""
    if 'fold' not in df.columns:
        raise ValueError('manifest has no fold column; call assign_folds first')
    train, val = df[df.fold != fold].copy(), df[df.fold == fold].copy()
    assert not set(train.patient_id) & set(val.patient_id)
    if exclude_benign_from_positive_patients:
        positive = set(train.patient_id[train.grade_group >= 1])
        train = train[(train.grade_group >= 1) | ~train.patient_id.isin(positive)]
    return train.reset_index(drop=True), val.reset_index(drop=True)


def labeled(df):
    return df[df.grade_group.notna()].reset_index(drop=True)


def three_state(grade_group):
    """0 benign, 1 GG1-2, 2 GG>=3."""
    g = np.asarray(grade_group)
    return np.where(g >= 3, 2, np.where(g >= 1, 1, 0))


def prompt_stats(train):
    """Min / max / mean of the continuous clinical values on the training split (used for min-max scaling)."""
    stats = {}
    for c in CONTINUOUS_PROMPTS:
        x = train[c].astype(float)
        if x.notna().sum() == 0:
            raise ValueError(f'no {c} values in the training split')
        lo, hi = float(x.min()), float(x.max())
        stats[c] = dict(min=lo, max=hi if hi > lo else lo + 1., mean=float(x.mean()))
    return stats


def prompt_vector(row, stats):
    v = []
    for c in CONTINUOUS_PROMPTS:
        x = row[c]
        x = stats[c]['mean'] if pd.isna(x) else float(x)
        v.append((x - stats[c]['min']) / (stats[c]['max'] - stats[c]['min']))
    for c in LOCATION_PROMPTS:
        v.append(0. if pd.isna(row[c]) else float(row[c]))
    return torch.tensor(v, dtype=torch.float32)


def load_array(path):
    if str(path).endswith('.npy'):
        x = np.load(path)
    else:
        img = PILImage.open(path)
        if img.mode in ('RGB', 'RGBA', 'P', 'LA', 'CMYK', 'YCbCr'):
            img = img.convert('L')  # colour images: luminance; grayscale (8/16-bit, float) keeps native values
        x = np.asarray(img)
    if x.ndim == 3:
        x = x[..., 0]
    return x.astype(np.float32)


class CoreDataset(Dataset):
    """B-mode frame, prostate/needle masks, normalized clinical prompts and labels for each core.

    Preprocessing follows ProstNFound+: min-max intensity scaling, channel normalization, resize to 1024,
    optional augmentation (``translate``: random translation up to 20%; ``random_crop``: random resized crop
    with probability 0.5), resize to ``image_size``; masks are resized with nearest interpolation to ``mask_size``.
    """

    def __init__(self, table, stats, mean, std, image_size=256, mask_size=64, augment='none'):
        self.table = table.reset_index(drop=True)
        self.stats, self.mean, self.std = stats, tuple(mean), tuple(std)
        self.image_size, self.mask_size, self.augment = image_size, mask_size, augment

    def __len__(self):
        return len(self.table)

    def __getitem__(self, i):
        row = self.table.iloc[i]
        bmode = torch.from_numpy(load_array(row.image))[None].repeat(3, 1, 1)
        lo = bmode.min()
        bmode = (bmode - lo) / (bmode.max() - lo).clamp_min(1e-6)
        bmode = tv_tensors.Image(T.Normalize(self.mean, self.std)(bmode))
        pm = tv_tensors.Mask(torch.from_numpy((load_array(row.prostate_mask) > 0).astype(np.float32))[None])
        nm = tv_tensors.Mask(torch.from_numpy((load_array(row.needle_mask) > 0).astype(np.float32))[None])
        bmode, nm, pm = T.Resize((1024, 1024), antialias=True)(bmode, nm, pm)
        if 'translate' in self.augment:
            bmode, nm, pm = T.RandomAffine([0, 0], [0.2, 0.2])(bmode, nm, pm)
        if 'random_crop' in self.augment and torch.rand(1).item() > .5:
            bmode, nm, pm = T.RandomResizedCrop((self.image_size, self.image_size), scale=(0.3, 1))(bmode, nm, pm)
        else:
            bmode = T.Resize((self.image_size, self.image_size))(bmode)
        resize = T.Resize((self.mask_size, self.mask_size), antialias=False, interpolation=InterpolationMode.NEAREST)
        gg = row.grade_group
        inv = row.involvement
        if pd.isna(inv) and not pd.isna(gg) and gg == 0:
            inv = 0.
        return dict(image=bmode.as_subclass(torch.Tensor),
                    prostate_mask=resize(pm).as_subclass(torch.Tensor),
                    needle_mask=resize(nm).as_subclass(torch.Tensor),
                    prompts=prompt_vector(row, self.stats),
                    grade_group=torch.tensor(-1 if pd.isna(gg) else int(gg)),
                    involvement=torch.tensor(float('nan') if pd.isna(inv) else float(inv)),
                    core_id=str(row.core_id), patient_id=str(row.patient_id), center=str(row.center))
