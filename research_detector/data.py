"""Manifest-based data loading for the research detector (replaces the research code's cohort-specific loaders).

Batches carry the fields the research trainer expects: ``bmode``, ``prostate_mask``, ``needle_mask``, ``label``
(GG >= 1), ``grade_group``, ``clinically_significant`` (GG >= 3), ``involvement`` (0 when unknown), ``involvement_valid``, ``cores_positive_for_patient``,
identifiers, and the five clinical prompts scaled as in Prost-RL / ProstNFound+. Images go through the same
transform as ``protocouncil.data.CoreDataset``.
"""
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset

from protocouncil import data as D
from protocouncil.import_prostrl import PROSTRL_PROMPT_STATS

PROMPT_FIELDS = ('age', 'psa', 'approx_psa_density', 'base_apex_encoding', 'mid_lateral_encoding')


class ResearchCoreDataset(Dataset):
    def __init__(self, table, stats, mean, std, image_size, mask_size, augment):
        self.table = table.reset_index(drop=True)
        self.core_ids = list(self.table.core_id)
        positives = self.table.groupby('patient_id').grade_group.apply(lambda g: int((g >= 1).sum()))
        self.positives = self.table.patient_id.map(positives).fillna(0).astype(int).values
        self.inner = D.CoreDataset(self.table, stats, mean, std, image_size, mask_size, augment)

    def __len__(self):
        return len(self.table)

    def __getitem__(self, i):
        x = self.inner[i]
        inv = x['involvement']
        valid = bool(torch.isfinite(inv))
        out = dict(bmode=x['image'], prostate_mask=x['prostate_mask'], needle_mask=x['needle_mask'],
                   label=(x['grade_group'] >= 1).long(), grade_group=x['grade_group'],
                   clinically_significant=(x['grade_group'] >= 3),
                   involvement=inv if valid else torch.tensor(0.), involvement_valid=torch.tensor(valid),
                   cores_positive_for_patient=torch.tensor(self.positives[i]),
                   core_id=x['core_id'], patient_id=x['patient_id'], center=x['center'])
        for name, value in zip(PROMPT_FIELDS, x['prompts']):
            out[name] = value.reshape(1)
        return out


def get_dataloaders(args, mode='train'):
    """Train / val loaders for ``args.fold`` of ``args.manifest`` (``test`` is empty: external cohorts are scored
    separately). ``mode='train'`` uses ``args.batch_size``; other modes use batch size 1, as in the research code."""
    df = D.read_manifest(args.manifest)
    if 'fold' not in df.columns:
        df = D.assign_folds(df, int(args.n_folds))
    train, val = D.fold_split(D.labeled(df), int(args.fold), bool(args.exclude_benign_cores_from_positive_patients))
    if args.get('limit_train_data') is not None:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=1 - float(args.limit_train_data),
                                     random_state=int(args.get('train_subsample_seed', 42)))
        idx, _ = next(sss.split(train.core_id, train.center))
        train = train.iloc[idx].reset_index(drop=True)
    stats = PROSTRL_PROMPT_STATS if args.get('prompt_stats', 'prost_rl') == 'prost_rl' else D.prompt_stats(train)
    kw = dict(stats=stats, mean=tuple(args.mean), std=tuple(args.std), image_size=int(args.image_size),
              mask_size=int(args.mask_size))
    train_ds = ResearchCoreDataset(train, augment=str(args.augmentations), **kw)
    val_ds = ResearchCoreDataset(val, augment='none', **kw)
    test_ds = ResearchCoreDataset(val.iloc[:0], augment='none', **kw)
    bs = int(args.batch_size) if mode == 'train' else 1
    common = dict(batch_size=bs, num_workers=int(args.num_workers), pin_memory=True)
    return dict(train=DataLoader(train_ds, shuffle=(mode == 'train') and len(train_ds) > 0, **common),
                val=DataLoader(val_ds, shuffle=False, **common),
                test=DataLoader(test_ds, shuffle=False, **common))


def grades_of(loader):
    """Grade groups of the cores of a loader's dataset (for fold-specific class weights)."""
    return loader.dataset.table.grade_group.astype(int).to_numpy()
