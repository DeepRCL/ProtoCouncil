"""Freeze a trained detector, extract its two 16x16 token grids and store anatomical region descriptors.

    python -m protocouncil.extract --detector runs/detectors/medsam/fold0/best.pth \
        --manifest data/cohort_a.csv --out runs/packs/medsam/fold0 [--external data/cohort_b.csv:cohort_b ...]

Writes ``train.pt`` and ``val.pt`` for the detector's fold (same cohort rules as detector training) and one pack per
external manifest (labels optional). Each pack holds, per core: ids, grade group, involvement, the 64x64 prostate
and needle masks, region descriptors of the encoder grid (``desc_enc``) and of the prompt-conditioned class-decoder
grid (``desc_ctx``), region occupancy and validity. Tokens are rounded to float16 before descriptors are computed and
descriptors are stored in float16. Cores without a needle track or without any valid sector cannot be scored by
the prototype head; they are listed in ``excluded_<split>.csv``.
"""
import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from . import data as D
from .models import prototypes as P
from .models.detector import load_detector
from .utils import device


@torch.no_grad()
def extract(det, ck, table, dev, batch_size=16, num_workers=4, enc_det=None):
    """``enc_det``: optional second detector whose image encoder gives the enc grid (ctx always comes from ``det``)."""
    ds = D.CoreDataset(table, ck['prompt_stats'], ck['mean'], ck['std'], image_size=ck['image_size'], mask_size=64)
    meta = {k: [] for k in ('core_id', 'patient_id', 'center')}
    gg, inv, pm, nm, dc, de, occ = [], [], [], [], [], [], []
    det.eval()
    for b in DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers):
        image = b['image'].to(dev).float()
        out = det(image, b['prompts'].to(dev), return_tokens=True,
                  prostate_mask=b['prostate_mask'].to(dev), needle_mask=b['needle_mask'].to(dev))
        if enc_det is not None:
            out['enc'] = enc_det.encoder(image).flatten(2).transpose(1, 2)
        p, n = b['prostate_mask'][:, 0] > .5, b['needle_mask'][:, 0] > .5
        reg = torch.zeros(len(p), len(P.REGIONS), 64, 64, dtype=torch.bool)
        has = p.flatten(1).any(1)  # empty prostate masks give empty regions -> excluded below
        if has.any():
            reg[has] = P.regions(p[has], n[has])
        d_ctx, o = P.descriptors(out['ctx'].half().cpu(), reg)
        d_enc, o2 = P.descriptors(out['enc'].half().cpu(), reg)
        assert torch.equal(o, o2)
        for k in meta:
            meta[k] += list(b[k])
        gg.append(b['grade_group']); inv.append(b['involvement']); pm.append(p); nm.append(n)
        dc.append(d_ctx.half()); de.append(d_enc.half()); occ.append(o)
    c = dict(meta, grade_group=torch.cat(gg).long(), involvement=torch.cat(inv).float(), prostate64=torch.cat(pm),
             needle64=torch.cat(nm), desc_ctx=torch.cat(dc), desc_enc=torch.cat(de), occupancy=torch.cat(occ))
    c['valid'] = P.validity(c['occupancy'])
    finite = torch.isfinite(c['desc_ctx'].float()).flatten(1).all(1) & torch.isfinite(c['desc_enc'].float()).flatten(1).all(1)
    keep = P.usable(c['valid']) & finite
    excluded = pd.DataFrame(dict(core_id=[k for k, ok in zip(c['core_id'], keep) if not ok],
                                 reason=['no needle track or no valid sector in the 64x64 masks' if not u else 'non-finite tokens'
                                         for u, ok in zip(P.usable(c['valid']).tolist(), keep.tolist()) if not ok]))
    idx = torch.nonzero(keep).flatten().tolist()
    c = {k: (v[keep] if torch.is_tensor(v) else [v[i] for i in idx]) for k, v in c.items()}
    return c, excluded


def parse_external(items):
    """``path:name`` items -> {name: path}. Names must be new, start with a letter or digit, use only letters, digits, ``_`` and ``-``, and not be train/val."""
    out = {}
    for item in items:
        path, sep, name = item.rpartition(':')
        if (not sep or not path or name in ('train', 'val') or name in out or not name[:1].isalnum()
                or not name.replace('_', '').replace('-', '').isalnum()):
            raise ValueError(f'external cohorts are given as path:name with a new alphanumeric name (not train/val): {item}')
        out[name] = path
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--detector', required=True, help='detector giving the ctx grid (and the enc grid by default)')
    p.add_argument('--enc-detector', help='optional detector whose image encoder gives the enc grid (e.g. Prost-RL RL stage)')
    p.add_argument('--manifest', required=True, help='development manifest (cohort used for cross-validation)')
    p.add_argument('--out', required=True)
    p.add_argument('--external', nargs='*', default=[], help='extra manifests as path:name (scored, never trained on)')
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--num-workers', type=int, default=4)
    a = p.parse_args()
    dev = device()
    det, ck = load_detector(a.detector)
    det.to(dev)
    enc_det = None
    if a.enc_detector:
        enc_det, eck = load_detector(a.enc_detector)
        assert (eck['encoder'], tuple(eck['mean']), tuple(eck['std']), eck['image_size']) == \
            (ck['encoder'], tuple(ck['mean']), tuple(ck['std']), ck['image_size']), 'detectors use different inputs'
        assert (eck['fold'], eck['n_folds']) == (ck['fold'], ck['n_folds']), 'detectors come from different folds'
        enc_det.to(dev).eval()
    df = D.read_manifest(a.manifest)
    if 'fold' not in df.columns:
        df = D.assign_folds(df, ck['n_folds'])
    train, val = D.fold_split(D.labeled(df), ck['fold'], ck['exclude_benign_from_positive_patients'])
    splits = {'train': train, 'val': val}
    for name, path in parse_external(a.external).items():
        splits[name] = D.read_manifest(path)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, table in splits.items():
        c, excluded = extract(det, ck, table, dev, a.batch_size, a.num_workers, enc_det)
        c.update(encoder=ck['encoder'], fold=ck['fold'], split=name)
        torch.save(c, out / f'{name}.pt')
        excluded.to_csv(out / f'excluded_{name}.csv', index=False)
        summary[name] = dict(cores=len(c['core_id']), patients=len(set(c['patient_id'])), excluded=len(excluded))
        print(name, summary[name], flush=True)
    (out / 'summary.json').write_text(json.dumps(dict(detector=str(a.detector), enc_detector=a.enc_detector, **summary), indent=1))


if __name__ == '__main__':
    main()
