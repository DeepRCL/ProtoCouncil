"""Train one encoder's anatomical prototype head on cached region descriptors (CPU).

    python -m protocouncil.train_heads --packs runs/packs/medsam/fold0 --out runs/heads/medsam/fold0/seed7201 --seed 7201

Procedure (paper Sec. 2.2): AdamW (lr 1e-3 projectors and prototypes, 1e-2 final layer, weight decay 1e-4, batch 96),
up to 40 epochs with early stopping (patience 10) on the validation fold's mean of all-cancer and csPCa AUROC;
gland-family dropout 0.5; projection of every prototype onto a real training region (Hungarian assignment, distinct
patients); then projectors and prototypes are frozen and only the final layer is refit (up to 15 epochs, the
projected initialization is eligible). The label-free fusion scale is the SD of class-centred validation logits.
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import three_state
from .metrics import selection_score
from .models import prototypes as P

READOUT_LR = 1e-2
LR = 1e-3
WEIGHT_DECAY = 1e-4
BATCH = 96
EPOCHS = 40
REFIT_EPOCHS = 15
PATIENCE = 10
CLIP = 5.


def load_pack(path):
    """Region-descriptor pack written by ``protocouncil.extract`` (adds the 3-state target)."""
    c = torch.load(path, weights_only=False)
    c['state'] = torch.as_tensor(three_state(c['grade_group'].numpy()), dtype=torch.long)
    c['involvement_valid'] = torch.isfinite(c['involvement'])
    return c


def batch(c, ix):
    return dict(ctx=c['desc_ctx'][ix], enc=c['desc_enc'][ix], valid=c['valid'][ix], state=c['state'][ix])


@torch.no_grad()
def infer(model, c, chunk=512):
    model.eval()
    outs = [model(batch(c, slice(i, i + chunk))) for i in range(0, len(c['core_id']), chunk)]
    res = {k: torch.cat([o[k] for o in outs]) for k in ('logits', 'activation', 'winner')}
    return res


def score(c, logits):
    return selection_score(c['grade_group'].numpy(), logits.softmax(1).numpy())


def fit(model, tr, va, params, epochs, seed, refit=False):
    """AdamW training with validation selection; ``refit`` also evaluates the starting point (epoch -1)."""
    readout = [q for n, q in model.named_parameters() if n in ('rho', 'bias') and any(q is x for x in params)]
    rest = [q for q in params if not any(q is x for x in readout)]
    groups = [g for g in [dict(params=rest, lr=LR), dict(params=readout, lr=READOUT_LR)] if g['params']]
    opt = torch.optim.AdamW(groups, weight_decay=WEIGHT_DECAY)
    w = P.class_weights(tr['state'])
    best, state, stale, history = -np.inf, None, 0, []
    g = torch.Generator().manual_seed(int(seed) % 2**31)
    for epoch in range(-1 if refit else 0, epochs):
        losses = []
        if epoch >= 0:
            model.train()
            for ix in torch.randperm(len(tr['core_id']), generator=g).split(BATCH):
                opt.zero_grad(set_to_none=True)
                loss, _ = P.objective(model, batch(tr, ix), w)
                assert torch.isfinite(loss)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, CLIP, error_if_nonfinite=True)
                opt.step()
                losses.append(float(loss.detach()))
        s = score(va, infer(model, va)['logits'])
        history.append(dict(epoch=epoch, val_score=s, loss=float(np.mean(losses)) if losses else None))
        if s > best + 1e-6:
            best, state, stale = s, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        if stale >= PATIENCE:
            break
    model.load_state_dict(state)
    return history


def predictions(c, logits):
    p = logits.softmax(1).numpy()
    t = pd.DataFrame({k: c[k] for k in ('core_id', 'patient_id', 'center')})
    t['grade_group'] = c['grade_group'].numpy()
    t['involvement'] = c['involvement'].numpy()
    for k in range(3):
        t[f'logit{k}'] = logits[:, k].numpy()
    t['p0'], t['p1'], t['p2'] = p[:, 0], p[:, 1], p[:, 2]
    t['p_cancer'], t['p_cspca'] = p[:, 1:].sum(1), p[:, 2]
    return t


def train(packs, out, seed, gland_dropout=.5, threads=4):
    torch.set_num_threads(threads)
    torch.manual_seed(seed); np.random.seed(seed)
    packs, out = Path(packs), Path(out)
    tr, va = load_pack(packs / 'train.pt'), load_pack(packs / 'val.pt')
    assert not set(tr['patient_id']) & set(va['patient_id']), 'patient overlap between train and val'
    torch.manual_seed(seed)
    m = P.AnatomicalPrototypeHead(tr['desc_ctx'].shape[-1], tr['desc_enc'].shape[-1], gland_dropout)
    history = {'learned': fit(m, tr, va, list(m.parameters()), EPOCHS, seed)}
    learned = copy.deepcopy(m).eval()
    records = P.project(m, tr['desc_ctx'], tr['desc_enc'], tr['valid'], tr['state'], tr['involvement'],
                        tr['involvement_valid'], tr['patient_id'], tr['occupancy'])
    m.f_abs.requires_grad_(False); m.f_rel.requires_grad_(False); m.prototypes.requires_grad_(False)
    history['refit'] = fit(m, tr, va, [m.rho, m.bias], REFIT_EPOCHS, seed, refit=True)
    m.requires_grad_(True); m.eval()
    err = P.grounding_error(m, tr['desc_ctx'], tr['desc_enc'], tr['valid'], records)
    assert err < 1e-5, ('projected prototypes are not exact region embeddings', err)
    for x in records:
        x['core_id'] = tr['core_id'][x['core_index']]
        x['center'] = tr['center'][x['core_index']]
        x['grade_group'] = int(tr['grade_group'][x['core_index']])
        x['involvement'] = float(tr['involvement'][x['core_index']])
    o = infer(m, va)
    scale = P.logit_scale(o['logits'])
    out.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state_dict=m.state_dict(), dims=m.dims, gland_dropout=gland_dropout, per_class=m.per_class,
                    fusion_scale=scale, projection=records, seed=seed, encoder=tr.get('encoder'), fold=tr.get('fold')),
               out / 'head.pth')
    torch.save(dict(state_dict=learned.state_dict(), dims=learned.dims, gland_dropout=gland_dropout), out / 'head_learned.pth')
    predictions(va, o['logits']).to_csv(out / 'val_predictions.csv', index=False)
    (out / 'history.json').write_text(json.dumps(history, indent=1))
    (out / 'projection.json').write_text(json.dumps(records, indent=1))
    print(json.dumps(dict(out=str(out), val_score=score(va, o['logits']), fusion_scale=scale, grounding_error=err)))
    return m


def load_head(path):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    m = P.AnatomicalPrototypeHead(*ck['dims'], ck['gland_dropout'], ck.get('per_class', P.PER_CLASS))
    m.load_state_dict(ck['state_dict'])
    return m.eval(), ck


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--packs', required=True, help='directory with train.pt and val.pt from protocouncil.extract')
    p.add_argument('--out', required=True)
    p.add_argument('--seed', type=int, default=7201)
    p.add_argument('--gland-dropout', type=float, default=.5)
    p.add_argument('--threads', type=int, default=4)
    a = p.parse_args()
    train(a.packs, a.out, a.seed, a.gland_dropout, a.threads)


if __name__ == '__main__':
    main()
