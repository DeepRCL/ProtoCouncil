"""Fine-tune a foundation-model encoder as a prompt-conditioned cancer detector on one cross-validation fold.

    python -m protocouncil.train_detector --config configs/detector_medsam.yaml \
        --manifest data/cohort_a.csv --fold 0 --out runs/detectors/medsam/fold0

Objective: needle-track involvement loss on the heatmap (proportion BCE, ProstNFound+) plus three-class
cross-entropy (benign / GG1-2 / GG>=3) on the class decoder. The checkpoint with the best validation-fold mean of
all-cancer and csPCa AUROC (class decoder) is kept as ``best.pth``.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from . import data as D
from .losses import balanced_cross_entropy, involvement_loss, needle_mean_probability
from .metrics import auroc, selection_score
from .models.detector import build_detector
from .utils import append_jsonl, autocast, device, load_config, set_seed, warmup_cosine


def loaders(cfg, manifest, fold, mean, std):
    df = D.read_manifest(manifest)
    if 'fold' not in df.columns:
        df = D.assign_folds(df, cfg['n_folds'])
    train, val = D.fold_split(D.labeled(df), fold, cfg['exclude_benign_from_positive_patients'])
    stats = D.prompt_stats(train)
    kw = dict(stats=stats, mean=mean, std=std, image_size=cfg['image_size'], mask_size=cfg['mask_size'])
    g = torch.Generator().manual_seed(cfg['seed'])
    tl = DataLoader(D.CoreDataset(train, augment=cfg['augment'], **kw), batch_size=cfg['batch_size'], shuffle=True,
                    generator=g, num_workers=cfg['num_workers'], drop_last=False, pin_memory=True)
    vl = DataLoader(D.CoreDataset(val, **kw), batch_size=cfg['batch_size'], shuffle=False,
                    num_workers=cfg['num_workers'], pin_memory=True)
    return tl, vl, stats, train, val


@torch.no_grad()
def evaluate(det, loader, dev, precision):
    det.eval()
    rows = []
    for b in loader:
        with autocast(precision, dev):
            out = det(b['image'].to(dev), b['prompts'].to(dev))
        p = out['class_logits'].float().softmax(1).cpu().numpy()
        h = needle_mean_probability(out['heatmap_logits'], b['prostate_mask'].to(dev), b['needle_mask'].to(dev)).cpu().numpy()
        for i in range(len(p)):
            rows.append(dict(core_id=b['core_id'][i], patient_id=b['patient_id'][i], center=b['center'][i],
                             grade_group=int(b['grade_group'][i]), involvement=float(b['involvement'][i]),
                             p0=p[i, 0], p1=p[i, 1], p2=p[i, 2], heatmap_needle_mean=h[i]))
    t = pd.DataFrame(rows)
    t['p_cancer'], t['p_cspca'] = t.p1 + t.p2, t.p2
    score = selection_score(t.grade_group.values, t[['p0', 'p1', 'p2']].values)
    return t, dict(val_score=score, val_all_auroc=auroc(t.grade_group >= 1, t.p_cancer),
                   val_cspca_auroc=auroc(t.grade_group >= 3, t.p_cspca),
                   val_heatmap_all_auroc=auroc(t.grade_group >= 1, t.heatmap_needle_mean))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', required=True)
    p.add_argument('--manifest', required=True)
    p.add_argument('--fold', type=int, required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--encoder-weights', help='MedSAM checkpoint (medsam) or adapted backbone from protocouncil.adapt')
    p.add_argument('--medsam-checkpoint', help='MedSAM release checkpoint (initializes prompt encoder and mask decoder)')
    p.add_argument('--init-decoders-from', help='trained detector (e.g. the MedSAM detector of the same fold) whose '
                                                'prompt encoder, prompt MLPs and decoders initialize this detector')
    p.add_argument('--epochs', type=int, help='override the number of epochs in the config (e.g. for a quick check)')
    a = p.parse_args()
    cfg = load_config(a.config, medsam_checkpoint=a.medsam_checkpoint, encoder_weights=a.encoder_weights,
                      init_decoders_from=a.init_decoders_from, epochs=a.epochs)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    set_seed(cfg['seed'])
    dev = device()
    weights = cfg.get('encoder_weights') or (cfg['medsam_checkpoint'] if cfg['encoder'] == 'medsam' else None)
    det = build_detector(cfg['encoder'], weights, cfg['medsam_checkpoint'])
    if cfg.get('init_decoders_from'):
        det.load_decoders_from(torch.load(cfg['init_decoders_from'], map_location='cpu', weights_only=False)['state_dict'])
    det.to(dev)
    tl, vl, stats, train, val = loaders(cfg, a.manifest, a.fold, det.mean, det.std)
    groups = [dict(params=det.encoder_parameters(), lr=cfg['encoder_lr']), dict(params=det.head_parameters(), lr=cfg['lr'])]
    opt = torch.optim.AdamW([g for g in groups if g['params']], weight_decay=cfg['weight_decay'])
    sched = warmup_cosine(opt, len(tl), cfg['epochs'], cfg['warmup_epochs'])
    scaler = torch.amp.GradScaler(enabled=cfg['precision'] == 'fp16')
    (out / 'config.json').write_text(json.dumps(dict(cfg, fold=a.fold, manifest=str(a.manifest), train_cores=len(train),
                                                     val_cores=len(val), prompt_stats=stats), indent=1))
    best = -np.inf
    for epoch in range(cfg['epochs']):
        det.train()
        losses = []
        for b in tl:
            image, prompts = b['image'].to(dev), b['prompts'].to(dev)
            pm, nm = b['prostate_mask'].to(dev), b['needle_mask'].to(dev)
            target = torch.as_tensor(D.three_state(b['grade_group'].numpy()), device=dev)
            with autocast(cfg['precision'], dev):
                o = det(image, prompts)
            loss = involvement_loss(o['heatmap_logits'], pm, nm, b['involvement'].to(dev), cfg['pos_weight'],
                                    cfg['entropy_weight']) + cfg['class_weight'] * balanced_cross_entropy(o['class_logits'], target)
            if not torch.isfinite(loss):
                raise RuntimeError(f'non-finite loss at epoch {epoch}')
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([q for g in opt.param_groups for q in g['params']], cfg['clip_grad'])
            scaler.step(opt); scaler.update(); sched.step()
            losses.append(float(loss.detach()))
        table, metrics = evaluate(det, vl, dev, cfg['precision'])
        row = dict(epoch=epoch, train_loss=float(np.mean(losses)), **metrics)
        append_jsonl(out / 'metrics.jsonl', row)
        print(json.dumps(row), flush=True)
        if metrics['val_score'] > best:
            best = metrics['val_score']
            torch.save(dict(state_dict=det.state_dict(), encoder=cfg['encoder'], prompts=det.prompts, prompt_stats=stats,
                            mean=det.mean, std=det.std, image_size=cfg['image_size'], fold=a.fold, epoch=epoch,
                            exclude_benign_from_positive_patients=cfg['exclude_benign_from_positive_patients'],
                            n_folds=cfg['n_folds'], **metrics), out / 'best.pth')
            table.to_csv(out / 'val_predictions_best.csv', index=False)
    print(json.dumps(dict(best_val_score=best, checkpoint=str(out / 'best.pth'))))


if __name__ == '__main__':
    main()
