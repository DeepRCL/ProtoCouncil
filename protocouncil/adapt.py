"""Adapt a pretrained DINOv3 or V-JEPA 2.1 backbone to micro-ultrasound on one fold (before detector training).

    python -m protocouncil.adapt --config configs/adapt_dinov3.yaml --weights dinov3_vitl16_pretrain_lvd1689m.pth \
        --manifest data/cohort_a.csv --fold 0 --out runs/adapt/dinov3/fold0

The full backbone and a 1x1 convolution over its final 16x16 patch features are trained with the needle-track
involvement loss (proportion BCE in the needle trace AND prostate, positive cores weighted by ``pos_weight``).
The epoch with the best validation all-cancer AUROC of the mean needle-track heatmap is exported as
``encoder.pth``, which ``train_detector --encoder-weights`` then loads.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .losses import involvement_loss, needle_mean_probability
from .metrics import auroc
from .models.encoders import build_encoder
from .train_detector import loaders
from .utils import append_jsonl, autocast, device, load_config, set_seed, warmup_cosine


class DenseHead(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.conv = nn.Conv2d(encoder.out_dim, 1, kernel_size=1)

    def forward(self, x):
        return self.conv(self.encoder(x))


@torch.no_grad()
def evaluate(model, loader, dev, precision):
    model.eval()
    y, s = [], []
    for b in loader:
        with autocast(precision, dev):
            logits = model(b['image'].to(dev))
        s.append(needle_mean_probability(logits, b['prostate_mask'].to(dev), b['needle_mask'].to(dev)).cpu().numpy())
        y.append(b['grade_group'].numpy() >= 1)
    return auroc(np.concatenate(y), np.concatenate(s))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', required=True)
    p.add_argument('--weights', required=True, help='official pretrained backbone checkpoint')
    p.add_argument('--manifest', required=True)
    p.add_argument('--fold', type=int, required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--epochs', type=int, help='override the number of epochs in the config')
    a = p.parse_args()
    cfg = load_config(a.config, epochs=a.epochs)
    assert cfg['encoder'] in ('dinov3', 'vjepa'), 'backbone adaptation is used for DINOv3 and V-JEPA 2.1'
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    set_seed(cfg['seed'])
    dev = device()
    encoder = build_encoder(cfg['encoder'], a.weights)
    model = DenseHead(encoder).to(dev)
    tl, vl, _, train, val = loaders(cfg, a.manifest, a.fold, encoder.mean, encoder.std)
    enc_params = [q for q in encoder.parameters() if q.requires_grad]
    opt = torch.optim.AdamW([dict(params=enc_params, lr=cfg['encoder_lr']), dict(params=model.conv.parameters(), lr=cfg['lr'])],
                            weight_decay=cfg['weight_decay'])
    sched = warmup_cosine(opt, len(tl), cfg['epochs'], cfg['warmup_epochs'])
    (out / 'config.json').write_text(json.dumps(dict(cfg, fold=a.fold, weights=str(a.weights), train_cores=len(train),
                                                     val_cores=len(val)), indent=1))
    best = -np.inf
    for epoch in range(cfg['epochs']):
        model.train()
        losses = []
        for b in tl:
            pm, nm = b['prostate_mask'].to(dev), b['needle_mask'].to(dev)
            with autocast(cfg['precision'], dev):
                logits = model(b['image'].to(dev))
            loss = involvement_loss(logits, pm, nm, b['involvement'].to(dev), cfg['pos_weight'], cfg['entropy_weight'])
            if not torch.isfinite(loss):
                raise RuntimeError(f'non-finite loss at epoch {epoch}')
            if loss.requires_grad:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['clip_grad'])
                opt.step()
            sched.step()
            losses.append(float(loss.detach()))
        score = evaluate(model, vl, dev, cfg['precision'])
        row = dict(epoch=epoch, train_loss=float(np.mean(losses)), val_heatmap_all_auroc=score)
        append_jsonl(out / 'metrics.jsonl', row)
        print(json.dumps(row), flush=True)
        if score > best:
            best = score
            torch.save(dict(encoder=cfg['encoder'], encoder_state_dict=encoder.state_dict(), fold=a.fold, epoch=epoch,
                            val_heatmap_all_auroc=score), out / 'encoder.pth')
    print(json.dumps(dict(best_val_heatmap_all_auroc=best, checkpoint=str(out / 'encoder.pth'))))


if __name__ == '__main__':
    main()
