"""Case-based explanation of one core: exact prototype contributions and the training regions behind them.

    python -m protocouncil.explain --heads runs/heads --packs runs/packs --fold 0 --seed seed7201 --split val \
        --core-id CORE123 [--manifest data/cohort_a.csv --figure core123.png]

For each encoder e, the class logit is z_c^(e) = b_c + sum_{k in P_c} w_k a_k, and the fused logit is
z_c = sum_e alpha_e z_c^(e). The tool lists, for the predicted class, the prototypes with the largest fused
contributions alpha_e w_k a_k, the test region each one matched (a_k is its rescaled cosine similarity) and the
training core and region it was projected onto. With ``--figure`` (and manifests that contain the test core and the
training exemplars) it draws the test region (orange) next to each exemplar region (blue).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .council import ENCODERS
from .models import prototypes as P
from .train_heads import load_head, load_pack

CLASSES = ('benign', 'GG1-2', 'GG>=3')


def explain(heads, packs, fold, seed, split, core_id, top=3, encoders=ENCODERS):
    parts, z, scales = {}, [], []
    for e in encoders:
        c = load_pack(Path(packs) / e / f'fold{fold}' / f'{split}.pt')
        i = list(c['core_id']).index(core_id)
        m, ck = load_head(Path(heads) / e / f'fold{fold}' / seed / 'head.pth')
        with torch.no_grad():
            out = m(dict(ctx=c['desc_ctx'][i:i + 1], enc=c['desc_enc'][i:i + 1], valid=c['valid'][i:i + 1]))
            contrib = m.contributions(out)[0]
        logits = out['logits'][0]
        exact = m.bias.detach() + torch.zeros(P.STATES).index_add(0, m.owner, contrib.sum(0))
        assert torch.allclose(exact, logits, atol=1e-5)
        parts[e] = dict(m=m, ck=ck, out=out, contrib=contrib, logits=logits, pack=c, index=i)
        z.append(logits[None].numpy()); scales.append(ck['fusion_scale'])
    alpha = P.fusion_weights(scales)
    fused = P.fuse(z, scales)[0]
    pred = int(np.argmax(fused))
    report = dict(core_id=core_id, fold=fold, seed=seed, fused_logits=fused.tolist(),
                  probabilities=P.softmax(fused[None])[0].tolist(), predicted=CLASSES[pred], encoders={})
    for (e, d), a in zip(parts.items(), alpha):
        m, ck = d['m'], d['ck']
        recs = {(r['family'], r['prototype']): r for r in ck['projection']}
        enc = dict(weight=float(a), bias=m.bias.tolist(), logits=d['logits'].tolist(), top=[])
        for f, fam in enumerate(m.families):
            for k in range(len(m.owner)):
                if int(m.owner[k]) != pred:
                    continue
                r = recs[(fam, k)]
                region = int(d['out']['winner'][0, f, k]) + (0 if fam == 'gland' else 1 if fam == 'needle' else 2)
                enc['top'].append(dict(family=fam, prototype=k, fused_contribution=float(a * d['contrib'][f, k]),
                                       similarity=float(d['out']['activation'][0, f, k]), weight=float(m.weights()[f, k].detach()),
                                       test_region=P.REGIONS[region], exemplar_core=r['core_id'],
                                       exemplar_region=r['region'], exemplar_grade_group=r.get('grade_group'),
                                       exemplar_involvement=r.get('involvement'), exemplar_index=r['core_index']))
        enc['top'] = sorted(enc['top'], key=lambda x: -x['fused_contribution'])[:top]
        report['encoders'][e] = enc
    return report, parts


def _region_mask(prostate64, needle64, region):
    reg = P.regions(prostate64[None], needle64[None])[0, P.REGIONS.index(region)].float()
    return F.interpolate(reg[None, None], size=(256, 256), mode='nearest')[0, 0].numpy() > .5


def figure(report, parts, manifests, path):
    import matplotlib.pyplot as plt
    import pandas as pd
    from .data import load_array, read_manifest
    table = pd.concat([read_manifest(m) for m in manifests]).drop_duplicates('core_id').set_index('core_id')

    def image(core):
        x = torch.from_numpy(load_array(table.loc[core, 'image']))[None, None]
        x = F.interpolate(x, size=(256, 256), mode='bilinear', antialias=True)[0, 0].numpy()
        return (x - x.min()) / max(1e-6, x.max() - x.min())

    items = [(e, t) for e, enc in report['encoders'].items() for t in enc['top']]
    train = {e: load_pack(Path(d['pack_dir']) / 'train.pt') for e, d in parts.items()}
    fig, axes = plt.subplots(len(items), 2, figsize=(5, 2.5 * len(items)), squeeze=False)
    for row, (e, t) in enumerate(items):
        c, i = parts[e]['pack'], parts[e]['index']
        for col, (core, region, colour) in enumerate([(report['core_id'], t['test_region'], 'tab:orange'),
                                                       (t['exemplar_core'], t['exemplar_region'], 'tab:blue')]):
            ax = axes[row, col]
            ax.imshow(image(core), cmap='gray'); ax.axis('off')
            src, j = (c, i) if col == 0 else (train[e], t['exemplar_index'])
            ax.contour(_region_mask(src['prostate64'][j], src['needle64'][j], region), levels=[.5], colors=colour, linewidths=1.5)
        axes[row, 0].set_title(f'{e} | {t["family"]}: {t["test_region"]}  a={t["similarity"]:.2f}', fontsize=8)
        axes[row, 1].set_title(f'exemplar GG{t["exemplar_grade_group"]}, +{t["fused_contribution"]:.2f}', fontsize=8)
    fig.suptitle(f'{report["core_id"]}: {report["predicted"]} (P(GG>=3) = {report["probabilities"][2]:.2f})', fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 1 - .3 / len(items)))
    fig.savefig(path, dpi=200)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--heads', required=True); p.add_argument('--packs', required=True)
    p.add_argument('--fold', type=int, required=True); p.add_argument('--seed', default='seed7201')
    p.add_argument('--split', required=True); p.add_argument('--core-id', required=True)
    p.add_argument('--top', type=int, default=3)
    p.add_argument('--manifest', nargs='*', default=[], help='manifests with image paths (test core and training cores)')
    p.add_argument('--figure')
    p.add_argument('--encoders', nargs='+', default=list(ENCODERS))
    a = p.parse_args()
    report, parts = explain(a.heads, a.packs, a.fold, a.seed, a.split, a.core_id, a.top, tuple(a.encoders))
    print(json.dumps(report, indent=1))
    if a.figure:
        for e in parts:
            parts[e]['pack_dir'] = Path(a.packs) / e / f'fold{a.fold}'
        figure(report, parts, a.manifest, a.figure)


if __name__ == '__main__':
    main()
