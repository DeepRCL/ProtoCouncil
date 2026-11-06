"""ProtoCouncil predictions, label-free fusion, thresholds and the encoder-disagreement gate.

    # fused predictions of one fold model for one pack (seed mean of the fused probabilities)
    python -m protocouncil.council predict --heads runs/heads --packs runs/packs --fold 0 --split val \
        --out runs/council/fold0/val.csv
    # csPCa thresholds at 80% specificity on pooled development-validation predictions
    python -m protocouncil.council calibrate --predictions runs/council/fold*/val.csv --out runs/council/thresholds.json
    # Table-1 metrics and selective prediction, mean +/- SD over fold models
    python -m protocouncil.council evaluate --predictions runs/council/fold*/cohort_b.csv --thresholds runs/council/thresholds.json

Fusion: fused logits are the weighted mean of the encoders' class logits with weights inversely proportional to each
encoder's label-free logit scale (SD of class-centred validation logits); a softmax gives P(cancer) = 1 - P(benign)
and P(csPCa) = P(GG>=3). Each fold model averages the probabilities of its head seeds.
Gate: every encoder and the fused model call csPCa when P(GG>=3) reaches its own threshold. The council reports the
fused call only when the three encoder calls agree and abstains otherwise. O = number of encoders whose call differs
from the fused call.
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit

from .metrics import auroc, detection_metrics
from .models import prototypes as P
from .train_heads import infer, load_head, load_pack

ENCODERS = ('medsam', 'dinov3', 'vjepa')


def predict(heads, packs, fold, split, encoders=ENCODERS, seeds=None):
    """Seed-mean fused and per-encoder probabilities of one fold model on ``packs/<enc>/fold<k>/<split>.pt``.

    ``seeds``: head seeds to average (e.g. [7201, 7202]); every encoder must have all of them. If omitted, the seeds
    found on disk are used and must be the same for every encoder.
    """
    found = {e: sorted(Path(d).name for d in glob.glob(f'{heads}/{e}/fold{fold}/seed*') if (Path(d) / 'head.pth').exists())
             for e in encoders}
    if seeds is None:
        seeds = found[encoders[0]]
        assert seeds and all(found[e] == seeds for e in encoders), f'head seeds differ between encoders: {found}'
    else:
        seeds = [f'seed{s}' for s in seeds]
        missing = {e: sorted(set(seeds) - set(found[e])) for e in encoders if set(seeds) - set(found[e])}
        assert not missing, f'missing trained heads: {missing}'
    probs = {k: [] for k in ('fused',) + tuple(encoders)}
    data = {e: load_pack(Path(packs) / e / f'fold{fold}' / f'{split}.pt') for e in encoders}
    meta = data[encoders[0]]
    for e in encoders:
        assert list(data[e]['core_id']) == list(meta['core_id']), 'packs of different encoders cover different cores'
    for s in seeds:
        z, scales = [], []
        for e in encoders:
            m, ck = load_head(Path(heads) / e / f'fold{fold}' / s / 'head.pth')
            logits = infer(m, data[e])['logits'].numpy()
            z.append(logits); scales.append(ck['fusion_scale'])
            probs[e].append(P.softmax(logits))
        probs['fused'].append(P.softmax(P.fuse(z, scales)))
    t = pd.DataFrame({k: meta[k] for k in ('core_id', 'patient_id', 'center')})
    t['grade_group'] = meta['grade_group'].numpy()
    t.loc[t.grade_group < 0, 'grade_group'] = np.nan
    t['involvement'] = meta['involvement'].numpy()
    p = np.mean(probs['fused'], 0)
    t['p0'], t['p1'], t['p2'] = p[:, 0], p[:, 1], p[:, 2]
    t['p_cancer'], t['p_cspca'] = 1 - p[:, 0], p[:, 2]
    for e in encoders:
        q = np.mean(probs[e], 0)
        t[f'p_cancer_{e}'], t[f'p_cspca_{e}'] = 1 - q[:, 0], q[:, 2]
    t.attrs['seeds'] = seeds
    return t


IDS = {'core_id': str, 'patient_id': str, 'center': str}


def read_predictions(path):
    """Prediction CSV with identifiers kept as strings (e.g. patient IDs with leading zeros)."""
    return pd.read_csv(path, dtype=IDS)


def calibrate(tables, specificity=.8, encoders=ENCODERS):
    """Per-model csPCa threshold at ``specificity`` on non-csPCa cores (GG<3) of pooled validation predictions."""
    t = pd.concat(tables, ignore_index=True)
    neg = t.grade_group.values < 3
    cols = {'fused': 'p_cspca', **{e: f'p_cspca_{e}' for e in encoders}}
    return {k: float(np.quantile(t[c].values[neg], specificity)) for k, c in cols.items()}


def council(t, thresholds, encoders=ENCODERS):
    """Adds csPCa calls, O (encoders opposing the fused call), the abstention flag and the confidence margin."""
    t = t.copy()
    t['call'] = t.p_cspca >= thresholds['fused']
    calls = np.column_stack([t[f'p_cspca_{e}'].values >= thresholds[e] for e in encoders])
    t['opposing'] = (calls != t.call.values[:, None]).sum(1)
    t['abstain'] = ~(calls.all(1) | (~calls).all(1))
    p = np.clip(t.p_cspca.values, 1e-7, 1 - 1e-7)
    t['confidence'] = np.abs(logit(p) - logit(thresholds['fused']))
    return t


def retained(t, keep):
    y, d = t.grade_group.values >= 3, t.call.values
    k = np.asarray(keep, bool)
    if not (y & k).any() or not (~y & k).any():  # balanced accuracy undefined without both classes
        return dict(coverage=float(k.mean()), bacc=float('nan'), auroc=float('nan'))
    sens = (d & y & k).sum() / (y & k).sum()
    spec = (~d & ~y & k).sum() / (~y & k).sum()
    return dict(coverage=float(k.mean()), bacc=float((sens + spec) / 2), auroc=auroc(y[k], t.p_cspca.values[k]))


def selective_metrics(t):
    """No abstention, disagreement gate, and the confidence gate at the disagreement gate's coverage."""
    out = {f'none_{k}': v for k, v in retained(t, np.ones(len(t))).items()}
    keep = ~t.abstain.values
    out.update({f'council_{k}': v for k, v in retained(t, keep).items()})
    n = int(keep.sum())
    conf = np.zeros(len(t), bool)
    conf[np.argsort(-t.confidence.values, kind='stable')[:n]] = True
    out.update({f'confidence_{k}': v for k, v in retained(t, conf).items()})
    correct = t.call.values == (t.grade_group.values >= 3)
    for o in range(4):
        m = t.opposing.values == o
        out[f'opposing{o}_share'] = float(m.mean())
        out[f'opposing{o}_accuracy'] = float(correct[m].mean()) if m.any() else float('nan')
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('predict')
    a.add_argument('--heads', required=True); a.add_argument('--packs', required=True)
    a.add_argument('--fold', type=int, required=True); a.add_argument('--split', required=True)
    a.add_argument('--out', required=True)
    a.add_argument('--encoders', nargs='+', default=list(ENCODERS))
    a.add_argument('--seeds', nargs='+', type=int, help='head seeds to average (default: all found, same for every encoder)')
    b = sub.add_parser('calibrate')
    b.add_argument('--predictions', nargs='+', required=True); b.add_argument('--out', required=True)
    b.add_argument('--specificity', type=float, default=.8)
    b.add_argument('--encoders', nargs='+', default=list(ENCODERS))
    c = sub.add_parser('evaluate')
    c.add_argument('--predictions', nargs='+', required=True, help='one file per fold model')
    c.add_argument('--thresholds', required=True)
    c.add_argument('--out')
    c.add_argument('--encoders', nargs='+', default=list(ENCODERS))
    args = p.parse_args()
    if args.cmd == 'predict':
        t = predict(args.heads, args.packs, args.fold, args.split, tuple(args.encoders), args.seeds)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        t.to_csv(args.out, index=False)
        print(f'{len(t)} cores, seeds {t.attrs["seeds"]} -> {args.out}')
    elif args.cmd == 'calibrate':
        th = calibrate([read_predictions(f) for f in args.predictions], args.specificity, tuple(args.encoders))
        Path(args.out).write_text(json.dumps(th, indent=1))
        print(json.dumps(th, indent=1))
    else:
        th = json.loads(Path(args.thresholds).read_text())
        rows = []
        for f in args.predictions:
            t = read_predictions(f).dropna(subset=['grade_group'])
            rows.append(dict(predictions=f, **detection_metrics(t), **selective_metrics(council(t, th, tuple(args.encoders)))))
        per = pd.DataFrame(rows)
        num = per.drop(columns='predictions')
        summary = pd.DataFrame({'mean': num.mean(), 'sd': num.std(ddof=0)})
        print((summary * 100).round(1).to_string())
        if args.out:
            per.to_csv(args.out, index=False)


if __name__ == '__main__':
    main()
