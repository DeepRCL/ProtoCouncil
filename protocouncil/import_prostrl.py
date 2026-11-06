"""Use trained Prost-RL models as the MedSAM detector of ProtoCouncil.

In the paper, the MedSAM branch reads the Prost-RL detectors of each fold (https://github.com/DeepRCL/Prost-RL):
    ctx grid  <- supervised stage (cfg/train/experiments/ppo/supervised_baseline.yaml), class-decoder image tokens
    enc grid  <- RL stage (cfg/train/exp4_prost_rl.yaml), MedSAM image-encoder output

    python -m protocouncil.import_prostrl --supervised <prost-rl>/PPO-supervised-baseline-fold0/best_rl.pth \
        --rl <prost-rl>/EXP4-pairwise-ranking-rl-fold0/best_rl.pth --fold 0 --out runs/detectors/medsam/fold0

writes ``ctx.pth`` and ``enc.pth`` (ProtoCouncil detector checkpoints) for
    python -m protocouncil.extract --detector runs/detectors/medsam/fold0/ctx.pth \
        --enc-detector runs/detectors/medsam/fold0/enc.pth ...
The fold recorded in each checkpoint must match ``--fold`` (patient assignments must also be the same folds as
the manifest's). Context tokens are computed by the ProstNFound+ part of the supervised model (image encoder, prompts, class decoder);
the Prost-RL attention policy is not used for them. Clinical prompts are scaled with Prost-RL's own constants.
"""
import argparse
from pathlib import Path

import torch

from .models.detector import PROMPTS, Detector
from .models.encoders import MedSAMEncoder

PREFIX = 'model.prostnfound.'
# Prost-RL / ProstNFound+ prompt scaling: (x - min) / (max - min), missing values -> mean.
PROSTRL_PROMPT_STATS = {'age': dict(min=0., max=79., mean=62.5816),
                        'psa': dict(min=0.2, max=32.95, mean=6.821426488456866),
                        'psad': dict(min=4.615739672282483e-06, max=0.000837278201784, mean=0.000175347951594383)}
PROMPT_NAMES = {'age': 'age', 'psa': 'psa', 'approx_psa_density': 'psad', 'base_apex_encoding': 'base_apex',
                'mid_lateral_encoding': 'mid_lateral'}


def convert(path):
    """Prost-RL checkpoint -> (ProtoCouncil Detector, Prost-RL training config)."""
    ck = torch.load(path, map_location='cpu', weights_only=False)
    sd, args = ck['model'], ck.get('args', {})
    args = args if isinstance(args, dict) else vars(args)
    new = {}
    for k, v in sd.items():
        if not k.startswith(PREFIX):
            continue  # attention policy, value network, wrapper buffers
        k = k[len(PREFIX):]
        if k.startswith('medsam_model.image_encoder.'):
            new['encoder.encoder.' + k[len('medsam_model.image_encoder.'):]] = v
        elif k.startswith('medsam_model.prompt_encoder.'):
            new['prompt_encoder.' + k[len('medsam_model.prompt_encoder.'):]] = v
        elif k.startswith('medsam_model.mask_decoder.') and '_output_upscaling' not in k:
            new['mask_decoder.' + k[len('medsam_model.mask_decoder.'):]] = v
        elif k.startswith('class_decoder.'):
            new[k] = v
        elif k.startswith('floating_point_prompt_modules.'):
            name, rest = k[len('floating_point_prompt_modules.'):].split('.', 1)
            new[f'prompt_mlps.{PROMPT_NAMES[name]}.{rest}'] = v
    for k in ('temperature', 'bias'):  # identity heatmap calibration in released models
        if PREFIX + k in sd:
            assert float(sd[PREFIX + k]) == (1. if k == 'temperature' else 0.), (k, sd[PREFIX + k])
    num_classes = new['class_decoder.output_cls_mlps.0.layers.2.weight'].shape[0]
    det = Detector(MedSAMEncoder(), prompts=PROMPTS, num_classes=num_classes)
    det.load_state_dict(new, strict=True)
    return det, args, num_classes


def recorded_split(args):
    """(fold, n_folds) recorded in a Prost-RL checkpoint's training config, or (None, None)."""
    data = args.get('data', {}) if isinstance(args, dict) else {}
    fold, n = data.get('fold'), data.get('n_folds')
    return (None if fold is None else int(fold)), (None if n is None else int(n))


def check_split(args, fold, n_folds, path):
    """The checkpoint must have been trained for the requested fold (when its config records one)."""
    rec_fold, rec_n = recorded_split(args)
    if rec_fold is not None and rec_fold != fold:
        raise ValueError(f'{path} was trained for fold {rec_fold}, not fold {fold}')
    if rec_n is not None and rec_n != n_folds:
        raise ValueError(f'{path} was trained with {rec_n} folds, not {n_folds}')
    if rec_fold is None or rec_n is None:
        print(f'warning: {path} does not record its fold split; make sure it was trained without the patients of fold {fold}')


def save(det, num_classes, args, fold, n_folds, path):
    data = args.get('data', {}) if isinstance(args, dict) else {}
    torch.save(dict(state_dict=det.state_dict(), encoder='medsam', prompts=det.prompts, num_classes=num_classes,
                    prompt_stats=PROSTRL_PROMPT_STATS, mean=(0., 0., 0.), std=(1., 1., 1.),
                    image_size=int(data.get('image_size', 256)), fold=fold, n_folds=n_folds,
                    exclude_benign_from_positive_patients=bool(data.get('exclude_benign_cores_from_positive_patients', True)),
                    source='Prost-RL', source_name=args.get('name')), path)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--supervised', required=True, help='Prost-RL supervised-stage checkpoint (context grid)')
    p.add_argument('--rl', required=True, help='Prost-RL RL-stage checkpoint (encoder grid)')
    p.add_argument('--fold', type=int, required=True)
    p.add_argument('--n-folds', type=int, default=5)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for src, name in ((a.supervised, 'ctx.pth'), (a.rl, 'enc.pth')):
        det, args, n = convert(src)
        check_split(args, a.fold, a.n_folds, src)
        save(det, n, args, a.fold, a.n_folds, out / name)
        print(f'{src} -> {out / name} ({n}-class Prost-RL detector)')


if __name__ == '__main__':
    main()
