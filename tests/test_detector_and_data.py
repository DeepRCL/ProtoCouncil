import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn

from protocouncil import data as D
from protocouncil.losses import involvement_loss, proportion_bce
from protocouncil.models.detector import Detector


class TinyEncoder(nn.Module):
    out_dim, mean, std = 32, (0., 0., 0.), (1., 1., 1.)

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 32, 16, 16)

    def forward(self, x):
        return self.conv(x)


def test_detector_outputs_and_token_grids():
    det = Detector(TinyEncoder()).eval()
    out = det(torch.randn(2, 3, 256, 256), torch.rand(2, 5), return_tokens=True)
    assert out['heatmap_logits'].shape == (2, 1, 64, 64)
    assert out['class_logits'].shape == (2, 3)
    assert out['enc'].shape == (2, 256, 32) and out['ctx'].shape == (2, 256, 256)


def test_involvement_loss_is_finite_when_saturated():
    for z in (-100., 100.):
        for t in (0., .5, 1.):
            logits = torch.full((16,), z, requires_grad=True)
            loss = proportion_bce(logits, torch.tensor(t))
            loss.backward()
            assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()
    h = torch.randn(2, 1, 64, 64)
    pm = torch.zeros(2, 1, 64, 64); pm[:, :, 10:50, 10:50] = 1
    nm = torch.zeros(2, 1, 64, 64); nm[0, :, 20:30] = 1             # core 1 has an empty needle track
    loss = involvement_loss(h, pm, nm, torch.tensor([.5, .2]))
    assert torch.isfinite(loss)


def make_manifest(tmp_path, n_patients=20):
    rows = []
    rng = np.random.default_rng(0)
    for p in range(n_patients):
        for k in range(3):
            cid = f'P{p}_{k}'
            img = (rng.random((120, 200)) * 255).astype(np.uint8)
            pm = np.zeros((120, 200), np.uint8); pm[20:100, 40:160] = 255
            nm = np.zeros((120, 200), np.uint8); nm[50:60, :] = 255
            for name, a in [('img', img), ('pm', pm), ('nm', nm)]:
                Image.fromarray(a).save(tmp_path / f'{cid}_{name}.png')
            gg = [0, 0, 2, 3][(p + k) % 4]
            rows.append(dict(core_id=cid, patient_id=f'P{p}', center='AB'[p % 2], image=f'{cid}_img.png',
                             prostate_mask=f'{cid}_pm.png', needle_mask=f'{cid}_nm.png', grade_group=gg,
                             involvement=0. if gg == 0 else .5, age=60 + p, psa=5., psad=np.nan if k == 0 else .1, base_apex=-1, mid_lateral=1))
    pd.DataFrame(rows).to_csv(tmp_path / 'manifest.csv', index=False)
    return tmp_path / 'manifest.csv'


def test_manifest_folds_and_dataset(tmp_path):
    df = D.assign_folds(D.read_manifest(make_manifest(tmp_path)), n_folds=5)
    assert df.groupby('patient_id').fold.nunique().max() == 1
    train, val = D.fold_split(df, 0)
    assert not set(train.patient_id) & set(val.patient_id)
    positive = set(train.patient_id[train.grade_group >= 1])
    assert not ((train.grade_group == 0) & train.patient_id.isin(positive)).any()
    stats = D.prompt_stats(train)
    item = D.CoreDataset(val, stats, (0, 0, 0), (1, 1, 1), augment='translate')[0]
    assert item['image'].shape == (3, 256, 256) and item['prostate_mask'].shape == (1, 64, 64)
    assert item['prompts'].shape == (5,) and torch.isfinite(item['prompts']).all()


def test_16_bit_images_keep_their_intensities(tmp_path):
    x = (np.arange(64 * 64).reshape(64, 64) * 16).astype(np.uint16)
    Image.fromarray(x).save(tmp_path / 'img16.png')
    y = D.load_array(tmp_path / 'img16.png')
    assert y.max() > 255 and np.array_equal(y, x.astype(np.float32))
