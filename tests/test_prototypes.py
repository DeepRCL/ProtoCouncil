import numpy as np
import torch

from protocouncil.models import prototypes as P
from protocouncil import train_heads as th


def ellipse(cx=32, cy=30, rx=22, ry=14):
    y, x = torch.meshgrid(torch.arange(64), torch.arange(64), indexing='ij')
    return ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1


def needle():
    m = torch.zeros(64, 64, dtype=torch.bool)
    for i in range(64):
        m[i, max(0, i - 2):i + 2] = True
    return m


def synthetic_pack(n=180, d=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    pm = torch.stack([ellipse(32 + i % 5, 30) for i in range(n)])
    nm = needle()[None].expand(n, -1, -1).clone()
    reg = P.regions(pm, nm)
    gg = torch.tensor([0, 1, 2, 3, 4, 5] * (n // 6))
    tokens = torch.randn(n, 256, d, generator=g) + gg[:, None, None].float() * .3
    ctx, occ = P.descriptors(tokens.half(), reg)
    enc, _ = P.descriptors(tokens.half() * 2, reg)
    return dict(core_id=[f'c{i}' for i in range(n)], patient_id=[f'p{i}' for i in range(n)], center=['X'] * n,
                grade_group=gg, involvement=torch.where(gg > 0, torch.tensor(.6), torch.tensor(0.)),
                prostate64=pm, needle64=nm, desc_ctx=ctx.half(), desc_enc=enc.half(), occupancy=occ, valid=P.validity(occ))


def test_regions_partition_gland():
    pm, nm = ellipse()[None], needle()[None]
    reg = P.regions(pm, nm)
    sectors = reg[0, P.SECTORS]
    assert torch.equal(sectors.any(0), pm[0])                      # sectors cover the gland
    assert int(sectors.sum(0).max()) == 1                           # and do not overlap
    assert torch.equal(reg[0, 1], pm[0] & nm[0])                    # needle track = needle AND gland


def test_descriptor_mean_and_max():
    reg = P.regions(ellipse()[None], needle()[None])
    tokens = torch.zeros(1, 256, 2)
    tokens[0, :, 0] = 1.
    tokens[0, 100, 1] = 7.
    d, occ = P.descriptors(tokens, reg)
    assert torch.allclose(d[0, 0, 0], torch.tensor(1.))             # mean of a constant channel
    assert occ[0, 0] > 0 and d.shape == (1, 8, 4)


def test_logits_are_exact_sums_of_prototype_contributions():
    c = synthetic_pack()
    torch.manual_seed(0)
    m = P.AnatomicalPrototypeHead(c['desc_ctx'].shape[-1], c['desc_enc'].shape[-1]).eval()
    out = m(th.batch({**c, 'state': torch.zeros(len(c['core_id']), dtype=torch.long)}, slice(0, 20)))
    contrib = m.contributions(out)
    z = m.bias + torch.zeros(len(contrib), 3).index_add(1, m.owner, contrib.sum(1))
    assert torch.allclose(z, out['logits'], atol=1e-5)
    assert (contrib >= 0).all()                                     # every contribution is non-negative


def test_gland_dropout_only_in_training():
    c = synthetic_pack()
    m = P.AnatomicalPrototypeHead(c['desc_ctx'].shape[-1], c['desc_enc'].shape[-1], gland_dropout=.99)
    b = th.batch({**c, 'state': torch.zeros(len(c['core_id']), dtype=torch.long)}, slice(0, 30))
    m.eval()
    a = m(b)['activation']
    m.train()
    a2 = m(b)['activation']
    assert torch.allclose(a[:, 1:], a2[:, 1:]) and not torch.allclose(a[:, 0], a2[:, 0])


def test_projection_grounds_prototypes_on_distinct_patients():
    c = synthetic_pack()
    c['state'] = torch.as_tensor(th.three_state(c['grade_group'].numpy()))
    torch.manual_seed(0)
    m = P.AnatomicalPrototypeHead(c['desc_ctx'].shape[-1], c['desc_enc'].shape[-1])
    recs = P.project(m, c['desc_ctx'], c['desc_enc'], c['valid'], c['state'], c['involvement'],
                     torch.ones(len(c['core_id']), dtype=torch.bool), c['patient_id'], c['occupancy'])
    assert len(recs) == 36 and len({r['patient_id'] for r in recs}) == 36
    assert all(int(c['state'][r['core_index']]) == r['state'] for r in recs)
    assert P.grounding_error(m, c['desc_ctx'], c['desc_enc'], c['valid'], recs) < 1e-5


def test_fusion_preserves_additivity():
    z1, z2 = np.random.randn(5, 3), np.random.randn(5, 3)
    w = P.fusion_weights([2., 1.])
    assert np.allclose(w, [1 / 3, 2 / 3])
    assert np.allclose(P.fuse([z1, z2], [2., 1.]), w[0] * z1 + w[1] * z2)


def test_train_heads_end_to_end(tmp_path, monkeypatch):
    tr, va = synthetic_pack(seed=0), synthetic_pack(n=60, seed=1)
    va['patient_id'] = [f'q{i}' for i in range(60)]
    monkeypatch.setattr(th, 'EPOCHS', 2); monkeypatch.setattr(th, 'REFIT_EPOCHS', 2)
    torch.save(tr, tmp_path / 'train.pt'); torch.save(va, tmp_path / 'val.pt')
    th.train(tmp_path, tmp_path / 'out', seed=7201, threads=1)
    m, ck = th.load_head(tmp_path / 'out' / 'head.pth')
    assert ck['fusion_scale'] > 0 and len(ck['projection']) == 36
