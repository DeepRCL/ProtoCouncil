"""Anatomical regions, region descriptors and the anatomical prototype head.

Regions (from each core's own 64x64 masks): the whole gland G, the needle track N (needle trace AND gland) and six
gland sectors (posterior / anterior halves by per-column relative depth x three equal-area thirds along the
horizontal axis). A region descriptor is the overlap-weighted mean of the 16x16 tokens in the region concatenated
with their channel-wise maximum over tokens at least half inside the region.

The head embeds the gland with the prompt-conditioned descriptor d_ctx(G) and the needle track and sectors with
d_enc(R) - d_enc(G) (image features relative to the core's own gland), using two projectors (one for the gland, one
shared by needle and sectors). Each family (gland, needle, sector) has 4 class-specific prototypes per class
(benign, GG1-2, GG>=3). A prototype's activation is its best rescaled cosine similarity over the regions of its
family, and each class logit is a bias plus a non-negatively weighted sum of the activations of that class's
prototypes:  z_c = b_c + sum_{k in P_c} softplus(rho_k) a_k.
"""
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.nn import functional as F

STATES = 3
PER_CLASS = 4
EPS = 1e-6
REGIONS = ('gland', 'needle', 'post_left', 'post_mid', 'post_right', 'ant_left', 'ant_mid', 'ant_right')
SECTORS = slice(2, 8)
FAMILIES = ('gland', 'needle', 'sector')
SECTOR_MIN_SUPPORT = .5  # token-area units
INIT_WEIGHT = 5.
# Projection candidates must be well-supported regions (token-area units).
EXEMPLAR_MIN_SUPPORT = {'gland': 30., 'needle': 4., 'sector': 4.}
CANCER_EXEMPLAR_MIN_INVOLVEMENT = .4


# ----------------------------------------------------------------------------- regions and descriptors
def regions(prostate, needle):
    """[B, 8, 64, 64] boolean regions from each core's own 64x64 prostate and needle masks."""
    pm, nm = prostate.bool(), needle.bool()
    B = len(pm)
    out = torch.zeros(B, len(REGIONS), 64, 64, dtype=torch.bool)
    out[:, 0] = pm
    out[:, 1] = pm & nm
    ys = torch.arange(64)[:, None].expand(64, 64)
    cols = torch.arange(64)[None].expand(64, 64).float()
    for b in range(B):
        p = pm[b]
        assert p.any(), 'empty prostate mask'
        has = p.any(0)
        top = torch.where(has, torch.where(p, ys, 99).min(0).values, 0)
        bottom = torch.where(has, torch.where(p, ys, -1).max(0).values, 0)
        depth = (ys - top[None]).float() / (bottom - top + 1).clamp_min(1)[None].float()
        xs = torch.nonzero(p)[:, 1].float()
        q1, q2 = torch.quantile(xs, torch.tensor([1 / 3, 2 / 3]))
        third = (cols >= q1).long() + (cols >= q2).long()
        for h in range(2):
            half = depth < .5 if h == 0 else depth >= .5
            for t in range(3):
                out[b, 2 + 3 * h + t] = p & half & (third == t)
    return out


def validity(occupancy):
    """Gland always valid; needle valid with any overlap; sectors need SECTOR_MIN_SUPPORT token-area units."""
    v = torch.zeros_like(occupancy, dtype=torch.bool)
    v[:, 0] = occupancy[:, 0] > 0
    v[:, 1] = occupancy[:, 1] > 0
    v[:, SECTORS] = occupancy[:, SECTORS] >= SECTOR_MIN_SUPPORT
    return v


def usable(valid):
    """Cores the head can score: valid gland, valid needle track and at least one valid sector."""
    return valid[:, 0] & valid[:, 1] & valid[:, SECTORS].any(1)


def descriptors(tokens, reg):
    """Region descriptors [B, 8, 2D] (mean, max) and occupancy [B, 8] from tokens [B, 256, D] and regions.

    The max uses tokens at least half inside the region, falling back to any overlapping token; empty regions get
    zeros and are marked invalid by ``validity``.
    """
    w = F.avg_pool2d(reg.float(), 4).flatten(2)
    x = tokens.float()
    mean = torch.einsum('brt,btd->brd', w, x) / w.sum(2, keepdim=True).clamp_min(EPS)
    inside = w >= .5
    sel = inside | ((w > 0) & ~inside.any(2, keepdim=True))
    mx = x[:, None].expand(-1, w.shape[1], -1, -1).masked_fill(~sel[..., None], float('-inf')).max(2).values
    occ = w.sum(2)
    mx = torch.where(occ[..., None] > 0, mx, torch.zeros_like(mx))
    return torch.cat([mean, mx], -1), occ


# ----------------------------------------------------------------------------- head
def projector(dim, latent=64):
    return nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 128), nn.GELU(), nn.Linear(128, latent))


class AnatomicalPrototypeHead(nn.Module):
    """Anatomical prototype head of one encoder.

    ``gland_dropout``: during training only, with this probability per core the gland-family activations are
    replaced by their stop-gradient batch mean (family balancing); inference always uses the full model.
    Inputs: ``ctx`` / ``enc`` region descriptors [B, 8, 2D] and region validity [B, 8].
    """

    families = FAMILIES

    def __init__(self, dim_ctx, dim_enc, gland_dropout=.5, per_class=PER_CLASS):
        super().__init__()
        assert 0 <= gland_dropout < 1
        self.dims = (dim_ctx, dim_enc)
        self.per_class = per_class
        self.gland_dropout = float(gland_dropout)
        self.f_abs = projector(dim_ctx)   # gland projector (prompt-conditioned descriptor)
        self.f_rel = projector(dim_enc)   # needle/sector projector (gland-relative image descriptor)
        n = STATES * per_class
        self.prototypes = nn.Parameter(F.normalize(torch.randn(len(self.families), n, 64), dim=-1))
        self.register_buffer('owner', torch.arange(STATES).repeat_interleave(per_class))
        # Identifiable positive weights w = softplus(rho), initialized at INIT_WEIGHT.
        self.rho = nn.Parameter(torch.full((len(self.families), n), float(np.log(np.expm1(INIT_WEIGHT)))))
        self.bias = nn.Parameter(torch.zeros(STATES))

    def embed(self, ctx, enc, valid):
        """{family: (unit embeddings [B, R, 64], valid [B, R], region indices)}."""
        def e(f, x):
            return F.normalize(f(x), dim=-1, eps=EPS)
        rel = enc - enc[:, :1]
        return {'gland': (e(self.f_abs, ctx[:, :1]), valid[:, :1], [0]),
                'needle': (e(self.f_rel, rel[:, 1:2]), valid[:, 1:2], [1]),
                'sector': (e(self.f_rel, rel[:, SECTORS]), valid[:, SECTORS], list(range(2, 8)))}

    def weights(self):
        return F.softplus(self.rho)

    def readout(self, act):
        """act [B, F, K] -> logits [B, 3], an exact sum of class-specific contributions plus bias."""
        contrib = act * self.weights()
        z = torch.zeros(len(act), STATES, dtype=act.dtype, device=act.device)
        return z.index_add(1, self.owner, contrib.sum(1)) + self.bias

    def forward(self, batch):
        emb = self.embed(batch['ctx'].float(), batch['enc'].float(), batch['valid'])
        acts, wins, sims = [], [], []
        for f, name in enumerate(self.families):
            h, v, _ = emb[name]
            s = (1 + torch.einsum('brd,kd->brk', h, F.normalize(self.prototypes[f], dim=-1, eps=EPS))) / 2
            s = s.masked_fill(~v[..., None], -1.)
            a, w = s.max(1)
            acts.append(a); wins.append(w); sims.append(s)
        act = torch.stack(acts, 1)
        assert (act >= 0).all(), 'a prototype family had no valid region'
        out = dict(logits=self.readout(act), activation=act, winner=torch.stack(wins, 1), similarity=sims, embeddings=emb)
        if self.training and self.gland_dropout > 0:
            drop = torch.rand(len(act), device=act.device) < self.gland_dropout
            if drop.any():
                act = act.clone()
                act[drop, 0] = act[:, 0].detach().mean(0, keepdim=True).expand(int(drop.sum()), -1)
                out['activation'] = act
                out['logits'] = self.readout(act)
        return out

    def contributions(self, out):
        """Exact per-prototype logit contributions w_k a_k [B, F, K] (to each prototype's own class logit)."""
        return out['activation'] * self.weights()


# ----------------------------------------------------------------------------- training objective
def class_weights(state):
    """Cross-entropy class weights proportional to the inverse square root of class frequency (mean 1)."""
    counts = torch.bincount(state, minlength=STATES).float()
    w = (counts.sum() / counts.clamp_min(1)).sqrt()
    return w / w.mean()


def prototype_losses(model, out, state):
    """Cluster / separation (ProtoPNet-style, on each prototype's best valid region) and within-class diversity,
    averaged over the three families."""
    cluster, separation, diversity = 0., 0., 0.
    for f, name in enumerate(model.families):
        h, v, _ = out['embeddings'][name]
        P = F.normalize(model.prototypes[f], dim=-1, eps=EPS)
        cos = torch.einsum('brd,kd->brk', h, P).masked_fill(~v[..., None], -2.).max(1).values
        own = model.owner[None] == state[:, None]
        cluster = cluster + (1 - cos.masked_fill(~own, -2.).max(1).values).mean()
        separation = separation + F.relu(cos.masked_fill(own, -2.).max(1).values - .2).mean()
        G = P @ P.T
        same = (model.owner[:, None] == model.owner[None]) & ~torch.eye(len(P), dtype=torch.bool, device=P.device)
        diversity = diversity + F.relu(G[same] - .3).square().mean()
    n = len(model.families)
    return cluster / n, separation / n, diversity / n


def objective(model, batch, weights, lambdas=(.1, .05, .1)):
    out = model(batch)
    loss = F.cross_entropy(out['logits'].float(), batch['state'], weight=weights)
    if model.prototypes.requires_grad:
        cl, sep, div = prototype_losses(model, out, batch['state'])
        loss = loss + lambdas[0] * cl + lambdas[1] * sep + lambdas[2] * div
    return loss, out


# ----------------------------------------------------------------------------- projection onto real training regions
@torch.no_grad()
def embed_all(model, ctx, enc, valid, batch=512):
    chunks = {}
    for i in range(0, len(valid), batch):
        e = model.embed(ctx[i:i + batch].float(), enc[i:i + batch].float(), valid[i:i + batch])
        for k, (h, v, idx) in e.items():
            chunks.setdefault(k, ([], [], idx))
            chunks[k][0].append(h); chunks[k][1].append(v)
    return {k: (torch.cat(a), torch.cat(b), idx) for k, (a, b, idx) in chunks.items()}


def _eligible(state, involvement, involvement_valid):
    return [(state == c) & ((c == 0) | (involvement_valid.bool() & (involvement >= CANCER_EXEMPLAR_MIN_INVOLVEMENT)))
            for c in range(STATES)]


@torch.no_grad()
def project(model, ctx, enc, valid, state, involvement, involvement_valid, patients, occupancy):
    """Replace every prototype by the embedding of a real training region of its family (in place).

    A single Hungarian assignment maps the prototypes to regions of distinct patients and maximizes total cosine
    similarity. Candidates are training cores of the prototype's class (cancer classes: involvement >= 40%) and
    regions with at least EXEMPLAR_MIN_SUPPORT token-area support. Returns one record per prototype.
    """
    emb = embed_all(model, ctx, enc, valid)
    names, pid = np.unique(np.asarray(patients, dtype=str), return_inverse=True)
    eligible = _eligible(state, involvement, involvement_valid)
    need = len(model.families) * model.per_class
    for c in range(STATES):
        n = len(set(np.asarray(patients, dtype=str)[eligible[c].numpy()]))
        assert n >= need, f'class {c} has only {n} eligible patients for projection (need {need})'
    rows = []
    for f, name in enumerate(model.families):
        h, v, idx = emb[name]
        support = v & (occupancy[:, idx] >= EXEMPLAR_MIN_SUPPORT[name])
        P = F.normalize(model.prototypes[f], dim=-1, eps=EPS)
        cos = torch.einsum('nrd,kd->nrk', h, P).masked_fill(~support[..., None], -3.)
        best, region = cos.max(1)
        for k in range(len(P)):
            rows.append((f, k, best[:, k].numpy(), region[:, k].numpy(), eligible[int(model.owner[k])].numpy()))
    cost = np.full((len(rows), len(names)), 1e3)
    choice = np.full((len(rows), len(names)), -1)
    for j, (f, k, best, region, ok) in enumerate(rows):
        ok = ok & (best > -2.5)
        assert ok.any(), ('no projection candidate', f, k)
        for core in np.flatnonzero(ok):
            if 1 - best[core] < cost[j, pid[core]]:
                cost[j, pid[core]] = 1 - best[core]; choice[j, pid[core]] = core
    r, c = linear_sum_assignment(cost)
    assert np.array_equal(r, np.arange(len(rows))) and (cost[r, c] < 1e3).all(), 'projection infeasible'
    records = []
    for j, col in zip(r, c):
        f, k, best, region, ok = rows[j]
        core = int(choice[j, col]); name = model.families[f]
        h, v, idx = emb[name]
        ri = int(region[core])
        model.prototypes.data[f, k] = h[core, ri]
        records.append(dict(family=name, prototype=int(k), state=int(model.owner[k]), core_index=core,
                            region=REGIONS[idx[ri]], region_index=int(idx[ri]), patient_id=str(names[col]),
                            cosine=float(best[core])))
    assert len({x['patient_id'] for x in records}) == len(records)
    return records


@torch.no_grad()
def grounding_error(model, ctx, enc, valid, records):
    """Max abs difference between each projected prototype and the embedding of its source region (should be ~0)."""
    err = 0.
    for x in records:
        f = model.families.index(x['family'])
        i = x['core_index']
        h, v, idx = model.embed(ctx[i:i + 1].float(), enc[i:i + 1].float(), valid[i:i + 1])[x['family']]
        ri = idx.index(x['region_index'])
        assert bool(v[0, ri])
        err = max(err, float((h[0, ri] - F.normalize(model.prototypes[f, x['prototype']], dim=-1)).abs().max()))
    return err


# ----------------------------------------------------------------------------- label-free fusion
def logit_scale(logits):
    """SD of class-centred logits over validation cores (label-free)."""
    z = torch.as_tensor(logits, dtype=torch.float32)
    s = float((z - z.mean(1, keepdim=True)).std())
    assert np.isfinite(s) and s > 0, ('invalid fusion scale', s)
    return s


def fusion_weights(scales):
    """Weights inversely proportional to each encoder's logit scale, summing to one."""
    inv = np.array([1 / s for s in scales], dtype=np.float64)
    return inv / inv.sum()


def fuse(logits, scales):
    """Fused logits: weighted mean of encoder logits (each still a bias plus a sum of prototype contributions)."""
    w = fusion_weights(scales)
    return sum(wi * np.asarray(z, dtype=np.float64) for wi, z in zip(w, logits))


def softmax(z):
    z = np.asarray(z, dtype=np.float64)
    e = np.exp(z - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)
