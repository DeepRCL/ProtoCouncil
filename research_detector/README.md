# Research detector (exact DINOv3 / V-JEPA 2.1 recipe)

This folder contains the code that trained the DINOv3 and V-JEPA 2.1 detectors behind the numbers in the paper. Use it if you want that recipe exactly. For a smaller detector that follows the description in the paper, use `protocouncil.train_detector` instead.

It is research code. Besides the backbone and the prompt-conditioned class decoders that give the context grid, its training objective includes auxiliary terms from our earlier experiments: occurrence prototypes, a hyperbolic whole-needle branch, a needle-axis involvement loss and a semantic-anchor loss. Their settings are fixed in `configs/`. Many other options in the code are switched off by these configs.

## Configs

| Config | Used for |
|---|---|
| `configs/dinov3.yaml` | DINOv3 context grid (heatmap loss weight 1.0) |
| `configs/dinov3_heatmap02.yaml` | DINOv3 encoder grid (heatmap loss weight 0.2) |
| `configs/vjepa.yaml` | V-JEPA 2.1 context and encoder grids |

Each detector starts from a backbone adapted with `protocouncil.adapt`. Its prompt encoder, clinical prompt modules and both class decoders start from the Prost-RL supervised checkpoint of the same fold.

## Usage

```bash
python -m research_detector.train --config research_detector/configs/dinov3.yaml --manifest data/cohort_a.csv --fold 0 \
    --backbone-weights runs/adapt/dinov3/fold0/encoder.pth \
    --prostrl-supervised prostrl/PPO-supervised-baseline-fold0/best_rl.pth --out runs/research/dinov3/fold0
python -m research_detector.train --config research_detector/configs/dinov3_heatmap02.yaml ... --out runs/research/dinov3_heatmap02/fold0
python -m research_detector.export --checkpoint runs/research/dinov3/fold0/best_interpretable.pth --out runs/detectors/dinov3/fold0/ctx.pth
python -m research_detector.export --checkpoint runs/research/dinov3_heatmap02/fold0/best_interpretable.pth --out runs/detectors/dinov3/fold0/enc.pth
python -m protocouncil.extract --detector runs/detectors/dinov3/fold0/ctx.pth --enc-detector runs/detectors/dinov3/fold0/enc.pth \
    --manifest data/cohort_a.csv --out runs/packs/dinov3/fold0
```

`scripts/run_fold.sh` does all of this with `DETECTOR=research`. Extra settings can be overridden with `--set key=value` (OmegaConf syntax).

**Semantic-anchor bank.** The paper's detectors used a bank of 60 projected PANDA tile prototypes from a pathology model. This bank is derived from PANDA (CC BY-NC-SA 4.0) and Prov-GigaPath features (research-use license), so it is not included. Pass it with `--anchor-bank bank.npz` if you have it (arrays `anchors` [60, 256], `state_labels`, `grade_labels`, `anchor_ids`). The default, `--anchor-bank random`, is the matched random bank we used as a control (unit Gaussian vectors with the same grade counts, seed 92121).

Training on GPU is not bit-reproducible, so retrained detectors differ slightly from ours even with identical code and data.
