#!/usr/bin/env bash
# Train and apply ProtoCouncil for one cross-validation fold of the development cohort.
#
#   bash scripts/run_fold.sh FOLD DEV_MANIFEST [EXTERNAL_MANIFEST:NAME ...]
#
# Example: bash scripts/run_fold.sh 0 data/cohort_a.csv data/cohort_b.csv:cohort_b data/cohort_c.csv:cohort_c
# Pretrained weights are read from $CKPT (default: checkpoints/), outputs go to $RUNS (default: runs/).
set -euo pipefail
FOLD=$1; DEV=$2; shift 2
EXTERNAL=("$@")
RUNS=${RUNS:-runs}
CKPT=${CKPT:-checkpoints}
MEDSAM=${MEDSAM:-$CKPT/medsam_vit_b.pth}
DINOV3=${DINOV3:-$CKPT/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth}
VJEPA=${VJEPA:-$CKPT/vjepa2_1_vitl_dist_vitG_384.pt}
SEEDS=${SEEDS:-"7201 7202"}
EPOCH_ARG=${EPOCHS:+--epochs $EPOCHS}   # optional override for quick checks

# 1. MedSAM detector (adapters only).
python -m protocouncil.train_detector --config configs/detector_medsam.yaml --manifest "$DEV" --fold "$FOLD" \
  --medsam-checkpoint "$MEDSAM" --out "$RUNS/detectors/medsam/fold$FOLD" $EPOCH_ARG

# 2. DINOv3 and V-JEPA 2.1: adapt the backbone, then train the detector (decoders initialized from MedSAM's).
for ENC in dinov3 vjepa; do
  WEIGHTS=$([ "$ENC" = dinov3 ] && echo "$DINOV3" || echo "$VJEPA")
  python -m protocouncil.adapt --config "configs/adapt_$ENC.yaml" --weights "$WEIGHTS" --manifest "$DEV" --fold "$FOLD" \
    --out "$RUNS/adapt/$ENC/fold$FOLD" $EPOCH_ARG
  python -m protocouncil.train_detector --config "configs/detector_$ENC.yaml" --manifest "$DEV" --fold "$FOLD" \
    --medsam-checkpoint "$MEDSAM" --encoder-weights "$RUNS/adapt/$ENC/fold$FOLD/encoder.pth" \
    --init-decoders-from "$RUNS/detectors/medsam/fold$FOLD/best.pth" --out "$RUNS/detectors/$ENC/fold$FOLD" $EPOCH_ARG
done

# 3. Freeze each detector and store region descriptors of its two token grids.
for ENC in medsam dinov3 vjepa; do
  python -m protocouncil.extract --detector "$RUNS/detectors/$ENC/fold$FOLD/best.pth" --manifest "$DEV" \
    --out "$RUNS/packs/$ENC/fold$FOLD" ${EXTERNAL[@]+--external "${EXTERNAL[@]}"}
done

# 4. Anatomical prototype heads (CPU), two head seeds per encoder.
for ENC in medsam dinov3 vjepa; do
  for SEED in $SEEDS; do
    python -m protocouncil.train_heads --packs "$RUNS/packs/$ENC/fold$FOLD" --seed "$SEED" \
      --out "$RUNS/heads/$ENC/fold$FOLD/seed$SEED"
  done
done

# 5. Council predictions (fused + per-encoder probabilities) on the validation fold and every external cohort.
for SPLIT in val ${EXTERNAL[@]+"${EXTERNAL[@]##*:}"}; do
  python -m protocouncil.council predict --heads "$RUNS/heads" --packs "$RUNS/packs" --fold "$FOLD" --split="$SPLIT" \
    --seeds $SEEDS --out "$RUNS/council/fold$FOLD/$SPLIT.csv"
done
