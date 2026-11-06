#!/usr/bin/env bash
# Train and apply ProtoCouncil for one cross-validation fold of the development cohort.
#
#   bash scripts/run_fold.sh FOLD DEV_MANIFEST [EXTERNAL_MANIFEST:NAME ...]
#
# Example: PROSTRL_SUP='prostrl/PPO-supervised-baseline-fold{fold}/best_rl.pth' \
#          PROSTRL_RL='prostrl/EXP4-pairwise-ranking-rl-fold{fold}/best_rl.pth' \
#          bash scripts/run_fold.sh 0 data/cohort_a.csv data/cohort_b.csv:cohort_b
#
# MedSAM:  with PROSTRL_SUP and PROSTRL_RL (Prost-RL checkpoints of each fold, '{fold}' is replaced) the MedSAM grids
#          come from Prost-RL as in the paper; without them a ProstNFound+-style MedSAM detector is trained.
# DINOv3 / V-JEPA 2.1: DETECTOR=minimal (default) trains the detector described in the paper;
#          DETECTOR=research uses the exact research recipe of the paper's detectors (research_detector/).
# Pretrained weights are read from $CKPT (default: checkpoints/), outputs go to $RUNS (default: runs/).
set -euo pipefail
FOLD=$1; DEV=$2; shift 2
EXTERNAL=("$@")
RUNS=${RUNS:-runs}
CKPT=${CKPT:-checkpoints}
MEDSAM=${MEDSAM:-$CKPT/medsam_vit_b.pth}
DINOV3=${DINOV3:-$CKPT/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth}
VJEPA=${VJEPA:-$CKPT/vjepa2_1_vitl_dist_vitG_384.pt}
DETECTOR=${DETECTOR:-minimal}
ANCHOR_BANK=${ANCHOR_BANK:-random}
SEEDS=${SEEDS:-"7201 7202"}
EPOCH_ARG=${EPOCHS:+--epochs $EPOCHS}   # optional override for quick checks (minimal detectors and adaptation)
D=$RUNS/detectors

# 1. MedSAM detector(s): ctx.pth gives the prompt-conditioned grid, enc.pth the encoder grid.
if [ -n "${PROSTRL_SUP:-}" ]; then
  python -m protocouncil.import_prostrl --supervised "${PROSTRL_SUP//\{fold\}/$FOLD}" --rl "${PROSTRL_RL//\{fold\}/$FOLD}" \
    --fold "$FOLD" --out "$D/medsam/fold$FOLD"
else
  python -m protocouncil.train_detector --config configs/detector_medsam.yaml --manifest "$DEV" --fold "$FOLD" \
    --medsam-checkpoint "$MEDSAM" --out "$D/medsam/fold$FOLD" $EPOCH_ARG
  cp "$D/medsam/fold$FOLD/best.pth" "$D/medsam/fold$FOLD/ctx.pth"; cp "$D/medsam/fold$FOLD/best.pth" "$D/medsam/fold$FOLD/enc.pth"
fi

# 2. DINOv3 and V-JEPA 2.1: adapt the backbone, then train the detector(s).
for ENC in dinov3 vjepa; do
  WEIGHTS=$([ "$ENC" = dinov3 ] && echo "$DINOV3" || echo "$VJEPA")
  python -m protocouncil.adapt --config "configs/adapt_$ENC.yaml" --weights "$WEIGHTS" --manifest "$DEV" --fold "$FOLD" \
    --out "$RUNS/adapt/$ENC/fold$FOLD" $EPOCH_ARG
  BACKBONE="$RUNS/adapt/$ENC/fold$FOLD/encoder.pth"
  if [ "$DETECTOR" = research ]; then
    : "${PROSTRL_SUP:?DETECTOR=research initializes its decoders from the Prost-RL supervised checkpoint (set PROSTRL_SUP)}"
    for CFG in $([ "$ENC" = dinov3 ] && echo "dinov3 dinov3_heatmap02" || echo vjepa); do
      python -m research_detector.train --config "research_detector/configs/$CFG.yaml" --manifest "$DEV" --fold "$FOLD" \
        --backbone-weights "$BACKBONE" --prostrl-supervised "${PROSTRL_SUP//\{fold\}/$FOLD}" --anchor-bank "$ANCHOR_BANK" \
        --out "$RUNS/research/$CFG/fold$FOLD" ${RESEARCH_SET:+--set $RESEARCH_SET}
    done
    ENC_CFG=$([ "$ENC" = dinov3 ] && echo dinov3_heatmap02 || echo vjepa)
    python -m research_detector.export --checkpoint "$RUNS/research/$ENC/fold$FOLD/best_interpretable.pth" --out "$D/$ENC/fold$FOLD/ctx.pth"
    python -m research_detector.export --checkpoint "$RUNS/research/$ENC_CFG/fold$FOLD/best_interpretable.pth" --out "$D/$ENC/fold$FOLD/enc.pth"
  else
    python -m protocouncil.train_detector --config "configs/detector_$ENC.yaml" --manifest "$DEV" --fold "$FOLD" \
      --medsam-checkpoint "$MEDSAM" --encoder-weights "$BACKBONE" --init-decoders-from "$D/medsam/fold$FOLD/ctx.pth" \
      --out "$D/$ENC/fold$FOLD" $EPOCH_ARG
    cp "$D/$ENC/fold$FOLD/best.pth" "$D/$ENC/fold$FOLD/ctx.pth"; cp "$D/$ENC/fold$FOLD/best.pth" "$D/$ENC/fold$FOLD/enc.pth"
  fi
done

# 3. Freeze the detectors and store region descriptors of their two token grids.
for ENC in medsam dinov3 vjepa; do
  python -m protocouncil.extract --detector "$D/$ENC/fold$FOLD/ctx.pth" --enc-detector "$D/$ENC/fold$FOLD/enc.pth" \
    --manifest "$DEV" --out "$RUNS/packs/$ENC/fold$FOLD" ${EXTERNAL[@]+--external "${EXTERNAL[@]}"}
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
