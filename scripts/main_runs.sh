#!/usr/bin/env bash
# Launch order used in the study (see docs/plan.md).
#   scripts/main_runs.sh seed1       ttt_on_seed1 and ttt_off_seed1 (3.0B tokens each)
#   scripts/main_runs.sh seed2       ttt_on_seed2 and ttt_off_seed2
#   scripts/main_runs.sh chunks      chunk512 and chunk1024: the ttt_on recipe and schedule, stopped by the
#                                    time limit after the 1B-token checkpoint (step 477) is saved, so the
#                                    learning-rate schedule matches the main arms' 1B checkpoints exactly
#   scripts/main_runs.sh calib       base model with and without YaRN at 32k and 64k (harness calibration)
#   scripts/main_runs.sh eval SEED   final checkpoints of one seed: ttt_on, its lesion (ttt_lr=0), ttt_off
#   scripts/main_runs.sh yarn SEED   the eval arms of one seed at 64k with YaRN x2 (the primary 64k setting)
#   scripts/main_runs.sh sweep SEED  inference-time ttt_lr sweep and clipping variant at 32k and 64k
#   scripts/main_runs.sh chunk_eval  1B-token checkpoints: chunk 512 / 1024 / 4096 and the 1B control
#   scripts/diagnostics.sh SEED     checks, gradient norms, delta norms and inference cost (one GPU)
# Env: SAMPLES (500), LENGTHS ("4k 8k 16k 32k 64k").
set -euo pipefail
cd "$(dirname "$0")/.."; source slurm/common.sh
export CPUS_PER_GPU=${CPUS_PER_GPU:-2} MEM_PER_GPU=${MEM_PER_GPU:-48}
SAMPLES=${SAMPLES:-500}; LENGTHS=${LENGTHS:-"4k 8k 16k 32k 64k"}
FINAL=checkpoints/global_step_1431/hf_ckpt; ONE_B=checkpoints/global_step_477/hf_ckpt
C=$STUDY_ROOT/checkpoints; BASE=$STUDY_ROOT/models/$MODEL
case "$1" in
  seed1) scripts/train.sh ttt_on_seed1  configs/train/ttt_on.yaml  1 4
         scripts/train.sh ttt_off_seed1 configs/train/ttt_off.yaml 1 4 ;;
  seed2) scripts/train.sh ttt_on_seed2  configs/train/ttt_on.yaml  2 4
         scripts/train.sh ttt_off_seed2 configs/train/ttt_off.yaml 2 4 ;;
  chunks) for c in 512 1024; do
            TIME=15:30:00 scripts/train.sh "chunk$c" configs/train/ttt_on.yaml 1 4 --model.foundation \
              "{\"ttt_layers\":[0,6,12,18,24,30],\"ttt_mode\":true,\"ttt_proj\":true,\"ttt_lr\":3,\"ttt_chunk\":$c}"
          done ;;
  calib) scripts/eval.sh calib "32k 64k" 100 base=$BASE base_yarn2=$BASE:yarn=2 ;;
  eval)  s=$2; scripts/eval.sh seed$s "$LENGTHS" "$SAMPLES" base=$BASE ttt_on_s$s=$C/ttt_on_seed$s/$FINAL \
           ttt_on_s${s}_frozen=$C/ttt_on_seed$s/$FINAL:ttt_lr=0 ttt_off_s$s=$C/ttt_off_seed$s/$FINAL ;;
  yarn)  s=$2; scripts/eval.sh yarn$s "64k" "$SAMPLES" base_yarn2=$BASE:yarn=2 ttt_on_s${s}_yarn2=$C/ttt_on_seed$s/$FINAL:yarn=2 \
           ttt_on_s${s}_frozen_yarn2=$C/ttt_on_seed$s/$FINAL:ttt_lr=0,yarn=2 ttt_off_s${s}_yarn2=$C/ttt_off_seed$s/$FINAL:yarn=2 ;;
  sweep) s=$2; scripts/eval.sh sweep$s "32k 64k" "$SAMPLES" ttt_on_s${s}_lr1=$C/ttt_on_seed$s/$FINAL:ttt_lr=1 \
           ttt_on_s${s}_lr6=$C/ttt_on_seed$s/$FINAL:ttt_lr=6 ttt_on_s${s}_clip=$C/ttt_on_seed$s/$FINAL:ttt_clip=1e-5 ;;
  chunk_eval) scripts/eval.sh chunk "32k 64k" "$SAMPLES" chunk512=$C/chunk512/$ONE_B chunk1024=$C/chunk1024/$ONE_B \
           chunk4096=$C/ttt_on_seed1/$ONE_B control_1b=$C/ttt_off_seed1/$ONE_B ;;
  *) echo "usage: $0 seed1|seed2|chunks|calib|eval SEED|yarn SEED|sweep SEED|chunk_eval"; exit 1 ;;
esac
