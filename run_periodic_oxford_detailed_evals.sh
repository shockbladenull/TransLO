#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/Localize/ljc/Projects/TransLO"
CKPT_GLOB="${REPO_ROOT}/experiment/oxford_0226_train_0552_val_ddp/checkpoints/translonet/*.pth.tar"
OXFORD_ROOT="/Localize/ljc/Dataset/Oxford"
OXFORD_H5_ROOT="/home/ljc/Downloads/2-h5data"
OXFORD_FULL_H5_ROOT="/home/ljc/Downloads/2-h5data"
OXFORD_POSE_ROOT="/home/ljc/Downloads/QEOxford"
GPU_IDS="0,1,2,4"
JOBS_PER_GPU=4
WORKERS=2
AFTER_EPOCH=100
EPOCH_STRIDE=5

run_eval() {
  local sequence_name="$1"
  local mask_name="$2"
  local output_dir="$3"

  echo
  echo "============================================================"
  echo "Running detailed periodic eval"
  echo "  sequence: ${sequence_name}"
  echo "  mask:     ${mask_name}"
  echo "  output:   ${output_dir}"
  echo "============================================================"

  pixi run python oxford_lo300_rank_ckpts.py \
    --ckpt_glob "${CKPT_GLOB}" \
    --oxford_root "${OXFORD_ROOT}" \
    --oxford_h5_root "${OXFORD_H5_ROOT}" \
    --oxford_full_h5_root "${OXFORD_FULL_H5_ROOT}" \
    --oxford_pose_root "${OXFORD_POSE_ROOT}" \
    --oxford_eval_seq "${sequence_name}" \
    --oxford_eval_mask_name "${mask_name}" \
    --output_dir "${output_dir}" \
    --gpu_ids "${GPU_IDS}" \
    --jobs_per_gpu "${JOBS_PER_GPU}" \
    --workers "${WORKERS}" \
    --after_epoch "${AFTER_EPOCH}" \
    --epoch_stride "${EPOCH_STRIDE}"
}

cd "${REPO_ROOT}"

run_eval \
  "2019-01-17-14-03-00-radar-oxford-10k" \
  "velodyne_left_calibrateFalse_LO300m.h5" \
  "${REPO_ROOT}/experiment/oxford_periodic_eval_0300_lo"

run_eval \
  "2019-01-14-12-05-52-radar-oxford-10k" \
  "velodyne_left_calibrateFalse_LO300m.h5" \
  "${REPO_ROOT}/experiment/oxford_periodic_eval_0552_lo"

run_eval \
  "2019-01-11-14-02-26-radar-oxford-10k" \
  "velodyne_left_calibrateFalse_SCR300m.h5" \
  "${REPO_ROOT}/experiment/oxford_periodic_eval_0226_scr"

echo
echo "All periodic detailed evaluations finished."
