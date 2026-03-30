#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TASK_NAME="${TASK_NAME:-oxford_0226_scr_turning_from_ep060}"
CKPT_DIR="${CKPT_DIR:-${PROJECT_ROOT}/experiment/${TASK_NAME}/checkpoints/translonet}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/experiment/${TASK_NAME}/eval/oxford_detailed}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/experiment/${TASK_NAME}/logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/oxford_detailed_watcher.log}"

TMPDIR="${TMPDIR:-/Localize/ljc/tmp}"
GPU_ID="${GPU_ID:-1}"
POLL_SECONDS="${POLL_SECONDS:-60}"
AFTER_EPOCH="${AFTER_EPOCH:-185}"
EPOCH_STRIDE="${EPOCH_STRIDE:-5}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-128}"
WORKERS="${WORKERS:-6}"

OXFORD_ROOT="${OXFORD_ROOT:-/Localize/ljc/Dataset/Oxford}"
OXFORD_H5_ROOT="${OXFORD_H5_ROOT:-/home/ljc/Downloads/2-h5data}"
OXFORD_FULL_H5_ROOT="${OXFORD_FULL_H5_ROOT:-/home/ljc/Downloads/2-h5data}"
OXFORD_FULL_H5_NAME="${OXFORD_FULL_H5_NAME:-velodyne_left_calibrateFalse.h5}"
OXFORD_POSE_ROOT="${OXFORD_POSE_ROOT:-/home/ljc/Downloads/QEOxford}"
OXFORD_POSE_TXT_TEMPLATE="${OXFORD_POSE_TXT_TEMPLATE:-Oxford_SLAM_result_{sequence_short}/gicp_Oxford{sequence_short}_050_v1.txt}"
OXFORD_POSE_SKIP_START="${OXFORD_POSE_SKIP_START:-5}"
OXFORD_POSE_SKIP_END="${OXFORD_POSE_SKIP_END:-5}"
OXFORD_TRIM_EDGES="${OXFORD_TRIM_EDGES:-0}"
FRAME_GAP="${FRAME_GAP:-1}"

# Default route choice matches the current history under eval/oxford_detailed:
# continue writing 0226 SCR outputs after epoch 185. Set WATCH_ROUTES=0226,0300
# if later epochs should also include the 0300 LO route.
WATCH_ROUTES="${WATCH_ROUTES:-0226}"

mkdir -p "${TMPDIR}" "${OUTPUT_ROOT}" "${LOG_DIR}"

log() {
  local message="$1"
  printf '[%s] %s\n' "$(date '+%F %T')" "${message}" | tee -a "${LOG_FILE}"
}

append_route_specs() {
  local route_key="$1"
  case "${route_key}" in
    0226)
      ROUTE_SPECS+=("2019-01-11-14-02-26-radar-oxford-10k|velodyne_left_calibrateFalse_SCR300m.h5")
      ;;
    0300)
      ROUTE_SPECS+=("2019-01-17-14-03-00-radar-oxford-10k|velodyne_left_calibrateFalse_LO300m.h5")
      ;;
    *)
      log "unknown WATCH_ROUTES entry: ${route_key}"
      exit 1
      ;;
  esac
}

ROUTE_SPECS=()
IFS=',' read -r -a requested_routes <<< "${WATCH_ROUTES}"
for route_key in "${requested_routes[@]}"; do
  route_key="${route_key//[[:space:]]/}"
  [[ -n "${route_key}" ]] || continue
  append_route_specs "${route_key}"
done

if [[ "${#ROUTE_SPECS[@]}" -eq 0 ]]; then
  log "WATCH_ROUTES resolved to zero routes"
  exit 1
fi

run_route_eval() {
  local ckpt_path="$1"
  local epoch_str="$2"
  local sequence_name="$3"
  local mask_name="$4"
  local output_dir="${OUTPUT_ROOT}/epoch_${epoch_str}/${sequence_name}"

  mkdir -p "${output_dir}"

  log "evaluating epoch=${epoch_str} sequence=${sequence_name} mask=${mask_name}"
  TMPDIR="${TMPDIR}" CUDA_VISIBLE_DEVICES="${GPU_ID}" pixi run python - \
    "${ckpt_path}" \
    "${output_dir}" \
    "${sequence_name}" \
    "${mask_name}" \
    "${OXFORD_ROOT}" \
    "${OXFORD_H5_ROOT}" \
    "${OXFORD_FULL_H5_ROOT}" \
    "${OXFORD_FULL_H5_NAME}" \
    "${OXFORD_POSE_ROOT}" \
    "${OXFORD_POSE_TXT_TEMPLATE}" \
    "${OXFORD_POSE_SKIP_START}" \
    "${OXFORD_POSE_SKIP_END}" \
    "${OXFORD_TRIM_EDGES}" \
    "${FRAME_GAP}" \
    "${EVAL_BATCH_SIZE}" \
    "${WORKERS}" \
    >> "${LOG_FILE}" 2>&1 <<'PY'
import json
import os
import sys
import time

from configs import finalize_translonet_args
from oxford_lo300_eval import build_parser, evaluate_segment, load_checkpoint_model, setup_device
from tools.oxford_train_eval import build_oxford_detailed_summary, load_oxford_detailed_sequence
from tools.oxford_eval_tools import qe_pose_vectors_to_matrices, save_full_route_plots

(
    ckpt_path,
    output_dir,
    sequence_name,
    mask_name,
    oxford_root,
    oxford_h5_root,
    oxford_full_h5_root,
    oxford_full_h5_name,
    oxford_pose_root,
    oxford_pose_txt_template,
    oxford_pose_skip_start,
    oxford_pose_skip_end,
    oxford_trim_edges,
    frame_gap,
    eval_batch_size,
    workers,
) = sys.argv[1:]

parser = build_parser()
args = finalize_translonet_args(
    parser.parse_args(
        [
            "--gpu",
            "0",
            "--ckpt",
            ckpt_path,
            "--output_dir",
            output_dir,
            "--oxford_root",
            oxford_root,
            "--oxford_h5_root",
            oxford_h5_root,
            "--oxford_full_h5_root",
            oxford_full_h5_root,
            "--oxford_full_h5_name",
            oxford_full_h5_name,
            "--oxford_pose_root",
            oxford_pose_root,
            "--oxford_pose_txt_template",
            oxford_pose_txt_template,
            "--oxford_pose_skip_start",
            oxford_pose_skip_start,
            "--oxford_pose_skip_end",
            oxford_pose_skip_end,
            "--oxford_trim_edges",
            oxford_trim_edges,
            "--oxford_eval_seq",
            sequence_name,
            "--oxford_eval_mask_name",
            mask_name,
            "--frame_gap",
            frame_gap,
            "--eval_batch_size",
            eval_batch_size,
            "--workers",
            workers,
        ]
    )
)
args.device = setup_device(args)
start_time = time.time()
model, _ = load_checkpoint_model(args, checkpoint_path=ckpt_path)
sequence_data, segments = load_oxford_detailed_sequence(args, sequence_name, mask_name)

segment_metrics = []
pred_trajectories = []
gt_trajectories = []
for segment in segments:
    metrics, pred_trajectory, gt_trajectory = evaluate_segment(
        model,
        args.device,
        segment,
        args,
        show_progress=False,
    )
    segment_metrics.append(metrics)
    pred_trajectories.append(pred_trajectory)
    gt_trajectories.append(gt_trajectory)

save_full_route_plots(
    "full_route",
    segments,
    gt_trajectories,
    pred_trajectories,
    output_dir,
    background_trajectory=qe_pose_vectors_to_matrices(sequence_data["aligned_poses"]),
)

summary = build_oxford_detailed_summary(
    sequence_name=sequence_name,
    mask_name=mask_name,
    epoch=int(os.path.basename(ckpt_path).split("_")[2]),
    output_dir=output_dir,
    sequence_data=sequence_data,
    segment_metrics=segment_metrics,
    elapsed_sec=time.time() - start_time,
)
with open(os.path.join(output_dir, "summary.json"), "w") as handle:
    json.dump(summary, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY
}

process_checkpoint() {
  local ckpt_path="$1"
  local ckpt_name epoch_str epoch_num

  ckpt_name="$(basename "${ckpt_path}")"
  if [[ ! "${ckpt_name}" =~ ^translo_model_([0-9]+)_.*\.pth\.tar$ ]]; then
    return 0
  fi

  epoch_str="${BASH_REMATCH[1]}"
  epoch_num=$((10#${epoch_str}))

  if (( epoch_num <= AFTER_EPOCH )); then
    return 0
  fi

  if (( epoch_num % EPOCH_STRIDE != 0 )); then
    return 0
  fi

  local route_spec sequence_name mask_name summary_path
  local pending=0
  for route_spec in "${ROUTE_SPECS[@]}"; do
    IFS='|' read -r sequence_name mask_name <<< "${route_spec}"
    summary_path="${OUTPUT_ROOT}/epoch_${epoch_str}/${sequence_name}/summary.json"
    if [[ ! -f "${summary_path}" ]]; then
      pending=1
      break
    fi
  done

  if (( pending == 0 )); then
    return 0
  fi

  for route_spec in "${ROUTE_SPECS[@]}"; do
    IFS='|' read -r sequence_name mask_name <<< "${route_spec}"
    summary_path="${OUTPUT_ROOT}/epoch_${epoch_str}/${sequence_name}/summary.json"
    if [[ -f "${summary_path}" ]]; then
      continue
    fi
    run_route_eval "${ckpt_path}" "${epoch_str}" "${sequence_name}" "${mask_name}"
  done
}

log "starting Oxford detailed eval watcher"
log "task=${TASK_NAME} ckpt_dir=${CKPT_DIR} output_root=${OUTPUT_ROOT} routes=${WATCH_ROUTES}"
log "gpu=${GPU_ID} after_epoch=${AFTER_EPOCH} stride=${EPOCH_STRIDE} poll=${POLL_SECONDS}s"

shopt -s nullglob

while true; do
  ckpt_paths=("${CKPT_DIR}"/translo_model_*.pth.tar)
  if [[ "${#ckpt_paths[@]}" -eq 0 ]]; then
    log "no checkpoints found under ${CKPT_DIR}; sleeping"
    sleep "${POLL_SECONDS}"
    continue
  fi

  IFS=$'\n' ckpt_paths=($(printf '%s\n' "${ckpt_paths[@]}" | sort -V))
  unset IFS

  for ckpt_path in "${ckpt_paths[@]}"; do
    process_checkpoint "${ckpt_path}"
  done

  sleep "${POLL_SECONDS}"
done
