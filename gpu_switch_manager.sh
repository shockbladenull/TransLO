#!/usr/bin/env bash
set -euo pipefail

# Daily GPU schedule:
# - Night: 21:00 -> next day 09:00, use GPUs 0,1,3,6
# - Day:   09:00 -> 21:00, use GPUs 1,3
#
# The job cannot hot-switch world_size inside a running DDP process.
# This script stops the current job at the schedule boundary and restarts
# from the most recent checkpoint with the new GPU set.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_NAME="oxford_0226_scr_turning_from_ep060"
INITIAL_CKPT="$PROJECT_ROOT/experiment/oxford_0226_train_0552_val_tb/checkpoints/translonet/translo_model_060_-24.997519.pth.tar"

DAY_START_HHMM=0900
NIGHT_START_HHMM=2100

# After widening Oxford projection from 1024 to 1792, use the current stable
# day-time setting and preserve the same effective global batch size at night:
# - Day train:   2 GPUs * 28 = 56
# - Night train: 4 GPUs * 14 = 56
# - Day eval:    2 GPUs * 128 = 256
# - Night eval:  4 GPUs * 64 = 256
DAY_BATCH_SIZE=28
DAY_EVAL_BATCH_SIZE=128
NIGHT_BATCH_SIZE=14
NIGHT_EVAL_BATCH_SIZE=64

DAY_GPUS="2,3"
DAY_NPROC=2
DAY_MASTER_PORT=29513

NIGHT_GPUS="2,3,4,6"
NIGHT_NPROC=4
NIGHT_MASTER_PORT=29531

SAVE_EVAL_INTERVAL=1
WORKERS=6

CKPT_DIR="$PROJECT_ROOT/experiment/$TASK_NAME/checkpoints/translonet"
LOG_DIR="$PROJECT_ROOT/experiment/$TASK_NAME/logs"
MANAGER_LOG="$LOG_DIR/gpu_switch_manager.log"

mkdir -p "$LOG_DIR"
cd "$PROJECT_ROOT"

COMMON_ARGS=(
  --train_dataset_type oxford_qe
  --val_dataset_type oxford_qe
  --test_dataset_type oxford_qe
  --oxford_root /Localize/ljc/Dataset/Oxford
  # Use the swapped turning split copy so SCR_turning keeps the "training-side"
  # name while pointing to the harder turning subset.
  --oxford_h5_name velodyne_left_calibrateFalse_SCR_turning.h5
  --oxford_h5_root /home/ljc/Downloads/h5filewithturn_swapped_turning/2-h5data
  # Keep detailed route validation on the original standard SCR/LO split.
  --oxford_detailed_h5_root /home/ljc/Downloads/2-h5data
  --oxford_full_h5_name velodyne_left_calibrateFalse.h5
  --oxford_full_h5_root /home/ljc/Downloads/2-h5data
  --oxford_pose_source txt
  --oxford_pose_root /home/ljc/Downloads/QEOxford
  --oxford_pose_txt_template "Oxford_SLAM_result_{sequence_short}/gicp_Oxford{sequence_short}_050_v1.txt"
  --oxford_pose_skip_start 5
  --oxford_pose_skip_end 5
  --oxford_trim_edges 0
  --oxford_train_seqs 2019-01-11-14-02-26-radar-oxford-10k
  # Fast pair-wise val still uses 0552 via val_loader.
  # Periodic detailed route val is fixed in code to:
  # - 0226 + SCR300m
  # - 0300 + LO300m
  --oxford_val_seqs 2019-01-14-12-05-52-radar-oxford-10k
  --frame_gap 1
  --workers "$WORKERS"
  --save_eval_interval "$SAVE_EVAL_INTERVAL"
  --oxford_detailed_val
  --oxford_detailed_val_interval 5
  --ddp_timeout_sec 3600
  --task_name "$TASK_NAME"
)

TRAIN_PID=""
TRAIN_PGID=""

log() {
  local message="$1"
  printf '[%s] %s\n' "$(date '+%F %T')" "$message" | tee -a "$MANAGER_LOG"
}

latest_ckpt() {
  local latest=""
  latest=$(ls -1t "$CKPT_DIR"/*.pth.tar 2>/dev/null | head -n 1 || true)
  if [[ -n "$latest" ]]; then
    printf '%s\n' "$latest"
    return 0
  fi
  if [[ -f "$INITIAL_CKPT" ]]; then
    printf '%s\n' "$INITIAL_CKPT"
    return 0
  fi
  return 1
}

current_mode() {
  local hhmm
  hhmm=$(date +%H%M)
  if (( 10#$hhmm >= 10#$NIGHT_START_HHMM || 10#$hhmm < 10#$DAY_START_HHMM )); then
    echo "night"
  else
    echo "day"
  fi
}

next_switch_ts() {
  local hhmm
  hhmm=$(date +%H%M)
  if (( 10#$hhmm < 10#$DAY_START_HHMM )); then
    date -d "today ${DAY_START_HHMM:0:2}:${DAY_START_HHMM:2:2}:00" +%s
  elif (( 10#$hhmm < 10#$NIGHT_START_HHMM )); then
    date -d "today ${NIGHT_START_HHMM:0:2}:${NIGHT_START_HHMM:2:2}:00" +%s
  else
    date -d "tomorrow ${DAY_START_HHMM:0:2}:${DAY_START_HHMM:2:2}:00" +%s
  fi
}

stop_train() {
  if [[ -z "${TRAIN_PGID:-}" ]]; then
    return 0
  fi

  log "stopping process group ${TRAIN_PGID}"
  kill -TERM -- "-${TRAIN_PGID}" 2>/dev/null || true

  for _ in $(seq 1 20); do
    if ! kill -0 "${TRAIN_PID}" 2>/dev/null; then
      return 0
    fi
    sleep 1
  done

  log "forcing process group ${TRAIN_PGID} to exit"
  kill -KILL -- "-${TRAIN_PGID}" 2>/dev/null || true
}

cleanup() {
  stop_train
}

trap cleanup EXIT INT TERM

launch_train() {
  local mode="$1"
  local gpus nproc master_port batch_size eval_batch_size job_log
  local ckpt=""
  local -a resume_args=()

  if [[ "$mode" == "night" ]]; then
    gpus="$NIGHT_GPUS"
    nproc="$NIGHT_NPROC"
    master_port="$NIGHT_MASTER_PORT"
    batch_size="$NIGHT_BATCH_SIZE"
    eval_batch_size="$NIGHT_EVAL_BATCH_SIZE"
    job_log="$LOG_DIR/${TASK_NAME}_night.log"
  else
    gpus="$DAY_GPUS"
    nproc="$DAY_NPROC"
    master_port="$DAY_MASTER_PORT"
    batch_size="$DAY_BATCH_SIZE"
    eval_batch_size="$DAY_EVAL_BATCH_SIZE"
    job_log="$LOG_DIR/${TASK_NAME}_day.log"
  fi

  if ! ckpt="$(latest_ckpt)"; then
    log "no checkpoint available: expected latest in $CKPT_DIR or bootstrap checkpoint $INITIAL_CKPT"
    exit 1
  fi
  resume_args=(--ckpt "$ckpt")

  log "launching mode=${mode} gpus=${gpus} batch=${batch_size} eval_batch=${eval_batch_size} ckpt=${ckpt:-none}"

  setsid env OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="$gpus" \
    pixi run python -m torch.distributed.run \
      --nproc_per_node="$nproc" \
      --master_port="$master_port" \
      train.py \
      "${COMMON_ARGS[@]}" \
      --batch_size "$batch_size" \
      --eval_batch_size "$eval_batch_size" \
      "${resume_args[@]}" \
      >> "$job_log" 2>&1 &

  TRAIN_PID=$!
  TRAIN_PGID=$(ps -o pgid= "$TRAIN_PID" | tr -d ' ')
  log "started pid=${TRAIN_PID} pgid=${TRAIN_PGID}"
}

while true; do
  mode="$(current_mode)"
  launch_train "$mode"
  switch_ts="$(next_switch_ts)"

  while kill -0 "$TRAIN_PID" 2>/dev/null; do
    now_ts=$(date +%s)
    if (( now_ts >= switch_ts )); then
      log "schedule boundary reached, switching mode"
      stop_train
      wait "$TRAIN_PID" 2>/dev/null || true
      TRAIN_PID=""
      TRAIN_PGID=""
      break
    fi
    sleep 30
  done

  if [[ -n "${TRAIN_PID:-}" ]] && ! kill -0 "$TRAIN_PID" 2>/dev/null; then
    wait "$TRAIN_PID" 2>/dev/null || true
    TRAIN_PID=""
    TRAIN_PGID=""
    log "training process exited before the next switch; restarting from latest checkpoint"
  fi

  sleep 5
done
