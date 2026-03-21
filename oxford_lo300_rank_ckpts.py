import argparse
import csv
import glob
import multiprocessing as mp
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack

from tqdm import tqdm

from configs import add_translonet_args, finalize_translonet_args
from oxford_lo300_eval import (
    add_oxford_lo300_eval_args,
    evaluate_checkpoint,
    load_segments_from_args,
    setup_device,
    validate_args,
    write_json,
)

DEFAULT_AFTER_EPOCH = 100
DEFAULT_EPOCH_STRIDE = 5
EVALUATION_METRIC_FIELDS = (
    ("pairwise_translation_mean_m", "aggregates.pairwise.translation_mean_m.mean"),
    ("pairwise_rotation_mean_deg", "aggregates.pairwise.rotation_mean_deg.mean"),
    ("trajectory_endpoint_translation_error_percent", "aggregates.trajectory_endpoint.translation_error_percent.mean"),
    ("trajectory_endpoint_rotation_error_deg_per_m", "aggregates.trajectory_endpoint.rotation_error_deg_per_m.mean"),
    ("trajectory_per_frame_translation_mean_m", "aggregates.trajectory_per_frame.translation_mean_m.mean"),
    ("trajectory_per_frame_rotation_mean_deg", "aggregates.trajectory_per_frame.rotation_mean_deg.mean"),
)
_WORKER_ARGS = None
_WORKER_PREPARED_SEGMENTS = None
_WORKER_GPU_ID = None


def _suppress_option_help(parser, option_strings):
    for option_string in option_strings:
        action = parser._option_string_actions.get(option_string)
        if action is not None:
            action.help = argparse.SUPPRESS



def build_parser():
    parser = argparse.ArgumentParser(
        description="Run detailed Oxford LO300 evaluation for periodic TransLO checkpoints"
    )
    add_translonet_args(parser)
    add_oxford_lo300_eval_args(parser, require_output_dir=True)
    _suppress_option_help(parser, ['--ckpt', '--skip_plots', '--skip_segment_artifacts', '--summary_only'])
    parser.add_argument(
        "--ckpt_glob",
        required=True,
        help="Glob pattern for checkpoints to evaluate, for example experiment/run/checkpoints/translonet/*.pth.tar",
    )
    parser.add_argument(
        "--after_epoch",
        type=int,
        default=DEFAULT_AFTER_EPOCH,
        help="Only evaluate checkpoints with epoch strictly greater than this value",
    )
    parser.add_argument(
        "--epoch_stride",
        type=int,
        default=DEFAULT_EPOCH_STRIDE,
        help="Evaluate every N epochs after --after_epoch",
    )
    parser.add_argument(
        "--gpu_ids",
        default=None,
        help="Comma-separated GPU ids for parallel checkpoint evaluation; defaults to --gpu only",
    )
    parser.add_argument(
        "--jobs_per_gpu",
        type=int,
        default=1,
        help="Number of independent checkpoint-evaluation workers to launch per GPU",
    )
    return parser


def checkpoint_label(checkpoint_path):
    basename = os.path.basename(checkpoint_path)
    stem = os.path.splitext(basename)[0]
    if stem.endswith(".pth"):
        stem = os.path.splitext(stem)[0]
    return stem


def extract_checkpoint_epoch(checkpoint_path):
    match = re.search(r"translo_model_(\d+)", os.path.basename(checkpoint_path))
    if match is None:
        raise ValueError("Could not parse checkpoint epoch from '{}'".format(checkpoint_path))
    return int(match.group(1))


def should_evaluate_checkpoint(epoch, after_epoch, epoch_stride):
    if epoch_stride < 1:
        raise ValueError("epoch_stride must be >= 1")
    return int(epoch) > int(after_epoch) and ((int(epoch) - int(after_epoch)) % int(epoch_stride) == 0)


def select_checkpoint_paths(checkpoint_paths, after_epoch, epoch_stride):
    selected_paths = []
    for checkpoint_path in checkpoint_paths:
        epoch = extract_checkpoint_epoch(checkpoint_path)
        if should_evaluate_checkpoint(epoch, after_epoch=after_epoch, epoch_stride=epoch_stride):
            selected_paths.append(checkpoint_path)
    return selected_paths


def get_nested_metric(payload, dotted_key):
    current = payload
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            raise KeyError("Metric key '{}' not found at '{}'".format(dotted_key, part))
        current = current[part]
    return float(current)


def build_evaluation_row(summary, checkpoint_output_dir):
    checkpoint_path = summary["checkpoint_path"]
    row = {
        "checkpoint_name": checkpoint_label(checkpoint_path),
        "checkpoint_path": checkpoint_path,
        "checkpoint_epoch": int(summary.get("checkpoint_epoch", -1)),
        "worker_gpu": summary.get("worker_gpu"),
        "output_dir": os.path.abspath(checkpoint_output_dir),
        "summary_json": os.path.join(os.path.abspath(checkpoint_output_dir), "summary.json"),
        "segment_count": int(summary["segment_count"]),
        "elapsed_sec": float(summary["elapsed_sec"]),
        "gpu_mem_gb": float(summary.get("gpu_mem_gb", 0.0)),
        "gpu_peak_mem_gb": float(summary.get("gpu_peak_mem_gb", 0.0)),
    }
    for field_name, summary_key in EVALUATION_METRIC_FIELDS:
        row[field_name] = get_nested_metric(summary, summary_key)
    return row


def sort_evaluation_rows(rows):
    return sorted(rows, key=lambda row: (row["checkpoint_epoch"], row["checkpoint_name"]))


def write_csv(path, rows):
    if not rows:
        raise ValueError("Cannot write an empty evaluation CSV")
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_gpu_ids(args):
    if args.gpu_ids is None or str(args.gpu_ids).strip() == "":
        return [int(args.gpu)]
    return [int(part) for part in str(args.gpu_ids).split(",") if part.strip()]


def build_worker_gpu_ids(gpu_ids, jobs_per_gpu):
    if jobs_per_gpu < 1:
        raise ValueError("jobs_per_gpu must be >= 1")
    worker_gpu_ids = []
    for gpu_id in gpu_ids:
        for _ in range(jobs_per_gpu):
            worker_gpu_ids.append(int(gpu_id))
    return worker_gpu_ids


def build_worker_assignments(checkpoint_paths, gpu_ids, jobs_per_gpu):
    worker_gpu_ids = build_worker_gpu_ids(gpu_ids, jobs_per_gpu)
    assignments = [[] for _ in worker_gpu_ids]
    for index, checkpoint_path in enumerate(checkpoint_paths):
        assignments[index % len(worker_gpu_ids)].append(checkpoint_path)
    return worker_gpu_ids, assignments


def build_checkpoint_gpu_pairs(checkpoint_paths, gpu_ids, jobs_per_gpu):
    worker_gpu_ids, assignments = build_worker_assignments(checkpoint_paths, gpu_ids, jobs_per_gpu)
    checkpoint_gpu_pairs = []
    max_assignment_length = max((len(assignment) for assignment in assignments), default=0)
    for assignment_index in range(max_assignment_length):
        for slot_index, gpu_id in enumerate(worker_gpu_ids):
            assignment = assignments[slot_index]
            if assignment_index < len(assignment):
                checkpoint_gpu_pairs.append((assignment[assignment_index], int(gpu_id)))
    return checkpoint_gpu_pairs, worker_gpu_ids


def shorten_checkpoint_name(checkpoint_name, max_len=24):
    checkpoint_name = str(checkpoint_name)
    if len(checkpoint_name) <= max_len:
        return checkpoint_name
    return "...{}".format(checkpoint_name[-(max_len - 3):])


def format_progress_postfix(last_row):
    if last_row is None:
        return ""
    return "last={}@g{} epoch={} {:.4f}/{:.4f}".format(
        shorten_checkpoint_name(last_row["checkpoint_name"]),
        last_row.get("worker_gpu", "?"),
        int(last_row["checkpoint_epoch"]),
        float(last_row["trajectory_endpoint_translation_error_percent"]),
        float(last_row["trajectory_endpoint_rotation_error_deg_per_m"]),
    )


def _get_worker_context(args_dict, gpu_id):
    global _WORKER_ARGS, _WORKER_PREPARED_SEGMENTS, _WORKER_GPU_ID

    gpu_id = int(gpu_id)
    if _WORKER_ARGS is None or _WORKER_PREPARED_SEGMENTS is None or _WORKER_GPU_ID != gpu_id:
        import argparse as _argparse

        args = _argparse.Namespace(**args_dict)
        args.gpu = gpu_id
        args.device = setup_device(args)
        prepared_segments = load_segments_from_args(args)
        _WORKER_ARGS = args
        _WORKER_PREPARED_SEGMENTS = prepared_segments
        _WORKER_GPU_ID = gpu_id
    return _WORKER_ARGS, _WORKER_PREPARED_SEGMENTS


def _evaluate_single_checkpoint(args_dict, checkpoint_path, gpu_id, output_dir):
    args, prepared_segments = _get_worker_context(args_dict, gpu_id)
    checkpoint_output_dir = os.path.join(output_dir, "checkpoints", checkpoint_label(checkpoint_path))
    summary, _, _ = evaluate_checkpoint(
        args,
        checkpoint_path=checkpoint_path,
        output_dir=checkpoint_output_dir,
        prepared_segments=prepared_segments,
        summary_only=False,
        skip_plots=False,
        skip_segment_artifacts=False,
        show_progress=False,
    )
    summary["worker_gpu"] = int(gpu_id)
    return build_evaluation_row(summary, checkpoint_output_dir)


def evaluate_checkpoints_parallel(args, checkpoint_paths, show_progress=True):
    gpu_ids = parse_gpu_ids(args)
    checkpoint_gpu_pairs, worker_gpu_ids = build_checkpoint_gpu_pairs(checkpoint_paths, gpu_ids, args.jobs_per_gpu)
    args_dict = vars(args).copy()
    args_dict.pop("device", None)

    evaluation_rows = []
    progress_bar = tqdm(
        total=len(checkpoint_paths),
        desc="Detailed eval",
        unit="ckpt",
        dynamic_ncols=True,
        disable=not show_progress,
    )

    def _consume_result(row):
        evaluation_rows.append(row)
        progress_bar.update(1)
        progress_bar.set_postfix_str(format_progress_postfix(row))

    try:
        if len(worker_gpu_ids) == 1:
            for checkpoint_path, gpu_id in checkpoint_gpu_pairs:
                row = _evaluate_single_checkpoint(args_dict, checkpoint_path, gpu_id, args.output_dir)
                _consume_result(row)
            return evaluation_rows

        mp_context = mp.get_context("spawn")
        with ExitStack() as stack:
            executors = {
                int(gpu_id): stack.enter_context(
                    ProcessPoolExecutor(max_workers=args.jobs_per_gpu, mp_context=mp_context)
                )
                for gpu_id in gpu_ids
            }
            futures = {}
            for checkpoint_path, gpu_id in checkpoint_gpu_pairs:
                future = executors[int(gpu_id)].submit(
                    _evaluate_single_checkpoint,
                    args_dict,
                    checkpoint_path,
                    gpu_id,
                    args.output_dir,
                )
                futures[future] = {
                    "checkpoint_path": checkpoint_path,
                    "gpu_id": int(gpu_id),
                }

            for future in as_completed(futures):
                meta = futures[future]
                try:
                    row = future.result()
                except Exception as exc:
                    raise RuntimeError(
                        "Checkpoint evaluation failed for {} on gpu {}".format(
                            meta["checkpoint_path"], meta["gpu_id"]
                        )
                    ) from exc
                _consume_result(row)
        return evaluation_rows
    finally:
        progress_bar.close()


def main():
    parser = build_parser()
    args = finalize_translonet_args(parser.parse_args())
    validate_args(parser, args, require_ckpt=False, require_output_dir=True)
    args.summary_only = False
    args.skip_plots = False
    args.skip_segment_artifacts = False

    if args.jobs_per_gpu < 1:
        parser.error("--jobs_per_gpu must be >= 1")
    if args.epoch_stride < 1:
        parser.error("--epoch_stride must be >= 1")

    checkpoint_paths = sorted(glob.glob(args.ckpt_glob))
    if not checkpoint_paths:
        parser.error("No checkpoints matched --ckpt_glob {}".format(args.ckpt_glob))

    selected_checkpoint_paths = select_checkpoint_paths(
        checkpoint_paths,
        after_epoch=args.after_epoch,
        epoch_stride=args.epoch_stride,
    )
    if not selected_checkpoint_paths:
        parser.error(
            "No checkpoints matched epoch filter: epoch > {} and every {} epochs thereafter".format(
                args.after_epoch,
                args.epoch_stride,
            )
        )

    gpu_ids = parse_gpu_ids(args)
    total_workers = len(gpu_ids) * int(args.jobs_per_gpu)
    selected_epochs = [extract_checkpoint_epoch(path) for path in selected_checkpoint_paths]
    print(
        "Detailed eval config: {} selected checkpoints (from {} matched) across GPUs {} with {} parallel workers".format(
            len(selected_checkpoint_paths),
            len(checkpoint_paths),
            ",".join(str(gpu_id) for gpu_id in gpu_ids),
            total_workers,
        )
    )
    print(
        "Epoch filter: epoch > {} and every {} epochs thereafter -> {}".format(
            args.after_epoch,
            args.epoch_stride,
            ", ".join(str(epoch) for epoch in selected_epochs),
        )
    )

    os.makedirs(args.output_dir, exist_ok=True)
    evaluation_rows = evaluate_checkpoints_parallel(args, selected_checkpoint_paths, show_progress=True)
    evaluation_rows = sort_evaluation_rows(evaluation_rows)
    manifest = {
        "sequence_name": args.oxford_eval_seq,
        "mask_h5_name": args.oxford_eval_mask_name,
        "checkpoint_glob": args.ckpt_glob,
        "matched_checkpoint_count": int(len(checkpoint_paths)),
        "selected_checkpoint_count": int(len(evaluation_rows)),
        "after_epoch": int(args.after_epoch),
        "epoch_stride": int(args.epoch_stride),
        "selected_epochs": [int(row["checkpoint_epoch"]) for row in evaluation_rows],
        "gpu_ids": gpu_ids,
        "jobs_per_gpu": int(args.jobs_per_gpu),
        "evaluations": evaluation_rows,
    }

    write_csv(os.path.join(args.output_dir, "detailed_evaluations.csv"), evaluation_rows)
    write_json(os.path.join(args.output_dir, "detailed_evaluations.json"), manifest)

    print("Detailed evaluation finished for {} checkpoints on {}".format(len(evaluation_rows), args.oxford_eval_seq))
    print("Evaluation manifest written to {}".format(os.path.join(args.output_dir, "detailed_evaluations.csv")))


if __name__ == "__main__":
    main()
