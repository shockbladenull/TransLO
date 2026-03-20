import argparse
import csv
import glob
import multiprocessing as mp
import os
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


DEFAULT_RANK_TRANSLATION_KEY = "aggregates.trajectory_endpoint.translation_error_percent.mean"
DEFAULT_RANK_ROTATION_KEY = "aggregates.trajectory_endpoint.rotation_error_deg_per_m.mean"
_WORKER_ARGS = None
_WORKER_PREPARED_SEGMENTS = None
_WORKER_GPU_ID = None


def build_parser():
    parser = argparse.ArgumentParser(description="Rank multiple TransLO checkpoints on Oxford LO300 metrics")
    add_translonet_args(parser)
    add_oxford_lo300_eval_args(parser, require_output_dir=True)
    parser.add_argument(
        "--ckpt_glob",
        required=True,
        help="Glob pattern for checkpoints to evaluate, for example experiment/run/checkpoints/translonet/*.pth.tar",
    )
    parser.add_argument("--top_k", type=int, default=5, help="Number of top checkpoints to report")
    parser.add_argument(
        "--rank_translation_key",
        default=DEFAULT_RANK_TRANSLATION_KEY,
        help="Dotted summary key used as the primary ranking metric",
    )
    parser.add_argument(
        "--rank_rotation_key",
        default=DEFAULT_RANK_ROTATION_KEY,
        help="Dotted summary key used as the secondary ranking metric",
    )
    parser.add_argument(
        "--save_per_ckpt_summary",
        action="store_true",
        help="Write summary.json for every checkpoint under output_dir/checkpoints/<ckpt_name>/",
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


def get_nested_metric(payload, dotted_key):
    current = payload
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            raise KeyError("Metric key '{}' not found at '{}'".format(dotted_key, part))
        current = current[part]
    return float(current)


def checkpoint_label(checkpoint_path):
    basename = os.path.basename(checkpoint_path)
    stem = os.path.splitext(basename)[0]
    if stem.endswith(".pth"):
        stem = os.path.splitext(stem)[0]
    return stem


def ranking_sort_key(row):
    return (
        row["translation_metric"],
        row["rotation_metric"],
        row["checkpoint_epoch"] if row["checkpoint_epoch"] >= 0 else float("inf"),
        row["checkpoint_name"],
    )


def build_ranking_row(summary, translation_key, rotation_key):
    checkpoint_path = summary["checkpoint_path"]
    return {
        "checkpoint_name": checkpoint_label(checkpoint_path),
        "checkpoint_path": checkpoint_path,
        "checkpoint_epoch": int(summary.get("checkpoint_epoch", -1)),
        "translation_metric_key": translation_key,
        "translation_metric": get_nested_metric(summary, translation_key),
        "rotation_metric_key": rotation_key,
        "rotation_metric": get_nested_metric(summary, rotation_key),
        "segment_count": int(summary["segment_count"]),
        "elapsed_sec": float(summary["elapsed_sec"]),
        "gpu_mem_gb": float(summary.get("gpu_mem_gb", 0.0)),
        "gpu_peak_mem_gb": float(summary.get("gpu_peak_mem_gb", 0.0)),
        "worker_gpu": summary.get("worker_gpu"),
    }


def sort_ranking_rows(rows):
    return sorted(rows, key=ranking_sort_key)


def best_ranking_row(rows):
    if not rows:
        return None
    return min(rows, key=ranking_sort_key)


def write_csv(path, rows):
    if not rows:
        raise ValueError("Cannot write an empty ranking CSV")
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


def format_progress_postfix(last_row, best_row):
    parts = []
    if last_row is not None:
        parts.append(
            "last={}@g{} {:.4f}/{:.4f}".format(
                shorten_checkpoint_name(last_row["checkpoint_name"]),
                last_row.get("worker_gpu", "?"),
                float(last_row["translation_metric"]),
                float(last_row["rotation_metric"]),
            )
        )
    if best_row is not None:
        parts.append(
            "best={} {:.4f}/{:.4f}".format(
                shorten_checkpoint_name(best_row["checkpoint_name"]),
                float(best_row["translation_metric"]),
                float(best_row["rotation_metric"]),
            )
        )
    return " | ".join(parts)


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


def _evaluate_single_checkpoint(
    args_dict,
    checkpoint_path,
    gpu_id,
    save_per_ckpt_summary,
    output_dir,
    translation_key,
    rotation_key,
):
    args, prepared_segments = _get_worker_context(args_dict, gpu_id)

    checkpoint_output_dir = None
    if save_per_ckpt_summary:
        checkpoint_output_dir = os.path.join(output_dir, "checkpoints", checkpoint_label(checkpoint_path))
    summary, _, _ = evaluate_checkpoint(
        args,
        checkpoint_path=checkpoint_path,
        output_dir=checkpoint_output_dir,
        prepared_segments=prepared_segments,
        summary_only=True,
        skip_plots=True,
        skip_segment_artifacts=True,
        show_progress=False,
    )
    summary["worker_gpu"] = int(gpu_id)
    return build_ranking_row(summary, translation_key, rotation_key)


def evaluate_rankings_parallel(args, checkpoint_paths, show_progress=True):
    gpu_ids = parse_gpu_ids(args)
    checkpoint_gpu_pairs, worker_gpu_ids = build_checkpoint_gpu_pairs(checkpoint_paths, gpu_ids, args.jobs_per_gpu)
    args_dict = vars(args).copy()
    args_dict.pop("device", None)

    ranking_rows = []
    progress_bar = tqdm(
        total=len(checkpoint_paths),
        desc="Sweep",
        unit="ckpt",
        dynamic_ncols=True,
        disable=not show_progress,
    )

    def _consume_result(row):
        ranking_rows.append(row)
        progress_bar.update(1)
        progress_bar.set_postfix_str(format_progress_postfix(row, best_ranking_row(ranking_rows)))

    try:
        if len(worker_gpu_ids) == 1:
            for checkpoint_path, gpu_id in checkpoint_gpu_pairs:
                row = _evaluate_single_checkpoint(
                    args_dict,
                    checkpoint_path,
                    gpu_id,
                    args.save_per_ckpt_summary,
                    args.output_dir,
                    args.rank_translation_key,
                    args.rank_rotation_key,
                )
                _consume_result(row)
            return ranking_rows

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
                    args.save_per_ckpt_summary,
                    args.output_dir,
                    args.rank_translation_key,
                    args.rank_rotation_key,
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
        return ranking_rows
    finally:
        progress_bar.close()


def main():
    parser = build_parser()
    args = finalize_translonet_args(parser.parse_args())
    validate_args(parser, args, require_ckpt=False, require_output_dir=True)
    args.summary_only = True
    args.skip_plots = True
    args.skip_segment_artifacts = True

    if args.top_k < 1:
        parser.error("--top_k must be >= 1")
    if args.jobs_per_gpu < 1:
        parser.error("--jobs_per_gpu must be >= 1")

    checkpoint_paths = sorted(glob.glob(args.ckpt_glob))
    if not checkpoint_paths:
        parser.error("No checkpoints matched --ckpt_glob {}".format(args.ckpt_glob))

    gpu_ids = parse_gpu_ids(args)
    total_workers = len(gpu_ids) * int(args.jobs_per_gpu)
    print(
        "Sweep config: {} checkpoints across GPUs {} with {} parallel workers".format(
            len(checkpoint_paths),
            ",".join(str(gpu_id) for gpu_id in gpu_ids),
            total_workers,
        )
    )

    os.makedirs(args.output_dir, exist_ok=True)
    ranking_rows = evaluate_rankings_parallel(args, checkpoint_paths, show_progress=True)
    ranking_rows = sort_ranking_rows(ranking_rows)
    top_k_rows = ranking_rows[: min(args.top_k, len(ranking_rows))]
    ranking_payload = {
        "sequence_name": args.oxford_eval_seq,
        "mask_h5_name": args.oxford_eval_mask_name,
        "checkpoint_glob": args.ckpt_glob,
        "checkpoint_count": int(len(ranking_rows)),
        "top_k": int(min(args.top_k, len(ranking_rows))),
        "rank_translation_key": args.rank_translation_key,
        "rank_rotation_key": args.rank_rotation_key,
        "gpu_ids": gpu_ids,
        "jobs_per_gpu": int(args.jobs_per_gpu),
        "ranking": ranking_rows,
        "top_k_ranking": top_k_rows,
    }

    write_csv(os.path.join(args.output_dir, "checkpoint_ranking.csv"), ranking_rows)
    write_json(os.path.join(args.output_dir, "checkpoint_ranking.json"), ranking_payload)
    write_json(os.path.join(args.output_dir, "best_checkpoint.json"), top_k_rows[0])
    if len(top_k_rows) > 1:
        write_json(os.path.join(args.output_dir, "top_k_checkpoints.json"), top_k_rows)

    best = top_k_rows[0]
    print("Ranked {} checkpoints on {}".format(len(ranking_rows), args.oxford_eval_seq))
    print("Best checkpoint: {}".format(best["checkpoint_path"]))
    print(
        "Best metrics: {}={:.6f}, {}={:.6f}".format(
            best["translation_metric_key"],
            best["translation_metric"],
            best["rotation_metric_key"],
            best["rotation_metric"],
        )
    )
    print("Ranking written to {}".format(os.path.join(args.output_dir, "checkpoint_ranking.csv")))


if __name__ == "__main__":
    main()
