import argparse
import csv
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs import add_translonet_args, finalize_translonet_args
from dataset_factory import (
    load_oxford_txt_masked_sequence,
    split_oxford_selected_sequence_into_segments,
)
from tools.oxford_eval_tools import (
    OxfordSegmentPairDataset,
    aggregate_segment_metrics,
    build_segment,
    build_segment_metrics,
    pose_array_to_rows,
    save_segment_plots,
    segment_metrics_to_row,
)
from translo_model import translo_model
from utils1.collate_functions import collate_pair


DEFAULT_OXFORD_EVAL_SEQ = "2019-01-17-14-03-00-radar-oxford-10k"
DEFAULT_OXFORD_LO_MASK = "velodyne_left_calibrateFalse_LO300m.h5"
DEFAULT_OXFORD_FULL_H5 = "velodyne_left_calibrateFalse.h5"
DEFAULT_OXFORD_POSE_TEMPLATE = "Oxford_SLAM_result_{sequence_short}/gicp_Oxford{sequence_short}_050_v1.txt"


def add_oxford_lo300_eval_args(parser, require_output_dir=True):
    parser.add_argument(
        "--output_dir",
        required=require_output_dir,
        default=None,
        help="Directory for Oxford LO300 evaluation outputs",
    )
    parser.add_argument("--oxford_eval_seq", default=DEFAULT_OXFORD_EVAL_SEQ, help="Oxford route to evaluate")
    parser.add_argument(
        "--oxford_eval_mask_name",
        default=DEFAULT_OXFORD_LO_MASK,
        help="Oxford mask H5 used to define LO test frames",
    )
    parser.add_argument(
        "--skip_plots",
        action="store_true",
        help="Skip per-segment trajectory/XYZ/RPY/path plots",
    )
    parser.add_argument(
        "--skip_segment_artifacts",
        action="store_true",
        help="Skip per-segment trajectory .npy files and metrics.json outputs",
    )
    parser.add_argument(
        "--summary_only",
        action="store_true",
        help="Only write summary.json; skip segment tables, plots, and per-segment artifacts",
    )
    parser.set_defaults(
        train_dataset_type="oxford_qe",
        val_dataset_type="oxford_qe",
        test_dataset_type="oxford_qe",
        oxford_pose_source="txt",
        oxford_full_h5_name=DEFAULT_OXFORD_FULL_H5,
        oxford_pose_txt_template=DEFAULT_OXFORD_POSE_TEMPLATE,
        oxford_pose_skip_start=5,
        oxford_pose_skip_end=5,
        oxford_trim_edges=0,
    )
    return parser


def build_parser():
    parser = argparse.ArgumentParser(description="Evaluate a TransLO checkpoint on Oxford LO300 contiguous subsegments")
    add_translonet_args(parser)
    add_oxford_lo300_eval_args(parser, require_output_dir=True)
    return parser


def validate_args(parser, args, require_ckpt=True, require_output_dir=True):
    required_paths = ["oxford_root", "oxford_h5_root", "oxford_full_h5_root", "oxford_pose_root"]
    if require_ckpt:
        required_paths.append("ckpt")
    if require_output_dir:
        required_paths.append("output_dir")

    missing = [name for name in required_paths if not getattr(args, name)]
    if missing:
        parser.error("Missing required arguments: {}".format(", ".join("--{}".format(name) for name in missing)))

    if args.oxford_pose_source != "txt":
        parser.error("oxford_lo300_eval.py requires --oxford_pose_source txt")
    if args.frame_gap != 1:
        parser.error("oxford_lo300_eval.py currently requires --frame_gap 1")
    if args.eval_batch_size < 1:
        parser.error("--eval_batch_size must be >= 1")
    if args.workers < 0:
        parser.error("--workers must be >= 0")


def setup_device(args):
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        return torch.device("cuda", args.gpu)
    return torch.device("cpu")


def safe_gpu_memory_stats(device):
    if device.type != "cuda" or not torch.cuda.is_available():
        return 0.0, 0.0
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    current_mem = torch.cuda.memory_allocated(device_index) / (1024 ** 3)
    peak_mem = torch.cuda.max_memory_allocated(device_index) / (1024 ** 3)
    return float(current_mem), float(peak_mem)


def load_checkpoint_model(args, checkpoint_path=None):
    checkpoint_path = checkpoint_path or args.ckpt
    model = translo_model(args, args.eval_batch_size, args.H_input, args.W_input, False).to(args.device)
    checkpoint = torch.load(checkpoint_path, map_location=args.device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    normalized_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            normalized_state_dict[key[len("module."):]] = value
        else:
            normalized_state_dict[key] = value
    model.load_state_dict(normalized_state_dict, strict=True)
    model.eval()
    return model, checkpoint


def move_batch_to_device(device, batch):
    pos2, pos1, sample_id, T_gt, T_trans, T_trans_inv, Tr = batch
    pos2 = [item.to(device, dtype=torch.float32, non_blocking=True) for item in pos2]
    pos1 = [item.to(device, dtype=torch.float32, non_blocking=True) for item in pos1]
    T_gt = T_gt.to(device, dtype=torch.float32, non_blocking=True)
    T_trans = T_trans.to(device, dtype=torch.float32, non_blocking=True)
    T_trans_inv = T_trans_inv.to(device, dtype=torch.float32, non_blocking=True)
    return pos2, pos1, sample_id, T_gt, T_trans, T_trans_inv, Tr


def evaluate_segment(model, device, segment, args, show_progress=True):
    dataset = OxfordSegmentPairDataset(
        scan_dir=segment.scan_dir,
        timestamps=segment.timestamps,
        poses=segment.poses,
        frame_gap=args.frame_gap,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_pair,
    )

    pred_q_batches = []
    pred_t_batches = []
    gt_q_batches = []
    gt_t_batches = []
    elapsed_sec = 0.0

    with torch.no_grad():
        for batch in tqdm(
            loader,
            total=len(loader),
            desc="segment {:02d}".format(segment.segment_index),
            dynamic_ncols=True,
            leave=False,
            disable=not show_progress,
        ):
            pos2, pos1, _, T_gt, T_trans, T_trans_inv, _ = move_batch_to_device(device, batch)
            start_time = time.time()
            l0_q, l0_t, _, _, _, _, _, _, _, q_gt, t_gt, _, _ = model(
                pos2,
                pos1,
                T_gt,
                T_trans,
                T_trans_inv,
            )
            elapsed_sec += time.time() - start_time

            pred_q_batches.append(l0_q.detach().cpu().numpy())
            pred_t_batches.append(l0_t.detach().cpu().numpy())
            gt_q_batches.append(q_gt.detach().cpu().numpy())
            gt_t = t_gt.detach().cpu().squeeze(-1)
            if gt_t.ndim == 1:
                gt_t = gt_t.unsqueeze(0)
            gt_t_batches.append(gt_t.numpy())

    pred_q = np.concatenate(pred_q_batches, axis=0)
    pred_t = np.concatenate(pred_t_batches, axis=0)
    gt_q = np.concatenate(gt_q_batches, axis=0)
    gt_t = np.concatenate(gt_t_batches, axis=0)
    metrics, pred_trajectory, gt_trajectory = build_segment_metrics(
        segment=segment,
        pred_q=pred_q,
        pred_t=pred_t,
        gt_q=gt_q,
        gt_t=gt_t,
    )
    metrics["inference_time_sec"] = float(elapsed_sec)
    metrics["avg_pair_inference_time_sec"] = float(elapsed_sec / max(len(dataset), 1))
    return metrics, pred_trajectory, gt_trajectory


def load_segments_from_args(args):
    sequence_data = load_oxford_txt_masked_sequence(
        root_dir=args.oxford_root,
        sequence_name=args.oxford_eval_seq,
        h5_name=args.oxford_eval_mask_name,
        h5_root=args.oxford_h5_root,
        full_h5_name=args.oxford_full_h5_name,
        full_h5_root=args.oxford_full_h5_root,
        pose_root=args.oxford_pose_root,
        pose_txt_template=args.oxford_pose_txt_template,
        pose_skip_start=args.oxford_pose_skip_start,
        pose_skip_end=args.oxford_pose_skip_end,
        trim_edges=args.oxford_trim_edges,
    )
    segment_dicts = split_oxford_selected_sequence_into_segments(
        selected_timestamps=sequence_data["selected_timestamps"],
        selected_poses=sequence_data["selected_poses"],
        selected_aligned_indices=sequence_data["selected_aligned_indices"],
    )
    segments = [
        build_segment(args.oxford_eval_seq, sequence_data["scan_dir"], segment_dict)
        for segment_dict in segment_dicts
        if len(segment_dict["timestamps"]) >= 2
    ]
    if not segments:
        raise RuntimeError("No Oxford LO segments with at least two frames were found")
    return sequence_data, segments


def build_summary(
    args,
    checkpoint,
    checkpoint_path,
    sequence_data,
    segment_metrics,
    elapsed_sec,
    output_dir=None,
    gpu_mem_gb=0.0,
    gpu_peak_mem_gb=0.0,
):
    aggregates = aggregate_segment_metrics(segment_metrics)
    summary = {
        "sequence_name": args.oxford_eval_seq,
        "mask_h5_name": args.oxford_eval_mask_name,
        "checkpoint_path": os.path.abspath(checkpoint_path),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)) if isinstance(checkpoint, dict) else -1,
        "device": str(args.device),
        "segment_count": int(len(segment_metrics)),
        "segment_lengths_frames": [int(item["frame_count"]) for item in segment_metrics],
        "total_frame_count": int(sum(item["frame_count"] for item in segment_metrics)),
        "total_pair_count": int(sum(item["pair_count"] for item in segment_metrics)),
        "aligned_frame_count": int(len(sequence_data["aligned_timestamps"])),
        "selected_frame_count": int(len(sequence_data["selected_timestamps"])),
        "output_dir": os.path.abspath(output_dir) if output_dir is not None else None,
        "elapsed_sec": float(elapsed_sec),
        "gpu_mem_gb": float(gpu_mem_gb),
        "gpu_peak_mem_gb": float(gpu_peak_mem_gb),
        "aggregates": aggregates,
        "ranking_metrics": {
            "trajectory_endpoint_translation_error_percent_mean": float(
                aggregates["trajectory_endpoint"]["translation_error_percent"]["mean"]
            ),
            "trajectory_endpoint_rotation_error_deg_per_m_mean": float(
                aggregates["trajectory_endpoint"]["rotation_error_deg_per_m"]["mean"]
            ),
            "pairwise_translation_mean_m": float(aggregates["pairwise"]["translation_mean_m"]["mean"]),
            "pairwise_rotation_mean_deg": float(aggregates["pairwise"]["rotation_mean_deg"]["mean"]),
        },
    }
    return summary


def write_json(path, payload):
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_segments_csv(path, rows):
    if not rows:
        raise ValueError("Cannot write an empty segments.csv")
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_segments_jsonl(path, payloads):
    with open(path, "w") as handle:
        for payload in payloads:
            handle.write(json.dumps(payload, sort_keys=True))
            handle.write("\n")


def evaluate_checkpoint(
    args,
    checkpoint_path=None,
    output_dir=None,
    prepared_segments=None,
    skip_plots=None,
    skip_segment_artifacts=None,
    summary_only=None,
    show_progress=True,
):
    checkpoint_path = checkpoint_path or args.ckpt
    output_dir = args.output_dir if output_dir is None else output_dir
    summary_only = args.summary_only if summary_only is None else summary_only
    skip_plots = args.skip_plots if skip_plots is None else skip_plots
    skip_segment_artifacts = args.skip_segment_artifacts if skip_segment_artifacts is None else skip_segment_artifacts
    if summary_only:
        skip_plots = True
        skip_segment_artifacts = True

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    if args.device.type == "cuda" and torch.cuda.is_available():
        device_index = args.device.index if args.device.index is not None else torch.cuda.current_device()
        torch.cuda.reset_peak_memory_stats(device_index)

    model, checkpoint = load_checkpoint_model(args, checkpoint_path=checkpoint_path)
    if prepared_segments is None:
        sequence_data, segments = load_segments_from_args(args)
    else:
        sequence_data, segments = prepared_segments

    segment_metrics = []
    segment_rows = []
    summary_start = time.time()
    segment_iterator = tqdm(
        segments,
        desc="Oxford LO300 segments",
        dynamic_ncols=True,
        disable=not show_progress,
    )
    for segment in segment_iterator:
        metrics, pred_trajectory, gt_trajectory = evaluate_segment(
            model,
            args.device,
            segment,
            args,
            show_progress=False,
        )
        if output_dir is not None and not summary_only:
            segment_name = "segment_{:02d}".format(segment.segment_index)
            segment_dir = os.path.join(output_dir, segment_name)
            if not skip_plots or not skip_segment_artifacts:
                os.makedirs(segment_dir, exist_ok=True)

            if not skip_segment_artifacts:
                pred_path = os.path.join(segment_dir, "pred_traj.npy")
                gt_path = os.path.join(segment_dir, "gt_traj.npy")
                np.save(pred_path, pose_array_to_rows(pred_trajectory))
                np.save(gt_path, pose_array_to_rows(gt_trajectory))
                metrics["artifacts"] = {
                    "pred_traj_npy": os.path.basename(pred_path),
                    "gt_traj_npy": os.path.basename(gt_path),
                    "plot_prefix": segment_name,
                }
                write_json(os.path.join(segment_dir, "metrics.json"), metrics)

            if not skip_plots:
                save_segment_plots(segment_name, gt_trajectory, pred_trajectory, segment_dir)

        segment_metrics.append(metrics)
        segment_rows.append(segment_metrics_to_row(metrics))

    elapsed_sec = time.time() - summary_start
    gpu_mem_gb, gpu_peak_mem_gb = safe_gpu_memory_stats(args.device)
    summary = build_summary(
        args=args,
        checkpoint=checkpoint,
        checkpoint_path=checkpoint_path,
        sequence_data=sequence_data,
        segment_metrics=segment_metrics,
        elapsed_sec=elapsed_sec,
        output_dir=output_dir,
        gpu_mem_gb=gpu_mem_gb,
        gpu_peak_mem_gb=gpu_peak_mem_gb,
    )

    if output_dir is not None:
        if not summary_only:
            write_segments_csv(os.path.join(output_dir, "segments.csv"), segment_rows)
            write_segments_jsonl(os.path.join(output_dir, "segments.jsonl"), segment_metrics)
        write_json(os.path.join(output_dir, "summary.json"), summary)

    return summary, segment_metrics, segment_rows


def main():
    parser = build_parser()
    args = finalize_translonet_args(parser.parse_args())
    validate_args(parser, args, require_ckpt=True, require_output_dir=True)
    args.device = setup_device(args)

    summary, _, _ = evaluate_checkpoint(args)
    print("Evaluated {} LO segments from {}".format(summary["segment_count"], args.oxford_eval_seq))
    print("Summary written to {}".format(os.path.join(args.output_dir, "summary.json")))


if __name__ == "__main__":
    main()
