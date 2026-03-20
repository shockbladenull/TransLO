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


def build_parser():
    parser = argparse.ArgumentParser(description="Evaluate a TransLO checkpoint on Oxford LO300 contiguous subsegments")
    add_translonet_args(parser)
    parser.add_argument("--output_dir", required=True, help="Directory for Oxford LO300 evaluation outputs")
    parser.add_argument("--oxford_eval_seq", default=DEFAULT_OXFORD_EVAL_SEQ, help="Oxford route to evaluate")
    parser.add_argument(
        "--oxford_eval_mask_name",
        default=DEFAULT_OXFORD_LO_MASK,
        help="Oxford mask H5 used to define LO test frames",
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


def validate_args(parser, args):
    required_paths = (
        "ckpt",
        "oxford_root",
        "oxford_h5_root",
        "oxford_full_h5_root",
        "oxford_pose_root",
        "output_dir",
    )
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


def load_checkpoint_model(args):
    model = translo_model(args, args.eval_batch_size, args.H_input, args.W_input, False).to(args.device)
    checkpoint = torch.load(args.ckpt, map_location=args.device)
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


def evaluate_segment(model, device, segment, args):
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


def main():
    parser = build_parser()
    args = finalize_translonet_args(parser.parse_args())
    validate_args(parser, args)
    args.device = setup_device(args)

    os.makedirs(args.output_dir, exist_ok=True)

    model, checkpoint = load_checkpoint_model(args)
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

    segment_metrics = []
    segment_rows = []
    summary_start = time.time()
    for segment in tqdm(segments, desc="Oxford LO300 segments", dynamic_ncols=True):
        metrics, pred_trajectory, gt_trajectory = evaluate_segment(model, args.device, segment, args)
        segment_name = "segment_{:02d}".format(segment.segment_index)
        segment_dir = os.path.join(args.output_dir, segment_name)
        os.makedirs(segment_dir, exist_ok=True)

        pred_path = os.path.join(segment_dir, "pred_traj.npy")
        gt_path = os.path.join(segment_dir, "gt_traj.npy")
        np.save(pred_path, pose_array_to_rows(pred_trajectory))
        np.save(gt_path, pose_array_to_rows(gt_trajectory))
        save_segment_plots(segment_name, gt_trajectory, pred_trajectory, segment_dir)

        metrics["artifacts"] = {
            "pred_traj_npy": os.path.basename(pred_path),
            "gt_traj_npy": os.path.basename(gt_path),
            "plot_prefix": segment_name,
        }
        write_json(os.path.join(segment_dir, "metrics.json"), metrics)
        segment_metrics.append(metrics)
        segment_rows.append(segment_metrics_to_row(metrics))

    aggregates = aggregate_segment_metrics(segment_metrics)
    summary = {
        "sequence_name": args.oxford_eval_seq,
        "mask_h5_name": args.oxford_eval_mask_name,
        "checkpoint_path": os.path.abspath(args.ckpt),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)) if isinstance(checkpoint, dict) else -1,
        "device": str(args.device),
        "segment_count": int(len(segment_metrics)),
        "segment_lengths_frames": [int(item["frame_count"]) for item in segment_metrics],
        "total_frame_count": int(sum(item["frame_count"] for item in segment_metrics)),
        "total_pair_count": int(sum(item["pair_count"] for item in segment_metrics)),
        "aligned_frame_count": int(len(sequence_data["aligned_timestamps"])),
        "selected_frame_count": int(len(sequence_data["selected_timestamps"])),
        "output_dir": os.path.abspath(args.output_dir),
        "elapsed_sec": float(time.time() - summary_start),
        "aggregates": aggregates,
    }

    write_segments_csv(os.path.join(args.output_dir, "segments.csv"), segment_rows)
    write_segments_jsonl(os.path.join(args.output_dir, "segments.jsonl"), segment_metrics)
    write_json(os.path.join(args.output_dir, "summary.json"), summary)

    print("Evaluated {} LO segments from {}".format(len(segment_metrics), args.oxford_eval_seq))
    print("Summary written to {}".format(os.path.join(args.output_dir, "summary.json")))


if __name__ == "__main__":
    main()
