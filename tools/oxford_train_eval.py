import json
import os
import time

from dataset_factory import load_oxford_txt_masked_sequence, split_oxford_selected_sequence_into_segments
from oxford_lo300_eval import DEFAULT_OXFORD_EVAL_SEQ, DEFAULT_OXFORD_LO_MASK, evaluate_segment
from tools.oxford_eval_tools import (
    aggregate_segment_metrics,
    build_segment,
    qe_pose_vectors_to_matrices,
    save_full_route_plots,
)
from tools.tensorboard_tools import log_oxford_route_images

DEFAULT_OXFORD_SCR_MASK = "velodyne_left_calibrateFalse_SCR300m.h5"


def should_run_oxford_detailed_val(args, epoch):
    return (
        getattr(args, "oxford_detailed_val", False)
        and getattr(args, "val_dataset_type", None) == "oxford_qe"
        and getattr(args, "oxford_pose_source", None) == "txt"
        and epoch > 0
        and epoch % int(args.oxford_detailed_val_interval) == 0
    )


def build_oxford_detailed_targets(args):
    if not getattr(args, "oxford_train_seqs", None):
        raise ValueError("Oxford detailed validation requires at least one training sequence")

    return [
        {
            "sequence_name": args.oxford_train_seqs[0],
            "mask_name": DEFAULT_OXFORD_SCR_MASK,
        },
        {
            "sequence_name": DEFAULT_OXFORD_EVAL_SEQ,
            "mask_name": DEFAULT_OXFORD_LO_MASK,
        },
    ]


def build_oxford_detailed_output_dir(eval_dir, epoch, sequence_name):
    return os.path.join(
        eval_dir,
        "oxford_detailed",
        "epoch_{:03d}".format(int(epoch)),
        str(sequence_name),
    )


def load_oxford_detailed_sequence(args, sequence_name, mask_name):
    sequence_data = load_oxford_txt_masked_sequence(
        root_dir=args.oxford_root,
        sequence_name=sequence_name,
        h5_name=mask_name,
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
        build_segment(sequence_name, sequence_data["scan_dir"], segment_dict)
        for segment_dict in segment_dicts
        if len(segment_dict["timestamps"]) >= 2
    ]
    if not segments:
        raise RuntimeError("No Oxford validation segments with at least two frames were found for {}".format(sequence_name))
    return sequence_data, segments


def build_oxford_detailed_summary(sequence_name, mask_name, epoch, output_dir, sequence_data, segment_metrics, elapsed_sec):
    aggregates = aggregate_segment_metrics(segment_metrics)
    return {
        "sequence_name": sequence_name,
        "mask_h5_name": mask_name,
        "epoch": int(epoch),
        "output_dir": os.path.abspath(output_dir),
        "elapsed_sec": float(elapsed_sec),
        "segment_count": int(len(segment_metrics)),
        "aligned_frame_count": int(len(sequence_data["aligned_timestamps"])),
        "selected_frame_count": int(len(sequence_data["selected_timestamps"])),
        "aggregates": aggregates,
        "ranking_metrics": {
            "pairwise_translation_mean_m": float(aggregates["pairwise"]["translation_mean_m"]["mean"]),
            "pairwise_rotation_mean_deg": float(aggregates["pairwise"]["rotation_mean_deg"]["mean"]),
            "trajectory_endpoint_translation_error_percent_mean": float(
                aggregates["trajectory_endpoint"]["translation_error_percent"]["mean"]
            ),
            "trajectory_endpoint_rotation_error_deg_per_m_mean": float(
                aggregates["trajectory_endpoint"]["rotation_error_deg_per_m"]["mean"]
            ),
        },
        "route_artifacts": {
            "full_route_path_png": "full_route_path.png",
            "full_route_path_pdf": "full_route_path.pdf",
            "full_route_path_3D_png": "full_route_path_3D.png",
            "full_route_path_3D_pdf": "full_route_path_3D.pdf",
        },
    }


def run_oxford_detailed_val(model, device, args, eval_dir, epoch, log_fn=None, show_progress=False, tb_writer=None):
    if args.oxford_pose_source != "txt":
        raise ValueError("Oxford detailed validation currently requires --oxford_pose_source txt")

    summaries = []
    for target in build_oxford_detailed_targets(args):
        sequence_name = target["sequence_name"]
        mask_name = target["mask_name"]
        start_time = time.time()
        sequence_data, segments = load_oxford_detailed_sequence(args, sequence_name, mask_name)

        segment_metrics = []
        pred_trajectories = []
        gt_trajectories = []
        for segment in segments:
            metrics, pred_trajectory, gt_trajectory = evaluate_segment(
                model,
                device,
                segment,
                args,
                show_progress=show_progress,
            )
            segment_metrics.append(metrics)
            pred_trajectories.append(pred_trajectory)
            gt_trajectories.append(gt_trajectory)

        output_dir = build_oxford_detailed_output_dir(eval_dir, epoch, sequence_name)
        os.makedirs(output_dir, exist_ok=True)
        save_full_route_plots(
            "full_route",
            segments,
            gt_trajectories,
            pred_trajectories,
            output_dir,
            background_trajectory=qe_pose_vectors_to_matrices(sequence_data["aligned_poses"]),
        )
        log_oxford_route_images(tb_writer, sequence_name, output_dir, epoch)

        summary = build_oxford_detailed_summary(
            sequence_name=sequence_name,
            mask_name=mask_name,
            epoch=epoch,
            output_dir=output_dir,
            sequence_data=sequence_data,
            segment_metrics=segment_metrics,
            elapsed_sec=time.time() - start_time,
        )
        with open(os.path.join(output_dir, "summary.json"), "w") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
            handle.write("\n")
        summaries.append(summary)

        if log_fn is not None:
            log_fn(
                "Epoch {:03d}: Oxford detailed val {} | pair_t: {:.6f} m | traj: {:.2f}% | route: {}".format(
                    epoch,
                    sequence_name,
                    summary["ranking_metrics"]["pairwise_translation_mean_m"],
                    summary["ranking_metrics"]["trajectory_endpoint_translation_error_percent_mean"],
                    os.path.join(output_dir, "full_route_path_3D.png"),
                )
            )

    return summaries
