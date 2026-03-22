import math
import os
from dataclasses import dataclass

import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt
import numpy as np
import tools.transformations as tr
import torch
import torch.utils.data as data

from dataset_factory import OxfordQEDataset
from tools.euler_tools import quat2mat

plt.switch_backend("agg")


@dataclass
class OxfordSegment:
    sequence_name: str
    segment_index: int
    timestamps: np.ndarray
    poses: np.ndarray
    aligned_indices: np.ndarray
    scan_dir: str
    start_timestamp: int
    end_timestamp: int
    start_aligned_index: int
    end_aligned_index: int


class OxfordSegmentPairDataset(data.Dataset):
    def __init__(self, scan_dir, timestamps, poses, frame_gap=1):
        timestamps = np.asarray(timestamps, dtype=np.int64)
        poses = np.asarray(poses, dtype=np.float32)
        if len(timestamps) != len(poses):
            raise ValueError("Oxford segment timestamps and poses must have identical lengths")
        if frame_gap != 1:
            raise ValueError("Oxford LO300 segment trajectory evaluation currently requires frame_gap == 1")
        if len(timestamps) <= frame_gap:
            raise ValueError("Oxford segment is too short for frame_gap={}".format(frame_gap))

        self.scan_dir = scan_dir
        self.timestamps = timestamps
        self.poses = poses
        self.frame_gap = frame_gap
        self.identity_transform = np.eye(4, dtype=np.float32)
        self.samples = [(curr_idx - frame_gap, curr_idx) for curr_idx in range(frame_gap, len(timestamps))]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        prev_idx, curr_idx = self.samples[index]
        prev_timestamp = int(self.timestamps[prev_idx])
        curr_timestamp = int(self.timestamps[curr_idx])
        prev_scan_path = os.path.join(self.scan_dir, "{}.bin".format(prev_timestamp))
        curr_scan_path = os.path.join(self.scan_dir, "{}.bin".format(curr_timestamp))

        point1 = OxfordQEDataset._load_oxford_scan(prev_scan_path)
        point2 = OxfordQEDataset._load_oxford_scan(curr_scan_path)
        T_prev = OxfordQEDataset._qe_pose_to_matrix(self.poses[prev_idx])
        T_curr = OxfordQEDataset._qe_pose_to_matrix(self.poses[curr_idx])
        T_gt = np.matmul(np.linalg.inv(T_curr), T_prev).astype(np.float32)

        return (
            torch.from_numpy(point2).float(),
            torch.from_numpy(point1).float(),
            index,
            T_gt,
            self.identity_transform.copy(),
            self.identity_transform.copy(),
            self.identity_transform.copy(),
        )


PAIRWISE_METRIC_FIELDS = (
    "translation_mean_m",
    "translation_rmse_m",
    "rotation_mean_deg",
    "rotation_rmse_deg",
)
TRAJECTORY_ENDPOINT_METRIC_FIELDS = (
    "path_length_m",
    "translation_error_m",
    "translation_error_ratio",
    "translation_error_percent",
    "rotation_error_rad",
    "rotation_error_deg",
    "rotation_error_rad_per_m",
    "rotation_error_deg_per_m",
)
TRAJECTORY_FRAME_METRIC_FIELDS = (
    "translation_mean_m",
    "translation_rmse_m",
    "rotation_mean_deg",
    "rotation_rmse_deg",
)


def build_segment(sequence_name, scan_dir, segment_data):
    return OxfordSegment(
        sequence_name=sequence_name,
        segment_index=int(segment_data["segment_index"]),
        timestamps=np.asarray(segment_data["timestamps"], dtype=np.int64),
        poses=np.asarray(segment_data["poses"], dtype=np.float32),
        aligned_indices=np.asarray(segment_data["aligned_indices"], dtype=np.int64),
        scan_dir=scan_dir,
        start_timestamp=int(segment_data["start_timestamp"]),
        end_timestamp=int(segment_data["end_timestamp"]),
        start_aligned_index=int(segment_data["start_aligned_index"]),
        end_aligned_index=int(segment_data["end_aligned_index"]),
    )


def quaternion_angle_error_deg_np(pred_q, gt_q):
    pred_q = np.asarray(pred_q, dtype=np.float64)
    gt_q = np.asarray(gt_q, dtype=np.float64)
    pred_q = pred_q / np.clip(np.linalg.norm(pred_q, axis=-1, keepdims=True), 1e-10, None)
    gt_q = gt_q / np.clip(np.linalg.norm(gt_q, axis=-1, keepdims=True), 1e-10, None)
    dot = np.abs(np.sum(pred_q * gt_q, axis=-1))
    dot = np.clip(dot, 0.0, 1.0)
    return (2.0 * np.arccos(dot)) * (180.0 / np.pi)


def pair_transform_from_quaternion_translation(quaternion, translation):
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = quat2mat(np.asarray(quaternion, dtype=np.float64).reshape(4))
    transform[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return transform


def qe_pose_vectors_to_matrices(pose_vectors):
    pose_vectors = np.asarray(pose_vectors, dtype=np.float32)
    return np.asarray([OxfordQEDataset._qe_pose_to_matrix(row) for row in pose_vectors], dtype=np.float64)


def global_pose_vectors_to_relative_pairs(pose_vectors):
    pose_matrices = qe_pose_vectors_to_matrices(pose_vectors)
    if len(pose_matrices) < 2:
        raise ValueError("At least two Oxford poses are required to form a relative pair")

    pair_transforms = []
    for curr_idx in range(1, len(pose_matrices)):
        prev_idx = curr_idx - 1
        pair_transforms.append(np.matmul(np.linalg.inv(pose_matrices[curr_idx]), pose_matrices[prev_idx]))
    return np.asarray(pair_transforms, dtype=np.float64)


def compose_pair_transforms(pair_transforms):
    pair_transforms = np.asarray(pair_transforms, dtype=np.float64)
    trajectory = [np.eye(4, dtype=np.float64)]
    current_pose = np.eye(4, dtype=np.float64)
    for transform in pair_transforms:
        # Oxford relative transforms are encoded as current<-previous, so the
        # segment-local pose must be accumulated by left-multiplication.
        current_pose = np.matmul(transform, current_pose)
        trajectory.append(current_pose.copy())
    return np.asarray(trajectory, dtype=np.float64)


def translation_error_m(pose_error):
    pose_error = np.asarray(pose_error, dtype=np.float64)
    return float(np.linalg.norm(pose_error[:3, 3]))


def rotation_error_rad(pose_error):
    pose_error = np.asarray(pose_error, dtype=np.float64)
    d_value = 0.5 * (pose_error[0, 0] + pose_error[1, 1] + pose_error[2, 2] - 1.0)
    return float(np.arccos(np.clip(d_value, -1.0, 1.0)))


def path_length_m(trajectory):
    trajectory = np.asarray(trajectory, dtype=np.float64)
    if len(trajectory) < 2:
        return 0.0
    positions = trajectory[:, :3, 3]
    deltas = positions[1:] - positions[:-1]
    return float(np.linalg.norm(deltas, axis=1).sum())


def compute_array_stats(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        raise ValueError("Cannot compute statistics for an empty array")
    return {
        "mean": float(values.mean()),
        "rmse": float(np.sqrt(np.mean(np.square(values)))),
        "count": int(values.size),
    }


def compute_pairwise_metrics(pred_q, pred_t, gt_q, gt_t):
    pred_q = np.asarray(pred_q, dtype=np.float64)
    gt_q = np.asarray(gt_q, dtype=np.float64)
    pred_t = np.asarray(pred_t, dtype=np.float64).reshape(len(pred_q), 3)
    gt_t = np.asarray(gt_t, dtype=np.float64).reshape(len(gt_q), 3)

    translation_errors = np.linalg.norm(pred_t - gt_t, axis=-1)
    rotation_errors = quaternion_angle_error_deg_np(pred_q, gt_q)
    trans_stats = compute_array_stats(translation_errors)
    rot_stats = compute_array_stats(rotation_errors)
    return {
        "translation_mean_m": trans_stats["mean"],
        "translation_rmse_m": trans_stats["rmse"],
        "rotation_mean_deg": rot_stats["mean"],
        "rotation_rmse_deg": rot_stats["rmse"],
        "translation_errors_m": translation_errors.tolist(),
        "rotation_errors_deg": rotation_errors.tolist(),
    }


def compute_segment_endpoint_metrics(pred_trajectory, gt_trajectory):
    pred_trajectory = np.asarray(pred_trajectory, dtype=np.float64)
    gt_trajectory = np.asarray(gt_trajectory, dtype=np.float64)
    if len(pred_trajectory) != len(gt_trajectory):
        raise ValueError("Predicted and GT trajectories must have identical lengths")

    segment_length = max(path_length_m(gt_trajectory), 1e-12)
    pose_error = np.matmul(np.linalg.inv(pred_trajectory[-1]), gt_trajectory[-1])
    t_err_m = translation_error_m(pose_error)
    r_err_rad = rotation_error_rad(pose_error)
    return {
        "path_length_m": float(segment_length),
        "translation_error_m": float(t_err_m),
        "translation_error_ratio": float(t_err_m / segment_length),
        "translation_error_percent": float((t_err_m / segment_length) * 100.0),
        "rotation_error_rad": float(r_err_rad),
        "rotation_error_deg": float(np.rad2deg(r_err_rad)),
        "rotation_error_rad_per_m": float(r_err_rad / segment_length),
        "rotation_error_deg_per_m": float(np.rad2deg(r_err_rad) / segment_length),
    }


def compute_trajectory_frame_metrics(pred_trajectory, gt_trajectory):
    pred_trajectory = np.asarray(pred_trajectory, dtype=np.float64)
    gt_trajectory = np.asarray(gt_trajectory, dtype=np.float64)
    if len(pred_trajectory) != len(gt_trajectory):
        raise ValueError("Predicted and GT trajectories must have identical lengths")

    translation_errors = []
    rotation_errors_deg = []
    for pred_pose, gt_pose in zip(pred_trajectory, gt_trajectory):
        pose_error = np.matmul(np.linalg.inv(pred_pose), gt_pose)
        translation_errors.append(translation_error_m(pose_error))
        rotation_errors_deg.append(np.rad2deg(rotation_error_rad(pose_error)))

    trans_stats = compute_array_stats(translation_errors)
    rot_stats = compute_array_stats(rotation_errors_deg)
    return {
        "translation_mean_m": trans_stats["mean"],
        "translation_rmse_m": trans_stats["rmse"],
        "rotation_mean_deg": rot_stats["mean"],
        "rotation_rmse_deg": rot_stats["rmse"],
        "translation_errors_m": list(map(float, translation_errors)),
        "rotation_errors_deg": list(map(float, rotation_errors_deg)),
    }


def build_segment_metrics(segment, pred_q, pred_t, gt_q, gt_t):
    pred_q = np.asarray(pred_q, dtype=np.float64)
    pred_t = np.asarray(pred_t, dtype=np.float64).reshape(len(pred_q), 3)
    gt_q = np.asarray(gt_q, dtype=np.float64)
    gt_t = np.asarray(gt_t, dtype=np.float64).reshape(len(gt_q), 3)
    if len(pred_q) != (len(segment.timestamps) - 1):
        raise ValueError("Segment {} prediction count does not match frame count".format(segment.segment_index))

    pairwise_metrics = compute_pairwise_metrics(pred_q=pred_q, pred_t=pred_t, gt_q=gt_q, gt_t=gt_t)
    pred_pair_transforms = np.asarray(
        [pair_transform_from_quaternion_translation(q, t) for q, t in zip(pred_q, pred_t)],
        dtype=np.float64,
    )
    gt_pair_transforms = global_pose_vectors_to_relative_pairs(segment.poses)
    gt_trajectory = compose_pair_transforms(gt_pair_transforms)
    pred_trajectory = compose_pair_transforms(pred_pair_transforms)
    trajectory_endpoint_metrics = compute_segment_endpoint_metrics(pred_trajectory, gt_trajectory)
    trajectory_frame_metrics = compute_trajectory_frame_metrics(pred_trajectory, gt_trajectory)

    metrics = {
        "sequence_name": segment.sequence_name,
        "segment_index": int(segment.segment_index),
        "frame_count": int(len(segment.timestamps)),
        "pair_count": int(len(pred_q)),
        "start_timestamp": int(segment.start_timestamp),
        "end_timestamp": int(segment.end_timestamp),
        "start_aligned_index": int(segment.start_aligned_index),
        "end_aligned_index": int(segment.end_aligned_index),
        "pairwise": pairwise_metrics,
        "trajectory_endpoint": trajectory_endpoint_metrics,
        "trajectory_per_frame": trajectory_frame_metrics,
    }
    return metrics, pred_trajectory, gt_trajectory


def aggregate_segment_metrics(segment_metrics):
    if not segment_metrics:
        raise ValueError("At least one evaluated segment is required for aggregation")

    def aggregate_values(section_name, field_names):
        section = {}
        for field_name in field_names:
            values = [float(item[section_name][field_name]) for item in segment_metrics]
            section[field_name] = {
                "mean": float(np.mean(values)),
                "rmse": float(np.sqrt(np.mean(np.square(values)))),
            }
        return section

    return {
        "segment_count": int(len(segment_metrics)),
        "pairwise": aggregate_values("pairwise", PAIRWISE_METRIC_FIELDS),
        "trajectory_endpoint": aggregate_values("trajectory_endpoint", TRAJECTORY_ENDPOINT_METRIC_FIELDS),
        "trajectory_per_frame": aggregate_values("trajectory_per_frame", TRAJECTORY_FRAME_METRIC_FIELDS),
    }


def segment_metrics_to_row(segment_metrics):
    return {
        "sequence_name": segment_metrics["sequence_name"],
        "segment_index": int(segment_metrics["segment_index"]),
        "frame_count": int(segment_metrics["frame_count"]),
        "pair_count": int(segment_metrics["pair_count"]),
        "start_timestamp": int(segment_metrics["start_timestamp"]),
        "end_timestamp": int(segment_metrics["end_timestamp"]),
        "start_aligned_index": int(segment_metrics["start_aligned_index"]),
        "end_aligned_index": int(segment_metrics["end_aligned_index"]),
        "pairwise_translation_mean_m": float(segment_metrics["pairwise"]["translation_mean_m"]),
        "pairwise_translation_rmse_m": float(segment_metrics["pairwise"]["translation_rmse_m"]),
        "pairwise_rotation_mean_deg": float(segment_metrics["pairwise"]["rotation_mean_deg"]),
        "pairwise_rotation_rmse_deg": float(segment_metrics["pairwise"]["rotation_rmse_deg"]),
        "trajectory_endpoint_path_length_m": float(segment_metrics["trajectory_endpoint"]["path_length_m"]),
        "trajectory_endpoint_translation_error_m": float(segment_metrics["trajectory_endpoint"]["translation_error_m"]),
        "trajectory_endpoint_translation_error_ratio": float(segment_metrics["trajectory_endpoint"]["translation_error_ratio"]),
        "trajectory_endpoint_translation_error_percent": float(segment_metrics["trajectory_endpoint"]["translation_error_percent"]),
        "trajectory_endpoint_rotation_error_deg": float(segment_metrics["trajectory_endpoint"]["rotation_error_deg"]),
        "trajectory_endpoint_rotation_error_deg_per_m": float(segment_metrics["trajectory_endpoint"]["rotation_error_deg_per_m"]),
        "trajectory_per_frame_translation_mean_m": float(segment_metrics["trajectory_per_frame"]["translation_mean_m"]),
        "trajectory_per_frame_translation_rmse_m": float(segment_metrics["trajectory_per_frame"]["translation_rmse_m"]),
        "trajectory_per_frame_rotation_mean_deg": float(segment_metrics["trajectory_per_frame"]["rotation_mean_deg"]),
        "trajectory_per_frame_rotation_rmse_deg": float(segment_metrics["trajectory_per_frame"]["rotation_rmse_deg"]),
    }


def pose_array_to_rows(trajectory):
    trajectory = np.asarray(trajectory, dtype=np.float64)
    return trajectory[:, :3, :].reshape(len(trajectory), 12)


def _trajectory_dict(trajectory):
    trajectory = np.asarray(trajectory, dtype=np.float64)
    return {idx: trajectory[idx] for idx in range(len(trajectory))}


def segment_local_trajectory_to_global(start_pose_vector, local_trajectory):
    start_pose = np.asarray(OxfordQEDataset._qe_pose_to_matrix(start_pose_vector), dtype=np.float64)
    local_trajectory = np.asarray(local_trajectory, dtype=np.float64)
    global_trajectory = []
    for local_pose in local_trajectory:
        global_trajectory.append(np.matmul(start_pose, np.linalg.inv(local_pose)))
    return np.asarray(global_trajectory, dtype=np.float64)


def convert_segment_trajectories_to_global(segments, trajectories):
    if len(segments) != len(trajectories):
        raise ValueError('Segment metadata and trajectory lists must have identical lengths')
    return [
        segment_local_trajectory_to_global(segment.poses[0], trajectory)
        for segment, trajectory in zip(segments, trajectories)
    ]


def _stack_positions(trajectories):
    positions = [np.asarray(trajectory, dtype=np.float64)[:, :3, 3] for trajectory in trajectories]
    if not positions:
        return np.zeros((0, 3), dtype=np.float64)
    return np.concatenate(positions, axis=0)


def _set_equal_axis_2d(ax, x_values, y_values):
    x_values = np.asarray(x_values, dtype=np.float64)
    y_values = np.asarray(y_values, dtype=np.float64)
    if x_values.size == 0 or y_values.size == 0:
        return
    x_min, x_max = np.min(x_values), np.max(x_values)
    y_min, y_max = np.min(y_values), np.max(y_values)
    x_mean = (x_min + x_max) / 2.0
    y_mean = (y_min + y_max) / 2.0
    plot_radius = max(x_max - x_min, y_max - y_min, 1e-6) / 2.0
    ax.set_xlim([x_mean - plot_radius, x_mean + plot_radius])
    ax.set_ylim([y_mean - plot_radius, y_mean + plot_radius])


def save_trajectory_plot(name, gt_trajectory, pred_trajectory, output_dir):
    gt_positions = np.asarray(gt_trajectory, dtype=np.float64)[:, :3, 3]
    pred_positions = np.asarray(pred_trajectory, dtype=np.float64)[:, :3, 3]

    fig, ax = plt.subplots(figsize=(8, 6), dpi=110)
    ax.plot(gt_positions[:, 0], gt_positions[:, 2], "r-", label="GT")
    ax.plot(pred_positions[:, 0], pred_positions[:, 2], "b-", label="Ours")
    ax.plot([0.0], [0.0], "ko", label="Start")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    ax.legend(loc="upper right")
    _set_equal_axis_2d(ax, np.concatenate([gt_positions[:, 0], pred_positions[:, 0]]), np.concatenate([gt_positions[:, 2], pred_positions[:, 2]]))

    png_path = os.path.join(output_dir, "{}_trajectory.png".format(name))
    pdf_path = os.path.join(output_dir, "{}_trajectory.pdf".format(name))
    fig.tight_layout()
    fig.savefig(png_path, bbox_inches="tight", pad_inches=0.1)
    pdf = matplotlib.backends.backend_pdf.PdfPages(pdf_path)
    pdf.savefig(fig)
    pdf.close()
    plt.close(fig)


def save_xyz_plot(name, gt_trajectory, pred_trajectory, output_dir):
    gt_xyz = np.asarray(gt_trajectory, dtype=np.float64)[:, :3, 3]
    pred_xyz = np.asarray(pred_trajectory, dtype=np.float64)[:, :3, 3]

    fig, axes = plt.subplots(3, sharex="col", figsize=(20, 10))
    x_axis = range(len(pred_xyz))
    y_labels = ["$x$ (m)", "$y$ (m)", "$z$ (m)"]
    for axis_index in range(3):
        axes[axis_index].plot(x_axis, pred_xyz[:, axis_index], "b-", label="Ours")
        axes[axis_index].plot(x_axis, gt_xyz[:, axis_index], "r-", label="GT")
        axes[axis_index].set_ylabel(y_labels[axis_index])
        axes[axis_index].legend(loc="upper right", frameon=True)
    axes[2].set_xlabel("index")
    axes[0].set_title("XYZ")

    png_path = os.path.join(output_dir, "{}_xyz.png".format(name))
    pdf_path = os.path.join(output_dir, "{}_xyz.pdf".format(name))
    fig.tight_layout()
    fig.savefig(png_path, bbox_inches="tight", pad_inches=0.1)
    pdf = matplotlib.backends.backend_pdf.PdfPages(pdf_path)
    pdf.savefig(fig)
    pdf.close()
    plt.close(fig)


def save_rpy_plot(name, gt_trajectory, pred_trajectory, output_dir, axes_code="szxy"):
    gt_rpy = np.asarray([tr.euler_from_matrix(pose, axes=axes_code) for pose in np.asarray(gt_trajectory, dtype=np.float64)])
    pred_rpy = np.asarray([tr.euler_from_matrix(pose, axes=axes_code) for pose in np.asarray(pred_trajectory, dtype=np.float64)])

    fig, axes = plt.subplots(3, sharex="col", figsize=(20, 10))
    x_axis = range(len(pred_rpy))
    y_labels = ["$roll$ (deg)", "$pitch$ (deg)", "$yaw$ (deg)"]
    for axis_index in range(3):
        axes[axis_index].plot(x_axis, np.rad2deg(pred_rpy[:, axis_index]), "b-", label="Ours")
        axes[axis_index].plot(x_axis, np.rad2deg(gt_rpy[:, axis_index]), "r-", label="GT")
        axes[axis_index].set_ylabel(y_labels[axis_index])
        axes[axis_index].legend(loc="upper right", frameon=True)
    axes[2].set_xlabel("index")
    axes[0].set_title("RPY")

    png_path = os.path.join(output_dir, "{}_rpy.png".format(name))
    pdf_path = os.path.join(output_dir, "{}_rpy.pdf".format(name))
    fig.tight_layout()
    fig.savefig(png_path, bbox_inches="tight", pad_inches=0.1)
    pdf = matplotlib.backends.backend_pdf.PdfPages(pdf_path)
    pdf.savefig(fig)
    pdf.close()
    plt.close(fig)


def save_path_plot(name, gt_trajectory, pred_trajectory, output_dir):
    gt_positions = np.asarray(gt_trajectory, dtype=np.float64)[:, :3, 3]
    pred_positions = np.asarray(pred_trajectory, dtype=np.float64)[:, :3, 3]

    fig = plt.figure(figsize=(20, 6), dpi=100)
    axes = [fig.add_subplot(1, 3, index + 1) for index in range(3)]
    projections = (
        (0, 2, "x (m)", "z (m)"),
        (0, 1, "x (m)", "y (m)"),
        (1, 2, "y (m)", "z (m)"),
    )
    for axis, (x_idx, y_idx, x_label, y_label) in zip(axes, projections):
        axis.plot(gt_positions[:, x_idx], gt_positions[:, y_idx], "r-", label="GT")
        axis.plot(pred_positions[:, x_idx], pred_positions[:, y_idx], "b-", label="Ours")
        axis.plot([0.0], [0.0], "ko", label="Start")
        axis.set_xlabel(x_label)
        axis.set_ylabel(y_label)
        axis.legend(loc="upper right")
        _set_equal_axis_2d(
            axis,
            np.concatenate([gt_positions[:, x_idx], pred_positions[:, x_idx]]),
            np.concatenate([gt_positions[:, y_idx], pred_positions[:, y_idx]]),
        )

    png_path = os.path.join(output_dir, "{}_path.png".format(name))
    pdf_path = os.path.join(output_dir, "{}_path.pdf".format(name))
    fig.tight_layout()
    fig.savefig(png_path, bbox_inches="tight", pad_inches=0.1)
    pdf = matplotlib.backends.backend_pdf.PdfPages(pdf_path)
    pdf.savefig(fig)
    pdf.close()
    plt.close(fig)


def save_path_3d_plot(name, gt_trajectory, pred_trajectory, output_dir):
    gt_positions = np.asarray(gt_trajectory, dtype=np.float64)[:, :3, 3]
    pred_positions = np.asarray(pred_trajectory, dtype=np.float64)[:, :3, 3]

    fig = plt.figure(figsize=(8, 8), dpi=110)
    axis = fig.add_subplot(111, projection="3d")
    axis.plot(pred_positions[:, 0], pred_positions[:, 2], pred_positions[:, 1], "b-", label="Ours")
    axis.plot(gt_positions[:, 0], gt_positions[:, 2], gt_positions[:, 1], "r-", label="GT")
    axis.plot([0.0], [0.0], [0.0], "ko", label="Start")
    axis.set_xlabel("x (m)")
    axis.set_ylabel("z (m)")
    axis.set_zlabel("y (m)")
    axis.view_init(elev=20.0, azim=-35.0)
    axis.legend(loc="upper right")

    all_points = np.concatenate([
        np.stack([pred_positions[:, 0], pred_positions[:, 2], pred_positions[:, 1]], axis=1),
        np.stack([gt_positions[:, 0], gt_positions[:, 2], gt_positions[:, 1]], axis=1),
    ], axis=0)
    center = np.mean(all_points, axis=0)
    max_radius = max(np.max(np.abs(all_points - center), axis=0).max(), 1e-6)
    axis.set_xlim([center[0] - max_radius, center[0] + max_radius])
    axis.set_ylim([center[1] - max_radius, center[1] + max_radius])
    axis.set_zlim([center[2] - max_radius, center[2] + max_radius])

    png_path = os.path.join(output_dir, "{}_path_3D.png".format(name))
    pdf_path = os.path.join(output_dir, "{}_path_3D.pdf".format(name))
    fig.tight_layout()
    fig.savefig(png_path, bbox_inches="tight", pad_inches=0.1)
    pdf = matplotlib.backends.backend_pdf.PdfPages(pdf_path)
    pdf.savefig(fig)
    pdf.close()
    plt.close(fig)


def save_segment_plots(name, gt_trajectory, pred_trajectory, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    save_trajectory_plot(name, gt_trajectory, pred_trajectory, output_dir)
    save_xyz_plot(name, gt_trajectory, pred_trajectory, output_dir)
    save_rpy_plot(name, gt_trajectory, pred_trajectory, output_dir)
    save_path_plot(name, gt_trajectory, pred_trajectory, output_dir)
    save_path_3d_plot(name, gt_trajectory, pred_trajectory, output_dir)


def save_stitched_path_plot(name, gt_trajectories, pred_trajectories, output_dir, background_trajectory=None):
    gt_positions_list = [np.asarray(trajectory, dtype=np.float64)[:, :3, 3] for trajectory in gt_trajectories]
    pred_positions_list = [np.asarray(trajectory, dtype=np.float64)[:, :3, 3] for trajectory in pred_trajectories]
    all_positions_parts = [_stack_positions(gt_trajectories), _stack_positions(pred_trajectories)]
    background_positions = None
    if background_trajectory is not None:
        background_positions = np.asarray(background_trajectory, dtype=np.float64)[:, :3, 3]
        all_positions_parts.append(background_positions)
    all_positions = np.concatenate(all_positions_parts, axis=0)

    fig = plt.figure(figsize=(20, 6), dpi=100)
    axes = [fig.add_subplot(1, 3, index + 1) for index in range(3)]
    projections = (
        (0, 2, 'x (m)', 'z (m)'),
        (0, 1, 'x (m)', 'y (m)'),
        (1, 2, 'y (m)', 'z (m)'),
    )
    for axis, (x_idx, y_idx, x_label, y_label) in zip(axes, projections):
        if background_positions is not None:
            axis.plot(
                background_positions[:, x_idx],
                background_positions[:, y_idx],
                color='#888888',
                linewidth=1.0,
                label='TXT aligned to full_h5',
            )
        for segment_index, gt_positions in enumerate(gt_positions_list):
            axis.plot(
                gt_positions[:, x_idx],
                gt_positions[:, y_idx],
                'r-',
                label='GT' if segment_index == 0 else None,
            )
            axis.plot(
                [gt_positions[0, x_idx]],
                [gt_positions[0, y_idx]],
                'ko',
                label='Start' if segment_index == 0 else None,
            )
        for segment_index, pred_positions in enumerate(pred_positions_list):
            axis.plot(
                pred_positions[:, x_idx],
                pred_positions[:, y_idx],
                'b-',
                label='Ours' if segment_index == 0 else None,
            )
        axis.set_xlabel(x_label)
        axis.set_ylabel(y_label)
        axis.legend(loc='upper right')
        _set_equal_axis_2d(axis, all_positions[:, x_idx], all_positions[:, y_idx])

    png_path = os.path.join(output_dir, '{}_path.png'.format(name))
    pdf_path = os.path.join(output_dir, '{}_path.pdf'.format(name))
    fig.tight_layout()
    fig.savefig(png_path, bbox_inches='tight', pad_inches=0.1)
    pdf = matplotlib.backends.backend_pdf.PdfPages(pdf_path)
    pdf.savefig(fig)
    pdf.close()
    plt.close(fig)


def save_stitched_path_3d_plot(name, gt_trajectories, pred_trajectories, output_dir, background_trajectory=None):
    gt_positions_list = [np.asarray(trajectory, dtype=np.float64)[:, :3, 3] for trajectory in gt_trajectories]
    pred_positions_list = [np.asarray(trajectory, dtype=np.float64)[:, :3, 3] for trajectory in pred_trajectories]
    background_positions = None
    if background_trajectory is not None:
        background_positions = np.asarray(background_trajectory, dtype=np.float64)[:, :3, 3]

    fig = plt.figure(figsize=(8, 8), dpi=110)
    axis = fig.add_subplot(111, projection='3d')
    if background_positions is not None:
        axis.plot(
            background_positions[:, 0],
            background_positions[:, 2],
            background_positions[:, 1],
            color='#888888',
            linewidth=1.0,
            label='TXT aligned to full_h5',
        )
    for segment_index, pred_positions in enumerate(pred_positions_list):
        axis.plot(
            pred_positions[:, 0],
            pred_positions[:, 2],
            pred_positions[:, 1],
            'b-',
            label='Ours' if segment_index == 0 else None,
        )
    for segment_index, gt_positions in enumerate(gt_positions_list):
        axis.plot(
            gt_positions[:, 0],
            gt_positions[:, 2],
            gt_positions[:, 1],
            'r-',
            label='GT' if segment_index == 0 else None,
        )
        axis.plot(
            [gt_positions[0, 0]],
            [gt_positions[0, 2]],
            [gt_positions[0, 1]],
            'ko',
            label='Start' if segment_index == 0 else None,
        )
    axis.set_xlabel('x (m)')
    axis.set_ylabel('z (m)')
    axis.set_zlabel('y (m)')
    axis.view_init(elev=20.0, azim=-35.0)
    axis.legend(loc='upper right')

    all_points_parts = [
        np.stack([_stack_positions(pred_trajectories)[:, 0], _stack_positions(pred_trajectories)[:, 2], _stack_positions(pred_trajectories)[:, 1]], axis=1),
        np.stack([_stack_positions(gt_trajectories)[:, 0], _stack_positions(gt_trajectories)[:, 2], _stack_positions(gt_trajectories)[:, 1]], axis=1),
    ]
    if background_positions is not None:
        all_points_parts.append(
            np.stack([background_positions[:, 0], background_positions[:, 2], background_positions[:, 1]], axis=1)
        )
    all_points = np.concatenate(all_points_parts, axis=0)
    center = np.mean(all_points, axis=0)
    max_radius = max(np.max(np.abs(all_points - center), axis=0).max(), 1e-6)
    axis.set_xlim([center[0] - max_radius, center[0] + max_radius])
    axis.set_ylim([center[1] - max_radius, center[1] + max_radius])
    axis.set_zlim([center[2] - max_radius, center[2] + max_radius])

    png_path = os.path.join(output_dir, '{}_path_3D.png'.format(name))
    pdf_path = os.path.join(output_dir, '{}_path_3D.pdf'.format(name))
    fig.tight_layout()
    fig.savefig(png_path, bbox_inches='tight', pad_inches=0.1)
    pdf = matplotlib.backends.backend_pdf.PdfPages(pdf_path)
    pdf.savefig(fig)
    pdf.close()
    plt.close(fig)


def save_full_route_plots(name, segments, gt_trajectories, pred_trajectories, output_dir, background_trajectory=None):
    os.makedirs(output_dir, exist_ok=True)
    gt_global_trajectories = convert_segment_trajectories_to_global(segments, gt_trajectories)
    pred_global_trajectories = convert_segment_trajectories_to_global(segments, pred_trajectories)
    save_stitched_path_plot(
        name,
        gt_global_trajectories,
        pred_global_trajectories,
        output_dir,
        background_trajectory=background_trajectory,
    )
    save_stitched_path_3d_plot(
        name,
        gt_global_trajectories,
        pred_global_trajectories,
        output_dir,
        background_trajectory=background_trajectory,
    )
