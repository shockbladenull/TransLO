import math

import numpy as np

from dataset_factory import split_oxford_selected_sequence_into_segments
from tools.oxford_eval_tools import (
    OxfordSegment,
    aggregate_segment_metrics,
    build_segment_metrics,
    compose_pair_transforms,
    global_pose_vectors_to_relative_pairs,
)


def _pose_vector(tx=0.0, ty=0.0, tz=0.0):
    return np.asarray([
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
        tx, ty, tz,
    ], dtype=np.float32)


def _translation_transform(tx=0.0, ty=0.0, tz=0.0):
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = np.asarray([tx, ty, tz], dtype=np.float64)
    return transform


def test_split_oxford_selected_sequence_into_segments_splits_on_full_timeline_gaps():
    timestamps = np.asarray([10, 20, 40, 50, 80], dtype=np.int64)
    poses = np.stack([_pose_vector(tx=float(index)) for index in range(len(timestamps))], axis=0)
    aligned_indices = np.asarray([1, 2, 4, 5, 8], dtype=np.int64)

    segments = split_oxford_selected_sequence_into_segments(
        selected_timestamps=timestamps,
        selected_poses=poses,
        selected_aligned_indices=aligned_indices,
    )

    assert [len(segment["timestamps"]) for segment in segments] == [2, 2, 1]
    assert [segment["start_timestamp"] for segment in segments] == [10, 40, 80]
    assert [segment["end_timestamp"] for segment in segments] == [20, 50, 80]


def test_compose_pair_transforms_builds_segment_local_trajectory():
    pair_transforms = np.asarray(
        [
            _translation_transform(tx=1.0),
            _translation_transform(tx=2.0),
        ],
        dtype=np.float64,
    )

    trajectory = compose_pair_transforms(pair_transforms)

    assert trajectory.shape == (3, 4, 4)
    np.testing.assert_allclose(trajectory[0], np.eye(4), atol=1e-8)
    np.testing.assert_allclose(trajectory[1][:3, 3], np.asarray([1.0, 0.0, 0.0]), atol=1e-8)
    np.testing.assert_allclose(trajectory[2][:3, 3], np.asarray([3.0, 0.0, 0.0]), atol=1e-8)


def test_build_segment_metrics_returns_zero_for_perfect_prediction():
    segment = OxfordSegment(
        sequence_name="2019-01-17-14-03-00-radar-oxford-10k",
        segment_index=1,
        timestamps=np.asarray([10, 20, 30], dtype=np.int64),
        poses=np.stack([
            _pose_vector(tx=0.0),
            _pose_vector(tx=1.0),
            _pose_vector(tx=3.0),
        ], axis=0),
        aligned_indices=np.asarray([0, 1, 2], dtype=np.int64),
        scan_dir="/tmp",
        start_timestamp=10,
        end_timestamp=30,
        start_aligned_index=0,
        end_aligned_index=2,
    )
    gt_pair_transforms = global_pose_vectors_to_relative_pairs(segment.poses)
    pred_q = np.tile(np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64), (len(gt_pair_transforms), 1))
    pred_t = gt_pair_transforms[:, :3, 3]
    gt_q = pred_q.copy()
    gt_t = pred_t.copy()

    metrics, pred_trajectory, gt_trajectory = build_segment_metrics(
        segment=segment,
        pred_q=pred_q,
        pred_t=pred_t,
        gt_q=gt_q,
        gt_t=gt_t,
    )

    assert metrics["pairwise"]["translation_mean_m"] == 0.0
    assert metrics["pairwise"]["translation_rmse_m"] == 0.0
    assert metrics["pairwise"]["rotation_mean_deg"] == 0.0
    assert metrics["pairwise"]["rotation_rmse_deg"] == 0.0
    assert metrics["trajectory_endpoint"]["translation_error_m"] == 0.0
    assert metrics["trajectory_endpoint"]["rotation_error_deg"] == 0.0
    assert metrics["trajectory_per_frame"]["translation_mean_m"] == 0.0
    assert metrics["trajectory_per_frame"]["rotation_mean_deg"] == 0.0
    np.testing.assert_allclose(pred_trajectory, gt_trajectory, atol=1e-8)


def test_aggregate_segment_metrics_keeps_mean_and_rmse():
    segment_metrics = [
        {
            "pairwise": {
                "translation_mean_m": 1.0,
                "translation_rmse_m": 2.0,
                "rotation_mean_deg": 3.0,
                "rotation_rmse_deg": 4.0,
            },
            "trajectory_endpoint": {
                "path_length_m": 100.0,
                "translation_error_m": 5.0,
                "translation_error_ratio": 0.05,
                "translation_error_percent": 5.0,
                "rotation_error_rad": 0.1,
                "rotation_error_deg": 10.0,
                "rotation_error_rad_per_m": 0.001,
                "rotation_error_deg_per_m": 0.1,
            },
            "trajectory_per_frame": {
                "translation_mean_m": 6.0,
                "translation_rmse_m": 7.0,
                "rotation_mean_deg": 8.0,
                "rotation_rmse_deg": 9.0,
            },
        },
        {
            "pairwise": {
                "translation_mean_m": 3.0,
                "translation_rmse_m": 4.0,
                "rotation_mean_deg": 5.0,
                "rotation_rmse_deg": 6.0,
            },
            "trajectory_endpoint": {
                "path_length_m": 200.0,
                "translation_error_m": 7.0,
                "translation_error_ratio": 0.07,
                "translation_error_percent": 7.0,
                "rotation_error_rad": 0.3,
                "rotation_error_deg": 30.0,
                "rotation_error_rad_per_m": 0.003,
                "rotation_error_deg_per_m": 0.3,
            },
            "trajectory_per_frame": {
                "translation_mean_m": 8.0,
                "translation_rmse_m": 9.0,
                "rotation_mean_deg": 10.0,
                "rotation_rmse_deg": 11.0,
            },
        },
    ]

    summary = aggregate_segment_metrics(segment_metrics)

    assert summary["segment_count"] == 2
    assert summary["pairwise"]["translation_mean_m"]["mean"] == 2.0
    assert math.isclose(summary["pairwise"]["translation_mean_m"]["rmse"], math.sqrt(5.0))
    assert summary["trajectory_endpoint"]["translation_error_percent"]["mean"] == 6.0
    assert math.isclose(
        summary["trajectory_endpoint"]["translation_error_percent"]["rmse"],
        math.sqrt((5.0 ** 2 + 7.0 ** 2) / 2.0),
    )
