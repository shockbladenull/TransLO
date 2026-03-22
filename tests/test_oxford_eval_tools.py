import math

import numpy as np

from dataset_factory import split_oxford_selected_sequence_into_segments
from oxford_lo300_rank_ckpts import (
    build_checkpoint_gpu_pairs,
    build_worker_assignments,
    extract_checkpoint_epoch,
    format_progress_postfix,
    get_nested_metric,
    parse_gpu_ids,
    select_checkpoint_paths,
    should_evaluate_checkpoint,
    sort_evaluation_rows,
)
from tools.oxford_eval_tools import (
    OxfordSegment,
    aggregate_segment_metrics,
    build_segment_metrics,
    compose_pair_transforms,
    global_pose_vectors_to_relative_pairs,
    qe_pose_vectors_to_matrices,
    save_full_route_plots,
    segment_local_trajectory_to_global,
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



def _pose_vector_from_matrix(transform):
    transform = np.asarray(transform, dtype=np.float64)
    return np.asarray([
        transform[0, 0], transform[0, 1], transform[0, 2],
        transform[1, 0], transform[1, 1], transform[1, 2],
        transform[2, 0], transform[2, 1], transform[2, 2],
        transform[0, 3], transform[1, 3], transform[2, 3],
    ], dtype=np.float32)



def _rotation_z_transform(degrees, tx=0.0, ty=0.0, tz=0.0):
    radians = np.deg2rad(degrees)
    cos_value = np.cos(radians)
    sin_value = np.sin(radians)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray([
        [cos_value, -sin_value, 0.0],
        [sin_value, cos_value, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
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


def test_segment_local_trajectory_to_global_recovers_original_global_poses():
    pose_vectors = np.stack([
        _pose_vector(tx=0.0),
        _pose_vector(tx=1.0),
        _pose_vector(tx=3.0),
    ], axis=0)

    local_trajectory = compose_pair_transforms(global_pose_vectors_to_relative_pairs(pose_vectors))
    recovered_global = segment_local_trajectory_to_global(pose_vectors[0], local_trajectory)

    np.testing.assert_allclose(recovered_global, qe_pose_vectors_to_matrices(pose_vectors), atol=1e-8)


def test_compose_pair_transforms_respects_non_commutative_rotation_chain():
    pose_matrices = np.asarray([
        _rotation_z_transform(0.0, tx=0.0, ty=0.0),
        _rotation_z_transform(30.0, tx=1.0, ty=0.0),
        _rotation_z_transform(60.0, tx=2.0, ty=1.0),
        _rotation_z_transform(95.0, tx=2.5, ty=2.0),
    ], dtype=np.float64)
    pose_vectors = np.stack([_pose_vector_from_matrix(transform) for transform in pose_matrices], axis=0)

    local_trajectory = compose_pair_transforms(global_pose_vectors_to_relative_pairs(pose_vectors))
    recovered_global = segment_local_trajectory_to_global(pose_vectors[0], local_trajectory)

    np.testing.assert_allclose(recovered_global, pose_matrices, atol=1e-8)



def test_save_full_route_plots_writes_combined_path_artifacts(tmp_path):
    segment_a = OxfordSegment(
        sequence_name='route',
        segment_index=0,
        timestamps=np.asarray([10, 20, 30], dtype=np.int64),
        poses=np.stack([
            _pose_vector(tx=0.0),
            _pose_vector(tx=1.0),
            _pose_vector(tx=2.0),
        ], axis=0),
        aligned_indices=np.asarray([0, 1, 2], dtype=np.int64),
        scan_dir='/tmp',
        start_timestamp=10,
        end_timestamp=30,
        start_aligned_index=0,
        end_aligned_index=2,
    )
    segment_b = OxfordSegment(
        sequence_name='route',
        segment_index=1,
        timestamps=np.asarray([50, 60, 70], dtype=np.int64),
        poses=np.stack([
            _pose_vector(tx=10.0),
            _pose_vector(tx=11.0),
            _pose_vector(tx=12.0),
        ], axis=0),
        aligned_indices=np.asarray([5, 6, 7], dtype=np.int64),
        scan_dir='/tmp',
        start_timestamp=50,
        end_timestamp=70,
        start_aligned_index=5,
        end_aligned_index=7,
    )

    gt_trajectories = [
        compose_pair_transforms(global_pose_vectors_to_relative_pairs(segment_a.poses)),
        compose_pair_transforms(global_pose_vectors_to_relative_pairs(segment_b.poses)),
    ]
    pred_trajectories = [trajectory.copy() for trajectory in gt_trajectories]
    background_trajectory = qe_pose_vectors_to_matrices(
        np.concatenate([segment_a.poses, segment_b.poses], axis=0)
    )

    save_full_route_plots(
        'full_route',
        [segment_a, segment_b],
        gt_trajectories,
        pred_trajectories,
        str(tmp_path),
        background_trajectory=background_trajectory,
    )

    assert (tmp_path / 'full_route_path.png').is_file()
    assert (tmp_path / 'full_route_path.pdf').is_file()
    assert (tmp_path / 'full_route_path_3D.png').is_file()
    assert (tmp_path / 'full_route_path_3D.pdf').is_file()


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


def test_get_nested_metric_reads_dotted_summary_key():
    payload = {
        "aggregates": {
            "trajectory_endpoint": {
                "translation_error_percent": {
                    "mean": 1.25,
                },
            },
        },
    }

    value = get_nested_metric(payload, "aggregates.trajectory_endpoint.translation_error_percent.mean")

    assert value == 1.25


def test_extract_checkpoint_epoch_reads_epoch_from_filename():
    checkpoint_path = '/tmp/translo_model_110_-15.000000.pth.tar'

    assert extract_checkpoint_epoch(checkpoint_path) == 110



def test_should_evaluate_checkpoint_filters_epochs_after_100_every_5_rounds():
    assert should_evaluate_checkpoint(105, after_epoch=100, epoch_stride=5)
    assert should_evaluate_checkpoint(110, after_epoch=100, epoch_stride=5)
    assert not should_evaluate_checkpoint(100, after_epoch=100, epoch_stride=5)
    assert not should_evaluate_checkpoint(108, after_epoch=100, epoch_stride=5)



def test_select_checkpoint_paths_keeps_periodic_epochs_only():
    checkpoint_paths = [
        '/tmp/translo_model_098_-1.0.pth.tar',
        '/tmp/translo_model_100_-1.0.pth.tar',
        '/tmp/translo_model_105_-1.0.pth.tar',
        '/tmp/translo_model_108_-1.0.pth.tar',
        '/tmp/translo_model_110_-1.0.pth.tar',
    ]

    selected_paths = select_checkpoint_paths(checkpoint_paths, after_epoch=100, epoch_stride=5)

    assert selected_paths == [
        '/tmp/translo_model_105_-1.0.pth.tar',
        '/tmp/translo_model_110_-1.0.pth.tar',
    ]



def test_sort_evaluation_rows_orders_by_epoch_then_name():
    rows = [
        {
            'checkpoint_name': 'epoch_110_b',
            'checkpoint_epoch': 110,
        },
        {
            'checkpoint_name': 'epoch_105',
            'checkpoint_epoch': 105,
        },
        {
            'checkpoint_name': 'epoch_110_a',
            'checkpoint_epoch': 110,
        },
    ]

    sorted_rows = sort_evaluation_rows(rows)

    assert [row['checkpoint_name'] for row in sorted_rows] == ['epoch_105', 'epoch_110_a', 'epoch_110_b']


def test_parse_gpu_ids_defaults_to_single_gpu_namespace():
    class Args:
        gpu = 3
        gpu_ids = None

    assert parse_gpu_ids(Args()) == [3]


def test_build_worker_assignments_repeats_gpu_ids_per_job_slot():
    checkpoint_paths = ["a", "b", "c", "d", "e"]

    worker_gpu_ids, assignments = build_worker_assignments(checkpoint_paths, gpu_ids=[0, 2], jobs_per_gpu=2)

    assert worker_gpu_ids == [0, 0, 2, 2]
    assert assignments == [["a", "e"], ["b"], ["c"], ["d"]]


def test_build_checkpoint_gpu_pairs_interleaves_worker_slots():
    checkpoint_paths = ["a", "b", "c", "d", "e"]

    checkpoint_gpu_pairs, worker_gpu_ids = build_checkpoint_gpu_pairs(checkpoint_paths, gpu_ids=[0, 2], jobs_per_gpu=2)

    assert worker_gpu_ids == [0, 0, 2, 2]
    assert checkpoint_gpu_pairs == [("a", 0), ("b", 0), ("c", 2), ("d", 2), ("e", 0)]


def test_format_progress_postfix_includes_epoch_and_primary_metrics():
    last_row = {
        'checkpoint_name': 'translo_model_110_-6.500000',
        'checkpoint_epoch': 110,
        'trajectory_endpoint_translation_error_percent': 1.25,
        'trajectory_endpoint_rotation_error_deg_per_m': 0.15,
        'worker_gpu': 2,
    }

    postfix = format_progress_postfix(last_row)

    assert 'last=' in postfix
    assert '@g2' in postfix
    assert 'epoch=110' in postfix
    assert '1.2500/0.1500' in postfix
