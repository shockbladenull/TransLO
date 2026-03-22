import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from tools import oxford_train_eval


def _pose_vector(tx):
    return np.asarray(
        [
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
            tx, 0.0, 0.0,
        ],
        dtype=np.float32,
    )


def _base_args():
    return SimpleNamespace(
        oxford_detailed_val=True,
        oxford_detailed_val_interval=5,
        val_dataset_type="oxford_qe",
        oxford_pose_source="txt",
        oxford_val_seqs=["seq_a", "seq_b"],
        oxford_root="/tmp/oxford",
        oxford_h5_name="mask.h5",
        oxford_h5_root="/tmp/h5",
        oxford_full_h5_name="full.h5",
        oxford_full_h5_root="/tmp/full_h5",
        oxford_pose_root="/tmp/poses",
        oxford_pose_txt_template="poses_{sequence_short}.txt",
        oxford_pose_skip_start=5,
        oxford_pose_skip_end=5,
        oxford_trim_edges=0,
        frame_gap=1,
        eval_batch_size=1,
        workers=0,
    )


def test_should_run_oxford_detailed_val_requires_enabled_txt_oxford():
    args = _base_args()
    assert not oxford_train_eval.should_run_oxford_detailed_val(args, 4)
    assert oxford_train_eval.should_run_oxford_detailed_val(args, 5)

    args.oxford_detailed_val = False
    assert not oxford_train_eval.should_run_oxford_detailed_val(args, 5)

    args = _base_args()
    args.val_dataset_type = "kitti"
    assert not oxford_train_eval.should_run_oxford_detailed_val(args, 5)

    args = _base_args()
    args.oxford_pose_source = "h5"
    assert not oxford_train_eval.should_run_oxford_detailed_val(args, 5)


def test_run_oxford_detailed_val_writes_full_route_outputs(tmp_path, monkeypatch):
    args = _base_args()
    calls = []

    def fake_load_sequence(**kwargs):
        sequence_name = kwargs["sequence_name"]
        base_tx = 0.0 if sequence_name == "seq_a" else 10.0
        poses = np.stack(
            [
                _pose_vector(base_tx + 0.0),
                _pose_vector(base_tx + 1.0),
                _pose_vector(base_tx + 2.0),
            ],
            axis=0,
        )
        return {
            "scan_dir": str(tmp_path / sequence_name / "velodyne_left"),
            "selected_timestamps": np.asarray([10, 20, 30], dtype=np.int64),
            "selected_poses": poses,
            "selected_aligned_indices": np.asarray([0, 1, 2], dtype=np.int64),
            "aligned_timestamps": np.asarray([10, 20, 30], dtype=np.int64),
            "aligned_poses": poses,
        }

    def fake_evaluate_segment(model, device, segment, eval_args, show_progress):
        trajectory = np.stack([np.eye(4, dtype=np.float64) for _ in range(len(segment.timestamps))], axis=0)
        return (
            {
                "pairwise": {
                    "translation_mean_m": 0.1,
                    "translation_rmse_m": 0.2,
                    "rotation_mean_deg": 0.3,
                    "rotation_rmse_deg": 0.4,
                },
                "trajectory_endpoint": {
                    "path_length_m": 10.0,
                    "translation_error_m": 1.0,
                    "translation_error_ratio": 0.1,
                    "translation_error_percent": 10.0,
                    "rotation_error_rad": 0.01,
                    "rotation_error_deg": 0.57,
                    "rotation_error_rad_per_m": 0.001,
                    "rotation_error_deg_per_m": 0.057,
                },
                "trajectory_per_frame": {
                    "translation_mean_m": 0.5,
                    "translation_rmse_m": 0.6,
                    "rotation_mean_deg": 0.7,
                    "rotation_rmse_deg": 0.8,
                },
            },
            trajectory,
            trajectory.copy(),
        )

    def fake_save_full_route_plots(name, segments, gt_trajectories, pred_trajectories, output_dir, background_trajectory=None):
        calls.append((name, output_dir, len(segments), len(gt_trajectories), len(pred_trajectories)))
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        for filename in (
            "full_route_path.png",
            "full_route_path.pdf",
            "full_route_path_3D.png",
            "full_route_path_3D.pdf",
        ):
            (output_path / filename).write_text("artifact", encoding="utf-8")

    monkeypatch.setattr(oxford_train_eval, "load_oxford_txt_masked_sequence", fake_load_sequence)
    monkeypatch.setattr(oxford_train_eval, "evaluate_segment", fake_evaluate_segment)
    monkeypatch.setattr(oxford_train_eval, "save_full_route_plots", fake_save_full_route_plots)

    summaries = oxford_train_eval.run_oxford_detailed_val(
        model=object(),
        device=torch.device("cpu"),
        args=args,
        eval_dir=str(tmp_path / "eval"),
        epoch=5,
        show_progress=False,
    )

    assert len(summaries) == 2
    assert len(calls) == 2

    for sequence_name in args.oxford_val_seqs:
        output_dir = tmp_path / "eval" / "oxford_detailed" / "epoch_005" / sequence_name
        summary_path = output_dir / "summary.json"
        assert summary_path.is_file()
        assert (output_dir / "full_route_path_3D.png").is_file()
        assert (output_dir / "full_route_path_3D.pdf").is_file()

        with summary_path.open() as handle:
            payload = json.load(handle)
        assert payload["sequence_name"] == sequence_name
        assert payload["mask_h5_name"] == "mask.h5"
        assert payload["segment_count"] == 1
        assert payload["ranking_metrics"]["pairwise_translation_mean_m"] == 0.1
        assert payload["route_artifacts"]["full_route_path_3D_png"] == "full_route_path_3D.png"
