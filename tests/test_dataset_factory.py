import numpy as np
import h5py

import dataset_factory
from dataset_factory import (
    OxfordQEDataset,
    _align_txt_poses_to_full_timestamps,
    _oxford_sequence_short_name,
    _resolve_oxford_sequence_file,
    _select_masked_txt_poses,
    _txt_pose_row_to_qe_pose,
)


def _write_h5(path, timestamps):
    with h5py.File(path, "w") as h5_file:
        h5_file.create_dataset("valid_timestamps", data=np.asarray(timestamps, dtype=np.int64))


def _write_scan(path):
    np.asarray([1.0, 2.0, 3.0, 1.0], dtype=np.float32).tofile(path)


def test_txt_pose_row_to_qe_pose_reorders_row_major_3x4():
    txt_row = np.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], dtype=np.float32)
    qe_pose = _txt_pose_row_to_qe_pose(txt_row)
    np.testing.assert_array_equal(qe_pose, np.asarray([1, 2, 3, 5, 6, 7, 9, 10, 11, 4, 8, 12], dtype=np.float32))


def test_resolve_oxford_sequence_file_falls_back_to_short_sequence_name(tmp_path):
    long_name = "2019-01-14-12-05-52-radar-oxford-10k"
    short_dir = tmp_path / "0552"
    short_dir.mkdir()
    target = short_dir / "velodyne_left_calibrateFalse_SCR300m.h5"
    target.write_bytes(b"test")

    resolved = _resolve_oxford_sequence_file(
        sequence_name=long_name,
        seq_dir=str(tmp_path / long_name),
        filename="velodyne_left_calibrateFalse_SCR300m.h5",
        root_override=str(tmp_path),
    )

    assert resolved == str(target)


def test_txt_alignment_and_mask_filter_drop_only_trimmed_edges():
    full_timestamps = np.asarray([10, 20, 30, 40, 50, 60], dtype=np.int64)
    txt_poses = np.asarray(
        [
            [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0],
            [1, 0, 0, 2, 0, 1, 0, 0, 0, 0, 1, 0],
            [1, 0, 0, 4, 0, 1, 0, 0, 0, 0, 1, 0],
            [1, 0, 0, 5, 0, 1, 0, 0, 0, 0, 1, 0],
        ],
        dtype=np.float32,
    )
    aligned_timestamps, aligned_poses = _align_txt_poses_to_full_timestamps(
        txt_poses=txt_poses,
        full_timestamps=full_timestamps,
        skip_start=1,
        skip_end=1,
    )

    selected_timestamps, selected_poses = _select_masked_txt_poses(
        mask_timestamps=np.asarray([10, 20, 40, 50, 60], dtype=np.int64),
        full_timestamps=full_timestamps,
        aligned_timestamps=aligned_timestamps,
        aligned_poses=aligned_poses,
        skip_start=1,
        skip_end=1,
    )

    np.testing.assert_array_equal(selected_timestamps, np.asarray([20, 40, 50], dtype=np.int64))
    np.testing.assert_allclose(selected_poses[:, 9], np.asarray([1, 4, 5], dtype=np.float32))


def test_oxford_txt_dataset_uses_masked_timestamps_and_txt_gt(tmp_path):
    sequence_name = "2019-01-14-12-05-52-radar-oxford-10k"
    sequence_short = _oxford_sequence_short_name(sequence_name)
    sequence_dir = tmp_path / sequence_name
    scan_dir = sequence_dir / "velodyne_left"
    pose_dir = tmp_path / "poses"
    scan_dir.mkdir(parents=True)
    pose_dir.mkdir()

    full_timestamps = np.asarray([10, 20, 30, 40, 50, 60], dtype=np.int64)
    mask_timestamps = np.asarray([10, 20, 40, 50, 60], dtype=np.int64)
    for timestamp in mask_timestamps:
        _write_scan(scan_dir / "{}.bin".format(int(timestamp)))

    _write_h5(sequence_dir / "velodyne_left_calibrateFalse.h5", full_timestamps)
    _write_h5(sequence_dir / "velodyne_left_calibrateFalse_SCR300m.h5", mask_timestamps)

    txt_poses = np.asarray(
        [
            [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0],
            [1, 0, 0, 2, 0, 1, 0, 0, 0, 0, 1, 0],
            [1, 0, 0, 4, 0, 1, 0, 0, 0, 0, 1, 0],
            [1, 0, 0, 5, 0, 1, 0, 0, 0, 0, 1, 0],
        ],
        dtype=np.float32,
    )
    np.savetxt(pose_dir / "gicp_{}.txt".format(sequence_short), txt_poses, fmt="%.6f")

    dataset = OxfordQEDataset(
        root_dir=str(tmp_path),
        sequence_list=[sequence_name],
        h5_name="velodyne_left_calibrateFalse_SCR300m.h5",
        pose_source="txt",
        full_h5_name="velodyne_left_calibrateFalse.h5",
        pose_root=str(pose_dir),
        pose_txt_template="gicp_{sequence_short}.txt",
        pose_skip_start=1,
        pose_skip_end=1,
        frame_gap=1,
        trim_edges=0,
        is_training=0,
    )

    sequence_meta = dataset.sequence_meta[sequence_name]
    np.testing.assert_array_equal(sequence_meta["timestamps"], np.asarray([20, 40, 50], dtype=np.int64))
    np.testing.assert_allclose(sequence_meta["poses"][:, 9], np.asarray([1, 4, 5], dtype=np.float32))
    assert [len(segment["timestamps"]) for segment in sequence_meta["segments"]] == [1, 2]
    assert len(dataset) == 1

    _, _, _, t_gt, _, _, _ = dataset[0]
    np.testing.assert_allclose(t_gt[:3, 3], np.asarray([-1.0, 0.0, 0.0], dtype=np.float32))


def test_oxford_training_dataset_uses_light_augmentation_helper(tmp_path, monkeypatch):
    sequence_name = "2019-01-14-12-05-52-radar-oxford-10k"
    sequence_short = _oxford_sequence_short_name(sequence_name)
    sequence_dir = tmp_path / sequence_name
    scan_dir = sequence_dir / "velodyne_left"
    pose_dir = tmp_path / "poses"
    scan_dir.mkdir(parents=True)
    pose_dir.mkdir()

    full_timestamps = np.asarray([10, 20, 30, 40, 50, 60], dtype=np.int64)
    mask_timestamps = np.asarray([10, 20, 40, 50, 60], dtype=np.int64)
    for timestamp in mask_timestamps:
        _write_scan(scan_dir / "{}.bin".format(int(timestamp)))

    _write_h5(sequence_dir / "velodyne_left_calibrateFalse.h5", full_timestamps)
    _write_h5(sequence_dir / "velodyne_left_calibrateFalse_SCR300m.h5", mask_timestamps)

    txt_poses = np.asarray(
        [
            [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0],
            [1, 0, 0, 2, 0, 1, 0, 0, 0, 0, 1, 0],
            [1, 0, 0, 4, 0, 1, 0, 0, 0, 0, 1, 0],
            [1, 0, 0, 5, 0, 1, 0, 0, 0, 0, 1, 0],
        ],
        dtype=np.float32,
    )
    np.savetxt(pose_dir / "gicp_{}.txt".format(sequence_short), txt_poses, fmt="%.6f")

    expected_transform = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.11],
            [0.0, 1.0, 0.0, 0.02],
            [0.0, 0.0, 1.0, 0.01],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    monkeypatch.setattr(dataset_factory, "aug_matrix_oxford_light", lambda: expected_transform.copy())
    monkeypatch.setattr(
        dataset_factory,
        "aug_matrix",
        lambda: (_ for _ in ()).throw(AssertionError("Oxford dataset should not call KITTI aug_matrix")),
    )

    dataset = OxfordQEDataset(
        root_dir=str(tmp_path),
        sequence_list=[sequence_name],
        h5_name="velodyne_left_calibrateFalse_SCR300m.h5",
        pose_source="txt",
        full_h5_name="velodyne_left_calibrateFalse.h5",
        pose_root=str(pose_dir),
        pose_txt_template="gicp_{sequence_short}.txt",
        pose_skip_start=1,
        pose_skip_end=1,
        frame_gap=1,
        trim_edges=0,
        is_training=1,
    )

    _, _, _, _, t_trans, t_trans_inv, _ = dataset[0]
    np.testing.assert_allclose(t_trans, expected_transform)
    np.testing.assert_allclose(t_trans_inv, np.linalg.inv(expected_transform).astype(np.float32))
