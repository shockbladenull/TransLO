import os

import numpy as np
import torch
import torch.utils.data as data

from kitti_pytorch import points_dataset
from tools.points_process import aug_matrix

try:
    import h5py
except ImportError:
    h5py = None


def _oxford_sequence_short_name(sequence_name):
    parts = sequence_name.split("-")
    if len(parts) >= 6 and all(part.isdigit() for part in parts[:6]):
        return "{}{}".format(parts[4], parts[5])
    return sequence_name


def _resolve_oxford_sequence_file(sequence_name, seq_dir, filename, root_override=None):
    if root_override is None:
        return os.path.join(seq_dir, filename)

    sequence_short = _oxford_sequence_short_name(sequence_name)
    candidates = [
        os.path.join(root_override, sequence_name, filename),
        os.path.join(root_override, sequence_short, filename),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate

    return candidates[0]


def _txt_pose_row_to_qe_pose(row):
    row = np.asarray(row, dtype=np.float32).reshape(-1)
    if row.size != 12:
        raise ValueError("Expected Oxford TXT pose row with 12 values, got shape {}".format(row.shape))

    return np.array(
        [
            row[0],
            row[1],
            row[2],
            row[4],
            row[5],
            row[6],
            row[8],
            row[9],
            row[10],
            row[3],
            row[7],
            row[11],
        ],
        dtype=np.float32,
    )


def _align_txt_poses_to_full_timestamps(txt_poses, full_timestamps, skip_start, skip_end):
    txt_poses = np.asarray(txt_poses, dtype=np.float32)
    if txt_poses.ndim == 1:
        txt_poses = txt_poses.reshape(1, -1)
    if txt_poses.shape[1] != 12:
        raise ValueError("Expected Oxford TXT poses with shape [N, 12], got {}".format(txt_poses.shape))

    skip_start = max(int(skip_start), 0)
    skip_end = max(int(skip_end), 0)
    stop = len(full_timestamps) - skip_end if skip_end > 0 else len(full_timestamps)
    if stop <= skip_start:
        raise ValueError("Invalid Oxford TXT skip range: start={}, end={}".format(skip_start, skip_end))

    aligned_timestamps = np.asarray(full_timestamps[skip_start:stop], dtype=np.int64)
    if len(aligned_timestamps) != len(txt_poses):
        raise ValueError(
            "Oxford TXT/full-H5 length mismatch: {} poses vs {} timestamps after skipping {} front and {} back".format(
                len(txt_poses),
                len(aligned_timestamps),
                skip_start,
                skip_end,
            )
        )

    aligned_poses = np.stack([_txt_pose_row_to_qe_pose(row) for row in txt_poses], axis=0).astype(np.float32)
    return aligned_timestamps, aligned_poses


def _select_masked_txt_poses(mask_timestamps, full_timestamps, aligned_timestamps, aligned_poses, skip_start, skip_end):
    aligned_lookup = {
        int(timestamp): aligned_poses[idx]
        for idx, timestamp in enumerate(np.asarray(aligned_timestamps, dtype=np.int64))
    }

    trimmed_out = set(int(timestamp) for timestamp in np.asarray(full_timestamps[: max(skip_start, 0)], dtype=np.int64))
    if skip_end > 0:
        trimmed_out.update(int(timestamp) for timestamp in np.asarray(full_timestamps[-skip_end:], dtype=np.int64))

    selected_timestamps = []
    selected_poses = []
    missing_timestamps = []
    for timestamp in np.asarray(mask_timestamps, dtype=np.int64):
        timestamp = int(timestamp)
        pose = aligned_lookup.get(timestamp)
        if pose is None:
            missing_timestamps.append(timestamp)
            continue
        selected_timestamps.append(timestamp)
        selected_poses.append(pose)

    unexpected_missing = [timestamp for timestamp in missing_timestamps if timestamp not in trimmed_out]
    if unexpected_missing:
        raise ValueError(
            "Oxford TXT is missing {} masked timestamps outside the configured front/back skip window".format(
                len(unexpected_missing)
            )
        )

    if not selected_timestamps:
        raise ValueError("Oxford TXT alignment removed every masked timestamp")

    return np.asarray(selected_timestamps, dtype=np.int64), np.asarray(selected_poses, dtype=np.float32)


class OxfordQEDataset(data.Dataset):
    def __init__(
        self,
        root_dir,
        sequence_list,
        h5_name="velodyne_left_calibrateFalse.h5",
        h5_root=None,
        pose_source="h5",
        full_h5_name="velodyne_left_calibrateFalse.h5",
        full_h5_root=None,
        pose_root=None,
        pose_txt_template="Oxford_SLAM_result_{sequence_short}/gicp_Oxford{sequence_short}_050_v1.txt",
        pose_skip_start=5,
        pose_skip_end=5,
        frame_gap=1,
        trim_edges=5,
        is_training=1,
    ):
        if root_dir is None:
            raise ValueError("oxford_root must be set when using the oxford_qe dataset")
        if frame_gap < 1:
            raise ValueError("frame_gap must be >= 1")
        if h5py is None:
            raise ImportError("h5py is required for the oxford_qe dataset. Please install it from requirements.txt")
        if pose_source not in ("h5", "txt"):
            raise ValueError("Unsupported Oxford pose source: {}".format(pose_source))

        self.root_dir = root_dir
        self.sequence_list = sorted(sequence_list)
        self.h5_name = h5_name
        self.h5_root = h5_root
        self.pose_source = pose_source
        self.full_h5_name = full_h5_name
        self.full_h5_root = full_h5_root
        self.pose_root = pose_root if pose_root is not None else root_dir
        self.pose_txt_template = pose_txt_template
        self.pose_skip_start = max(int(pose_skip_start), 0)
        self.pose_skip_end = max(int(pose_skip_end), 0)
        self.frame_gap = frame_gap
        self.trim_edges = max(trim_edges, 0)
        self.is_training = is_training
        self.identity_transform = np.eye(4, dtype=np.float32)
        self.sequence_meta = {}
        self.samples = []

        for sequence_name in self.sequence_list:
            seq_dir = os.path.join(self.root_dir, sequence_name)
            scan_dir = os.path.join(seq_dir, "velodyne_left")
            if not os.path.isdir(scan_dir):
                raise FileNotFoundError("Oxford scan directory not found: {}".format(scan_dir))

            if self.pose_source == "h5":
                timestamps, poses = self._load_h5_sequence(seq_dir)
            else:
                timestamps, poses = self._load_txt_sequence(seq_dir, sequence_name)

            if self.trim_edges > 0:
                timestamps = timestamps[self.trim_edges:-self.trim_edges]
                poses = poses[self.trim_edges:-self.trim_edges]

            if len(timestamps) != len(poses):
                raise ValueError("Oxford timestamps/poses length mismatch in sequence {}".format(sequence_name))
            if len(timestamps) <= self.frame_gap:
                raise ValueError("Sequence {} is too short after trimming".format(sequence_name))

            self.sequence_meta[sequence_name] = {
                "scan_dir": scan_dir,
                "timestamps": timestamps.astype(np.int64),
                "poses": poses.astype(np.float32),
            }
            for curr_idx in range(self.frame_gap, len(timestamps)):
                self.samples.append((sequence_name, curr_idx - self.frame_gap, curr_idx))

    def _load_h5_sequence(self, seq_dir):
        sequence_name = os.path.basename(seq_dir)
        h5_path = _resolve_oxford_sequence_file(
            sequence_name=sequence_name,
            seq_dir=seq_dir,
            filename=self.h5_name,
            root_override=self.h5_root,
        )
        if not os.path.isfile(h5_path):
            raise FileNotFoundError("Oxford pose file not found: {}".format(h5_path))

        with h5py.File(h5_path, "r") as h5_file:
            timestamps = np.asarray(h5_file["valid_timestamps"])
            poses = np.asarray(h5_file["poses"])
        return timestamps, poses

    def _load_txt_sequence(self, seq_dir, sequence_name):
        mask_h5_path = _resolve_oxford_sequence_file(
            sequence_name=sequence_name,
            seq_dir=seq_dir,
            filename=self.h5_name,
            root_override=self.h5_root,
        )
        if not os.path.isfile(mask_h5_path):
            raise FileNotFoundError("Oxford mask H5 not found: {}".format(mask_h5_path))

        full_h5_path = _resolve_oxford_sequence_file(
            sequence_name=sequence_name,
            seq_dir=seq_dir,
            filename=self.full_h5_name,
            root_override=self.full_h5_root,
        )
        if not os.path.isfile(full_h5_path):
            raise FileNotFoundError("Oxford full H5 not found: {}".format(full_h5_path))

        txt_path = self._resolve_pose_txt_path(sequence_name)
        if not os.path.isfile(txt_path):
            raise FileNotFoundError("Oxford TXT pose file not found: {}".format(txt_path))

        with h5py.File(mask_h5_path, "r") as h5_file:
            mask_timestamps = np.asarray(h5_file["valid_timestamps"], dtype=np.int64)
        with h5py.File(full_h5_path, "r") as h5_file:
            full_timestamps = np.asarray(h5_file["valid_timestamps"], dtype=np.int64)

        txt_poses = np.loadtxt(txt_path, dtype=np.float32)
        aligned_timestamps, aligned_poses = _align_txt_poses_to_full_timestamps(
            txt_poses=txt_poses,
            full_timestamps=full_timestamps,
            skip_start=self.pose_skip_start,
            skip_end=self.pose_skip_end,
        )
        return _select_masked_txt_poses(
            mask_timestamps=mask_timestamps,
            full_timestamps=full_timestamps,
            aligned_timestamps=aligned_timestamps,
            aligned_poses=aligned_poses,
            skip_start=self.pose_skip_start,
            skip_end=self.pose_skip_end,
        )

    def _resolve_pose_txt_path(self, sequence_name):
        sequence_short = _oxford_sequence_short_name(sequence_name)
        relative_path = self.pose_txt_template.format(
            sequence=sequence_name,
            sequence_short=sequence_short,
        )
        return os.path.join(self.pose_root, relative_path)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sequence_name, prev_idx, curr_idx = self.samples[index]
        sequence_meta = self.sequence_meta[sequence_name]

        prev_timestamp = int(sequence_meta["timestamps"][prev_idx])
        curr_timestamp = int(sequence_meta["timestamps"][curr_idx])
        prev_scan_path = os.path.join(sequence_meta["scan_dir"], "{}.bin".format(prev_timestamp))
        curr_scan_path = os.path.join(sequence_meta["scan_dir"], "{}.bin".format(curr_timestamp))

        point1 = self._load_oxford_scan(prev_scan_path)
        point2 = self._load_oxford_scan(curr_scan_path)
        T_prev = self._qe_pose_to_matrix(sequence_meta["poses"][prev_idx])
        T_curr = self._qe_pose_to_matrix(sequence_meta["poses"][curr_idx])
        # TransLO predicts the transform from the current frame (pos2 / frame1) to the previous frame (pos1 / frame2).
        T_gt = np.matmul(np.linalg.inv(T_curr), T_prev).astype(np.float32)

        if self.is_training:
            T_trans = aug_matrix()
        else:
            T_trans = self.identity_transform.copy()
        T_trans_inv = np.linalg.inv(T_trans).astype(np.float32)

        return (
            torch.from_numpy(point2).float(),
            torch.from_numpy(point1).float(),
            index,
            T_gt,
            T_trans.astype(np.float32),
            T_trans_inv,
            self.identity_transform.copy(),
        )

    @staticmethod
    def _load_oxford_scan(scan_path):
        scan = np.fromfile(scan_path, dtype=np.float32)
        if scan.size == 0 or scan.size % 4 != 0:
            raise ValueError("Invalid Oxford scan file: {}".format(scan_path))
        scan = scan.reshape(4, -1).transpose()[:, :3]
        scan[:, 2] *= -1.0
        return np.ascontiguousarray(scan.astype(np.float32))

    @staticmethod
    def _qe_pose_to_matrix(pose_vector):
        pose_vector = np.asarray(pose_vector, dtype=np.float32).reshape(-1)
        if pose_vector.size != 12:
            raise ValueError("Expected QEOxford pose in R(9)+t(3) format, got shape {}".format(pose_vector.shape))
        transform = np.eye(4, dtype=np.float32)
        transform[:3, :3] = pose_vector[:9].reshape(3, 3)
        transform[:3, 3] = pose_vector[9:]
        return transform


def build_dataset(split, config, is_training):
    dataset_type = getattr(config, "{}_dataset_type".format(split))
    if dataset_type == "kitti":
        if split == "train":
            sequence_list = config.kitti_train_seqs
        elif split == "val":
            sequence_list = config.kitti_val_seqs
        elif split == "test":
            sequence_list = config.kitti_test_seqs
        else:
            raise ValueError("Unsupported split: {}".format(split))

        return points_dataset(
            is_training=is_training,
            num_point=config.num_points,
            data_dir_list=list(sequence_list),
            config=config,
        )

    if dataset_type == "oxford_qe":
        if split == "train":
            sequence_list = config.oxford_train_seqs
        elif split in ("val", "test"):
            sequence_list = config.oxford_val_seqs
        else:
            raise ValueError("Unsupported split: {}".format(split))

        return OxfordQEDataset(
            root_dir=config.oxford_root,
            sequence_list=sequence_list,
            h5_name=config.oxford_h5_name,
            h5_root=config.oxford_h5_root,
            pose_source=config.oxford_pose_source,
            full_h5_name=config.oxford_full_h5_name,
            full_h5_root=config.oxford_full_h5_root,
            pose_root=config.oxford_pose_root,
            pose_txt_template=config.oxford_pose_txt_template,
            pose_skip_start=config.oxford_pose_skip_start,
            pose_skip_end=config.oxford_pose_skip_end,
            frame_gap=config.frame_gap,
            trim_edges=config.oxford_trim_edges,
            is_training=is_training,
        )

    raise ValueError("Unsupported dataset type: {}".format(dataset_type))
