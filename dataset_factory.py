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


class OxfordQEDataset(data.Dataset):
    def __init__(
        self,
        root_dir,
        sequence_list,
        h5_name="velodyne_left_calibrateFalse.h5",
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

        self.root_dir = root_dir
        self.sequence_list = sorted(sequence_list)
        self.h5_name = h5_name
        self.frame_gap = frame_gap
        self.trim_edges = max(trim_edges, 0)
        self.is_training = is_training
        self.identity_transform = np.eye(4, dtype=np.float32)
        self.sequence_meta = {}
        self.samples = []

        for sequence_name in self.sequence_list:
            seq_dir = os.path.join(self.root_dir, sequence_name)
            scan_dir = os.path.join(seq_dir, "velodyne_left")
            h5_path = os.path.join(seq_dir, self.h5_name)
            if not os.path.isdir(scan_dir):
                raise FileNotFoundError("Oxford scan directory not found: {}".format(scan_dir))
            if not os.path.isfile(h5_path):
                raise FileNotFoundError("Oxford pose file not found: {}".format(h5_path))

            with h5py.File(h5_path, "r") as h5_file:
                timestamps = np.asarray(h5_file["valid_timestamps"])
                poses = np.asarray(h5_file["poses"])

            if self.trim_edges > 0:
                timestamps = timestamps[self.trim_edges:-self.trim_edges]
                poses = poses[self.trim_edges:-self.trim_edges]

            if len(timestamps) != len(poses):
                raise ValueError("Oxford timestamps/poses length mismatch in {}".format(h5_path))
            if len(timestamps) <= self.frame_gap:
                raise ValueError("Sequence {} is too short after trimming".format(sequence_name))

            self.sequence_meta[sequence_name] = {
                "scan_dir": scan_dir,
                "timestamps": timestamps.astype(np.int64),
                "poses": poses.astype(np.float32),
            }
            for curr_idx in range(self.frame_gap, len(timestamps)):
                self.samples.append((sequence_name, curr_idx - self.frame_gap, curr_idx))

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
            frame_gap=config.frame_gap,
            trim_edges=config.oxford_trim_edges,
            is_training=is_training,
        )

    raise ValueError("Unsupported dataset type: {}".format(dataset_type))
