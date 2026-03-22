# -*- coding:UTF-8 -*-

import argparse


"""
  config
"""

DEFAULT_KITTI_TRAIN_SEQS = [0, 1, 2, 3, 4, 5, 6]
DEFAULT_KITTI_VAL_SEQS = [7, 8, 9, 10]
DEFAULT_KITTI_TEST_SEQS = [7, 8, 9, 10]

DEFAULT_OXFORD_TRAIN_SEQS = [
    "2019-01-11-14-02-26-radar-oxford-10k",
]
DEFAULT_OXFORD_VAL_SEQS = [
    "2019-01-14-12-05-52-radar-oxford-10k",
]

SENSOR_PROFILES = {
    "kitti_hdl64": {
        "H_input": 64,
        "W_input": 1792,
        "vertical_view_up": 2.0,
        "vertical_view_down": -24.8,
    },
    # Keep a 64-row projection grid for compatibility with the current backbone.
    "oxford_hdl32": {
        "H_input": 64,
        "W_input": 1024,
        "vertical_view_up": 10.67,
        "vertical_view_down": -30.67,
    },
}


def _normalize_list_arg(value, cast):
    if value is None:
        return None

    if isinstance(value, (list, tuple)):
        raw_items = []
        for item in value:
            if isinstance(item, str):
                raw_items.extend([part for part in item.split(",") if part])
            else:
                raw_items.append(item)
    elif isinstance(value, str):
        raw_items = [part for part in value.split(",") if part]
    else:
        raw_items = [value]

    return [cast(item) for item in raw_items]


def _resolve_sensor_profile(args):
    if args.sensor_profile == "auto":
        sensor_profile = "oxford_hdl32" if args.train_dataset_type == "oxford_qe" else "kitti_hdl64"
    else:
        sensor_profile = args.sensor_profile

    profile = SENSOR_PROFILES[sensor_profile]
    for field, default_value in profile.items():
        if getattr(args, field) is None:
            setattr(args, field, default_value)

    args.sensor_profile = sensor_profile
    return args


def add_translonet_args(parser):
    parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 3]')
    parser.add_argument('--multi_gpu', type=str, default=None, help='The gpu [default : null]')
    parser.add_argument('--limit_or_filter', type=bool, default=True, help='if False, filter will reserve 40m~50m points')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch Size during training [default: 16]')
    parser.add_argument('--eval_batch_size', type=int, default=8, help='Batch Size during evaling [default: 64]')
    parser.add_argument('--eval_before', type=int, default=0, help='if 1, eval before train')
    parser.add_argument('--save_eval_interval', type=int, default=2,
                        help='Save checkpoints and run validation every N epochs')

    parser.add_argument('--train_dataset_type', choices=['kitti', 'oxford_qe'], default='kitti',
                        help='Dataset used for training')
    parser.add_argument('--val_dataset_type', choices=['kitti', 'oxford_qe'], default='kitti',
                        help='Dataset used for validation')
    parser.add_argument('--test_dataset_type', choices=['kitti', 'oxford_qe'], default='kitti',
                        help='Dataset kept for explicit testing')

    parser.add_argument('--lidar_root', default='/dataset/data_odometry_velodyne/dataset', help='KITTI dataset directory [default: /dataset]')
    parser.add_argument('--image_root', default='/dataset/data_odometry_color', help='Dataset directory [default: /dataset]')
    parser.add_argument('--oxford_root', default=None, help='QEOxford dataset root')
    parser.add_argument('--oxford_h5_name', default='velodyne_left_calibrateFalse.h5',
                        help='QEOxford H5 file used for timestamps and, when pose_source=h5, poses')
    parser.add_argument('--oxford_h5_root', default=None,
                        help='Optional root directory containing oxford_h5_name sequence folders')
    parser.add_argument('--oxford_pose_source', choices=['h5', 'txt'], default='h5',
                        help='Read Oxford poses from the selected H5 or an external TXT trajectory')
    parser.add_argument('--oxford_full_h5_name', default='velodyne_left_calibrateFalse.h5',
                        help='Full-sequence Oxford H5 used to align TXT poses onto the LiDAR timeline')
    parser.add_argument('--oxford_full_h5_root', default=None,
                        help='Optional root directory containing oxford_full_h5_name sequence folders')
    parser.add_argument('--oxford_pose_root', default=None,
                        help='Root directory containing Oxford TXT pose files; defaults to oxford_root')
    parser.add_argument('--oxford_pose_txt_template',
                        default='Oxford_SLAM_result_{sequence_short}/gicp_Oxford{sequence_short}_050_v1.txt',
                        help='TXT pose path template relative to oxford_pose_root')
    parser.add_argument('--oxford_pose_skip_start', type=int, default=5,
                        help='Skip N full-H5 timestamps from the front before aligning TXT poses')
    parser.add_argument('--oxford_pose_skip_end', type=int, default=5,
                        help='Skip N full-H5 timestamps from the back before aligning TXT poses')
    parser.add_argument('--oxford_train_seqs', nargs='+', default=list(DEFAULT_OXFORD_TRAIN_SEQS),
                        help='QEOxford training sequences')
    parser.add_argument('--oxford_val_seqs', nargs='+', default=list(DEFAULT_OXFORD_VAL_SEQS),
                        help='QEOxford validation sequences')
    parser.add_argument('--oxford_trim_edges', type=int, default=None,
                        help='Additional trim applied after timestamps and poses are aligned')
    parser.add_argument('--oxford_detailed_val', action='store_true',
                        help='Run periodic Oxford route validation plots in addition to fast pair-wise validation')
    parser.add_argument('--oxford_detailed_val_interval', type=int, default=5,
                        help='Run Oxford detailed route validation every N epochs')
    parser.add_argument('--frame_gap', type=int, default=1,
                        help='Temporal gap used to form relative pose pairs')
    parser.add_argument('--kitti_train_seqs', nargs='+', type=int, default=list(DEFAULT_KITTI_TRAIN_SEQS),
                        help='KITTI training sequences')
    parser.add_argument('--kitti_val_seqs', nargs='+', type=int, default=list(DEFAULT_KITTI_VAL_SEQS),
                        help='KITTI validation sequences')
    parser.add_argument('--kitti_test_seqs', nargs='+', type=int, default=list(DEFAULT_KITTI_TEST_SEQS),
                        help='KITTI test sequences')
    parser.add_argument('--log_dir', default='log_train', help='Log dir [default: log_train]')

    parser.add_argument('--num_points', type=int, default=150000, help='Point Number [default: 2048]')

    parser.add_argument('--sensor_profile', choices=['auto', 'kitti_hdl64', 'oxford_hdl32'], default='auto',
                        help='Projection profile for spherical range images')
    parser.add_argument('--H_input', type=int, default=None, help='Projection height')
    parser.add_argument('--W_input', type=int, default=None, help='Projection width')
    parser.add_argument('--vertical_view_up', type=float, default=None, help='Upper vertical FoV in degrees')
    parser.add_argument('--vertical_view_down', type=float, default=None, help='Lower vertical FoV in degrees')

    parser.add_argument('--max_epoch', type=int, default=301, help='Epoch to run [default: 151]')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='The Weight decay [default : 0.0001]')
    parser.add_argument('--workers', type=int, default=6,
                        help='Sets how many child processes can be used [default : 16]')
    parser.add_argument('--model_name', type=str, default='pwclonet', help='base_dir_name [default: pwclonet]')
    parser.add_argument('--task_name', type=str, default=None, help='who can replace model_name ')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')

    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--optimizer', default='Adam', help='adam or momentum [default: adam]')

    parser.add_argument('--initial_lr', type=bool, default=False, help='Initial learning rate or not [default: False]')
    parser.add_argument('--learning_rate_clip', type=float, default=1e-6, help='learning_rate_clip [default : 1e-5]')
    parser.add_argument('--lr_stepsize', type=int, default=13, help="lr_stepsize")
    parser.add_argument('--lr_gamma', type=float, default=0.7, help="lr_gamma")
    parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
    parser.add_argument('--is_training', type=bool, default=True, help='is_training [default : True]')

    ##Trnasformers
    #Encoder
    parser.add_argument('--d_embed', type=int, default=128, help='Number of dimensions to encode into')
    parser.add_argument('--scale', type=float, default=1.0, help='for pos embedding')
    parser.add_argument('--attention_type', type=str, default='dot_prod', help='attention type')
    parser.add_argument('--nhead', type=int, default=8, help='heads of Multi-Head Attention')
    parser.add_argument('--d_feedforward', type=int, default=512, help='')
    parser.add_argument('--dropout', type=float, default=0.0, help='drop out')
    parser.add_argument('--H_anchors', type=int, default=4, help='size of H patches')
    parser.add_argument('--W_anchors', type=int, default=4, help='size of W patches')
    parser.add_argument('--seq_len', type=int, default=1, help='sequence length')
    parser.add_argument('--pre_norm', type=bool, default=True, help='Normalization type')
    parser.add_argument('--transformer_act', type=str, default='relu', help='transformer activation type')
    parser.add_argument('--num_encoder_layers', type=int, default=6, help='number of encoder layers')
    parser.add_argument('--transformer_encoder_has_pos_emb', type=bool, default=True,
                        help='if transformer encoder has pos emb')
    parser.add_argument('--sa_val_has_pos_emb', type=bool, default=True, help='if f1 has pos emb')
    parser.add_argument('--ca_val_has_pos_emb', type=bool, default=True, help='if f2 has pos emb')
    #Decoder
    parser.add_argument('--corr_decoder_has_pos_emb', type=bool, default=True, help='if decoder has pos emb')

    return parser


def build_translonet_parser():
    parser = argparse.ArgumentParser()
    return add_translonet_args(parser)


def finalize_translonet_args(args):
    args.kitti_train_seqs = _normalize_list_arg(args.kitti_train_seqs, int)
    args.kitti_val_seqs = _normalize_list_arg(args.kitti_val_seqs, int)
    args.kitti_test_seqs = _normalize_list_arg(args.kitti_test_seqs, int)
    args.oxford_train_seqs = _normalize_list_arg(args.oxford_train_seqs, str)
    args.oxford_val_seqs = _normalize_list_arg(args.oxford_val_seqs, str)
    if args.oxford_trim_edges is None:
        args.oxford_trim_edges = 0 if args.oxford_pose_source == 'txt' else 5
    if args.save_eval_interval <= 0:
        raise ValueError('--save_eval_interval must be a positive integer')
    if args.oxford_detailed_val_interval <= 0:
        raise ValueError('--oxford_detailed_val_interval must be a positive integer')
    args = _resolve_sensor_profile(args)
    return args


def translonet_args(argv=None):
    parser = build_translonet_parser()
    args = parser.parse_args(argv)

    return finalize_translonet_args(args)
