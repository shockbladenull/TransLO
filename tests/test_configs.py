from configs import translonet_args


def test_oxford_auto_profile_uses_kitti_like_projection_width():
    args = translonet_args(
        [
            "--train_dataset_type",
            "oxford_qe",
            "--val_dataset_type",
            "oxford_qe",
        ]
    )

    assert args.sensor_profile == "oxford_hdl32"
    assert args.H_input == 64
    assert args.W_input == 1792
    assert args.vertical_view_up == 10.67
    assert args.vertical_view_down == -30.67
