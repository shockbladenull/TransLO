import torch

import translo_model_utils


def test_duplicate_range_image_rows_repeats_each_row_in_order():
    range_image = torch.tensor(
        [[[[1.0]], [[2.0]], [[3.0]]]],
        dtype=torch.float32,
    )

    duplicated = translo_model_utils.duplicate_range_image_rows(range_image)

    assert duplicated.shape == (1, 6, 1, 1)
    assert duplicated[:, :, 0, 0].tolist() == [[1.0, 1.0, 2.0, 2.0, 3.0, 3.0]]


def test_project_oxford_32_to_64_duplicates_projection_and_mask_rows(monkeypatch):
    calls = []

    def fake_project(pc, Feature=None, H_input=64, W_input=1800, vertical_view_down=-24.8, vertical_view_up=2.0):
        calls.append(
            {
                "H_input": H_input,
                "W_input": W_input,
                "vertical_view_down": vertical_view_down,
                "vertical_view_up": vertical_view_up,
            }
        )
        projection = torch.arange(32, dtype=torch.float32).view(1, 32, 1, 1)
        mask = (100 + torch.arange(32, dtype=torch.float32)).view(1, 32, 1, 1)
        return projection, mask

    monkeypatch.setattr(translo_model_utils, "ProjectPCimg2SphericalRing", fake_project)

    projection, mask = translo_model_utils.ProjectOxford32To64SphericalRing(
        [torch.zeros((1, 3), dtype=torch.float32)],
        W_input=1792,
        vertical_view_down=-30.67,
        vertical_view_up=10.67,
    )

    assert calls == [
        {
            "H_input": 32,
            "W_input": 1792,
            "vertical_view_down": -30.67,
            "vertical_view_up": 10.67,
        }
    ]
    assert projection.shape == (1, 64, 1, 1)
    assert mask.shape == (1, 64, 1, 1)
    assert projection[0, :6, 0, 0].tolist() == [0.0, 0.0, 1.0, 1.0, 2.0, 2.0]
    assert mask[0, :6, 0, 0].tolist() == [100.0, 100.0, 101.0, 101.0, 102.0, 102.0]
