import math

import numpy as np

from tools import points_process


def _matrix_to_euler_deg(rotation):
    r11 = rotation[0, 0]
    r12 = rotation[0, 1]
    r13 = rotation[0, 2]
    r23 = rotation[1, 2]
    r33 = rotation[2, 2]

    cy = math.sqrt(r33 * r33 + r23 * r23)
    z = math.degrees(math.atan2(-r12, r11))
    y = math.degrees(math.atan2(r13, cy))
    x = math.degrees(math.atan2(-r23, r33))
    return z, y, x


def test_aug_matrix_preserves_existing_default_scales(monkeypatch):
    monkeypatch.setattr(points_process.np.random, "randn", lambda: 1.0)
    monkeypatch.setattr(
        points_process.np.random,
        "uniform",
        lambda low, high, size: np.ones(size, dtype=np.float32),
    )

    transform = points_process.aug_matrix()
    z_deg, y_deg, x_deg = _matrix_to_euler_deg(transform[:3, :3])

    np.testing.assert_allclose([z_deg, y_deg, x_deg], [2.25, 0.45, 0.45], atol=1e-4)
    np.testing.assert_allclose(transform[:3, 3], [0.5, 0.1, 0.05], atol=1e-6)


def test_aug_matrix_oxford_light_uses_reduced_rotation_and_translation(monkeypatch):
    monkeypatch.setattr(points_process.np.random, "randn", lambda: 1.0)
    monkeypatch.setattr(
        points_process.np.random,
        "uniform",
        lambda low, high, size: np.ones(size, dtype=np.float32),
    )

    transform = points_process.aug_matrix_oxford_light()
    z_deg, y_deg, x_deg = _matrix_to_euler_deg(transform[:3, :3])

    np.testing.assert_allclose([z_deg, y_deg, x_deg], [0.15, 0.05, 0.05], atol=1e-4)
    np.testing.assert_allclose(transform[:3, 3], [0.075, 0.015, 0.0075], atol=1e-6)
