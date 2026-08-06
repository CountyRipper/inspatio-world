import numpy as np

from phase1_lsm.trajectory import (
    A_KEYFRAMES,
    APRIME_KEYFRAMES,
    fixed_controls,
    read_trajectory,
    validate_target_c2w,
    write_trajectory,
)


def _c2w_from_yaw(yaw_degree):
    angle = np.deg2rad(yaw_degree)
    result = np.repeat(np.eye(4, dtype=np.float32)[None], len(angle), axis=0)
    result[:, 0, 0] = np.cos(angle)
    result[:, 0, 2] = np.sin(angle)
    result[:, 2, 0] = -np.sin(angle)
    result[:, 2, 2] = np.cos(angle)
    result[:, :3, 3] = np.array([1.25, -2.0, 0.5], dtype=np.float32)
    return result


def test_fixed_trajectory_and_actual_pose_contract(tmp_path):
    for name, sign in (("P", 1), ("N", -1)):
        path = write_trajectory(tmp_path / f"{name}.txt", sign)
        controls = read_trajectory(path)
        assert controls.shape == (3, 240)
        assert np.array_equal(controls[0], np.zeros(240))
        assert np.array_equal(controls[2], np.zeros(240))
        np.testing.assert_array_equal(controls[1, 57:69], sign * np.ones(12) * 45)
        np.testing.assert_array_equal(controls[1, 225:240], sign * np.ones(15) * 45)
        np.testing.assert_array_equal(controls[1, A_KEYFRAMES], controls[1, APRIME_KEYFRAMES])
        metrics = validate_target_c2w(_c2w_from_yaw(controls[1]))
        assert metrics["max_camera_center_drift"] == 0.0
        assert metrics["max_rotation_speed_degree_per_frame"] <= 0.8 + 1e-6


def test_fixed_controls_do_not_need_resampling():
    for sign in (-1, 1):
        pitch, yaw, radius = fixed_controls(sign)
        assert len(pitch) == len(yaw) == len(radius) == 240
        assert np.max(np.abs(np.diff(yaw))) <= 0.8
