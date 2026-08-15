from __future__ import annotations

import numpy as np


def as_homogeneous(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape == (4, 4):
        return matrix
    if matrix.shape == (3, 4):
        return np.vstack([matrix, np.array([[0.0, 0.0, 0.0, 1.0]])])
    raise ValueError(f"Expected a 3x4 or 4x4 pose, got {matrix.shape}")


def tcw_to_c2w(tcw: np.ndarray) -> np.ndarray:
    return np.linalg.inv(as_homogeneous(tcw))


def c2w_to_tcw(c2w: np.ndarray) -> np.ndarray:
    return np.linalg.inv(as_homogeneous(c2w))


def to_cut3r_c2w(c2w: np.ndarray) -> np.ndarray:
    """Apply VMem's CUT3R convention: negate c2w Y/Z basis columns."""
    transform = np.eye(4, dtype=np.float64)
    transform[1, 1] = -1.0
    transform[2, 2] = -1.0
    return as_homogeneous(c2w) @ transform


def from_cut3r_c2w(c2w: np.ndarray) -> np.ndarray:
    # The convention transform is its own inverse.
    return to_cut3r_c2w(c2w)


def rotation_geodesic(rotation_a: np.ndarray, rotation_b: np.ndarray) -> float:
    relative = np.asarray(rotation_a).T @ np.asarray(rotation_b)
    cosine = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.arccos(cosine))


def pose_distance(
    c2w_a: np.ndarray,
    c2w_b: np.ndarray,
    *,
    translation_weight: float = 1.0,
    rotation_weight: float = 1.0,
) -> tuple[float, float, float]:
    a = as_homogeneous(c2w_a)
    b = as_homogeneous(c2w_b)
    translation = float(np.linalg.norm(a[:3, 3] - b[:3, 3]))
    rotation = rotation_geodesic(a[:3, :3], b[:3, :3])
    combined = translation_weight * translation + rotation_weight * rotation
    return combined, translation, rotation


def rotation_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    """Return a normalized quaternion in [w, x, y, z] order."""
    r = np.asarray(rotation, dtype=np.float64)
    trace = np.trace(r)
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2
        q = np.array([0.25 * s, (r[2, 1] - r[1, 2]) / s,
                      (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s])
    else:
        i = int(np.argmax(np.diag(r)))
        if i == 0:
            s = np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2
            q = np.array([(r[2, 1] - r[1, 2]) / s, 0.25 * s,
                          (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s])
        elif i == 1:
            s = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2
            q = np.array([(r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s,
                          0.25 * s, (r[1, 2] + r[2, 1]) / s])
        else:
            s = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2
            q = np.array([(r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s,
                          (r[1, 2] + r[2, 1]) / s, 0.25 * s])
    return q / np.linalg.norm(q)


def quaternion_to_rotation(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quaternion, dtype=np.float64)
    n = np.linalg.norm([w, x, y, z])
    if n == 0:
        raise ValueError("Cannot convert a zero quaternion")
    w, x, y, z = np.array([w, x, y, z]) / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def average_camera_pose(c2ws: np.ndarray) -> np.ndarray:
    c2ws = np.asarray(c2ws, dtype=np.float64)
    if c2ws.ndim != 3 or c2ws.shape[1:] != (4, 4):
        raise ValueError(f"Expected [N,4,4] c2ws, got {c2ws.shape}")
    quaternions = np.stack([rotation_to_quaternion(pose[:3, :3]) for pose in c2ws])
    reference = quaternions[0]
    quaternions[np.sum(quaternions * reference, axis=1) < 0] *= -1
    accumulator = quaternions.T @ quaternions
    _, eigenvectors = np.linalg.eigh(accumulator)
    quaternion = eigenvectors[:, -1]
    if np.dot(quaternion, reference) < 0:
        quaternion *= -1
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = quaternion_to_rotation(quaternion)
    result[:3, 3] = c2ws[:, :3, 3].mean(axis=0)
    return result


def scale_intrinsics(
    intrinsic: np.ndarray,
    *,
    source_hw: tuple[int, int],
    target_hw: tuple[int, int],
) -> np.ndarray:
    source_h, source_w = source_hw
    target_h, target_w = target_hw
    scaled = np.asarray(intrinsic, dtype=np.float64).copy()
    scaled[0] *= target_w / source_w
    scaled[1] *= target_h / source_h
    return scaled
