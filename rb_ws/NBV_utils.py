"""Object-frame occupancy map을 위한 기본 NBV scoring utilities.

이 모듈은 Isaac Sim에 의존하지 않고 NumPy만 사용한다. 고정 카메라와
후보 object pose가 주어졌을 때 object frame에서 ray tracing을 수행하고,
현재 보일 것으로 예상되는 voxel들의 entropy를 합산한다.
"""

from __future__ import annotations

import math
import os
import time
from collections import OrderedDict

import numpy as np


OCCUPANCY_UNKNOWN = -1
OCCUPANCY_FREE = 0
OCCUPANCY_OCCUPIED = 1


class RayGeometryCache:
    """Bounded, per-sequence LRU for geometric ray paths, never map state."""

    def __init__(self, max_bytes=64 * 1024 * 1024):
        self.max_bytes = max(0, int(max_bytes))
        self.bytes_used = 0
        self.hits = 0
        self.misses = 0
        self._entries = OrderedDict()

    def get(self, key):
        value = self._entries.get(key)
        if value is None:
            self.misses += 1
        else:
            self.hits += 1
            self._entries.move_to_end(key)
        return value

    def put(self, key, arrays):
        size = sum(array.nbytes for array in arrays)
        if size > self.max_bytes:
            return
        previous = self._entries.pop(key, None)
        if previous is not None:
            self.bytes_used -= sum(array.nbytes for array in previous)
        while self._entries and self.bytes_used + size > self.max_bytes:
            _, evicted = self._entries.popitem(last=False)
            self.bytes_used -= sum(array.nbytes for array in evicted)
        self._entries[key] = arrays
        self.bytes_used += size

# Isaac/USD camera: +X right, +Y up, -Z forward
# CV camera:        +X right, +Y down, +Z forward
CAMERA_CV_TO_USD = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


def _validate_transform(matrix, name):
    matrix = np.asarray(matrix, dtype=np.float64)

    if matrix.shape != (4, 4):
        raise ValueError(
            f"{name} shape가 (4, 4)가 아닙니다: {matrix.shape}"
        )

    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name}에 NaN 또는 Inf가 있습니다.")

    if not np.allclose(
        matrix[3],
        np.array([0.0, 0.0, 0.0, 1.0]),
        rtol=0.0,
        atol=1e-7,
    ):
        raise ValueError(
            f"{name}의 마지막 행이 [0, 0, 0, 1]이 아닙니다. "
            "Gf.Matrix4d를 np.asarray(...).T로 변환했는지 확인하세요."
        )

    return matrix


def load_occupancy_map(file_path):
    """Occupancy_utils.py가 저장한 .npz를 검증하여 dict로 반환한다."""
    if not os.path.isfile(file_path):
        raise FileNotFoundError(
            f"Occupancy map 파일이 없습니다:\n{file_path}"
        )

    with np.load(file_path) as data:
        required_keys = {
            "origin",
            "root_size",
            "leaf_size",
            "grid_shape",
        }
        missing_keys = required_keys - set(data.files)

        if missing_keys:
            raise ValueError(
                "Occupancy map에 필요한 항목이 없습니다: "
                f"{sorted(missing_keys)}"
            )

        origin = np.asarray(data["origin"], dtype=np.float64)
        root_size = float(data["root_size"])
        leaf_size = float(data["leaf_size"])
        grid_shape = tuple(
            np.asarray(data["grid_shape"], dtype=np.int32).tolist()
        )

        if "probability" in data.files:
            probability = np.asarray(
                data["probability"],
                dtype=np.float64,
            )
        elif "log_odds" in data.files:
            log_odds = np.asarray(data["log_odds"], dtype=np.float64)
            probability = 1.0 / (1.0 + np.exp(-log_odds))
        else:
            raise ValueError(
                "Occupancy map에 probability 또는 log_odds가 없습니다."
            )

        if "state" in data.files:
            state = np.asarray(data["state"], dtype=np.int8)
        elif "observation_count" in data.files:
            observation_count = np.asarray(
                data["observation_count"],
                dtype=np.uint16,
            )
            state = np.full(
                grid_shape,
                OCCUPANCY_UNKNOWN,
                dtype=np.int8,
            )
            observed = observation_count > 0
            state[observed & (probability < 0.5)] = OCCUPANCY_FREE
            state[observed & (probability >= 0.5)] = OCCUPANCY_OCCUPIED
        else:
            raise ValueError(
                "Occupancy map에 state 또는 observation_count가 없습니다."
            )

    if origin.shape != (3,):
        raise ValueError(f"origin shape가 (3,)이 아닙니다: {origin.shape}")

    if root_size <= 0.0 or leaf_size <= 0.0:
        raise ValueError("root_size와 leaf_size는 0보다 커야 합니다.")

    if state.shape != grid_shape:
        raise ValueError(
            f"state shape {state.shape} != grid_shape {grid_shape}"
        )

    if probability.shape != grid_shape:
        raise ValueError(
            "probability shape가 grid_shape와 다릅니다: "
            f"{probability.shape} != {grid_shape}"
        )

    unexpected_states = np.setdiff1d(
        np.unique(state),
        np.array(
            [OCCUPANCY_UNKNOWN, OCCUPANCY_FREE, OCCUPANCY_OCCUPIED],
            dtype=np.int8,
        ),
    )

    if len(unexpected_states) > 0:
        raise ValueError(
            "지원하지 않는 occupancy state가 있습니다: "
            f"{unexpected_states.tolist()}"
        )

    return {
        "origin": origin,
        "root_size": root_size,
        "leaf_size": leaf_size,
        "grid_shape": grid_shape,
        "probability": probability,
        "state": state,
    }


def binary_entropy(probability):
    """Bernoulli occupancy entropy를 bit 단위로 계산한다."""
    probability = np.asarray(probability, dtype=np.float64)
    clipped = np.clip(probability, 1e-12, 1.0 - 1e-12)

    return -(
        clipped * np.log2(clipped)
        + (1.0 - clipped) * np.log2(1.0 - clipped)
    )


def build_unknown_voxel_visibility_bitsets(
    occupancy_map,
    candidate_results,
):
    """Build candidate bitsets for currently visible UNKNOWN voxels.

    An unobserved occupancy voxel has ``p=0.5`` and therefore exactly one
    bit of Bernoulli entropy. Consequently, the bit count of a candidate's
    mask is its unknown-voxel entropy gain, and bitwise union automatically
    removes overlap between views during route search.

    ``candidate_results`` must have been produced with
    ``return_visible_indices=True``. The returned bit indices use the same
    C-order flattening as ``occupancy_map['state'].reshape(-1)``.
    """
    if isinstance(occupancy_map, (str, os.PathLike)):
        occupancy_map = load_occupancy_map(occupancy_map)

    state = np.asarray(occupancy_map["state"], dtype=np.int8)
    state_flat = state.reshape(-1)
    voxel_count = int(state_flat.size)
    visibility_bits = {}
    entropy_bits_by_candidate = {}
    reachable_bits = 0

    for result in candidate_results:
        candidate_name = result.get("candidate_name")
        if candidate_name is None:
            raise ValueError("candidate_result에 candidate_name이 없습니다.")
        if "visible_voxel_indices" not in result:
            raise ValueError(
                f"{candidate_name}: visible_voxel_indices가 없습니다. "
                "score_candidate_poses(..., return_visible_indices=True)를 "
                "사용하세요."
            )

        indices = np.asarray(
            result["visible_voxel_indices"],
            dtype=np.int64,
        ).reshape(-1, 3)
        if len(indices) == 0:
            candidate_bits = 0
        else:
            in_bounds = np.all(
                (indices >= 0)
                & (indices < np.asarray(state.shape, dtype=np.int64)),
                axis=1,
            )
            if not np.all(in_bounds):
                raise ValueError(
                    f"{candidate_name}: occupancy grid 밖의 visible voxel "
                    "index가 있습니다."
                )
            flat_indices = np.ravel_multi_index(
                indices.T,
                state.shape,
            )
            unknown_flat = np.unique(
                flat_indices[
                    state_flat[flat_indices] == OCCUPANCY_UNKNOWN
                ]
            )
            unknown_mask = np.zeros(voxel_count, dtype=np.uint8)
            unknown_mask[unknown_flat] = 1
            packed = np.packbits(unknown_mask, bitorder="little")
            candidate_bits = int.from_bytes(
                packed.tobytes(),
                byteorder="little",
            )

        visibility_bits[candidate_name] = candidate_bits
        entropy_bits_by_candidate[candidate_name] = int(
            candidate_bits.bit_count()
        )
        reachable_bits |= candidate_bits

    return {
        "visibility_bits": visibility_bits,
        "entropy_bits_by_candidate": entropy_bits_by_candidate,
        "reachable_bits": reachable_bits,
        "reachable_entropy_bits": int(reachable_bits.bit_count()),
        "voxel_count": voxel_count,
        "grid_shape": tuple(int(value) for value in state.shape),
    }


def make_object_world_matrix(position, rotation_xyz_deg):
    """고정 위치와 XYZ Euler 각도로 candidate object matrix를 만든다.

    현재 후보들은 한 번에 X 또는 Y 한 축만 회전하므로 Isaac의
    RotateXYZOp와 동일하게 사용할 수 있다.
    """
    position = np.asarray(position, dtype=np.float64)
    rotation_xyz_deg = np.asarray(rotation_xyz_deg, dtype=np.float64)

    if position.shape != (3,):
        raise ValueError(f"position shape가 (3,)이 아닙니다: {position.shape}")

    if rotation_xyz_deg.shape != (3,):
        raise ValueError(
            "rotation_xyz_deg shape가 (3,)이 아닙니다: "
            f"{rotation_xyz_deg.shape}"
        )

    rx, ry, rz = np.deg2rad(rotation_xyz_deg)

    rotation_x = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(rx), -np.sin(rx)],
            [0.0, np.sin(rx), np.cos(rx)],
        ],
        dtype=np.float64,
    )
    rotation_y = np.array(
        [
            [np.cos(ry), 0.0, np.sin(ry)],
            [0.0, 1.0, 0.0],
            [-np.sin(ry), 0.0, np.cos(ry)],
        ],
        dtype=np.float64,
    )
    rotation_z = np.array(
        [
            [np.cos(rz), -np.sin(rz), 0.0],
            [np.sin(rz), np.cos(rz), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation_z @ rotation_y @ rotation_x
    matrix[:3, 3] = position
    return matrix


def make_roi_mask(
    occupancy_map,
    roi_min=None,
    roi_max=None,
):
    """Object-frame ROI bounding box 내부 voxel mask를 만든다."""
    origin = np.asarray(occupancy_map["origin"], dtype=np.float64)
    leaf_size = float(occupancy_map["leaf_size"])
    grid_shape = tuple(occupancy_map["grid_shape"])

    if roi_min is None and roi_max is None:
        return np.ones(grid_shape, dtype=bool)

    if roi_min is None or roi_max is None:
        raise ValueError("roi_min과 roi_max는 함께 지정해야 합니다.")

    roi_min = np.asarray(roi_min, dtype=np.float64)
    roi_max = np.asarray(roi_max, dtype=np.float64)

    if roi_min.shape != (3,) or roi_max.shape != (3,):
        raise ValueError("roi_min과 roi_max shape는 각각 (3,)이어야 합니다.")

    if np.any(roi_min >= roi_max):
        raise ValueError("모든 축에서 roi_min < roi_max여야 합니다.")

    indices = np.indices(grid_shape, dtype=np.float64)
    centers = origin[:, None, None, None] + (
        indices + 0.5
    ) * leaf_size

    return np.all(
        (centers >= roi_min[:, None, None, None])
        & (centers < roi_max[:, None, None, None]),
        axis=0,
    )


def generate_camera_rays(
    intrinsic_matrix,
    image_size=None,
    pixel_stride=16,
):
    """샘플된 image pixel의 CV-camera ray direction을 반환한다."""
    intrinsic_matrix = np.asarray(intrinsic_matrix, dtype=np.float64)

    if intrinsic_matrix.shape != (3, 3):
        raise ValueError(
            "intrinsic_matrix shape가 (3, 3)이 아닙니다: "
            f"{intrinsic_matrix.shape}"
        )

    if pixel_stride <= 0:
        raise ValueError("pixel_stride는 1 이상이어야 합니다.")

    fx = float(intrinsic_matrix[0, 0])
    fy = float(intrinsic_matrix[1, 1])
    cx = float(intrinsic_matrix[0, 2])
    cy = float(intrinsic_matrix[1, 2])

    if fx <= 0.0 or fy <= 0.0:
        raise ValueError("fx와 fy는 0보다 커야 합니다.")

    if image_size is None:
        width = int(round(2.0 * cx))
        height = int(round(2.0 * cy))
    else:
        width, height = (int(value) for value in image_size)

    if width <= 0 or height <= 0:
        raise ValueError(f"잘못된 image_size입니다: {(width, height)}")

    u_values = np.arange(0, width, pixel_stride, dtype=np.float64) + 0.5
    v_values = np.arange(0, height, pixel_stride, dtype=np.float64) + 0.5
    pixel_u, pixel_v = np.meshgrid(u_values, v_values)

    directions = np.stack(
        [
            (pixel_u.ravel() - cx) / fx,
            (pixel_v.ravel() - cy) / fy,
            np.ones(pixel_u.size, dtype=np.float64),
        ],
        axis=1,
    )
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    return directions


def ray_voxel_indices(
    sensor_origin,
    ray_direction,
    grid_origin,
    root_size,
    leaf_size,
):
    """하나의 ray가 map AABB 내부에서 지나는 voxel index를 반환한다."""
    sensor_origin = np.asarray(sensor_origin, dtype=np.float64)
    ray_direction = np.asarray(ray_direction, dtype=np.float64)
    grid_origin = np.asarray(grid_origin, dtype=np.float64)
    bounds_max = grid_origin + float(root_size)

    direction_norm = np.linalg.norm(ray_direction)

    if direction_norm < 1e-12:
        return np.empty((0, 3), dtype=np.int32)

    ray_direction = ray_direction / direction_norm
    t_enter = 0.0
    t_exit = np.inf

    for axis in range(3):
        if abs(ray_direction[axis]) < 1e-12:
            if not (
                grid_origin[axis]
                <= sensor_origin[axis]
                <= bounds_max[axis]
            ):
                return np.empty((0, 3), dtype=np.int32)
            continue

        axis_t1 = (
            grid_origin[axis] - sensor_origin[axis]
        ) / ray_direction[axis]
        axis_t2 = (
            bounds_max[axis] - sensor_origin[axis]
        ) / ray_direction[axis]

        t_enter = max(t_enter, min(axis_t1, axis_t2))
        t_exit = min(t_exit, max(axis_t1, axis_t2))

        if t_enter > t_exit:
            return np.empty((0, 3), dtype=np.int32)

    if t_exit < 0.0 or not np.isfinite(t_exit):
        return np.empty((0, 3), dtype=np.int32)

    t_enter = max(t_enter, 0.0)
    segment_length = t_exit - t_enter

    if segment_length < 0.0:
        return np.empty((0, 3), dtype=np.int32)

    sample_step = float(leaf_size) * 0.5
    sample_count = max(
        2,
        int(np.ceil(segment_length / sample_step)) + 1,
    )
    sample_t = np.linspace(t_enter, t_exit, sample_count)
    sample_points = (
        sensor_origin[None, :]
        + sample_t[:, None] * ray_direction[None, :]
    )

    epsilon = float(leaf_size) * 1e-6
    sample_points = np.clip(
        sample_points,
        grid_origin + epsilon,
        bounds_max - epsilon,
    )
    voxel_indices = np.floor(
        (sample_points - grid_origin[None, :]) / float(leaf_size)
    ).astype(np.int32)

    keep_mask = np.ones(len(voxel_indices), dtype=bool)

    if len(voxel_indices) > 1:
        keep_mask[1:] = np.any(
            voxel_indices[1:] != voxel_indices[:-1],
            axis=1,
        )

    return voxel_indices[keep_mask]


def ray_voxel_indices_batch(
    sensor_origin,
    ray_directions,
    grid_origin,
    root_size,
    leaf_size,
):
    """여러 ray가 지나는 voxel을 한 번에 구한다 (벡터화).

    ray_voxel_indices()를 ray마다 호출하면 광선당 약 87개 샘플을
    파이썬 루프로 순회하며 voxel마다 tuple을 만든다. 후보 6개 채점에
    2.6초가 걸리던 원인이다. 같은 슬랩 판정과 샘플링을 numpy로 한 번에
    처리한다.

    반환:
        flat_indices  (n_rays, n_samples) 격자 평탄 인덱스
        ray_valid     (n_rays,) AABB와 만나는 ray인지
        sample_valid  (n_rays, n_samples) 그 ray에서 실제로 쓰는 샘플인지
    """
    sensor_origin = np.asarray(sensor_origin, dtype=np.float64)
    directions = np.asarray(ray_directions, dtype=np.float64)
    grid_origin = np.asarray(grid_origin, dtype=np.float64)
    root_size = float(root_size)
    leaf_size = float(leaf_size)
    bounds_max = grid_origin + root_size

    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    safe_norms = np.where(norms < 1e-12, 1.0, norms)
    directions = directions / safe_norms

    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = (grid_origin[None, :] - sensor_origin[None, :]) / directions
        t2 = (bounds_max[None, :] - sensor_origin[None, :]) / directions

    near = np.minimum(t1, t2)
    far = np.maximum(t1, t2)
    # 축과 평행한 성분은 슬랩 판정에서 제외하되, 원점이 범위 밖이면
    # 그 ray 자체를 버린다.
    parallel = np.abs(directions) < 1e-12
    inside_slab = (
        (sensor_origin[None, :] >= grid_origin[None, :])
        & (sensor_origin[None, :] <= bounds_max[None, :])
    )
    near = np.where(parallel, -np.inf, near)
    far = np.where(parallel, np.inf, far)

    t_enter = np.maximum(np.max(near, axis=1), 0.0)
    t_exit = np.min(far, axis=1)
    ray_valid = (
        (t_enter <= t_exit)
        & np.isfinite(t_exit)
        & (t_exit >= 0.0)
        & (norms[:, 0] >= 1e-12)
        & np.all(parallel <= inside_slab, axis=1)
    )

    cells_per_axis = int(round(root_size / leaf_size))
    # 샘플 수는 ray마다 자기 구간 길이로 정한다. 전부 대각선 기준
    # 고정값을 쓰면 짧은 ray가 더 조밀하게 찍혀, 기존 구현이 건너뛰던
    # 모서리 voxel까지 잡힌다 (실측: 가시 voxel 3036 -> 3129). NBV
    # 동작이 조용히 달라지므로 기존 규칙을 그대로 재현한다.
    sample_step = leaf_size * 0.5
    segment_length = np.maximum(t_exit - t_enter, 0.0)
    per_ray_count = np.maximum(
        2,
        np.ceil(segment_length / sample_step).astype(np.int64) + 1,
    )
    sample_count = int(per_ray_count.max()) if len(per_ray_count) else 2
    steps = np.arange(sample_count)[None, :]
    sample_valid = steps < per_ray_count[:, None]
    fractions = np.minimum(
        steps / np.maximum(per_ray_count[:, None] - 1, 1),
        1.0,
    )
    sample_t = (
        t_enter[:, None]
        + (t_exit - t_enter)[:, None] * fractions
    )
    points = (
        sensor_origin[None, None, :]
        + sample_t[:, :, None] * directions[:, None, :]
    )
    epsilon = leaf_size * 1e-6
    points = np.clip(
        points,
        grid_origin[None, None, :] + epsilon,
        (bounds_max - epsilon)[None, None, :],
    )
    indices = np.floor(
        (points - grid_origin[None, None, :]) / leaf_size
    ).astype(np.int64)
    indices = np.clip(indices, 0, cells_per_axis - 1)
    flat_indices = (
        indices[:, :, 0] * cells_per_axis * cells_per_axis
        + indices[:, :, 1] * cells_per_axis
        + indices[:, :, 2]
    )

    return flat_indices, ray_valid, sample_valid


def score_directional_surfels(
    surface_model,
    object_from_camera_cv,
    intrinsic_matrix,
    image_size,
    pixel_bin_size=8,
    occlusion_tolerance=0.003,
    min_facing_cos=0.05,
    top_weight=1.0,
    bottom_weight=1.0,
    side_weight=1.0,
):
    """Count camera-visible, still-unobserved oriented surface samples.

    A coarse z-buffer removes self-occluded samples. Opposite faces of a thin
    part stay independent because front-facing is evaluated from each
    surfel's normal rather than from its occupancy voxel.
    """
    points = np.asarray(surface_model["points"], dtype=np.float64)
    normals = np.asarray(surface_model["normals"], dtype=np.float64)
    observed = np.asarray(surface_model["observed"], dtype=bool)
    top_weight = max(0.0, float(top_weight))
    bottom_weight = max(0.0, float(bottom_weight))
    side_weight = max(0.0, float(side_weight))

    if points.shape != normals.shape or points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("surface_model points/normals shape는 (N, 3)이어야 합니다.")
    if observed.shape != (len(points),):
        raise ValueError("surface_model observed shape가 surfel 수와 다릅니다.")

    intrinsic_matrix = np.asarray(intrinsic_matrix, dtype=np.float64)
    if image_size is None:
        width = int(round(2.0 * intrinsic_matrix[0, 2]))
        height = int(round(2.0 * intrinsic_matrix[1, 2]))
    else:
        width, height = (int(value) for value in image_size)

    sensor_origin = object_from_camera_cv[:3, 3]
    view = sensor_origin[None, :] - points
    view_norm = np.linalg.norm(view, axis=1)
    front_facing = (
        np.einsum("ij,ij->i", normals, view)
        > float(min_facing_cos) * view_norm
    )

    camera_cv_from_object = np.linalg.inv(object_from_camera_cv)
    points_camera = (
        points @ camera_cv_from_object[:3, :3].T
        + camera_cv_from_object[:3, 3]
    )
    depth = points_camera[:, 2]
    valid_depth = depth > 1e-6
    u = np.full(len(points), np.nan, dtype=np.float64)
    v = np.full(len(points), np.nan, dtype=np.float64)
    u[valid_depth] = (
        intrinsic_matrix[0, 2]
        + intrinsic_matrix[0, 0]
        * points_camera[valid_depth, 0]
        / depth[valid_depth]
    )
    v[valid_depth] = (
        intrinsic_matrix[1, 2]
        + intrinsic_matrix[1, 1]
        * points_camera[valid_depth, 1]
        / depth[valid_depth]
    )
    in_frame = (
        front_facing
        & valid_depth
        & (u >= 0.0)
        & (u < width)
        & (v >= 0.0)
        & (v < height)
    )
    candidate_indices = np.flatnonzero(in_frame)

    if len(candidate_indices) == 0:
        return {
            "visible_count": 0,
            "unobserved_visible_count": 0,
            "weighted_unobserved_score": 0.0,
            "unobserved_visible_groups": {
                "top(+Z)": 0,
                "bottom(-Z)": 0,
                "side": 0,
            },
            "visible_mask": np.zeros(len(points), dtype=bool),
        }

    bin_size = max(1, int(pixel_bin_size))
    bins_x = int(math.ceil(width / bin_size))
    bin_x = (u[candidate_indices] // bin_size).astype(np.int64)
    bin_y = (v[candidate_indices] // bin_size).astype(np.int64)
    bin_keys = bin_y * bins_x + bin_x
    min_depth_by_bin = np.full(
        int(math.ceil(height / bin_size)) * bins_x,
        np.inf,
        dtype=np.float64,
    )
    np.minimum.at(
        min_depth_by_bin,
        bin_keys,
        depth[candidate_indices],
    )
    visible_candidates = (
        depth[candidate_indices]
        <= min_depth_by_bin[bin_keys] + float(occlusion_tolerance)
    )
    visible_indices = candidate_indices[visible_candidates]
    visible_mask = np.zeros(len(points), dtype=bool)
    visible_mask[visible_indices] = True
    unobserved_visible = visible_mask & ~observed
    top_mask = normals[:, 2] >= 0.5
    bottom_mask = normals[:, 2] <= -0.5
    side_mask = ~(top_mask | bottom_mask)
    surfel_weights = np.full(len(points), side_weight, dtype=np.float64)
    surfel_weights[top_mask] = top_weight
    surfel_weights[bottom_mask] = bottom_weight

    return {
        "visible_count": int(np.count_nonzero(visible_mask)),
        "unobserved_visible_count": int(
            np.count_nonzero(unobserved_visible)
        ),
        "weighted_unobserved_score": float(
            surfel_weights[unobserved_visible].sum()
        ),
        "unobserved_visible_groups": {
            "top(+Z)": int(
                np.count_nonzero(unobserved_visible & top_mask)
            ),
            "bottom(-Z)": int(
                np.count_nonzero(unobserved_visible & bottom_mask)
            ),
            "side": int(
                np.count_nonzero(unobserved_visible & side_mask)
            ),
        },
        "visible_mask": visible_mask,
    }


def score_candidate_pose(
    occupancy_map,
    candidate_object_world_matrix,
    camera_world_matrix,
    intrinsic_matrix,
    image_size=None,
    pixel_stride=16,
    roi_min=None,
    roi_max=None,
    candidate_name=None,
    return_visible_indices=False,
    min_visibility_probability=None,
    surface_mask=None,
    surface_model=None,
    surface_pixel_bin_size=8,
    surface_occlusion_tolerance=0.003,
    surface_min_facing_cos=0.05,
    surface_top_weight=1.0,
    surface_bottom_weight=1.0,
    surface_side_weight=1.0,
    background_weight=1.0,
    score_unknown_only=False,
    ray_geometry_cache=None,
):
    """후보 object pose의 expected visible-entropy score를 계산한다.

    논문식 NBV에 맞춰 ROI 안에서 ray로 관측 가능한 voxel들의
    occupancy entropy를 단순 합산한다. Free와 Unknown은 통과하고,
    Occupied는 score에 포함한 뒤 해당 ray를 종료한다. Unknown을
    지났다는 이유로 뒤쪽 voxel score를 감쇠하지 않는다.

    score_unknown_only=True이면 이미 한 번 이상 관측된 free/occupied
    voxel의 잔여 posterior entropy는 제외하고, 현재 UNKNOWN voxel의
    entropy만 합산한다. UNKNOWN은 p=0.5이므로 voxel당 정확히 1 bit다.

    ray_geometry_cache는 동일 자세/카메라/격자의 광선 경로만 재사용한다.
    ROI, occupied 가림, UNKNOWN 상태와 entropy는 매번 현재 map으로
    계산한다. 캐시는 선택적으로 제공하며 시퀀스 밖으로 공유하지 않는다.

    surface_mask가 주어지면 물체 표면 voxel의 entropy를 1.0으로,
    그 밖의 voxel은 background_weight로 가중한다.

    score_unknown_only 모드에서는 roi_min/roi_max가 정보 탐색 범위를
    결정한다. ROI를 물체 주변으로 제한해야 멀리 떨어진 배경 unknown이
    점수를 지배하지 않는다. surface_mask/surface_model은 알려진 표면을
    이용하는 legacy 선택 모드에만 필요하다.

    background_weight=1.0이면 이전 동작과 완전히 같다.
    """
    if isinstance(occupancy_map, (str, os.PathLike)):
        occupancy_map = load_occupancy_map(occupancy_map)

    candidate_object_world_matrix = _validate_transform(
        candidate_object_world_matrix,
        "candidate_object_world_matrix",
    )
    camera_world_matrix = _validate_transform(
        camera_world_matrix,
        "camera_world_matrix",
    )
    if min_visibility_probability is not None:
        min_visibility_probability = float(
            min_visibility_probability
        )

        if not 0.0 < min_visibility_probability <= 1.0:
            raise ValueError(
                "min_visibility_probability는 0보다 크고 "
                "1 이하여야 합니다."
            )

    object_from_camera_cv = (
        np.linalg.inv(candidate_object_world_matrix)
        @ camera_world_matrix
        @ CAMERA_CV_TO_USD
    )
    sensor_origin_object = object_from_camera_cv[:3, 3]

    state = np.asarray(occupancy_map["state"], dtype=np.int8)
    probability = np.asarray(
        occupancy_map["probability"],
        dtype=np.float64,
    )
    roi_mask = make_roi_mask(
        occupancy_map,
        roi_min=roi_min,
        roi_max=roi_max,
    )
    grid_origin = np.asarray(occupancy_map["origin"], dtype=np.float64)
    root_size = float(occupancy_map["root_size"])
    leaf_size = float(occupancy_map["leaf_size"])

    # ray마다 파이썬 루프를 돌면 voxel 하나당 tuple을 만들게 되어
    # 후보 6개 채점에 2.6초가 걸렸다. 슬랩 판정/샘플링/가시 voxel
    # 수집을 전부 numpy로 처리한다. 판정 규칙은 동일하다:
    # occupied voxel을 '포함해서' ray를 끊고, ROI 안의 voxel만 센다.
    geometry = None
    if ray_geometry_cache is not None:
        # Exact geometry keys invalidate on camera/object motion, intrinsics,
        # resolution, sampling or grid changes. ROI and occupancy are evaluated
        # below on every call, so newly occupied cells still stop the rays.
        geometry_key = (
            object_from_camera_cv.tobytes(),
            np.asarray(intrinsic_matrix, dtype=np.float64).tobytes(),
            tuple(image_size) if image_size is not None else None,
            float(pixel_stride), grid_origin.tobytes(),
            root_size, leaf_size, state.shape,
        )
        geometry = ray_geometry_cache.get(geometry_key)
    if geometry is None:
        camera_rays = generate_camera_rays(
            intrinsic_matrix=intrinsic_matrix,
            image_size=image_size,
            pixel_stride=pixel_stride,
        )
        geometry = ray_voxel_indices_batch(
            sensor_origin=sensor_origin_object,
            ray_directions=camera_rays @ object_from_camera_cv[:3, :3].T,
            grid_origin=grid_origin,
            root_size=root_size,
            leaf_size=leaf_size,
        )
        if ray_geometry_cache is not None:
            ray_geometry_cache.put(geometry_key, geometry)
    flat_indices, ray_valid, sample_valid = geometry
    cells_per_axis = state.shape[0]
    state_flat = state.reshape(-1)
    roi_flat = roi_mask.reshape(-1)

    occupied_hit = (
        (state_flat[flat_indices] == OCCUPANCY_OCCUPIED) & sample_valid
    )
    has_hit = occupied_hit.any(axis=1)
    sample_count = flat_indices.shape[1]
    first_hit = np.where(
        has_hit,
        occupied_hit.argmax(axis=1),
        sample_count - 1,
    )
    positions = np.arange(sample_count)[None, :]
    keep = (
        ray_valid[:, None]
        & sample_valid
        & (positions <= first_hit[:, None])
    )

    map_intersecting_ray_count = int(np.count_nonzero(ray_valid))
    occupied_stopped_ray_count = int(
        np.count_nonzero(ray_valid & has_hit)
    )

    visible_flat = np.unique(flat_indices[keep])

    if len(visible_flat) > 0:
        visible_flat = visible_flat[roi_flat[visible_flat]]

    if len(visible_flat) > 0:
        visible_indices = np.stack(
            [
                visible_flat // (cells_per_axis * cells_per_axis),
                (visible_flat // cells_per_axis) % cells_per_axis,
                visible_flat % cells_per_axis,
            ],
            axis=1,
        ).astype(np.int32)
        visible_state = state_flat[visible_flat]
        visible_probability = probability.reshape(-1)[visible_flat]
    else:
        visible_indices = np.empty((0, 3), dtype=np.int32)
        visible_state = np.empty(0, dtype=np.int8)
        visible_probability = np.empty(0, dtype=np.float64)

    visible_entropy = binary_entropy(
        visible_probability
    )
    raw_entropy_sum = float(visible_entropy.sum())
    unknown_mask = (
        visible_state == OCCUPANCY_UNKNOWN
    )
    # UNKNOWN cells have p=0.5 in the occupancy map, so this is one bit per
    # cell. Keep it explicit because route planning must count only genuinely
    # new information and remove voxel overlap across candidate views.
    unknown_entropy_sum = float(visible_entropy[unknown_mask].sum())

    # 표면 가중. flat index 규약은 state.reshape(-1)과 같은 C order라
    # 같은 격자 모양이면 그대로 인덱싱할 수 있다.
    surface_visible_count = 0
    surface_unknown_visible_count = 0
    surface_entropy_sum = 0.0

    if surface_mask is None:
        entropy_score = (
            unknown_entropy_sum
            if score_unknown_only
            else raw_entropy_sum
        )
    else:
        surface_grid = np.asarray(surface_mask, dtype=bool)

        if surface_grid.shape != state.shape:
            raise ValueError(
                f"surface_mask shape {surface_grid.shape} != "
                f"occupancy state shape {state.shape}"
            )

        if len(visible_flat) > 0:
            visible_is_surface = surface_grid.reshape(-1)[visible_flat]
            surface_entropy_sum = float(
                visible_entropy[visible_is_surface].sum()
            )
            surface_visible_count = int(
                np.count_nonzero(visible_is_surface)
            )
            surface_unknown_visible_count = int(
                np.count_nonzero(visible_is_surface & unknown_mask)
            )
            entropy_score = float(
                surface_entropy_sum
                + float(background_weight)
                * float(
                    visible_entropy[~visible_is_surface].sum()
                )
            )
        else:
            entropy_score = 0.0
    expected_unknown_visible_count = float(
        np.count_nonzero(unknown_mask)
    )
    directional_surface_visible_count = 0
    directional_surface_unknown_visible_count = 0

    if surface_model is not None:
        directional = score_directional_surfels(
            surface_model=surface_model,
            object_from_camera_cv=object_from_camera_cv,
            intrinsic_matrix=intrinsic_matrix,
            image_size=image_size,
            pixel_bin_size=surface_pixel_bin_size,
            occlusion_tolerance=surface_occlusion_tolerance,
            min_facing_cos=surface_min_facing_cos,
            top_weight=surface_top_weight,
            bottom_weight=surface_bottom_weight,
            side_weight=surface_side_weight,
        )
        directional_surface_visible_count = directional["visible_count"]
        directional_surface_unknown_visible_count = directional[
            "unobserved_visible_count"
        ]
        # Directional surfels are the primary inspection objective. The coarse
        # occupancy entropy remains only as a small tie-break, matching the
        # previous background_weight behavior.
        entropy_score = float(
            directional["weighted_unobserved_score"]
            + float(background_weight) * raw_entropy_sum
        )
        surface_visible_count = directional_surface_visible_count
        surface_unknown_visible_count = (
            directional_surface_unknown_visible_count
        )
        surface_entropy_sum = float(
            directional_surface_unknown_visible_count
        )
        expected_unknown_visible_count = float(
            directional_surface_unknown_visible_count
        )
    visible_count = len(visible_indices)
    result = {
        "candidate_name": candidate_name,
        "entropy_score": entropy_score,
        "raw_entropy_sum": raw_entropy_sum,
        "unknown_entropy_sum": unknown_entropy_sum,
        "expected_unknown_visible_count": (
            expected_unknown_visible_count
        ),
        "unknown_visible_count": int(
            np.count_nonzero(unknown_mask)
        ),
        "free_visible_count": int(
            np.count_nonzero(visible_state == OCCUPANCY_FREE)
        ),
        "occupied_visible_count": int(
            np.count_nonzero(visible_state == OCCUPANCY_OCCUPIED)
        ),
        "visible_voxel_count": int(visible_count),
        "surface_entropy_sum": surface_entropy_sum,
        "surface_visible_count": surface_visible_count,
        "surface_unknown_visible_count": surface_unknown_visible_count,
        "directional_surface_visible_count": (
            directional_surface_visible_count
        ),
        "directional_surface_unknown_visible_count": (
            directional_surface_unknown_visible_count
        ),
        "directional_surface_weighted_unknown_score": (
            float(directional["weighted_unobserved_score"])
            if surface_model is not None
            else 0.0
        ),
        "directional_surface_unknown_groups": (
            directional["unobserved_visible_groups"]
            if surface_model is not None
            else {}
        ),
        "mean_visible_entropy": (
            entropy_score / visible_count if visible_count > 0 else 0.0
        ),
        "mean_visibility_probability": (
            1.0 if visible_count > 0 else 0.0
        ),
        "sampled_ray_count": int(len(ray_valid)),
        "map_intersecting_ray_count": int(map_intersecting_ray_count),
        "occupied_stopped_ray_count": int(occupied_stopped_ray_count),
        "attenuation_stopped_ray_count": 0,
        "sensor_origin_object": sensor_origin_object.copy(),
    }

    if return_visible_indices:
        result["visible_voxel_indices"] = visible_indices
        result["visible_voxel_probabilities"] = (
            np.ones(visible_count, dtype=np.float64)
        )
        result["visible_voxel_entropies"] = visible_entropy

    return result


def score_candidate_poses(
    occupancy_map,
    candidates,
    camera_world_matrix,
    intrinsic_matrix,
    progress_label="[NBV]",
    **score_kwargs,
):
    """[(name, object_world_matrix), ...]의 점수를 계산한다."""
    if isinstance(occupancy_map, (str, os.PathLike)):
        occupancy_map = load_occupancy_map(occupancy_map)

    candidates = list(candidates)
    results = []
    candidate_count = len(candidates)
    scoring_started_at = time.monotonic()

    for candidate_index, (candidate_name, candidate_matrix) in enumerate(
        candidates,
        start=1,
    ):
        print(
            f"{progress_label} Scoring candidate "
            f"{candidate_index}/{candidate_count}: {candidate_name}",
            flush=True,
        )
        result = score_candidate_pose(
            occupancy_map=occupancy_map,
            candidate_object_world_matrix=candidate_matrix,
            camera_world_matrix=camera_world_matrix,
            intrinsic_matrix=intrinsic_matrix,
            candidate_name=candidate_name,
            **score_kwargs,
        )
        results.append(result)
        elapsed = time.monotonic() - scoring_started_at
        estimated_remaining = (
            elapsed / candidate_index
        ) * (candidate_count - candidate_index)
        print(
            f"{progress_label} Scored candidate "
            f"{candidate_index}/{candidate_count}: {candidate_name}, "
            f"score={result['entropy_score']:.4f}, "
            f"elapsed={elapsed:.1f}s, ETA={estimated_remaining:.1f}s",
            flush=True,
        )

    return results


def select_next_best_pose(results, score_key="entropy_score"):
    """가장 높은 score의 candidate result를 반환한다."""
    if not results:
        raise ValueError("선택할 candidate score가 없습니다.")

    for result in results:
        if score_key not in result:
            raise KeyError(
                f"candidate result에 {score_key!r}가 없습니다."
            )

        if not np.isfinite(result[score_key]):
            raise ValueError(
                f"candidate score가 유한하지 않습니다: {result}"
            )

    return max(results, key=lambda result: result[score_key])


def print_candidate_scores(results):
    """후보 점수를 사람이 확인하기 쉬운 형태로 출력한다."""
    print("\n========== Candidate NBV Scores ==========")

    for result in sorted(
        results,
        key=lambda item: item["entropy_score"],
        reverse=True,
    ):
        surface_groups = result.get(
            "directional_surface_unknown_groups",
            {},
        )
        print(
            f"{result['candidate_name']}: "
            f"visible_entropy={result['entropy_score']:.4f}, "
            f"visible_unknown="
            f"{result['expected_unknown_visible_count']:.2f}, "
            f"unknown={result['unknown_visible_count']}, "
            f"free={result['free_visible_count']}, "
            f"occupied={result['occupied_visible_count']}, "
            f"visible={result['visible_voxel_count']}, "
            f"surface_unknown="
            f"{result.get('surface_unknown_visible_count', 0)}, "
            "surface_unknown_groups="
            f"top:{surface_groups.get('top(+Z)', 0)}/"
            f"side:{surface_groups.get('side', 0)}/"
            f"bottom:{surface_groups.get('bottom(-Z)', 0)}, "
            f"mean_entropy="
            f"{result['mean_visible_entropy']:.4f}"
        )

    best = select_next_best_pose(results)
    print("Top-scoring NBV:", best["candidate_name"])
    print("==========================================\n")
    return best
