"""Object-frame probabilistic occupancy map utilities."""

import os

import numpy as np


OCCUPANCY_UNKNOWN = -1
OCCUPANCY_FREE = 0
OCCUPANCY_OCCUPIED = 1


def probability_to_log_odds(probability):
    probability = np.asarray(
        probability,
        dtype=np.float64,
    )

    return np.log(
        probability / (1.0 - probability)
    )


def log_odds_to_probability(log_odds):
    log_odds = np.asarray(
        log_odds,
        dtype=np.float64,
    )

    return 1.0 / (1.0 + np.exp(-log_odds))


def create_object_occupancy_map(
    origin,
    root_size,
    leaf_size,
):
    origin = np.asarray(
        origin,
        dtype=np.float32,
    )

    root_size = float(root_size)
    leaf_size = float(leaf_size)

    if origin.shape != (3,):
        raise ValueError(
            f"occupancy origin shape가 (3,)이 아닙니다: {origin.shape}"
        )

    if root_size <= 0.0 or leaf_size <= 0.0:
        raise ValueError(
            "root_size와 leaf_size는 0보다 커야 합니다."
        )

    cells_float = root_size / leaf_size
    cells_per_axis = int(round(cells_float))

    if not np.isclose(
        cells_float,
        cells_per_axis,
        rtol=0.0,
        atol=1e-6,
    ):
        raise ValueError(
            "root_size가 leaf_size의 정수 배가 아닙니다: "
            f"root_size={root_size}, leaf_size={leaf_size}"
        )

    grid_shape = (
        cells_per_axis,
        cells_per_axis,
        cells_per_axis,
    )

    return {
        "origin": origin,
        "root_size": np.float32(root_size),
        "leaf_size": np.float32(leaf_size),
        "grid_shape": np.asarray(
            grid_shape,
            dtype=np.int32,
        ),
        # L=0은 p=0.5를 의미한다.
        "log_odds": np.zeros(
            grid_shape,
            dtype=np.float32,
        ),
        # L=0인 미관측 voxel과 상반된 관측이 누적된 voxel을 구분한다.
        "observation_count": np.zeros(
            grid_shape,
            dtype=np.uint16,
        ),
        # 정확한 ray 시각화를 위한 누적 camera-origin / endpoint 기록.
        "ray_origins": np.empty((0, 3), dtype=np.float32),
        "ray_endpoints": np.empty((0, 3), dtype=np.float32),
        "ray_frame_ids": np.empty(0, dtype=np.int32),
    }


def load_object_occupancy_map(file_path):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(
            f"Occupancy map 파일이 없습니다:\n{file_path}"
        )

    with np.load(file_path) as data:
        occupancy_map = {
            "origin": data["origin"].astype(np.float32),
            "root_size": np.float32(data["root_size"]),
            "leaf_size": np.float32(data["leaf_size"]),
            "grid_shape": data["grid_shape"].astype(np.int32),
            "log_odds": data["log_odds"].astype(np.float32),
            "observation_count": data[
                "observation_count"
            ].astype(np.uint16),
            # 이전 버전의 .npz에는 ray 정보가 없으므로 빈 배열로 읽는다.
            "ray_origins": (
                data["ray_origins"].astype(np.float32)
                if "ray_origins" in data.files
                else np.empty((0, 3), dtype=np.float32)
            ),
            "ray_endpoints": (
                data["ray_endpoints"].astype(np.float32)
                if "ray_endpoints" in data.files
                else np.empty((0, 3), dtype=np.float32)
            ),
            "ray_frame_ids": (
                data["ray_frame_ids"].astype(np.int32)
                if "ray_frame_ids" in data.files
                else np.empty(0, dtype=np.int32)
            ),
        }

    expected_shape = tuple(
        occupancy_map["grid_shape"].tolist()
    )

    if occupancy_map["log_odds"].shape != expected_shape:
        raise ValueError(
            "저장된 log_odds shape가 grid_shape와 다릅니다: "
            f"{occupancy_map['log_odds'].shape} != {expected_shape}"
        )

    if occupancy_map["observation_count"].shape != expected_shape:
        raise ValueError(
            "저장된 observation_count shape가 grid_shape와 다릅니다: "
            f"{occupancy_map['observation_count'].shape} != "
            f"{expected_shape}"
        )

    ray_count = len(occupancy_map["ray_origins"])

    if occupancy_map["ray_origins"].shape != (ray_count, 3):
        raise ValueError(
            "ray_origins shape가 (N, 3)이 아닙니다: "
            f"{occupancy_map['ray_origins'].shape}"
        )

    if occupancy_map["ray_endpoints"].shape != (ray_count, 3):
        raise ValueError(
            "ray_endpoints shape가 ray_origins와 다릅니다: "
            f"{occupancy_map['ray_endpoints'].shape}"
        )

    if occupancy_map["ray_frame_ids"].shape != (ray_count,):
        raise ValueError(
            "ray_frame_ids shape가 ray 개수와 다릅니다: "
            f"{occupancy_map['ray_frame_ids'].shape}"
        )

    return occupancy_map


def save_object_occupancy_map(
    occupancy_map,
    file_path,
    save_label=None,
):
    save_dir = os.path.dirname(file_path)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    log_odds = np.asarray(
        occupancy_map["log_odds"],
        dtype=np.float32,
    )
    observation_count = np.asarray(
        occupancy_map["observation_count"],
        dtype=np.uint16,
    )
    ray_origins = np.asarray(
        occupancy_map.get(
            "ray_origins",
            np.empty((0, 3), dtype=np.float32),
        ),
        dtype=np.float32,
    ).reshape(-1, 3)
    ray_endpoints = np.asarray(
        occupancy_map.get(
            "ray_endpoints",
            np.empty((0, 3), dtype=np.float32),
        ),
        dtype=np.float32,
    ).reshape(-1, 3)
    ray_frame_ids = np.asarray(
        occupancy_map.get(
            "ray_frame_ids",
            np.empty(0, dtype=np.int32),
        ),
        dtype=np.int32,
    ).reshape(-1)

    if not (
        len(ray_origins)
        == len(ray_endpoints)
        == len(ray_frame_ids)
    ):
        raise ValueError(
            "ray_origins, ray_endpoints, ray_frame_ids의 길이가 다릅니다."
        )

    probability = log_odds_to_probability(
        log_odds
    ).astype(np.float32)

    state = np.full(
        log_odds.shape,
        OCCUPANCY_UNKNOWN,
        dtype=np.int8,
    )

    observed_mask = observation_count > 0
    state[
        observed_mask & (probability < 0.5)
    ] = OCCUPANCY_FREE
    state[
        observed_mask & (probability >= 0.5)
    ] = OCCUPANCY_OCCUPIED

    np.savez_compressed(
        file_path,
        origin=np.asarray(
            occupancy_map["origin"],
            dtype=np.float32,
        ),
        root_size=np.float32(
            occupancy_map["root_size"]
        ),
        leaf_size=np.float32(
            occupancy_map["leaf_size"]
        ),
        grid_shape=np.asarray(
            occupancy_map["grid_shape"],
            dtype=np.int32,
        ),
        log_odds=log_odds,
        probability=probability,
        observation_count=observation_count,
        state=state,
        ray_origins=ray_origins,
        ray_endpoints=ray_endpoints,
        ray_frame_ids=ray_frame_ids,
    )

    state_values, state_counts = np.unique(
        state,
        return_counts=True,
    )
    state_count_dict = dict(
        zip(
            state_values.tolist(),
            state_counts.tolist(),
        )
    )

    if save_label is None:
        print("[OCCUPANCY] Saved:", file_path)
    else:
        print(
            "[OCCUPANCY] Saved:",
            save_label,
            file_path,
        )

    print(
        "[OCCUPANCY] State counts:",
        {
            "unknown": state_count_dict.get(
                OCCUPANCY_UNKNOWN,
                0,
            ),
            "free": state_count_dict.get(
                OCCUPANCY_FREE,
                0,
            ),
            "occupied": state_count_dict.get(
                OCCUPANCY_OCCUPIED,
                0,
            ),
        },
    )
    print(
        "[OCCUPANCY] Saved ray count:",
        len(ray_origins),
    )


def sample_segment_voxel_indices(
    start_point,
    end_point,
    origin,
    root_size,
    leaf_size,
):
    """선분이 고정 grid 내부에서 지나는 voxel index를 순서대로 반환한다."""
    start_point = np.asarray(start_point, dtype=np.float64)
    end_point = np.asarray(end_point, dtype=np.float64)
    origin = np.asarray(origin, dtype=np.float64)

    direction = end_point - start_point
    bounds_min = origin
    bounds_max = origin + float(root_size)

    t_enter = 0.0
    t_exit = 1.0

    # Slab 방식으로 선분과 object-map AABB의 교차 구간을 구한다.
    for axis in range(3):
        if abs(direction[axis]) < 1e-12:
            if (
                start_point[axis] < bounds_min[axis]
                or start_point[axis] > bounds_max[axis]
            ):
                return np.empty((0, 3), dtype=np.int32)

            continue

        axis_t1 = (
            bounds_min[axis] - start_point[axis]
        ) / direction[axis]
        axis_t2 = (
            bounds_max[axis] - start_point[axis]
        ) / direction[axis]

        axis_enter = min(axis_t1, axis_t2)
        axis_exit = max(axis_t1, axis_t2)

        t_enter = max(t_enter, axis_enter)
        t_exit = min(t_exit, axis_exit)

        if t_enter > t_exit:
            return np.empty((0, 3), dtype=np.int32)

    clipped_start = start_point + t_enter * direction
    clipped_end = start_point + t_exit * direction
    clipped_length = np.linalg.norm(
        clipped_end - clipped_start
    )

    # leaf 크기의 절반 간격으로 샘플링하면 통과 voxel을 빠뜨리지 않는다.
    sample_step = float(leaf_size) * 0.5
    sample_count = max(
        2,
        int(np.ceil(clipped_length / sample_step)) + 1,
    )

    sample_t = np.linspace(
        0.0,
        1.0,
        sample_count,
        dtype=np.float64,
    )
    sample_points = (
        clipped_start[None, :]
        + sample_t[:, None]
        * (clipped_end - clipped_start)[None, :]
    )

    # AABB 경계에서 floor 결과가 grid 크기와 같아지는 것을 방지한다.
    epsilon = float(leaf_size) * 1e-6
    sample_points = np.clip(
        sample_points,
        bounds_min + epsilon,
        bounds_max - epsilon,
    )

    voxel_indices = np.floor(
        (sample_points - origin[None, :])
        / float(leaf_size)
    ).astype(np.int32)

    # 같은 voxel 안에서 연속 샘플된 중복 index를 제거한다.
    keep_mask = np.ones(
        len(voxel_indices),
        dtype=bool,
    )

    if len(voxel_indices) > 1:
        keep_mask[1:] = np.any(
            voxel_indices[1:] != voxel_indices[:-1],
            axis=1,
        )

    return voxel_indices[keep_mask]


def update_object_occupancy_map(
    occupancy_map,
    sensor_origin_object,
    target_points_object,
    frame_id=None,
    occupied_probability=0.7,
    free_probability=0.4,
    clamp_probability_min=0.12,
    clamp_probability_max=0.97,
):
    """Target depth endpoint와 line-of-sight로 occupancy log-odds를 갱신한다."""
    sensor_origin_object = np.asarray(
        sensor_origin_object,
        dtype=np.float64,
    )
    target_points_object = np.asarray(
        target_points_object,
        dtype=np.float64,
    )

    if sensor_origin_object.shape != (3,):
        raise ValueError(
            "sensor_origin_object shape가 (3,)이 아닙니다: "
            f"{sensor_origin_object.shape}"
        )

    if (
        target_points_object.ndim != 2
        or target_points_object.shape[1] != 3
    ):
        raise ValueError(
            "target_points_object shape가 (N, 3)이 아닙니다: "
            f"{target_points_object.shape}"
        )

    origin = np.asarray(
        occupancy_map["origin"],
        dtype=np.float64,
    )
    root_size = float(occupancy_map["root_size"])
    leaf_size = float(occupancy_map["leaf_size"])
    grid_shape = np.asarray(
        occupancy_map["grid_shape"],
        dtype=np.int32,
    )

    valid_mask = np.all(
        np.isfinite(target_points_object),
        axis=1,
    )

    bounds_max = origin + root_size
    valid_mask &= np.all(
        target_points_object >= origin[None, :],
        axis=1,
    )
    valid_mask &= np.all(
        target_points_object < bounds_max[None, :],
        axis=1,
    )

    valid_target_points = target_points_object[valid_mask]
    removed_point_count = (
        len(target_points_object)
        - len(valid_target_points)
    )

    if len(valid_target_points) == 0:
        print(
            "[OCCUPANCY] WARNING: 고정 grid 내부의 target point가 없습니다."
        )
        return occupancy_map

    endpoint_indices = np.floor(
        (valid_target_points - origin[None, :])
        / leaf_size
    ).astype(np.int32)

    # 한 프레임에서 같은 endpoint voxel은 한 번만 ray update한다.
    unique_endpoint_indices, representative_indices = np.unique(
        endpoint_indices,
        axis=0,
        return_index=True,
    )
    representative_points = valid_target_points[
        representative_indices
    ]

    ray_frame_id = (
        -1
        if frame_id is None
        else int(frame_id)
    )
    new_ray_origins = np.repeat(
        sensor_origin_object[None, :],
        len(representative_points),
        axis=0,
    ).astype(np.float32)
    new_ray_endpoints = representative_points.astype(
        np.float32
    )
    new_ray_frame_ids = np.full(
        len(representative_points),
        ray_frame_id,
        dtype=np.int32,
    )

    occupancy_map["ray_origins"] = np.concatenate(
        [
            np.asarray(
                occupancy_map.get(
                    "ray_origins",
                    np.empty((0, 3), dtype=np.float32),
                ),
                dtype=np.float32,
            ).reshape(-1, 3),
            new_ray_origins,
        ],
        axis=0,
    )
    occupancy_map["ray_endpoints"] = np.concatenate(
        [
            np.asarray(
                occupancy_map.get(
                    "ray_endpoints",
                    np.empty((0, 3), dtype=np.float32),
                ),
                dtype=np.float32,
            ).reshape(-1, 3),
            new_ray_endpoints,
        ],
        axis=0,
    )
    occupancy_map["ray_frame_ids"] = np.concatenate(
        [
            np.asarray(
                occupancy_map.get(
                    "ray_frame_ids",
                    np.empty(0, dtype=np.int32),
                ),
                dtype=np.int32,
            ).reshape(-1),
            new_ray_frame_ids,
        ],
        axis=0,
    )

    occupied_keys = {
        tuple(index.tolist())
        for index in unique_endpoint_indices
    }
    free_keys = set()

    for endpoint_point, endpoint_index in zip(
        representative_points,
        unique_endpoint_indices,
    ):
        ray_indices = sample_segment_voxel_indices(
            start_point=sensor_origin_object,
            end_point=endpoint_point,
            origin=origin,
            root_size=root_size,
            leaf_size=leaf_size,
        )

        endpoint_key = tuple(endpoint_index.tolist())

        for ray_index in ray_indices:
            ray_key = tuple(ray_index.tolist())

            if ray_key != endpoint_key:
                free_keys.add(ray_key)

    # 동일 frame에서 occupied endpoint로 관측된 voxel이 항상 우선한다.
    free_keys.difference_update(occupied_keys)

    log_odds = occupancy_map["log_odds"]
    observation_count = occupancy_map[
        "observation_count"
    ]

    hit_log_odds = float(
        probability_to_log_odds(occupied_probability)
    )
    miss_log_odds = float(
        probability_to_log_odds(free_probability)
    )
    clamp_min_log_odds = float(
        probability_to_log_odds(clamp_probability_min)
    )
    clamp_max_log_odds = float(
        probability_to_log_odds(clamp_probability_max)
    )

    def keys_to_index_arrays(keys):
        if not keys:
            return None

        indices = np.asarray(
            sorted(keys),
            dtype=np.int32,
        )

        in_bounds = np.all(
            (indices >= 0)
            & (indices < grid_shape[None, :]),
            axis=1,
        )
        indices = indices[in_bounds]

        if len(indices) == 0:
            return None

        return tuple(indices.T)

    free_index_arrays = keys_to_index_arrays(free_keys)
    occupied_index_arrays = keys_to_index_arrays(occupied_keys)

    if free_index_arrays is not None:
        log_odds[free_index_arrays] += miss_log_odds
        observation_count[free_index_arrays] += 1

    if occupied_index_arrays is not None:
        log_odds[occupied_index_arrays] += hit_log_odds
        observation_count[occupied_index_arrays] += 1

    np.clip(
        log_odds,
        clamp_min_log_odds,
        clamp_max_log_odds,
        out=log_odds,
    )

    print("[OCCUPANCY] Sensor origin:", sensor_origin_object)
    print(
        "[OCCUPANCY] Target points used:",
        len(valid_target_points),
    )
    print(
        "[OCCUPANCY] Unique occupied endpoints:",
        len(occupied_keys),
    )
    print(
        "[OCCUPANCY] Unique free voxels:",
        len(free_keys),
    )
    print(
        "[OCCUPANCY] Ray history count:",
        len(occupancy_map["ray_origins"]),
    )

    if removed_point_count > 0:
        print(
            "[OCCUPANCY] WARNING: grid 밖 target points removed:",
            removed_point_count,
        )

    return occupancy_map


def update_and_save_object_occupancy_map(
    frame_id,
    zivid_save_dir,
    origin,
    root_size,
    leaf_size,
    sensor_origin_object,
    target_points_object,
    occupied_probability=0.7,
    free_probability=0.4,
    clamp_probability_min=0.12,
    clamp_probability_max=0.97,
):
    """한 capture의 occupancy map 로드, 갱신, snapshot 저장을 수행한다."""
    origin = np.asarray(origin, dtype=np.float32)
    root_size = float(root_size)
    leaf_size = float(leaf_size)

    occupancy_map_dir = os.path.join(
        zivid_save_dir,
        "object_occupancy",
    )
    occupancy_latest_path = os.path.join(
        occupancy_map_dir,
        "object_occupancy_latest.npz",
    )
    occupancy_frame_path = os.path.join(
        occupancy_map_dir,
        f"object_occupancy_{frame_id:06d}.npz",
    )

    # frame 0은 새로운 NBV episode의 시작으로 보고 map을 초기화한다.
    if (
        frame_id == 0
        or not os.path.isfile(occupancy_latest_path)
    ):
        occupancy_map = create_object_occupancy_map(
            origin=origin,
            root_size=root_size,
            leaf_size=leaf_size,
        )
        print(
            "[OCCUPANCY] New map initialized:",
            occupancy_map["grid_shape"],
        )
    else:
        occupancy_map = load_object_occupancy_map(
            occupancy_latest_path
        )

        if not np.allclose(
            occupancy_map["origin"],
            origin,
            rtol=0.0,
            atol=1e-7,
        ):
            raise ValueError(
                "기존 occupancy map의 origin이 현재 설정과 다릅니다."
            )

        if not np.isclose(
            float(occupancy_map["root_size"]),
            root_size,
            rtol=0.0,
            atol=1e-7,
        ):
            raise ValueError(
                "기존 occupancy map의 root_size가 현재 설정과 다릅니다."
            )

        if not np.isclose(
            float(occupancy_map["leaf_size"]),
            leaf_size,
            rtol=0.0,
            atol=1e-7,
        ):
            raise ValueError(
                "기존 occupancy map의 leaf_size가 현재 설정과 다릅니다."
            )

        print(
            "[OCCUPANCY] Previous map loaded:",
            occupancy_latest_path,
        )

    print(
        "[OCCUPANCY] Updating map for frame:",
        frame_id,
    )

    occupancy_map = update_object_occupancy_map(
        occupancy_map=occupancy_map,
        sensor_origin_object=sensor_origin_object,
        target_points_object=target_points_object,
        frame_id=frame_id,
        occupied_probability=occupied_probability,
        free_probability=free_probability,
        clamp_probability_min=clamp_probability_min,
        clamp_probability_max=clamp_probability_max,
    )

    # latest는 다음 frame 갱신용, frame 파일은 단계별 검증용이다.
    save_object_occupancy_map(
        occupancy_map,
        occupancy_latest_path,
        save_label="latest",
    )
    save_object_occupancy_map(
        occupancy_map,
        occupancy_frame_path,
        save_label=f"frame_{frame_id:06d}",
    )

    return occupancy_map