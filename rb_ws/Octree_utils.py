"""Octree voxel creation, coloring, and storage helpers."""

import os

import numpy as np

try:
    from rb_ws.Visualization_config import VOXEL_SIZE
except ModuleNotFoundError:
    from Visualization_config import VOXEL_SIZE


def labels_to_colors(labels):
    labels = np.asarray(labels, dtype=np.int64)

    palette = np.array(
        [
            [0.50, 0.50, 0.50],  # background
            [1.00, 0.10, 0.10],
            [0.10, 0.55, 1.00],
            [0.10, 0.85, 0.25],
            [1.00, 0.75, 0.10],
            [0.75, 0.20, 1.00],
            [0.10, 0.90, 0.90],
            [1.00, 0.45, 0.10],
        ],
        dtype=np.float32,
    )

    colors = np.empty((len(labels), 3), dtype=np.float32)

    # -1과 0 모두 background/unknown
    background_mask = labels <= 0
    colors[background_mask] = palette[0]

    known_labels = labels[~background_mask]

    color_indices = (
        (known_labels - 1)
        % (len(palette) - 1)
    ) + 1

    colors[~background_mask] = palette[color_indices]

    return colors


def create_octree_voxels(
    points,
    labels=None,
    leaf_size=VOXEL_SIZE,
    max_depth=None,
    size_expand=0.01,
    fixed_origin=None,
    fixed_root_size=None,
):
    points = np.asarray(
        points,
        dtype=np.float32,
    )

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(
            f"points shape가 (N, 3)이 아닙니다: {points.shape}"
        )

    valid_mask = np.all(
        np.isfinite(points),
        axis=1,
    )

    if labels is not None:
        labels = np.asarray(labels)

        if len(labels) != len(points):
            raise ValueError(
                f"labels 길이 {len(labels)} != points 길이 {len(points)}"
            )

        labels = labels[valid_mask].astype(np.int32)

    points = points[valid_mask]

    if len(points) == 0:
        raise RuntimeError(
            "Octree로 만들 Point Cloud가 비어 있습니다."
        )

    if leaf_size <= 0.0:
        raise ValueError(
            f"leaf_size는 0보다 커야 합니다: {leaf_size}"
        )

    # ============================================================
    # Octree Root 영역 결정
    # ============================================================

    use_fixed_bounds = (
        fixed_origin is not None
        or fixed_root_size is not None
    )

    if use_fixed_bounds:

        if (
            fixed_origin is None
            or fixed_root_size is None
        ):
            raise ValueError(
                "fixed_origin과 fixed_root_size는 "
                "함께 지정해야 합니다."
            )

        origin = np.asarray(
            fixed_origin,
            dtype=np.float32,
        )

        if origin.shape != (3,):
            raise ValueError(
                f"fixed_origin shape가 (3,)이 아닙니다: "
                f"{origin.shape}"
            )

        root_size = float(
            fixed_root_size
        )

        if root_size <= 0.0:
            raise ValueError(
                f"fixed_root_size는 0보다 커야 합니다: "
                f"{root_size}"
            )

        root_max = (
            origin
            + root_size
        )

        # 고정 Octree 영역 안에 들어오는 point만 사용
        inside_mask = np.all(
            (
                points >= origin
            )
            & (
                points < root_max
            ),
            axis=1,
        )

        removed_count = (
            len(points)
            - np.count_nonzero(inside_mask)
        )

        if removed_count > 0:
            print(
                "[Octree] Fixed root 밖의 point 제거:",
                removed_count,
            )

        points = points[
            inside_mask
        ]

        if labels is not None:
            labels = labels[
                inside_mask
            ]

        if len(points) == 0:
            raise RuntimeError(
                "고정 Octree 영역 안에 point가 없습니다."
            )

    else:
        # 기존 방식: 현재 point 범위로 root 자동 결정
        min_bound = points.min(axis=0)
        max_bound = points.max(axis=0)
        center = (
            min_bound
            + max_bound
        ) * 0.5

        root_size = float(
            np.max(
                max_bound
                - min_bound
            )
        )

        if root_size <= 0.0:
            root_size = float(
                leaf_size
            )

        root_size *= (
            1.0
            + float(size_expand)
        )

        origin = (
            center
            - root_size * 0.5
        )

    if max_depth is None:
        max_depth = int(
            np.ceil(
                np.log2(root_size / leaf_size)
            )
        )

    max_depth = max(0, int(max_depth))
    division_count = 2 ** max_depth
    actual_leaf_size = root_size / division_count

    normalized = (points - origin) / root_size
    normalized = np.clip(
        normalized,
        0.0,
        np.nextafter(1.0, 0.0),
    )

    point_leaf_indices = np.floor(
        normalized * division_count
    ).astype(np.int32)

    leaf_indices, inverse_indices = np.unique(
        point_leaf_indices,
        axis=0,
        return_inverse=True,
    )

    leaf_centers = (
        origin
        + (
            leaf_indices.astype(np.float32)
            + 0.5
        )
        * actual_leaf_size
    ).astype(np.float32)

    octree_data = {
        "leaf_indices": leaf_indices,
        "leaf_centers": leaf_centers,
        "origin": origin.astype(np.float32),
        "root_size": np.array(root_size, dtype=np.float32),
        "leaf_size": np.array(actual_leaf_size, dtype=np.float32),
        "max_depth": np.array(max_depth, dtype=np.int32),
        "point_count": np.array(len(points), dtype=np.int64),
    }

    if labels is not None:
        leaf_label_pairs = np.column_stack(
            [
                inverse_indices,
                labels,
            ]
        )

        unique_pairs, counts = np.unique(
            leaf_label_pairs,
            axis=0,
            return_counts=True,
        )

        leaf_labels = np.full(
            len(leaf_indices),
            -1,
            dtype=np.int32,
        )

        leaf_label_counts = np.zeros(
            len(leaf_indices),
            dtype=np.int32,
        )

        for pair, count in zip(
            unique_pairs,
            counts,
        ):
            leaf_index = int(pair[0])

            if count > leaf_label_counts[leaf_index]:
                leaf_labels[leaf_index] = int(pair[1])
                leaf_label_counts[leaf_index] = int(count)

        octree_data["leaf_labels"] = leaf_labels
        octree_data["leaf_label_counts"] = leaf_label_counts

    return octree_data


def save_octree_voxels(octree_data, save_path):
    save_dir = os.path.dirname(save_path)

    if save_dir:
        os.makedirs(
            save_dir,
            exist_ok=True,
        )

    np.savez_compressed(
        save_path,
        **octree_data,
    )

    print(
        "[Octree] Saved:",
        save_path,
        "leaf count:",
        len(octree_data["leaf_indices"]),
    )


def load_octree_voxels(file_path):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(
            f"Octree voxel 파일이 없습니다:\n{file_path}"
        )

    data = np.load(file_path)

    octree_data = {
        "leaf_indices": data["leaf_indices"],
        "leaf_centers": data["leaf_centers"],
        "origin": data["origin"],
        "root_size": data["root_size"],
        "leaf_size": data["leaf_size"],
        "max_depth": data["max_depth"],
        "point_count": data["point_count"],
    }

    if "leaf_labels" in data.files:
        octree_data["leaf_labels"] = data["leaf_labels"]

    if "leaf_label_counts" in data.files:
        octree_data["leaf_label_counts"] = data["leaf_label_counts"]

    return octree_data
