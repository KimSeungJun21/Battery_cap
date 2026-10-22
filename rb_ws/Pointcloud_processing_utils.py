"""Point cloud conversion, voxelization, and PLY storage helpers."""

import numpy as np


def save_pointcloud_ply(
    points,
    save_path,
    colors=None,
    max_points=None,
):
    """
    Colored Point Cloud 저장

    points:
        (N, 3)

    colors:
        (N, 3)
        0~1 또는 0~255

    save_path:
        *.ply
    """

    points = np.asarray(
        points,
        dtype=np.float32,
    )

    # ========================================================
    # invalid point 제거
    # ========================================================

    valid = np.all(
        np.isfinite(points),
        axis=1,
    )

    points = points[
        valid
    ]

    if colors is not None:

        colors = np.asarray(
            colors
        )

        colors = colors[
            valid
        ]

    # ========================================================
    # Downsampling
    # ========================================================

    if (
        max_points is not None
        and len(points) > max_points
    ):

        indices = np.linspace(
            0,
            len(points) - 1,
            max_points,
            dtype=np.int64,
        )

        points = points[
            indices
        ]

        if colors is not None:

            colors = colors[
                indices
            ]

    # ========================================================
    # Color 변환
    # ========================================================

    if colors is not None:

        # 0~1이면 0~255로 변환
        if colors.max() <= 1.0:

            colors = (
                colors * 255.0
            )

        colors = np.clip(
            colors,
            0,
            255,
        ).astype(
            np.uint8
        )

    # ========================================================
    # PLY 저장
    # ========================================================

    with open(
        save_path,
        "w"
    ) as f:

        f.write("ply\n")
        f.write("format ascii 1.0\n")

        f.write(
            f"element vertex "
            f"{len(points)}\n"
        )

        f.write(
            "property float x\n"
        )

        f.write(
            "property float y\n"
        )

        f.write(
            "property float z\n"
        )

        if colors is not None:

            f.write(
                "property uchar red\n"
            )

            f.write(
                "property uchar green\n"
            )

            f.write(
                "property uchar blue\n"
            )

        f.write(
            "end_header\n"
        )

        # ----------------------------------------------------
        # XYZ + RGB
        # ----------------------------------------------------

        if colors is not None:

            for point, color in zip(
                points,
                colors,
            ):

                f.write(
                    f"{point[0]:.6f} "
                    f"{point[1]:.6f} "
                    f"{point[2]:.6f} "
                    f"{color[0]} "
                    f"{color[1]} "
                    f"{color[2]}\n"
                )

        # ----------------------------------------------------
        # XYZ only
        # ----------------------------------------------------

        else:

            np.savetxt(
                f,
                points,
                fmt="%.6f %.6f %.6f",
            )

    print(
        "[PCD] Saved:",
        save_path,
        "points:",
        len(points),
    )


def depth_to_pointcloud(
    depth,
    fx,
    fy,
    cx,
    cy,
    mask=None,
    semantic_image=None,
    target_semantic_id=None,
    target_semantic_ids=None,
    return_semantic_ids=False,
):
    """
    Depth image -> Point Cloud

    mask:
        실제 PCD로 만들 pixel 영역.
        예: ROI mask

    semantic_image:
        Isaac Sim semantic segmentation ID image

    target_semantic_id:
        target_object의 semantic ID (단일)

    target_semantic_ids:
        물체 표면으로 취급할 semantic ID들. defect decal처럼 class는
        다르지만 물리적으로는 같은 표면인 경우를 함께 빨간색으로
        칠하기 위한 것이다. 주어지면 target_semantic_id보다 우선한다.

    return:
        points: (N, 3)
        colors: (N, 3), RGB [0, 1]
        semantic_ids: (N,), return_semantic_ids=True일 때만
    """

    depth = np.asarray(
        depth,
        dtype=np.float32,
    ).squeeze()

    height, width = depth.shape

    v, u = np.indices(
        (height, width)
    )

    # ========================================================
    # 유효 Depth
    # ========================================================

    valid = (
        np.isfinite(depth)
        & (depth > 0.0)
    )

    # ROI 적용
    if mask is not None:

        if mask.shape != depth.shape:
            raise ValueError(
                f"mask shape {mask.shape} != "
                f"depth shape {depth.shape}"
            )

        valid &= mask

    # ========================================================
    # 유효 pixel 추출
    # ========================================================

    u_valid = (
        u[valid]
        .astype(np.float32)
    )

    v_valid = (
        v[valid]
        .astype(np.float32)
    )

    z = depth[valid]

    # ========================================================
    # Depth -> XYZ
    # ========================================================

    x = (
        (u_valid - cx)
        * z
        / fx
    )

    y = (
        (v_valid - cy)
        * z
        / fy
    )

    points = np.stack(
        [
            x,
            y,
            z,
        ],
        axis=1,
    )

    # ========================================================
    # 기본 PCD 색상
    #
    # Background = 회색
    # ========================================================

    colors = np.full(
        (
            len(points),
            3,
        ),
        0.5,
        dtype=np.float32,
    )

    # ========================================================
    # Target Object 색상
    #
    # Target = 빨간색
    # ========================================================

    semantic_ids = None

    if semantic_image is not None:

        semantic_image = np.asarray(
            semantic_image
        ).squeeze()

        if semantic_image.shape != depth.shape:
            raise ValueError(
                f"semantic shape "
                f"{semantic_image.shape} != "
                f"depth shape {depth.shape}"
            )

        semantic_valid = semantic_image[
            valid
        ].astype(np.int32)

        semantic_ids = semantic_valid

        if target_semantic_ids is not None:
            target_mask = np.isin(
                semantic_valid,
                np.asarray(
                    list(target_semantic_ids),
                    dtype=np.int32,
                ),
            )
        elif target_semantic_id is not None:
            target_mask = (
                semantic_valid
                == target_semantic_id
            )
        else:
            target_mask = None

        if target_mask is not None:

            colors[
                target_mask
            ] = [
                1.0,
                0.0,
                0.0,
            ]

            print(
                "[PCD] Target points:",
                np.count_nonzero(
                    target_mask
                ),
            )

            print(
                "[PCD] Background points:",
                len(points)
                - np.count_nonzero(
                    target_mask
                ),
            )

    if return_semantic_ids:
        return (
            points,
            colors,
            semantic_ids,
        )

    return (
        points,
        colors,
    )


def pointcloud_to_voxels(
    points,
    voxel_size=0.01,
):
    """
    points:
        (N, 3) numpy array
        하나의 공통 좌표계에 존재하는 point cloud

    voxel_size:
        voxel 한 변의 길이 [m]

    return:
        voxel_indices : (M, 3)
        voxel_centers : (M, 3)
    """

    points = np.asarray(
        points,
        dtype=np.float32,
    )

    # invalid point 제거
    valid = np.all(
        np.isfinite(points),
        axis=1,
    )

    points = points[valid]

    if len(points) == 0:
        return (
            np.empty((0, 3), dtype=np.int32),
            np.empty((0, 3), dtype=np.float32),
        )

    # ---------------------------------------------------------
    # point → voxel coordinate
    # ---------------------------------------------------------

    voxel_indices = np.floor(
        points / voxel_size
    ).astype(np.int32)

    # 같은 voxel에 들어간 point 중복 제거
    voxel_indices = np.unique(
        voxel_indices,
        axis=0,
    )

    # voxel center 계산
    voxel_centers = (
        voxel_indices.astype(np.float32)
        + 0.5
    ) * voxel_size

    return (
        voxel_indices,
        voxel_centers,
    )
