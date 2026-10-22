"""Open3D visualization helpers for saved Octree voxel data."""

import numpy as np

try:
    from rb_ws.Open3D_utils import (
        get_open3d,
        make_coordinate_frame,
        numpy_to_open3d_pcd,
    )
    from rb_ws.Octree_utils import (
        labels_to_colors,
        load_octree_voxels,
    )
except ModuleNotFoundError:
    from Open3D_utils import (
        get_open3d,
        make_coordinate_frame,
        numpy_to_open3d_pcd,
    )
    from Octree_utils import (
        labels_to_colors,
        load_octree_voxels,
    )


def visualize_octree_voxels_open3d(
    octree_data_or_path,
    target_only=False,
):
    """
    저장된 Octree leaf voxel을 label 색상으로 시각화한다.

    target_only=False:
        배경(-1 또는 0)과 target instance(1, 2, ...)를 함께 표시

    target_only=True:
        target instance label이 1 이상인 leaf만 표시
    """
    o3d = get_open3d()

    if isinstance(octree_data_or_path, str):
        octree_data = load_octree_voxels(octree_data_or_path)
    else:
        octree_data = octree_data_or_path

    leaf_centers = np.asarray(
        octree_data["leaf_centers"],
        dtype=np.float64,
    )
    leaf_size = float(octree_data["leaf_size"])

    if len(leaf_centers) == 0:
        raise RuntimeError(
            "시각화할 Octree leaf가 없습니다."
        )

    leaf_labels = None

    if "leaf_labels" in octree_data:
        leaf_labels = np.asarray(
            octree_data["leaf_labels"],
            dtype=np.int32,
        )

        if len(leaf_labels) != len(leaf_centers):
            raise ValueError(
                f"leaf label 수 {len(leaf_labels)} != "
                f"leaf center 수 {len(leaf_centers)}"
            )

    if target_only:
        if leaf_labels is None:
            raise RuntimeError(
                "이 Octree에는 leaf_labels가 없습니다. "
                "pcd_*.ply에서 Octree를 다시 만들지 말고 "
                "save_zivid_capture()가 저장한 "
                "octree_voxels_*.npz를 사용하세요."
            )

        display_mask = leaf_labels > 0

        if not np.any(display_mask):
            raise RuntimeError(
                "Octree에 target leaf(label > 0)가 없습니다. "
                "[OCTREE] Point labels와 "
                "[ZIVID] Octree leaf labels 로그를 확인하세요."
            )

        display_centers = leaf_centers[display_mask]
        display_labels = leaf_labels[display_mask]

    else:
        display_centers = leaf_centers
        display_labels = leaf_labels

    pcd = numpy_to_open3d_pcd(display_centers)

    if display_labels is not None:
        pcd.colors = o3d.utility.Vector3dVector(
            labels_to_colors(display_labels).astype(
                np.float64
            )
        )

    # leaf center가 이미 Octree의 고정 grid에 있으므로 한 점당 하나의
    # voxel이 생성된다. 색상은 각 leaf label 색상을 그대로 사용한다.
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(
        pcd,
        voxel_size=leaf_size,
    )

    print("[Octree Voxel Visualization]")
    print("Source point count:", int(octree_data["point_count"]))
    print("Max depth:", int(octree_data["max_depth"]))
    print("Leaf size:", leaf_size, "m")
    print("Total leaf count:", len(leaf_centers))
    print("Displayed leaf count:", len(display_centers))
    print("Target only:", target_only)
    print("Origin:", octree_data["origin"])
    print("Root size:", float(octree_data["root_size"]))

    if leaf_labels is not None:
        unique_labels, counts = np.unique(
            leaf_labels,
            return_counts=True,
        )
        print(
            "Leaf labels:",
            dict(
                zip(
                    unique_labels.tolist(),
                    counts.tolist(),
                )
            ),
        )

        target_labels = leaf_labels[leaf_labels > 0]

        if len(target_labels) > 0:
            target_unique, target_counts = np.unique(
                target_labels,
                return_counts=True,
            )
            print(
                "Target leaf labels:",
                dict(
                    zip(
                        target_unique.tolist(),
                        target_counts.tolist(),
                    )
                ),
            )
        else:
            print(
                "WARNING: target leaf(label > 0)가 없습니다."
            )
    else:
        print(
            "WARNING: leaf_labels가 없어 instance 색상을 "
            "표시할 수 없습니다. octree_voxels_*.npz를 사용하세요."
        )

    o3d.visualization.draw_geometries(
        [
            voxel_grid,
            make_coordinate_frame(),
        ],
        window_name=(
            "Zivid Target Octree Voxels"
            if target_only
            else "Zivid Octree Voxels"
        ),
        width=1280,
        height=720,
    )


def compare_octree_voxels_open3d(
    octree_path_a,
    octree_path_b,
):
    """
    같은 object-frame 고정 격자에 저장된 두 Octree를 비교한다.

    색상:
        빨강: frame A에만 존재
        파랑: frame B에만 존재
        초록: 두 frame에 공통으로 존재
    """
    o3d = get_open3d()

    octree_a = load_octree_voxels(octree_path_a)
    octree_b = load_octree_voxels(octree_path_b)

    origin_a = np.asarray(octree_a["origin"], dtype=np.float64)
    origin_b = np.asarray(octree_b["origin"], dtype=np.float64)
    root_size_a = float(octree_a["root_size"])
    root_size_b = float(octree_b["root_size"])
    leaf_size_a = float(octree_a["leaf_size"])
    leaf_size_b = float(octree_b["leaf_size"])
    max_depth_a = int(octree_a["max_depth"])
    max_depth_b = int(octree_b["max_depth"])

    if not np.allclose(origin_a, origin_b, rtol=0.0, atol=1e-7):
        raise ValueError(
            "두 Octree의 origin이 다릅니다:\n"
            f"A: {origin_a}\n"
            f"B: {origin_b}"
        )

    if not np.isclose(
        root_size_a,
        root_size_b,
        rtol=0.0,
        atol=1e-7,
    ):
        raise ValueError(
            "두 Octree의 root_size가 다릅니다: "
            f"A={root_size_a}, B={root_size_b}"
        )

    if not np.isclose(
        leaf_size_a,
        leaf_size_b,
        rtol=0.0,
        atol=1e-7,
    ):
        raise ValueError(
            "두 Octree의 leaf_size가 다릅니다: "
            f"A={leaf_size_a}, B={leaf_size_b}"
        )

    if max_depth_a != max_depth_b:
        raise ValueError(
            "두 Octree의 max_depth가 다릅니다: "
            f"A={max_depth_a}, B={max_depth_b}"
        )

    def get_target_indices(octree_data):
        leaf_indices = np.asarray(
            octree_data["leaf_indices"],
            dtype=np.int64,
        )

        if leaf_indices.ndim != 2 or leaf_indices.shape[1] != 3:
            raise ValueError(
                "leaf_indices shape가 (N, 3)이 아닙니다: "
                f"{leaf_indices.shape}"
            )

        if "leaf_labels" in octree_data:
            leaf_labels = np.asarray(
                octree_data["leaf_labels"],
                dtype=np.int32,
            )

            if len(leaf_labels) != len(leaf_indices):
                raise ValueError(
                    f"leaf label 수 {len(leaf_labels)} != "
                    f"leaf index 수 {len(leaf_indices)}"
                )

            leaf_indices = leaf_indices[leaf_labels > 0]

        return np.unique(leaf_indices, axis=0)

    indices_a = get_target_indices(octree_a)
    indices_b = get_target_indices(octree_b)

    set_a = {tuple(index) for index in indices_a}
    set_b = {tuple(index) for index in indices_b}

    common_set = set_a & set_b
    only_a_set = set_a - set_b
    only_b_set = set_b - set_a
    union_set = set_a | set_b

    def set_to_indices(index_set):
        if not index_set:
            return np.empty((0, 3), dtype=np.int64)

        return np.asarray(
            sorted(index_set),
            dtype=np.int64,
        )

    common_indices = set_to_indices(common_set)
    only_a_indices = set_to_indices(only_a_set)
    only_b_indices = set_to_indices(only_b_set)

    def indices_to_centers(indices):
        return origin_a + (
            indices.astype(np.float64) + 0.5
        ) * leaf_size_a

    common_centers = indices_to_centers(common_indices)
    only_a_centers = indices_to_centers(only_a_indices)
    only_b_centers = indices_to_centers(only_b_indices)

    display_centers = np.concatenate(
        [
            common_centers,
            only_a_centers,
            only_b_centers,
        ],
        axis=0,
    )

    if len(display_centers) == 0:
        raise RuntimeError("비교할 target leaf가 없습니다.")

    display_colors = np.concatenate(
        [
            np.tile([0.10, 0.90, 0.20], (len(common_centers), 1)),
            np.tile([1.00, 0.10, 0.10], (len(only_a_centers), 1)),
            np.tile([0.10, 0.45, 1.00], (len(only_b_centers), 1)),
        ],
        axis=0,
    ).astype(np.float64)

    pcd = numpy_to_open3d_pcd(display_centers)
    pcd.colors = o3d.utility.Vector3dVector(display_colors)

    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(
        pcd,
        voxel_size=leaf_size_a,
    )

    common_count = len(common_set)
    union_count = len(union_set)
    jaccard = common_count / union_count if union_count else 0.0
    overlap_a = common_count / len(set_a) if set_a else 0.0
    overlap_b = common_count / len(set_b) if set_b else 0.0

    print("[Object-frame Octree Comparison]")
    print("Frame A:", octree_path_a)
    print("Frame B:", octree_path_b)
    print("Origin:", origin_a)
    print("Root size:", root_size_a)
    print("Leaf size:", leaf_size_a, "m")
    print("Max depth:", max_depth_a)
    print("Frame A target leaves:", len(set_a))
    print("Frame B target leaves:", len(set_b))
    print("Common leaves (green):", common_count)
    print("Frame A only (red):", len(only_a_set))
    print("Frame B only (blue):", len(only_b_set))
    print("Union leaves:", union_count)
    print("Jaccard:", f"{jaccard:.4f}")
    print("Common / Frame A:", f"{overlap_a:.4f}")
    print("Common / Frame B:", f"{overlap_b:.4f}")
    print("Color legend: common=green, A-only=red, B-only=blue")

    if common_count == 0:
        print(
            "WARNING: 공통 leaf가 없습니다. object-frame 변환과 "
            "촬영 pose 동기화를 확인하세요."
        )

    o3d.visualization.draw_geometries(
        [
            voxel_grid,
            make_coordinate_frame(),
        ],
        window_name="Object-frame Octree Comparison",
        width=1280,
        height=720,
    )
