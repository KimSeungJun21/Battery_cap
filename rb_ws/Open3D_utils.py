"""Open3D loading and basic point cloud visualization helpers."""

import os

import numpy as np

try:
    from rb_ws.Visualization_config import VOXEL_SIZE
except ModuleNotFoundError:
    from Visualization_config import VOXEL_SIZE


def get_open3d():
    try:
        import open3d as o3d
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "open3d가 설치되어 있지 않습니다. "
            "이 스크립트는 Open3D 환경에서 실행해야 합니다."
        ) from exc

    return o3d


def numpy_to_open3d_pcd(points):
    o3d = get_open3d()

    points = np.asarray(
        points,
        dtype=np.float64,
    )

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(
            f"points shape가 (N, 3)이 아닙니다: {points.shape}"
        )

    valid_mask = np.all(
        np.isfinite(points),
        axis=1,
    )

    points = points[valid_mask]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(
        points
    )

    return pcd


def ensure_open3d_pcd(pointcloud):
    o3d = get_open3d()

    if isinstance(pointcloud, o3d.geometry.PointCloud):
        return pointcloud

    return numpy_to_open3d_pcd(pointcloud)


def get_valid_xyz_points(points):
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

    return points[valid_mask]


def load_pointcloud(file_path):
    o3d = get_open3d()

    if not os.path.isfile(file_path):
        raise FileNotFoundError(
            f"PCD 파일이 없습니다:\n{file_path}"
        )

    extension = os.path.splitext(file_path)[1].lower()

    if extension in [
        ".ply",
        ".pcd",
        ".xyz",
    ]:
        pcd = o3d.io.read_point_cloud(file_path)

        if pcd.is_empty():
            raise RuntimeError(
                "읽은 Point Cloud가 비어 있습니다."
            )

        return pcd

    if extension == ".npy":
        points = np.load(file_path)
        return numpy_to_open3d_pcd(points)

    raise ValueError(
        f"지원하지 않는 파일 형식: {extension}"
    )


def make_coordinate_frame(size=0.2):
    o3d = get_open3d()

    return o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=size,
        origin=[
            0.0,
            0.0,
            0.0,
        ],
    )


def visualize_pointcloud_open3d(pointcloud):
    o3d = get_open3d()

    pcd = ensure_open3d_pcd(pointcloud)

    print("[PCD Visualization]")
    print("Point count:", len(pcd.points))

    o3d.visualization.draw_geometries(
        [
            pcd,
            make_coordinate_frame(),
        ],
        window_name="Zivid Point Cloud",
        width=1280,
        height=720,
    )


def create_voxel_grid(pointcloud, voxel_size=VOXEL_SIZE):
    o3d = get_open3d()

    pcd = ensure_open3d_pcd(pointcloud)

    if pcd.is_empty():
        raise RuntimeError(
            "Point Cloud가 비어 있습니다."
        )

    return o3d.geometry.VoxelGrid.create_from_point_cloud(
        pcd,
        voxel_size=voxel_size,
    )


def visualize_voxels_open3d(pointcloud, voxel_size=VOXEL_SIZE):
    o3d = get_open3d()

    voxel_grid = create_voxel_grid(
        pointcloud,
        voxel_size=voxel_size,
    )

    voxels = voxel_grid.get_voxels()

    print("[Voxel Visualization]")
    print("Voxel size:", voxel_size, "m")
    print("Voxel count:", len(voxels))
    print("Min bound:", voxel_grid.get_min_bound())
    print("Max bound:", voxel_grid.get_max_bound())

    o3d.visualization.draw_geometries(
        [
            voxel_grid,
            make_coordinate_frame(),
        ],
        window_name="Zivid Voxel Grid",
        width=1280,
        height=720,
    )
