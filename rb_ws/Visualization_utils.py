"""Compatibility wrapper and CLI for Zivid point cloud visualization."""

import argparse

import numpy as np

try:
    from rb_ws.Visualization_config import (
        COMPARE_OCTREE_PATH_A,
        COMPARE_OCTREE_PATH_B,
        OCCUPANCY_PATH,
        OCTREE_PATH,
        PCD_PATH,
        VOXEL_SIZE,
    )
    from rb_ws.Open3D_utils import (
        create_voxel_grid,
        ensure_open3d_pcd,
        get_open3d,
        get_valid_xyz_points,
        load_pointcloud,
        make_coordinate_frame,
        numpy_to_open3d_pcd,
        visualize_pointcloud_open3d,
        visualize_voxels_open3d,
    )
    from rb_ws.Octree_utils import (
        create_octree_voxels,
        labels_to_colors,
        load_octree_voxels,
        save_octree_voxels,
    )
    from rb_ws.Octree_visualization_utils import (
        compare_octree_voxels_open3d,
        visualize_octree_voxels_open3d,
    )
    from rb_ws.Occupancy_visualization_utils import (
        save_occupancy_sequence_open3d,
        visualize_occupancy_map_open3d,
    )
except ModuleNotFoundError:
    from Visualization_config import (
        COMPARE_OCTREE_PATH_A,
        COMPARE_OCTREE_PATH_B,
        OCCUPANCY_PATH,
        OCTREE_PATH,
        PCD_PATH,
        VOXEL_SIZE,
    )
    from Open3D_utils import (
        create_voxel_grid,
        ensure_open3d_pcd,
        get_open3d,
        get_valid_xyz_points,
        load_pointcloud,
        make_coordinate_frame,
        numpy_to_open3d_pcd,
        visualize_pointcloud_open3d,
        visualize_voxels_open3d,
    )
    from Octree_utils import (
        create_octree_voxels,
        labels_to_colors,
        load_octree_voxels,
        save_octree_voxels,
    )
    from Octree_visualization_utils import (
        compare_octree_voxels_open3d,
        visualize_octree_voxels_open3d,
    )
    from Occupancy_visualization_utils import (
        save_occupancy_sequence_open3d,
        visualize_occupancy_map_open3d,
    )


__all__ = [
    "COMPARE_OCTREE_PATH_A",
    "COMPARE_OCTREE_PATH_B",
    "OCCUPANCY_PATH",
    "OCTREE_PATH",
    "PCD_PATH",
    "VOXEL_SIZE",
    "compare_octree_voxels_open3d",
    "create_octree_voxels",
    "create_voxel_grid",
    "ensure_open3d_pcd",
    "get_open3d",
    "get_valid_xyz_points",
    "labels_to_colors",
    "load_octree_voxels",
    "load_pointcloud",
    "make_coordinate_frame",
    "numpy_to_open3d_pcd",
    "save_octree_voxels",
    "save_occupancy_sequence_open3d",
    "visualize_occupancy_map_open3d",
    "visualize_octree_voxels_open3d",
    "visualize_pointcloud_open3d",
    "visualize_voxels_open3d",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize a Zivid point cloud, voxel grid, or octree voxels."
    )

    parser.add_argument(
        "--path",
        default=PCD_PATH,
        help="Point cloud file path. Supports .ply, .pcd, .xyz, .npy.",
    )

    parser.add_argument(
        "--voxel-size",
        type=float,
        default=VOXEL_SIZE,
        help="Voxel size in meters.",
    )

    parser.add_argument(
        "--mode",
        choices=[
            "all",
            "pcd",
            "voxel",
            "octree",
            "saved-octree",
            "compare-octrees",
            "occupancy",
        ],
        default="saved-octree",
        help="Visualization mode.",
    )

    parser.add_argument(
        "--octree-path",
        default=OCTREE_PATH,
        help="Saved octree voxel .npz path.",
    )

    parser.add_argument(
        "--octree-path-a",
        default=COMPARE_OCTREE_PATH_A,
        help="First object-frame octree .npz path for comparison.",
    )

    parser.add_argument(
        "--octree-path-b",
        default=COMPARE_OCTREE_PATH_B,
        help="Second object-frame octree .npz path for comparison.",
    )

    parser.add_argument(
        "--occupancy-path",
        default=OCCUPANCY_PATH,
        help="Saved object-frame occupancy map .npz path.",
    )

    parser.add_argument(
        "--occupancy-show",
        choices=[
            "observed",
            "occupied",
            "free",
            "unknown",
            "all",
        ],
        default="observed",
        help=(
            "Occupancy voxels to display. "
            "Default shows free and occupied voxels."
        ),
    )

    parser.add_argument(
        "--occupancy-clearance",
        type=int,
        default=1,
        help=(
            "all 모드에서 관측 영역 주변에 숨길 Unknown voxel 두께. "
            "0이면 Unknown을 전부 표시합니다."
        ),
    )

    parser.add_argument(
        "--occupancy-crop-margin",
        type=int,
        default=None,
        help=(
            "occupancy 표시를 '물체(occupied voxel) bounding box + 이만큼의 "
            "voxel 여유'로 자른다. 격자는 0.25m 정육면체인데 물체는 "
            "0.10 x 0.19 x 0.06m뿐이라 voxel의 대부분이 원거리 공기다. "
            "예: --occupancy-crop-margin 3"
        ),
    )
    parser.add_argument(
        "--no-reverse-image",
        dest="save_reverse_image",
        action="store_false",
        help=(
            "occupancy PNG를 정면 시점만 저장한다. 기본값은 정확히 "
            "반대편에서 본 화면도 <이름>_reverse.png로 함께 저장하는 "
            "것이다 (한쪽에서만 보면 물체 뒤편 voxel이 가려진다)."
        ),
    )
    parser.set_defaults(save_reverse_image=True)
    parser.add_argument(
        "--show-rays",
        action="store_true",
        help=(
            "occupancy .npz에 저장된 camera-to-endpoint ray를 "
            "주황색 LineSet으로 표시합니다."
        ),
    )

    parser.add_argument(
        "--ray-frame",
        type=int,
        default=None,
        help=(
            "표시할 ray의 frame_id. 생략하면 누적된 모든 frame의 "
            "ray를 사용합니다."
        ),
    )

    parser.add_argument(
        "--max-rays",
        type=int,
        default=200,
        help="화면에 표시할 최대 ray 개수입니다.",
    )

    parser.add_argument(
        "--save-image",
        default=None,
        help="현재 occupancy 시각화를 지정한 PNG 파일로 저장합니다.",
    )

    parser.add_argument(
        "--save-sequence",
        action="store_true",
        help="NBV 촬영의 누적 occupancy snapshot 전체를 PNG로 저장합니다.",
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "PNG sequence 저장 경로. 기본값은 "
            "object_occupancy/nbv_occupancy_frames입니다."
        ),
    )

    parser.add_argument(
        "--manifest-path",
        default=None,
        help="Battery_rb.py가 저장한 NBV sequence manifest 경로입니다.",
    )

    parser.add_argument(
        "--sequence-rays",
        choices=["current", "all", "fixed"],
        default="current",
        help=(
            "sequence 이미지의 ray 표시 방식: current=해당 frame, "
            "all=현재까지 누적, fixed=--ray-frame으로 고정."
        ),
    )

    parser.add_argument(
        "--watch",
        action="store_true",
        help="NBV 촬영이 진행되는 동안 새 snapshot을 계속 PNG로 저장합니다.",
    )

    parser.add_argument(
        "--poll-interval",
        type=float,
        default=1.0,
        help="--watch 사용 시 새 frame을 확인하는 주기입니다. 단위: 초.",
    )

    parser.add_argument(
        "--gif-path",
        default=None,
        help="전체 PNG sequence를 애니메이션 GIF로 함께 저장합니다.",
    )

    parser.add_argument(
        "--gif-duration-ms",
        type=int,
        default=900,
        help="GIF에서 각 frame을 표시하는 시간입니다. 단위: ms.",
    )

    parser.add_argument(
        "--image-width",
        type=int,
        default=1280,
        help="PNG/GIF 이미지 너비입니다.",
    )

    parser.add_argument(
        "--image-height",
        type=int,
        default=720,
        help="PNG/GIF 이미지 높이입니다.",
    )

    parser.add_argument(
        "--octree-depth",
        type=int,
        default=None,
        help="Fixed octree depth. If omitted, depth is chosen from --voxel-size.",
    )

    parser.add_argument(
        "--target-only",
        action="store_true",
        help=(
            "saved-octree 모드에서 background를 숨기고 "
            "target instance leaf(label > 0)만 표시합니다."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print()
    print("======================================")
    print("Zivid PCD / Voxel Visualization")
    print("======================================")
    print("PCD path:", args.path)
    print("Voxel size:", args.voxel_size, "m")
    print("Mode:", args.mode)
    print("======================================")
    print()

    if args.mode == "saved-octree":
        visualize_octree_voxels_open3d(
            args.octree_path,
            target_only=args.target_only,
        )
        return

    if args.mode == "compare-octrees":
        compare_octree_voxels_open3d(
            args.octree_path_a,
            args.octree_path_b,
        )
        return

    if args.mode == "occupancy":
        if args.save_sequence or args.watch or args.gif_path:
            if (
                args.ray_frame is not None
                and args.sequence_rays != "fixed"
            ):
                print(
                    "[OCCUPANCY SEQUENCE] --ray-frame은 "
                    "--sequence-rays fixed에서만 사용됩니다. "
                    "기본 current 모드는 각 frame 자신의 ray를 표시합니다."
                )

            save_occupancy_sequence_open3d(
                occupancy_path=args.occupancy_path,
                output_dir=args.output_dir,
                manifest_path=args.manifest_path,
                display_mode=args.occupancy_show,
                unknown_clearance=args.occupancy_clearance,
                show_rays=args.show_rays,
                crop_margin_voxels=args.occupancy_crop_margin,
                save_reverse_image=args.save_reverse_image,
                ray_mode=args.sequence_rays,
                ray_frame=args.ray_frame,
                max_rays=args.max_rays,
                image_width=args.image_width,
                image_height=args.image_height,
                watch=args.watch,
                poll_interval=args.poll_interval,
                gif_path=args.gif_path,
                gif_duration_ms=args.gif_duration_ms,
            )
            return

        visualize_occupancy_map_open3d(
            args.occupancy_path,
            display_mode=args.occupancy_show,
            unknown_clearance=args.occupancy_clearance,
            crop_margin_voxels=args.occupancy_crop_margin,
            save_reverse_image=args.save_reverse_image,
            show_rays=args.show_rays,
            ray_frame=args.ray_frame,
            max_rays=args.max_rays,
            image_path=args.save_image,
            show_window=args.save_image is None,
            image_width=args.image_width,
            image_height=args.image_height,
        )
        return

    pcd = load_pointcloud(args.path)

    print("[LOAD] Point count:", len(pcd.points))

    points = np.asarray(pcd.points)
    print("[LOAD] Point min:", points.min(axis=0))
    print("[LOAD] Point max:", points.max(axis=0))
    print()

    if args.mode in ["all", "pcd"]:
        print("원본 Point Cloud 창을 엽니다.")
        visualize_pointcloud_open3d(pcd)

    if args.mode in ["all", "voxel"]:
        print()
        print("Voxel Grid 창을 엽니다.")
        visualize_voxels_open3d(
            pcd,
            voxel_size=args.voxel_size,
        )

    if args.mode in ["all", "octree"]:
        print()
        print("Octree voxel을 생성하고 저장합니다.")
        print(
            "[WARNING] pcd 파일에서 새로 생성한 Octree에는 "
            "semantic/instance label이 없습니다. target instance를 "
            "보려면 --mode saved-octree와 capture에서 저장된 "
            "octree_voxels_*.npz를 사용하세요."
        )
        octree_data = create_octree_voxels(
            points,
            leaf_size=args.voxel_size,
            max_depth=args.octree_depth,
        )
        save_octree_voxels(
            octree_data,
            args.octree_path,
        )
        visualize_octree_voxels_open3d(octree_data)


if __name__ == "__main__":
    main()