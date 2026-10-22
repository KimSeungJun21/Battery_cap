"""Open3D visualization helpers for object-frame occupancy maps."""

import glob
import json
import os
import time

import numpy as np

try:
    from rb_ws.Open3D_utils import (
        get_open3d,
        make_coordinate_frame,
        numpy_to_open3d_pcd,
    )
except ModuleNotFoundError:
    from Open3D_utils import (
        get_open3d,
        make_coordinate_frame,
        numpy_to_open3d_pcd,
    )


def visualize_occupancy_map_open3d(
    occupancy_path,
    display_mode="observed",
    unknown_clearance=1,
    crop_margin_voxels=None,
    save_reverse_image=True,
    show_rays=False,
    ray_frame=None,
    max_rays=200,
    image_path=None,
    show_window=True,
    image_width=1280,
    image_height=720,
    frame_metadata=None,
):
    """
    저장된 object-frame occupancy map을 voxel 상태별로 시각화한다.

    색상:
        회색: unknown
        파랑: free
        초록: occupied
    """
    o3d = get_open3d()

    unknown_clearance = int(unknown_clearance)

    if unknown_clearance < 0:
        raise ValueError(
            "unknown_clearance는 0 이상의 정수여야 합니다."
        )

    max_rays = int(max_rays)

    if max_rays <= 0:
        raise ValueError("max_rays는 1 이상이어야 합니다.")

    image_width = int(image_width)
    image_height = int(image_height)

    if image_width <= 0 or image_height <= 0:
        raise ValueError(
            "image_width와 image_height는 1 이상이어야 합니다."
        )

    if not os.path.isfile(occupancy_path):
        raise FileNotFoundError(
            f"Occupancy map 파일이 없습니다:\n{occupancy_path}"
        )

    with np.load(occupancy_path) as data:
        required_keys = {
            "origin",
            "root_size",
            "leaf_size",
            "grid_shape",
            "state",
        }
        missing_keys = required_keys - set(data.files)

        if missing_keys:
            raise ValueError(
                "Occupancy map에 필요한 항목이 없습니다: "
                f"{sorted(missing_keys)}"
            )

        origin = np.asarray(
            data["origin"],
            dtype=np.float64,
        )
        root_size = float(data["root_size"])
        leaf_size = float(data["leaf_size"])
        grid_shape = tuple(
            np.asarray(
                data["grid_shape"],
                dtype=np.int32,
            ).tolist()
        )
        state = np.asarray(
            data["state"],
            dtype=np.int8,
        )

        if "probability" in data.files:
            probability = np.asarray(
                data["probability"],
                dtype=np.float32,
            )
        elif "log_odds" in data.files:
            log_odds = np.asarray(
                data["log_odds"],
                dtype=np.float32,
            )
            probability = 1.0 / (
                1.0 + np.exp(-log_odds)
            )
        else:
            probability = None

        ray_keys = {
            "ray_origins",
            "ray_endpoints",
            "ray_frame_ids",
        }
        available_ray_keys = ray_keys & set(data.files)

        if available_ray_keys and available_ray_keys != ray_keys:
            raise ValueError(
                "Occupancy map의 ray 저장 항목이 불완전합니다: "
                f"{sorted(available_ray_keys)}"
            )

        if available_ray_keys == ray_keys:
            ray_origins = np.asarray(
                data["ray_origins"],
                dtype=np.float64,
            )
            ray_endpoints = np.asarray(
                data["ray_endpoints"],
                dtype=np.float64,
            )
            ray_frame_ids = np.asarray(
                data["ray_frame_ids"],
                dtype=np.int32,
            )
        else:
            ray_origins = np.empty((0, 3), dtype=np.float64)
            ray_endpoints = np.empty((0, 3), dtype=np.float64)
            ray_frame_ids = np.empty(0, dtype=np.int32)

    if origin.shape != (3,):
        raise ValueError(
            f"origin shape가 (3,)이 아닙니다: {origin.shape}"
        )

    if state.shape != grid_shape:
        raise ValueError(
            f"state shape {state.shape} != grid_shape {grid_shape}"
        )

    unexpected_states = np.setdiff1d(
        np.unique(state),
        np.array([-1, 0, 1], dtype=np.int8),
    )

    if len(unexpected_states) > 0:
        raise ValueError(
            "지원하지 않는 occupancy state가 있습니다: "
            f"{unexpected_states.tolist()}"
        )

    if (
        probability is not None
        and probability.shape != grid_shape
    ):
        raise ValueError(
            "probability shape가 grid_shape와 다릅니다: "
            f"{probability.shape} != {grid_shape}"
        )

    ray_count = len(ray_origins)

    if ray_origins.shape != (ray_count, 3):
        raise ValueError(
            f"ray_origins shape가 (N, 3)이 아닙니다: {ray_origins.shape}"
        )

    if ray_endpoints.shape != (ray_count, 3):
        raise ValueError(
            "ray_endpoints shape가 ray_origins와 다릅니다: "
            f"{ray_endpoints.shape}"
        )

    if ray_frame_ids.shape != (ray_count,):
        raise ValueError(
            "ray_frame_ids shape가 ray 개수와 다릅니다: "
            f"{ray_frame_ids.shape}"
        )

    state_masks = {
        "unknown": state == -1,
        "free": state == 0,
        "occupied": state == 1,
        "observed": state != -1,
        "all": np.ones(state.shape, dtype=bool),
    }

    if display_mode not in state_masks:
        raise ValueError(
            f"지원하지 않는 occupancy display mode: {display_mode}"
        )

    # 격자는 0.25m 정육면체인데 물체는 0.10 x 0.19 x 0.06m뿐이라, voxel의
    # 대부분이 물체에서 한참 떨어진 원거리 공기다. 물체 주변만 잘라서
    # 보면 실제로 관측된 영역의 상태가 훨씬 잘 보인다.
    crop_mask = np.ones(state.shape, dtype=bool)
    crop_info = None

    if crop_margin_voxels is not None:
        margin = int(crop_margin_voxels)

        if margin < 0:
            raise ValueError(
                "crop_margin_voxels는 0 이상의 정수여야 합니다."
            )

        occupied_indices = np.argwhere(state == 1)

        if len(occupied_indices) == 0:
            print(
                "[OCCUPANCY CROP] occupied voxel이 없어 crop을 "
                "건너뜁니다."
            )
        else:
            low = np.maximum(
                occupied_indices.min(axis=0) - margin,
                0,
            )
            high = np.minimum(
                occupied_indices.max(axis=0) + margin,
                np.array(state.shape) - 1,
            )
            crop_mask = np.zeros(state.shape, dtype=bool)
            crop_mask[
                low[0]:high[0] + 1,
                low[1]:high[1] + 1,
                low[2]:high[2] + 1,
            ] = True
            crop_info = {
                "low": low,
                "high": high,
                "margin": margin,
                "kept": int(crop_mask.sum()),
                "total": int(state.size),
            }

    display_mask = state_masks[display_mode] & crop_mask
    display_indices = np.argwhere(display_mask)

    if len(display_indices) == 0:
        raise RuntimeError(
            f"표시할 occupancy voxel이 없습니다: {display_mode}"
        )

    state_colors = {
        -1: np.array([0.60, 0.60, 0.60], dtype=np.float64),
        0: np.array([0.10, 0.45, 1.00], dtype=np.float64),
        1: np.array([0.10, 0.90, 0.20], dtype=np.float64),
    }

    def centers_for_mask(mask):
        indices = np.argwhere(mask & crop_mask)

        return origin[None, :] + (
            indices.astype(np.float64) + 0.5
        ) * leaf_size

    def centers_for_state(state_value):
        return centers_for_mask(state == state_value)

    def dilate_mask(mask, radius):
        """3-D voxel mask를 radius만큼 확장한다."""
        if radius == 0:
            return mask.copy()

        padded = np.pad(
            mask,
            pad_width=radius,
            mode="constant",
            constant_values=False,
        )
        dilated = np.zeros_like(mask, dtype=bool)
        shape_x, shape_y, shape_z = mask.shape

        for offset_x in range(2 * radius + 1):
            for offset_y in range(2 * radius + 1):
                for offset_z in range(2 * radius + 1):
                    dilated |= padded[
                        offset_x:offset_x + shape_x,
                        offset_y:offset_y + shape_y,
                        offset_z:offset_z + shape_z,
                    ]

        return dilated

    def make_colored_pointcloud(centers, color):
        pointcloud = numpy_to_open3d_pcd(centers)
        colors = np.tile(
            color[None, :],
            (len(centers), 1),
        )
        pointcloud.colors = o3d.utility.Vector3dVector(colors)
        return pointcloud

    def make_solid_voxels(centers, color):
        pointcloud = make_colored_pointcloud(
            centers,
            color,
        )

        return o3d.geometry.VoxelGrid.create_from_point_cloud(
            pointcloud,
            voxel_size=leaf_size,
        )

    geometries = []
    hidden_unknown_count = 0
    displayed_ray_count = 0

    # 단일 상태를 검사할 때는 기존처럼 solid voxel로 표시한다.
    if display_mode in ["occupied", "free", "unknown"]:
        state_value = {
            "unknown": -1,
            "free": 0,
            "occupied": 1,
        }[display_mode]
        geometries.append(
            make_solid_voxels(
                centers_for_state(state_value),
                state_colors[state_value],
            )
        )

    else:
        # observed/all에서는 Occupied를 solid voxel로 유지하고,
        # Free만 point cloud로 표시해 뒤쪽 표면을 가리지 않는다.
        occupied_centers = centers_for_state(1)

        if len(occupied_centers) > 0:
            geometries.append(
                make_solid_voxels(
                    occupied_centers,
                    state_colors[1],
                )
            )

        free_centers = centers_for_state(0)

        if len(free_centers) > 0:
            geometries.append(
                make_colored_pointcloud(
                    free_centers,
                    state_colors[0],
                )
            )

        if display_mode == "all":
            # 각 voxel의 상태는 상호 배타적이므로 Unknown과 Free의
            # 실제 3-D 교집합은 없다. 대신 관측 영역 주위의 Unknown만
            # 시각화에서 제거해 내부 Occupied가 보이는 통로를 만든다.
            observed_clearance = dilate_mask(
                state != -1,
                unknown_clearance,
            )
            visible_unknown_mask = (
                (state == -1)
                & ~observed_clearance
            )
            hidden_unknown_count = int(
                np.count_nonzero(
                    (state == -1)
                    & observed_clearance
                )
            )
            unknown_centers = centers_for_mask(
                visible_unknown_mask
            )

            if len(unknown_centers) > 0:
                geometries.append(
                    make_solid_voxels(
                        unknown_centers,
                        state_colors[-1],
                    )
                )

    if show_rays:
        if ray_count == 0:
            raise RuntimeError(
                "저장된 ray 정보가 없습니다. 수정된 Occupancy_utils.py로 "
                "다시 촬영한 occupancy .npz가 필요합니다."
            )

        ray_mask = np.ones(ray_count, dtype=bool)

        if ray_frame is not None:
            ray_mask &= (
                ray_frame_ids == int(ray_frame)
            )

        selected_ray_origins = ray_origins[ray_mask]
        selected_ray_endpoints = ray_endpoints[ray_mask]

        if len(selected_ray_origins) == 0:
            available_frames = np.unique(
                ray_frame_ids
            ).tolist()
            raise RuntimeError(
                f"frame {ray_frame}에 저장된 ray가 없습니다. "
                f"사용 가능한 frame: {available_frames}"
            )

        if len(selected_ray_origins) > max_rays:
            sample_indices = np.linspace(
                0,
                len(selected_ray_origins) - 1,
                max_rays,
                dtype=np.int64,
            )
            selected_ray_origins = selected_ray_origins[
                sample_indices
            ]
            selected_ray_endpoints = selected_ray_endpoints[
                sample_indices
            ]

        displayed_ray_count = len(
            selected_ray_origins
        )
        line_points = np.concatenate(
            [
                selected_ray_origins,
                selected_ray_endpoints,
            ],
            axis=0,
        )
        line_indices = np.column_stack(
            [
                np.arange(displayed_ray_count),
                np.arange(displayed_ray_count)
                + displayed_ray_count,
            ]
        ).astype(np.int32)
        line_colors = np.tile(
            np.array(
                [[1.00, 0.55, 0.05]],
                dtype=np.float64,
            ),
            (displayed_ray_count, 1),
        )

        ray_line_set = o3d.geometry.LineSet()
        ray_line_set.points = o3d.utility.Vector3dVector(
            line_points
        )
        ray_line_set.lines = o3d.utility.Vector2iVector(
            line_indices
        )
        ray_line_set.colors = o3d.utility.Vector3dVector(
            line_colors
        )
        geometries.append(ray_line_set)

        unique_camera_origins = np.unique(
            selected_ray_origins,
            axis=0,
        )
        camera_origin_pcd = make_colored_pointcloud(
            unique_camera_origins,
            np.array(
                [1.00, 0.10, 0.10],
                dtype=np.float64,
            ),
        )
        geometries.append(camera_origin_pcd)

    unknown_count = int(np.count_nonzero(state == -1))
    free_count = int(np.count_nonzero(state == 0))
    occupied_count = int(np.count_nonzero(state == 1))

    print("[Object-frame Occupancy Visualization]")
    print("Path:", occupancy_path)
    if crop_info is not None:
        print(
            "Crop to object bbox + "
            f"{crop_info['margin']} voxel margin: "
            f"index {crop_info['low'].tolist()} ~ "
            f"{crop_info['high'].tolist()}, "
            f"{crop_info['kept']} / {crop_info['total']} voxels kept"
        )

    print("Display mode:", display_mode)
    print("Origin:", origin)
    print("Root size:", root_size)
    print("Leaf size:", leaf_size, "m")
    print("Grid shape:", grid_shape)
    print("Total voxel count:", state.size)
    print("Unknown voxels (gray):", unknown_count)
    print("Free voxels (blue):", free_count)
    print("Occupied voxels (green):", occupied_count)
    print(
        "Displayed voxel count:",
        len(display_indices) - hidden_unknown_count,
    )

    if display_mode == "observed":
        print(
            "Render style: occupied=solid voxels, "
            "free=points"
        )
    elif display_mode == "all":
        print(
            "Render style: occupied/unknown=solid voxels, "
            "free=points"
        )
        print(
            "Unknown clearance radius:",
            unknown_clearance,
            "voxel(s)",
        )
        print(
            "Hidden Unknown clearance voxels:",
            hidden_unknown_count,
        )
    else:
        print("Render style: selected state=solid voxels")

    if probability is not None:
        observed_probability = probability[state != -1]

        if len(observed_probability) > 0:
            print(
                "Observed probability min/max:",
                float(observed_probability.min()),
                float(observed_probability.max()),
            )

    print("Color legend: unknown=gray, free=blue, occupied=green")

    if show_rays:
        print("Saved ray count:", ray_count)
        print("Ray frame filter:", ray_frame)
        print("Displayed ray count (orange):", displayed_ray_count)
        print("Camera origin color: red")

    geometries.append(make_coordinate_frame())

    window_name = (
        "Object-frame Occupancy Map "
        f"({display_mode})"
    )

    if image_path is not None:
        save_occupancy_open3d_image(
            save_reverse=save_reverse_image,
            geometries=geometries,
            image_path=image_path,
            grid_origin=origin,
            root_size=root_size,
            image_width=image_width,
            image_height=image_height,
            frame_metadata=frame_metadata,
            occupancy_counts={
                "unknown": unknown_count,
                "free": free_count,
                "occupied": occupied_count,
            },
        )

    if show_window:
        o3d.visualization.draw_geometries(
            geometries,
            window_name=window_name,
            width=image_width,
            height=image_height,
        )

    return {
        "occupancy_path": occupancy_path,
        "image_path": image_path,
        "unknown_count": unknown_count,
        "free_count": free_count,
        "occupied_count": occupied_count,
        "displayed_ray_count": displayed_ray_count,
    }


OCCUPANCY_VIEW_DIRECTION = (1.0, -1.2, 1.6)


def _set_fixed_occupancy_view(
    o3d,
    visualizer,
    grid_origin,
    root_size,
    image_width,
    image_height,
    view_direction=None,
):
    """모든 PNG에 동일한 object-frame 시점을 적용해 누적 변화만 보이게 한다.

    view_direction을 뒤집으면 정확히 반대편에서 본 화면이 된다.
    """
    look_at = np.asarray(
        grid_origin,
        dtype=np.float64,
    ) + float(root_size) / 2.0

    if view_direction is None:
        view_direction = OCCUPANCY_VIEW_DIRECTION

    view_direction = np.asarray(
        view_direction,
        dtype=np.float64,
    )
    view_direction /= np.linalg.norm(view_direction)
    camera_position = (
        look_at
        + view_direction * float(root_size) * 2.6
    )
    forward = look_at - camera_position
    forward /= np.linalg.norm(forward)
    world_up = np.asarray(
        [0.0, 0.0, 1.0],
        dtype=np.float64,
    )
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.stack(
        [right, down, forward],
        axis=0,
    )
    extrinsic = np.eye(4, dtype=np.float64)
    extrinsic[:3, :3] = rotation
    extrinsic[:3, 3] = -rotation @ camera_position

    vertical_field_of_view = np.deg2rad(55.0)
    focal_length = image_height / (
        2.0 * np.tan(vertical_field_of_view / 2.0)
    )
    parameters = o3d.camera.PinholeCameraParameters()
    parameters.intrinsic = o3d.camera.PinholeCameraIntrinsic(
        image_width,
        image_height,
        focal_length,
        focal_length,
        (image_width - 1.0) / 2.0,
        (image_height - 1.0) / 2.0,
    )
    parameters.extrinsic = extrinsic
    view_control = visualizer.get_view_control()

    try:
        converted = view_control.convert_from_pinhole_camera_parameters(
            parameters,
            allow_arbitrary=True,
        )
    except TypeError:
        converted = view_control.convert_from_pinhole_camera_parameters(
            parameters
        )

    if converted is False:
        raise RuntimeError(
            "Open3D 고정 시점을 설정하지 못했습니다."
        )


def _annotate_occupancy_image(
    image_path,
    frame_metadata,
    occupancy_counts,
):
    """PNG에 frame, pose 및 Unknown 감소량을 알아보기 쉽게 표시한다."""
    if frame_metadata is None:
        return

    try:
        from PIL import Image, ImageDraw, ImageFont
    except ModuleNotFoundError:
        print(
            "[OCCUPANCY IMAGE] Pillow가 없어 텍스트 표시를 생략합니다."
        )
        return

    with Image.open(image_path) as image:
        annotated = image.convert("RGBA")
        overlay = Image.new(
            "RGBA",
            annotated.size,
            (0, 0, 0, 0),
        )
        draw = ImageDraw.Draw(overlay)

        try:
            title_font = ImageFont.truetype(
                "DejaVuSans.ttf",
                size=24,
            )
            detail_font = ImageFont.truetype(
                "DejaVuSans.ttf",
                size=18,
            )
        except OSError:
            title_font = ImageFont.load_default()
            detail_font = title_font

        frame_id = int(frame_metadata["frame_id"])
        pose_name = frame_metadata.get(
            "pose_name",
            "unknown_pose",
        )
        title = f"Frame {frame_id:03d} | {pose_name}"
        counts = (
            f"Unknown: {occupancy_counts['unknown']:,}    "
            f"Free: {occupancy_counts['free']:,}    "
            f"Occupied: {occupancy_counts['occupied']:,}"
        )
        score = frame_metadata.get(
            "visible_entropy",
            frame_metadata.get("expected_entropy"),
        )
        lines = [title, counts]

        if score is not None:
            lines.append(
                f"NBV visible entropy: {float(score):.3f}"
            )

        banner_height = 32 + len(lines) * 30
        draw.rounded_rectangle(
            [
                16,
                16,
                max(540, annotated.width // 2),
                16 + banner_height,
            ],
            radius=12,
            fill=(18, 25, 35, 205),
        )

        for line_index, line in enumerate(lines):
            draw.text(
                (32, 30 + line_index * 30),
                line,
                fill=(255, 255, 255, 255),
                font=title_font if line_index == 0 else detail_font,
            )

        Image.alpha_composite(
            annotated,
            overlay,
        ).convert("RGB").save(image_path)


def save_occupancy_open3d_image(
    geometries,
    image_path,
    grid_origin,
    root_size,
    image_width=1280,
    image_height=720,
    frame_metadata=None,
    occupancy_counts=None,
    save_reverse=True,
):
    """기존 Open3D geometry를 창을 띄우지 않고 동일 시점 PNG로 저장한다.

    save_reverse=True면 정확히 반대편에서 본 화면도 <이름>_reverse.png로
    함께 저장한다. 한쪽에서만 보면 물체 뒤편 voxel 상태가 가려져
    보이지 않기 때문이다. 같은 렌더 창을 재사용하므로 비용이 작다.
    """
    o3d = get_open3d()
    image_path = os.path.abspath(image_path)
    os.makedirs(
        os.path.dirname(image_path),
        exist_ok=True,
    )
    visualizer = o3d.visualization.Visualizer()

    created = visualizer.create_window(
        window_name="Object-frame Occupancy Export",
        width=int(image_width),
        height=int(image_height),
        visible=False,
    )

    if created is False:
        raise RuntimeError(
            "Open3D 이미지 저장용 렌더링 창을 만들지 못했습니다. "
            "GUI가 연결된 환경에서 실행하세요."
        )

    try:
        for geometry in geometries:
            visualizer.add_geometry(geometry)

        render_option = visualizer.get_render_option()
        render_option.background_color = np.asarray(
            [0.97, 0.97, 0.98],
            dtype=np.float64,
        )
        render_option.point_size = 5.0

        if hasattr(render_option, "mesh_show_back_face"):
            render_option.mesh_show_back_face = True

        stem, extension = os.path.splitext(image_path)
        reverse_path = f"{stem}_reverse{extension}"
        shots = [(image_path, OCCUPANCY_VIEW_DIRECTION)]

        if save_reverse:
            shots.append(
                (
                    reverse_path,
                    tuple(-v for v in OCCUPANCY_VIEW_DIRECTION),
                )
            )

        for target_path, direction in shots:
            _set_fixed_occupancy_view(
                o3d=o3d,
                visualizer=visualizer,
                grid_origin=grid_origin,
                root_size=root_size,
                image_width=int(image_width),
                image_height=int(image_height),
                view_direction=direction,
            )
            visualizer.poll_events()
            visualizer.update_renderer()
            visualizer.capture_screen_image(
                target_path,
                do_render=True,
            )
    finally:
        visualizer.destroy_window()

    saved_paths = []

    for target_path, _ in shots:
        if not os.path.isfile(target_path):
            raise RuntimeError(
                f"Open3D PNG 저장에 실패했습니다: {target_path}"
            )

        _annotate_occupancy_image(
            image_path=target_path,
            frame_metadata=frame_metadata,
            occupancy_counts=occupancy_counts or {},
        )
        saved_paths.append(target_path)
        print("[OCCUPANCY IMAGE] Saved:", target_path)

    return image_path


def _load_nbv_sequence_frames(
    occupancy_path,
    manifest_path=None,
):
    occupancy_dir = os.path.dirname(
        os.path.abspath(occupancy_path)
    )
    manifest_path = (
        os.path.abspath(manifest_path)
        if manifest_path is not None
        else os.path.join(
            occupancy_dir,
            "nbv_sequence_manifest.json",
        )
    )

    if os.path.isfile(manifest_path):
        with open(
            manifest_path,
            "r",
            encoding="utf-8",
        ) as manifest_file:
            manifest = json.load(manifest_file)

        frames = sorted(
            manifest.get("frames", []),
            key=lambda item: int(item["frame_id"]),
        )

        return frames, manifest.get("status", "unknown"), manifest_path

    snapshot_paths = sorted(
        glob.glob(
            os.path.join(
                occupancy_dir,
                "object_occupancy_[0-9][0-9][0-9]"
                "[0-9][0-9][0-9].npz",
            )
        )
    )
    frames = []

    for snapshot_path in snapshot_paths:
        frame_id = int(
            os.path.splitext(
                os.path.basename(snapshot_path)
            )[0].rsplit("_", 1)[1]
        )
        frames.append(
            {
                "frame_id": frame_id,
                "pose_name": f"frame_{frame_id:06d}",
                "occupancy_path": snapshot_path,
            }
        )

    return frames, "unknown", manifest_path


def _save_occupancy_animation(
    image_paths,
    gif_path,
    frame_duration_ms,
):
    if not image_paths:
        raise ValueError("GIF로 저장할 occupancy 이미지가 없습니다.")

    from PIL import Image

    gif_path = os.path.abspath(gif_path)
    os.makedirs(
        os.path.dirname(gif_path),
        exist_ok=True,
    )
    images = []

    for image_path in image_paths:
        with Image.open(image_path) as image:
            images.append(image.convert("RGB"))

    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:],
        duration=int(frame_duration_ms),
        loop=0,
    )

    print("[OCCUPANCY GIF] Saved:", gif_path)
    return gif_path


def save_occupancy_sequence_open3d(
    occupancy_path,
    output_dir=None,
    manifest_path=None,
    display_mode="all",
    crop_margin_voxels=None,
    save_reverse_image=True,
    unknown_clearance=1,
    show_rays=True,
    ray_mode="current",
    ray_frame=None,
    max_rays=200,
    image_width=1280,
    image_height=720,
    watch=False,
    poll_interval=1.0,
    gif_path=None,
    gif_duration_ms=900,
):
    """누적 occupancy snapshot을 순서대로 PNG 및 선택적 GIF로 저장한다."""
    if ray_mode not in {"current", "all", "fixed"}:
        raise ValueError(
            f"지원하지 않는 sequence ray mode: {ray_mode}"
        )

    if ray_mode == "fixed" and ray_frame is None:
        raise ValueError(
            "ray_mode='fixed'에는 --ray-frame 값이 필요합니다."
        )

    poll_interval = float(poll_interval)

    if poll_interval <= 0.0:
        raise ValueError("poll_interval은 0보다 커야 합니다.")

    occupancy_path = os.path.abspath(occupancy_path)
    occupancy_dir = os.path.dirname(occupancy_path)
    output_dir = (
        os.path.abspath(output_dir)
        if output_dir is not None
        else os.path.join(
            occupancy_dir,
            "nbv_occupancy_frames",
        )
    )
    os.makedirs(output_dir, exist_ok=True)

    rendered_frame_ids = set()
    image_paths_by_frame = {}

    while True:
        frames, sequence_status, resolved_manifest_path = (
            _load_nbv_sequence_frames(
                occupancy_path=occupancy_path,
                manifest_path=manifest_path,
            )
        )

        if not frames and not watch:
            raise RuntimeError(
                "저장할 occupancy snapshot이 없습니다. 먼저 "
                "Battery_rb.py를 실행해 NBV 촬영을 수행하세요."
            )

        for frame_metadata in frames:
            frame_id = int(frame_metadata["frame_id"])

            if frame_id in rendered_frame_ids:
                continue

            snapshot_path = frame_metadata["occupancy_path"]

            if not os.path.isabs(snapshot_path):
                snapshot_path = os.path.join(
                    occupancy_dir,
                    snapshot_path,
                )

            if not os.path.isfile(snapshot_path):
                if watch:
                    continue

                raise FileNotFoundError(
                    "NBV manifest에 기록된 occupancy snapshot이 "
                    f"없습니다: {snapshot_path}"
                )

            if ray_mode == "current":
                selected_ray_frame = frame_id
            elif ray_mode == "fixed":
                selected_ray_frame = int(ray_frame)
            else:
                selected_ray_frame = None

            image_path = os.path.join(
                output_dir,
                f"occupancy_{frame_id:06d}.png",
            )

            print(
                "\n[OCCUPANCY SEQUENCE] Rendering frame:",
                frame_id,
                frame_metadata.get("pose_name"),
            )

            visualize_occupancy_map_open3d(
                occupancy_path=snapshot_path,
                display_mode=display_mode,
                crop_margin_voxels=crop_margin_voxels,
                save_reverse_image=save_reverse_image,
                unknown_clearance=unknown_clearance,
                show_rays=show_rays,
                ray_frame=selected_ray_frame,
                max_rays=max_rays,
                image_path=image_path,
                show_window=False,
                image_width=image_width,
                image_height=image_height,
                frame_metadata=frame_metadata,
            )

            rendered_frame_ids.add(frame_id)
            image_paths_by_frame[frame_id] = image_path

        if not watch or sequence_status == "completed":
            break

        if sequence_status == "failed":
            raise RuntimeError(
                "NBV capture sequence가 실패했습니다: "
                f"{resolved_manifest_path}"
            )

        time.sleep(poll_interval)

    image_paths = [
        image_paths_by_frame[frame_id]
        for frame_id in sorted(image_paths_by_frame)
    ]

    if gif_path is not None:
        _save_occupancy_animation(
            image_paths=image_paths,
            gif_path=gif_path,
            frame_duration_ms=gif_duration_ms,
        )

    print(
        "\n[OCCUPANCY SEQUENCE] Saved frames:",
        len(image_paths),
    )
    print("[OCCUPANCY SEQUENCE] Directory:", output_dir)

    return image_paths
