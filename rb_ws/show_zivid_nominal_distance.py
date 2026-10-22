"""Isaac Sim Script Editor helper for the 450 mm reference distance.

Run this file from Isaac Sim's Script Editor while Battery_rb.py is active:

    exec(open(
        "/home/seungjun/isaac_ws/rb_ws/show_zivid_nominal_distance.py"
    ).read())

After loading it, change the distance interactively in metres:

    draw_zivid_nominal_distance(0.30)
    draw_zivid_nominal_distance(0.45)
    draw_zivid_nominal_distance(0.60)

The geometry is drawn only in the viewport through Isaac Sim debug draw.  It
does not become part of the USD stage and is not included in Zivid captures.
"""

import math

import omni.usd
from pxr import Gf, Usd, UsdGeom
from isaacsim.util.debug_draw import _debug_draw


CAMERA_PATH = (
    "/World/Battery_env/Zivid_camera_01/Geometry/zivid_base_link/"
    "zivid_optical_frame/Zivid_Sensor"
)
TARGET_PATHS = (
    "/World/BatteryCap",
    "/World/MustardBottle",
)

# Viewport projection reference. The runtime defect-recall GSD is configured
# separately in Battery_defect_config.py.
REFERENCE_DISTANCE_M = 0.450
REFERENCE_FX_PX = 2120.0
REFERENCE_FY_PX = 2430.6
IMAGE_WIDTH_PX = 2448
IMAGE_HEIGHT_PX = 2048


def _world_point(camera_to_world, camera_point):
    point = camera_to_world.Transform(Gf.Vec3d(*camera_point))
    return (float(point[0]), float(point[1]), float(point[2]))


def _add_segment(starts, ends, colors, widths, start, end, color, width):
    starts.append(tuple(start))
    ends.append(tuple(end))
    colors.append(tuple(color))
    widths.append(float(width))


def _add_cross(
    starts,
    ends,
    colors,
    widths,
    camera_to_world,
    center_camera,
    radius,
    color,
    width,
):
    cx, cy, cz = center_camera
    for start, end in (
        ((cx - radius, cy, cz), (cx + radius, cy, cz)),
        ((cx, cy - radius, cz), (cx, cy + radius, cz)),
    ):
        _add_segment(
            starts,
            ends,
            colors,
            widths,
            _world_point(camera_to_world, start),
            _world_point(camera_to_world, end),
            color,
            width,
        )


def _target_bbox_center(stage):
    for path in TARGET_PATHS:
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid() or not prim.IsActive():
            continue

        bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            includedPurposes=[UsdGeom.Tokens.default_],
            useExtentsHint=True,
        )
        bbox_range = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
        minimum = bbox_range.GetMin()
        maximum = bbox_range.GetMax()
        center = Gf.Vec3d(
            0.5 * (float(minimum[0]) + float(maximum[0])),
            0.5 * (float(minimum[1]) + float(maximum[1])),
            0.5 * (float(minimum[2]) + float(maximum[2])),
        )
        return path, center

    return None, None


def draw_zivid_nominal_distance(
    distance_m=REFERENCE_DISTANCE_M,
    clear_existing=True,
):
    """Draw the reference image plane at an adjustable optical distance.

    Args:
        distance_m: Camera optical-axis distance in metres.
        clear_existing: Clear viewport debug lines before drawing.  This also
            clears other debug-draw lines, such as the startup frustum.
    """
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("열려 있는 USD stage가 없습니다.")

    camera_prim = stage.GetPrimAtPath(CAMERA_PATH)
    if not camera_prim.IsValid() or not camera_prim.IsA(UsdGeom.Camera):
        raise RuntimeError(f"Zivid camera prim을 찾지 못했습니다: {CAMERA_PATH}")

    camera_to_world = UsdGeom.Xformable(
        camera_prim
    ).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    world_to_camera = camera_to_world.GetInverse()

    distance = float(distance_m)
    if not 0.05 <= distance <= 3.0:
        raise ValueError(
            "distance_m은 0.05~3.0 m 범위로 지정하세요: "
            f"{distance}"
        )
    cx = 0.5 * IMAGE_WIDTH_PX
    cy = 0.5 * IMAGE_HEIGHT_PX

    # USD camera: +X image right, +Y image up, -Z camera forward.
    corners_camera = [
        (
            (u - cx) * distance / REFERENCE_FX_PX,
            (cy - v) * distance / REFERENCE_FY_PX,
            -distance,
        )
        for u, v in (
            (0.0, 0.0),
            (float(IMAGE_WIDTH_PX), 0.0),
            (float(IMAGE_WIDTH_PX), float(IMAGE_HEIGHT_PX)),
            (0.0, float(IMAGE_HEIGHT_PX)),
        )
    ]
    corners_world = [
        _world_point(camera_to_world, point) for point in corners_camera
    ]
    camera_origin_world = _world_point(camera_to_world, (0.0, 0.0, 0.0))
    nominal_center_world = _world_point(
        camera_to_world,
        (0.0, 0.0, -distance),
    )

    starts = []
    ends = []
    colors = []
    widths = []

    cyan = (0.0, 0.9, 1.0, 1.0)
    yellow = (1.0, 0.85, 0.0, 1.0)
    white = (1.0, 1.0, 1.0, 0.9)
    green = (0.1, 1.0, 0.2, 1.0)

    # Cyan rectangle: complete Zivid image plane at the selected distance.
    for index in range(4):
        _add_segment(
            starts,
            ends,
            colors,
            widths,
            corners_world[index],
            corners_world[(index + 1) % 4],
            cyan,
            4.0,
        )

    # Yellow optical axis and center marker.
    _add_segment(
        starts,
        ends,
        colors,
        widths,
        camera_origin_world,
        nominal_center_world,
        yellow,
        4.0,
    )
    _add_cross(
        starts,
        ends,
        colors,
        widths,
        camera_to_world,
        (0.0, 0.0, -distance),
        0.025,
        yellow,
        5.0,
    )

    # White ticks every 10 cm; the longer final tick is the selected distance.
    tick_distances = [
        0.1 * index
        for index in range(1, int(distance / 0.1) + 1)
        if 0.1 * index < distance - 1e-6
    ]
    tick_distances.append(distance)
    for tick_distance in tick_distances:
        tick_radius = 0.018 if tick_distance == distance else 0.010
        start = _world_point(
            camera_to_world,
            (-tick_radius, 0.0, -tick_distance),
        )
        end = _world_point(
            camera_to_world,
            (tick_radius, 0.0, -tick_distance),
        )
        _add_segment(
            starts,
            ends,
            colors,
            widths,
            start,
            end,
            white,
            2.0,
        )

    target_path, target_center_world = _target_bbox_center(stage)
    target_report = "활성 batterycap/mustard를 찾지 못했습니다."

    if target_center_world is not None:
        target_camera = world_to_camera.Transform(target_center_world)
        tx = float(target_camera[0])
        ty = float(target_camera[1])
        tz = float(target_camera[2])
        optical_depth = -tz
        straight_distance = math.sqrt(tx * tx + ty * ty + tz * tz)

        _add_segment(
            starts,
            ends,
            colors,
            widths,
            camera_origin_world,
            tuple(float(value) for value in target_center_world),
            green,
            2.5,
        )
        _add_cross(
            starts,
            ends,
            colors,
            widths,
            camera_to_world,
            (tx, ty, tz),
            0.018,
            green,
            5.0,
        )

        if optical_depth > 0.0:
            pixel_u = REFERENCE_FX_PX * tx / optical_depth + cx
            pixel_v = cy - REFERENCE_FY_PX * ty / optical_depth
            inside = (
                0.0 <= pixel_u < IMAGE_WIDTH_PX
                and 0.0 <= pixel_v < IMAGE_HEIGHT_PX
            )
            gsd_x = 1000.0 * optical_depth / REFERENCE_FX_PX
            gsd_y = 1000.0 * optical_depth / REFERENCE_FY_PX
            target_report = (
                f"{target_path}\n"
                f"  광축 깊이 Z       : {optical_depth:.4f} m\n"
                f"  직선거리          : {straight_distance:.4f} m\n"
                f"  기준 영상 투영점  : ({pixel_u:.1f}, {pixel_v:.1f}) px\n"
                f"  화면 중심 포함    : {inside}\n"
                f"  중심 깊이 GSD     : X={gsd_x:.3f}, Y={gsd_y:.3f} mm/px"
            )
        else:
            target_report = (
                f"{target_path}: 카메라 뒤쪽에 있습니다 "
                f"(camera-local Z={tz:.4f} m)."
            )

    draw = _debug_draw.acquire_debug_draw_interface()
    if clear_existing:
        draw.clear_lines()
    draw.draw_lines(starts, ends, colors, widths)

    plane_width = distance * IMAGE_WIDTH_PX / REFERENCE_FX_PX
    plane_height = distance * IMAGE_HEIGHT_PX / REFERENCE_FY_PX
    print("\n========== Zivid 기준 거리 시각화 ==========")
    print(f"카메라             : {CAMERA_PATH}")
    print(f"공칭 광축 거리     : {distance:.3f} m")
    print(
        "공칭 거리 영상면   : "
        f"{plane_width * 1000.0:.1f} x {plane_height * 1000.0:.1f} mm"
    )
    print(
        "색상                : 청록=기준 영상면, "
        "노랑=광축, 초록=물체 중심"
    )
    print("현재 물체:")
    print(target_report)
    print(
        "제거 방법          : Script Editor에서 "
        "_debug_draw.acquire_debug_draw_interface().clear_lines()"
    )
    print("====================================================\n")


draw_zivid_nominal_distance(REFERENCE_DISTANCE_M)
