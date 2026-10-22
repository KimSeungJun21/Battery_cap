"""Viewport-only camera frustum visualization for Battery_rb.py."""

import numpy as np
from pxr import Gf, UsdGeom


def _camera_plane_corners(intrinsic_matrix, image_size, distance):
    """Return the four image corners in USD camera-local coordinates."""
    width, height = image_size
    fx = float(intrinsic_matrix[0, 0])
    fy = float(intrinsic_matrix[1, 1])
    cx = float(intrinsic_matrix[0, 2])
    cy = float(intrinsic_matrix[1, 2])

    if fx <= 0.0 or fy <= 0.0:
        raise ValueError("Camera focal lengths must be positive.")

    # USD cameras look along local -Z with +Y pointing upward. Image pixel
    # coordinates point downward, so the local Y component is inverted.
    return [
        (
            (u - cx) * distance / fx,
            (cy - v) * distance / fy,
            -distance,
        )
        for u, v in (
            (0.0, 0.0),
            (float(width), 0.0),
            (float(width), float(height)),
            (0.0, float(height)),
        )
    ]


def _transform_point(matrix, point):
    transformed = matrix.Transform(Gf.Vec3d(*point))
    return (
        float(transformed[0]),
        float(transformed[1]),
        float(transformed[2]),
    )


def _append_dashed_edge(
    start_points,
    end_points,
    start,
    end,
    dash_length,
    gap_length,
):
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    edge = end - start
    edge_length = float(np.linalg.norm(edge))
    if edge_length <= 1e-9:
        return

    direction = edge / edge_length
    offset = 0.0
    while offset < edge_length:
        dash_end = min(offset + dash_length, edge_length)
        start_points.append(tuple(start + direction * offset))
        end_points.append(tuple(start + direction * dash_end))
        offset += dash_length + gap_length


def draw_camera_frustum(
    camera_prim,
    intrinsic_matrix,
    image_size,
    near_distance=0.10,
    far_distance=1.50,
    dash_length=0.04,
    gap_length=0.025,
    color=(0.0, 1.0, 0.4, 1.0),
    line_width=2.0,
):
    """Draw one persistent dotted frustum for a fixed USD camera."""
    if not camera_prim.IsValid() or not camera_prim.IsA(UsdGeom.Camera):
        raise ValueError("camera_prim must be a valid USD Camera prim.")
    if near_distance <= 0.0 or far_distance <= near_distance:
        raise ValueError("Frustum distances must satisfy 0 < near < far.")
    if dash_length <= 0.0 or gap_length < 0.0:
        raise ValueError("Dash length must be positive and gap non-negative.")

    intrinsic_matrix = np.asarray(intrinsic_matrix, dtype=np.float64)
    if intrinsic_matrix.shape != (3, 3):
        raise ValueError("intrinsic_matrix must have shape (3, 3).")

    camera_to_world = (
        UsdGeom.Xformable(camera_prim)
        .ComputeLocalToWorldTransform(0.0)
    )
    origin = _transform_point(camera_to_world, (0.0, 0.0, 0.0))
    near_corners = [
        _transform_point(camera_to_world, point)
        for point in _camera_plane_corners(
            intrinsic_matrix,
            image_size,
            near_distance,
        )
    ]
    far_corners = [
        _transform_point(camera_to_world, point)
        for point in _camera_plane_corners(
            intrinsic_matrix,
            image_size,
            far_distance,
        )
    ]

    edges = []
    edges.extend((origin, corner) for corner in far_corners)
    for corners in (near_corners, far_corners):
        edges.extend(
            (corners[index], corners[(index + 1) % 4])
            for index in range(4)
        )

    start_points = []
    end_points = []
    for start, end in edges:
        _append_dashed_edge(
            start_points=start_points,
            end_points=end_points,
            start=start,
            end=end,
            dash_length=dash_length,
            gap_length=gap_length,
        )

    # Imported lazily so Battery_rb.py can enable the extension first.
    from isaacsim.util.debug_draw import _debug_draw

    draw = _debug_draw.acquire_debug_draw_interface()
    draw.draw_lines(
        start_points,
        end_points,
        [color] * len(start_points),
        [float(line_width)] * len(start_points),
    )
    print(
        "[CAMERA] Fixed Zivid frustum drawn:",
        f"near={near_distance:.2f}m",
        f"far={far_distance:.2f}m",
        f"segments={len(start_points)}",
    )
    return draw
