"""Zivid render product, intrinsic, and capture helpers."""

import os

import numpy as np
import omni.usd
from pxr import UsdGeom

try:
    from rb_ws.PCD_utils import save_zivid_capture
except ModuleNotFoundError:
    from PCD_utils import save_zivid_capture


def create_zivid_render_stack(
    rep_module,
    camera_path,
    resolution,
):
    rep_module.orchestrator.set_capture_on_play(False)

    render_product = rep_module.create.render_product(
        camera_path,
        resolution=resolution,
        name="ZividRenderProduct",
    )

    render_product.hydra_texture.set_updates_enabled(False)

    rgb_annotator = (
        rep_module.AnnotatorRegistry.get_annotator("rgb")
    )
    rgb_annotator.attach(
        render_product
    )

    depth_annotator = (
        rep_module.AnnotatorRegistry.get_annotator(
            "distance_to_image_plane"
        )
    )
    depth_annotator.attach(
        render_product
    )

    semantic_annotator = (
        rep_module.AnnotatorRegistry.get_annotator(
            "semantic_segmentation",
            init_params={
                "colorize": False,
            },
        )
    )
    semantic_annotator.attach(
        render_product
    )

    instance_annotator = (
        rep_module.AnnotatorRegistry.get_annotator(
            "instance_segmentation",
            init_params={
                "colorize": False,
            },
        )
    )
    instance_annotator.attach(
        render_product
    )

    bbox_annotator = (
        rep_module.AnnotatorRegistry.get_annotator(
            "bounding_box_2d_tight"
        )
    )
    bbox_annotator.attach(
        render_product
    )

    print(
        "Zivid RenderProduct initialized:",
        camera_path,
    )

    return (
        render_product,
        rgb_annotator,
        depth_annotator,
        semantic_annotator,
        instance_annotator,
        bbox_annotator,
    )


# camera prim path -> focal length as authored in the USD (before zoom).
_AUTHORED_FOCAL_LENGTHS = {}


def square_pixel_intrinsic(focal_length, horizontal_aperture, resolution):
    """Pinhole K and vertical aperture for the renderer's square pixels."""
    width, height = resolution
    if min(width, height, focal_length, horizontal_aperture) <= 0:
        raise ValueError("camera resolution, focal length and aperture must be positive")
    focal_pixels = width * focal_length / horizontal_aperture
    return (
        np.array([[focal_pixels, 0.0, width / 2.0],
                  [0.0, focal_pixels, height / 2.0],
                  [0.0, 0.0, 1.0]], dtype=np.float64),
        horizontal_aperture * height / width,
    )


def compute_zivid_intrinsic(
    camera_prim,
    resolution,
):
    usd_camera = UsdGeom.Camera(
        camera_prim
    )

    focal_length = (
        usd_camera
        .GetFocalLengthAttr()
        .Get()
    )
    # Simulation-only zoom: scale the authored focal length (narrower FOV,
    # fx and pixel density scale with it). Idempotent per prim so a second
    # call cannot zoom twice. The real Zivid 2+ has a fixed lens.
    zoom = float(os.environ.get("BATTERY_ZIVID_ZOOM", "1.0"))
    prim_key = str(camera_prim.GetPath())
    authored_focal_length = _AUTHORED_FOCAL_LENGTHS.setdefault(
        prim_key, float(focal_length)
    )
    if zoom > 0.0 and not np.isclose(
        focal_length, authored_focal_length * zoom, rtol=1e-9
    ):
        focal_length = authored_focal_length * zoom
        usd_camera.GetFocalLengthAttr().Set(focal_length)
        print(
            "[ZIVID][ZOOM] focal length",
            f"{authored_focal_length:.6f} -> {focal_length:.6f}",
            f"(x{zoom:.3f}, BATTERY_ZIVID_ZOOM)",
        )

    horizontal_aperture = (
        usd_camera
        .GetHorizontalApertureAttr()
        .Get()
    )

    vertical_aperture = (
        usd_camera
        .GetVerticalApertureAttr()
        .Get()
    )

    width, height = resolution

    intrinsic_matrix, effective_vertical_aperture = square_pixel_intrinsic(
        focal_length, horizontal_aperture, resolution,
    )
    # Isaac's pinhole renderer maintains square pixels. The authored sensor
    # aspect ratio can differ from the render-product aspect ratio. Using the
    # stale vertical aperture made fy=2430.6 while rendered fy was 2120.0;
    # reconstructed top points missed the 3 mm surface matching radius.
    # Synchronize the live camera as well as K, before the first capture.
    if not np.isclose(vertical_aperture, effective_vertical_aperture, rtol=1e-6):
        print(
            "[ZIVID][CALIBRATION] vertical aperture synchronized to resolution:",
            f"{vertical_aperture:.6f} -> {effective_vertical_aperture:.6f}",
            "(square pixels, fx=fy)",
        )
        usd_camera.GetVerticalApertureAttr().Set(effective_vertical_aperture)
    vertical_aperture = effective_vertical_aperture

    print("\n========== Zivid Intrinsic ==========")
    print(
        "resolution:",
        width,
        "x",
        height,
    )
    print(
        "focal_length:",
        focal_length,
    )
    print(
        "horizontal_aperture:",
        horizontal_aperture,
    )
    print(
        "vertical_aperture:",
        vertical_aperture,
    )
    print("\nK =")
    print(intrinsic_matrix)
    print("=====================================")

    return intrinsic_matrix


def capture_current_zivid_frame(
    frame_id,
    pose_name,
    camera_prim,
    object_prim,
    zivid_render_product,
    zivid_rgb_annotator,
    zivid_depth_annotator,
    zivid_semantic_annotator,
    zivid_instance_annotator,
    zivid_bbox_annotator,
    orchestrator,
    zivid_save_dir,
    intrinsic_matrix,
    pause_timeline=True,
    voxel_size=0.01,
    object_grid_root_size=0.32,
    object_grid_origin=None,
):
    camera_world_matrix_gf = (
        omni.usd.get_world_transform_matrix(
            camera_prim
        )
    )

    object_world_matrix_gf = (
        omni.usd.get_world_transform_matrix(
            object_prim
        )
    )

    camera_world_matrix = np.asarray(
        camera_world_matrix_gf,
        dtype=np.float64,
    ).T
    object_world_matrix = np.asarray(
        object_world_matrix_gf,
        dtype=np.float64,
    ).T

    print(
        "\n========== Capture Poses =========="
    )
    print(
        "[POSE] Frame:",
        frame_id,
        pose_name,
    )
    print(
        "[POSE] Camera world matrix:"
    )
    print(camera_world_matrix_gf)

    print(
        "[POSE] Target object world matrix:"
    )
    print(object_world_matrix_gf)
    print(
        "===================================\n"
    )

    capture_succeeded = save_zivid_capture(
        frame_id=frame_id,
        zivid_render_product=zivid_render_product,
        zivid_rgb_annotator=zivid_rgb_annotator,
        zivid_depth_annotator=zivid_depth_annotator,
        zivid_semantic_annotator=zivid_semantic_annotator,
        zivid_instance_annotator=zivid_instance_annotator,
        zivid_bbox_annotator=zivid_bbox_annotator,
        orchestrator=orchestrator,
        zivid_save_dir=zivid_save_dir,
        intrinsic_matrix=intrinsic_matrix,
        camera_world_matrix=camera_world_matrix,
        object_world_matrix=object_world_matrix,
        pause_timeline=pause_timeline,
        voxel_size=voxel_size,
        object_grid_root_size=object_grid_root_size,
        object_grid_origin=object_grid_origin,
    )

    if not capture_succeeded:
        raise RuntimeError(
            "Zivid capture가 완료되지 않았습니다: "
            f"frame={frame_id}, pose={pose_name}"
        )

    return (
        camera_world_matrix,
        object_world_matrix,
    )
