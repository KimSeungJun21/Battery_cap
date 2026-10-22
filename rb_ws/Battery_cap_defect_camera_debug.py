#!/usr/bin/env python3
"""로봇팔 없이 battery cap의 결함을 카메라 정면에서 확대 촬영한다.

각 결함의 object-frame 위치/법선을 이용해 결함을 Zivid 광축 중앙으로
옮기고 정면을 향하게 한 뒤 다음 파일을 저장한다.

    <output>/batterycap_defect_camera_YYYYmmdd_HHMMSS/
        01_<defect>/raw.png
        01_<defect>/bbox.png
        01_<defect>/crop.png
        overview.png
        captures.json

물체 자세만 직접 바꾸며 RB5, ROS, suction gripper, PhysX를 초기화하거나
사용하지 않는다. bbox는 semantic GT의 실제 결함 픽셀에서 계산하므로
물체 전체 bbox가 아니라 결함 자체에 맞는다.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime


DEFAULT_OUTPUT_ROOT = "/home/seungjun/isaac_ws/seungjun"


def parse_options():
    parser = argparse.ArgumentParser(
        description="Capture close, front-facing battery-cap defect images."
    )
    parser.add_argument(
        "--output-root",
        default=DEFAULT_OUTPUT_ROOT,
        help="촬영 폴더를 만들 상위 경로",
    )
    parser.add_argument(
        "--distance",
        type=float,
        default=0.22,
        help="결함 표면부터 카메라까지 거리(m), 기본 0.22",
    )
    parser.add_argument(
        "--zoom",
        type=float,
        default=1.4,
        help="시뮬레이션 카메라 focal-length 배율, 기본 1.4",
    )
    parser.add_argument(
        "--bbox-padding",
        type=int,
        default=24,
        help="실제 결함 bbox 둘레 표시 여백(px), 기본 24",
    )
    parser.add_argument(
        "--crop-size",
        type=int,
        default=1000,
        help="저장할 결함 확대 crop의 긴 변(px), 기본 1000",
    )
    parser.add_argument(
        "--settle-frames",
        type=int,
        default=6,
        help="물체 자세 변경 후 기다릴 렌더 update 수",
    )
    parser.add_argument(
        "--debug-visible",
        action="store_true",
        help="결함을 형광/확대 표시(위치 확인용; 실제 외관 평가에는 사용 금지)",
    )
    options, remaining = parser.parse_known_args()

    if not 0.08 <= options.distance <= 1.0:
        parser.error("--distance는 0.08~1.0 m 범위여야 합니다.")
    if not 0.2 <= options.zoom <= 5.0:
        parser.error("--zoom은 0.2~5.0 범위여야 합니다.")
    if options.bbox_padding < 0:
        parser.error("--bbox-padding은 0 이상이어야 합니다.")
    if options.crop_size < 128:
        parser.error("--crop-size는 128 이상이어야 합니다.")

    # Kit가 알지 못하는 이 스크립트 전용 인자를 제거한다.
    sys.argv = [sys.argv[0], *remaining]
    return options


OPTIONS = parse_options()
RUN_STAMP = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
RUN_DIR = os.path.abspath(
    os.path.join(
        os.path.expanduser(OPTIONS.output_root),
        f"batterycap_defect_camera_{RUN_STAMP}",
    )
)
os.makedirs(RUN_DIR, exist_ok=False)

# 아래 모듈들은 import 시 환경변수를 읽으므로 SimulationApp보다 먼저 둔다.
os.environ["BATTERY_TARGET_OBJECT"] = "batterycap"
os.environ["BATTERY_ZIVID_ZOOM"] = str(OPTIONS.zoom)
os.environ["BATTERY_DEFECT_DEBUG_VISIBLE"] = (
    "1" if OPTIONS.debug_visible else "0"
)
os.environ["BATTERY_DEFECT_TEXTURE_DIR"] = os.path.join(
    RUN_DIR, "defect_textures"
)
os.environ["BATTERY_DEFECT_GT_DIR"] = os.path.join(RUN_DIR, "defect_gt")

from isaacsim import SimulationApp


simulation_app = SimulationApp(
    {"headless": False},
    experience="/home/seungjun/isaacsim/apps/isaacsim.exp.full.kit",
)

import numpy as np
import omni.replicator.core as rep
import omni.usd
from PIL import Image, ImageDraw, ImageFont, ImageOps
from pxr import Gf, Usd, UsdGeom

from Battery_config import (
    RB_PRIM_PATH,
    RB_USD_PATH,
    STAGE_LIGHT_DIM_FACTOR,
    ZIVID_AMBIENT_INTENSITY,
    ZIVID_CAMERA_PATH,
    ZIVID_PROJECTOR_BASELINE_M,
    ZIVID_PROJECTOR_COLOR_TEMPERATURE,
    ZIVID_PROJECTOR_ENABLE,
    ZIVID_PROJECTOR_HEIGHT_M,
    ZIVID_PROJECTOR_INTENSITY,
    ZIVID_PROJECTOR_PRIM_NAME,
    ZIVID_PROJECTOR_WIDTH_M,
    ZIVID_RESOLUTION,
    get_target_object_config,
)
from Battery_defect_utils import setup_defects
from Battery_lighting_utils import setup_inspection_lighting
from Battery_stage_setup_utils import open_stage_or_raise, update_simulation_frames
from Battery_target_utils import (
    add_referenced_target_prim,
    add_target_labels,
    set_target_pose,
)
from Battery_zivid_setup_utils import compute_zivid_intrinsic


FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
BOX_TARGET = (255, 205, 0)
BOX_OTHER = (0, 225, 255)
BOX_WIDTH = 8
MIN_CROP_SIDE = 300
CROP_CONTEXT_SCALE = 4.0


def safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "defect"


def load_font(size):
    try:
        return ImageFont.truetype(FONT_PATH, size=size)
    except OSError:
        return ImageFont.load_default()


def matrix_from_conventional(transform):
    """열벡터 4x4 numpy 행렬을 USD의 행벡터 Gf 행렬로 바꾼다."""
    values = np.asarray(transform, dtype=np.float64).T.reshape(-1)
    return Gf.Matrix4d(*[float(value) for value in values])


def normalize(vector):
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError("길이가 0인 방향 벡터입니다.")
    return vector / norm


def make_defect_camera_transform(camera_prim, record, distance):
    """결함 tangent/bitangent/normal을 camera +X/+Y/+Z에 정렬한다."""
    camera_to_world = np.asarray(
        omni.usd.get_world_transform_matrix(camera_prim), dtype=np.float64
    ).T
    camera_rotation = camera_to_world[:3, :3]

    object_frame = record["object_frame"]
    tangent = normalize(object_frame["tangent"])
    bitangent = normalize(object_frame["bitangent"])
    normal = normalize(object_frame["normal"])
    source_basis = np.column_stack((tangent, bitangent, normal))

    # 카메라는 로컬 -Z를 바라보므로 표면의 바깥 법선은 카메라 쪽인
    # 로컬 +Z를 향해야 한다. tangent는 영상의 오른쪽, bitangent는 위쪽.
    desired_basis = camera_rotation @ np.eye(3, dtype=np.float64)
    object_rotation = desired_basis @ source_basis.T
    if np.linalg.det(object_rotation) < 0.999:
        raise RuntimeError(
            f"회전행렬이 올바르지 않습니다: det={np.linalg.det(object_rotation)}"
        )

    surface_point = np.asarray(object_frame["surface_point"], dtype=np.float64)
    desired_surface_world = camera_to_world @ np.array(
        [0.0, 0.0, -float(distance), 1.0], dtype=np.float64
    )
    translation = desired_surface_world[:3] - object_rotation @ surface_point

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = object_rotation
    transform[:3, 3] = translation
    return transform


def set_target_transform(target_prim, transform):
    xformable = UsdGeom.Xformable(target_prim)
    xformable.ClearXformOpOrder()
    xformable.AddTransformOp(
        precision=UsdGeom.XformOp.PrecisionDouble
    ).Set(matrix_from_conventional(transform))


def defect_semantic_ids(semantic_info):
    """semantic id -> label metadata에서 결함 이름별 ID를 복원한다."""
    mapping = {}
    for semantic_id, labels in semantic_info.get("idToLabels", {}).items():
        if not isinstance(labels, dict):
            continue
        classes = [
            item.strip()
            for item in str(labels.get("class", "")).split(",")
            if item.strip()
        ]
        if "defect" not in classes:
            continue
        objects = [
            item.strip()
            for item in str(labels.get("object", "")).split(",")
            if item.strip()
        ]
        name = None
        if len(classes) == len(objects):
            name = objects[classes.index("defect")]
        elif len(objects) == 1:
            name = objects[0]
        if name:
            mapping[name] = int(semantic_id)
    return mapping


def mask_box(mask):
    rows, columns = np.nonzero(mask)
    if len(rows) == 0:
        return None
    return (
        int(columns.min()),
        int(rows.min()),
        int(columns.max()) + 1,
        int(rows.max()) + 1,
    )


def expand_box(box, padding, width, height, minimum_side=0):
    x0, y0, x1, y1 = box
    center_x = 0.5 * (x0 + x1)
    center_y = 0.5 * (y0 + y1)
    side_x = max(float(minimum_side), float(x1 - x0 + 2 * padding))
    side_y = max(float(minimum_side), float(y1 - y0 + 2 * padding))
    x0 = max(0, int(round(center_x - 0.5 * side_x)))
    y0 = max(0, int(round(center_y - 0.5 * side_y)))
    x1 = min(width, int(round(center_x + 0.5 * side_x)))
    y1 = min(height, int(round(center_y + 0.5 * side_y)))
    return x0, y0, x1, y1


def crop_box_for_detail(box, width, height):
    defect_side = max(box[2] - box[0], box[3] - box[1])
    side = max(MIN_CROP_SIDE, int(round(defect_side * CROP_CONTEXT_SCALE)))
    return expand_box(box, 0, width, height, minimum_side=side)


def draw_box_label(draw, box, label, color, font):
    x0, y0, x1, y1 = box
    draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline=color, width=BOX_WIDTH)
    text_box = draw.textbbox((0, 0), label, font=font, stroke_width=1)
    text_width = text_box[2] - text_box[0]
    text_height = text_box[3] - text_box[1]
    label_y = y0 - text_height - 16
    if label_y < 0:
        label_y = min(y1 + 8, max(0, draw._image.size[1] - text_height - 12))
    background = (
        max(0, x0),
        max(0, label_y),
        min(draw._image.size[0], x0 + text_width + 18),
        min(draw._image.size[1], label_y + text_height + 12),
    )
    draw.rectangle(background, fill=(0, 0, 0))
    draw.text(
        (background[0] + 9, background[1] + 4),
        label,
        fill=color,
        font=font,
        stroke_width=1,
        stroke_fill=(0, 0, 0),
    )


def save_capture(rgb, semantic, semantic_info, target_name, view_dir):
    if rgb.ndim == 3 and rgb.shape[2] == 4:
        rgb = rgb[:, :, :3]
    rgb = np.asarray(rgb, dtype=np.uint8)
    semantic = np.asarray(semantic).squeeze()
    height, width = semantic.shape
    image = Image.fromarray(rgb)
    image.save(os.path.join(view_dir, "raw.png"), compress_level=2)

    ids = defect_semantic_ids(semantic_info)
    observations = []
    for name, semantic_id in sorted(ids.items()):
        mask = semantic == semantic_id
        box = mask_box(mask)
        if box is None:
            continue
        observations.append(
            {
                "name": name,
                "semantic_id": semantic_id,
                "pixels": int(np.count_nonzero(mask)),
                "mask_bbox_xyxy": list(box),
                "is_target": name == target_name,
            }
        )

    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    font = load_font(34)
    for item in sorted(observations, key=lambda value: not value["is_target"]):
        color = BOX_TARGET if item["is_target"] else BOX_OTHER
        display_box = expand_box(
            item["mask_bbox_xyxy"], OPTIONS.bbox_padding, width, height
        )
        item["display_bbox_xyxy"] = list(display_box)
        label = f"TARGET: {item['name']}" if item["is_target"] else item["name"]
        draw_box_label(draw, display_box, label, color, font)

    annotated.save(os.path.join(view_dir, "bbox.png"), compress_level=2)
    target = next((item for item in observations if item["is_target"]), None)
    if target is not None:
        detail_box = crop_box_for_detail(target["mask_bbox_xyxy"], width, height)
        detail = image.crop(detail_box)
        detail = ImageOps.contain(
            detail,
            (OPTIONS.crop_size, OPTIONS.crop_size),
            method=Image.Resampling.LANCZOS,
        )
        detail_draw = ImageDraw.Draw(detail)
        detail_draw.rectangle(
            (2, 2, detail.width - 3, detail.height - 3),
            outline=BOX_TARGET,
            width=6,
        )
        detail.save(os.path.join(view_dir, "crop.png"), compress_level=2)
        target["crop_bbox_xyxy"] = list(detail_box)
    return observations, target is not None


def make_overview(image_paths, output_path):
    if not image_paths:
        return
    thumb_size = (560, 470)
    columns = 2
    rows = (len(image_paths) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * thumb_size[0], rows * thumb_size[1]), "#202020")
    title_font = load_font(20)
    for index, (label, path) in enumerate(image_paths):
        image = Image.open(path).convert("RGB")
        thumb = ImageOps.contain(image, (540, 410), method=Image.Resampling.LANCZOS)
        x = (index % columns) * thumb_size[0] + (thumb_size[0] - thumb.width) // 2
        y = (index // columns) * thumb_size[1] + 48
        sheet.paste(thumb, (x, y))
        short_label = label.replace("batterycap_", "")
        if len(short_label) > 43:
            short_label = short_label[:40] + "..."
        ImageDraw.Draw(sheet).text(
            ((index % columns) * thumb_size[0] + 12, (index // columns) * thumb_size[1] + 10),
            short_label,
            fill=BOX_TARGET,
            font=title_font,
        )
    sheet.save(output_path, compress_level=2)


def main():
    print(f"[DEFECT CAMERA] output: {RUN_DIR}")
    stage = open_stage_or_raise(simulation_app=simulation_app, usd_path=RB_USD_PATH)
    update_simulation_frames(simulation_app=simulation_app, update_count=20)

    # 환경 USD 안의 카메라/조명만 쓰고 RB5 전체는 렌더와 실행에서 제외한다.
    robot_prim = stage.GetPrimAtPath(RB_PRIM_PATH)
    if robot_prim.IsValid():
        robot_prim.SetActive(False)
        print(f"[DEFECT CAMERA] robot disabled: {RB_PRIM_PATH}")

    camera_prim = stage.GetPrimAtPath(ZIVID_CAMERA_PATH)
    if not camera_prim.IsValid() or not camera_prim.IsA(UsdGeom.Camera):
        raise RuntimeError(f"Zivid camera를 찾지 못했습니다: {ZIVID_CAMERA_PATH}")

    setup_inspection_lighting(
        stage=stage,
        camera_prim_path=ZIVID_CAMERA_PATH,
        projector_enable=ZIVID_PROJECTOR_ENABLE,
        prim_name=ZIVID_PROJECTOR_PRIM_NAME,
        baseline_m=ZIVID_PROJECTOR_BASELINE_M,
        width_m=ZIVID_PROJECTOR_WIDTH_M,
        height_m=ZIVID_PROJECTOR_HEIGHT_M,
        intensity=ZIVID_PROJECTOR_INTENSITY,
        color_temperature=ZIVID_PROJECTOR_COLOR_TEMPERATURE,
        ambient_intensity=ZIVID_AMBIENT_INTENSITY,
        dim_factor=STAGE_LIGHT_DIM_FACTOR,
    )

    target_config = get_target_object_config(assets_root_path=None)
    target_prim = add_referenced_target_prim(
        stage=stage,
        simulation_app=simulation_app,
        usd_path=target_config.usd_path,
        prim_path=target_config.prim_path,
        description=target_config.description,
    )
    set_target_pose(
        target_prim=target_prim,
        position=target_config.position,
        rotation_xyz_deg=target_config.neutral_rotation,
        scale=target_config.scale,
        log_prefix="[DEFECT CAMERA]",
    )
    add_target_labels(
        stage=stage,
        target_prim=target_prim,
        target_prim_path=target_config.prim_path,
        class_label="target_object",
        object_label=target_config.object_label,
    )
    records = setup_defects(
        stage=stage,
        target_prim_path=target_config.prim_path,
        simulation_app=simulation_app,
        target_object_type=target_config.key,
        long_axis_index=target_config.defect_long_axis_index,
        theta_zero_axis_index=target_config.defect_theta_zero_axis_index,
        theta_ninety_axis_index=target_config.defect_theta_ninety_axis_index,
        log_prefix="[DEFECT CAMERA][DEFECT]",
    )
    if not records:
        raise RuntimeError("촬영할 battery cap 결함이 없습니다.")

    rep.orchestrator.set_capture_on_play(False)
    render_product = rep.create.render_product(
        ZIVID_CAMERA_PATH,
        resolution=ZIVID_RESOLUTION,
        name="BatteryCapDefectCloseup",
    )
    render_product.hydra_texture.set_updates_enabled(False)
    rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
    semantic_annotator = rep.AnnotatorRegistry.get_annotator(
        "semantic_segmentation", init_params={"colorize": False}
    )
    rgb_annotator.attach(render_product)
    semantic_annotator.attach(render_product)
    intrinsic = compute_zivid_intrinsic(camera_prim, ZIVID_RESOLUTION)
    np.save(os.path.join(RUN_DIR, "intrinsic.npy"), intrinsic)

    captures = []
    overview_paths = []
    for index, record in enumerate(records, start=1):
        name = record["name"]
        view_name = f"{index:02d}_{safe_name(name)}"
        view_dir = os.path.join(RUN_DIR, view_name)
        os.makedirs(view_dir, exist_ok=False)
        transform = make_defect_camera_transform(camera_prim, record, OPTIONS.distance)
        set_target_transform(target_prim, transform)
        for _ in range(OPTIONS.settle_frames):
            simulation_app.update()

        render_product.hydra_texture.set_updates_enabled(True)
        try:
            rep.orchestrator.step(rt_subframes=4, pause_timeline=True, delta_time=0.0)
            rgb = np.asarray(rgb_annotator.get_data())
            semantic_data = semantic_annotator.get_data()
        finally:
            render_product.hydra_texture.set_updates_enabled(False)

        if rgb.size == 0 or semantic_data is None:
            raise RuntimeError(f"렌더 annotator 데이터가 없습니다: {name}")
        observations, target_visible = save_capture(
            rgb=rgb,
            semantic=semantic_data["data"],
            semantic_info=semantic_data.get("info", {}),
            target_name=name,
            view_dir=view_dir,
        )
        captures.append(
            {
                "index": index,
                "target_defect": name,
                "target_visible": target_visible,
                "distance_m": OPTIONS.distance,
                "object_to_world": transform.tolist(),
                "observations": observations,
                "directory": view_name,
            }
        )
        overview_paths.append((f"{index:02d}  {name}", os.path.join(view_dir, "bbox.png")))
        print(
            f"[DEFECT CAMERA] {index:02d}/{len(records):02d} {name}: "
            f"target_visible={target_visible}, observed={len(observations)}"
        )

    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "output_dir": RUN_DIR,
        "robot_arm_used": False,
        "target": "batterycap",
        "resolution": list(ZIVID_RESOLUTION),
        "distance_m": OPTIONS.distance,
        "zoom": OPTIONS.zoom,
        "bbox_padding_px": OPTIONS.bbox_padding,
        "debug_visible": OPTIONS.debug_visible,
        "captures": captures,
    }
    with open(os.path.join(RUN_DIR, "captures.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    make_overview(overview_paths, os.path.join(RUN_DIR, "overview.png"))

    missing = [item["target_defect"] for item in captures if not item["target_visible"]]
    print(f"[DEFECT CAMERA] 완료: {RUN_DIR}")
    print(f"[DEFECT CAMERA] 결함별 촬영 {len(captures)}장, 미관측 {len(missing)}개")
    if missing:
        print("[DEFECT CAMERA][WARN] 미관측:", ", ".join(missing))


try:
    main()
finally:
    simulation_app.close()
