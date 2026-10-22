"""Zivid annotator capture pipeline and segmentation helpers."""

import json
import os
import re

import numpy as np
from PIL import Image

# Lossless PNG: lower compression reduces CPU time, not pixel quality.
PNG_COMPRESSION_LEVEL = min(9, max(0, int(
    os.environ.get("BATTERY_PNG_COMPRESSION_LEVEL", "6")
)))

try:
    from rb_ws.Battery_timing_utils import StageTimer
except ModuleNotFoundError:
    from Battery_timing_utils import StageTimer

try:
    from rb_ws.Occupancy_utils import update_and_save_object_occupancy_map
    from rb_ws.Octree_utils import (
        create_octree_voxels,
        labels_to_colors,
        save_octree_voxels,
    )
    from rb_ws.Octree_visualization_utils import visualize_octree_voxels_open3d
    from rb_ws.Open3D_utils import visualize_voxels_open3d
    from rb_ws.Pointcloud_processing_utils import (
        depth_to_pointcloud,
        pointcloud_to_voxels,
        save_pointcloud_ply,
    )
except ModuleNotFoundError:
    from Occupancy_utils import update_and_save_object_occupancy_map
    from Octree_utils import (
        create_octree_voxels,
        labels_to_colors,
        save_octree_voxels,
    )
    from Octree_visualization_utils import visualize_octree_voxels_open3d
    from Open3D_utils import visualize_voxels_open3d
    from Pointcloud_processing_utils import (
        depth_to_pointcloud,
        pointcloud_to_voxels,
        save_pointcloud_ply,
    )


# 물체 표면으로 취급할 semantic class들.
#
# defect decal은 별도 prim이라 class=defect로 라벨링되지만, 물리적으로는
# 물체 표면 그 자체다(표면보다 0.2mm 앞에 떠 있을 뿐). 배경으로 떨어뜨리면
# 흠집 자리의 voxel이 occupied 증거를 못 받아 영원히 UNKNOWN으로 남는다.
TARGET_OBJECT_SEMANTIC_LABEL = "target_object"

# 시각화/디버그용 산출물을 저장할지.
#
# NBV 루프가 실제로 읽는 파일은 object_occupancy_latest.npz 하나뿐이다.
# pcd/voxels/octree PLY와 전체 장면 옥트리는 시각화 전용인데, 실측상
# 캡처 5.43초 중 약 4.1초를 차지한다 (전체 장면 옥트리 2.29초,
# PLY 3종 1.5초). 시간 예산이 빠듯할 때 끈다.
SAVE_DEBUG_ARTIFACTS = (
    os.environ.get(
        "BATTERY_SAVE_DEBUG_ARTIFACTS",
        "1",
    )
    .strip()
    .lower()
    not in (
        "0",
        "false",
        "no",
        "off",
    )
)

try:
    from Battery_defect_config import (
        DEFECT_CLASS_LABEL as DEFECT_SEMANTIC_LABEL,
    )
except ImportError:
    DEFECT_SEMANTIC_LABEL = "defect"

# 프레임 RGB/depth 원본을 저장할지.
#
# 2448x2048 PNG 쓰기 + 20MB depth npy 쓰기에 프레임당 0.62초가 든다
# (실측). NBV 루프는 둘 다 안 읽는다. 결함 검출 평가용 데이터가
# 필요할 때만 켠다.
SAVE_FRAME_IMAGES = (
    os.environ.get(
        "BATTERY_SAVE_FRAME_IMAGES",
        "1",
    )
    .strip()
    .lower()
    not in (
        "0",
        "false",
        "no",
        "off",
    )
)

# 직전 save_zivid_capture()의 구간별 소요 시간 (초).
#
# render는 Isaac의 Replicator가 쓰는 시간이라 실기 Zivid에는 없는
# 비용이다. 검사 시간 예산에서 이 몫을 빼고 보려면 호출한 쪽이
# 이 값을 읽는다. summary()가 돌려주는 dict와 같은 키를 갖는다.
LAST_CAPTURE_SPANS = {}

# 직전 target 캡처의 조밀한 object-frame 표면점과 카메라 원점.
#
# 10 mm occupancy ray history는 같은 voxel의 점을 한 개로 줄이므로 2 mm
# 두께의 batterycap 앞/뒷면을 구분할 수 없다. arm NBV runner가 이 원본
# 관측을 즉시 directional surfel model에 대응시킨다. 디스크 저장용이
# 아니라 같은 프로세스 안에서 다음 NBV 계산까지 유지하는 버퍼다.
LAST_TARGET_SURFACE_OBSERVATION = {}


SURFACE_SEMANTIC_LABELS = (
    TARGET_OBJECT_SEMANTIC_LABEL,
    DEFECT_SEMANTIC_LABEL,
)


PCD_ROI_CROP_LEFT = int(
    os.environ.get("BATTERY_PCD_ROI_CROP_LEFT", "500")
)
PCD_ROI_CROP_RIGHT = int(
    os.environ.get("BATTERY_PCD_ROI_CROP_RIGHT", "500")
)
PCD_ROI_CROP_BOTTOM = int(
    os.environ.get("BATTERY_PCD_ROI_CROP_BOTTOM", "500")
)
PCD_ROI_CROP_TOP = int(
    os.environ.get("BATTERY_PCD_ROI_CROP_TOP", "0")
)


def create_pcd_roi_mask(
    image_shape,
    crop_left=None,
    crop_right=None,
    crop_bottom=None,
    crop_top=None,
):
    """
    전체 이미지에서 PCD로 사용할 pixel 영역을 정의한다.

    기본 설정:
        왼쪽   500 px 제거
        오른쪽 500 px 제거
        하단   500 px 제거
        상단은 유지

    return:
        roi_mask: (H, W) bool
    """

    height, width = image_shape

    crop_left = (
        PCD_ROI_CROP_LEFT if crop_left is None else int(crop_left)
    )
    crop_right = (
        PCD_ROI_CROP_RIGHT if crop_right is None else int(crop_right)
    )
    crop_bottom = (
        PCD_ROI_CROP_BOTTOM if crop_bottom is None else int(crop_bottom)
    )
    crop_top = PCD_ROI_CROP_TOP if crop_top is None else int(crop_top)

    roi_mask = np.zeros(
        (height, width),
        dtype=bool,
    )

    x_start = crop_left
    x_end = width - crop_right

    y_start = crop_top
    y_end = height - crop_bottom

    if x_start >= x_end:
        raise ValueError(
            f"잘못된 horizontal crop: "
            f"{crop_left=}, {crop_right=}, "
            f"image width={width}"
        )

    if y_start >= y_end:
        raise ValueError(
            f"잘못된 vertical crop: "
            f"{crop_bottom=}, "
            f"image height={height}"
        )

    roi_mask[
        y_start:y_end,
        x_start:x_end,
    ] = True

    print(
        "[PCD ROI]",
        f"x = {x_start}:{x_end},",
        f"y = {y_start}:{y_end}",
    )

    print(
        "[PCD ROI] valid pixels:",
        np.count_nonzero(roi_mask),
    )

    return roi_mask


def _iter_semantic_label_tokens(label_info):
    """idToLabels 한 항목에서 라벨 토큰들을 뽑는다.

    Isaac은 instance_name별 dict로 주기도 하고 문자열로 주기도 하며,
    한 prim에 라벨이 여러 개면 "a,b"처럼 쉼표로 붙여 주기도 한다.
    세 경우를 모두 토큰 단위로 편다.
    """
    if isinstance(label_info, dict):
        raw_values = [str(value) for value in label_info.values()]
    else:
        raw_values = [str(label_info)]

    tokens = []

    for value in raw_values:
        for token in value.split(","):
            token = token.strip()

            if token:
                tokens.append(token)

    return tokens


def map_defect_semantic_ids(
    semantic_info,
    class_label=DEFECT_SEMANTIC_LABEL,
):
    """결함 '이름'별 semantic ID를 찾는다. {name: id}를 돌려준다.

    find_semantic_ids()는 class 라벨("defect")로만 묶어서 ID 목록을
    주므로 어느 결함인지 구분되지 않는다. Recall은 '결함 몇 개를
    봤는가'라서 개체를 구분해야 한다.

    setup_defects()가 decal prim마다 두 라벨을 붙인다:
        class  = "defect"          (target_object와 함께 "a,b"로 합쳐짐)
        object = "<결함 이름>"      (마찬가지로 합쳐짐)

    그래서 idToSemantics 한 항목이 이렇게 온다:
        {'class': 'target_object,defect',
         'object': 'mustard_bottle,side_plain_cluster'}

    class 토큰에 "defect"가 있는 항목만 고르고, object 토큰 중
    대상 물체 이름이 아닌 쪽을 결함 이름으로 본다.
    """
    id_to_labels = semantic_info.get("idToLabels", {})
    mapping = {}

    for semantic_id, label_info in id_to_labels.items():
        if not isinstance(label_info, dict):
            continue

        class_tokens = [
            token.strip()
            for token in str(label_info.get("class", "")).split(",")
            if token.strip()
        ]

        if class_label not in class_tokens:
            continue

        object_tokens = [
            token.strip()
            for token in str(label_info.get("object", "")).split(",")
            if token.strip()
        ]

        # class 토큰과 object 토큰은 같은 순서로 붙으므로, "defect"가
        # 놓인 자리의 object 토큰이 그 결함의 이름이다. 길이가 어긋나면
        # 대상 물체 이름을 뺀 나머지로 되돌아간다.
        name = None

        if len(object_tokens) == len(class_tokens):
            name = object_tokens[class_tokens.index(class_label)]
        elif len(object_tokens) == 1:
            name = object_tokens[0]

        if name:
            mapping[name] = int(semantic_id)

    return mapping


def record_defect_observation(
    save_dir,
    frame_id,
    semantic_image,
    defect_name_to_id,
    valid_depth_mask=None,
    reset=False,
    filename="defect_observations.json",
):
    """이 뷰에서 결함별로 몇 픽셀이 보였는지 누적 기록한다.

    Recall 계산의 원자료다. 여기서는 '판정'을 하지 않고 픽셀 수만
    남긴다. 몇 픽셀부터 검출로 칠지는 평가 쪽에서 정한다.

    valid_depth_mask를 주면 depth가 유효한 픽셀만 따로 센다. 결함이
    RGB에는 찍혔지만 depth가 비어 3D로 못 올라간 경우를 구분하기
    위한 것이다.
    """
    path = os.path.join(save_dir, filename)

    if reset or not os.path.isfile(path):
        payload = {"frames": []}
    else:
        try:
            with open(path, encoding="utf-8") as handle:
                payload = json.load(handle)
        except (ValueError, OSError):
            payload = {"frames": []}

    per_defect = {}

    for name, semantic_id in defect_name_to_id.items():
        pixel_mask = semantic_image == semantic_id
        pixel_count = int(np.count_nonzero(pixel_mask))

        if pixel_count == 0:
            continue

        entry = {"pixels": pixel_count}

        if valid_depth_mask is not None:
            entry["pixels_with_depth"] = int(
                np.count_nonzero(pixel_mask & valid_depth_mask)
            )

        # 결함이 이미지의 '어디'에 찍혔는지. defect_mask_*.png는 결함
        # 전체의 합집합이라 blob이 어느 결함인지 되찾을 수 없어서,
        # 확대 crop을 만들려면 픽셀 수로 짐작해야 했다. 여기서 이름별
        # bbox를 같이 남기면 Battery_defect_zoom.py가 정확히 자른다.
        rows = np.any(pixel_mask, axis=1)
        columns = np.any(pixel_mask, axis=0)
        y_indices = np.flatnonzero(rows)
        x_indices = np.flatnonzero(columns)
        entry["bbox_xyxy"] = [
            int(x_indices[0]),
            int(y_indices[0]),
            int(x_indices[-1]) + 1,
            int(y_indices[-1]) + 1,
        ]
        centroid_y, centroid_x = np.nonzero(pixel_mask)
        entry["centroid_xy"] = [
            float(centroid_x.mean()),
            float(centroid_y.mean()),
        ]

        per_defect[name] = entry

    # 같은 frame_id를 다시 찍으면 덮어쓴다.
    payload["frames"] = [
        frame
        for frame in payload.get("frames", [])
        if int(frame.get("frame_id", -1)) != int(frame_id)
    ]
    payload["frames"].append(
        {
            "frame_id": int(frame_id),
            "defects": per_defect,
        }
    )
    payload["frames"].sort(key=lambda frame: frame["frame_id"])

    temporary_path = path + ".tmp"

    with open(temporary_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    os.replace(temporary_path, path)

    print(
        f"[DEFECT GT] frame {frame_id}: 결함 {len(per_defect)}개 관측 "
        + (
            ", ".join(
                f"{name}={entry['pixels']}px"
                for name, entry in sorted(per_defect.items())
            )
            or "(없음)"
        )
    )

    return path


def find_semantic_ids(
    semantic_info,
    target_labels=("target_object",),
    verbose=True,
):
    """여러 semantic class의 ID를 한 번에 찾는다.

    {label: [semantic_id, ...]} dict를 돌려준다. 못 찾은 라벨은 빠진다.

    라벨당 ID가 '여러 개'인 것이 핵심이다. decal prim마다
    object=<결함 이름> 라벨이 다르기 때문에, class=defect가 같아도
    Isaac은 결함마다 서로 다른 semantic ID를 부여한다. 라벨당 하나만
    받으면 나머지 결함이 전부 배경으로 떨어져, 그 자리의 voxel이
    occupied 증거를 못 받고 영원히 UNKNOWN으로 남는다.
    """
    id_to_labels = semantic_info.get(
        "idToLabels",
        {}
    )

    if verbose:
        print(
            "[SEG] idToLabels:",
            id_to_labels
        )

    wanted = {str(label) for label in target_labels}
    found = {}

    for semantic_id, label_info in id_to_labels.items():
        tokens = _iter_semantic_label_tokens(label_info)

        for token in tokens:
            if token not in wanted:
                continue

            bucket = found.setdefault(token, [])
            value = int(semantic_id)

            if value not in bucket:
                bucket.append(value)

    for bucket in found.values():
        bucket.sort()

    return found


def flatten_semantic_ids(found):
    """{label: [ids]}를 정렬된 단일 ID 리스트로 편다."""
    ids = set()

    for bucket in found.values():
        ids.update(bucket)

    return sorted(ids)


def find_semantic_id(
    semantic_info,
    target_label="target_object",
):
    """단일 라벨의 첫 번째 ID (하위 호환용 래퍼)."""
    found = find_semantic_ids(
        semantic_info,
        target_labels=(target_label,),
    )
    bucket = found.get(target_label)

    return bucket[0] if bucket else None


def sanitize_label_for_filename(label):
    label = str(label)
    label = re.sub(r"[^0-9A-Za-z_.-]+", "_", label)
    return label.strip("_") or "unknown"


def image_values_for_valid_depth(
    depth,
    values_image,
    mask=None,
):
    depth = np.asarray(
        depth,
        dtype=np.float32,
    ).squeeze()

    values_image = np.asarray(
        values_image
    ).squeeze()

    if values_image.shape != depth.shape:
        raise ValueError(
            f"values image shape {values_image.shape} != "
            f"depth shape {depth.shape}"
        )

    valid = (
        np.isfinite(depth)
        & (depth > 0.0)
    )

    if mask is not None:
        if mask.shape != depth.shape:
            raise ValueError(
                f"mask shape {mask.shape} != depth shape {depth.shape}"
            )

        valid &= mask

    return values_image[valid].astype(np.int32)


def resolve_object_grid_bounds(
    root_size,
    leaf_size,
    origin=None,
):
    root_size = float(root_size)
    leaf_size = float(leaf_size)

    if root_size <= 0.0:
        raise ValueError(
            f"object grid root_size는 0보다 커야 합니다: {root_size}"
        )

    if leaf_size <= 0.0:
        raise ValueError(
            f"object grid leaf_size는 0보다 커야 합니다: {leaf_size}"
        )

    cells_float = root_size / leaf_size
    cells_per_axis = int(round(cells_float))

    if not np.isclose(
        cells_float,
        cells_per_axis,
        rtol=0.0,
        atol=1e-6,
    ):
        raise ValueError(
            "object grid root_size는 leaf_size의 정수 배여야 합니다: "
            f"root_size={root_size}, leaf_size={leaf_size}, "
            f"root_size / leaf_size={cells_float}"
        )

    if origin is None:
        origin = np.full(
            3,
            -0.5 * root_size,
            dtype=np.float32,
        )
    else:
        origin = np.asarray(
            origin,
            dtype=np.float32,
        )

        if origin.shape != (3,):
            raise ValueError(
                f"object grid origin shape가 (3,)이 아닙니다: {origin.shape}"
            )

    return (
        origin,
        root_size,
        leaf_size,
        cells_per_axis,
    )


def assign_defect_points_to_objects(
    points,
    octree_point_labels,
    defect_point_mask,
):
    """흠집 점을 가장 가까운 물체 instance의 octree label로 흡수시킨다.

    흠집은 독립된 객체가 아니라 물체 표면의 속성이다. 자기 instance
    label을 새로 받으면 octree에서 별도 객체로 분리되고, label별로
    나뉘어 저장되는 파일도 쪼개진다.

    octree_point_labels를 제자리에서 수정하고, 배정된 점 수를 돌려준다.
    """
    assigned_mask = (octree_point_labels > 0) & ~defect_point_mask

    if not np.any(assigned_mask) or not np.any(defect_point_mask):
        return 0

    object_labels = np.unique(octree_point_labels[assigned_mask])

    if len(object_labels) == 1:
        octree_point_labels[defect_point_mask] = int(object_labels[0])

        return int(np.count_nonzero(defect_point_mask))

    # 물체가 여러 개면 label별 centroid까지의 거리로 배정한다.
    centroids = np.stack(
        [
            points[
                assigned_mask
                & (octree_point_labels == label)
            ].mean(axis=0)
            for label in object_labels
        ],
        axis=0,
    )
    defect_points = points[defect_point_mask]
    distances = np.linalg.norm(
        defect_points[:, None, :] - centroids[None, :, :],
        axis=2,
    )
    nearest = object_labels[np.argmin(distances, axis=1)]
    octree_point_labels[defect_point_mask] = nearest.astype(np.int32)

    return int(np.count_nonzero(defect_point_mask))


def save_octree_voxels_by_label(
    points,
    labels,
    save_dir,
    frame_id,
    leaf_size=0.01,
    skip_labels=(0, -1),
    fixed_origin=None,
    fixed_root_size=None,
):
    labels = np.asarray(
        labels,
        dtype=np.int32,
    )

    os.makedirs(
        save_dir,
        exist_ok=True,
    )

    for label in np.unique(labels):
        label = int(label)

        if label in skip_labels:
            continue

        label_mask = labels == label
        label_points = points[label_mask]

        if len(label_points) == 0:
            continue

        try:
            label_octree = create_octree_voxels(
                label_points,
                labels=np.full(
                    len(label_points),
                    label,
                    dtype=np.int32,
                ),
                leaf_size=leaf_size,
                fixed_origin=fixed_origin,
                fixed_root_size=fixed_root_size,
            )
        except RuntimeError as exc:
            print(
                "[Octree Object] WARNING:",
                "label:",
                label,
                "skipped:",
                exc,
            )
            continue

        safe_label = sanitize_label_for_filename(label)

        npz_path = os.path.join(
            save_dir,
            f"object_{safe_label}_{frame_id:06d}.npz",
        )

        ply_path = os.path.join(
            save_dir,
            f"object_{safe_label}_{frame_id:06d}.ply",
        )

        save_octree_voxels(
            label_octree,
            npz_path,
        )

        save_pointcloud_ply(
            label_octree["leaf_centers"],
            ply_path,
            colors=labels_to_colors(
                label_octree["leaf_labels"]
            ),
            max_points=None,
        )

        print(
            "[Octree Object]",
            "label:",
            label,
            "points:",
            len(label_points),
            "leaves:",
            len(label_octree["leaf_indices"]),
        )


def save_zivid_capture(
    frame_id,
    zivid_render_product,
    zivid_rgb_annotator,
    zivid_depth_annotator,
    zivid_semantic_annotator,
    zivid_instance_annotator,
    zivid_bbox_annotator,
    orchestrator,
    zivid_save_dir,
    intrinsic_matrix,

    # 추가
    camera_world_matrix,
    object_world_matrix,

    visualize=False,
    pause_timeline=True,
    voxel_size=0.01,
    object_grid_root_size=0.32,
    object_grid_origin=None,
    surface_labels=SURFACE_SEMANTIC_LABELS,
    save_debug_artifacts=None,
    save_frame_images=None,
):
    """
    Zivid annotator의 RGB, depth, semantic/instance segmentation을 같은
    프레임에서 읽고 point cloud, voxel, target-aware Octree를 저장한다.

    Octree label 규칙:
        -1: target_object가 아닌 배경
         1: 첫 번째 target instance
         2: 두 번째 target instance
         ...
    """
    # ============================================================
    # Camera / Object World Matrix 확인
    # ============================================================
    camera_world_matrix = np.asarray(
    camera_world_matrix,
    dtype=np.float64,
    )

    object_world_matrix = np.asarray(
        object_world_matrix,
        dtype=np.float64,
    )

    # 실패한 캡처가 이전 프레임의 표면점을 재사용하지 못하게 먼저
    # 비운다. dict 자체는 유지해야 import한 쪽의 참조도 최신 상태다.
    LAST_TARGET_SURFACE_OBSERVATION.clear()
    LAST_TARGET_SURFACE_OBSERVATION["frame_id"] = int(frame_id)

    if camera_world_matrix.shape != (4, 4):
        raise ValueError(
            f"camera_world_matrix shape: "
            f"{camera_world_matrix.shape}"
        )

    if object_world_matrix.shape != (4, 4):
        raise ValueError(
            f"object_world_matrix shape: "
            f"{object_world_matrix.shape}"
        )

    print(
        "[POSE] Camera matrix shape:",
        camera_world_matrix.shape,
    )

    print(
        "[POSE] Object matrix shape:",
        object_world_matrix.shape,
    )


    (
        object_octree_origin,
        object_octree_root_size,
        voxel_size,
        object_grid_cells_per_axis,
    ) = resolve_object_grid_bounds(
        root_size=object_grid_root_size,
        leaf_size=voxel_size,
        origin=object_grid_origin,
    )

    print(
        "[OBJECT GRID]",
        "origin=",
        object_octree_origin,
        "root_size=",
        object_octree_root_size,
        "leaf_size=",
        voxel_size,
        "cells_per_axis=",
        object_grid_cells_per_axis,
    )

    print(f"Capturing Zivid frame {frame_id}...")
    timer = StageTimer(f"capture[{frame_id}]")

    # 모든 annotator를 같은 렌더 프레임에서 읽는다.
    zivid_render_product.hydra_texture.set_updates_enabled(True)

    try:
        orchestrator.step(
            rt_subframes=1,
            pause_timeline=pause_timeline,
            delta_time=0.0,
        )

        rgb = zivid_rgb_annotator.get_data()
        depth = zivid_depth_annotator.get_data()
        semantic = zivid_semantic_annotator.get_data()

        instance = None
        if zivid_instance_annotator is not None:
            instance = zivid_instance_annotator.get_data()

        bbox = None

        # bbox는 로그 출력에만 쓰인다. 2448x2048 rendervar를 호스트로
        # 복사하는 비용이 있으므로 진단이 꺼져 있으면 건너뛴다.
        if (
            zivid_bbox_annotator is not None
            and (
                SAVE_DEBUG_ARTIFACTS
                if save_debug_artifacts is None
                else save_debug_artifacts
            )
        ):
            bbox = zivid_bbox_annotator.get_data()

    finally:
        zivid_render_product.hydra_texture.set_updates_enabled(False)

    timer.mark("render")

    if rgb is None:
        print("[ZIVID] RGB data is not ready")
        return False

    if depth is None:
        print("[ZIVID] Depth data is not ready")
        return False

    if semantic is None:
        print("[ZIVID] Semantic segmentation data is not ready")
        return False

    rgb = np.asarray(rgb)
    depth = np.asarray(depth, dtype=np.float32).squeeze()

    if depth.ndim != 2:
        raise ValueError(
            f"[ZIVID] depth shape가 (H, W)가 아닙니다: {depth.shape}"
        )

    semantic_image = np.asarray(semantic["data"]).squeeze()
    semantic_info = semantic.get("info", {})

    if semantic_image.shape != depth.shape:
        raise ValueError(
            f"[ZIVID] semantic shape {semantic_image.shape} != "
            f"depth shape {depth.shape}"
        )

    instance_image = None
    instance_info = {}

    if instance is not None:
        instance_image = np.asarray(instance["data"]).squeeze()
        instance_info = instance.get("info", {})

        if instance_image.shape != depth.shape:
            raise ValueError(
                f"[ZIVID] instance shape {instance_image.shape} != "
                f"depth shape {depth.shape}"
            )

    timer.mark("annotator_parse")
    print("[SEG] shape:", semantic_image.shape)
    print("[SEG] info:", semantic_info)

    if instance_image is not None:
        print("[INST] shape:", instance_image.shape)
        print("[INST] info:", instance_info)

    if bbox is not None:
        print("[BBOX] data:", bbox.get("data"))
        print("[BBOX] info:", bbox.get("info", {}))

    # 물체 표면으로 취급할 semantic class들의 ID를 찾는다.
    # defect decal은 별도 class지만 물리적으로는 물체 표면이므로
    # 반드시 함께 포함해야 한다.
    surface_semantic_ids = find_semantic_ids(
        semantic_info,
        target_labels=surface_labels,
    )
    target_semantic_id_list = surface_semantic_ids.get(
        TARGET_OBJECT_SEMANTIC_LABEL,
        [],
    )
    defect_semantic_id_list = surface_semantic_ids.get(
        DEFECT_SEMANTIC_LABEL,
        [],
    )
    target_semantic_id = (
        target_semantic_id_list[0]
        if target_semantic_id_list
        else None
    )

    if target_semantic_id is None:
        print(
            "[SEG] WARNING: target_object semantic ID를 "
            "찾지 못했습니다. Octree는 모두 배경으로 표시됩니다."
        )
    else:
        print(
            "[SEG] target_object IDs:",
            target_semantic_id_list,
        )

    if not defect_semantic_id_list:
        print(
            "[SEG] defect semantic class 없음 "
            "(결함 비활성이거나 이 뷰에서 안 보임)."
        )
    else:
        # 결함마다 ID가 다르므로 리스트 전체를 써야 한다.
        print(
            "[SEG] defect IDs:",
            defect_semantic_id_list,
        )

    surface_semantic_id_list = flatten_semantic_ids(
        surface_semantic_ids
    )

    # point cloud에 포함할 영상 영역을 정의한다.
    roi_mask = create_pcd_roi_mask(depth.shape)

    # The ROI is a fixed pixel window, tuned for wherever the camera happened
    # to be. Move or tilt Zivid_camera_01 and the part lands somewhere else in
    # the frame - after one such move the part projected to v=1651..2605 while
    # the window ends at v=1548, so every target point was cropped away,
    # "Target point count: 0" followed, the capture was rejected and the
    # occupancy map stopped updating even though segmentation could still see
    # the object. Whatever the window is, it must never discard a pixel that
    # carries the target or defect label, so those are added back and the
    # rescue is reported - a large count means the crop needs retuning via
    # BATTERY_PCD_ROI_CROP_{LEFT,RIGHT,BOTTOM,TOP}.
    surface_pixel_mask = np.isin(
        semantic_image,
        list(target_semantic_id_list) + list(defect_semantic_id_list),
    )
    rescued = int(np.count_nonzero(surface_pixel_mask & ~roi_mask))

    if rescued:
        roi_mask = roi_mask | surface_pixel_mask
        print(
            "[PCD ROI] WARNING: ROI 밖의 대상 픽셀",
            rescued,
            "개를 살렸습니다. 카메라를 옮겼다면",
            "BATTERY_PCD_ROI_CROP_{LEFT,RIGHT,BOTTOM,TOP}을",
            "다시 맞추세요.",
        )

    fx = float(intrinsic_matrix[0, 0])
    fy = float(intrinsic_matrix[1, 1])
    cx = float(intrinsic_matrix[0, 2])
    cy = float(intrinsic_matrix[1, 2])

    points_camera, point_colors, point_semantic_ids = depth_to_pointcloud(
        depth=depth,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        mask=roi_mask,
        semantic_image=semantic_image,
        target_semantic_id=target_semantic_id,
        target_semantic_ids=surface_semantic_id_list,
        return_semantic_ids=True,
    )

    timer.mark("pointcloud")

    if len(points_camera) == 0:
        print(
            "[ZIVID] ROI 안에 유효한 point가 없습니다."
        )
        return False

    # ============================================================
    # Camera optical frame → Object frame
    # ============================================================

    # depth_to_pointcloud() 좌표계:
    #   +X: 이미지 오른쪽
    #   +Y: 이미지 아래
    #   +Z: 카메라 전방
    #
    # USD Camera 좌표계:
    #   +X: 이미지 오른쪽
    #   +Y: 이미지 위
    #   -Z: 카메라 전방
    camera_cv_to_usd = np.array(
        [
            [1.0,  0.0,  0.0, 0.0],
            [0.0, -1.0,  0.0, 0.0],
            [0.0,  0.0, -1.0, 0.0],
            [0.0,  0.0,  0.0, 1.0],
        ],
        dtype=np.float64,
    )

    # Camera local(CV) → Object local
    object_from_camera = (
        np.linalg.inv(object_world_matrix)
        @ camera_world_matrix
        @ camera_cv_to_usd
    )

    # (N, 3) → homogeneous (N, 4)
    points_camera_h = np.concatenate(
        [
            points_camera.astype(np.float64),
            np.ones(
                (len(points_camera), 1),
                dtype=np.float64,
            ),
        ],
        axis=1,
    )

    # Object 좌표계로 변환
    points_object_h = (
        object_from_camera
        @ points_camera_h.T
    ).T

    points_object = points_object_h[:, :3].astype(
        np.float32
    )

    print(
        "[POSE] Object-from-camera matrix:"
    )

    print(
        object_from_camera
    )

    print(
        "[POSE] Object-frame point min:",
        points_object.min(axis=0),
    )

    print(
        "[POSE] Object-frame point max:",
        points_object.max(axis=0),
    )


    point_instance_ids = None

    if instance_image is not None:
        point_instance_ids = image_values_for_valid_depth(
            depth=depth,
            values_image=instance_image,
            mask=roi_mask,
        )

        if len(point_instance_ids) != len(points_camera):
            raise RuntimeError(
                f"[ZIVID] instance ID 수 {len(point_instance_ids)} != "
                f"point 수 {len(points_camera)}"
            )

        unique_instances, instance_counts = np.unique(
            point_instance_ids,
            return_counts=True,
        )
        print(
            "[INST] Point instances:",
            dict(zip(unique_instances.tolist(), instance_counts.tolist())),
        )

    print("[ZIVID] ROI Point cloud:", points_camera.shape)

    # ------------------------------------------------------------
    # target-aware Octree label 생성
    # ------------------------------------------------------------
    # 배경은 -1로 둔다. 현재 labels_to_colors()가 음수 label을 회색으로
    # 표시하므로 instance ID 0과 색이 충돌하지 않는다.
    octree_point_labels = np.full(
        len(points_camera),
        -1,
        dtype=np.int32,
    )

    instance_id_mapping = {}
    target_point_mask = np.zeros(len(points_camera), dtype=bool)
    defect_point_mask = np.zeros(len(points_camera), dtype=bool)

    if (
    surface_semantic_id_list
    and point_semantic_ids is not None
    ):
        # defect를 포함한 '물체 표면' 전체가 target이다.
        target_point_mask = np.isin(
            point_semantic_ids,
            surface_semantic_id_list,
        )

    if (
    defect_semantic_id_list
    and point_semantic_ids is not None
    ):
        defect_point_mask = np.isin(
            point_semantic_ids,
            defect_semantic_id_list,
        )

    # 흠집을 뺀 순수 물체 표면. instance 열거의 기준이 된다.
    object_point_mask = target_point_mask & ~defect_point_mask

    target_points_object = points_object[
    target_point_mask
    ]

    LAST_TARGET_SURFACE_OBSERVATION.update(
        {
            "sensor_origin_object": np.asarray(
                object_from_camera[:3, 3],
                dtype=np.float32,
            ).copy(),
            "target_points_object": np.asarray(
                target_points_object,
                dtype=np.float32,
            ).copy(),
        }
    )

    if len(target_points_object) > 0:
        print(
            "[POSE] Target object-frame min:",
            target_points_object.min(axis=0),
        )

        print(
            "[POSE] Target object-frame max:",
            target_points_object.max(axis=0),
        )

        print(
            "[POSE] Target object-frame center:",
            target_points_object.mean(axis=0),
        )

    target_point_count = int(np.count_nonzero(target_point_mask))
    defect_point_count = int(np.count_nonzero(defect_point_mask))
    object_point_count = int(np.count_nonzero(object_point_mask))
    print("[OCTREE] Target point count:", target_point_count)
    print(
        "[OCTREE]   object surface:",
        object_point_count,
        "| defect:",
        defect_point_count,
    )

    if target_point_count == 0:
        print(
            "[OCTREE] WARNING: target_object point가 없습니다. "
            "대상 prim의 semantic label과 ROI를 확인하세요."
        )

    elif point_instance_ids is None:
        # Instance annotator가 없으면 모든 target point를 한 객체로 취급한다.
        octree_point_labels[target_point_mask] = 1
        instance_id_mapping["semantic_target"] = 1

    else:
        # 물체 instance만 1, 2, 3, ...으로 재매핑한다. Isaac의 원본
        # instance ID는 배경과 팔레트 색상 계산에 직접 쓰지 않는다.
        #
        # defect decal은 별도 prim이라 자기 instance ID를 갖는다. 그대로
        # 열거하면 흠집이 독립된 '두 번째 객체'로 잡히므로, 여기서는
        # 빼두고 아래에서 부모 물체 label로 흡수시킨다.
        target_instance_ids = np.unique(
            point_instance_ids[object_point_mask]
        )

        print(
            "[OCTREE] Original target instance IDs:",
            target_instance_ids.tolist(),
        )

        for octree_label, instance_id in enumerate(
            target_instance_ids,
            start=1,
        ):
            current_instance_mask = (
                object_point_mask
                & (point_instance_ids == instance_id)
            )
            octree_point_labels[current_instance_mask] = octree_label
            instance_id_mapping[int(instance_id)] = int(octree_label)

        if defect_point_count > 0:
            assign_defect_points_to_objects(
                points=points_camera,
                octree_point_labels=octree_point_labels,
                defect_point_mask=defect_point_mask,
            )

    unique_point_labels, point_label_counts = np.unique(
        octree_point_labels,
        return_counts=True,
    )

    timer.mark("labeling")
    print("[OCTREE] Instance ID mapping:", instance_id_mapping)
    print(
        "[OCTREE] Point labels:",
        dict(zip(unique_point_labels.tolist(), point_label_counts.tolist())),
    )

    if save_debug_artifacts is None:
        save_debug_artifacts = SAVE_DEBUG_ARTIFACTS

    if save_frame_images is None:
        save_frame_images = SAVE_FRAME_IMAGES

    # 전체 장면(배경 포함 130만 점) 기준 uniform voxel과 Octree는
    # 시각화 전용이다. NBV는 target_points_object만 쓴다.
    voxel_centers = None
    octree_data = None

    if save_debug_artifacts:
        voxel_indices, voxel_centers = pointcloud_to_voxels(
            points_camera,
            voxel_size=voxel_size,
        )

        octree_data = create_octree_voxels(
            points=points_camera,
            labels=octree_point_labels,
            leaf_size=voxel_size,
        )

        unique_leaf_labels, leaf_label_counts = np.unique(
            octree_data["leaf_labels"],
            return_counts=True,
        )

        print("[ZIVID] Occupied voxel count:", len(voxel_indices))
        print(
            "[ZIVID] Octree leaf count:",
            len(octree_data["leaf_indices"]),
        )
        print(
            "[ZIVID] Octree leaf labels:",
            dict(
                zip(
                    unique_leaf_labels.tolist(),
                    leaf_label_counts.tolist(),
                )
            ),
        )
    else:
        print(
            "[ZIVID] 디버그 산출물 생략 "
            "(BATTERY_SAVE_DEBUG_ARTIFACTS=0)"
        )
    timer.mark("debug_geometry")

    # 출력 경로를 먼저 만들고 모든 파일을 저장한다.
    os.makedirs(zivid_save_dir, exist_ok=True)

    rgb_path = os.path.join(
        zivid_save_dir,
        f"rgb_{frame_id:06d}.png",
    )
    depth_path = os.path.join(
        zivid_save_dir,
        f"depth_{frame_id:06d}.npy",
    )
    intrinsic_path = os.path.join(zivid_save_dir, "intrinsic.npy")
    intrinsic_txt_path = os.path.join(zivid_save_dir, "intrinsic.txt")
    pcd_path = os.path.join(
        zivid_save_dir,
        f"pcd_{frame_id:06d}.ply",
    )
    voxel_path = os.path.join(
        zivid_save_dir,
        f"voxels_{frame_id:06d}.ply",
    )
    octree_path = os.path.join(
        zivid_save_dir,
        f"octree_voxels_{frame_id:06d}.npz",
    )
    octree_ply_path = os.path.join(
        zivid_save_dir,
        f"octree_voxels_{frame_id:06d}.ply",
    )
    octree_object_dir = os.path.join(
        zivid_save_dir,
        f"octree_objects_{frame_id:06d}",
    )

    if rgb.ndim == 3 and rgb.shape[2] == 4:
        rgb = rgb[:, :, :3]
    rgb = rgb.astype(np.uint8)

    if save_frame_images:
        Image.fromarray(rgb).save(rgb_path, compress_level=PNG_COMPRESSION_LEVEL)
        np.save(depth_path, depth)
    else:
        print(
            "[ZIVID] RGB/depth 저장 생략 "
            "(BATTERY_SAVE_FRAME_IMAGES=0)"
        )

    # 흠집 GT 마스크(영상 공간). 검출 성능 평가의 정답으로 쓴다.
    if defect_semantic_id_list:
        defect_mask_image = np.isin(
            semantic_image,
            defect_semantic_id_list,
        )
        defect_mask_path = os.path.join(
            zivid_save_dir,
            f"defect_mask_{frame_id:06d}.png",
        )
        Image.fromarray(
            (defect_mask_image * 255).astype(np.uint8)
        ).save(defect_mask_path, compress_level=PNG_COMPRESSION_LEVEL)
        print(
            "[DEFECT GT] mask:",
            defect_mask_path,
            "pixels:",
            int(np.count_nonzero(defect_mask_image)),
        )

    # Recall 원자료. 결함이 하나도 안 보인 뷰도 기록해야 분모가 맞으므로
    # defect_semantic_id_list가 비어 있어도 부른다.
    record_defect_observation(
        save_dir=zivid_save_dir,
        frame_id=frame_id,
        semantic_image=semantic_image,
        defect_name_to_id=map_defect_semantic_ids(semantic_info),
        valid_depth_mask=np.isfinite(depth) & (depth > 0.0),
        reset=(int(frame_id) == 0),
    )
    timer.mark("save_frame_images")

    # A new sequence can change camera calibration. Never retain an older K
    # next to freshly captured depth images.
    if int(frame_id) == 0 or not os.path.isfile(intrinsic_path):
        np.save(intrinsic_path, intrinsic_matrix)
        np.savetxt(intrinsic_txt_path, intrinsic_matrix, fmt="%.10f")

    timer.mark("defect_mask")

    if save_debug_artifacts:
        # PCD는 target class를 빨간색, 나머지를 회색으로 저장한다.
        save_pointcloud_ply(
            points_camera,
            pcd_path,
            colors=point_colors,
            max_points=300000,
        )

        # Uniform voxel PLY는 기하 확인용이라 색상 없이 저장한다.
        save_pointcloud_ply(
            voxel_centers,
            voxel_path,
            max_points=None,
        )

        # NPZ에는 Octree 구조와 leaf label이 함께 저장된다.
        save_octree_voxels(octree_data, octree_path)

        octree_colors = labels_to_colors(octree_data["leaf_labels"])
        save_pointcloud_ply(
            octree_data["leaf_centers"],
            octree_ply_path,
            colors=octree_colors,
            max_points=None,
        )

    timer.mark("debug_artifacts")

    if np.any(octree_point_labels > 0):
        if save_debug_artifacts:
            save_octree_voxels_by_label(
                points=points_object,
                labels=octree_point_labels,
                save_dir=octree_object_dir,
                frame_id=frame_id,
                leaf_size=voxel_size,
                fixed_origin=object_octree_origin,
                fixed_root_size=object_octree_root_size,
            )

        camera_origin_object = object_from_camera[
            :3,
            3,
        ]

        update_and_save_object_occupancy_map(
            frame_id=frame_id,
            zivid_save_dir=zivid_save_dir,
            origin=object_octree_origin,
            root_size=object_octree_root_size,
            leaf_size=voxel_size,
            sensor_origin_object=camera_origin_object,
            target_points_object=target_points_object,
            occupied_probability=0.7,
            free_probability=0.4,
            clamp_probability_min=0.12,
            clamp_probability_max=0.97,
        )

    timer.mark("occupancy")

    if visualize:
        try:
            visualize_voxels_open3d(
                points_camera,
                voxel_size=voxel_size,
            )
        except ModuleNotFoundError as exc:
            print("[ZIVID] Voxel visualization skipped:", exc)

        if octree_data is None:
            print(
                "[ZIVID] Octree visualization skipped: "
                "디버그 산출물이 꺼져 있어 octree를 만들지 않았습니다."
            )
        else:
            try:
                visualize_octree_voxels_open3d(octree_data)
            except ModuleNotFoundError as exc:
                print("[ZIVID] Octree visualization skipped:", exc)

    valid_depth = depth[np.isfinite(depth) & (depth > 0.0)]

    print("\n========== Zivid Capture ==========")
    print("RGB:", rgb_path)
    print("Depth:", depth_path)
    print("Point cloud:", pcd_path)
    print("Octree:", octree_path)
    print("Octree PLY:", octree_ply_path)
    print("RGB shape:", rgb.shape)
    print("Depth shape:", depth.shape)

    if valid_depth.size > 0:
        print("Valid depth min/max:", valid_depth.min(), valid_depth.max())
    else:
        print("WARNING: No valid depth values")

    LAST_CAPTURE_SPANS.clear()
    LAST_CAPTURE_SPANS.update(
        timer.summary(
            extra={
                "target_points": int(target_point_count),
                "all_points": int(len(points_camera)),
            }
        )
    )
    print("===================================\n")
    if target_point_count == 0:
        print(
            "[ZIVID] Capture rejected: target_object is not visible; "
            "occupancy map was not updated."
        )
        return False

    return True
