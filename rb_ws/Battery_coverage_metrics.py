"""검사 동작 성공률: 미관측 voxel 해소 곡선과 시간 예산 판정.

물체 주변 occupancy 격자에서 UNKNOWN voxel이 뷰를 거듭할수록 얼마나
줄어드는지를 재고, 주어진 시간/뷰 예산 안에 목표 임계값에 도달했는지
판정한다.

분모 정의가 이 지표의 핵심이다. 전체 격자나 ROI 전체를 분모로 쓰면
'물리적으로 절대 관측할 수 없는 voxel'(물체 내부, 카메라가 닿지 않는
구역)이 섞여 들어가 100%가 원리적으로 불가능해진다. 그래서 세 가지를
함께 보고한다.

    roi           : ROI 안의 모든 voxel        (보수적, 해석 쉬움)
    achievable    : 후보 pose 전체를 다 찍었을 때 관측 가능한 voxel의
                    합집합. ray casting으로 계산한다. 이게 의미 있는
                    분모다.
    surface       : occupied로 판정된 voxel     (물체 표면 자체)

Isaac Sim에 의존하지 않는다 (numpy만). 저장된 스냅샷과 실행 로그만으로
동작하므로 시뮬레이터를 끄고 분석할 수 있다.
"""

from __future__ import annotations

import datetime
import glob
import os
import re

import numpy as np

try:
    from rb_ws.NBV_utils import (
        CAMERA_CV_TO_USD,
        OCCUPANCY_OCCUPIED,
        OCCUPANCY_UNKNOWN,
        ray_voxel_indices,
    )
except ModuleNotFoundError:
    from NBV_utils import (
        CAMERA_CV_TO_USD,
        OCCUPANCY_OCCUPIED,
        OCCUPANCY_UNKNOWN,
        ray_voxel_indices,
    )


DEFAULT_THRESHOLDS = (0.5, 0.7, 0.9)


# ============================================================
# 스냅샷 로드
# ============================================================


def load_occupancy_snapshots(occupancy_dir):
    """object_occupancy_%06d.npz들을 frame 순서로 읽는다.

    파일이 프레임 인덱스로 덮어쓰이기 때문에 서로 다른 실행이 섞일 수
    있다. 파일 mtime이 단조 증가하지 않으면 경고한다.
    """
    paths = sorted(
        p for p in glob.glob(
            os.path.join(occupancy_dir, "object_occupancy_*.npz")
        )
        if "latest" not in os.path.basename(p)
    )

    if not paths:
        raise FileNotFoundError(
            f"occupancy 스냅샷이 없습니다: {occupancy_dir}"
        )

    snapshots = []
    previous_mtime = None
    mixed = False

    for path in paths:
        frame_id = int(
            re.search(r"(\d+)\.npz$", os.path.basename(path)).group(1)
        )
        mtime = os.path.getmtime(path)

        if previous_mtime is not None and mtime < previous_mtime:
            mixed = True

        previous_mtime = mtime

        with np.load(path) as data:
            snapshots.append(
                {
                    "frame_id": frame_id,
                    "path": path,
                    "mtime": mtime,
                    "state": np.asarray(data["state"], dtype=np.int8),
                    "origin": np.asarray(
                        data["origin"], dtype=np.float64
                    ),
                    "leaf_size": float(data["leaf_size"]),
                    "root_size": float(data["root_size"]),
                }
            )

    if mixed:
        print(
            "[COVERAGE][WARN] 스냅샷 파일의 수정 시각이 단조 증가하지 "
            "않습니다. 서로 다른 실행의 프레임이 섞였을 수 있습니다. "
            "에피소드마다 디렉터리를 비우거나 옮기세요."
        )

    return snapshots


def voxel_centers(state_shape, origin, leaf_size):
    grid = np.stack(
        np.meshgrid(
            *[np.arange(s) for s in state_shape],
            indexing="ij",
        ),
        axis=-1,
    )

    return origin + (grid + 0.5) * leaf_size


def roi_mask_from_bounds(state_shape, origin, leaf_size, roi_min, roi_max):
    centers = voxel_centers(state_shape, origin, leaf_size)

    return np.all(
        (centers >= np.asarray(roi_min))
        & (centers <= np.asarray(roi_max)),
        axis=-1,
    )


# ============================================================
# 도달 가능 집합
# ============================================================


def compute_achievable_mask(
    state_shape,
    origin,
    leaf_size,
    root_size,
    camera_world_matrix,
    object_world_position,
    candidate_rotations_deg,
    intrinsic_matrix,
    image_size,
    pixel_stride=48,
    roi_mask=None,
    occupied_mask=None,
    log_prefix="[COVERAGE]",
):
    """후보 pose를 모두 찍었을 때 관측 가능한 voxel의 합집합.

    occupied_mask를 주면 ray가 occupied voxel에서 멈춘다. 이게
    중요하다. 가림을 무시하면 ROI의 거의 모든 voxel이 '도달 가능'으로
    나와 분모가 무의미해진다 (실측: 6143/6144). 실제 한계는 시야가
    아니라 물체가 자기 뒤를 가리는 것이다. score_candidate_pose도
    같은 규칙으로 ray를 끊는다.

    occupied_mask는 보통 마지막 occupancy map의 occupied 집합을 쓴다.
    그 맵 자체가 불완전하므로 결과는 상한이라는 점에 유의한다.
    """
    camera_world_matrix = np.asarray(
        camera_world_matrix, dtype=np.float64
    )
    achievable = np.zeros(state_shape, dtype=bool)
    rays = _camera_rays(intrinsic_matrix, image_size, pixel_stride)

    for index, rotation_deg in enumerate(candidate_rotations_deg):
        object_world = _object_world_matrix(
            object_world_position, rotation_deg
        )
        object_from_camera = (
            np.linalg.inv(object_world)
            @ camera_world_matrix
            @ CAMERA_CV_TO_USD
        )
        sensor_origin = object_from_camera[:3, 3]
        directions = rays @ object_from_camera[:3, :3].T

        for direction in directions:
            indices = ray_voxel_indices(
                sensor_origin=sensor_origin,
                ray_direction=direction,
                grid_origin=origin,
                root_size=root_size,
                leaf_size=leaf_size,
            )

            for voxel in indices:
                key = tuple(int(v) for v in voxel)
                achievable[key] = True

                if occupied_mask is not None and occupied_mask[key]:
                    break

        if (index + 1) % 5 == 0:
            print(
                f"{log_prefix} 도달 가능 집합 계산 "
                f"{index + 1}/{len(candidate_rotations_deg)} "
                f"누적 {int(achievable.sum())} voxel"
            )

    if roi_mask is not None:
        achievable &= roi_mask

    return achievable


def _camera_rays(intrinsic_matrix, image_size, pixel_stride):
    intrinsic_matrix = np.asarray(intrinsic_matrix, dtype=np.float64)
    width, height = image_size
    us = np.arange(0, width, pixel_stride, dtype=np.float64)
    vs = np.arange(0, height, pixel_stride, dtype=np.float64)
    grid_u, grid_v = np.meshgrid(us, vs, indexing="xy")
    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]
    cx = intrinsic_matrix[0, 2]
    cy = intrinsic_matrix[1, 2]
    x = (grid_u.ravel() - cx) / fx
    y = (grid_v.ravel() - cy) / fy
    rays = np.stack([x, y, np.ones_like(x)], axis=1)

    return rays / np.linalg.norm(rays, axis=1, keepdims=True)


def _object_world_matrix(position, rotation_xyz_deg):
    rx, ry, rz = np.radians(np.asarray(rotation_xyz_deg, dtype=np.float64))
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    rot_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    rot_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rot_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    matrix = np.eye(4)
    matrix[:3, :3] = rot_z @ rot_y @ rot_x
    matrix[:3, 3] = np.asarray(position, dtype=np.float64)

    return matrix


# ============================================================
# 실행 로그 타임라인
# ============================================================


LOG_TIMESTAMP = re.compile(
    r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d+)\] .*?\] (.*)$"
)


def parse_run_timeline(log_path):
    """실행 로그에서 캡처/이동 이벤트 시각을 뽑는다."""
    events = []

    with open(log_path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = LOG_TIMESTAMP.match(line.rstrip("\n"))

            if not match:
                continue

            stamp = datetime.datetime.strptime(
                match.group(1), "%Y-%m-%d %H:%M:%S.%f"
            )
            message = match.group(2)

            for needle, tag in (
                ("Capturing Zivid frame", "capture_begin"),
                ("========== Zivid Capture", "capture_done"),
                ("Selecting pose for frame", "select_begin"),
                ("Selected NBV:", "select_done"),
                ("Executing selected pose", "move_begin"),
                ("Skipping unreachable candidate", "move_failed"),
                ("Move succeeded", "move_ok"),
            ):
                if needle in message:
                    events.append((stamp, tag, message))
                    break

    return events


def summarize_timeline(events):
    """구간별 소요 시간 통계."""
    if not events:
        return {}

    start = events[0][0]
    spans = {
        "select": [],
        "move_ok": [],
        "move_failed": [],
        "capture": [],
    }
    open_tags = {}

    for stamp, tag, _ in events:
        if tag == "select_begin":
            open_tags["select"] = stamp
        elif tag == "select_done" and "select" in open_tags:
            spans["select"].append(
                (stamp - open_tags.pop("select")).total_seconds()
            )
        elif tag == "move_begin":
            open_tags["move"] = stamp
        elif tag in ("move_ok", "capture_done") and "move" in open_tags:
            spans["move_ok"].append(
                (stamp - open_tags.pop("move")).total_seconds()
            )
        elif tag == "move_failed" and "move" in open_tags:
            spans["move_failed"].append(
                (stamp - open_tags.pop("move")).total_seconds()
            )

    capture_times = [
        (stamp - start).total_seconds()
        for stamp, tag, _ in events
        if tag == "capture_done"
    ]

    return {
        "capture_elapsed_s": capture_times,
        "total_s": (events[-1][0] - start).total_seconds(),
        "spans": {
            key: {
                "count": len(values),
                "mean": float(np.mean(values)) if values else 0.0,
                "max": float(np.max(values)) if values else 0.0,
                "total": float(np.sum(values)) if values else 0.0,
            }
            for key, values in spans.items()
        },
    }


# ============================================================
# 지표 계산
# ============================================================


def coverage_curve(snapshots, masks):
    """프레임별 해소 비율. masks는 {이름: bool 격자}."""
    rows = []

    for snapshot in snapshots:
        state = snapshot["state"]
        known = state != OCCUPANCY_UNKNOWN
        row = {"frame_id": snapshot["frame_id"]}

        for name, mask in masks.items():
            total = int(mask.sum())
            resolved = int((known & mask).sum())
            row[name] = {
                "total": total,
                "resolved": resolved,
                "fraction": resolved / total if total else 0.0,
            }

        row["occupied"] = int(
            (state == OCCUPANCY_OCCUPIED).sum()
        )
        rows.append(row)

    return rows


def first_reaching(rows, key, threshold, elapsed_by_frame=None):
    """임계값에 처음 도달한 (뷰 수, 경과 시간)."""
    for index, row in enumerate(rows):
        if row[key]["fraction"] >= threshold:
            elapsed = None

            if elapsed_by_frame and index < len(elapsed_by_frame):
                elapsed = elapsed_by_frame[index]

            return index + 1, elapsed

    return None, None


def evaluate_budget(
    rows,
    key,
    threshold,
    budget_s=None,
    budget_views=None,
    elapsed_by_frame=None,
):
    """예산 안에 임계값 도달했는지 판정한다."""
    views, elapsed = first_reaching(
        rows, key, threshold, elapsed_by_frame
    )
    final = rows[-1][key]["fraction"] if rows else 0.0

    if views is None:
        return {
            "success": False,
            "reason": (
                f"{threshold:.0%}에 도달하지 못함 "
                f"(최종 {final:.1%})"
            ),
            "views": None,
            "elapsed_s": None,
            "final_fraction": final,
        }

    ok = True
    reasons = []

    if budget_views is not None and views > budget_views:
        ok = False
        reasons.append(f"뷰 {views} > 예산 {budget_views}")

    if budget_s is not None:
        if elapsed is None:
            reasons.append("경과 시간 정보 없음")
        elif elapsed > budget_s:
            ok = False
            reasons.append(f"{elapsed:.1f}s > 예산 {budget_s:.1f}s")

    return {
        "success": ok,
        "reason": "; ".join(reasons) if reasons else "예산 내 도달",
        "views": views,
        "elapsed_s": elapsed,
        "final_fraction": final,
    }


def load_arm_candidate_rotations(
    config_path=None,
    max_abs_deg=60.0,
    limit=20,
):
    """Battery_config.py에서 팔 후보 pose 각도를 읽는다.

    Battery_config는 pxr에 의존하므로 Isaac 밖에서는 import할 수 없다.
    소스에서 각도 리스트만 파싱해 쓴다.
    """
    import ast

    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "Battery_config.py",
        )

    source = open(config_path, encoding="utf-8").read()
    tree = ast.parse(source)

    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue

        targets = [
            t.id for t in node.targets if isinstance(t, ast.Name)
        ]

        if "MUSTARD_BOTTLE_CAPTURE_POSE_DEGREES" not in targets:
            continue

        entries = ast.literal_eval(node.value)

        return [
            rotation
            for _, rotation in entries
            if max(abs(v) for v in rotation) <= max_abs_deg
        ][:limit]

    raise RuntimeError(
        "MUSTARD_BOTTLE_CAPTURE_POSE_DEGREES를 찾지 못했습니다."
    )


def zivid_camera_world_matrix(
    position=(0.0206, 0.9978, 1.2248),
    forward=(0.0, -1.0, 0.0),
    up=(0.0, 0.0, 1.0),
):
    """측정한 Zivid 카메라 자세로 USD 규약 4x4를 만든다.

    USD 카메라는 로컬 -Z가 전방, +Y가 위다. 따라서 열벡터는
    [right, up, back=-forward, position] 순서다.
    """
    back = -np.asarray(forward, dtype=np.float64)
    back /= np.linalg.norm(back)
    up_vec = np.asarray(up, dtype=np.float64)
    right = np.cross(up_vec, back)
    right /= np.linalg.norm(right)
    up_ortho = np.cross(back, right)
    matrix = np.eye(4)
    matrix[:3, 0] = right
    matrix[:3, 1] = up_ortho
    matrix[:3, 2] = back
    matrix[:3, 3] = np.asarray(position, dtype=np.float64)

    return matrix


# ============================================================
# 리포트
# ============================================================


def report(
    occupancy_dir="/home/seungjun/isaac_ws/zivid_capture/object_occupancy",
    intrinsic_path="/home/seungjun/isaac_ws/zivid_capture/intrinsic.npy",
    log_path=None,
    roi_min=(-0.08, -0.12, -0.08),
    roi_max=(0.08, 0.12, 0.08),
    object_world_position=(0.0206, 0.548, 1.2248),
    thresholds=DEFAULT_THRESHOLDS,
    budget_s=10.0,
    budget_views=None,
    pixel_stride=32,
):
    """검사 동작 커버리지 리포트를 출력하고 dict로 돌려준다."""
    snapshots = load_occupancy_snapshots(occupancy_dir)
    first = snapshots[0]
    final_state = snapshots[-1]["state"]
    roi = roi_mask_from_bounds(
        first["state"].shape,
        first["origin"],
        first["leaf_size"],
        roi_min,
        roi_max,
    )
    achievable = compute_achievable_mask(
        first["state"].shape,
        first["origin"],
        first["leaf_size"],
        first["root_size"],
        camera_world_matrix=zivid_camera_world_matrix(),
        object_world_position=object_world_position,
        candidate_rotations_deg=load_arm_candidate_rotations(),
        intrinsic_matrix=np.load(intrinsic_path),
        image_size=(2448, 2048),
        pixel_stride=pixel_stride,
        roi_mask=roi,
        occupied_mask=final_state == OCCUPANCY_OCCUPIED,
    )
    masks = {"roi": roi, "achievable": achievable}
    rows = coverage_curve(snapshots, masks)

    elapsed = None
    timeline = None

    if log_path:
        events = parse_run_timeline(log_path)
        timeline = summarize_timeline(events)
        elapsed = timeline.get("capture_elapsed_s")

    print()
    print("=" * 66)
    print("검사 동작 커버리지 리포트")
    print("=" * 66)
    print(
        f"  ROI voxel        {int(roi.sum()):>6}"
        f"   도달 가능 {int(achievable.sum()):>6}"
        f" ({achievable.sum() / max(roi.sum(), 1):.1%})"
    )
    print()
    header = f"{'뷰':>4}{'경과s':>8}{'ROI해소':>10}{'도달가능대비':>13}{'occupied':>10}"
    print(header)
    print("-" * len(header))

    for index, row in enumerate(rows):
        seconds = (
            f"{elapsed[index]:8.1f}"
            if elapsed and index < len(elapsed)
            else f"{'-':>8}"
        )
        print(
            f"{index + 1:>4}{seconds}"
            f"{row['roi']['fraction']:>9.1%}"
            f"{row['achievable']['fraction']:>13.1%}"
            f"{row['occupied']:>10}"
        )

    print()
    print(f"  시간 예산 {budget_s:.0f}s 기준 판정")

    results = {}

    for threshold in thresholds:
        verdict = evaluate_budget(
            rows,
            "achievable",
            threshold,
            budget_s=budget_s,
            budget_views=budget_views,
            elapsed_by_frame=elapsed,
        )
        results[threshold] = verdict
        mark = "성공" if verdict["success"] else "실패"
        detail = (
            f"뷰 {verdict['views']}"
            + (
                f", {verdict['elapsed_s']:.1f}s"
                if verdict["elapsed_s"] is not None
                else ""
            )
            if verdict["views"]
            else "미도달"
        )
        print(
            f"    {threshold:>5.0%} : {mark:<4} ({detail})"
            f"  {verdict['reason']}"
        )

    if timeline:
        print()
        print("  시간 분해")
        for key, label in (
            ("select", "NBV 점수 계산"),
            ("move_ok", "이동+캡처 성공"),
            ("move_failed", "이동 실패(도달불가)"),
        ):
            span = timeline["spans"].get(key, {})

            if span.get("count"):
                print(
                    f"    {label:<20} {span['count']:>3}회 "
                    f"평균 {span['mean']:6.1f}s "
                    f"최대 {span['max']:6.1f}s "
                    f"합계 {span['total']:7.1f}s"
                )

        print(f"    {'전체':<20}     {timeline['total_s']:.1f}s")

    print("=" * 66)

    return {
        "rows": rows,
        "results": results,
        "timeline": timeline,
        "roi_total": int(roi.sum()),
        "achievable_total": int(achievable.sum()),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="검사 동작 커버리지 리포트",
    )
    parser.add_argument("--occupancy-dir", default=None)
    parser.add_argument("--log", default=None)
    parser.add_argument("--budget-s", type=float, default=10.0)
    parser.add_argument("--budget-views", type=int, default=None)
    args = parser.parse_args()

    kwargs = {"budget_s": args.budget_s, "budget_views": args.budget_views}

    if args.occupancy_dir:
        kwargs["occupancy_dir"] = args.occupancy_dir

    if args.log:
        kwargs["log_path"] = args.log

    report(**kwargs)


# ============================================================
# 물체 표면 기준 커버리지
# ============================================================
#
# 격자 전체를 분모로 쓰면 95%가 물체와 무관한 원거리 공기라 수치가
# 의미를 잃는다 (실측: 격자 기준 24%인데 표면 기준으로는 84.7%였다).
# 검사 성공률은 '물체 표면을 얼마나 봤는가'이므로, 실제 메시를 같은
# 격자에 이산화해 그것을 분모로 삼는다.


def surface_voxel_mask(
    vertices,
    triangles,
    origin,
    leaf_size,
    grid_shape,
    samples=500000,
    seed=0,
):
    """메시 표면을 occupancy 격자와 같은 해상도로 이산화한다.

    삼각형 면적에 비례해 표면을 조밀 샘플링한 뒤 각 점이 속한 voxel을
    모은다. 격자 밖으로 나가는 점은 버린다.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.int32)
    origin = np.asarray(origin, dtype=np.float64)
    leaf_size = float(leaf_size)
    grid_shape = tuple(int(v) for v in grid_shape)

    corner0 = vertices[triangles[:, 0]]
    edge1 = vertices[triangles[:, 1]] - corner0
    edge2 = vertices[triangles[:, 2]] - corner0
    areas = 0.5 * np.linalg.norm(np.cross(edge1, edge2), axis=1)
    total_area = float(areas.sum())

    if total_area <= 0.0:
        raise ValueError("메시 표면적이 0입니다.")

    rng = np.random.default_rng(seed)
    picked = rng.choice(
        len(triangles),
        size=int(samples),
        p=areas / total_area,
    )
    u = rng.random(int(samples))
    v = rng.random(int(samples))
    over = u + v > 1.0
    u[over] = 1.0 - u[over]
    v[over] = 1.0 - v[over]
    points = (
        corner0[picked]
        + u[:, None] * edge1[picked]
        + v[:, None] * edge2[picked]
    )
    indices = np.floor((points - origin) / leaf_size).astype(np.int64)
    inside = np.all(
        (indices >= 0) & (indices < np.array(grid_shape)),
        axis=1,
    )
    mask = np.zeros(grid_shape, dtype=bool)

    if np.any(inside):
        mask[tuple(indices[inside].T)] = True

    return mask, total_area


def surface_coverage(state, surface_mask):
    """표면 voxel 중 occupied로 관측된 비율."""
    surface_total = int(surface_mask.sum())

    if surface_total == 0:
        return 0.0, 0, 0

    observed = int(
        np.count_nonzero(
            (state == OCCUPANCY_OCCUPIED) & surface_mask
        )
    )

    return observed / surface_total, observed, surface_total


# ============================================================
# 방향성 표면(surfel) 커버리지
# ============================================================


def sample_directional_surfels(
    vertices,
    triangles,
    samples=50000,
    seed=0,
):
    """메시 표면을 위치+법선을 가진 동일 면적 표본으로 만든다.

    삼각형을 면적에 비례해 뽑으므로 각 surfel은 같은 표면적을 대표한다.
    위치가 같은 얇은 판의 앞/뒷면도 반대 법선을 가진 별도 표본으로
    유지된다. 이것이 위치만 boolean voxel로 합치던 기존 방식과의 핵심
    차이다.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.int32)
    sample_count = int(samples)

    if sample_count <= 0:
        raise ValueError("directional surfel samples는 0보다 커야 합니다.")
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("vertices shape는 (N, 3)이어야 합니다.")
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("triangles shape는 (M, 3)이어야 합니다.")

    corner0 = vertices[triangles[:, 0]]
    edge1 = vertices[triangles[:, 1]] - corner0
    edge2 = vertices[triangles[:, 2]] - corner0
    cross = np.cross(edge1, edge2)
    double_areas = np.linalg.norm(cross, axis=1)
    valid = double_areas > 1e-15

    if not np.any(valid):
        raise ValueError("메시 표면적이 0입니다.")

    corner0 = corner0[valid]
    edge1 = edge1[valid]
    edge2 = edge2[valid]
    cross = cross[valid]
    double_areas = double_areas[valid]
    areas = 0.5 * double_areas
    total_area = float(areas.sum())
    face_normals = cross / double_areas[:, None]

    rng = np.random.default_rng(seed)
    picked = rng.choice(
        len(areas),
        size=sample_count,
        p=areas / total_area,
    )
    u = rng.random(sample_count)
    v = rng.random(sample_count)
    over = u + v > 1.0
    u[over] = 1.0 - u[over]
    v[over] = 1.0 - v[over]
    points = (
        corner0[picked]
        + u[:, None] * edge1[picked]
        + v[:, None] * edge2[picked]
    )

    return {
        "points": points.astype(np.float32),
        "normals": face_normals[picked].astype(np.float32),
        "observed": np.zeros(sample_count, dtype=bool),
        "total_area": total_area,
    }


def _ray_hits_facing_surfel(tree, normals, origins, directions, max_range, step, radius):
    """True where a ray (origin + s*direction, s >= 2*radius) meets a surfel facing back at it.

    Only surfels with normal . direction < -0.5 count, so a ray grazing a
    parallel surface (a wall it runs along, a lip just above it) is no hit.
    """
    padded = np.vstack([normals, np.zeros((1, 3))])  # index len(points) = no neighbour
    hit = np.zeros(len(origins), dtype=bool)
    for distance in np.arange(2.0 * radius, max_range, step):
        pending = np.flatnonzero(~hit)
        if len(pending) == 0:
            break
        _, indices = tree.query(
            origins[pending] + distance * directions[pending],
            k=8,
            distance_upper_bound=radius,
        )
        facing = np.einsum("ikj,ij->ik", padded[indices], directions[pending])
        hit[pending[np.min(facing, axis=1) < -0.5]] = True
    return hit


def inner_side_mask(points, normals, step=0.0015, radius=0.0012):
    """Side surfels on the inside of an overhang, e.g. the legs' inner faces.

    A side surfel (|n_z| < 0.5) counts as inner when an overhang covers it
    (its +Z ray meets a downward-facing surfel) AND its own normal ray meets
    a wall facing back at it, i.e. it faces another wall under the overhang.
    The battery cap's legs face each other under the top plate (inner); the
    legs' +/-Y faces and the plate's edges face open space (outer).
    Needs scipy; returns an all-False mask without it.
    """
    points = np.asarray(points, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)
    side = np.flatnonzero(np.abs(normals[:, 2]) < 0.5)
    mask = np.zeros(len(points), dtype=bool)
    if len(side) == 0:
        return mask
    try:
        from scipy.spatial import cKDTree
    except (ImportError, ModuleNotFoundError):
        return mask

    tree = cKDTree(points)  # all surfels: the rays must meet any part surface
    extent = points.max(axis=0) - points.min(axis=0)
    up = np.tile([0.0, 0.0, 1.0], (len(side), 1))
    covered = side[
        _ray_hits_facing_surfel(
            tree, normals, points[side], up, float(extent[2]) + step, step, radius
        )
    ]
    facing = _ray_hits_facing_surfel(
        tree,
        normals,
        points[covered],
        normals[covered],
        float(np.linalg.norm(extent)),
        step,
        radius,
    )
    mask[covered[facing]] = True
    # A sparse sample can miss a ray hit; let coplanar neighbours vote.
    vote_radius = 0.004
    for index, neighbours in zip(side, tree.query_ball_point(points[side], r=vote_radius)):
        neighbours = np.asarray(neighbours, dtype=np.int64)
        coplanar = neighbours[
            (normals[neighbours] @ normals[index] > 0.95)
            & (np.abs((points[neighbours] - points[index]) @ normals[index]) < 0.0005)
        ]
        if len(coplanar) and np.count_nonzero(mask[coplanar]) * 2 > len(coplanar):
            mask[index] = True
    return mask


def thin_side_mask(points, normals, radius=0.01, min_width=0.008, band=0.0015, plane_tol=0.0005):
    """Side surfels on a strip narrower than ``min_width``: edges, steps, rims.

    For each side surfel, the coplanar surfels within ``radius`` (same normal,
    within ``plane_tol`` of its plane) are measured along a vertical and a
    horizontal line through it (``band`` wide). If either line is shorter
    than ``min_width`` the surfel sits on a thin strip: the plate's 2-5 mm
    edges and the top's step walls; the legs' 20 x 70 mm end faces are not.
    Needs scipy; returns an all-False mask without it.
    """
    points = np.asarray(points, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)
    side = np.flatnonzero(np.abs(normals[:, 2]) < 0.5)
    mask = np.zeros(len(points), dtype=bool)
    if len(side) == 0:
        return mask
    try:
        from scipy.spatial import cKDTree
    except (ImportError, ModuleNotFoundError):
        return mask

    tree = cKDTree(points[side])
    for index, neighbours in zip(side, tree.query_ball_point(points[side], r=radius)):
        neighbours = side[np.asarray(neighbours, dtype=np.int64)]
        normal = normals[index]
        offsets = points[neighbours] - points[index]
        coplanar = (normals[neighbours] @ normal > 0.95) & (np.abs(offsets @ normal) < plane_tol)
        offsets = offsets[coplanar]
        vertical = np.array([0.0, 0.0, 1.0]) - normal[2] * normal
        vertical /= np.linalg.norm(vertical)
        along_v = offsets @ vertical
        along_h = offsets @ np.cross(normal, vertical)
        column = along_v[np.abs(along_h) < band]
        row = along_h[np.abs(along_v) < band]
        if np.ptp(column) < min_width or np.ptp(row) < min_width:
            mask[index] = True
    return mask


def buried_top_mask(points, normals, radius=0.0012, min_gap=0.0005):
    """Top surfels with another horizontal surface of the part right above them.

    The CAD keeps overlapping layers: under each end pad (top at 50.35 mm)
    the plate top (50.1) and pocket floor (49.9) are still there, and a 1 mm
    sheet lies on the centre. Such a +Z surfel has a surfel with |n_z| > 0.9
    within ``radius`` in XY and more than ``min_gap`` higher; no camera can
    see it, yet the 3 mm depth match marked the layers within 3 mm as seen.
    Needs scipy; returns an all-False mask without it.
    """
    points = np.asarray(points, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)
    top = np.flatnonzero(normals[:, 2] >= 0.5)
    flat = np.flatnonzero(np.abs(normals[:, 2]) > 0.9)
    mask = np.zeros(len(points), dtype=bool)
    if len(top) == 0 or len(flat) == 0:
        return mask
    try:
        from scipy.spatial import cKDTree
    except (ImportError, ModuleNotFoundError):
        return mask

    tree = cKDTree(points[flat, :2])
    for index, neighbours in zip(top, tree.query_ball_point(points[top, :2], r=radius)):
        if neighbours and np.any(points[flat[neighbours], 2] > points[index, 2] + min_gap):
            mask[index] = True
    return mask


def coverage_exclusion_masks(points, normals):
    """Surfels left out of top/side coverage.

    Side counts only the legs' outer faces:
    * ``inner``: legs' inner faces under the top plate (inner_side_mask)
    * ``y_facing``: every +/-Y facing side surfel - on this cap they are thin
      lines (plate edges, step walls, the legs' 3 mm wide Y faces)
    * ``thin``: any other strip narrower than 8 mm (thin_side_mask), e.g. the
      top's X step walls and the plate's end wings beside the legs
    Top counts only surfaces a camera can reach:
    * ``buried_top``: CAD layers under another top surface (buried_top_mask)
    ``excluded`` is the union.
    """
    normals = np.asarray(normals, dtype=np.float64)
    side = np.abs(normals[:, 2]) < 0.5
    inner = inner_side_mask(points, normals)
    y_facing = side & (np.abs(normals[:, 1]) > np.abs(normals[:, 0])) & ~inner
    thin = thin_side_mask(points, normals) & ~inner & ~y_facing
    buried_top = buried_top_mask(points, normals)
    return {
        "inner": inner,
        "y_facing": y_facing,
        "thin": thin,
        "buried_top": buried_top,
        "excluded": inner | y_facing | thin | buried_top,
    }


def _nearest_point_distances(query_points, reference_points, radius):
    """Nearest distance with scipy acceleration and a numpy-only fallback."""
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(reference_points)
        try:
            distances, _ = tree.query(
                query_points,
                k=1,
                distance_upper_bound=radius,
                workers=-1,
            )
        except TypeError:
            distances, _ = tree.query(
                query_points,
                k=1,
                distance_upper_bound=radius,
            )

        return np.asarray(distances, dtype=np.float64)
    except (ImportError, ModuleNotFoundError):
        pass

    # Isaac 환경에서 scipy가 빠진 경우를 위한 반경 spatial hash. 같은
    # cell과 인접 26개 cell만 보면 radius 이내의 최근접점을 놓치지 않는다.
    cell_size = float(radius)
    reference_cells = np.floor(
        reference_points / cell_size
    ).astype(np.int64)
    buckets = {}

    for index, cell in enumerate(reference_cells):
        buckets.setdefault(tuple(cell), []).append(index)

    distances = np.full(len(query_points), np.inf, dtype=np.float64)
    offsets = [
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
    ]

    for query_index, point in enumerate(query_points):
        cell = np.floor(point / cell_size).astype(np.int64)
        nearby = []

        for offset in offsets:
            key = tuple(cell + np.asarray(offset, dtype=np.int64))
            nearby.extend(buckets.get(key, ()))

        if nearby:
            delta = reference_points[np.asarray(nearby)] - point
            distances[query_index] = float(
                np.sqrt(np.min(np.einsum("ij,ij->i", delta, delta)))
            )

    return distances


def update_directional_surfel_observations(
    surface_model,
    target_points_object,
    sensor_origin_object,
    match_distance=0.003,
    min_facing_cos=0.05,
    return_visible_mask=False,
):
    """한 depth capture로 실제 보인 방향성 surfel만 관측 처리한다."""
    points = np.asarray(surface_model["points"], dtype=np.float64)
    normals = np.asarray(surface_model["normals"], dtype=np.float64)
    observed = np.asarray(surface_model["observed"], dtype=bool)
    targets = np.asarray(target_points_object, dtype=np.float64).reshape(-1, 3)
    sensor = np.asarray(sensor_origin_object, dtype=np.float64)
    radius = float(match_distance)
    facing_cos = float(min_facing_cos)

    if radius <= 0.0:
        raise ValueError("surfel match_distance는 0보다 커야 합니다.")
    if not 0.0 <= facing_cos < 1.0:
        raise ValueError("surfel min_facing_cos는 [0, 1)이어야 합니다.")
    if sensor.shape != (3,):
        raise ValueError("sensor_origin_object shape는 (3,)이어야 합니다.")
    if len(targets) == 0:
        return 0

    view = sensor[None, :] - points
    view_norm = np.linalg.norm(view, axis=1)
    front_facing = (
        np.einsum("ij,ij->i", normals, view)
        > facing_cos * view_norm
    )
    # The persistent IMSI route cache needs the complete set seen from this
    # pose, including surfels that an earlier frame had already observed.
    # Legacy callers only need newly observed surfels, so retain the cheaper
    # unobserved-only query unless the full mask is explicitly requested.
    candidates = np.flatnonzero(
        front_facing if return_visible_mask else (~observed & front_facing)
    )

    if len(candidates) == 0:
        if return_visible_mask:
            return 0, np.zeros(len(points), dtype=bool)
        return 0

    distances = _nearest_point_distances(
        points[candidates],
        targets,
        radius,
    )
    visible_indices = candidates[distances <= radius]
    newly_observed_count = int(
        np.count_nonzero(~observed[visible_indices])
    )
    observed[visible_indices] = True
    surface_model["observed"] = observed
    if return_visible_mask:
        visible_mask = np.zeros(len(points), dtype=bool)
        visible_mask[visible_indices] = True
        return newly_observed_count, visible_mask
    return newly_observed_count


def directional_surface_coverage(surface_model):
    """전체 및 object-local +/-Z/side 방향별 surfel coverage.

    surface_model["coverage_excluded"](coverage_exclusion_masks)가 있으면
    top/side 분모와 분자에서 뺀다: side는 다리 바깥면만, top은 카메라가 닿는
    (다른 층에 묻히지 않은) 면만. 뺀 surfel은 result["excluded"]로 따로 준다.
    """
    normals = np.asarray(surface_model["normals"], dtype=np.float64)
    observed = np.asarray(surface_model["observed"], dtype=bool)
    excluded = surface_model.get("coverage_excluded")
    excluded = (
        np.zeros(len(normals), dtype=bool)
        if excluded is None
        else np.asarray(excluded, dtype=bool)
    )
    groups = {
        "top(+Z)": (normals[:, 2] >= 0.5) & ~excluded,
        "bottom(-Z)": normals[:, 2] <= -0.5,
        "side": (np.abs(normals[:, 2]) < 0.5) & ~excluded,
    }

    def summarize(mask):
        total = int(np.count_nonzero(mask))
        count = int(np.count_nonzero(observed & mask))
        return {
            "fraction": count / total if total else 0.0,
            "observed": count,
            "total": total,
        }

    result = summarize(np.ones(len(observed), dtype=bool))
    result["groups"] = {
        name: summarize(mask) for name, mask in groups.items()
    }
    result["excluded"] = summarize(excluded)
    return result
