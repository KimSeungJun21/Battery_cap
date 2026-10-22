"""NBV capture sequence manifest helpers."""

import json
import os

import numpy as np


def save_nbv_sequence_manifest(
    manifest_path,
    frame_records,
    status,
    started_at,
    max_capture_count,
):
    """현재 NBV episode에서 실제 촬영된 snapshot 목록만 원자적으로 저장한다."""
    os.makedirs(
        os.path.dirname(manifest_path),
        exist_ok=True,
    )

    payload = {
        "status": status,
        "started_at": started_at,
        "max_capture_count": int(max_capture_count),
        "captured_frame_count": len(frame_records),
        "frames": frame_records,
    }
    temporary_path = manifest_path + ".tmp"

    with open(
        temporary_path,
        "w",
        encoding="utf-8",
    ) as manifest_file:
        json.dump(
            payload,
            manifest_file,
            indent=2,
            ensure_ascii=False,
        )

    os.replace(
        temporary_path,
        manifest_path,
    )

    print(
        "[NBV] Sequence manifest updated:",
        status,
        "frames=",
        len(frame_records),
        manifest_path,
    )


def make_nbv_frame_record(
    frame_id,
    pose_name,
    rotation_xyz_deg,
    occupancy_snapshot_path,
    candidate_result=None,
    joint_positions=None,
    tcp_position=None,
    tcp_orientation_xyzw=None,
    elapsed_s=None,
    elapsed_excl_render_s=None,
    surface_coverage=None,
):
    """누적 occupancy 상태와 해당 촬영 자세를 시각화용으로 기록한다.

    elapsed_s는 '물체를 들어올린 시점'(= NBV 시퀀스 시작) 기준 경과
    시간이다. 검사 동작이 시간 예산 안에 드는지 재는 기준이 된다.

    elapsed_excl_render_s는 거기서 Isaac의 Replicator 렌더 시간을 뺀
    값이다. 실기 Zivid는 오프스크린 렌더 비용을 치르지 않으므로,
    시뮬레이터에 종속되지 않는 시간 예산을 보려면 이쪽을 본다.
    """
    if not os.path.isfile(occupancy_snapshot_path):
        raise RuntimeError(
            "촬영 후 occupancy snapshot이 생성되지 않았습니다: "
            f"{occupancy_snapshot_path}"
        )

    with np.load(occupancy_snapshot_path) as occupancy_data:
        state = np.asarray(
            occupancy_data["state"],
            dtype=np.int8,
        )
        ray_count = (
            len(occupancy_data["ray_frame_ids"])
            if "ray_frame_ids" in occupancy_data.files
            else 0
        )

    frame_record = {
        "frame_id": int(frame_id),
        "pose_name": pose_name,
        "rotation_xyz_deg": np.asarray(
            rotation_xyz_deg,
            dtype=np.float64,
        ).tolist(),
        "occupancy_path": occupancy_snapshot_path,
        "unknown_count": int(np.count_nonzero(state == -1)),
        "free_count": int(np.count_nonzero(state == 0)),
        "occupied_count": int(np.count_nonzero(state == 1)),
        "ray_count": int(ray_count),
    }

    # 물체를 들어올린 시점(= NBV 시퀀스 시작) 기준 경과 시간.
    if elapsed_s is not None:
        frame_record["elapsed_s"] = round(float(elapsed_s), 3)

    if elapsed_excl_render_s is not None:
        frame_record["elapsed_excl_render_s"] = round(
            float(elapsed_excl_render_s), 3
        )

    if surface_coverage is not None:
        frame_record["surface_coverage"] = round(
            float(surface_coverage), 4
        )

    if joint_positions is not None:
        frame_record["joint_positions_rad"] = {
            name: float(value)
            for name, value in joint_positions.items()
        }

    if tcp_position is not None:
        frame_record["tcp_position"] = [
            float(component) for component in tcp_position
        ]

    if tcp_orientation_xyzw is not None:
        frame_record["tcp_orientation_xyzw"] = [
            float(component) for component in tcp_orientation_xyzw
        ]

    if candidate_result is not None:
        frame_record["visible_entropy"] = float(
            candidate_result["entropy_score"]
        )
        frame_record["expected_entropy"] = float(
            candidate_result["entropy_score"]
        )
        frame_record["visible_unknown_count"] = int(
            candidate_result["unknown_visible_count"]
        )
        frame_record["visible_free_count"] = int(
            candidate_result["free_visible_count"]
        )
        frame_record["visible_occupied_count"] = int(
            candidate_result["occupied_visible_count"]
        )
        frame_record["visible_voxel_count"] = int(
            candidate_result["visible_voxel_count"]
        )
        frame_record["expected_unknown_visible_count"] = float(
            candidate_result["expected_unknown_visible_count"]
        )

    return frame_record
