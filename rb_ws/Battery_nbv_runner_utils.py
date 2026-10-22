"""NBV capture sequence runner for Battery_rb.py."""

import os
from datetime import datetime

import numpy as np

try:
    from rb_ws.Battery_nbv_sequence_utils import (
        make_nbv_frame_record,
        save_nbv_sequence_manifest,
    )
    from rb_ws.Battery_stage_setup_utils import update_simulation_frames
    from rb_ws.Battery_target_utils import (
        restore_target_dynamic_state,
        set_target_kinematic,
        set_target_pose,
    )
    from rb_ws.Battery_zivid_setup_utils import capture_current_zivid_frame
    from rb_ws.NBV_utils import (
        make_object_world_matrix,
        print_candidate_scores,
        score_candidate_poses,
    )
except ModuleNotFoundError:
    from Battery_nbv_sequence_utils import (
        make_nbv_frame_record,
        save_nbv_sequence_manifest,
    )
    from Battery_stage_setup_utils import update_simulation_frames
    from Battery_target_utils import (
        restore_target_dynamic_state,
        set_target_kinematic,
        set_target_pose,
    )
    from Battery_zivid_setup_utils import capture_current_zivid_frame
    from NBV_utils import (
        make_object_world_matrix,
        print_candidate_scores,
        score_candidate_poses,
    )


def restore_after_nbv_capture_sequence(
    timeline,
    simulation_app,
    stage,
    target_prim,
    target_prim_path,
    target_position,
    target_rotation,
    target_scale,
    pose_settle_update_count,
    log_prefix="[MUSTARD]",
):
    restore_target_dynamic_state(
        stage=stage,
        target_prim=target_prim,
        target_prim_path=target_prim_path,
        position=target_position,
        rotation_xyz_deg=target_rotation,
        scale=target_scale,
        log_prefix=log_prefix,
    )
    timeline.play()
    update_simulation_frames(
        simulation_app=simulation_app,
        update_count=pose_settle_update_count,
    )

    print(
        "[NBV] Timeline resumed after target physics restore."
    )


def run_nbv_capture_sequence(
    timeline,
    simulation_app,
    stage,
    target_prim,
    target_prim_path,
    target_position,
    target_scale,
    capture_poses,
    pose_settle_update_count,
    camera_prim,
    zivid_render_product,
    zivid_rgb_annotator,
    zivid_depth_annotator,
    zivid_semantic_annotator,
    zivid_instance_annotator,
    zivid_bbox_annotator,
    orchestrator,
    zivid_save_dir,
    intrinsic_matrix,
    image_size,
    nbv_roi_min,
    nbv_roi_max,
    nbv_pixel_stride,
    nbv_max_capture_count,
    nbv_min_expected_entropy,
    object_grid_leaf_size,
    object_grid_root_size,
    object_grid_origin,
    log_prefix="[MUSTARD]",
):
    print(
        "[ZIVID] Trying aligned captures..."
    )

    initial_pose_name, initial_rotation = (
        capture_poses[0]
    )

    timeline.pause()

    try:
        set_target_kinematic(
            stage=stage,
            target_prim=target_prim,
            target_prim_path=target_prim_path,
            enabled=True,
            log_prefix=log_prefix,
        )

        print(
            "[NBV] Initial capture:",
            initial_pose_name,
            initial_rotation,
        )

        set_target_pose(
            target_prim=target_prim,
            position=target_position,
            rotation_xyz_deg=initial_rotation,
            scale=target_scale,
            log_prefix=log_prefix,
        )

        update_simulation_frames(
            simulation_app=simulation_app,
            update_count=pose_settle_update_count,
        )

        (
            camera_world_matrix,
            initial_object_world_matrix,
        ) = capture_current_zivid_frame(
            frame_id=0,
            pose_name=initial_pose_name,
            camera_prim=camera_prim,
            object_prim=target_prim,
            zivid_render_product=zivid_render_product,
            zivid_rgb_annotator=zivid_rgb_annotator,
            zivid_depth_annotator=zivid_depth_annotator,
            zivid_semantic_annotator=zivid_semantic_annotator,
            zivid_instance_annotator=zivid_instance_annotator,
            zivid_bbox_annotator=zivid_bbox_annotator,
            orchestrator=orchestrator,
            zivid_save_dir=zivid_save_dir,
            intrinsic_matrix=intrinsic_matrix,
            pause_timeline=True,
            voxel_size=object_grid_leaf_size,
            object_grid_root_size=object_grid_root_size,
            object_grid_origin=object_grid_origin,
        )

        occupancy_map_dir = os.path.join(
            zivid_save_dir,
            "object_occupancy",
        )
        occupancy_latest_path = os.path.join(
            occupancy_map_dir,
            "object_occupancy_latest.npz",
        )
        nbv_manifest_path = os.path.join(
            occupancy_map_dir,
            "nbv_sequence_manifest.json",
        )
        max_capture_count = min(
            int(nbv_max_capture_count),
            len(capture_poses),
        )

        if max_capture_count < 1:
            raise ValueError(
                "NBV_MAX_CAPTURE_COUNT는 1 이상이어야 합니다."
            )

        sequence_started_at = datetime.now().isoformat()
        frame_records = [
            make_nbv_frame_record(
                frame_id=0,
                pose_name=initial_pose_name,
                rotation_xyz_deg=initial_rotation,
                occupancy_snapshot_path=os.path.join(
                    occupancy_map_dir,
                    "object_occupancy_000000.npz",
                ),
            )
        ]

        save_nbv_sequence_manifest(
            manifest_path=nbv_manifest_path,
            frame_records=frame_records,
            status="running",
            started_at=sequence_started_at,
            max_capture_count=max_capture_count,
        )

        candidate_position = (
            initial_object_world_matrix[:3, 3]
        )
        remaining_pose_rotations = {
            pose_name: rotation_xyz_deg
            for pose_name, rotation_xyz_deg
            in capture_poses[1:]
        }

        print("[NBV] Occupancy:", occupancy_latest_path)
        print("[NBV] ROI min:", nbv_roi_min)
        print("[NBV] ROI max:", nbv_roi_max)
        print("[NBV] Pixel stride:", nbv_pixel_stride)
        print("[NBV] Candidate pose pool:", len(capture_poses))
        print("[NBV] Maximum captures:", max_capture_count)

        while (
            len(frame_records) < max_capture_count
            and remaining_pose_rotations
        ):
            next_frame_id = len(frame_records)
            candidate_poses = [
                (
                    pose_name,
                    make_object_world_matrix(
                        position=candidate_position,
                        rotation_xyz_deg=np.asarray(
                            rotation_xyz_deg,
                            dtype=np.float64,
                        ),
                    ),
                )
                for pose_name, rotation_xyz_deg
                in remaining_pose_rotations.items()
            ]

            print(
                "\n[NBV] Selecting pose for frame:",
                next_frame_id,
            )
            print(
                "[NBV] Remaining candidate poses:",
                len(candidate_poses),
            )

            candidate_results = score_candidate_poses(
                occupancy_map=occupancy_latest_path,
                candidates=candidate_poses,
                camera_world_matrix=camera_world_matrix,
                intrinsic_matrix=intrinsic_matrix,
                image_size=image_size,
                pixel_stride=nbv_pixel_stride,
                roi_min=nbv_roi_min,
                roi_max=nbv_roi_max,
            )

            selected_nbv = print_candidate_scores(
                candidate_results
            )

            if (
                selected_nbv["entropy_score"]
                <= nbv_min_expected_entropy
            ):
                print(
                    "[NBV] Stopping: remaining visible entropy "
                    "is below the configured threshold:",
                    selected_nbv["entropy_score"],
                )
                break

            selected_pose_name = selected_nbv[
                "candidate_name"
            ]
            selected_rotation = remaining_pose_rotations[
                selected_pose_name
            ]

            print(
                "[NBV] Executing selected pose:",
                selected_pose_name,
                selected_rotation,
            )

            set_target_pose(
                target_prim=target_prim,
                position=target_position,
                rotation_xyz_deg=selected_rotation,
                scale=target_scale,
                log_prefix=log_prefix,
            )

            update_simulation_frames(
                simulation_app=simulation_app,
                update_count=pose_settle_update_count,
            )

            camera_world_matrix, _ = (
                capture_current_zivid_frame(
                    frame_id=next_frame_id,
                    pose_name=selected_pose_name,
                    camera_prim=camera_prim,
                    object_prim=target_prim,
                    zivid_render_product=zivid_render_product,
                    zivid_rgb_annotator=zivid_rgb_annotator,
                    zivid_depth_annotator=zivid_depth_annotator,
                    zivid_semantic_annotator=zivid_semantic_annotator,
                    zivid_instance_annotator=zivid_instance_annotator,
                    zivid_bbox_annotator=zivid_bbox_annotator,
                    orchestrator=orchestrator,
                    zivid_save_dir=zivid_save_dir,
                    intrinsic_matrix=intrinsic_matrix,
                    pause_timeline=True,
                    voxel_size=object_grid_leaf_size,
                    object_grid_root_size=object_grid_root_size,
                    object_grid_origin=object_grid_origin,
                )
            )

            frame_record = make_nbv_frame_record(
                frame_id=next_frame_id,
                pose_name=selected_pose_name,
                rotation_xyz_deg=selected_rotation,
                occupancy_snapshot_path=os.path.join(
                    occupancy_map_dir,
                    "object_occupancy_"
                    f"{next_frame_id:06d}.npz",
                ),
                candidate_result=selected_nbv,
            )
            previous_unknown_count = frame_records[-1][
                "unknown_count"
            ]

            frame_records.append(frame_record)
            remaining_pose_rotations.pop(
                selected_pose_name
            )

            print(
                "[NBV] Cumulative occupancy:",
                "unknown=",
                frame_record["unknown_count"],
                "free=",
                frame_record["free_count"],
                "occupied=",
                frame_record["occupied_count"],
                "newly observed=",
                previous_unknown_count
                - frame_record["unknown_count"],
            )

            save_nbv_sequence_manifest(
                manifest_path=nbv_manifest_path,
                frame_records=frame_records,
                status="running",
                started_at=sequence_started_at,
                max_capture_count=max_capture_count,
            )

        save_nbv_sequence_manifest(
            manifest_path=nbv_manifest_path,
            frame_records=frame_records,
            status="completed",
            started_at=sequence_started_at,
            max_capture_count=max_capture_count,
        )

        print(
            "[ZIVID] NBV capture sequence finished:",
            len(frame_records),
            "frames",
        )

        return frame_records

    finally:
        restore_after_nbv_capture_sequence(
            timeline=timeline,
            simulation_app=simulation_app,
            stage=stage,
            target_prim=target_prim,
            target_prim_path=target_prim_path,
            target_position=target_position,
            target_rotation=initial_rotation,
            target_scale=target_scale,
            pose_settle_update_count=pose_settle_update_count,
            log_prefix=log_prefix,
        )
