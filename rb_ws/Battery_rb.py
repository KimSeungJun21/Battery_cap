import argparse
import os
import sys


def apply_battery_cli_options():
    """Consume Battery-specific options before Kit parses its own argv."""
    parser = argparse.ArgumentParser(
        description="Isaac Sim battery inspection scenario."
    )
    parser.add_argument(
        "--target-object",
        choices=("batterycap", "mustard"),
        help="Inspection target (default: batterycap).",
    )
    parser.add_argument(
        "--collision",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable target-object PhysX collision.",
    )
    parser.add_argument(
        "--gravity",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable gravity on the target rigid body.",
    )
    options, remaining = parser.parse_known_args()

    if options.target_object is not None:
        os.environ["BATTERY_TARGET_OBJECT"] = options.target_object
    if options.collision is not None:
        os.environ["BATTERY_TARGET_COLLISION_ENABLED"] = (
            "1" if options.collision else "0"
        )
    if options.gravity is not None:
        os.environ["BATTERY_TARGET_GRAVITY_ENABLED"] = (
            "1" if options.gravity else "0"
        )

    # Preserve any ROS/Kit options that do not belong to this script.
    sys.argv = [sys.argv[0], *remaining]


apply_battery_cli_options()

from isaacsim import SimulationApp

simulation_app = SimulationApp(
    {"headless": False},
    experience="/home/seungjun/isaacsim/apps/isaacsim.exp.full.kit",
)

import carb
import carb.settings
import omni.timeline
import omni.usd

from isaacsim.core.utils.extensions import enable_extension
from isaacsim.storage.native import get_assets_root_path
from Battery_config import (
    AUTO_RUN_NBV_CAPTURE,
    LIFT_DETECTION_HEIGHT,
    LIFT_DETECTION_RISE_M,
    LIFT_STABILITY_CHECK_COUNT,
    LIFT_STABILITY_TOLERANCE,
    NBV_MAX_CAPTURE_COUNT,
    NBV_BACKGROUND_SCORE_WEIGHT,
    NBV_MIN_EXPECTED_ENTROPY,
    NBV_FRAME_MARGIN_PX,
    NBV_MIN_IN_FRAME_FRACTION,
    NBV_MIN_SURFACE_COVERAGE_GAIN,
    NBV_MIN_SURFACE_COVERAGE_GAIN_CONSECUTIVE,
    NBV_MIN_VIEW_SEPARATION_DEG,
    NBV_MAX_CONSECUTIVE_MOVE_FAILURES,
    NBV_MOVE_RESULT_TIMEOUT,
    NBV_SURFACE_COVERAGE_TARGET,
    NBV_SURFACE_BOTTOM_SCORE_WEIGHT,
    NBV_SURFACE_SIDE_SCORE_WEIGHT,
    NBV_SURFACE_TOP_SCORE_WEIGHT,
    NBV_SIDE_SURFACE_COVERAGE_TARGET,
    NBV_SURFEL_MATCH_DISTANCE,
    NBV_SURFEL_MIN_FACING_COS,
    NBV_SURFEL_OCCLUSION_TOLERANCE,
    NBV_SURFEL_PIXEL_BIN_SIZE,
    NBV_SURFEL_SAMPLE_COUNT,
    NBV_TOP_SURFACE_COVERAGE_TARGET,
    NBV_PIXEL_STRIDE,
    PICK_DETECTION_CHECK_INTERVAL,
    PUBLISH_TARGET_OBJECT_POSE,
    POSE_SETTLE_UPDATE_COUNT,
    RB_ARM_LINK_COLLISION_ENABLED,
    RB_PRIM_PATH,
    RB_USD_PATH,
    ENABLE_SUCTION_GRIPPER,
    SUCTION_COAXIAL_FORCE_LIMIT,
    SUCTION_CUP_PRIM_PATH,
    SUCTION_GRIPPER_TOPIC,
    SUCTION_MAX_GRIP_DISTANCE,
    SUCTION_PROXY_INERTIA,
    SUCTION_PROXY_MASS,
    SUCTION_RETRY_INTERVAL,
    SUCTION_SHEAR_FORCE_LIMIT,
    SUCTION_TCP_PRIM_PATH,
    TCP_PRIM_PATH,
    TARGET_COLLISION_APPROXIMATION,
    TARGET_COLLISION_ENABLED,
    TARGET_GRAVITY_ENABLED,
    TARGET_OBJECT_POSE_FRAME_ID,
    TARGET_OBJECT_GRASP_Z_OFFSET,
    TARGET_OBJECT_POSE_PUBLISH_HZ,
    TARGET_OBJECT_POSE_TOPIC,
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
    ZIVID_SAVE_DIR,
    STAGE_LIGHT_DIM_FACTOR,
    get_target_object_config,
)
from Battery_camera_frustum_utils import draw_camera_frustum
from Battery_logging_utils import setup_run_logging
from Battery_manual_capture_utils import ManualCaptureController
from Battery_nbv_arm_runner_utils import (
    create_nbv_move_client,
    run_nbv_capture_sequence_with_arm,
)
from Battery_robot_setup_utils import (
    configure_rb5_position_control,
    create_rb_articulation,
    set_rb5_arm_link_collisions_enabled,
)
from Battery_stage_setup_utils import (
    add_ground_plane,
    find_articulation_roots,
    open_stage_or_raise,
    print_prim_children,
    print_relevant_stage_prims,
    require_prim,
    update_simulation_frames,
)
from Battery_defect_utils import setup_defects
from Battery_lighting_utils import setup_inspection_lighting
from Battery_target_utils import (
    add_referenced_target_prim,
    add_target_labels,
    configure_target_physics,
    set_target_pose,
)
from Battery_zivid_setup_utils import (
    compute_zivid_intrinsic,
    create_zivid_render_stack,
)


RUN_LOG_PATH = setup_run_logging()

ROS_ENVIRONMENT_KEYS = [
    "ROS_DISTRO",
    "ROS_DOMAIN_ID",
    "RMW_IMPLEMENTATION",
    "ROS_AUTOMATIC_DISCOVERY_RANGE",
    "ROS_LOCALHOST_ONLY",
    "FASTRTPS_DEFAULT_PROFILES_FILE",
    "CYCLONEDDS_URI",
]


def print_ros_environment_summary():
    print("[ROS ENV] Isaac Sim process ROS environment:")

    for key in ROS_ENVIRONMENT_KEYS:
        print(
            f"[ROS ENV] {key}="
            f"{os.environ.get(key, '<unset>')}"
        )


print_ros_environment_summary()

# ============================================================
# ROS 2 Bridge 활성화
# ============================================================

enable_extension("isaacsim.ros2.bridge")
enable_extension("isaacsim.sensors.physics")
enable_extension("isaacsim.robot.surface_gripper")
enable_extension("isaacsim.util.debug_draw")
simulation_app.update()

from Battery_object_pose_publisher import TargetObjectPosePublisher
from Battery_suction_gripper_utils import (
    SuctionGripperController,
    report_grip_mass_ratio,
    setup_epick_surface_gripper,
)

# ============================================================
# Isaac Sim Asset Root 설정
# ============================================================

settings = carb.settings.get_settings()
settings.set(
    "/persistent/isaac/asset_root/default",
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/"
    "Assets/Isaac/6.0",
)

assets_root_path = get_assets_root_path()

if assets_root_path is None:
    simulation_app.close()
    raise RuntimeError("Could not find Isaac Sim assets root path")

print("Assets root:", assets_root_path)

import omni.replicator.core as rep
# ============================================================
# RB5 Stage 열기
# ============================================================

stage = open_stage_or_raise(
    simulation_app=simulation_app,
    usd_path=RB_USD_PATH,
)

update_simulation_frames(
    simulation_app=simulation_app,
    update_count=30,
)

print_relevant_stage_prims(
    stage=stage,
    keywords=[
        "rb5",
        "battery",
        "robot",
    ],
)

robot_prim = require_prim(
    stage=stage,
    prim_path=RB_PRIM_PATH,
    description="RB5",
)
print_prim_children(
    prim=robot_prim,
    prim_path=RB_PRIM_PATH,
)
articulation_roots = find_articulation_roots(
    stage=stage,
    robot_prim_path=RB_PRIM_PATH,
)

# Imported RB body colliders are instance proxies, so override them before
# PhysX parses the stage.  Explicit link1..link6 paths leave link0 and the
# EPick subtree (including its suction cup collider) unchanged.
set_rb5_arm_link_collisions_enabled(
    stage=stage,
    robot_prim_path=RB_PRIM_PATH,
    enabled=RB_ARM_LINK_COLLISION_ENABLED,
)

# ============================================================
# Epick Suction Gripper
# ============================================================
# Must run before timeline.play() - PhysX only registers gripper/joint
# prims that exist at its initial scene parse.

suction_gripper_controller = None

if ENABLE_SUCTION_GRIPPER:
    suction_gripper_prim_path = setup_epick_surface_gripper(
        stage=stage,
        suction_cup_prim_path=SUCTION_CUP_PRIM_PATH,
        tcp_prim_path=SUCTION_TCP_PRIM_PATH,
        max_grip_distance=SUCTION_MAX_GRIP_DISTANCE,
        coaxial_force_limit=SUCTION_COAXIAL_FORCE_LIMIT,
        shear_force_limit=SUCTION_SHEAR_FORCE_LIMIT,
        retry_interval=SUCTION_RETRY_INTERVAL,
        proxy_mass=SUCTION_PROXY_MASS,
        proxy_inertia=SUCTION_PROXY_INERTIA,
        log_prefix="[SUCTION]",
    )
    suction_gripper_controller = SuctionGripperController(
        gripper_prim_path=suction_gripper_prim_path,
        topic_name=SUCTION_GRIPPER_TOPIC,
        log_prefix="[SUCTION]",
    )
else:
    print(
        "[SUCTION] Suction gripper disabled: "
        "BATTERY_ENABLE_SUCTION_GRIPPER=0"
    )

# ============================================================
# Ground Plane 추가
# ============================================================

add_ground_plane(
    assets_root_path=assets_root_path,
    simulation_app=simulation_app,
)

os.makedirs(
    ZIVID_SAVE_DIR,
    exist_ok=True,
)

zivid_prim = require_prim(
    stage=stage,
    prim_path=ZIVID_CAMERA_PATH,
    description="Zivid Camera",
)
print(
    "Zivid Camera Prim:",
    ZIVID_CAMERA_PATH
)

# ============================================================
# 검사 조명 리그
# ============================================================
# 스테이지 기본 조명은 전방향이라 정반사 구조를 씻어낸다.
# roughness만 다른 결함은 정반사 근처에서만 대비가 생기므로,
# 프로젝터를 주 광원으로 세우고 기존 조명을 감쇠시킨다.
# 렌더 스택보다 먼저 구성해야 첫 프레임부터 반영된다.

lighting_rig = setup_inspection_lighting(
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

simulation_app.update()

# ============================================================
# Zivid Render Product + Annotators
# ============================================================

(
    zivid_render_product,
    zivid_rgb_annotator,
    zivid_depth_annotator,
    zivid_semantic_annotator,
    zivid_instance_annotator,
    zivid_bbox_annotator,
) = create_zivid_render_stack(
    rep_module=rep,
    camera_path=ZIVID_CAMERA_PATH,
    resolution=ZIVID_RESOLUTION,
)

simulation_app.update()

print(
    "Zivid CameraSensor initialized"
)

# ============================================================
# Camera Intrinsic 계산
# ============================================================

K = compute_zivid_intrinsic(
    camera_prim=zivid_prim,
    resolution=ZIVID_RESOLUTION,
)

zivid_frustum_draw = draw_camera_frustum(
    camera_prim=zivid_prim,
    intrinsic_matrix=K,
    image_size=ZIVID_RESOLUTION,
)

# ============================================================
# Target Object 생성
# ============================================================
print("[DEBUG 1] Intrinsic finished", flush=True)


# ============================================================
# 선택한 검사 대상 생성
# ============================================================

target_config = get_target_object_config(
    assets_root_path
)
target_prim_path = target_config.prim_path
target_log_prefix = target_config.log_prefix

print(
    "[DEBUG 2] Before target object:",
    target_config.key,
    "asset=",
    target_config.usd_path,
    "collision=",
    TARGET_COLLISION_ENABLED,
    "gravity=",
    TARGET_GRAVITY_ENABLED,
    flush=True,
)

target_prim = add_referenced_target_prim(
    stage=stage,
    simulation_app=simulation_app,
    usd_path=target_config.usd_path,
    prim_path=target_prim_path,
    description=target_config.description,
)

set_target_pose(
    target_prim=target_prim,
    position=target_config.position,
    rotation_xyz_deg=target_config.neutral_rotation,
    scale=target_config.scale,
    log_prefix=target_log_prefix,
)

configure_target_physics(
    stage=stage,
    target_prim=target_prim,
    target_prim_path=target_prim_path,
    collision_enabled=TARGET_COLLISION_ENABLED,
    gravity_enabled=TARGET_GRAVITY_ENABLED,
    create_missing_colliders=(
        target_config.create_missing_colliders
    ),
    collision_approximation=TARGET_COLLISION_APPROXIMATION,
    log_prefix=target_log_prefix,
)

add_target_labels(
    stage=stage,
    target_prim=target_prim,
    target_prim_path=target_prim_path,
    class_label="target_object",
    object_label=target_config.object_label,
)

# add_target_labels()가 대상 prim 아래 모든 Mesh에 class=target_object를
# 덮어쓰므로, decal의 class=defect 라벨이 살아남으려면 반드시 그 뒤에서
# 호출해야 한다. BATTERY_ENABLE_DEFECTS=0이면 통째로 건너뛴다.
# 그리퍼 설정은 timeline.play() 전에 끝나야 하고 물체는 그 뒤에
# 생성되므로, 질량비 진단은 여기서 따로 한다. 질량비가 크게 어긋나면
# attachment joint가 불안정해져 이동 중 물체를 놓친다.
if suction_gripper_controller is not None:
    report_grip_mass_ratio(
        stage=stage,
        payload_prim_path=target_prim_path,
        proxy_mass=SUCTION_PROXY_MASS,
        log_prefix="[SUCTION]",
    )

defect_records = setup_defects(
    stage=stage,
    target_prim_path=target_prim_path,
    simulation_app=simulation_app,
    target_object_type=target_config.key,
    long_axis_index=target_config.defect_long_axis_index,
    theta_zero_axis_index=(
        target_config.defect_theta_zero_axis_index
    ),
    theta_ninety_axis_index=(
        target_config.defect_theta_ninety_axis_index
    ),
    log_prefix=f"{target_log_prefix}[DEFECT]",
)

print(
    "[DEBUG 3] After target object:",
    target_config.key,
    target_config.usd_path,
    flush=True,
)


# ============================================================
# ROS 2 Joint Graph
# ============================================================

print("[DEBUG 4] Before importing rb_action_graph", flush=True)

from rb_action_graph import create_ros2_joint_graph

print("[DEBUG 5] rb_action_graph imported", flush=True)

print("[DEBUG 6] Before create_ros2_joint_graph", flush=True)

create_ros2_joint_graph(
    robot_prim_path=articulation_roots[0],
    joint_state_topic="/isaac_joint_states",
    joint_command_topic="/isaac_joint_commands",
)

print("[DEBUG 7] After create_ros2_joint_graph", flush=True)


print("[DEBUG 8] Before simulation_app.update()", flush=True)

simulation_app.update()

print("[DEBUG 9] After simulation_app.update()", flush=True)


# ============================================================
# Simulation 시작
# ============================================================

timeline = omni.timeline.get_timeline_interface()

print("[DEBUG 10] Before timeline.play()", flush=True)

timeline.play()

print("[DEBUG 11] After timeline.play()", flush=True)

for i in range(30):

    if i % 5 == 0:
        print(
            f"[DEBUG 12] startup update {i}/30",
            flush=True,
        )

    simulation_app.update()

print("[DEBUG 13] startup updates finished", flush=True)

rb_robot = create_rb_articulation(
    articulation_root=articulation_roots[0],
    before_log="[DEBUG 14] Before Articulation()",
    after_log="[DEBUG 15] Articulation created",
    flush=True,
)
configure_rb5_position_control(
    rb_robot
)

# from set_objects import set_box_conveyor

# flat_box = set_box_conveyor(
#     assets_root_path=assets_root_path
# )

# simulation_app.update()


# ============================================================
# 박스 이동 설정
# ============================================================

box_start_x = -1.0
box_start_y = 0.56

box_target_x = -0.12
box_target_y = 0.56

box_move_duration = 40.0
box_move_start_time = timeline.get_current_time()
box_is_moving = True

# ============================================================
# Simulation Loop
# ============================================================

# ============================================================
# Simulation Loop
# ============================================================

frame_count = 0
nbv_started = False
nbv_move_client = None
last_object_height = None
height_stable_count = 0
# 흡착 전 안착 높이. 들어올림 판정의 기준선이 된다.
object_rest_height = None
target_pose_publisher = None

if PUBLISH_TARGET_OBJECT_POSE:
    robot_link0_prim = stage.GetPrimAtPath(
        articulation_roots[0]
    )

    # The grasp pose is published from the face the cup actually lands on,
    # which for the batterycap is the flat middle board rather than the whole
    # assembly. Falls back to the assembly root if the sub-prim is missing.
    grasp_prim = target_prim
    grasp_suffix = target_config.grasp_prim_path_suffix

    if grasp_suffix:
        grasp_prim_path = target_prim_path + grasp_suffix
        candidate_prim = stage.GetPrimAtPath(
            grasp_prim_path
        )

        if candidate_prim.IsValid():
            grasp_prim = candidate_prim
            print(
                target_log_prefix,
                "Grasp pose source:",
                grasp_prim_path,
            )
        else:
            print(
                target_log_prefix,
                "WARNING: grasp prim을 찾지 못해 대상 root를 씁니다:",
                grasp_prim_path,
            )

    target_pose_publisher = TargetObjectPosePublisher(
        target_prim=grasp_prim,
        reference_prim=robot_link0_prim,
        topic_name=TARGET_OBJECT_POSE_TOPIC,
        frame_id=TARGET_OBJECT_POSE_FRAME_ID,
        publish_hz=TARGET_OBJECT_POSE_PUBLISH_HZ,
        pose_mode=target_config.grasp_pose_mode,
        grasp_z_offset=TARGET_OBJECT_GRASP_Z_OFFSET,
        log_prefix=target_log_prefix,
    )
    target_pose_publisher.publish_if_due(
        force=True
    )
else:
    print(
        f"{target_log_prefix} Target object pose publisher disabled: "
        "BATTERY_PUBLISH_TARGET_OBJECT_POSE=0"
    )

manual_capture_controller = ManualCaptureController(
    camera_prim=zivid_prim,
    object_prim=target_prim,
    zivid_render_product=zivid_render_product,
    zivid_rgb_annotator=zivid_rgb_annotator,
    zivid_depth_annotator=zivid_depth_annotator,
    zivid_semantic_annotator=zivid_semantic_annotator,
    zivid_instance_annotator=zivid_instance_annotator,
    zivid_bbox_annotator=zivid_bbox_annotator,
    orchestrator=rep.orchestrator,
    zivid_save_dir=ZIVID_SAVE_DIR,
    intrinsic_matrix=K,
    voxel_size=target_config.object_grid_leaf_size,
    object_grid_root_size=target_config.object_grid_root_size,
    object_grid_origin=target_config.object_grid_origin,
)

try:
    while simulation_app.is_running():

        simulation_app.update()

        frame_count += 1

        if target_pose_publisher is not None:
            target_pose_publisher.publish_if_due()

        if suction_gripper_controller is not None:
            suction_gripper_controller.spin_once()

        manual_capture_controller.spin_once()

        # 로봇이 흡착으로 실제로 들어올렸는지 감지되면 NBV를 시작한다
        # (물체를 직접 텔레포트하는 대신 팔을 움직여 재관측한다).
        if (
            not nbv_started
            and AUTO_RUN_NBV_CAPTURE
            and suction_gripper_controller is not None
            and frame_count % PICK_DETECTION_CHECK_INTERVAL == 0
        ):
            gripped_objects = (
                suction_gripper_controller.gripper_interface
                .get_gripped_objects(
                    suction_gripper_controller.gripper_prim_path
                )
            )

            object_height = omni.usd.get_world_transform_matrix(
                target_prim
            ).ExtractTranslation()[2]

            if target_prim_path not in gripped_objects:
                # 아직 안 잡혔다. 현재 높이를 안착 기준선으로 계속 갱신한다.
                object_rest_height = object_height
                height_stable_count = 0
                last_object_height = None
            else:
                if object_rest_height is None:
                    # 첫 폴링부터 잡혀 있었다면 지금 높이를 기준으로 삼는다.
                    object_rest_height = object_height

                lift_rise = object_height - object_rest_height
                lifted = (
                    lift_rise >= LIFT_DETECTION_RISE_M
                    and object_height >= LIFT_DETECTION_HEIGHT
                )

                if lifted:
                    if (
                        last_object_height is not None
                        and abs(object_height - last_object_height)
                        < LIFT_STABILITY_TOLERANCE
                    ):
                        height_stable_count += 1
                    else:
                        height_stable_count = 0
                    last_object_height = object_height
                else:
                    height_stable_count = 0
                    last_object_height = None

                if height_stable_count >= LIFT_STABILITY_CHECK_COUNT:
                    nbv_started = True
                    print(
                        "[NBV] Pick settled (height=",
                        round(object_height, 4),
                        "rest=",
                        round(object_rest_height, 4),
                        "rise=",
                        round(lift_rise, 4),
                        ") - starting arm-driven NBV capture sequence",
                    )

                    if nbv_move_client is None:
                        nbv_move_client = create_nbv_move_client(
                            simulation_app=simulation_app,
                            result_timeout=NBV_MOVE_RESULT_TIMEOUT,
                            log_prefix="[NBV-ARM]",
                        )

                    # Debug-draw lines are visible to this render product.
                    # Remove the viewport frustum before collecting NBV data.
                    zivid_frustum_draw.clear_lines()
                    print(
                        "[CAMERA] Frustum lines cleared before NBV capture."
                    )

                    run_nbv_capture_sequence_with_arm(
                        stage=stage,
                        simulation_app=simulation_app,
                        move_client=nbv_move_client,
                        rb_robot=rb_robot,
                        target_prim=target_prim,
                        target_prim_path=target_prim_path,
                        tcp_prim_path=TCP_PRIM_PATH,
                        robot_base_prim_path=articulation_roots[0],
                        capture_poses=target_config.arm_capture_poses,
                        pose_settle_update_count=POSE_SETTLE_UPDATE_COUNT,
                        camera_prim=zivid_prim,
                        zivid_render_product=zivid_render_product,
                        zivid_rgb_annotator=zivid_rgb_annotator,
                        zivid_depth_annotator=zivid_depth_annotator,
                        zivid_semantic_annotator=zivid_semantic_annotator,
                        zivid_instance_annotator=zivid_instance_annotator,
                        zivid_bbox_annotator=zivid_bbox_annotator,
                        orchestrator=rep.orchestrator,
                        zivid_save_dir=ZIVID_SAVE_DIR,
                        intrinsic_matrix=K,
                        image_size=ZIVID_RESOLUTION,
                        nbv_roi_min=target_config.nbv_roi_min,
                        nbv_roi_max=target_config.nbv_roi_max,
                        nbv_pixel_stride=NBV_PIXEL_STRIDE,
                        nbv_max_capture_count=NBV_MAX_CAPTURE_COUNT,
                        nbv_min_expected_entropy=NBV_MIN_EXPECTED_ENTROPY,
                        nbv_min_view_separation_deg=(
                            NBV_MIN_VIEW_SEPARATION_DEG
                        ),
                        nbv_surface_coverage_target=(
                            NBV_SURFACE_COVERAGE_TARGET
                        ),
                        nbv_top_surface_coverage_target=(
                            NBV_TOP_SURFACE_COVERAGE_TARGET
                        ),
                        nbv_side_surface_coverage_target=(
                            NBV_SIDE_SURFACE_COVERAGE_TARGET
                        ),
                        nbv_min_surface_coverage_gain=(
                            NBV_MIN_SURFACE_COVERAGE_GAIN
                        ),
                        nbv_min_surface_coverage_gain_consecutive=(
                            NBV_MIN_SURFACE_COVERAGE_GAIN_CONSECUTIVE
                        ),
                        nbv_bottom_surface_coverage_target=(
                            target_config.bottom_surface_coverage_target
                        ),
                        nbv_max_consecutive_move_failures=(
                            NBV_MAX_CONSECUTIVE_MOVE_FAILURES
                        ),
                        nbv_view_reference_local_point=(
                            target_config.view_reference_local_point
                        ),
                        nbv_min_in_frame_fraction=(
                            NBV_MIN_IN_FRAME_FRACTION
                        ),
                        nbv_frame_margin_px=NBV_FRAME_MARGIN_PX,
                        nbv_background_score_weight=(
                            NBV_BACKGROUND_SCORE_WEIGHT
                        ),
                        nbv_surface_top_score_weight=(
                            NBV_SURFACE_TOP_SCORE_WEIGHT
                        ),
                        nbv_surface_bottom_score_weight=(
                            NBV_SURFACE_BOTTOM_SCORE_WEIGHT
                        ),
                        nbv_surface_side_score_weight=(
                            NBV_SURFACE_SIDE_SCORE_WEIGHT
                        ),
                        nbv_surfel_sample_count=NBV_SURFEL_SAMPLE_COUNT,
                        nbv_surfel_match_distance=(
                            NBV_SURFEL_MATCH_DISTANCE
                        ),
                        nbv_surfel_min_facing_cos=(
                            NBV_SURFEL_MIN_FACING_COS
                        ),
                        nbv_surfel_pixel_bin_size=(
                            NBV_SURFEL_PIXEL_BIN_SIZE
                        ),
                        nbv_surfel_occlusion_tolerance=(
                            NBV_SURFEL_OCCLUSION_TOLERANCE
                        ),
                        object_grid_leaf_size=(
                            target_config.object_grid_leaf_size
                        ),
                        object_grid_root_size=(
                            target_config.object_grid_root_size
                        ),
                        object_grid_origin=(
                            target_config.object_grid_origin
                        ),
                        is_still_gripped=(
                            lambda: target_prim_path
                            in suction_gripper_controller.gripper_interface
                            .get_gripped_objects(
                                suction_gripper_controller.gripper_prim_path
                            )
                        ),
                        log_prefix=f"{target_log_prefix}[NBV-ARM]",
                    )

finally:
    if target_pose_publisher is not None:
        target_pose_publisher.destroy()

    if suction_gripper_controller is not None:
        suction_gripper_controller.destroy()

    if nbv_move_client is not None:
        nbv_move_client.destroy()

    timeline.stop()
    simulation_app.close()
