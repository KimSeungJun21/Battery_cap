from isaacsim import SimulationApp

simulation_app = SimulationApp(
    {"headless": False},
    experience="/home/seungjun/isaacsim/apps/isaacsim.exp.full.kit",
)

# SimulationApp 생성 후 import
import os
import carb
import carb.settings
import omni.timeline
import omni.usd

from pxr import UsdPhysics

import isaacsim.core.experimental.utils.stage as stage_utils
from isaacsim.core.utils.extensions import enable_extension
from isaacsim.storage.native import get_assets_root_path

# ============================================================
# ROS 2 Bridge 활성화
# ============================================================

enable_extension("isaacsim.ros2.bridge")
enable_extension("isaacsim.sensors.physics")
simulation_app.update()

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

# ============================================================
# RB5 Stage 열기
# ============================================================

RB_USD_PATH = "/home/seungjun/isaac_ws/Assets/urdf/rb5_850e_u_epick_isaac/rb5_850e_u_epick_isaac.usda"
RB_PRIM_PATH = "/rb5_850e_u"

if not os.path.isfile(RB_USD_PATH):
    simulation_app.close()
    raise FileNotFoundError(
        f"RB5 USD 파일을 찾을 수 없습니다: {RB_USD_PATH}"
    )

print("Opening RB5 stage:", RB_USD_PATH)

result = omni.usd.get_context().open_stage(RB_USD_PATH)

if not result:
    simulation_app.close()
    raise RuntimeError(
        f"RB5 USD Stage를 열지 못했습니다: {RB_USD_PATH}"
    )

# Stage 로딩 대기
for _ in range(30):
    simulation_app.update()

stage = omni.usd.get_context().get_stage()
robot_prim = stage.GetPrimAtPath(RB_PRIM_PATH)

print(f"{RB_PRIM_PATH} prim valid:", robot_prim.IsValid())

if not robot_prim.IsValid():
    simulation_app.close()
    raise RuntimeError(
        f"RB5 Prim을 찾지 못했습니다: {RB_PRIM_PATH}"
    )

print(
    f"{RB_PRIM_PATH} children:",
    [str(child.GetPath()) for child in robot_prim.GetChildren()],
)

# ============================================================
# Articulation Root 검색
# ============================================================

articulation_roots = []

for prim in stage.Traverse():
    prim_path = str(prim.GetPath())

    if (
        prim_path.startswith(RB_PRIM_PATH)
        and prim.HasAPI(UsdPhysics.ArticulationRootAPI)
    ):
        articulation_roots.append(prim_path)

if articulation_roots:
    print("RB5 Articulation Root:", articulation_roots[0])
else:
    print("WARNING: RB5 Articulation Root를 찾지 못했습니다.")
    print(
        "Expected root joint:",
        "/rb5_850e/Physics/root_joint",
    )







# ============================================================
# Ground Plane 추가
# ============================================================

stage_utils.add_reference_to_stage(
    usd_path=(
        assets_root_path
        + "/Isaac/Environments/Grid/default_environment.usd"
    ),
    path="/World/ground",
)

for _ in range(10):
    simulation_app.update()

# ============================================================
# 박스 생성 — 한 번만 호출
# ============================================================

from set_objects import set_box_conveyor

flat_box = set_box_conveyor(
    assets_root_path=assets_root_path
)

simulation_app.update()

#===========================================================
# ROS 2 Joint Graph 생성
#===========================================================   
# RB5 ROS2 Action Graph 생성
from rb_action_graph import create_ros2_joint_graph

create_ros2_joint_graph(
    robot_prim_path=articulation_roots[0],
    # Isaac의 실제 물리 상태
    joint_state_topic="/isaac_joint_states",

    # Isaac으로 보낼 JointState command
    joint_command_topic="/isaac_joint_commands",
)

simulation_app.update()



# ============================================================
# 시뮬레이션 시작
# ============================================================

timeline = omni.timeline.get_timeline_interface()
timeline.play()

for _ in range(5):
    simulation_app.update()


from isaacsim.core.experimental.prims import Articulation

rb_robot = Articulation(
    paths=articulation_roots[0]
)

print("RB5 DOF names:", rb_robot.dof_names)

# Gain을 적용하기 전 현재 자세를 목표 자세로 저장
current_joint_positions = rb_robot.get_dof_positions()

rb_robot.set_dof_position_targets(
    current_joint_positions
)

# 모든 관절을 position control 모드로 설정
rb_robot.switch_dof_control_mode("position")

# 순서:
# base, shoulder, elbow, wrist1, wrist2, wrist3
rb_robot.set_dof_gains(
    stiffnesses=[
        400.0,
        600.0,
        500.0,
        200.0,
        150.0,
        100.0,
    ],
    dampings=[
        40.0,
        80.0,
        60.0,
        20.0,
        15.0,
        10.0,
    ],
)

# URDF의 10 Nm보다 높게 설정
rb_robot.set_dof_max_efforts(
    max_efforts=[
        300.0,
        300.0,
        200.0,
        100.0,
        100.0,
        50.0,
    ]
)


stiffnesses, dampings = rb_robot.get_dof_gains()
max_efforts = rb_robot.get_dof_max_efforts()
max_velocities = rb_robot.get_dof_max_velocities()

stiffnesses = stiffnesses.numpy()[0]
dampings = dampings.numpy()[0]
max_efforts = max_efforts.numpy()[0]
max_velocities = max_velocities.numpy()[0]

print("\n========== RB5 Drive Settings ==========")

for index, joint_name in enumerate(rb_robot.dof_names):
    print(
        f"{joint_name}: "
        f"stiffness={stiffnesses[index]:.6f}, "
        f"damping={dampings[index]:.6f}, "
        f"max_effort={max_efforts[index]:.6f}, "
        f"max_velocity={max_velocities[index]:.6f}"
    )

print("========================================")


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

try:
    while simulation_app.is_running():
        simulation_app.update()

        positions, orientations = flat_box.get_world_poses()
        positions = positions.numpy()

        if box_is_moving:
            elapsed_time = (
                timeline.get_current_time()
                - box_move_start_time
            )

            progress = min(
                max(
                    elapsed_time / box_move_duration,
                    0.0,
                ),
                1.0,
            )

            positions[0, 0] = (
                box_start_x
                + (box_target_x - box_start_x)
                * progress
            )

            positions[0, 1] = (
                box_start_y
                + (box_target_y - box_start_y)
                * progress
            )

            if progress >= 1.0:
                box_is_moving = False

                print(
                    "FlatBox reached target:",
                    box_target_x,
                    box_target_y,
                )

        else:
            positions[0, 0] = box_target_x
            positions[0, 1] = box_target_y

        flat_box.set_world_poses(
            positions=positions,
            orientations=orientations,
        )

        # x/y 속도만 제거하고 z 방향 중력 운동은 유지
        linear_velocities, _ = flat_box.get_velocities()
        linear_velocities = linear_velocities.numpy()

        linear_velocities[0, 0] = 0.0
        linear_velocities[0, 1] = 0.0

        flat_box.set_velocities(
            linear_velocities=linear_velocities
        )

finally:
    timeline.stop()
    simulation_app.close()
