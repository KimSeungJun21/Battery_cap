from isaacsim import SimulationApp

#simulation_app = SimulationApp({"headless": False})
simulation_app = SimulationApp(
    {"headless": False},
    experience="/home/seungjun/isaacsim/apps/isaacsim.exp.full.kit",
)


# SimulationApp 생성 후 import
from isaacsim.core.utils.extensions import enable_extension

# ROS2 Bridge 먼저 켜기
enable_extension("isaacsim.ros2.bridge")
simulation_app.update()

import numpy as np
import omni.timeline
import omni.usd
import carb

import isaacsim.core.experimental.utils.stage as stage_utils
from isaacsim.storage.native import get_assets_root_path

# Cloud asset root를 강제로 사용하고 싶으면 이 줄을 켜도 됨
# 기존에 local asset root가 잘못 잡혀 있으면 이게 중요함
import carb.settings
settings = carb.settings.get_settings()
settings.set(
    "/persistent/isaac/asset_root/default",
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0",
)


assets_root_path = get_assets_root_path()

if assets_root_path is None:
    carb.log_error("Could not find Isaac Sim assets root path")
    simulation_app.close()
    raise RuntimeError("Could not find Isaac Sim assets root path")

print("Assets root:", assets_root_path)

# Ground plane
stage_utils.add_reference_to_stage(
    usd_path=assets_root_path + "/Isaac/Environments/Grid/default_environment.usd",
    path="/World/ground",
)
from set_objects import set_objects
flat_box = set_objects(assets_root_path=assets_root_path)

# Start simulation
timeline = omni.timeline.get_timeline_interface()
timeline.play()
simulation_app.update()

stage = omni.usd.get_context().get_stage()
robot_prim = stage.GetPrimAtPath("/World/UR5e")
print("/World/UR5e prim valid:", robot_prim.IsValid())
if robot_prim.IsValid():
    print("/World/UR5e children:", [str(child.GetPath()) for child in robot_prim.GetChildren()])

from action_graph import create_ros2_joint_graph
create_ros2_joint_graph(robot_prim_path="/World/UR5e/root_joint")

box_start_x = -1.0
box_start_y = 0.56
box_target_x = -0.12
box_target_y = 0.56
box_move_duration = 40.0  # 시뮬레이션 시간 기준 40초
box_move_start_time = timeline.get_current_time()
box_is_moving = True

while simulation_app.is_running():
    simulation_app.update()

    positions, orientations = flat_box.get_world_poses()
    positions = positions.numpy()

    if box_is_moving:
        elapsed_time = timeline.get_current_time() - box_move_start_time
        progress = min(max(elapsed_time / box_move_duration, 0.0), 1.0)
        positions[0, 0] = box_start_x + (box_target_x - box_start_x) * progress
        positions[0, 1] = box_start_y + (box_target_y - box_start_y) * progress

        if progress >= 1.0:
            box_is_moving = False
            print("FlatBox reached target (x, y):", box_target_x, box_target_y)
    else:
        # 도착 후에도 충돌이나 마찰로 x/y 위치가 밀리지 않도록 유지한다.
        positions[0, 0] = box_target_x
        positions[0, 1] = box_target_y

    flat_box.set_world_poses(positions=positions, orientations=orientations)

    # x/y는 직접 제어하고 z 속도는 보존해 중력과 낙하를 유지한다.
    linear_velocities, _ = flat_box.get_velocities()
    linear_velocities = linear_velocities.numpy()
    linear_velocities[0, 0] = 0.0
    linear_velocities[0, 1] = 0.0
    flat_box.set_velocities(linear_velocities=linear_velocities)

simulation_app.close()
