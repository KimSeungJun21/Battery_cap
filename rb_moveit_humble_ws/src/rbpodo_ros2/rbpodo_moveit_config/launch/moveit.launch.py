import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.conditions import IfCondition, UnlessCondition
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
from moveit_configs_utils import MoveItConfigsBuilder


robot_ip = LaunchConfiguration("robot_ip")
use_fake_hardware = LaunchConfiguration("use_fake_hardware")
fake_sensor_commands = LaunchConfiguration("fake_sensor_commands")
model_id = LaunchConfiguration("model_id")
cb_simulation = LaunchConfiguration("cb_simulation")
use_isaac_bridge = LaunchConfiguration("use_isaac_bridge")
use_sim_time = LaunchConfiguration("use_sim_time")

def generate_launch_description():

    declared_arguments = []
    declared_arguments.append(
        DeclareLaunchArgument(
            "rviz_config",
            default_value="moveit.rviz",
            description="RViz configuration file",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "robot_ip",
            default_value="10.0.2.7",
            description="RB Cobot Control Box IP Address",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "use_fake_hardware",
            default_value="true",
            description="True if there's no RB Cobot Control Box",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "fake_sensor_commands",
            default_value="false",
            description="True when use fake sensor commands",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "cb_simulation",
            default_value="Simulation",
            description="Select RB Control Box mode, Simulation or Real",
        )
    )

    declared_arguments.append(
        DeclareLaunchArgument(
            "model_id",
            default_value="rb5_850e_u",
            description="RB Series currently using",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "use_isaac_bridge",
            default_value="true",
            description=(
                "Use Isaac Sim JointState bridge instead of ros2_control "
                "fake hardware for trajectory execution"
            ),
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="true",
            description="Use /clock time from Isaac Sim",
        )
    )
    return LaunchDescription(
        declared_arguments + [OpaqueFunction(function=launch_setup)]
    )


def launch_setup(context, *args, **kwargs):
    mappings = {
        "robot_ip": robot_ip,
        "use_fake_hardware": use_fake_hardware,
        "fake_sensor_commands": fake_sensor_commands,
        "model_id": model_id,
        "cb_simulation": cb_simulation,
    }

    moveit_config = (
        MoveItConfigsBuilder("rbpodo")
        .robot_description(file_path="config/rbpodo.urdf.xacro", mappings=mappings)
        .trajectory_execution(file_path="config/moveit_controllers.yaml")
        .planning_scene_monitor(
            publish_robot_description=True, publish_robot_description_semantic=True
        )
        .planning_pipelines(
            pipelines=["ompl", "chomp", "pilz_industrial_motion_planner"]
        )
        .to_moveit_configs()
    )

    # Start the actual move_group node/action server
    run_move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            moveit_config.to_dict(),
            {
                "use_sim_time": use_sim_time,
            },
        ],
    )

    rviz_base = LaunchConfiguration("rviz_config")
    rviz_config = PathJoinSubstitution(
        [FindPackageShare("rbpodo_moveit_config"), "config", rviz_base]
    )

    # RViz
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="log",
        arguments=["-d", rviz_config],
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            moveit_config.planning_pipelines,
            moveit_config.joint_limits,
            {
                "use_sim_time": use_sim_time,
            },
        ],
    )

    # Static TF
    static_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="static_transform_publisher",
        output="log",
        arguments=["0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "world", "link0"],
    )

    # Publish TF
    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="both",
        parameters=[
            moveit_config.robot_description,
            {
                "use_sim_time": use_sim_time,
            },
        ],
    )

    # ros2_control using FakeSystem as hardware
    ros2_controllers_path = os.path.join(
        get_package_share_directory("rbpodo_bringup"),
        "config",
        "controllers.yaml",
    )
    ros2_control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[
            moveit_config.robot_description,
            ros2_controllers_path,
            {
                "use_sim_time": use_sim_time,
            },
        ],
        output="both",
        condition=UnlessCondition(use_isaac_bridge),
    )

    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "joint_state_broadcaster",
            "--controller-manager-timeout",
            "300",
            "--controller-manager",
            "/controller_manager",
        ],
        condition=UnlessCondition(use_isaac_bridge),
    )

    arm_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_trajectory_controller", "-c", "/controller_manager"],
        condition=UnlessCondition(use_isaac_bridge),
    )

    isaac_trajectory_bridge = Node(
        package="rbpodo_moveit_config",
        executable="isaac_joint_trajectory_bridge.py",
        name="isaac_joint_trajectory_bridge",
        output="screen",
        parameters=[
            {
                "isaac_joint_states_topic": "/isaac_joint_states",
                "joint_states_topic": "/joint_states",
                "isaac_joint_commands_topic": "/isaac_joint_commands",
                "goal_tolerance": 0.05,
                "goal_wait_timeout": 30.0,
                "goal_stable_duration": 0.2,
                "shoulder_min_position": -1.0471975511965976,
                "shoulder_max_position": 1.5707963267948966,
                "use_sim_time": use_sim_time,
            }
        ],
        condition=IfCondition(use_isaac_bridge),
    )

    nodes_to_start = [
        rviz_node,
        static_tf,
        robot_state_publisher,
        run_move_group_node,
        ros2_control_node,
        joint_state_broadcaster_spawner,
        arm_controller_spawner,
        isaac_trajectory_bridge,
    ]

    return nodes_to_start
