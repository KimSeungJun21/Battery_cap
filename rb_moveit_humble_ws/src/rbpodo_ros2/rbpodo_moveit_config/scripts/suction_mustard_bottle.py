#!/usr/bin/env python3
"""Pick the YCB mustard bottle from above in Isaac Sim using MoveIt."""

from __future__ import annotations

import argparse
import sys
import time

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import Pose, PoseStamped, Quaternion
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    Constraints,
    MoveItErrorCodes,
    OrientationConstraint,
    PositionConstraint,
)
from moveit_msgs.srv import GetCartesianPath
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.utilities import remove_ros_args
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import Bool
from std_srvs.srv import SetBool


# Fixed robot/ROS configuration. These values are not operator controls.
GROUP_NAME = "mainpulation"
TCP_LINK = "tcp"
PLANNING_FRAME = "world"
MOVE_ACTION = "/move_action"
CARTESIAN_SERVICE = "/compute_cartesian_path"
OBJECT_POSE_TOPIC = "/target_object/pose"
JOINT_COMMAND_TOPIC = "/isaac_joint_commands"
EPICK_SERVICE = "/grip_cmd"
ISAAC_SUCTION_TOPIC = "/epick/suction_cmd"

# Lift target is 10 cm above Zivid_camera_01's optical-center height in
# MoveIt's planning frame.
# Battery_rb.py publishes object poses relative to the robot root while naming
# that frame "world": 1.2248 - 0.7000 + 0.1000.
CAMERA_LIFT_TARGET_Z = 0.7248

ARM_JOINT_NAMES = (
    "base",
    "shoulder",
    "elbow",
    "wrist1",
    "wrist2",
    "wrist3",
)

# The TCP is offset along local -Y. A +90-degree world-X rotation therefore
# points the suction axis straight down along world -Z.
TCP_QUATERNION = (0.7071068, 0.0, 0.0, 0.7071068)

PLANNING_TIME = 15.0
PLANNING_ATTEMPTS = 30
POSITION_TOLERANCE = 0.01
ORIENTATION_TOLERANCE = 0.03
YAW_TOLERANCE = 0.15
SERVER_TIMEOUT = 20.0
OBJECT_POSE_TIMEOUT = 3.0

CARTESIAN_MAX_STEP = 0.005
CARTESIAN_MIN_FRACTION = 0.99
CARTESIAN_POINT_DURATION = 0.05

ZERO_START_COMMAND_DURATION = 5.0
FINAL_COMMAND_HOLD = 0.3

SUCTION_REPEAT_COUNT = 5
SUCTION_REPEAT_PERIOD = 0.05
SUCTION_SETTLE_TIME = 0.5
GRIPPER_RELEASE_SETTLE_TIME = 0.5


def parse_arguments(
    argv,
    object_description="YCB mustard bottle",
    default_avoid_collisions=False,
):
    parser = argparse.ArgumentParser(
        description=f"Top-down suction pick for the {object_description}."
    )
    parser.add_argument(
        "--contact-position",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        help=(
            "Direct TCP contact position in world coordinates. If omitted, "
            f"the position is read from {OBJECT_POSE_TOPIC}."
        ),
    )
    parser.add_argument(
        "--contact-offset",
        nargs=3,
        type=float,
        metavar=("DX", "DY", "DZ"),
        default=(0.0, 0.0, 0.0),
        help="Offset added to the object pose, in meters.",
    )
    parser.add_argument(
        "--pregrasp-height",
        type=float,
        default=0.30,
        help="Distance above the contact point before descending (meters).",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=0.2,
        help="MoveIt velocity and acceleration scaling from 0.01 to 1.0.",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Plan the pregrasp motion without commanding Isaac Sim.",
    )
    parser.add_argument(
        "--no-zero-start",
        action="store_true",
        help="Do not command all six arm joints to zero before planning.",
    )
    parser.add_argument(
        "--release-after-lift",
        action="store_true",
        help="Turn suction off after reaching the lift pose.",
    )
    collision_group = parser.add_mutually_exclusive_group()
    collision_group.add_argument(
        "--avoid-collisions",
        dest="avoid_collisions",
        action="store_true",
        help="Enable MoveIt collision checking for Cartesian paths.",
    )
    collision_group.add_argument(
        "--allow-collisions",
        dest="avoid_collisions",
        action="store_false",
        help="Disable MoveIt collision checking for Cartesian paths.",
    )
    parser.set_defaults(
        avoid_collisions=bool(default_avoid_collisions)
    )
    return parser.parse_args(argv)


def make_quaternion():
    quaternion = Quaternion()
    (
        quaternion.x,
        quaternion.y,
        quaternion.z,
        quaternion.w,
    ) = TCP_QUATERNION
    return quaternion


def make_pose(x, y, z):
    pose = Pose()
    pose.position.x = float(x)
    pose.position.y = float(y)
    pose.position.z = float(z)
    pose.orientation = make_quaternion()
    return pose


def offset_vertical(pose, distance):
    return make_pose(
        pose.position.x,
        pose.position.y,
        pose.position.z + float(distance),
    )


def make_camera_height_lift_pose(contact_pose):
    """Raise the suction TCP to 10 cm above the camera height."""
    target_z = max(
        float(contact_pose.position.z),
        CAMERA_LIFT_TARGET_Z,
    )
    return make_pose(
        contact_pose.position.x,
        contact_pose.position.y,
        target_z,
    )


def make_goal_constraints(pose, constraint_name="mustard_tcp_pose"):
    region = SolidPrimitive()
    region.type = SolidPrimitive.SPHERE
    region.dimensions = [POSITION_TOLERANCE]

    position = PositionConstraint()
    position.header.frame_id = PLANNING_FRAME
    position.link_name = TCP_LINK
    position.constraint_region.primitives.append(region)
    position.constraint_region.primitive_poses.append(pose)
    position.weight = 1.0

    orientation = OrientationConstraint()
    orientation.header.frame_id = PLANNING_FRAME
    orientation.link_name = TCP_LINK
    orientation.orientation = pose.orientation
    orientation.absolute_x_axis_tolerance = ORIENTATION_TOLERANCE
    orientation.absolute_y_axis_tolerance = ORIENTATION_TOLERANCE
    orientation.absolute_z_axis_tolerance = YAW_TOLERANCE
    orientation.weight = 1.0

    constraints = Constraints()
    constraints.name = constraint_name
    constraints.position_constraints.append(position)
    constraints.orientation_constraints.append(orientation)
    return constraints


def duration_seconds(duration):
    return float(duration.sec) + float(duration.nanosec) * 1e-9


def set_duration(duration, seconds):
    seconds = max(0.0, float(seconds))
    duration.sec = int(seconds)
    duration.nanosec = int(round((seconds - duration.sec) * 1e9))


def ensure_trajectory_timing(trajectory):
    previous = -1.0
    valid = bool(trajectory.points)
    for point in trajectory.points:
        current = duration_seconds(point.time_from_start)
        if current <= previous:
            valid = False
            break
        previous = current

    if valid:
        return

    for index, point in enumerate(trajectory.points, start=1):
        set_duration(
            point.time_from_start,
            CARTESIAN_POINT_DURATION * index,
        )


def spin_sleep(node, duration):
    deadline = time.monotonic() + max(0.0, float(duration))
    while rclpy.ok() and time.monotonic() < deadline:
        rclpy.spin_once(
            node,
            timeout_sec=min(0.05, deadline - time.monotonic()),
        )


class JointCommander:
    def __init__(self, node):
        self.node = node
        self.publisher = node.create_publisher(
            JointState,
            JOINT_COMMAND_TOPIC,
            10,
        )

    def wait_until_ready(self):
        deadline = time.monotonic() + SERVER_TIMEOUT
        while rclpy.ok() and time.monotonic() < deadline:
            if self.publisher.get_subscription_count() > 0:
                return
            rclpy.spin_once(self.node, timeout_sec=0.1)
        raise RuntimeError(
            f"No subscriber is connected to {JOINT_COMMAND_TOPIC}."
        )

    def publish_positions(self, joint_names, positions):
        if len(joint_names) != len(positions):
            raise RuntimeError("Joint names and positions do not match.")

        message = JointState()
        message.header.stamp = self.node.get_clock().now().to_msg()
        message.name = list(joint_names)
        message.position = list(positions)
        self.publisher.publish(message)

    def publish_trajectory(self, trajectory):
        names = list(trajectory.joint_names)
        points = list(trajectory.points)
        if not names or not points:
            raise RuntimeError("MoveIt returned an empty joint trajectory.")

        self.node.get_logger().info(
            f"Commanding {len(points)} trajectory points: "
            + ", ".join(names)
        )
        previous = 0.0
        for point in points:
            current = duration_seconds(point.time_from_start)
            spin_sleep(self.node, max(0.0, current - previous))
            self.publish_positions(names, point.positions)
            previous = current

        deadline = time.monotonic() + FINAL_COMMAND_HOLD
        final_positions = points[-1].positions
        while rclpy.ok() and time.monotonic() < deadline:
            self.publish_positions(names, final_positions)
            spin_sleep(self.node, 0.05)


class SuctionController:
    def __init__(self, node):
        self.node = node
        self.interface = None
        self.service = node.create_client(SetBool, EPICK_SERVICE)
        self.publisher = node.create_publisher(
            Bool,
            ISAAC_SUCTION_TOPIC,
            10,
        )

    def wait_until_ready(self):
        if self.service.wait_for_service(timeout_sec=2.0):
            self.interface = "service"
            self.node.get_logger().info(
                f"Using EPick SetBool service: {EPICK_SERVICE}"
            )
            return

        deadline = time.monotonic() + 2.0
        while rclpy.ok() and time.monotonic() < deadline:
            if self.publisher.get_subscription_count() > 0:
                self.interface = "topic"
                self.node.get_logger().info(
                    "Using Isaac suction Bool topic: "
                    f"{ISAAC_SUCTION_TOPIC}"
                )
                return
            rclpy.spin_once(self.node, timeout_sec=0.1)
        raise RuntimeError(
            "No suction interface is available: "
            f"service={EPICK_SERVICE}, topic={ISAAC_SUCTION_TOPIC}."
        )

    def set_enabled(self, enabled):
        if self.interface == "service":
            request = SetBool.Request()
            request.data = bool(enabled)
            future = self.service.call_async(request)
            rclpy.spin_until_future_complete(self.node, future)
            response = future.result()
            if response is None or not response.success:
                message = response.message if response is not None else ""
                raise RuntimeError(
                    f"EPick service command failed: {message}"
                )
            return

        if self.interface != "topic":
            raise RuntimeError("Suction interface was not initialized.")

        message = Bool()
        message.data = bool(enabled)
        for _ in range(SUCTION_REPEAT_COUNT):
            self.publisher.publish(message)
            rclpy.spin_once(self.node, timeout_sec=0.0)
            time.sleep(SUCTION_REPEAT_PERIOD)


class MustardBottlePicker(Node):
    NODE_NAME = "suction_mustard_bottle"
    CONSTRAINT_NAME = "mustard_tcp_pose"

    def __init__(self, options):
        super().__init__(self.NODE_NAME)
        self.options = options
        self.speed = min(1.0, max(0.01, float(options.speed)))
        self.latest_object_pose = None

        self.move_group = ActionClient(self, MoveGroup, MOVE_ACTION)
        self.cartesian = self.create_client(
            GetCartesianPath,
            CARTESIAN_SERVICE,
        )
        self.joints = JointCommander(self)
        self.suction = SuctionController(self)

        if options.contact_position is None:
            self.create_subscription(
                PoseStamped,
                OBJECT_POSE_TOPIC,
                self._on_object_pose,
                10,
            )

    def _on_object_pose(self, message):
        self.latest_object_pose = message

    def wait_until_ready(self):
        self.get_logger().info("Waiting for MoveIt and Isaac Sim interfaces.")
        if not self.move_group.wait_for_server(timeout_sec=SERVER_TIMEOUT):
            raise RuntimeError(f"MoveIt action is unavailable: {MOVE_ACTION}")
        if not self.cartesian.wait_for_service(timeout_sec=SERVER_TIMEOUT):
            raise RuntimeError(
                f"Cartesian service is unavailable: {CARTESIAN_SERVICE}"
            )
        if not self.options.plan_only:
            self.joints.wait_until_ready()
            self.suction.wait_until_ready()

    def run(self):
        if not self.options.plan_only:
            self.open_gripper_for_start()
        if not self.options.no_zero_start:
            self.move_to_zero_start_pose()
        if not self.options.plan_only:
            self.wait_for_operator_approval()

        contact = self.make_contact_pose()
        pregrasp = offset_vertical(
            contact,
            max(0.0, self.options.pregrasp_height),
        )
        lift = make_camera_height_lift_pose(contact)

        self.log_pose("pregrasp", pregrasp)
        trajectory = self.plan_pose(pregrasp)
        if self.options.plan_only:
            self.get_logger().info(
                "Pregrasp planning succeeded; --plan-only stops here."
            )
            return

        self.execute_joint_trajectory("pregrasp", trajectory)
        self.move_cartesian("contact", contact)

        self.get_logger().info("Enabling suction.")
        self.suction.set_enabled(True)
        spin_sleep(self, SUCTION_SETTLE_TIME)

        self.move_cartesian("lift", lift)
        if self.options.release_after_lift:
            self.get_logger().info("Disabling suction after lift.")
            self.suction.set_enabled(False)

    def open_gripper_for_start(self):
        """Open EPick before zeroing so any attached object is released."""
        self.get_logger().info(
            "Opening EPick before zero start to release attached objects."
        )
        self.suction.set_enabled(False)
        spin_sleep(self, GRIPPER_RELEASE_SETTLE_TIME)

    def move_to_zero_start_pose(self):
        """Command all six arm joints to zero and verify convergence."""
        if self.options.plan_only:
            self.get_logger().info(
                "Skipping zero start because --plan-only is enabled."
            )
            return

        names = list(ARM_JOINT_NAMES)
        zeros = [0.0] * len(names)
        self.get_logger().info(
            "Commanding zero start pose: " + ", ".join(names)
        )
        deadline = time.monotonic() + ZERO_START_COMMAND_DURATION
        while rclpy.ok() and time.monotonic() < deadline:
            self.joints.publish_positions(names, zeros)
            spin_sleep(self, 0.05)
        self.get_logger().info(
            "Zero start command duration completed; joint tolerance "
            "verification is disabled."
        )

    def wait_for_operator_approval(self):
        """Block all pick motion until the operator explicitly approves."""
        preparation = (
            "Robot preparation"
            if self.options.no_zero_start
            else "Zero start"
        )
        self.get_logger().warning(
            f"{preparation} is complete. Operator approval is required "
            "before the pick sequence can continue."
        )

        while rclpy.ok():
            try:
                answer = input(
                    "Continue to pregrasp and suction? [y/yes, n/no]: "
                ).strip().lower()
            except (EOFError, KeyboardInterrupt) as exc:
                raise RuntimeError(
                    "Operator approval was not received; pick cancelled."
                ) from exc

            if answer in ("y", "yes"):
                self.get_logger().info(
                    "Operator approved the pick sequence."
                )
                return
            if answer in ("n", "no", "q", "quit"):
                raise RuntimeError(
                    "Operator rejected the pick sequence."
                )

            print("Enter y/yes to continue or n/no to cancel.", flush=True)

        raise RuntimeError("ROS shut down before operator approval.")

    def wait_for_object_pose(self):
        deadline = time.monotonic() + OBJECT_POSE_TIMEOUT
        while (
            rclpy.ok()
            and self.latest_object_pose is None
            and time.monotonic() < deadline
        ):
            rclpy.spin_once(self, timeout_sec=0.1)

        if self.latest_object_pose is None:
            raise RuntimeError(
                f"No object pose received from {OBJECT_POSE_TOPIC}. "
                "Pass --contact-position or start Battery_rb.py with target "
                "pose publishing enabled."
            )
        return self.latest_object_pose

    def make_contact_pose(self):
        if self.options.contact_position is not None:
            x, y, z = self.options.contact_position
        else:
            message = self.wait_for_object_pose()
            if (
                message.header.frame_id
                and message.header.frame_id != PLANNING_FRAME
            ):
                self.get_logger().warning(
                    "Object pose frame differs from planning frame: "
                    f"{message.header.frame_id} != {PLANNING_FRAME}"
                )
            x = message.pose.position.x
            y = message.pose.position.y
            z = message.pose.position.z
            dx, dy, dz = self.options.contact_offset
            x += dx
            y += dy
            z += dz

        contact = make_pose(x, y, z)
        self.log_pose("contact", contact)
        return contact

    def plan_pose(self, pose):
        goal = MoveGroup.Goal()
        goal.request.group_name = GROUP_NAME
        goal.request.num_planning_attempts = PLANNING_ATTEMPTS
        goal.request.allowed_planning_time = PLANNING_TIME
        goal.request.max_velocity_scaling_factor = self.speed
        goal.request.max_acceleration_scaling_factor = self.speed
        goal.request.start_state.is_diff = True
        goal.request.goal_constraints.append(
            make_goal_constraints(
                pose,
                constraint_name=self.CONSTRAINT_NAME,
            )
        )
        goal.planning_options.plan_only = True
        goal.planning_options.replan = True
        goal.planning_options.planning_scene_diff.is_diff = True
        goal.planning_options.planning_scene_diff.robot_state.is_diff = True

        future = self.move_group.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)
        handle = future.result()
        if handle is None or not handle.accepted:
            raise RuntimeError("MoveIt rejected the pregrasp goal.")

        result_future = handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        wrapped = result_future.result()
        if wrapped is None:
            raise RuntimeError("MoveIt did not return a planning result.")

        result = wrapped.result
        if (
            wrapped.status != GoalStatus.STATUS_SUCCEEDED
            or result.error_code.val != MoveItErrorCodes.SUCCESS
        ):
            raise RuntimeError(
                "MoveIt pregrasp planning failed: "
                f"status={wrapped.status}, error={result.error_code.val}."
            )

        trajectory = result.planned_trajectory.joint_trajectory
        if not trajectory.joint_names or not trajectory.points:
            raise RuntimeError("MoveIt returned an empty pregrasp trajectory.")
        return trajectory

    def move_cartesian(self, label, pose):
        self.log_pose(label, pose)
        request = GetCartesianPath.Request()
        request.header.frame_id = PLANNING_FRAME
        request.header.stamp = self.get_clock().now().to_msg()
        request.start_state.is_diff = True
        request.group_name = GROUP_NAME
        request.link_name = TCP_LINK
        request.waypoints = [pose]
        request.max_step = CARTESIAN_MAX_STEP
        request.jump_threshold = 0.0
        request.avoid_collisions = bool(
            self.options.avoid_collisions
        )
        if hasattr(request, "max_velocity_scaling_factor"):
            request.max_velocity_scaling_factor = self.speed
        if hasattr(request, "max_acceleration_scaling_factor"):
            request.max_acceleration_scaling_factor = self.speed

        future = self.cartesian.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        response = future.result()
        if response is None:
            raise RuntimeError(f"Cartesian service failed for {label}.")
        if response.error_code.val != MoveItErrorCodes.SUCCESS:
            raise RuntimeError(
                f"Cartesian planning failed for {label}: "
                f"error={response.error_code.val}."
            )
        if response.fraction < CARTESIAN_MIN_FRACTION:
            raise RuntimeError(
                f"Cartesian path for {label} is incomplete: "
                f"{response.fraction:.3f} < {CARTESIAN_MIN_FRACTION:.3f}."
            )

        trajectory = response.solution.joint_trajectory
        if not trajectory.joint_names or not trajectory.points:
            raise RuntimeError(f"Empty Cartesian trajectory for {label}.")
        ensure_trajectory_timing(trajectory)
        self.execute_joint_trajectory(label, trajectory)

    def execute_joint_trajectory(self, label, trajectory):
        self.joints.publish_trajectory(trajectory)
        self.get_logger().info(
            f"Finished commanding {label}; joint tolerance verification "
            "is disabled."
        )

    def log_pose(self, label, pose):
        self.get_logger().info(
            f"{label} pose: "
            f"position=({pose.position.x:.4f}, "
            f"{pose.position.y:.4f}, {pose.position.z:.4f}), "
            f"orientation=({pose.orientation.x:.4f}, "
            f"{pose.orientation.y:.4f}, {pose.orientation.z:.4f}, "
            f"{pose.orientation.w:.4f})"
        )


def main(argv=None):
    argv = sys.argv if argv is None else argv
    options = parse_arguments(remove_ros_args(argv)[1:])
    rclpy.init(args=argv)
    node = MustardBottlePicker(options)

    try:
        node.wait_until_ready()
        node.run()
        node.get_logger().info("Mustard bottle suction sequence finished.")
    except Exception as exc:
        node.get_logger().error(str(exc))
        raise
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
