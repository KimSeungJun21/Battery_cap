#!/usr/bin/env python3
"""Relay NBV pose requests from Isaac Sim to a MoveGroup action.

Battery_rb.py runs inside Isaac Sim's bundled minimal ROS2 Python
environment, which only ships a small set of message packages
(std_msgs/geometry_msgs/action_msgs) and does not have moveit_msgs
available. This script runs instead in a normal ROS2/MoveIt-capable shell
(same environment as suction_mustard_bottle.py) and bridges the gap with
plain geometry_msgs/std_msgs topics that Isaac's bundled rclpy can use
directly: it receives a target PoseStamped on --request-topic, drives it
through MoveGroup, and reports success/failure as a Bool on --result-topic.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from builtin_interfaces.msg import Duration
from moveit_msgs.action import ExecuteTrajectory, MoveGroup
from moveit_msgs.msg import (
    Constraints,
    JointConstraint,
    MoveItErrorCodes,
    OrientationConstraint,
    PositionConstraint,
    PositionIKRequest,
)
from moveit_msgs.srv import GetPositionFK, GetPositionIK
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.utilities import remove_ros_args
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import Bool, String


MOVEIT_ERROR_NAMES = [
    "SUCCESS",
    "FAILURE",
    "PLANNING_FAILED",
    "INVALID_MOTION_PLAN",
    "MOTION_PLAN_INVALIDATED_BY_ENVIRONMENT_CHANGE",
    "CONTROL_FAILED",
    "UNABLE_TO_AQUIRE_SENSOR_DATA",
    "TIMED_OUT",
    "PREEMPTED",
    "START_STATE_IN_COLLISION",
    "START_STATE_VIOLATES_PATH_CONSTRAINTS",
    "GOAL_IN_COLLISION",
    "GOAL_VIOLATES_PATH_CONSTRAINTS",
    "GOAL_CONSTRAINTS_VIOLATED",
    "INVALID_GROUP_NAME",
    "INVALID_GOAL_CONSTRAINTS",
    "INVALID_ROBOT_STATE",
    "INVALID_LINK_NAME",
    "INVALID_OBJECT_NAME",
    "FRAME_TRANSFORM_FAILURE",
    "COLLISION_CHECKING_UNAVAILABLE",
    "ROBOT_STATE_STALE",
    "SENSOR_INFO_STALE",
    "COMMUNICATION_FAILURE",
    "NO_IK_SOLUTION",
]


def moveit_error_name(error_code_value):
    for name in MOVEIT_ERROR_NAMES:
        if getattr(MoveItErrorCodes, name, None) == error_code_value:
            return name
    return f"UNKNOWN_{error_code_value}"


def duration_seconds(duration):
    return float(duration.sec) + float(duration.nanosec) * 1e-9


def matrix_multiply(left, right):
    return [
        [
            sum(left[row][inner] * right[inner][column] for inner in range(4))
            for column in range(4)
        ]
        for row in range(4)
    ]


def pose_matrix(pose):
    x = float(pose.orientation.x)
    y = float(pose.orientation.y)
    z = float(pose.orientation.z)
    w = float(pose.orientation.w)
    norm = math.sqrt(x * x + y * y + z * z + w * w)

    if norm <= 1e-12:
        raise ValueError("FK returned a zero-length TCP quaternion")

    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return [
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - w * z),
            2.0 * (x * z + w * y),
            float(pose.position.x),
        ],
        [
            2.0 * (x * y + w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - w * x),
            float(pose.position.y),
        ],
        [
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
            1.0 - 2.0 * (x * x + y * y),
            float(pose.position.z),
        ],
        [0.0, 0.0, 0.0, 1.0],
    ]


def transform_point(matrix, point):
    return [
        sum(matrix[row][column] * point[column] for column in range(3))
        + matrix[row][3]
        for row in range(3)
    ]


def dot(left, right):
    return sum(a * b for a, b in zip(left, right))


def validate_frustum_config(payload):
    if int(payload.get("schema_version", 0)) != 1:
        raise ValueError("unsupported frustum config schema")

    camera = payload["camera_matrix"]
    intrinsic = payload["intrinsic_matrix"]
    image_size = payload["image_size"]
    corners = payload["corners_local"]
    grasp_offset = payload["grasp_offset"]

    if len(camera) != 4 or any(len(row) != 4 for row in camera):
        raise ValueError("camera_matrix must be 4x4")
    if len(intrinsic) != 3 or any(len(row) != 3 for row in intrinsic):
        raise ValueError("intrinsic_matrix must be 3x3")
    if len(image_size) != 2 or min(image_size) <= 0:
        raise ValueError("image_size must contain positive width and height")
    if not corners or any(len(point) != 3 for point in corners):
        raise ValueError("corners_local must contain 3D points")
    if len(grasp_offset) != 4 or any(len(row) != 4 for row in grasp_offset):
        raise ValueError("grasp_offset must be 4x4")

    margin = float(payload.get("margin_px", 0.0))
    fraction = float(payload.get("min_in_frame_fraction", 1.0))
    if margin < 0.0:
        raise ValueError("margin_px must not be negative")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("min_in_frame_fraction must be in (0, 1]")

    # Convert every numeric field now so malformed JSON is rejected before a
    # motion request can reach MoveIt.
    payload["camera_matrix"] = [
        [float(value) for value in row] for row in camera
    ]
    payload["intrinsic_matrix"] = [
        [float(value) for value in row] for row in intrinsic
    ]
    payload["image_size"] = [int(image_size[0]), int(image_size[1])]
    payload["corners_local"] = [
        [float(value) for value in point] for point in corners
    ]
    payload["grasp_offset"] = [
        [float(value) for value in row] for row in grasp_offset
    ]
    payload["margin_px"] = margin
    payload["min_in_frame_fraction"] = fraction
    return payload


def make_position_constraint(frame_id, link_name, pose, tolerance):
    primitive = SolidPrimitive()
    primitive.type = SolidPrimitive.SPHERE
    primitive.dimensions = [float(tolerance)]

    constraint = PositionConstraint()
    constraint.header.frame_id = frame_id
    constraint.link_name = link_name
    constraint.constraint_region.primitives.append(primitive)
    constraint.constraint_region.primitive_poses.append(pose)
    constraint.weight = 1.0
    return constraint


def make_orientation_constraint(frame_id, link_name, orientation, tolerance):
    constraint = OrientationConstraint()
    constraint.header.frame_id = frame_id
    constraint.link_name = link_name
    constraint.orientation = orientation
    constraint.absolute_x_axis_tolerance = float(tolerance)
    constraint.absolute_y_axis_tolerance = float(tolerance)
    constraint.absolute_z_axis_tolerance = float(tolerance)
    constraint.weight = 1.0
    return constraint


def make_goal_constraints(
    frame_id,
    link_name,
    pose,
    position_tolerance,
    orientation_tolerance,
):
    constraints = Constraints()
    constraints.name = "nbv_tcp_pose_goal"
    constraints.position_constraints.append(
        make_position_constraint(
            frame_id=frame_id,
            link_name=link_name,
            pose=pose,
            tolerance=position_tolerance,
        )
    )
    constraints.orientation_constraints.append(
        make_orientation_constraint(
            frame_id=frame_id,
            link_name=link_name,
            orientation=pose.orientation,
            tolerance=orientation_tolerance,
        )
    )
    return constraints


def make_joint_goal_constraints(
    joint_names,
    positions,
    tolerance=0.001,
):
    if len(joint_names) != len(positions) or not joint_names:
        raise ValueError("joint_names/positions must be non-empty and equal")

    constraints = Constraints()
    constraints.name = "nbv_cached_joint_goal"
    for name, position in zip(joint_names, positions):
        constraint = JointConstraint()
        constraint.joint_name = str(name)
        constraint.position = float(position)
        constraint.tolerance_above = float(tolerance)
        constraint.tolerance_below = float(tolerance)
        constraint.weight = 1.0
        constraints.joint_constraints.append(constraint)
    return constraints


def parse_arguments(argv):
    parser = argparse.ArgumentParser(
        description=(
            "Relay PoseStamped goals from Isaac Sim to MoveGroup for the "
            "arm-driven NBV sweep, reporting success/failure back as Bool."
        )
    )
    parser.add_argument("--move-action", default="/move_action")
    parser.add_argument(
        "--execute-action",
        default="/execute_trajectory",
        help="MoveIt action used after a planned trajectory passes the guard.",
    )
    parser.add_argument("--request-topic", default="/nbv/move_request")
    parser.add_argument("--result-topic", default="/nbv/move_result")
    parser.add_argument(
        "--preflight-request-topic",
        default="/nbv/preflight_request",
    )
    parser.add_argument(
        "--preflight-result-topic",
        default="/nbv/preflight_result",
    )
    parser.add_argument(
        "--joint-request-topic",
        default="/nbv/joint_request",
    )
    parser.add_argument(
        "--isaac-joint-command-topic",
        default="/isaac_joint_commands",
        help=(
            "Isaac JointState command topic used only by cached-joint IMSI "
            "requests. This bypasses the FollowJointTrajectory bridge's "
            "joint-space convergence wait; Isaac validates the stopped TCP "
            "and object poses instead."
        ),
    )
    parser.add_argument(
        "--direct-publish-rate",
        type=float,
        default=50.0,
        help="Publish rate for cached-joint IMSI trajectories.",
    )
    parser.add_argument(
        "--direct-final-hold",
        type=float,
        default=0.05,
        help="Seconds to keep publishing the final IMSI joint command.",
    )
    parser.add_argument(
        "--frustum-config-topic",
        default="/nbv/frustum_guard_config",
        help=(
            "JSON String topic carrying the camera, object bbox and rigid "
            "TCP-to-object transform used by the trajectory guard."
        ),
    )
    parser.add_argument("--group-name", default="mainpulation")
    parser.add_argument("--link-name", default="tcp")
    parser.add_argument("--frame-id", default="world")
    parser.add_argument("--position-tolerance", type=float, default=0.01)
    parser.add_argument("--orientation-tolerance", type=float, default=0.05)
    # IK 사전 검사가 도달 불가 후보를 먼저 거르므로, 여기까지 오는
    # 목표는 대부분 해가 있다. 10초 x 10회는 성공 경로에서도 시간을
    # 잡아먹는다. 짧게 잡고, 실패가 늘면 다시 올린다.
    parser.add_argument("--planning-time", type=float, default=2.0)
    parser.add_argument("--planning-attempts", type=int, default=2)
    # MoveIt의 scaling factor는 URDF에 선언된 관절 속도/가속 한계에 대한
    # 비율이다. 1.0이면 그 한계까지 쓴다 (순간이동이 아니라, IK로 푼
    # 궤적을 로봇이 낼 수 있는 최대 속도로 실행).
    parser.add_argument("--velocity-scaling", type=float, default=1.0)
    parser.add_argument("--acceleration-scaling", type=float, default=1.0)
    parser.add_argument("--server-timeout", type=float, default=20.0)
    parser.add_argument(
        "--goal-timeout",
        type=float,
        default=40.0,
        help=(
            "Give up on a MoveGroup goal after this many seconds and "
            "cancel it. The Isaac trajectory bridge may wait up to 30s "
            "for physical joint convergence, so this must stay above that "
            "limit. Set 0 to wait forever."
        ),
    )
    parser.add_argument(
        "--ik-service",
        default="/compute_ik",
        help="MoveIt IK service used for the reachability pre-check.",
    )
    parser.add_argument(
        "--fk-service",
        default="/compute_fk",
        help="MoveIt FK service used to inspect planned trajectory samples.",
    )
    parser.add_argument(
        "--no-ik-precheck",
        dest="ik_precheck",
        action="store_false",
        help=(
            "Disable the IK reachability pre-check. Without it every "
            "unreachable NBV candidate goes through the full MoveGroup "
            "planning pipeline: measured 35-52s each, which was 79%% of a "
            "620s NBV sweep (493s across 14 failures). The pre-check "
            "rejects those in milliseconds."
        ),
    )
    parser.set_defaults(ik_precheck=True)
    parser.add_argument(
        "--ik-timeout",
        type=float,
        default=0.2,
        help="Per-attempt IK solve timeout in seconds.",
    )
    parser.add_argument(
        "--ik-attempts",
        type=int,
        default=2,
        help=(
            "IK solve attempts before declaring a pose unreachable. "
            "MoveIt seeds IK randomly, so a single failure is not "
            "conclusive."
        ),
    )
    parser.add_argument(
        "--ik-avoid-collisions",
        dest="ik_avoid_collisions",
        action="store_true",
        help="Make the IK pre-check reject poses that are in collision.",
    )
    parser.add_argument(
        "--ik-allow-collisions",
        dest="ik_avoid_collisions",
        action="store_false",
    )
    parser.set_defaults(ik_avoid_collisions=True)
    parser.add_argument(
        "--ik-service-timeout",
        type=float,
        default=5.0,
        help="How long to wait for the IK service at startup.",
    )
    parser.add_argument(
        "--fk-timeout",
        type=float,
        default=1.0,
        help="Timeout for the final stopped-pose FK request.",
    )
    parser.add_argument(
        "--frustum-sample-period",
        type=float,
        default=0.05,
        help=(
            "Deprecated compatibility option. Frustum visibility is checked "
            "only at the final stopped pose."
        ),
    )
    parser.add_argument(
        "--frustum-max-joint-step",
        type=float,
        default=0.05,
        help=(
            "Deprecated compatibility option. Frustum visibility is checked "
            "only at the final stopped pose."
        ),
    )
    parser.add_argument(
        "--no-frustum-guard",
        dest="frustum_guard",
        action="store_false",
        help=(
            "Disable final-pose camera-frustum validation and use the legacy "
            "plan-and-execute behavior."
        ),
    )
    parser.set_defaults(frustum_guard=True)
    parser.add_argument(
        "--reason-topic",
        default="/nbv/move_reason",
        help=(
            "std_msgs/String topic carrying why each request succeeded or "
            "failed (ik_unreachable / plan_failed / succeeded / ...). "
            "Isaac's bundled rclpy has std_msgs, so it can subscribe."
        ),
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help=(
            "Send each goal to MoveGroup with planning_options.plan_only=True "
            "instead of plan-and-execute. The robot never actually moves - "
            "MoveIt just plans and (if RViz's MotionPlanning display is open) "
            "shows the resulting trajectory/goal pose, so you can visually "
            "inspect reachability for each NBV candidate before trusting the "
            "real run. Result Bool reflects planning success, not execution."
        ),
    )
    parser.add_argument(
        "--publish-only",
        action="store_true",
        help=(
            "Never talk to MoveGroup at all - just republish each incoming "
            "target pose on --preview-topic (for an RViz PoseStamped/Axes "
            "display) and immediately report success. Use this to eyeball "
            "where the selected NBV candidate's TCP target actually is "
            "(e.g. behind/below the robot, flipped away from the camera) "
            "without waiting through a real plan or plan-only attempt."
        ),
    )
    parser.add_argument(
        "--preview-topic",
        default="/nbv/target_pose_preview",
        help="Topic --publish-only republishes each target pose on.",
    )
    parser.add_argument(
        "--settle-time",
        type=float,
        default=0.3,
        help=(
            "Delay after each MoveGroup result before accepting the next "
            "goal, so move_group's own trajectory_execution_manager has "
            "time to fully release the previous one. Measured NBV cycles "
            "spent 2.0s here per view; 0.3s is usually enough. Raise it "
            "again if you see 'Cannot push a new trajectory while another "
            "is being executed'."
        ),
    )
    return parser.parse_args(argv)


class NbvArmMoveRelay(Node):
    def __init__(self, options):
        super().__init__("nbv_arm_move_relay")
        self.options = options
        self.pending_request = None
        self.pending_preflight_request = None
        self.pending_joint_request = None

        self.move_group_client = ActionClient(
            self,
            MoveGroup,
            options.move_action,
        )
        self.execute_trajectory_client = ActionClient(
            self,
            ExecuteTrajectory,
            options.execute_action,
        )
        self.result_publisher = self.create_publisher(
            Bool,
            options.result_topic,
            10,
        )
        self.reason_publisher = self.create_publisher(
            String,
            options.reason_topic,
            10,
        )
        self.preflight_result_publisher = self.create_publisher(
            String,
            options.preflight_result_topic,
            10,
        )
        self.isaac_joint_command_publisher = self.create_publisher(
            JointState,
            options.isaac_joint_command_topic,
            10,
        )
        self.ik_client = self.create_client(
            GetPositionIK,
            options.ik_service,
        )
        self.fk_client = self.create_client(
            GetPositionFK,
            options.fk_service,
        )
        self.ik_available = False
        self.ik_warned = False
        self.fk_available = False
        self.frustum_config = None
        self.last_move_failure_reason = None
        self.preview_publisher = self.create_publisher(
            PoseStamped,
            options.preview_topic,
            10,
        )
        self.request_subscription = self.create_subscription(
            PoseStamped,
            options.request_topic,
            self.handle_request,
            10,
        )
        self.preflight_request_subscription = self.create_subscription(
            String,
            options.preflight_request_topic,
            self.handle_preflight_request,
            10,
        )
        self.joint_request_subscription = self.create_subscription(
            String,
            options.joint_request_topic,
            self.handle_joint_request,
            10,
        )
        self.frustum_config_subscription = self.create_subscription(
            String,
            options.frustum_config_topic,
            self.handle_frustum_config,
            10,
        )

    def handle_frustum_config(self, message):
        try:
            payload = json.loads(message.data)
            self.frustum_config = validate_frustum_config(payload)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self.frustum_config = None
            self.get_logger().error(
                "Rejected invalid frustum guard configuration: "
                f"{type(exc).__name__}: {exc}"
            )
            return

        self.get_logger().info(
            "Frustum guard configured: "
            f"corners={len(self.frustum_config['corners_local'])}, "
            f"margin={self.frustum_config['margin_px']:.0f}px, "
            "required="
            f"{self.frustum_config['min_in_frame_fraction'] * 100:.0f}%"
        )

    def wait_until_ready(self):
        if self.options.ik_precheck and not self.options.publish_only:
            self.ik_available = self.ik_client.wait_for_service(
                timeout_sec=self.options.ik_service_timeout
            )

            if self.ik_available:
                self.get_logger().info(
                    "IK pre-check enabled: "
                    f"{self.options.ik_service} "
                    f"(timeout={self.options.ik_timeout}s, "
                    f"attempts={self.options.ik_attempts}, "
                    f"avoid_collisions={self.options.ik_avoid_collisions})"
                )
            else:
                self.get_logger().warning(
                    f"IK service {self.options.ik_service} not available - "
                    "pre-check disabled. Unreachable candidates will again "
                    "cost a full planning timeout each."
                )

        if self.options.publish_only:
            self.get_logger().info(
                "--publish-only: skipping MoveGroup action server wait."
            )
            return
        self.get_logger().info(
            f"Waiting for MoveGroup action server: {self.options.move_action}"
        )
        if not self.move_group_client.wait_for_server(
            timeout_sec=self.options.server_timeout
        ):
            raise RuntimeError(
                "MoveGroup action server is not available: "
                f"{self.options.move_action}"
            )

        if self.options.frustum_guard and not self.options.plan_only:
            self.get_logger().info(
                "Waiting for trajectory execution action server: "
                f"{self.options.execute_action}"
            )
            if not self.execute_trajectory_client.wait_for_server(
                timeout_sec=self.options.server_timeout
            ):
                raise RuntimeError(
                    "ExecuteTrajectory action server is not available: "
                    f"{self.options.execute_action}"
                )

            self.fk_available = self.fk_client.wait_for_service(
                timeout_sec=self.options.server_timeout
            )
            if not self.fk_available:
                raise RuntimeError(
                    "Frustum guard requires the MoveIt FK service: "
                    f"{self.options.fk_service}"
                )
            self.get_logger().info(
                "Final stopped-pose frustum guard enabled; visibility "
                "during motion is allowed to leave the camera frame."
            )

    def handle_request(self, message):
        self.pending_request = message

    def handle_preflight_request(self, message):
        self.pending_preflight_request = str(message.data)

    def handle_joint_request(self, message):
        self.pending_joint_request = str(message.data)

    def _pose_from_preflight_payload(self, payload):
        if int(payload.get("schema_version", 0)) != 1:
            raise ValueError("unsupported preflight schema")
        position = payload["position"]
        orientation = payload["orientation_xyzw"]
        if len(position) != 3 or len(orientation) != 4:
            raise ValueError("invalid pose dimensions")

        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = str(
            payload.get("frame_id") or self.options.frame_id
        )
        pose_stamped.header.stamp = self.get_clock().now().to_msg()
        pose_stamped.pose.position.x = float(position[0])
        pose_stamped.pose.position.y = float(position[1])
        pose_stamped.pose.position.z = float(position[2])
        pose_stamped.pose.orientation.x = float(orientation[0])
        pose_stamped.pose.orientation.y = float(orientation[1])
        pose_stamped.pose.orientation.z = float(orientation[2])
        pose_stamped.pose.orientation.w = float(orientation[3])
        return pose_stamped

    def process_pending_preflight_request(self):
        if self.pending_preflight_request is None:
            return

        raw_request = self.pending_preflight_request
        self.pending_preflight_request = None
        result_payload = {
            "schema_version": 1,
            "success": False,
            "reason": "preflight_failed",
        }

        try:
            payload = json.loads(raw_request)
            label = str(payload.get("label", "unnamed"))
            result_payload["label"] = label
            pose_stamped = self._pose_from_preflight_payload(payload)
            self.last_move_timing = None
            started_at = time.monotonic()
            reachable, ik_reason = self.check_ik(pose_stamped)
            ik_seconds = time.monotonic() - started_at

            if not reachable:
                result_payload["reason"] = ik_reason
            else:
                original_plan_only = self.options.plan_only
                try:
                    # This request must never execute.  It only asks MoveIt
                    # for a trajectory and records that trajectory's final
                    # joint state (the IK branch MoveIt actually selected).
                    self.options.plan_only = True
                    self.last_move_failure_reason = None
                    success = self.move_to_pose(pose_stamped=pose_stamped)
                finally:
                    self.options.plan_only = original_plan_only

                target = self.last_planned_joint_target
                if not success or target is None:
                    result_payload["reason"] = (
                        self.last_move_failure_reason or "plan_failed"
                    )
                else:
                    if self.options.frustum_guard:
                        tcp_pose = self.compute_tcp_pose(
                            target["joint_names"],
                            target["positions"],
                        )
                        fraction = self.project_object_framing(tcp_pose)
                        required = self.frustum_config[
                            "min_in_frame_fraction"
                        ]
                        if fraction + 1e-12 < required:
                            raise ValueError(
                                "endpoint_visibility_failed: "
                                f"{fraction:.3f} < {required:.3f}"
                            )
                        target["in_frame_fraction"] = float(fraction)

                    result_payload.update(target)
                    result_payload.update(
                        {
                            "success": True,
                            "reason": "preflight_ok",
                            "ik_time_s": float(ik_seconds),
                            "joint_source": (
                                "planned_trajectory_final_state"
                            ),
                        }
                    )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            result_payload["reason"] = (
                f"invalid_preflight:{type(exc).__name__}:{exc}"
            )
        except RuntimeError as exc:
            result_payload["reason"] = f"preflight_runtime_error:{exc}"

        message = String()
        message.data = json.dumps(
            result_payload,
            separators=(",", ":"),
        )
        self.preflight_result_publisher.publish(message)
        self.get_logger().info(
            "Preflight result: "
            f"label={result_payload.get('label', 'unknown')} "
            f"success={result_payload['success']} "
            f"reason={result_payload['reason']}"
        )
        # A plan-only MoveGroup request can remain busy briefly after its
        # result.  Give it time to release before the next candidate.
        time.sleep(self.options.settle_time)

    def process_pending_joint_request(self):
        if self.pending_joint_request is None:
            return

        raw_request = self.pending_joint_request
        self.pending_joint_request = None
        success = False
        reason = "invalid_joint_request"
        touched_move_group = False

        try:
            payload = json.loads(raw_request)
            if int(payload.get("schema_version", 0)) != 1:
                raise ValueError("unsupported joint request schema")
            label = str(payload.get("label", "cached_joint_target"))
            joint_names = [str(name) for name in payload["joint_names"]]
            positions = [float(value) for value in payload["positions"]]
            if len(joint_names) != len(positions) or not joint_names:
                raise ValueError("joint_names/positions size mismatch")
            if self.options.plan_only or self.options.publish_only:
                raise ValueError(
                    "cached joint execution requires normal relay mode"
                )

            self.get_logger().info(
                f"Received cached joint goal: {label}, "
                f"joints={len(joint_names)}"
            )
            touched_move_group = True
            self.last_move_timing = None
            self.last_ik_seconds = 0.0
            self.last_move_failure_reason = None
            success = self.move_to_pose(
                joint_target=(joint_names, positions),
                direct_stream=True,
            )
            reason = (
                "cached_joint_succeeded"
                if success
                else self.last_move_failure_reason or "joint_plan_failed"
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            reason = f"invalid_joint_request:{type(exc).__name__}:{exc}"

        timing = getattr(self, "last_move_timing", None)
        if timing is not None:
            reason = (
                f"{reason} plan={timing['plan']:.3f} "
                f"traj={timing['traj']:.3f} "
                f"ik={timing['ik']:.3f} "
                f"accept={timing['accept']:.3f} "
                f"guard={timing.get('guard', 0.0):.3f} "
                f"exec={timing['exec']:.3f}"
            )
            self.last_move_timing = None

        reason_message = String()
        reason_message.data = reason
        self.reason_publisher.publish(reason_message)
        result = Bool()
        result.data = bool(success)
        self.result_publisher.publish(result)

        if touched_move_group:
            time.sleep(self.options.settle_time)

    def process_pending_request(self):
        if self.pending_request is None:
            return

        request = self.pending_request
        self.pending_request = None

        self.get_logger().info(
            "Received NBV move request: "
            f"position=({request.pose.position.x:.4f}, "
            f"{request.pose.position.y:.4f}, "
            f"{request.pose.position.z:.4f})"
        )

        touched_move_group = False

        if self.options.publish_only:
            self.preview_publisher.publish(request)
            self.get_logger().info(
                f"--publish-only: republished on {self.options.preview_topic} "
                "- robot not moved."
            )
            success, reason = True, "publish_only"
        else:
            started_at = time.monotonic()
            reachable, ik_reason = self.check_ik(request)

            if not reachable:
                # MoveGroup을 건드리지 않았으므로 settle도 필요 없다.
                elapsed = time.monotonic() - started_at
                self.get_logger().info(
                    f"IK pre-check rejected the pose in {elapsed * 1000:.0f}ms "
                    f"({ik_reason}) - skipping MoveGroup entirely."
                )
                success, reason = False, ik_reason
            else:
                self.last_ik_seconds = time.monotonic() - started_at
                touched_move_group = True
                self.last_move_failure_reason = None
                success = self.move_to_pose(request)
                reason = (
                    "succeeded"
                    if success
                    else self.last_move_failure_reason or "plan_failed"
                )

        # reason을 result보다 먼저 보낸다. Isaac 쪽은 Bool 결과가
        # 도착하면 대기 루프를 빠져나가므로, 반대 순서로 보내면 계측
        # 문자열을 놓칠 수 있다.
        timing = getattr(self, "last_move_timing", None)

        if timing is not None:
            reason = (
                f"{reason} plan={timing['plan']:.3f} "
                f"traj={timing['traj']:.3f} "
                f"ik={timing['ik']:.3f} "
                f"accept={timing['accept']:.3f} "
                f"guard={timing.get('guard', 0.0):.3f} "
                f"exec={timing['exec']:.3f}"
            )
            self.last_move_timing = None

        reason_message = String()
        reason_message.data = reason
        self.reason_publisher.publish(reason_message)

        result = Bool()
        result.data = bool(success)
        self.result_publisher.publish(result)

        # move_group's own trajectory_execution_manager can still consider
        # itself "busy" for a moment after the action result is sent -
        # sending the next goal immediately then fails instantly with
        # "Cannot push a new trajectory while another is being executed".
        # Give it a moment to fully settle before accepting another goal.
        # IK로 걸러낸 요청은 MoveGroup을 건드리지 않았으니 건너뛴다.
        if touched_move_group:
            time.sleep(self.options.settle_time)

    def check_ik(self, pose_stamped):
        """이 자세에 IK 해가 있는가.

        (도달가능?, 사유) 튜플을 돌려준다. 서비스가 없거나 호출이
        실패하면 '도달 가능'으로 보고 넘긴다. 사전 검사 오류로 멀쩡한
        후보를 버리는 것이 플래닝 시간을 낭비하는 것보다 나쁘다.
        """
        if not (self.options.ik_precheck and self.ik_available):
            return True, "ik_skipped"

        frame_id = (
            pose_stamped.header.frame_id
            if pose_stamped.header.frame_id
            else self.options.frame_id
        )
        timeout_ns = int(self.options.ik_timeout * 1e9)
        last_error = MoveItErrorCodes.FAILURE

        for _ in range(max(1, self.options.ik_attempts)):
            request = GetPositionIK.Request()
            ik_request = PositionIKRequest()
            ik_request.group_name = self.options.group_name
            ik_request.ik_link_name = self.options.link_name
            ik_request.avoid_collisions = bool(
                self.options.ik_avoid_collisions
            )
            # is_diff=True면 현재 planning scene의 로봇 상태를 기준으로
            # 푼다. 매 후보마다 실제 현재 자세에서 출발하게 된다.
            ik_request.robot_state.is_diff = True
            ik_request.pose_stamped.header.frame_id = frame_id
            ik_request.pose_stamped.pose = pose_stamped.pose
            ik_request.timeout = Duration(
                sec=timeout_ns // 1000000000,
                nanosec=timeout_ns % 1000000000,
            )
            request.ik_request = ik_request

            future = self.ik_client.call_async(request)
            rclpy.spin_until_future_complete(
                self,
                future,
                timeout_sec=self.options.ik_timeout * 4.0 + 1.0,
            )
            response = future.result()

            if response is None:
                if not self.ik_warned:
                    self.get_logger().warning(
                        "IK service call returned nothing - treating the "
                        "pose as reachable and falling through to MoveGroup."
                    )
                    self.ik_warned = True

                return True, "ik_no_response"

            error_value = response.error_code.val

            if error_value == MoveItErrorCodes.SUCCESS:
                return True, "ik_ok"

            last_error = error_value

        return False, f"ik_{moveit_error_name(last_error).lower()}"

    def move_to_pose(
        self,
        pose_stamped=None,
        joint_target=None,
        direct_stream=False,
    ):
        self.last_planned_joint_target = None
        goal = MoveGroup.Goal()
        goal.request.group_name = self.options.group_name
        goal.request.num_planning_attempts = self.options.planning_attempts
        goal.request.allowed_planning_time = self.options.planning_time
        goal.request.max_velocity_scaling_factor = (
            self.options.velocity_scaling
        )
        goal.request.max_acceleration_scaling_factor = (
            self.options.acceleration_scaling
        )
        goal.request.start_state.is_diff = True
        if joint_target is None:
            if pose_stamped is None:
                raise ValueError("pose_stamped or joint_target is required")
            frame_id = (
                pose_stamped.header.frame_id
                if pose_stamped.header.frame_id
                else self.options.frame_id
            )
            goal.request.goal_constraints.append(
                make_goal_constraints(
                    frame_id=frame_id,
                    link_name=self.options.link_name,
                    pose=pose_stamped.pose,
                    position_tolerance=self.options.position_tolerance,
                    orientation_tolerance=self.options.orientation_tolerance,
                )
            )
        else:
            joint_names, positions = joint_target
            goal.request.goal_constraints.append(
                make_joint_goal_constraints(joint_names, positions)
            )
        # With the guard enabled, planning and execution must be separate:
        # MoveGroup first returns the RobotTrajectory, we inspect its final
        # stopped pose, and only an in-frame endpoint is executed. Intermediate
        # trajectory poses are intentionally allowed to leave the camera view.
        separate_execution = (
            (self.options.frustum_guard or direct_stream)
            and not self.options.plan_only
        )
        goal.planning_options.plan_only = (
            self.options.plan_only or separate_execution
        )
        goal.planning_options.look_around = False
        goal.planning_options.replan = True
        goal.planning_options.planning_scene_diff.is_diff = True
        goal.planning_options.planning_scene_diff.robot_state.is_diff = True

        timeout = (
            self.options.goal_timeout
            if self.options.goal_timeout > 0.0
            else None
        )

        # move_ok(1.86s 평균) 중 실제 주행은 계획 궤적 기준 0.60s뿐인데
        # 나머지 0.96s가 어디로 가는지 알 수 없었다. goal 왕복을 두
        # 구간으로 쪼개 어느 쪽이 먹는지 본다:
        #   accept = goal 전송 -> MoveGroup이 수락 (플래닝은 보통 여기)
        #   exec   = 수락 -> 결과 도착 (실제 실행 + 컨트롤러 정착)
        goal_sent_at = time.monotonic()
        future = self.move_group_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)
        goal_handle = future.result()
        accept_seconds = time.monotonic() - goal_sent_at

        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().warning("MoveIt goal was rejected.")
            self.last_move_failure_reason = "plan_rejected"
            return False

        accepted_at = time.monotonic()
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(
            self,
            result_future,
            timeout_sec=timeout,
        )
        wrapped_result = result_future.result()

        if wrapped_result is None:
            # 제한 시간 안에 안 끝났다. 목표를 취소하지 않으면 다음
            # 목표를 보낼 때 move_group이 여전히 바쁘다고 거부한다.
            self.get_logger().warning(
                f"MoveIt goal exceeded {self.options.goal_timeout:.1f}s "
                "- cancelling."
            )
            cancel_future = goal_handle.cancel_goal_async()
            rclpy.spin_until_future_complete(
                self,
                cancel_future,
                timeout_sec=2.0,
            )

            self.last_move_failure_reason = "plan_timeout"
            return False

        error_value = wrapped_result.result.error_code.val

        if (
            wrapped_result.status != GoalStatus.STATUS_SUCCEEDED
            or error_value != MoveItErrorCodes.SUCCESS
        ):
            self.get_logger().warning(
                "MoveIt move failed: "
                f"{moveit_error_name(error_value)} ({error_value})"
            )
            self.last_move_failure_reason = (
                f"plan_{moveit_error_name(error_value).lower()}"
            )
            return False

        # 플래닝과 실제 실행 시간을 나눠 기록한다. 뷰포인트를 미리
        # 고정해두면 플래닝 비용은 사라지므로, 순수 이동 시간이
        # 얼마인지가 시간 예산의 하한이 된다.
        result = wrapped_result.result
        planning_time = float(getattr(result, "planning_time", 0.0))
        trajectory_duration = 0.0

        try:
            joint_trajectory = result.planned_trajectory.joint_trajectory
            points = joint_trajectory.points

            if points:
                stamp = points[-1].time_from_start
                trajectory_duration = (
                    stamp.sec + stamp.nanosec * 1e-9
                )
                self.last_planned_joint_target = {
                    "joint_names": [
                        str(name) for name in joint_trajectory.joint_names
                    ],
                    "positions": [
                        float(value) for value in points[-1].positions
                    ],
                    "planning_time_s": planning_time,
                    "trajectory_duration_s": trajectory_duration,
                }
        except AttributeError:
            pass

        planning_result_seconds = time.monotonic() - accepted_at
        guard_seconds = 0.0
        execution_seconds = planning_result_seconds

        if separate_execution:
            if self.options.frustum_guard:
                guard_started_at = time.monotonic()
                guard_ok = self.validate_trajectory_endpoint_frustum(
                    result.planned_trajectory,
                    trajectory_start=result.trajectory_start,
                )
                guard_seconds = time.monotonic() - guard_started_at

                if not guard_ok:
                    self.last_move_timing = {
                        "plan": planning_time,
                        "traj": trajectory_duration,
                        "ik": float(getattr(self, "last_ik_seconds", 0.0)),
                        "accept": accept_seconds,
                        "guard": guard_seconds,
                        "exec": 0.0,
                    }
                    return False

            execution_started_at = time.monotonic()
            if direct_stream:
                executed = self.stream_trajectory_to_isaac(
                    result.planned_trajectory,
                    trajectory_start=result.trajectory_start,
                )
            else:
                executed = self.execute_trajectory(
                    result.planned_trajectory
                )
            if not executed:
                execution_seconds = time.monotonic() - execution_started_at
                self.last_move_timing = {
                    "plan": planning_time,
                    "traj": trajectory_duration,
                    "ik": float(getattr(self, "last_ik_seconds", 0.0)),
                    "accept": accept_seconds,
                    "guard": guard_seconds,
                    "exec": execution_seconds,
                }
                return False
            execution_seconds = time.monotonic() - execution_started_at

        # reason 토픽으로도 내보내려고 남겨둔다. Isaac 쪽 rclpy는
        # moveit_msgs가 없어 이 결과를 직접 못 읽으므로, 문자열로
        # 실어 보내는 것이 유일한 경로다.
        self.last_move_timing = {
            "plan": planning_time,
            "traj": trajectory_duration,
            "ik": float(getattr(self, "last_ik_seconds", 0.0)),
            "accept": accept_seconds,
            "guard": guard_seconds,
            "exec": execution_seconds,
        }

        self.get_logger().info(
            "MoveIt move succeeded. "
            f"[TIMING] moveit planning={planning_time:.3f} "
            f"trajectory={trajectory_duration:.3f} "
            f"ik={self.last_move_timing['ik']:.3f} "
            f"accept={accept_seconds:.3f} "
            f"guard={guard_seconds:.3f} "
            f"exec={execution_seconds:.3f}"
        )

        return True

    def stream_trajectory_to_isaac(
        self,
        trajectory,
        trajectory_start=None,
    ):
        """Stream a cached-joint plan without joint-space success gating.

        The normal ExecuteTrajectory path is intentionally retained for all
        pose requests.  Only the temporary cached-joint experiment uses this
        path; its actual stopped TCP/object pose is measured in Isaac before
        the candidate is accepted and captured.
        """
        joint_trajectory = trajectory.joint_trajectory
        joint_names = [str(name) for name in joint_trajectory.joint_names]
        point_data = [
            (
                duration_seconds(point.time_from_start),
                [float(value) for value in point.positions],
            )
            for point in joint_trajectory.points
        ]

        if not joint_names or not point_data:
            self.last_move_failure_reason = "direct_stream_empty_trajectory"
            return False
        if any(len(positions) != len(joint_names) for _, positions in point_data):
            self.last_move_failure_reason = "direct_stream_invalid_trajectory"
            return False

        if trajectory_start is not None:
            start_by_name = dict(
                zip(
                    trajectory_start.joint_state.name,
                    trajectory_start.joint_state.position,
                )
            )
            if all(name in start_by_name for name in joint_names):
                point_data.insert(
                    0,
                    (
                        0.0,
                        [float(start_by_name[name]) for name in joint_names],
                    ),
                )

        point_data.sort(key=lambda item: item[0])
        final_time = max(0.0, point_data[-1][0])
        rate_hz = max(1.0, float(self.options.direct_publish_rate))
        period = 1.0 / rate_hz
        started_at = time.monotonic()
        segment_index = 0

        def publish(positions):
            message = JointState()
            message.header.stamp = self.get_clock().now().to_msg()
            message.name = joint_names
            message.position = [float(value) for value in positions]
            self.isaac_joint_command_publisher.publish(message)

        while True:
            elapsed = time.monotonic() - started_at
            if elapsed >= final_time:
                break

            while (
                segment_index + 1 < len(point_data)
                and point_data[segment_index + 1][0] <= elapsed
            ):
                segment_index += 1

            if segment_index + 1 >= len(point_data):
                positions = point_data[-1][1]
            else:
                start_time, start_positions = point_data[segment_index]
                end_time, end_positions = point_data[segment_index + 1]
                duration = max(1e-9, end_time - start_time)
                ratio = min(1.0, max(0.0, (elapsed - start_time) / duration))
                positions = [
                    start_value + ratio * (end_value - start_value)
                    for start_value, end_value in zip(
                        start_positions,
                        end_positions,
                    )
                ]

            publish(positions)
            sleep_time = period - ((time.monotonic() - started_at) % period)
            time.sleep(max(0.001, sleep_time))

        final_positions = point_data[-1][1]
        hold_until = time.monotonic() + max(
            0.0,
            float(self.options.direct_final_hold),
        )
        while time.monotonic() < hold_until:
            publish(final_positions)
            time.sleep(period)
        publish(final_positions)

        self.get_logger().info(
            "Cached-joint IMSI trajectory streamed directly to Isaac: "
            f"duration={final_time:.3f}s, rate={rate_hz:.0f}Hz. "
            "Final success will be decided from measured TCP/object pose."
        )
        return True

    def sampled_trajectory_positions(
        self,
        joint_trajectory,
        trajectory_start=None,
    ):
        """Interpolate a MoveIt trajectory in both time and joint space."""
        point_data = [
            (
                duration_seconds(point.time_from_start),
                [float(value) for value in point.positions],
            )
            for point in joint_trajectory.points
        ]

        # Isaac's FollowJointTrajectory bridge prepends the currently measured
        # joints at t=0 before interpolating. MoveGroup's trajectory_start is
        # the closest pre-execution equivalent, so validate that first segment
        # too instead of assuming the first planned point is the exact start.
        if trajectory_start is not None:
            start_by_name = dict(
                zip(
                    trajectory_start.joint_state.name,
                    trajectory_start.joint_state.position,
                )
            )
            if all(
                name in start_by_name
                for name in joint_trajectory.joint_names
            ):
                point_data.insert(
                    0,
                    (
                        0.0,
                        [
                            float(start_by_name[name])
                            for name in joint_trajectory.joint_names
                        ],
                    ),
                )

        if not point_data:
            return []

        sample_period = max(1e-4, self.options.frustum_sample_period)
        max_joint_step = max(1e-4, self.options.frustum_max_joint_step)
        samples = [point_data[0]]

        for start, end in zip(point_data, point_data[1:]):
            start_time, start_positions = start
            end_time, end_positions = end
            duration = max(0.0, end_time - start_time)

            if len(start_positions) != len(end_positions):
                raise ValueError("trajectory point position lengths differ")

            max_delta = max(
                (
                    abs(end_value - start_value)
                    for start_value, end_value in zip(
                        start_positions,
                        end_positions,
                    )
                ),
                default=0.0,
            )
            steps = max(
                1,
                int(math.ceil(duration / sample_period)),
                int(math.ceil(max_delta / max_joint_step)),
            )

            for step in range(1, steps + 1):
                ratio = step / steps
                positions = [
                    start_value + ratio * (end_value - start_value)
                    for start_value, end_value in zip(
                        start_positions,
                        end_positions,
                    )
                ]
                samples.append(
                    (start_time + ratio * duration, positions)
                )

                if len(samples) > 500:
                    raise ValueError(
                        "trajectory guard sample count exceeded 500"
                    )

        return samples

    def compute_tcp_pose(self, joint_names, joint_positions):
        request = GetPositionFK.Request()
        request.header.frame_id = self.options.frame_id
        request.fk_link_names = [self.options.link_name]
        request.robot_state.is_diff = True
        request.robot_state.joint_state.name = list(joint_names)
        request.robot_state.joint_state.position = list(joint_positions)

        future = self.fk_client.call_async(request)
        rclpy.spin_until_future_complete(
            self,
            future,
            timeout_sec=self.options.fk_timeout,
        )
        response = future.result()

        if response is None:
            raise RuntimeError("compute_fk timed out")
        if response.error_code.val != MoveItErrorCodes.SUCCESS:
            raise RuntimeError(
                "compute_fk failed: "
                f"{moveit_error_name(response.error_code.val)}"
            )
        if not response.pose_stamped:
            raise RuntimeError("compute_fk returned no TCP pose")

        return response.pose_stamped[0].pose

    def project_object_framing(self, tcp_pose):
        config = self.frustum_config
        tcp_matrix = pose_matrix(tcp_pose)
        object_matrix = matrix_multiply(
            tcp_matrix,
            config["grasp_offset"],
        )
        camera = config["camera_matrix"]
        intrinsic = config["intrinsic_matrix"]
        width, height = config["image_size"]
        margin = config["margin_px"]

        camera_position = [camera[row][3] for row in range(3)]
        camera_right = [camera[row][0] for row in range(3)]
        camera_up = [camera[row][1] for row in range(3)]
        camera_view = [-camera[row][2] for row in range(3)]
        fx = intrinsic[0][0]
        fy = intrinsic[1][1]
        cx = intrinsic[0][2]
        cy = intrinsic[1][2]
        inside = 0

        for corner in config["corners_local"]:
            point = transform_point(object_matrix, corner)
            offset = [
                point[index] - camera_position[index]
                for index in range(3)
            ]
            depth = dot(offset, camera_view)

            if depth <= 1e-6:
                continue

            u = cx + fx * dot(offset, camera_right) / depth
            v = cy - fy * dot(offset, camera_up) / depth
            if margin <= u < width - margin and margin <= v < height - margin:
                inside += 1

        return inside / len(config["corners_local"])

    def validate_trajectory_endpoint_frustum(
        self,
        trajectory,
        trajectory_start=None,
    ):
        """Require only the final stopped trajectory pose to be in frame."""
        del trajectory_start  # Kept in the signature for call-site compatibility.

        if self.frustum_config is None:
            self.get_logger().error(
                "Frustum guard has no valid configuration; refusing to "
                "execute the planned trajectory."
            )
            self.last_move_failure_reason = "frustum_guard_missing"
            return False
        if not self.fk_available:
            self.last_move_failure_reason = "frustum_guard_fk_unavailable"
            return False

        joint_trajectory = trajectory.joint_trajectory
        joint_names = list(joint_trajectory.joint_names)

        if not joint_names or not joint_trajectory.points:
            self.get_logger().error(
                "MoveIt returned an empty joint trajectory; refusing execution."
            )
            self.last_move_failure_reason = "frustum_guard_empty_trajectory"
            return False

        required_fraction = self.frustum_config[
            "min_in_frame_fraction"
        ]
        final_point = joint_trajectory.points[-1]
        final_positions = [float(value) for value in final_point.positions]

        if len(final_positions) != len(joint_names):
            self.get_logger().error(
                "Final trajectory point has a different number of joint "
                "positions than joint names; refusing execution."
            )
            self.last_move_failure_reason = "frustum_guard_invalid_trajectory"
            return False

        try:
            tcp_pose = self.compute_tcp_pose(joint_names, final_positions)
            fraction = self.project_object_framing(tcp_pose)
        except (RuntimeError, ValueError) as exc:
            self.get_logger().error(
                f"Final-pose frustum FK/projection failed: {exc}"
            )
            self.last_move_failure_reason = "frustum_guard_fk_failed"
            return False

        if fraction + 1e-12 < required_fraction:
            self.get_logger().warning(
                "Rejected trajectory before execution: object is outside "
                "the camera frame at the final stopped pose, "
                f"in_frame={fraction * 100:.0f}% < "
                f"{required_fraction * 100:.0f}%"
            )
            self.last_move_failure_reason = "endpoint_visibility_failed"
            return False

        self.get_logger().info(
            "Final-pose frustum guard passed: "
            f"in_frame={fraction * 100:.0f}%"
        )
        return True

    def execute_trajectory(self, trajectory):
        goal = ExecuteTrajectory.Goal()
        goal.trajectory = trajectory
        timeout = (
            self.options.goal_timeout
            if self.options.goal_timeout > 0.0
            else None
        )

        future = self.execute_trajectory_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)
        goal_handle = future.result()

        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().warning(
                "ExecuteTrajectory goal was rejected after guard passed."
            )
            self.last_move_failure_reason = "execute_rejected"
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(
            self,
            result_future,
            timeout_sec=timeout,
        )
        wrapped_result = result_future.result()

        if wrapped_result is None:
            self.get_logger().warning(
                f"ExecuteTrajectory exceeded {self.options.goal_timeout:.1f}s "
                "- cancelling."
            )
            cancel_future = goal_handle.cancel_goal_async()
            rclpy.spin_until_future_complete(
                self,
                cancel_future,
                timeout_sec=2.0,
            )
            self.last_move_failure_reason = "execute_timeout"
            return False

        error_value = wrapped_result.result.error_code.val
        if (
            wrapped_result.status != GoalStatus.STATUS_SUCCEEDED
            or error_value != MoveItErrorCodes.SUCCESS
        ):
            self.get_logger().warning(
                "ExecuteTrajectory failed: "
                f"{moveit_error_name(error_value)} ({error_value})"
            )
            self.last_move_failure_reason = (
                f"execute_{moveit_error_name(error_value).lower()}"
            )
            return False

        return True


def main(argv=None):
    argv = sys.argv if argv is None else argv
    cli_argv = remove_ros_args(argv)[1:]
    options = parse_arguments(cli_argv)

    rclpy.init(args=argv)
    node = NbvArmMoveRelay(options)

    try:
        node.wait_until_ready()
        if options.publish_only:
            node.get_logger().warning(
                "Running with --publish-only: MoveGroup is never contacted - "
                f"each target pose is just republished on {options.preview_topic} "
                "for RViz. The robot will not move."
            )
        elif options.plan_only:
            node.get_logger().warning(
                "Running with --plan-only: goals will be PLANNED but never "
                "EXECUTED - the robot will not move. Check RViz's Motion "
                "Planning display for the previewed trajectory/goal pose."
            )
        node.get_logger().info("NBV arm move relay ready.")

        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)
            node.process_pending_preflight_request()
            node.process_pending_joint_request()
            node.process_pending_request()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
