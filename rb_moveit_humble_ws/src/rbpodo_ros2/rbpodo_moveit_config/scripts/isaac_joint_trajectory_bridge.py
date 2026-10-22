#!/usr/bin/env python3
"""Bridge MoveIt FollowJointTrajectory goals to Isaac JointState commands."""

import math
import threading
import time

import rclpy
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint


DEFAULT_JOINT_NAMES = [
    "base",
    "shoulder",
    "elbow",
    "wrist1",
    "wrist2",
    "wrist3",
]
DEFAULT_SHOULDER_MIN_POSITION = -math.pi
DEFAULT_SHOULDER_MAX_POSITION = math.pi


def duration_to_seconds(duration):
    return float(duration.sec) + float(duration.nanosec) * 1e-9


def make_point(
    positions,
    time_from_start=0.0,
):
    point = JointTrajectoryPoint()
    point.positions = list(positions)
    sec = int(math.floor(time_from_start))
    point.time_from_start.sec = sec
    point.time_from_start.nanosec = int(
        round((time_from_start - sec) * 1e9)
    )
    return point


def interpolate_positions(
    start_positions,
    end_positions,
    ratio,
):
    ratio = min(
        1.0,
        max(
            0.0,
            ratio,
        ),
    )
    return [
        start + (end - start) * ratio
        for start, end in zip(
            start_positions,
            end_positions,
        )
    ]


def angular_position_error(
    target_position,
    current_position,
):
    difference = float(target_position) - float(current_position)
    return abs(
        math.atan2(
            math.sin(difference),
            math.cos(difference),
        )
    )


def format_joint_position_errors(
    joint_names,
    target_positions,
    current_positions,
):
    lines = []

    for joint_name, target_position, current_position in zip(
        joint_names,
        target_positions,
        current_positions,
    ):
        error = angular_position_error(
            target_position,
            current_position,
        )
        lines.append(
            f"{joint_name}: target={target_position:.5f}, "
            f"current={current_position:.5f}, "
            f"error={error:.5f} rad "
            f"({math.degrees(error):.2f} deg)"
        )

    return "; ".join(
        lines
    )


class IsaacJointTrajectoryBridge(Node):
    def __init__(self):
        super().__init__("isaac_joint_trajectory_bridge")

        self.declare_parameter(
            "joint_names",
            DEFAULT_JOINT_NAMES,
        )
        self.declare_parameter(
            "isaac_joint_states_topic",
            "/isaac_joint_states",
        )
        self.declare_parameter(
            "joint_states_topic",
            "/joint_states",
        )
        self.declare_parameter(
            "isaac_joint_commands_topic",
            "/isaac_joint_commands",
        )
        self.declare_parameter(
            "publish_rate_hz",
            50.0,
        )
        self.declare_parameter(
            "final_hold_duration",
            0.4,
        )
        self.declare_parameter(
            "goal_tolerance",
            0.03,
        )
        self.declare_parameter(
            "goal_wait_timeout",
            5.0,
        )
        self.declare_parameter(
            "goal_stable_duration",
            0.2,
        )
        self.declare_parameter(
            "shoulder_min_position",
            DEFAULT_SHOULDER_MIN_POSITION,
        )
        self.declare_parameter(
            "shoulder_max_position",
            DEFAULT_SHOULDER_MAX_POSITION,
        )

        self.joint_names = list(
            self.get_parameter("joint_names").value
        )
        self.isaac_joint_states_topic = self.get_parameter(
            "isaac_joint_states_topic"
        ).value
        self.joint_states_topic = self.get_parameter(
            "joint_states_topic"
        ).value
        self.isaac_joint_commands_topic = self.get_parameter(
            "isaac_joint_commands_topic"
        ).value
        self.publish_rate_hz = float(
            self.get_parameter("publish_rate_hz").value
        )
        self.final_hold_duration = float(
            self.get_parameter("final_hold_duration").value
        )
        self.goal_tolerance = float(
            self.get_parameter("goal_tolerance").value
        )
        self.goal_wait_timeout = float(
            self.get_parameter("goal_wait_timeout").value
        )
        self.goal_stable_duration = float(
            self.get_parameter("goal_stable_duration").value
        )
        self.shoulder_min_position = float(
            self.get_parameter("shoulder_min_position").value
        )
        self.shoulder_max_position = float(
            self.get_parameter("shoulder_max_position").value
        )

        self.callback_group = ReentrantCallbackGroup()
        self.state_lock = threading.Lock()
        self.latest_joint_state = None

        self.command_publisher = self.create_publisher(
            JointState,
            self.isaac_joint_commands_topic,
            10,
        )
        self.joint_state_publisher = self.create_publisher(
            JointState,
            self.joint_states_topic,
            10,
        )
        self.joint_state_subscription = self.create_subscription(
            JointState,
            self.isaac_joint_states_topic,
            self.handle_isaac_joint_state,
            10,
            callback_group=self.callback_group,
        )
        self.action_server = ActionServer(
            self,
            FollowJointTrajectory,
            "joint_trajectory_controller/follow_joint_trajectory",
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=self.callback_group,
        )

        self.get_logger().info(
            "Isaac trajectory bridge ready: "
            f"action=/joint_trajectory_controller/follow_joint_trajectory, "
            f"state_in={self.isaac_joint_states_topic}, "
            f"state_out={self.joint_states_topic}, "
            f"command_out={self.isaac_joint_commands_topic}, "
            f"goal_tolerance={self.goal_tolerance:.4f} rad, "
            f"shoulder_min={self.shoulder_min_position:.4f} rad, "
            f"shoulder_max={self.shoulder_max_position:.4f} rad"
        )

    def _now_seconds(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def handle_isaac_joint_state(
        self,
        message,
    ):
        republished_message = JointState()
        republished_message.header = message.header
        republished_message.header.stamp = self.get_clock().now().to_msg()
        republished_message.name = list(message.name)
        republished_message.position = list(message.position)
        republished_message.velocity = list(message.velocity)
        republished_message.effort = list(message.effort)

        with self.state_lock:
            self.latest_joint_state = republished_message

        self.joint_state_publisher.publish(republished_message)

    def goal_callback(
        self,
        goal_request,
    ):
        trajectory = goal_request.trajectory
        missing_joints = [
            joint_name
            for joint_name in trajectory.joint_names
            if joint_name not in self.joint_names
        ]

        if missing_joints:
            self.get_logger().error(
                "Rejecting trajectory with unknown joints: "
                f"{missing_joints}"
            )
            return GoalResponse.REJECT

        if not trajectory.points:
            self.get_logger().error(
                "Rejecting empty trajectory."
            )
            return GoalResponse.REJECT

        limit_error = self.find_trajectory_limit_error(
            joint_names=list(trajectory.joint_names),
            points=list(trajectory.points),
        )
        if limit_error is not None:
            self.get_logger().error(
                "Rejecting trajectory outside configured joint limits: "
                f"{limit_error}"
            )
            return GoalResponse.REJECT

        self.get_logger().info(
            "Accepted trajectory goal with "
            f"{len(trajectory.points)} points for joints "
            f"{list(trajectory.joint_names)}"
        )
        return GoalResponse.ACCEPT

    def find_trajectory_limit_error(
        self,
        joint_names,
        points,
    ):
        if "shoulder" not in joint_names:
            return None

        shoulder_index = joint_names.index(
            "shoulder"
        )

        for point_index, point in enumerate(
            points
        ):
            if shoulder_index >= len(point.positions):
                return (
                    "shoulder position is missing at trajectory point "
                    f"{point_index}"
                )

            shoulder_position = float(
                point.positions[shoulder_index]
            )

            if shoulder_position < self.shoulder_min_position:
                return (
                    f"shoulder={shoulder_position:.5f} rad "
                    f"({math.degrees(shoulder_position):.2f} deg) "
                    f"below min={self.shoulder_min_position:.5f} rad "
                    f"({math.degrees(self.shoulder_min_position):.2f} deg) "
                    f"at point {point_index}"
                )

            if shoulder_position > self.shoulder_max_position:
                return (
                    f"shoulder={shoulder_position:.5f} rad "
                    f"({math.degrees(shoulder_position):.2f} deg) "
                    f"above max={self.shoulder_max_position:.5f} rad "
                    f"({math.degrees(self.shoulder_max_position):.2f} deg) "
                    f"at point {point_index}"
                )

        return None

    def cancel_callback(
        self,
        goal_handle,
    ):
        self.get_logger().warn(
            "Trajectory cancel requested."
        )
        return CancelResponse.ACCEPT

    def execute_callback(
        self,
        goal_handle,
    ):
        trajectory = goal_handle.request.trajectory
        result = FollowJointTrajectory.Result()

        try:
            self.publish_trajectory(
                goal_handle=goal_handle,
                joint_names=list(trajectory.joint_names),
                points=list(trajectory.points),
            )
        except Exception as exc:
            result.error_code = (
                FollowJointTrajectory.Result.INVALID_GOAL
            )
            result.error_string = str(exc)
            goal_handle.abort()
            self.get_logger().error(
                f"Trajectory execution aborted: {exc}"
            )
            return result

        if goal_handle.is_cancel_requested:
            goal_handle.canceled()
            result.error_code = (
                FollowJointTrajectory.Result.SUCCESSFUL
            )
            result.error_string = "Trajectory canceled."
            return result

        goal_handle.succeed()
        result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
        result.error_string = "Trajectory sent to Isaac Sim."
        self.get_logger().info(
            "Trajectory execution finished."
        )
        return result

    def publish_trajectory(
        self,
        goal_handle,
        joint_names,
        points,
    ):
        ordered_points = self.make_ordered_points(
            joint_names=joint_names,
            points=points,
        )

        start_time = self._now_seconds()
        final_time = duration_to_seconds(
            ordered_points[-1].time_from_start
        )
        period = 1.0 / max(
            1.0,
            self.publish_rate_hz,
        )

        while rclpy.ok():
            if goal_handle.is_cancel_requested:
                return

            elapsed = self._now_seconds() - start_time

            if elapsed >= final_time:
                break

            point = self.interpolate_point(
                points=ordered_points,
                elapsed=elapsed,
            )
            self.publish_command(
                joint_names=self.joint_names,
                positions=point.positions,
            )
            self.publish_feedback(
                goal_handle=goal_handle,
                joint_names=self.joint_names,
                desired=point,
            )
            time.sleep(period)

        final_point = ordered_points[-1]
        self.publish_final_hold(
            goal_handle=goal_handle,
            final_point=final_point,
            period=period,
        )
        self.wait_until_goal_reached(
            goal_handle=goal_handle,
            final_point=final_point,
            period=period,
        )

    def make_ordered_points(
        self,
        joint_names,
        points,
    ):
        current_positions = self.get_current_positions()
        ordered_points = [
            make_point(
                current_positions,
                0.0,
            )
        ]

        for point in points:
            ordered_positions = self.order_positions(
                joint_names=joint_names,
                positions=point.positions,
                fallback_positions=ordered_points[-1].positions,
            )
            ordered_point = make_point(
                ordered_positions,
                duration_to_seconds(point.time_from_start),
            )
            ordered_points.append(ordered_point)

        ordered_points.sort(
            key=lambda item: duration_to_seconds(
                item.time_from_start
            )
        )
        return ordered_points

    def get_current_positions(self):
        with self.state_lock:
            message = self.latest_joint_state

        if message is None:
            return [0.0] * len(self.joint_names)

        position_by_name = {
            name: position
            for name, position in zip(
                message.name,
                message.position,
            )
        }
        return [
            position_by_name.get(
                joint_name,
                0.0,
            )
            for joint_name in self.joint_names
        ]

    def order_positions(
        self,
        joint_names,
        positions,
        fallback_positions,
    ):
        position_by_name = {
            name: position
            for name, position in zip(
                joint_names,
                positions,
            )
        }
        return [
            position_by_name.get(
                joint_name,
                fallback,
            )
            for joint_name, fallback in zip(
                self.joint_names,
                fallback_positions,
            )
        ]

    def interpolate_point(
        self,
        points,
        elapsed,
    ):
        previous_point = points[0]

        for next_point in points[1:]:
            previous_time = duration_to_seconds(
                previous_point.time_from_start
            )
            next_time = duration_to_seconds(
                next_point.time_from_start
            )

            if elapsed <= next_time:
                duration = max(
                    next_time - previous_time,
                    1e-6,
                )
                ratio = (elapsed - previous_time) / duration
                positions = interpolate_positions(
                    previous_point.positions,
                    next_point.positions,
                    ratio,
                )
                return make_point(
                    positions,
                    elapsed,
                )

            previous_point = next_point

        return points[-1]

    def publish_final_hold(
        self,
        goal_handle,
        final_point,
        period,
    ):
        hold_start = self._now_seconds()

        while (
            rclpy.ok()
            and self._now_seconds() - hold_start
            < self.final_hold_duration
        ):
            if goal_handle.is_cancel_requested:
                return

            self.publish_command(
                joint_names=self.joint_names,
                positions=final_point.positions,
            )
            self.publish_feedback(
                goal_handle=goal_handle,
                joint_names=self.joint_names,
                desired=final_point,
            )
            time.sleep(period)

    def wait_until_goal_reached(
        self,
        goal_handle,
        final_point,
        period,
    ):
        if self.goal_wait_timeout <= 0.0:
            return

        deadline = self._now_seconds() + self.goal_wait_timeout
        stable_start = None
        last_error = None

        while rclpy.ok() and self._now_seconds() < deadline:
            if goal_handle.is_cancel_requested:
                return

            self.publish_command(
                joint_names=self.joint_names,
                positions=final_point.positions,
            )
            self.publish_feedback(
                goal_handle=goal_handle,
                joint_names=self.joint_names,
                desired=final_point,
            )

            errors = self.compute_position_errors(
                target_positions=final_point.positions,
            )
            last_error = max(errors)
            now = self._now_seconds()

            if last_error <= self.goal_tolerance:
                if stable_start is None:
                    stable_start = now
                if now - stable_start >= self.goal_stable_duration:
                    self.get_logger().info(
                        "Isaac joints reached trajectory goal: "
                        f"max_error={last_error:.5f} rad"
                    )
                    return
            else:
                stable_start = None

            time.sleep(period)

        raise RuntimeError(
            "Isaac joints did not reach trajectory goal within "
            f"{self.goal_wait_timeout:.2f}s: "
            f"max_error={last_error if last_error is not None else float('inf'):.5f} rad, "
            f"tolerance={self.goal_tolerance:.5f} rad; "
            f"{self.format_current_goal_error(final_point.positions)}"
        )

    def compute_position_errors(
        self,
        target_positions,
    ):
        current_positions = self.get_current_positions()
        return [
            angular_position_error(
                target_position,
                current_position,
            )
            for target_position, current_position in zip(
                target_positions,
                current_positions,
            )
        ]

    def format_current_goal_error(
        self,
        target_positions,
    ):
        current_positions = self.get_current_positions()
        return format_joint_position_errors(
            joint_names=self.joint_names,
            target_positions=target_positions,
            current_positions=current_positions,
        )

    def publish_command(
        self,
        joint_names,
        positions,
    ):
        message = JointState()
        message.header.stamp = self.get_clock().now().to_msg()
        message.name = list(joint_names)
        message.position = list(positions)
        self.command_publisher.publish(message)

    def publish_feedback(
        self,
        goal_handle,
        joint_names,
        desired,
    ):
        feedback = FollowJointTrajectory.Feedback()
        feedback.header.stamp = self.get_clock().now().to_msg()
        feedback.joint_names = list(joint_names)
        feedback.desired = desired
        feedback.actual = make_point(
            self.get_current_positions(),
            duration_to_seconds(desired.time_from_start),
        )
        feedback.error = make_point(
            [
                desired_position - actual_position
                for desired_position, actual_position
                in zip(
                    desired.positions,
                    feedback.actual.positions,
                )
            ],
            duration_to_seconds(desired.time_from_start),
        )
        goal_handle.publish_feedback(feedback)


def main():
    rclpy.init()
    node = IsaacJointTrajectoryBridge()
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
