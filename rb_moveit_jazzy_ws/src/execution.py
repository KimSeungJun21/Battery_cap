#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint

class RBMoveClient(Node):

    def __init__(self):
        super().__init__("rb_move_client")

        self.client = ActionClient(
            self,
            MoveGroup,
            "/move_action",
        )

    def move_joints(self, positions):

        joint_names = [
            "base",
            "shoulder",
            "elbow",
            "wrist1",
            "wrist2",
            "wrist3",
        ]

        self.get_logger().info("Waiting for MoveIt...")
        self.client.wait_for_server()

        goal = MoveGroup.Goal()

        # ==========================================
        # Motion planning 설정
        # ==========================================

        request = goal.request

        # RB 공식 SRDF의 오타 그대로 사용
        request.group_name = "mainpulation"

        request.pipeline_id = "ompl"
        request.planner_id = "RRTConnectkConfigDefault"

        request.num_planning_attempts = 5
        request.allowed_planning_time = 5.0

        request.max_velocity_scaling_factor = 0.2
        request.max_acceleration_scaling_factor = 0.2

        # 현재 로봇 상태에서 시작
        request.start_state.is_diff = True

        # ==========================================
        # Joint goal
        # ==========================================

        constraints = Constraints()

        for name, position in zip(joint_names, positions):

            joint_constraint = JointConstraint()

            joint_constraint.joint_name = name
            joint_constraint.position = float(position)

            joint_constraint.tolerance_above = 0.01
            joint_constraint.tolerance_below = 0.01
            joint_constraint.weight = 1.0

            constraints.joint_constraints.append(
                joint_constraint
            )

        request.goal_constraints.append(constraints)

        # ==========================================
        # Plan + Execute
        # ==========================================

        goal.planning_options.plan_only = False
        goal.planning_options.look_around = False
        goal.planning_options.replan = True

        goal.planning_options.planning_scene_diff.is_diff = True
        goal.planning_options.planning_scene_diff.robot_state.is_diff = True

        self.get_logger().info(
            f"Sending joint target: {positions}"
        )

        send_future = self.client.send_goal_async(goal)

        rclpy.spin_until_future_complete(
            self,
            send_future,
        )

        goal_handle = send_future.result()

        if not goal_handle.accepted:
            self.get_logger().error(
                "MoveIt goal rejected"
            )
            return False

        self.get_logger().info(
            "MoveIt goal accepted"
        )

        result_future = goal_handle.get_result_async()

        rclpy.spin_until_future_complete(
            self,
            result_future,
        )

        result = result_future.result().result

        if result.error_code.val == 1:
            self.get_logger().info(
                "Motion completed successfully"
            )
            return True

        self.get_logger().error(
            f"Motion failed: {result.error_code.val}"
        )

        return False

def main():

    rclpy.init()

    node = RBMoveClient()

    # radian
    target = [
        0.10,    # base
        -0.20,   # shoulder
        0.20,    # elbow
        0.0,     # wrist1
        0.0,     # wrist2
        0.0,     # wrist3
    ]

    node.move_joints(target)

    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()