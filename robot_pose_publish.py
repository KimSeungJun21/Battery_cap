#!/usr/bin/env python3

import time
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Empty

class JointSequencePublisher(Node):
    def __init__(self):
        super().__init__("joint_sequence_publisher")

        self.pub = self.create_publisher(JointState, "/joint_command", 10)

        # 신호 받는 topic
        self.trigger_sub = self.create_subscription(
            Empty,
            "/start_sequence",
            self.trigger_callback,
            10,
        )

        self.trigger_requested = False

        self.joint_names = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]

    def trigger_callback(self, msg):
        self.get_logger().info("Trigger received: start pose3 -> pose4 -> pose5 -> pose2")
        self.trigger_requested = True

    def publish_position(self, position):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        msg.position = list(position)
        self.pub.publish(msg)

    def hold_position(self, position, duration=3.0, rate=10):
        dt = 1.0 / rate
        steps = int(duration * rate)

        for _ in range(steps):
            self.publish_position(position)
            rclpy.spin_once(self, timeout_sec=0.0)
            time.sleep(dt)

    def interpolate_position(self, start, goal, duration=3.0, rate=30):
        start = np.array(start, dtype=float)
        goal = np.array(goal, dtype=float)

        dt = 1.0 / rate
        steps = int(duration * rate)

        for i in range(steps + 1):
            alpha = i / steps
            pos = (1.0 - alpha) * start + alpha * goal
            self.publish_position(pos)
            rclpy.spin_once(self, timeout_sec=0.0)
            time.sleep(dt)

    def wait_for_trigger(self, hold_pose, rate=10):
        self.get_logger().info("Waiting for trigger on /start_sequence ...")

        dt = 1.0 / rate

        while rclpy.ok() and not self.trigger_requested:
            # 기다리는 동안 pose2 유지
            self.publish_position(hold_pose)
            rclpy.spin_once(self, timeout_sec=0.0)
            time.sleep(dt)

        self.trigger_requested = False

def main():
    rclpy.init()

    node = JointSequencePublisher()

    pose1 = [0.0, -1.57, 1.57, -1.57, -1.57, 0.0]

    pose2 = [1.57, -1.57, 1.57, -1.57, -1.57, 0.0]

    pose3 = [
        1.0472,    # 60°
        -1.5708,   # -90°
        1.5708,    # 90°
        -1.0472,   # -60°
        -1.0472,   # -60°
        0.0,
    ]

    pose4 = [
        1.7453,    # 120°
        -1.5708,   # -90°
        1.5708,    # 90°
        -1.0472,   # -60°
        -2.0944,   # -120°
        0.0,
    ]

    pose5 = [
        1.5708,    # 90°
        -1.0472,   # -60°
        1.0472,    # 60°
        -1.0472,   # -60°
        -1.5708,   # -90°
        0.0,
    ]

    # 처음에는 pose1 -> pose2까지만 진행
    node.get_logger().info("Moving to pose1")
    node.interpolate_position(pose1, pose1, duration=1.0)
    node.hold_position(pose1, duration=1.0)

    node.get_logger().info("Moving to pose2")
    node.interpolate_position(pose1, pose2, duration=3.0)
    node.hold_position(pose2, duration=1.0)

    # 이후에는 신호를 기다렸다가 3 -> 4 -> 5 -> 2 실행
    while rclpy.ok():
        node.wait_for_trigger(hold_pose=pose2)

        node.get_logger().info("Moving to pose3")
        node.interpolate_position(pose2, pose3, duration=3.0)
        node.hold_position(pose3, duration=1.0)

        node.get_logger().info("Moving to pose4")
        node.interpolate_position(pose3, pose4, duration=3.0)
        node.hold_position(pose4, duration=1.0)

        node.get_logger().info("Moving to pose5")
        node.interpolate_position(pose4, pose5, duration=3.0)
        node.hold_position(pose5, duration=1.0)

        node.get_logger().info("Returning to pose2")
        node.interpolate_position(pose5, pose2, duration=3.0)
        node.hold_position(pose2, duration=1.0)

        node.get_logger().info("Sequence done. Waiting for next trigger.")

    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()