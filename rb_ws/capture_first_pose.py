"""Print the robot's current pose as BATTERY_IMSI_FIRST_POSE_* env lines.

Run inside the ROS 2 Humble container while Isaac Sim is publishing:

    docker exec rb_humble bash -lc 'source /opt/ros/humble/setup.bash; \
        ROS_DOMAIN_ID=42 RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
        python3 /home/seungjun/isaac_ws/rb_ws/capture_first_pose.py'

Joints come from /joint_states; the TCP pose is the tf lookup
``world -> tcp`` (MoveIt world == robot link0). The pose is only reported
once the joints have stopped moving.
"""

import sys
import time

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformListener


JOINT_NAMES = ("base", "shoulder", "elbow", "wrist1", "wrist2", "wrist3")
BASE_FRAME = "world"
TCP_FRAME = "tcp"
STILL_RAD = 1e-4
STILL_SAMPLES = 5
TIMEOUT_S = 20.0


class PoseCapture(Node):
    def __init__(self):
        super().__init__("battery_first_pose_capture")
        self.samples = []
        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, self)
        self.create_subscription(JointState, "/joint_states", self.on_joints, 10)

    def on_joints(self, message):
        by_name = dict(zip(message.name, message.position))
        if all(name in by_name for name in JOINT_NAMES):
            self.samples.append([by_name[name] for name in JOINT_NAMES])
            self.samples = self.samples[-STILL_SAMPLES:]

    def still(self):
        if len(self.samples) < STILL_SAMPLES:
            return False
        first = self.samples[0]
        return all(
            abs(value - first[index]) <= STILL_RAD
            for sample in self.samples
            for index, value in enumerate(sample)
        )


def main():
    rclpy.init()
    node = PoseCapture()
    deadline = time.monotonic() + TIMEOUT_S
    transform = None
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
        if not node.still():
            continue
        try:
            transform = node.buffer.lookup_transform(
                BASE_FRAME, TCP_FRAME, rclpy.time.Time(),
                timeout=Duration(seconds=0.5),
            )
            break
        except Exception:
            continue
    if transform is None:
        print(
            "# FAILED: joints not still or tf world->tcp unavailable within "
            f"{TIMEOUT_S:.0f}s (samples={len(node.samples)})",
            file=sys.stderr,
        )
        return 1

    joints = node.samples[-1]
    t = transform.transform.translation
    q = transform.transform.rotation
    print(f'export BATTERY_IMSI_FIRST_POSE_JOINT_NAMES="{",".join(JOINT_NAMES)}"')
    print(
        'export BATTERY_IMSI_FIRST_POSE_JOINTS="'
        + ",".join(f"{value:.6f}" for value in joints) + '"'
    )
    print(
        'export BATTERY_IMSI_FIRST_POSE_TCP="'
        + ",".join(f"{value:.6f}" for value in (t.x, t.y, t.z, q.x, q.y, q.z, q.w))
        + '"'
    )
    node.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
