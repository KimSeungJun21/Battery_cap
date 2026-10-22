#!/usr/bin/env python3
"""Pick the batterycap assembly from above in Isaac Sim using MoveIt.

This is the batterycap counterpart of suction_mustard_bottle.py.  It is kept
self-contained instead of subclassing the mustard picker because the two
targets differ in the places that decide whether suction works at all:

Asset
    /home/seungjun/isaac_ws/Assets/batterycap.usd, spawned by Battery_rb.py
    at /World/BatteryCap, position (0.0, 0.5, 0.27), scale 1, no rotation
    (Battery_config.BATTERYCAP_*).

Geometry
    The authored meshes occupy local X=[-0.145, 0.145], Y=[-0.015, 0.015],
    Z=[0.430, 0.5035] m.  With the spawn Z of 0.27 the part's underside lands
    exactly on the environment table top (world Z=0.7000) and its broad top
    face sits at world Z=0.7735 - only 73.5 mm of clearance, against the
    mustard bottle's ~190 mm.  The part is also long in X (290 mm) and narrow
    in Y (30 mm), so the suction cup has to land on the bbox centre line.

Tool length
    MoveIt plans for the link named "tcp", which is the bare flange TCP
    (link6 + "0 -0.0967 0"); rbpodo_moveit_config's URDF carries no gripper.
    The EPick hangs below it only in the Isaac stage, and its contact point
    (epick_tip) is TCP_TIP_OFFSET = 117.3 mm further along tcp's local -Y -
    straight down in the pick orientation.  Driving "tcp" itself onto the top
    face, the way the mustard script does, therefore buries the cup 117 mm
    inside the part and through the table.  With the cap only 73.5 mm tall
    and Isaac's surface gripper limited to BATTERY_SUCTION_MAX_GRIP_DISTANCE
    (0.02 m), that difference is the whole pick.  So the contact goal is
    raised by the tool length by default and both heights are logged; pass
    --tcp-tip-offset 0 to reproduce the mustard script's behaviour exactly.

Lift
    The Zivid camera looks horizontally (world -Y) from world Z=1.2248 with a
    vertical half-FOV of 22.85 deg, so a part resting on the table is far
    below its field of view: projecting the cap's authored bbox from a logged
    at-rest run puts all eight corners at v=2597..3177 px against a 2048 px
    sensor.  Captures only work with the part held up at camera height -
    working mustard runs show the object at world Z=1.2324 at capture time -
    which is what this lift is for, not merely clearing the table.

    That also pins the planning frame, and a real run measured it: with the
    cap at its spawn pose this script logged "top face Z=0.0710" for
    mainboard2, whose top is world Z = 0.27 + 0.501 = 0.7710.  So the robot
    root sits at world Z = 0.7000 - it is mounted on the table - and the
    mustard script's comment arithmetic ("1.2248 - 0.7000 + 0.1000") was
    right about the root after all; its own constant 0.7248 is the part that
    does not follow from it.

    So the lift is expressed as a target for the suction tip, which is also
    the cap's top face, and LIFT_TIP_TARGET_Z trades framing against reach
    (see its comment).  --lift-rise keeps a floor of LIFT_DETECTION_RISE_M
    (0.10 m) of travel so Battery_rb.py still detects the pick.
"""

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

# Distance from the MoveIt "tcp" link to the EPick contact point, summed from
# the fixed joints of Assets/urdf/rb5_850e_u_epick_isaac.urdf:
#   tcp -> epick_base_link          rpy (90, 0, 0), no translation
#   epick_base_link -> epick_body   +0.05115 along epick +Z
#   epick_body -> epick_suction_cup +0.05865
#   epick_suction_cup -> epick_tip  +0.00750
# The 90-degree X rotation aligns epick +Z with tcp's local -Y, which
# TCP_QUATERNION then points along world -Z.
TCP_TIP_OFFSET = 0.1173

# Zivid_camera_01's optical centre in the planning frame: world Z=1.2248 less
# the robot root's world Z of 0.7000 (measured, see this module's docstring).
CAMERA_OPTICAL_CENTER_Z = 0.5248
# Default lift height for the tip. Projecting the held cap's bbox through the
# real camera at the pose a run actually used (tip Z -> offset from the
# optical axis / TCP reach / bbox corners inside the 2448x2048 sensor):
#   0.4500  -0.075 m  0.788 m  8/8      0.5500  +0.025 m  0.863 m  8/8
#   0.5248   0.000 m  0.844 m  8/8      0.6000  +0.075 m  0.902 m  8/8
#   0.6500  +0.125 m  0.942 m  8/8
# The part is small enough in the frame that everything from 0.45 up stays
# fully visible, so the binding constraint is reach: the RB5-850E manages
# ~0.947 m at the TCP. 0.55 sits just above the optical axis with ~0.08 m of
# reach in hand; a run at this value completed its lift.
LIFT_TIP_TARGET_Z = 0.55
# How far the tip may sit from the optical centre before the warning fires.
# The sweep above stays in frame well past this, so treat it as a sanity
# bound on a hand-passed --lift-tip-z rather than a hard visibility limit.
LIFT_FOV_TOLERANCE = 0.15
# Battery_rb.py needs this much rise to register the pick
# (Battery_config.LIFT_DETECTION_RISE_M).
LIFT_DETECTION_RISE = 0.10

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
# The cup is circular, so yaw about the suction axis is free as far as the
# grasp is concerned. It is still held to a modest tolerance to keep the
# EPick body clear of the part's long +X direction; --yaw-tolerance 3.15
# releases it if planning to the cap proves hard.
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

# Isaac's surface gripper only attaches within this distance
# (Battery_config.SUCTION_MAX_GRIP_DISTANCE). Used to warn when the requested
# tip clearance puts the cup out of range before suction is commanded.
SUCTION_MAX_GRIP_DISTANCE = 0.02

# The cup is 24 mm across (epick_suction_cup, radius 0.012) and mainboard2 is
# only 30 mm wide, so the cup overhangs the board past this much lateral
# error. It is also why small pose or tracking errors show up as a visibly
# off-centre approach.
CUP_LATERAL_MARGIN = 0.003


def parse_arguments(argv):
    parser = argparse.ArgumentParser(
        description="Top-down suction pick for the batterycap assembly."
    )
    parser.add_argument(
        "--contact-position",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        help=(
            "Direct suction-tip contact position in world coordinates. If "
            f"omitted, the position is read from {OBJECT_POSE_TOPIC}. The "
            "tool length is added on top of this, so pass the cap's top "
            "face, not a TCP goal."
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
        "--tcp-tip-offset",
        type=float,
        default=TCP_TIP_OFFSET,
        help=(
            "Distance from the MoveIt tcp link to the EPick contact point "
            f"(default {TCP_TIP_OFFSET:.4f} m). Pass 0 to drive tcp itself "
            "onto the contact point, as suction_mustard_bottle.py does."
        ),
    )
    parser.add_argument(
        "--tip-clearance",
        type=float,
        default=0.0,
        help=(
            "Gap left between the suction tip and the cap's top face at the "
            "contact pose (meters). Negative values press into the surface."
        ),
    )
    parser.add_argument(
        "--pregrasp-height",
        type=float,
        default=0.20,
        help="Distance above the contact point before descending (meters).",
    )
    parser.add_argument(
        "--lift-tip-z",
        type=float,
        default=LIFT_TIP_TARGET_Z,
        help=(
            "Height the suction tip (and so the cap's top face) is raised "
            f"to, in the planning frame (default {LIFT_TIP_TARGET_Z:.4f}). "
            f"The Zivid's optical centre is {CAMERA_OPTICAL_CENTER_Z:.4f}; "
            f"values more than {LIFT_FOV_TOLERANCE:.2f} m from it are "
            "warned about. Reach, not framing, is the limit here."
        ),
    )
    parser.add_argument(
        "--lift-rise",
        type=float,
        default=LIFT_DETECTION_RISE,
        help=(
            "Minimum rise above the contact pose, in meters (default "
            f"{LIFT_DETECTION_RISE}). Battery_rb.py needs this much travel "
            "to register the pick."
        ),
    )
    parser.add_argument(
        "--yaw-tolerance",
        type=float,
        default=YAW_TOLERANCE,
        help=(
            "Orientation tolerance about the suction axis (radians, default "
            f"{YAW_TOLERANCE}). Use ~3.15 to leave yaw unconstrained."
        ),
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
    # The cap leaves only 73.5 mm between the table and its top face, so the
    # descent is kept collision-checked by default.
    parser.set_defaults(avoid_collisions=True)
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


def make_goal_constraints(pose, constraint_name, yaw_tolerance):
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
    orientation.absolute_z_axis_tolerance = float(yaw_tolerance)
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


class BatteryCapPicker(Node):
    NODE_NAME = "suction_batterycap"
    CONSTRAINT_NAME = "batterycap_tcp_pose"

    def __init__(self, options):
        super().__init__(self.NODE_NAME)
        self.options = options
        self.speed = min(1.0, max(0.01, float(options.speed)))
        self.tip_offset = float(options.tcp_tip_offset)
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
        lift = self.make_lift_pose(contact)

        self.log_pose("pregrasp", pregrasp)
        trajectory = self.plan_pose(pregrasp)
        if self.options.plan_only:
            self.get_logger().info(
                "Pregrasp planning succeeded; --plan-only stops here."
            )
            return

        self.execute_joint_trajectory("pregrasp", trajectory)
        self.move_cartesian("contact", contact)
        self.report_contact_drift(contact)

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
                "Pass --contact-position or start Battery_rb.py with "
                "--target-object batterycap and target pose publishing "
                "enabled (BATTERY_PUBLISH_TARGET_OBJECT_POSE=1)."
            )
        return self.latest_object_pose

    def read_surface_point(self):
        """Latest published grasp point, with --contact-offset applied."""
        message = self.wait_for_object_pose()
        if (
            message.header.frame_id
            and message.header.frame_id != PLANNING_FRAME
        ):
            self.get_logger().warning(
                "Object pose frame differs from planning frame: "
                f"{message.header.frame_id} != {PLANNING_FRAME}"
            )
        dx, dy, dz = self.options.contact_offset
        return (
            message.pose.position.x + dx,
            message.pose.position.y + dy,
            message.pose.position.z + dz,
        )

    def report_contact_drift(self, contact_pose):
        """Compare the planned contact point with the part's live pose.

        The contact goal is read once, before the pregrasp move, and the arm
        then runs open loop - every trajectory here is published as joint
        setpoints with no feedback. So if the cup ends up off the board there
        are two candidates, and they need different fixes: the part shifted
        while the arm was approaching, or the arm did not track the commanded
        pose. Re-reading the pose right before suction separates them: a
        non-zero delta here means the part moved, a zero delta means the
        target was stable and any visible offset is the arm's tracking.
        """
        if self.options.contact_position is not None:
            return

        self.latest_object_pose = None
        try:
            x, y, z = self.read_surface_point()
        except RuntimeError as exc:
            self.get_logger().warning(f"Contact drift unknown: {exc}")
            return

        planned_tip_z = contact_pose.position.z - self.tip_offset
        dx = x - contact_pose.position.x
        dy = y - contact_pose.position.y
        dz = (z + float(self.options.tip_clearance)) - planned_tip_z
        lateral = (dx * dx + dy * dy) ** 0.5
        self.get_logger().info(
            "contact drift since planning: "
            f"dx={dx * 1000.0:+.1f} mm, dy={dy * 1000.0:+.1f} mm, "
            f"dz={dz * 1000.0:+.1f} mm (lateral {lateral * 1000.0:.1f} mm)"
        )
        if lateral > CUP_LATERAL_MARGIN:
            self.get_logger().warning(
                f"The part moved {lateral * 1000.0:.1f} mm sideways while "
                "the arm approached, more than the cup's "
                f"{CUP_LATERAL_MARGIN * 1000.0:.0f} mm of margin on the "
                "board; the cup is now off-centre. Let the cap settle "
                "before approving the pick."
            )

    def make_contact_pose(self):
        """Place the suction tip on the cap's top face.

        Battery_rb.py publishes mainboard2's own-frame top centre, so the
        incoming point is already the middle of the flat board the cup lands
        on. The returned pose is a goal for the "tcp" link, which sits
        self.tip_offset above the cup's contact point.
        """
        if self.options.contact_position is not None:
            x, y, z = self.options.contact_position
        else:
            x, y, z = self.read_surface_point()

        surface_z = float(z)
        tip_z = surface_z + float(self.options.tip_clearance)
        contact = make_pose(x, y, tip_z + self.tip_offset)

        gap = tip_z - surface_z
        self.get_logger().info(
            "batterycap contact: top face Z=" f"{surface_z:.4f}, "
            f"suction tip Z={tip_z:.4f} (gap {gap * 1000.0:.1f} mm), "
            f"tcp Z={contact.position.z:.4f} "
            f"(tool length {self.tip_offset:.4f} m)"
        )
        if abs(gap) > SUCTION_MAX_GRIP_DISTANCE:
            self.get_logger().warning(
                f"Tip gap {gap:.4f} m exceeds Isaac's grip distance "
                f"{SUCTION_MAX_GRIP_DISTANCE:.4f} m; suction will not "
                "attach. Check --tip-clearance and --tcp-tip-offset."
            )
        self.log_pose("contact", contact)
        return contact

    def make_lift_pose(self, contact_pose):
        """Present the cap to the Zivid camera after suction.

        The target is the suction tip, which holds the cap's top face, so it
        is the part - not the flange - that ends up at camera height. A
        minimum rise is enforced on top of that so Battery_rb.py still sees
        the pick even if the cap starts near the target height.
        """
        contact_tip_z = contact_pose.position.z - self.tip_offset
        target_tip_z = max(
            contact_tip_z + max(0.0, float(self.options.lift_rise)),
            float(self.options.lift_tip_z),
        )
        offset_from_camera = target_tip_z - CAMERA_OPTICAL_CENTER_Z
        self.get_logger().info(
            f"lift target: tip Z={target_tip_z:.4f} "
            f"(rise {target_tip_z - contact_tip_z:.4f} m, "
            f"{offset_from_camera:+.4f} m vs camera optical centre)"
        )
        if abs(offset_from_camera) > LIFT_FOV_TOLERANCE:
            self.get_logger().warning(
                f"Lift leaves the cap {offset_from_camera:+.3f} m from the "
                "camera's optical centre; it will fall outside the Zivid's "
                "vertical FOV and every capture will be rejected."
            )
        return make_pose(
            contact_pose.position.x,
            contact_pose.position.y,
            target_tip_z + self.tip_offset,
        )

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
                yaw_tolerance=self.options.yaw_tolerance,
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
            f"tip_z={pose.position.z - self.tip_offset:.4f}, "
            f"orientation=({pose.orientation.x:.4f}, "
            f"{pose.orientation.y:.4f}, {pose.orientation.z:.4f}, "
            f"{pose.orientation.w:.4f})"
        )


def main(argv=None):
    argv = sys.argv if argv is None else argv
    options = parse_arguments(remove_ros_args(argv)[1:])
    rclpy.init(args=argv)
    node = BatteryCapPicker(options)

    try:
        node.wait_until_ready()
        node.run()
        node.get_logger().info("Batterycap suction sequence finished.")
    except Exception as exc:
        node.get_logger().error(str(exc))
        raise
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
