"""ROS 2 publisher for target object poses in Isaac Sim."""

import time

import omni.usd
import rclpy
from geometry_msgs.msg import PoseStamped
from pxr import Gf, Usd, UsdGeom


def create_pose_stamped_from_prim(
    node,
    prim,
    frame_id,
    reference_prim=None,
    pose_mode="bbox_top_center",
    grasp_z_offset=0.0,
):
    world_matrix = omni.usd.get_world_transform_matrix(
        prim
    )

    if pose_mode == "local_bbox_top_center":
        translation, orientation_xyzw = compute_local_bbox_top_center(
            prim=prim,
            reference_prim=reference_prim,
            grasp_z_offset=grasp_z_offset,
        )
    elif pose_mode == "bbox_top_center":
        translation = compute_bbox_top_center(
            prim=prim,
            reference_prim=reference_prim,
            grasp_z_offset=grasp_z_offset,
        )
        orientation_xyzw = (
            0.0,
            0.0,
            0.0,
            1.0,
        )
    else:
        rotation = world_matrix.RemoveScaleShear().ExtractRotationQuat()
        if reference_prim is not None:
            reference_world_matrix = omni.usd.get_world_transform_matrix(
                reference_prim
            )
            pose_matrix = (
                world_matrix
                * reference_world_matrix.GetInverse()
            )
        else:
            pose_matrix = world_matrix

        translation = pose_matrix.ExtractTranslation()
        imaginary = rotation.GetImaginary()
        orientation_xyzw = (
            float(imaginary[0]),
            float(imaginary[1]),
            float(imaginary[2]),
            float(rotation.GetReal()),
        )

    message = PoseStamped()
    message.header.stamp = node.get_clock().now().to_msg()
    message.header.frame_id = frame_id
    message.pose.position.x = float(
        translation[0]
    )
    message.pose.position.y = float(
        translation[1]
    )
    message.pose.position.z = float(
        translation[2]
    )
    message.pose.orientation.x = float(
        orientation_xyzw[0]
    )
    message.pose.orientation.y = float(
        orientation_xyzw[1]
    )
    message.pose.orientation.z = float(
        orientation_xyzw[2]
    )
    message.pose.orientation.w = float(
        orientation_xyzw[3]
    )
    return message


def compute_bbox_top_center(
    prim,
    reference_prim=None,
    grasp_z_offset=0.0,
):
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        includedPurposes=[
            UsdGeom.Tokens.default_,
        ],
        useExtentsHint=True,
    )
    bbox_range = (
        bbox_cache
        .ComputeWorldBound(prim)
        .ComputeAlignedRange()
    )
    min_point = bbox_range.GetMin()
    max_point = bbox_range.GetMax()
    world_point = Gf.Vec3d(
        (float(min_point[0]) + float(max_point[0])) * 0.5,
        (float(min_point[1]) + float(max_point[1])) * 0.5,
        float(max_point[2]) + float(grasp_z_offset),
    )

    if reference_prim is None:
        return world_point

    reference_world_matrix = omni.usd.get_world_transform_matrix(
        reference_prim
    )
    return reference_world_matrix.GetInverse().Transform(
        world_point
    )


def compute_local_bbox_top_center(
    prim,
    reference_prim=None,
    grasp_z_offset=0.0,
):
    """Centre of the prim's top face, in the prim's OWN frame.

    compute_bbox_top_center() takes the centre of the world axis-aligned box
    and its global maximum Z. Those belong to different points as soon as the
    part is tilted: for the batterycap's 290 mm board a 7 degree tilt lifts
    max Z about 0.145 * sin(7 deg) = 18 mm above the surface at the centre,
    which is most of Isaac's 20 mm grip distance. So take the box in the
    prim's own space instead, where the top face is flat, and carry the point
    (and the surface's orientation) out to world through the box's matrix.

    grasp_z_offset is applied along the part's local +Z, i.e. along the
    surface normal rather than world up.

    Returns (translation, orientation_xyzw).
    """
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        includedPurposes=[
            UsdGeom.Tokens.default_,
        ],
        useExtentsHint=True,
    )
    bbox = bbox_cache.ComputeWorldBound(prim)
    box_range = bbox.GetRange()
    box_matrix = bbox.GetMatrix()

    min_point = box_range.GetMin()
    max_point = box_range.GetMax()
    local_point = Gf.Vec3d(
        (float(min_point[0]) + float(max_point[0])) * 0.5,
        (float(min_point[1]) + float(max_point[1])) * 0.5,
        float(max_point[2]) + float(grasp_z_offset),
    )

    pose_matrix = box_matrix
    if reference_prim is not None:
        reference_world_matrix = omni.usd.get_world_transform_matrix(
            reference_prim
        )
        pose_matrix = box_matrix * reference_world_matrix.GetInverse()

    translation = pose_matrix.Transform(local_point)
    rotation = pose_matrix.RemoveScaleShear().ExtractRotationQuat()
    imaginary = rotation.GetImaginary()
    orientation_xyzw = (
        float(imaginary[0]),
        float(imaginary[1]),
        float(imaginary[2]),
        float(rotation.GetReal()),
    )
    return translation, orientation_xyzw


class TargetObjectPosePublisher:
    def __init__(
        self,
        target_prim,
        reference_prim=None,
        topic_name="/target_object/pose",
        frame_id="world",
        publish_hz=20.0,
        pose_mode="bbox_top_center",
        grasp_z_offset=0.0,
        log_prefix="[OBJECT POSE]",
    ):
        if not rclpy.ok():
            rclpy.init(
                args=None
            )

        self.target_prim = target_prim
        self.reference_prim = reference_prim
        self.topic_name = topic_name
        self.frame_id = frame_id
        self.pose_mode = pose_mode
        self.grasp_z_offset = float(
            grasp_z_offset
        )
        self.publish_period = 1.0 / max(
            1.0,
            float(publish_hz),
        )
        self.log_prefix = log_prefix
        self.last_publish_time = 0.0
        self.node = rclpy.create_node(
            "battery_target_object_pose_publisher"
        )
        self.publisher = self.node.create_publisher(
            PoseStamped,
            self.topic_name,
            10,
        )

        print(
            self.log_prefix,
            "Publishing target object pose:",
            self.topic_name,
            "frame_id=",
            self.frame_id,
            "publish_hz=",
            float(publish_hz),
            "pose_mode=",
            self.pose_mode,
            "grasp_z_offset=",
            self.grasp_z_offset,
            "reference_prim=",
            (
                str(reference_prim.GetPath())
                if reference_prim is not None
                else "<none>"
            ),
        )

    def publish_if_due(
        self,
        force=False,
    ):
        now = time.monotonic()

        if (
            not force
            and now - self.last_publish_time
            < self.publish_period
        ):
            return

        message = create_pose_stamped_from_prim(
            node=self.node,
            prim=self.target_prim,
            frame_id=self.frame_id,
            reference_prim=self.reference_prim,
            pose_mode=self.pose_mode,
            grasp_z_offset=self.grasp_z_offset,
        )
        self.publisher.publish(
            message
        )
        rclpy.spin_once(
            self.node,
            timeout_sec=0.0,
        )
        self.last_publish_time = now

    def destroy(
        self,
    ):
        if self.node is not None:
            self.node.destroy_node()
            self.node = None

        if rclpy.ok():
            rclpy.shutdown()
