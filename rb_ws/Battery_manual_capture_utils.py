"""On-demand Zivid capture, decoupled from the pick/NBV-rotate sequence.

Battery_rb.py's automatic NBV sweep only fires after a suction pick is
detected. This gives a way to capture whatever pose the object is
*currently* in - resting, mid-air, gripped, doesn't matter - without going
through pick/lift/rotate at all. Saves under zivid_capture/manual/ so it
never collides with or perturbs the numbered NBV sequence frames.
"""

import os

import numpy as np
import rclpy
from std_msgs.msg import Bool

try:
    from rb_ws.Battery_zivid_setup_utils import capture_current_zivid_frame
except ModuleNotFoundError:
    from Battery_zivid_setup_utils import capture_current_zivid_frame


# Same USD-camera (-Z forward) -> CV-camera (+Z forward) convention used by
# Zivid_capture_utils.save_zivid_capture, duplicated here so a manual capture
# can save its own object_from_camera matrix without touching shared code.
CAMERA_CV_TO_USD = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


class ManualCaptureController:
    def __init__(
        self,
        camera_prim,
        object_prim,
        zivid_render_product,
        zivid_rgb_annotator,
        zivid_depth_annotator,
        zivid_semantic_annotator,
        zivid_instance_annotator,
        zivid_bbox_annotator,
        orchestrator,
        zivid_save_dir,
        intrinsic_matrix,
        voxel_size,
        object_grid_root_size,
        object_grid_origin,
        topic_name="/zivid/manual_capture",
        log_prefix="[MANUAL-CAPTURE]",
    ):
        if not rclpy.ok():
            rclpy.init(args=None)

        self.camera_prim = camera_prim
        self.object_prim = object_prim
        self.zivid_render_product = zivid_render_product
        self.zivid_rgb_annotator = zivid_rgb_annotator
        self.zivid_depth_annotator = zivid_depth_annotator
        self.zivid_semantic_annotator = zivid_semantic_annotator
        self.zivid_instance_annotator = zivid_instance_annotator
        self.zivid_bbox_annotator = zivid_bbox_annotator
        self.orchestrator = orchestrator
        self.zivid_save_dir = os.path.join(zivid_save_dir, "manual")
        self.intrinsic_matrix = intrinsic_matrix
        self.voxel_size = voxel_size
        self.object_grid_root_size = object_grid_root_size
        self.object_grid_origin = object_grid_origin
        self.log_prefix = log_prefix
        self.next_frame_id = 0

        os.makedirs(self.zivid_save_dir, exist_ok=True)

        self.node = rclpy.create_node(
            "battery_manual_capture_controller"
        )
        self.subscription = self.node.create_subscription(
            Bool,
            topic_name,
            self.handle_command,
            10,
        )

        print(
            self.log_prefix,
            "Manual capture controller listening on:",
            topic_name,
            "- saving under:",
            self.zivid_save_dir,
        )

    def handle_command(self, message):
        if not message.data:
            return

        frame_id = self.next_frame_id
        self.next_frame_id += 1

        print(self.log_prefix, "Capturing manual frame:", frame_id)

        camera_world_matrix, object_world_matrix = capture_current_zivid_frame(
            frame_id=frame_id,
            pose_name="manual",
            camera_prim=self.camera_prim,
            object_prim=self.object_prim,
            zivid_render_product=self.zivid_render_product,
            zivid_rgb_annotator=self.zivid_rgb_annotator,
            zivid_depth_annotator=self.zivid_depth_annotator,
            zivid_semantic_annotator=self.zivid_semantic_annotator,
            zivid_instance_annotator=self.zivid_instance_annotator,
            zivid_bbox_annotator=self.zivid_bbox_annotator,
            orchestrator=self.orchestrator,
            zivid_save_dir=self.zivid_save_dir,
            intrinsic_matrix=self.intrinsic_matrix,
            pause_timeline=True,
            voxel_size=self.voxel_size,
            object_grid_root_size=self.object_grid_root_size,
            object_grid_origin=self.object_grid_origin,
        )

        object_from_camera = (
            np.linalg.inv(
                np.asarray(object_world_matrix, dtype=np.float64)
            )
            @ np.asarray(camera_world_matrix, dtype=np.float64)
            @ CAMERA_CV_TO_USD
        )
        pose_path = os.path.join(
            self.zivid_save_dir,
            f"object_from_camera_{frame_id:06d}.npy",
        )
        np.save(pose_path, object_from_camera)

        print(
            self.log_prefix,
            "Saved manual capture",
            frame_id,
            "->",
            self.zivid_save_dir,
        )

    def spin_once(self):
        rclpy.spin_once(
            self.node,
            timeout_sec=0.0,
        )

    def destroy(self):
        if self.node is not None:
            self.node.destroy_node()
            self.node = None
