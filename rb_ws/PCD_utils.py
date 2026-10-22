"""Compatibility wrapper for PCD utility imports.

The implementation is split across focused modules, but this file keeps a
single import surface for capture scripts.
"""

try:
    from rb_ws.Occupancy_utils import update_and_save_object_occupancy_map
    from rb_ws.Octree_utils import (
        create_octree_voxels,
        labels_to_colors,
        save_octree_voxels,
    )
    from rb_ws.Octree_visualization_utils import visualize_octree_voxels_open3d
    from rb_ws.Open3D_utils import visualize_voxels_open3d
    from rb_ws.Pointcloud_processing_utils import (
        depth_to_pointcloud,
        pointcloud_to_voxels,
        save_pointcloud_ply,
    )
    from rb_ws.Zivid_capture_utils import (
        SURFACE_SEMANTIC_LABELS,
        assign_defect_points_to_objects,
        create_pcd_roi_mask,
        find_semantic_id,
        find_semantic_ids,
        flatten_semantic_ids,
        image_values_for_valid_depth,
        sanitize_label_for_filename,
        save_octree_voxels_by_label,
        save_zivid_capture,
    )
except ModuleNotFoundError:
    from Occupancy_utils import update_and_save_object_occupancy_map
    from Octree_utils import (
        create_octree_voxels,
        labels_to_colors,
        save_octree_voxels,
    )
    from Octree_visualization_utils import visualize_octree_voxels_open3d
    from Open3D_utils import visualize_voxels_open3d
    from Pointcloud_processing_utils import (
        depth_to_pointcloud,
        pointcloud_to_voxels,
        save_pointcloud_ply,
    )
    from Zivid_capture_utils import (
        SURFACE_SEMANTIC_LABELS,
        assign_defect_points_to_objects,
        create_pcd_roi_mask,
        find_semantic_id,
        find_semantic_ids,
        flatten_semantic_ids,
        image_values_for_valid_depth,
        sanitize_label_for_filename,
        save_octree_voxels_by_label,
        save_zivid_capture,
    )


__all__ = [
    "SURFACE_SEMANTIC_LABELS",
    "assign_defect_points_to_objects",
    "create_octree_voxels",
    "create_pcd_roi_mask",
    "depth_to_pointcloud",
    "find_semantic_id",
    "find_semantic_ids",
    "flatten_semantic_ids",
    "image_values_for_valid_depth",
    "labels_to_colors",
    "pointcloud_to_voxels",
    "sanitize_label_for_filename",
    "save_octree_voxels",
    "save_octree_voxels_by_label",
    "save_pointcloud_ply",
    "save_zivid_capture",
    "update_and_save_object_occupancy_map",
    "visualize_octree_voxels_open3d",
    "visualize_voxels_open3d",
]
