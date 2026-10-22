"""Quality coverage / close-up pose helpers, without Isaac or ROS."""

import ast
import math
from pathlib import Path
import unittest

import numpy as np


SOURCE = Path(__file__).resolve().parents[1] / "Battery_nbv_arm_runner_utils.py"
TREE = ast.parse(SOURCE.read_text(encoding="utf-8"))


def euler_matrix(rotation_xyz_deg):
    # Same convention as NBV_utils.make_object_world_matrix (Rz @ Ry @ Rx).
    rx, ry, rz = np.deg2rad(rotation_xyz_deg)
    x = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    y = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    z = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    matrix = np.eye(4)
    matrix[:3, :3] = z @ y @ x
    return matrix


def load_functions():
    names = {
        "quality_face_group_masks", "surfel_pixel_density",
        "update_quality_observations", "quality_coverage_fractions",
        "predicted_face_density", "rank_quality_candidates",
        "proximity_gain_weights", "rotation_toward",
        "rotation_xyz_deg_from_matrix", "solve_tcp_pose_for_candidate",
        "rotation_about_axis", "frontal_top_delta_rotation", "face_deficit_candidates",
        "predict_object_matrix",
    }
    body = [n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name in names]
    body += [n for n in TREE.body if isinstance(n, ast.Assign)
             and any(getattr(t, "id", None) == "QUALITY_FACE_NORMALS" for t in n.targets)]
    namespace = {
        "math": math, "np": np,
        "make_object_world_matrix": lambda position, rotation_xyz_deg: euler_matrix(rotation_xyz_deg),
    }
    exec(compile(ast.Module(body=body, type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace


class QualityCoverageTests(unittest.TestCase):
    def setUp(self):
        self.f = load_functions()

    def test_side_split_by_dominant_normal_axis(self):
        normals = np.array([[0, 0, 1], [1, 0.2, 0], [-1, 0, 0.1], [0.3, 1, 0], [0, -1, 0], [0, 0, -1]], float)
        masks = self.f["quality_face_group_masks"](normals, ["+Z", "+X", "-X", "+Y", "-Y"])
        self.assertEqual({k: list(np.flatnonzero(v)) for k, v in masks.items()},
                         {"+Z": [0], "+X": [1], "-X": [2], "+Y": [3], "-Y": [4]})

    def test_density_matches_recall_basis(self):
        density = self.f["surfel_pixel_density"]
        points = np.zeros((3, 3))
        normals = np.array([[0, 0, 1], [0, 0, 1], [0, 0, -1]], float)
        sensor = np.array([0, 0, 0.4])
        self.assertTrue(np.allclose(density(points, normals, sensor, 0.4), [1, 1, 0]))
        oblique = np.array([0.5 * math.sin(math.radians(60)), 0, 0.5 * math.cos(math.radians(60))])
        # 60 deg at 0.5 m: 0.64 * 0.5 = 0.32 < 0.5 -> not a quality view.
        self.assertAlmostEqual(density(points[:1], normals[:1], oblique, 0.4)[0], 0.32)

    def test_quality_observations_accumulate_only_dense_visible_surfels(self):
        model = {"points": np.zeros((3, 3)), "normals": np.array([[0, 0, 1]] * 3, float)}
        visible = np.array([True, False, True])
        near, far = np.array([0, 0, 0.4]), np.array([0, 0, 0.7])
        self.assertEqual(self.f["update_quality_observations"](model, visible, far, 0.4, 0.5), 0)
        self.assertEqual(self.f["update_quality_observations"](model, visible, near, 0.4, 0.5), 2)
        self.assertEqual(self.f["update_quality_observations"](model, visible, near, 0.4, 0.5), 0)
        masks = {"+Z": np.array([True, True, True])}
        self.assertAlmostEqual(self.f["quality_coverage_fractions"](model, masks)["+Z"], 2 / 3)

    def test_rank_prefers_serving_more_deficit_per_second_and_reports_unreachable(self):
        rank = self.f["rank_quality_candidates"]
        dirs = {"end": [1, 0, 0.2], "top": [0, 0.3, 1], "both": [0.7, 0, 0.7], "far_end": [1, 0, 0]}
        dist = {"end": 0.42, "top": 0.42, "both": 0.40, "far_end": 0.60}
        cost = {"end": 1.0, "top": 1.0, "both": 1.2, "far_end": 0.5}
        ranked, unreachable = rank(list(dirs), dirs, dist, cost,
                                   {"+X": 0.4, "+Z": 0.1, "-Y": 0.3}, 0.4, 0.5)
        self.assertEqual([name for name, _, _ in ranked], ["both", "end", "top"])
        self.assertEqual(ranked[0][1], ["+X", "+Z"])
        self.assertEqual(unreachable, ["-Y"])  # far_end at 0.6 m fails density

    def test_proximity_weight_is_clipped(self):
        weights = self.f["proximity_gain_weights"]({"near": 0.42, "lift": 0.496, "closer": 0.30}, 0.496, 1.5)
        self.assertAlmostEqual(weights["near"], (0.496 / 0.42) ** 2)
        self.assertEqual(weights["lift"], 1.0)
        self.assertEqual(weights["closer"], 1.5)

    def test_rotation_toward_moves_normal_by_tilt_and_never_overshoots(self):
        rotate = self.f["rotation_toward"]
        normal, camera = np.array([0, 0, 1.0]), np.array([0, 0.8, 0.6])
        target_angle = math.degrees(math.acos(0.6))
        for tilt in (15, 25):
            moved = rotate(normal, camera, tilt) @ normal
            self.assertAlmostEqual(math.degrees(math.acos(moved @ normal)), tilt)
            self.assertAlmostEqual(math.degrees(math.acos(moved @ camera)), target_angle - tilt)
        self.assertTrue(np.allclose(rotate(normal, camera, 90) @ normal, camera))

    def test_euler_record_round_trip(self):
        to_euler = self.f["rotation_xyz_deg_from_matrix"]
        for angles in [(20, 0, 0), (0, -30, -45), (10, 20, 30)]:
            self.assertTrue(np.allclose(to_euler(euler_matrix(angles)[:3, :3]), angles, atol=1e-6))

    def test_pivot_override_translates_geometry_centre_only(self):
        solve = self.f["solve_tcp_pose_for_candidate"]
        initial = np.eye(4); initial[:3, 3] = [0.0, 0.5, 0.7]
        grasp = np.eye(4); grasp[:3, 3] = [0, 0, -0.15]
        pivot_local = np.array([0, 0, 0.46675])
        args = dict(initial_object_rotation_base_np=np.eye(3), grasp_offset_np=grasp,
                    reference_tcp_position=None, rotation_xyz_deg=(20, 0, 0),
                    initial_object_base_matrix=initial, object_pivot_local_point=pivot_local)
        _, base_object = solve(**args)
        target = np.array([0.0, 0.6, 1.25])
        _, moved_object = solve(**args, pivot_base_override=target)
        self.assertTrue(np.allclose(moved_object[:3, :3], base_object[:3, :3]))
        self.assertTrue(np.allclose(moved_object[:3, :3] @ pivot_local + moved_object[:3, 3], target))
        rotation = euler_matrix((20, 0, 0))[:3, :3]
        _, explicit = solve(**{**args, "rotation_xyz_deg": None}, delta_rotation_np=rotation)
        self.assertTrue(np.allclose(explicit, base_object))

    def test_frontal_top_faces_camera_for_every_roll(self):
        frontal = self.f["frontal_top_delta_rotation"]
        forward = np.array([0.0, -1.0, -1.0]) / math.sqrt(2)
        top = np.array([0.0, 0.0, 1.0])
        long_axes = []
        for roll in (0, 90, 180, -90):
            rotation = frontal(top, forward, roll)
            self.assertTrue(np.allclose(rotation @ top, -forward))
            self.assertAlmostEqual(np.linalg.det(rotation), 1.0)
            long_axes.append(rotation @ np.array([1.0, 0.0, 0.0]))
        self.assertAlmostEqual(float(long_axes[0] @ long_axes[1]), 0.0)
        self.assertAlmostEqual(float(long_axes[0] @ long_axes[2]), -1.0)
        # Anti-parallel start still ends frontal.
        self.assertTrue(np.allclose(frontal(forward, forward, 0) @ forward, -forward))

    def test_face_deficit_targets_the_unseen_face_only_with_facing_views(self):
        pick = self.f["face_deficit_candidates"]
        masks = {"+Y": np.array([1, 1, 0, 0, 0, 0], bool), "-Y": np.array([0, 0, 1, 1, 1, 1], bool)}
        observed = np.array([1, 1, 0, 0, 0, 1], bool)  # -Y 25% seen
        dirs = {"diag": [0.7, 0.7, 0.1], "back": [0.1, -0.95, -0.1], "oblique": [0.9, -0.4, 0.0]}
        face, names, fraction = pick(observed, masks, dirs, 0.5, 60.0)
        self.assertEqual((face, names, fraction), ("-Y", ["back"], 0.25))
        self.assertIsNone(pick(observed, masks, dirs, 0.2, 60.0))       # nothing below 20%
        self.assertIsNone(pick(observed, masks, dirs, 0.5, 60.0, {"-Y"}))  # already tried


if __name__ == "__main__":
    unittest.main()
