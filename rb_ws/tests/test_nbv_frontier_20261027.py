"""Battery_frontier_map_20261027: carving, frontier and visibility."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Battery_frontier_map_20261027 import (  # noqa: E402
    FREE, OCCUPIED, UNKNOWN, FrontierMap, frontier_candidate_results, frontier_route_problem,
)

K = np.array([[100.0, 0, 50], [0, 100.0, 50], [0, 0, 1]])
IMAGE = (100, 100)


def camera_looking_down(height):
    """USD camera (looks along -Z) above the box, in the object frame."""
    T = np.eye(4)
    T[2, 3] = height
    return T


class FrontierMapTests(unittest.TestCase):
    def make(self):
        return FrontierMap([0, 0, 0], [0.1, 0.1, 0.1], leaf_size=0.01, pixel_stride=1)

    def test_part_ray_carves_path_and_marks_part_endpoint(self):
        m = self.make()
        m.integrate_rays([0.055, 0.055, 0.5], [[0.055, 0.055, 0.055]], [True])
        state = m.state()
        self.assertEqual(state[5, 5, 5], OCCUPIED)
        self.assertTrue(np.all(state[5, 5, 6:] == FREE))
        self.assertTrue(np.all(state[5, 5, :5] == UNKNOWN))   # behind the surface
        self.assertEqual(m.part_occupied().sum(), 1)

    def test_background_ray_carves_whole_crossing_without_hits(self):
        m = self.make()
        # endpoint far below the box: every crossed voxel is FREE, none OCCUPIED
        m.integrate_rays([0.025, 0.025, 0.5], [[0.025, 0.025, -1.0]], [False])
        state = m.state()
        self.assertTrue(np.all(state[2, 2, :] == FREE))
        self.assertEqual(int((state == OCCUPIED).sum()), 0)
        self.assertEqual(m.stats()["frontier"], 0)  # no part, no frontier

    def test_non_part_hit_blocks_but_seeds_no_frontier(self):
        m = self.make()
        m.integrate_rays([0.055, 0.055, 0.5], [[0.055, 0.055, 0.075]], [False])  # gripper
        self.assertEqual(m.state()[5, 5, 7], OCCUPIED)
        self.assertEqual(m.part_occupied().sum(), 0)
        self.assertEqual(m.stats()["frontier"], 0)

    def test_frontier_shrinks_when_neighbours_are_observed(self):
        m = self.make()
        m.integrate_rays([0.055, 0.055, 0.5], [[0.055, 0.055, 0.055]], [True])
        before = m.stats()["frontier"]
        self.assertEqual(before, 26 - 1)  # 3x3x3 around the part minus the carved voxel above
        # carve the column beside the part with a background ray
        m.integrate_rays([0.045, 0.055, 0.5], [[0.045, 0.055, -1.0]], [False])
        # the column (4,5,z) holds three frontier voxels next to the part (z=4,5,6)
        self.assertEqual(m.stats()["frontier"], before - 3)
        self.assertFalse(m.frontier_mask()[4, 5, 4:7].any())

    def test_visibility_respects_view_and_occlusion(self):
        m = self.make()
        m.integrate_rays([0.055, 0.055, 0.5], [[0.055, 0.055, 0.055]], [True])
        frontier = m.frontier_indices()
        visible = set(m.visible_frontier(frontier, camera_looking_down(0.5), K, IMAGE).tolist())
        below = np.ravel_multi_index((5, 5, 4), m.shape)
        beside = np.ravel_multi_index((6, 5, 5), m.shape)
        self.assertNotIn(below, visible)   # behind the part voxel
        self.assertIn(beside, visible)
        # a camera looking the other way sees nothing
        away = camera_looking_down(0.5)
        away[:3, :3] = np.diag([1.0, -1.0, -1.0])  # now looks along +Z (up)
        self.assertEqual(len(m.visible_frontier(frontier, away, K, IMAGE)), 0)

    def test_route_problem_bits_match_visible_sets(self):
        frontier = np.array([10, 20, 30, 40])
        problem = frontier_route_problem(frontier, {"a": np.array([10, 30]), "b": np.array([30, 40]), "c": np.array([], int)})
        self.assertEqual(problem["entropy_bits_by_candidate"], {"a": 2, "b": 2, "c": 0})
        self.assertEqual(problem["visibility_bits"]["a"], 0b0101)
        self.assertEqual(problem["visibility_bits"]["b"], 0b1100)
        self.assertEqual(problem["reachable_entropy_bits"], 3)
        self.assertEqual(problem["voxel_count"], 4)
        results = {r["candidate_name"]: r for r in frontier_candidate_results({"a": np.array([10, 30])})}
        self.assertEqual(results["a"]["entropy_score"], 2.0)
        self.assertEqual(results["a"]["score_source"], "frontier_voxels")

    def test_depth_integration_uses_cv_camera_convention(self):
        m = self.make()
        depth = np.full((10, 10), np.nan)
        depth[5, 5] = 0.445  # pixel (u=5, v=5) -> camera ray through the principal point
        Kd = np.array([[10.0, 0, 5], [0, 10.0, 5], [0, 0, 1]])
        T = np.eye(4)
        T[:3, :3] = np.diag([1.0, -1.0, -1.0])  # CV camera looking down -Z of the object
        T[:3, 3] = [0.055, 0.055, 0.5]
        m.integrate_depth(depth, Kd, T, np.ones((10, 10), bool))
        self.assertEqual(m.state()[5, 5, 5], OCCUPIED)
        self.assertEqual(m.part_occupied().sum(), 1)

    def test_underside_frontier_is_not_counted(self):
        full = self.make()
        cut = FrontierMap([0, 0, 0], [0.1, 0.1, 0.1], leaf_size=0.01, pixel_stride=1, exclude_down_cos=0.5)
        for m in (full, cut):
            m.integrate_rays([0.055, 0.055, 0.5], [[0.055, 0.055, 0.055]], [True])
        below = cut.frontier_mask(include_down=True) & ~cut.frontier_mask()
        # the 3x3 layer under the part: straight down and 45 deg down-diagonals
        self.assertEqual(int(below.sum()), 9)
        self.assertTrue(np.all(below[4:7, 4:7, 4]))
        self.assertEqual(cut.stats()["frontier"], full.stats()["frontier"] - 9)
        self.assertEqual(cut.stats()["frontier_down_excluded"], 9)
        # beside and above the part is still frontier
        self.assertTrue(cut.frontier_mask()[6, 5, 5])
        self.assertTrue(cut.frontier_mask()[4, 4, 6])


if __name__ == "__main__":
    unittest.main()
