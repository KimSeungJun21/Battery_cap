"""Cached geometry must preserve fresh occupancy/occlusion scoring exactly."""

import unittest

import numpy as np

from rb_ws.NBV_utils import RayGeometryCache, score_candidate_pose


class RayCacheTests(unittest.TestCase):
    def setUp(self):
        shape = (8, 8, 8)
        self.occupancy = {
            "state": np.full(shape, -1, dtype=np.int8),
            "probability": np.full(shape, 0.5),
            "origin": np.array([-1., -1., -3.]),
            "root_size": 2., "leaf_size": .25, "grid_shape": shape,
        }
        self.kwargs = dict(
            occupancy_map=self.occupancy,
            candidate_object_world_matrix=np.eye(4),
            camera_world_matrix=np.eye(4),
            intrinsic_matrix=np.array([[16., 0, 8.], [0, 16., 8.], [0, 0, 1.]]),
            image_size=(16, 16), pixel_stride=1,
            score_unknown_only=True, return_visible_indices=True,
        )

    def compare(self, cache):
        fresh = score_candidate_pose(**self.kwargs)
        cached = score_candidate_pose(**self.kwargs, ray_geometry_cache=cache)
        for key in fresh:
            np.testing.assert_equal(cached[key], fresh[key], err_msg=key)
        return cached

    def test_map_changes_update_scores_and_occlusion_on_cache_hit(self):
        cache = RayGeometryCache()
        before = self.compare(cache)
        # A new occupied slab blocks the far UNKNOWN voxels on the same rays.
        self.occupancy["state"][:, :, 5] = 1
        self.occupancy["probability"][:, :, 5] = .9
        after = self.compare(cache)
        self.assertLess(after["entropy_score"], before["entropy_score"])
        self.assertGreater(after["occupied_stopped_ray_count"], 0)
        self.assertEqual((cache.hits, cache.misses), (1, 1))
        self.kwargs["roi_min"] = [-.25, -.25, -2.]
        self.kwargs["roi_max"] = [.25, .25, -1.]
        self.compare(cache)  # ROI is applied fresh too, not cached.
        self.assertEqual(cache.hits, 2)

    def test_geometry_changes_invalidate_cache(self):
        cache = RayGeometryCache()
        self.compare(cache)
        self.kwargs["camera_world_matrix"][0, 3] = .1
        self.compare(cache)
        self.kwargs["candidate_object_world_matrix"][1, 3] = .1
        self.compare(cache)
        self.kwargs["intrinsic_matrix"][0, 0] = 20.
        self.compare(cache)
        self.kwargs["pixel_stride"] = 2
        self.compare(cache)
        self.kwargs["image_size"] = (12, 12)
        self.compare(cache)
        self.occupancy["origin"][2] = -2.5
        self.compare(cache)
        self.assertEqual((cache.hits, cache.misses), (0, 7))

    def test_memory_bound_and_eviction(self):
        cache = RayGeometryCache(max_bytes=12)
        cache.put("a", (np.zeros(8, dtype=np.uint8),))
        cache.put("b", (np.zeros(8, dtype=np.uint8),))
        self.assertIsNone(cache.get("a"))
        self.assertIsNotNone(cache.get("b"))
        cache.put("large", (np.zeros(13, dtype=np.uint8),))
        self.assertIsNone(cache.get("large"))
        self.assertEqual(cache.bytes_used, 8)


if __name__ == "__main__":
    unittest.main()
