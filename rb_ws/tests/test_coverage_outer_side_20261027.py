"""Coverage counts what a camera can inspect.

Side: the legs' outer faces only (the legs' inner faces, every +/-Y face -
thin lines on this cap - and other strips narrower than 8 mm are excluded).
Top: CAD layers buried under another top surface are excluded.
"""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Battery_coverage_metrics import (  # noqa: E402
    coverage_exclusion_masks, directional_surface_coverage, inner_side_mask, sample_directional_surfels,
)

# Battery cap shape: top plate on two end legs (a "⊓" seen from +Y).
PLATE = ([-0.145, -0.015, 0.499], [0.145, 0.015, 0.5035])
LEGS = (([0.142, -0.010, 0.430], [0.145, 0.010, 0.499]),
        ([-0.145, -0.010, 0.430], [-0.142, 0.010, 0.499]))
# end pad laid on the plate: the plate top under it stays in the mesh (buried)
PAD = ([0.115, -0.008, 0.5035], [0.140, 0.008, 0.5060])


def box_mesh(lo, hi):
    (x0, y0, z0), (x1, y1, z1) = lo, hi
    v = np.array([[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)])
    # outward-wound quads: -X, +X, -Y, +Y, -Z, +Z
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    tris = [t for a, b, c, d in quads for t in ((a, b, c), (a, c, d))]
    return v, np.array(tris)


def cap_surfels(samples=40000):  # ~0.6 mm2 per surfel, as the 50k CAD model
    vertices, triangles = [], []
    for lo, hi in (PLATE,) + LEGS + (PAD,):
        v, t = box_mesh(lo, hi)
        triangles.append(t + sum(len(x) for x in vertices))
        vertices.append(v)
    return sample_directional_surfels(np.vstack(vertices), np.vstack(triangles), samples=samples, seed=0)


class OuterSideTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = cap_surfels()
        cls.points = cls.model["points"].astype(float)
        cls.normals = cls.model["normals"].astype(float)
        cls.inner = inner_side_mask(cls.points, cls.normals)
        cls.masks = coverage_exclusion_masks(cls.points, cls.normals)

    def test_only_leg_faces_facing_each_other_are_inner(self):
        p, n = self.points, self.normals
        leg_inner = (np.abs(n[:, 0]) > 0.9) & (n[:, 0] * p[:, 0] < 0) \
            & (np.abs(np.abs(p[:, 0]) - 0.142) < 1e-6) & (p[:, 2] < 0.499)
        self.assertGreater(leg_inner.sum(), 100)
        self.assertEqual(int((self.inner & ~leg_inner).sum()), 0)  # nothing outer is dropped
        self.assertGreater((self.inner & leg_inner).sum() / leg_inner.sum(), 0.95)

    def test_leg_outer_faces_and_plate_edges_stay_outer(self):
        p, n = self.points, self.normals
        end_faces = (np.abs(n[:, 0]) > 0.9) & (np.abs(np.abs(p[:, 0]) - 0.145) < 1e-6)
        leg_y_faces = (np.abs(n[:, 1]) > 0.9) & (p[:, 2] < 0.499)
        plate_edges = (np.abs(n[:, 1]) > 0.9) & (p[:, 2] > 0.499)
        for mask in (end_faces, leg_y_faces, plate_edges):
            self.assertGreater(mask.sum(), 10)
            self.assertFalse(self.inner[mask].any())

    def test_side_keeps_only_leg_end_faces(self):
        p, n = self.points, self.normals
        side = np.abs(n[:, 2]) < 0.5
        kept = side & ~self.masks["excluded"]
        on_end_plane = (np.abs(np.abs(p[:, 0]) - 0.145) < 1e-6) & (n[:, 0] * p[:, 0] > 0)
        leg_end = on_end_plane & (np.abs(p[:, 1]) < 0.0095) & (p[:, 2] < 0.498)
        # nothing but end faces is left, bar a few junction samples the rays miss
        self.assertGreater(on_end_plane[kept].mean(), 0.995)
        self.assertGreater((kept & leg_end).sum() / leg_end.sum(), 0.95)

    def test_y_faces_and_plate_end_wings_are_excluded(self):
        p, n = self.points, self.normals
        y_faces = (np.abs(n[:, 2]) < 0.5) & (np.abs(n[:, 1]) > 0.9)
        self.assertTrue(np.all(self.masks["y_facing"][y_faces]))
        # plate end wings beside the legs, 4.5 mm tall (the 1.5 mm next to the
        # leg's end face line up with it and stay)
        wings = (np.abs(np.abs(p[:, 0]) - 0.145) < 1e-6) & (np.abs(p[:, 1]) > 0.0115)
        self.assertGreater(wings.sum(), 10)
        self.assertTrue(np.all(self.masks["thin"][wings]))

    def test_plate_top_under_the_pad_is_buried(self):
        p, n = self.points, self.normals
        (x0, y0, _), (x1, y1, z1) = PAD
        top = n[:, 2] > 0.9
        inside = (p[:, 0] > x0 + 0.0015) & (p[:, 0] < x1 - 0.0015) & (np.abs(p[:, 1]) < y1 - 0.0015)
        outside = (p[:, 0] < x0 - 0.0015) | (p[:, 0] > x1 + 0.0015) | (np.abs(p[:, 1]) > y1 + 0.0015)
        under = top & inside & (p[:, 2] < z1 - 1e-6)
        self.assertGreater(under.sum(), 50)
        self.assertTrue(np.all(self.masks["buried_top"][under]))
        self.assertFalse(self.masks["buried_top"][top & (np.abs(p[:, 2] - z1) < 1e-6)].any())  # pad top
        plate_top = top & outside & (np.abs(p[:, 2] - PLATE[1][2]) < 1e-6)
        self.assertFalse(self.masks["buried_top"][plate_top].any())                            # open plate top
        leg_tops = top & (np.abs(p[:, 2] - LEGS[0][1][2]) < 1e-6)                              # inside the plate
        self.assertGreater(self.masks["buried_top"][leg_tops].mean(), 0.95)

    def test_coverage_side_group_excludes_masked_surfels(self):
        excluded = self.masks["excluded"]
        model = dict(self.model, observed=excluded.copy())  # only excluded surfels seen
        side = np.abs(self.normals[:, 2]) < 0.5
        plain = directional_surface_coverage(model)
        self.assertEqual(plain["groups"]["side"]["total"], int(side.sum()))
        model["coverage_excluded"] = excluded
        masked = directional_surface_coverage(model)
        top = self.normals[:, 2] >= 0.5
        self.assertEqual(masked["groups"]["side"]["total"], int((side & ~excluded).sum()))
        self.assertEqual(masked["groups"]["top(+Z)"]["total"], int((top & ~excluded).sum()))
        self.assertEqual(masked["groups"]["side"]["observed"], 0)
        self.assertEqual(masked["groups"]["top(+Z)"]["observed"], 0)
        self.assertEqual(masked["excluded"]["observed"], int(excluded.sum()))

if __name__ == "__main__":
    unittest.main()
