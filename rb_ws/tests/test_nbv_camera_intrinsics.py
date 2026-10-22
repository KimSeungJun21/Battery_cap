"""Camera calibration regression checks without Isaac or ROS imports."""

import ast
import os
from pathlib import Path
import unittest
import unittest.mock
from unittest.mock import Mock

import numpy as np


SOURCE = Path(__file__).resolve().parents[1] / "Battery_zivid_setup_utils.py"


class CameraIntrinsicsTests(unittest.TestCase):
    def setUp(self):
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        names = {"square_pixel_intrinsic", "compute_zivid_intrinsic"}
        module = ast.Module(body=[n for n in tree.body
                                  if isinstance(n, ast.FunctionDef) and n.name in names],
                            type_ignores=[])
        self.usd = Mock()
        self.namespace = {"np": np, "UsdGeom": self.usd, "os": os,
                          "_AUTHORED_FOCAL_LENGTHS": {}}
        exec(compile(module, str(SOURCE), "exec"), self.namespace)

    def test_live_camera_and_depth_intrinsics_agree_at_non_native_resolution(self):
        camera = self.usd.Camera.return_value
        camera.GetFocalLengthAttr.return_value.Get.return_value = 18.147562
        camera.GetHorizontalApertureAttr.return_value.Get.return_value = 20.955
        camera.GetVerticalApertureAttr.return_value.Get.return_value = 15.2908
        k = self.namespace["compute_zivid_intrinsic"](Mock(), (2448, 2048))
        self.assertAlmostEqual(k[0, 0], 2120.03015, places=4)
        self.assertEqual(k[1, 1], k[0, 0])
        camera.GetVerticalApertureAttr.return_value.Set.assert_called_once_with(
            20.955 * 2048 / 2448)
        # Reconstruct a noncentral depth sample from the rendered projection.
        point = np.array([.1, .15, .6])
        pixel = k @ point
        pixel /= pixel[2]
        np.testing.assert_allclose(np.linalg.solve(k, pixel) * point[2], point)

    def test_matching_aspect_ratio_needs_no_camera_edit(self):
        camera = self.usd.Camera.return_value
        camera.GetFocalLengthAttr.return_value.Get.return_value = 20.0
        camera.GetHorizontalApertureAttr.return_value.Get.return_value = 24.0
        camera.GetVerticalApertureAttr.return_value.Get.return_value = 18.0
        self.namespace["compute_zivid_intrinsic"](Mock(), (800, 600))
        camera.GetVerticalApertureAttr.return_value.Set.assert_not_called()

    def test_zoom_scales_focal_length_once(self):
        camera = self.usd.Camera.return_value
        focal = camera.GetFocalLengthAttr.return_value
        focal.Get.return_value = 18.147562
        camera.GetHorizontalApertureAttr.return_value.Get.return_value = 20.955
        camera.GetVerticalApertureAttr.return_value.Get.return_value = 20.955 * 2048 / 2448
        prim = Mock()
        with unittest.mock.patch.dict(os.environ, {"BATTERY_ZIVID_ZOOM": "1.2"}):
            k = self.namespace["compute_zivid_intrinsic"](prim, (2448, 2048))
            focal.Set.assert_called_once_with(18.147562 * 1.2)
            self.assertAlmostEqual(k[0, 0], 2120.03015 * 1.2, places=3)
            focal.Get.return_value = 18.147562 * 1.2  # USD now holds the zoomed value
            self.namespace["compute_zivid_intrinsic"](prim, (2448, 2048))
            focal.Set.assert_called_once()

    def test_invalid_calibration_rejected(self):
        for focal, aperture, resolution in [(0, 20, (800, 600)),
                                           (20, 0, (800, 600)),
                                           (20, 20, (800, 0))]:
            with self.assertRaises(ValueError):
                self.namespace["square_pixel_intrinsic"](focal, aperture, resolution)


if __name__ == "__main__":
    unittest.main()
