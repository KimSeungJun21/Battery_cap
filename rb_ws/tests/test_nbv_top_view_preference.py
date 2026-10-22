"""Planner regression checks that run without starting Isaac/ROS or the arm."""

import ast
import math
from pathlib import Path
import unittest
from unittest.mock import Mock

import numpy as np


SOURCE = Path(__file__).resolve().parents[1] / "Battery_nbv_arm_runner_utils.py"
TREE = ast.parse(SOURCE.read_text(encoding="utf-8"))


def planner_functions():
    # Load the actual pure functions without executing simulator imports.
    names = {
        "_boolean_mask_to_bits", "_route_coverage_counts",
        "top_view_gain_weights", "plan_motion_aware_nbv_route",
        "top_completion_candidates",
        "inspection_first_candidates",
    }
    module = ast.Module(
        body=[n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name in names],
        type_ignores=[],
    )
    namespace = {"math": math, "np": np}
    exec(compile(module, str(SOURCE), "exec"), namespace)
    return namespace


class TopViewPreferenceTests(unittest.TestCase):
    def setUp(self):
        self.functions = planner_functions()
        self.directions = {"top": [0, 0, 1], "side": [1, 0, 0], "back": [0, 0, -1]}

    def plan(self, weights=None, branch_width=2, masks=None, goal=2, max_views=1):
        return self.functions["plan_motion_aware_nbv_route"](
            observed_mask=0,
            candidate_visibility_bits=masks or {"top": 3, "side": 3},
            candidate_names=["top", "side"],
            current_edge_costs={"top": 2.0, "side": 1.0},
            pair_edge_costs={"top": {"side": 1.0}, "side": {"top": 2.0}},
            group_bits={"voxel_entropy": 7}, goal_counts={"voxel_entropy": goal},
            max_views=max_views, beam_width=8, branch_width=branch_width,
            candidate_gain_weights=weights,
        )

    def test_bias_tracks_direction_and_turns_off_at_target(self):
        weight = self.functions["top_view_gain_weights"]
        self.assertEqual(weight(self.directions, 0.816, 0.9), {"top": 3, "side": 1, "back": 1})
        self.assertEqual(weight(self.directions, 0.9, 0.9), {"top": 1, "side": 1, "back": 1})
        self.assertEqual(weight(self.directions, 0, 0.9, 1), {"top": 1, "side": 1, "back": 1})

    def test_bias_affects_both_branch_pruning_and_completed_path(self):
        self.assertEqual(self.plan()["route"], ["side"])
        for width in (1, 2):
            plan = self.plan({"top": 3, "side": 1}, branch_width=width)
            self.assertEqual(plan["route"], ["top"])
            self.assertEqual(plan["cost"], 2.0)  # Real motion estimate is preserved.
            self.assertAlmostEqual(plan["selection_cost"], 2.0 / 3.0)
            self.assertEqual(plan["coverage_counts"]["voxel_entropy"], 2)

    def test_bias_cannot_create_information_or_double_count_overlap(self):
        no_gain = self.plan({"top": 3}, masks={"top": 0, "side": 3})
        self.assertEqual(no_gain["route"], ["side"])
        overlap = self.plan({"top": 3}, masks={"top": 3, "side": 6}, goal=3, max_views=2)
        self.assertTrue(overlap["complete"])
        self.assertEqual(len(overlap["route"]), 2)
        self.assertEqual(overlap["coverage_counts"]["voxel_entropy"], 3)

    def test_finish_top_focus_preserves_side_work_and_has_no_gain_fallback(self):
        focus = self.functions["top_completion_candidates"]
        results = [
            {"candidate_name": "top", "entropy_score": 2.0},
            {"candidate_name": "back", "entropy_score": 10.0},
        ]
        self.assertEqual(focus(results, self.directions, .884, .899, .9, .9, 0),
                         (results, False))
        self.assertEqual(focus(results, self.directions, .884, .91, .9, .9, 0),
                         ([results[0]], True))
        self.assertEqual(focus(results, self.directions, .9, .91, .9, .9, 0),
                         (results, False))
        results[0]["entropy_score"] = 0
        self.assertEqual(focus(results, self.directions, .884, .91, .9, .9, 0),
                         (results, False))

    def test_deficient_top_cannot_be_postponed_by_large_side_entropy(self):
        choose = self.functions["inspection_first_candidates"]
        results = [{"candidate_name": name, "entropy_score": gain}
                   for name, gain in [("top", 2), ("side", 50), ("back", 100)]]
        eligible, focus = choose(results, self.directions, [], .248, .807, .9, .9, 0)
        self.assertEqual((eligible, focus), (["top"], "top"))
        # At exactly 90%, a further measurement is needed for the strict >90% KPI.
        self.assertEqual(choose(results, self.directions, [], .9, .95, .9, .9, 0)[1], "top")

    def test_side_focus_prefers_unseen_azimuth_after_top_is_complete(self):
        choose = self.functions["inspection_first_candidates"]
        directions = {"top": [0, 0, 1], "seen": [1, 0, 0], "other": [-1, 0, 0]}
        results = [{"candidate_name": name, "entropy_score": 1}
                   for name in directions]
        eligible, focus = choose(results, directions, [[1, 0, 0], [0, 0, 1]],
                                 .95, .6, .9, .9, 0)
        self.assertEqual((eligible, focus), (["other"], "side"))

    def test_top_focus_avoids_already_observed_direction_when_possible(self):
        choose = self.functions["inspection_first_candidates"]
        directions = {"seen": [0, 0, 1], "novel": [.6, 0, .8]}
        results = [{"candidate_name": name, "entropy_score": 1}
                   for name in directions]
        eligible, _ = choose(results, directions, [[0, 0, 1]], .6, .85, .9, .9, 0)
        self.assertEqual(eligible, ["novel"])

    def test_inspection_hard_rejects_views_under_twenty_degrees(self):
        choose = self.functions["inspection_first_candidates"]
        directions = {
            "close": [math.sin(math.radians(10)), 0, math.cos(math.radians(10))],
            "distinct": [0.5, 0, math.sqrt(0.75)],
        }
        results = [{"candidate_name": name, "entropy_score": gain}
                   for name, gain in [("close", 100), ("distinct", 1)]]
        eligible, focus = choose(
            results, directions, [[0, 0, 1]], .5, .95, .9, .9, 0,
        )
        self.assertEqual((eligible, focus), (["distinct"], "top"))

    def test_top_focus_uses_azimuth_diversity_before_voxel_entropy(self):
        choose = self.functions["inspection_first_candidates"]
        history = [[0.5, 0, math.sqrt(0.75)]]
        directions = {
            # Both are at least 20 degrees from the measurement. The much
            # higher-entropy candidate repeats its azimuth, so it must wait.
            "same_azimuth": [math.sqrt(0.75), 0, 0.5],
            "new_azimuth": [-0.5, 0, math.sqrt(0.75)],
        }
        results = [{"candidate_name": name, "entropy_score": gain}
                   for name, gain in [("same_azimuth", 100), ("new_azimuth", 1)]]
        eligible, focus = choose(
            results, directions, history, .5, .95, .9, .9, 0,
        )
        self.assertEqual((eligible, focus), (["new_azimuth"], "top"))

    def test_side_focus_rejects_bottom_facing_pose_even_with_more_entropy(self):
        choose = self.functions["inspection_first_candidates"]
        directions = {
            "bottom_oblique": [math.sqrt(0.96), 0, -0.2],
            "side": [math.sqrt(1 - 0.04 ** 2), 0, 0.04],
        }
        results = [{"candidate_name": name, "entropy_score": gain}
                   for name, gain in [("bottom_oblique", 100), ("side", 1)]]
        eligible, focus = choose(
            results, directions, [], .95, .5, .9, .9, 0,
        )
        self.assertEqual((eligible, focus), (["side"], "side"))

    def test_beam_first_view_constraint_keeps_complementary_later_views(self):
        planner = self.functions["plan_motion_aware_nbv_route"]
        args = dict(
            observed_mask=0, candidate_visibility_bits={"top": 1, "side": 7},
            candidate_names=["top", "side"],
            current_edge_costs={"top": 2, "side": 1},
            pair_edge_costs={"top": {"side": 1}, "side": {"top": 1}},
            group_bits={"voxel_entropy": 7}, goal_counts={"voxel_entropy": 3},
            max_views=2,
        )
        self.assertEqual(planner(**args)["route"], ["side"])
        plan = planner(**args, first_candidate_names=["top"])
        self.assertEqual(plan["route"], ["top", "side"])
        self.assertEqual(plan["coverage_counts"]["voxel_entropy"], 3)
        self.assertIsNone(planner(**args, first_candidate_names=[]))

    def test_no_gain_is_not_invented_to_satisfy_direction(self):
        choose = self.functions["inspection_first_candidates"]
        results = [{"candidate_name": "top", "entropy_score": 0},
                   {"candidate_name": "side", "entropy_score": 2}]
        eligible, focus = choose(results, self.directions, [], .1, .1, .9, .9, 0)
        self.assertEqual((eligible, focus), (["side"], "side"))

    def test_initial_scores_reused_once_then_capture_is_rescored(self):
        nested = next(n for n in ast.walk(TREE) if isinstance(n, ast.FunctionDef)
                      and n.name == "run_voxel_entropy_scoring")
        factory = ast.parse("def factory(initial_voxel_results):\n    pass\n").body[0]
        factory.body = [nested, ast.Return(value=ast.Name(id=nested.name, ctx=ast.Load()))]
        module = ast.fix_missing_locations(ast.Module(body=[factory], type_ignores=[]))
        score = Mock(side_effect=lambda **kw: [
            {"candidate_name": name, "unknown_entropy_sum": 2.0}
            for name, _ in kw["candidates"]
        ])
        namespace = dict.fromkeys([
            "occupancy_latest_path", "camera_world_matrix", "intrinsic_matrix",
            "image_size", "nbv_pixel_stride", "nbv_roi_min", "nbv_roi_max",
            "voxel_ray_cache",
        ])
        namespace.update(score_candidate_poses=score, log_prefix="[TEST]", max_capture_count=10)
        exec(compile(module, str(SOURCE), "exec"), namespace)
        cached = {"candidate_name": "top", "unknown_entropy_sum": 8.0}
        run = namespace["factory"]([cached])
        self.assertEqual(run([("top", None)], 1)[0]["entropy_score"], 8.0)
        score.assert_not_called()
        self.assertNotIn("entropy_score", cached)
        self.assertEqual(run([("top", None)], 2)[0]["entropy_score"], 2.0)
        score.assert_called_once()
        self.assertTrue(score.call_args.kwargs["score_unknown_only"])
        self.assertIsNone(score.call_args.kwargs["surface_model"])


if __name__ == "__main__":
    unittest.main()
