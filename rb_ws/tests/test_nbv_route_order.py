"""Route ordering / inspection lift checks without starting Isaac or ROS."""

import ast
import itertools
import math
from pathlib import Path
import random
import unittest

import numpy as np


SOURCE = Path(__file__).resolve().parents[1] / "Battery_nbv_arm_runner_utils.py"
TREE = ast.parse(SOURCE.read_text(encoding="utf-8"))


def load_functions():
    names = {
        "_boolean_mask_to_bits", "_route_coverage_counts",
        "plan_motion_aware_nbv_route", "order_route_by_motion_cost",
        "inspection_constraint_lift_reason", "plan_route_with_mandatory",
    }
    module = ast.Module(
        body=[n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name in names],
        type_ignores=[],
    )
    namespace = {"math": math, "np": np}
    exec(compile(module, str(SOURCE), "exec"), namespace)
    return namespace


def path_cost(order, start, pair):
    return sum(start[n] if i == 0 else pair[order[i - 1]][n] for i, n in enumerate(order))


class RouteOrderTests(unittest.TestCase):
    def setUp(self):
        self.functions = load_functions()
        self.order = self.functions["order_route_by_motion_cost"]

    def test_matches_brute_force_on_joint_space_instances(self):
        rng = random.Random(7)
        for _ in range(20):
            names = [f"v{i}" for i in range(6)]
            joints = {n: [rng.uniform(-3, 3) for _ in range(6)] for n in names}
            start_q = [0.0] * 6

            def cost(a, b):
                return 0.7 + 1.336 * max(abs(x - y) for x, y in zip(a, b)) / math.pi

            start = {n: cost(start_q, joints[n]) for n in names}
            pair = {a: {b: cost(joints[a], joints[b]) for b in names if b != a} for a in names}
            ordered, before, after = self.order(names, start, pair)
            best = min(itertools.permutations(names), key=lambda o: path_cost(o, start, pair))
            self.assertEqual(sorted(ordered), sorted(names))
            self.assertAlmostEqual(after, path_cost(best, start, pair))
            self.assertAlmostEqual(after, path_cost(ordered, start, pair))
            self.assertAlmostEqual(before, path_cost(names, start, pair))

    def test_removes_wrist_flip_zigzag(self):
        # +/- wrist flips alternating in the Beam order (as in the measured run).
        wrist = {"a": 1.9, "b": -0.6, "c": 1.4, "d": -2.9}
        start = {n: abs(q - 1.6) for n, q in wrist.items()}
        pair = {a: {b: abs(wrist[a] - wrist[b]) for b in wrist if b != a} for a in wrist}
        ordered, before, after = self.order(["a", "b", "c", "d"], start, pair)
        self.assertEqual(ordered, ["a", "c", "b", "d"])
        self.assertLess(after, before)

    def test_keeps_beam_order_on_ties_trivial_or_oversized_routes(self):
        start = {"a": 1.0, "b": 1.0}
        pair = {"a": {"b": 1.0}, "b": {"a": 1.0}}
        self.assertEqual(self.order(["a", "b"], start, pair)[0], ["a", "b"])
        self.assertEqual(self.order(["b"], start, pair), (["b"], 1.0, 1.0))
        self.assertEqual(self.order([], start, pair), ([], 0.0, 0.0))
        start = {"a": 5.0, "b": 1.0}
        self.assertEqual(self.order(["a", "b"], start, pair, max_exact_views=1)[0], ["a", "b"])

    def test_infinite_edges_are_never_chosen(self):
        inf = float("inf")
        start = {"a": 1.0, "b": 1.0, "c": 0.5}
        pair = {"a": {"b": 1.0, "c": 1.0}, "b": {"a": 1.0, "c": inf},
                "c": {"a": 1.0, "b": inf}}
        ordered, _, after = self.order(["a", "b", "c"], start, pair)
        self.assertEqual(ordered, ["c", "a", "b"])
        self.assertTrue(math.isfinite(after))

    def test_reordering_preserves_planned_coverage(self):
        planner = self.functions["plan_motion_aware_nbv_route"]
        bits = {"a": 0b0011, "b": 0b1100, "c": 0b0110}
        start = {"a": 3.0, "b": 0.5, "c": 1.0}
        pair = {x: {y: 1.0 for y in bits if y != x} for x in bits}
        plan = planner(
            observed_mask=0, candidate_visibility_bits=bits, candidate_names=list(bits),
            current_edge_costs=start, pair_edge_costs=pair,
            group_bits={"voxel_entropy": 0b1111}, goal_counts={"voxel_entropy": 4},
            max_views=3,
        )
        ordered, _, _ = self.order(plan["route"], start, pair)
        union = lambda route: sum(1 << i for i in range(4) if any(bits[n] >> i & 1 for n in route))
        self.assertEqual(union(ordered), union(plan["route"]))
        self.assertEqual(sorted(ordered), sorted(plan["route"]))


class MandatoryRouteTests(unittest.TestCase):
    def setUp(self):
        self.plan = load_functions()["plan_route_with_mandatory"]
        # bits: m1 sees 0-1, m2 sees 2, a sees 0-2 (fully redundant with m1+m2),
        # b sees 3, c sees 3-4.
        self.bits = {"m1": 0b00011, "m2": 0b00100, "a": 0b00111, "b": 0b01000, "c": 0b11000}
        names = list(self.bits)
        self.start = {n: 1.0 for n in names}
        self.pair = {x: {y: 1.0 for y in names if y != x} for x in names}

    def run_plan(self, mandatory, goal=5, max_views=5):
        return self.plan(
            mandatory, self.bits, list(self.bits), self.start, self.pair,
            {"voxel_entropy": 0b11111}, {"voxel_entropy": goal}, max_views,
        )

    def test_mandatory_voxels_are_reflected_before_the_search(self):
        plan = self.run_plan(["m1", "m2"])
        self.assertEqual(plan["route"][:2], ["m1", "m2"])
        # "a" only repeats what the mandatory views see; "c" alone finishes.
        self.assertEqual(plan["route"][2:], ["c"])
        self.assertEqual(plan["mandatory_coverage_counts"]["voxel_entropy"], 3)
        self.assertTrue(plan["complete"])

    def test_mandatory_views_are_kept_even_when_they_add_nothing(self):
        plan = self.run_plan(["m2"], goal=3)
        self.assertIn("m2", plan["route"])
        plan = self.run_plan(["m1", "m2"], goal=3)
        self.assertEqual(plan["route"], ["m1", "m2"])
        self.assertTrue(plan["complete"])

    def test_view_budget_counts_mandatory_nodes(self):
        plan = self.run_plan(["m1", "m2"], max_views=2)
        self.assertEqual(plan["route"], ["m1", "m2"])
        self.assertFalse(plan["complete"])

    def test_unknown_mandatory_names_are_ignored(self):
        self.assertEqual(self.run_plan(["missing"])["mandatory"], [])


class InspectionLiftTests(unittest.TestCase):
    def setUp(self):
        self.lift = load_functions()["inspection_constraint_lift_reason"]

    def test_lifts_on_too_few_eligible_views(self):
        self.assertIsNotNone(self.lift(["only"], "side", 3, None))
        self.assertIsNone(self.lift(["a", "b", "c"], "side", 3, None))

    def test_lifts_after_same_focus_failed_but_not_other_focus(self):
        many = ["a", "b", "c", "d"]
        self.assertIsNotNone(self.lift(many, "side", 3, "side"))
        self.assertIsNone(self.lift(many, "top", 3, "side"))

    def test_non_directional_focus_is_never_lifted(self):
        for focus in ("complete", "direction_unavailable", "no_information"):
            self.assertIsNone(self.lift([], focus, 3, focus))


if __name__ == "__main__":
    unittest.main()
