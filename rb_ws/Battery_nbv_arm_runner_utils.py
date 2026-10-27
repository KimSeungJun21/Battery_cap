"""NBV capture sequence that reorients a gripped object by moving the arm.

Unlike Battery_nbv_runner_utils.py (which teleports a kinematic object), this
module assumes the target object is rigidly held by the epick suction gripper
and achieves each candidate view by sending MoveIt goals that reorient the
tcp link, letting the physically-attached object follow along.

Isaac Sim's bundled rclpy environment does not ship moveit_msgs, so the
actual MoveGroup goal is not sent from here - it is relayed over plain
geometry_msgs/std_msgs topics to nbv_arm_move_relay.py, a small standalone
node run from a normal ROS2/MoveIt-capable shell (see
rbpodo_moveit_config/scripts/nbv_arm_move_relay.py).
"""

from __future__ import annotations

import hashlib
import math
import json
import os
import time
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import omni.timeline
import omni.usd
import rclpy
from geometry_msgs.msg import Pose, PoseStamped, Quaternion
from std_msgs.msg import Bool, String

try:
    from rb_ws.Battery_timing_utils import StageTimer
    from rb_ws.Battery_nbv_sequence_utils import (
        make_nbv_frame_record,
        save_nbv_sequence_manifest,
    )
    from rb_ws.Battery_suction_gripper_utils import compute_relative_pose
    from rb_ws.Battery_stage_setup_utils import update_simulation_frames
    from rb_ws.Battery_zivid_setup_utils import capture_current_zivid_frame
    from rb_ws.NBV_utils import (
        RayGeometryCache,
        build_unknown_voxel_visibility_bitsets,
        make_object_world_matrix,
        print_candidate_scores,
        score_candidate_poses,
    )
except ModuleNotFoundError:
    from Battery_timing_utils import StageTimer
    from Battery_nbv_sequence_utils import (
        make_nbv_frame_record,
        save_nbv_sequence_manifest,
    )
    from Battery_suction_gripper_utils import compute_relative_pose
    from Battery_stage_setup_utils import update_simulation_frames
    from Battery_zivid_setup_utils import capture_current_zivid_frame
    from NBV_utils import (
        RayGeometryCache,
        build_unknown_voxel_visibility_bitsets,
        make_object_world_matrix,
        print_candidate_scores,
        score_candidate_poses,
    )


def quaternion_from_rotation_matrix(rotation):
    """Trace-method conversion of a 3x3 numpy rotation matrix to geometry_msgs/Quaternion."""
    r00, r01, r02 = rotation[0]
    r10, r11, r12 = rotation[1]
    r20, r21, r22 = rotation[2]
    trace = r00 + r11 + r22

    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (r21 - r12) / scale
        y = (r02 - r20) / scale
        z = (r10 - r01) / scale
    elif r00 > r11 and r00 > r22:
        scale = math.sqrt(1.0 + r00 - r11 - r22) * 2.0
        w = (r21 - r12) / scale
        x = 0.25 * scale
        y = (r01 + r10) / scale
        z = (r02 + r20) / scale
    elif r11 > r22:
        scale = math.sqrt(1.0 + r11 - r00 - r22) * 2.0
        w = (r02 - r20) / scale
        x = (r01 + r10) / scale
        y = 0.25 * scale
        z = (r12 + r21) / scale
    else:
        scale = math.sqrt(1.0 + r22 - r00 - r11) * 2.0
        w = (r10 - r01) / scale
        x = (r02 + r20) / scale
        y = (r12 + r21) / scale
        z = 0.25 * scale

    quaternion = Quaternion()
    quaternion.x = float(x)
    quaternion.y = float(y)
    quaternion.z = float(z)
    quaternion.w = float(w)
    return quaternion


def matrix_np_from_gf(gf_matrix):
    return np.asarray(gf_matrix, dtype=np.float64).T


def matrix_np_from_pos_quat(position, quat):
    """Build a numpy column-vector-convention 4x4 from a Gf position/quat.

    Avoids Gf.Matrix4d.SetTransform() (uncertain 2-arg overload) - this is
    the standard closed-form quaternion-to-rotation-matrix formula, applied
    directly in the same numpy convention as the rest of this module.
    """
    w = float(quat.GetReal())
    x, y, z = [float(c) for c in quat.GetImaginary()]
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 1e-12:
        raise ValueError("grasp offset quaternion must not be zero length.")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm

    matrix = np.eye(4, dtype=np.float64)
    matrix[0, 0] = 1.0 - 2.0 * (y * y + z * z)
    matrix[0, 1] = 2.0 * (x * y - w * z)
    matrix[0, 2] = 2.0 * (x * z + w * y)
    matrix[1, 0] = 2.0 * (x * y + w * z)
    matrix[1, 1] = 1.0 - 2.0 * (x * x + z * z)
    matrix[1, 2] = 2.0 * (y * z - w * x)
    matrix[2, 0] = 2.0 * (x * z - w * y)
    matrix[2, 1] = 2.0 * (y * z + w * x)
    matrix[2, 2] = 1.0 - 2.0 * (x * x + y * y)
    matrix[0, 3] = float(position[0])
    matrix[1, 3] = float(position[1])
    matrix[2, 3] = float(position[2])
    return matrix


def pose_matrix_errors(actual_matrix, target_matrix):
    """Return translation metres and shortest 3D rotation error degrees."""
    actual_matrix = np.asarray(actual_matrix, dtype=np.float64)
    target_matrix = np.asarray(target_matrix, dtype=np.float64)
    position_error_m = float(
        np.linalg.norm(actual_matrix[:3, 3] - target_matrix[:3, 3])
    )
    # USD world matrices may contain a harmless authored scale.  Project both
    # 3x3 blocks onto SO(3) first so scale/shear cannot look like angle error.
    def nearest_rotation(matrix):
        left, _, right_transposed = np.linalg.svd(matrix)
        rotation = left @ right_transposed
        if np.linalg.det(rotation) < 0.0:
            left[:, -1] *= -1.0
            rotation = left @ right_transposed
        return rotation

    actual_rotation = nearest_rotation(actual_matrix[:3, :3])
    target_rotation = nearest_rotation(target_matrix[:3, :3])
    relative_rotation = target_rotation.T @ actual_rotation
    cosine = float((np.trace(relative_rotation) - 1.0) * 0.5)
    orientation_error_deg = math.degrees(
        math.acos(max(-1.0, min(1.0, cosine)))
    )
    return position_error_m, orientation_error_deg


def _boolean_mask_to_bits(mask):
    """Pack a numpy boolean mask into a fast Python integer bitset."""

    packed = np.packbits(
        np.asarray(mask, dtype=np.uint8),
        bitorder="little",
    )
    return int.from_bytes(packed.tobytes(), byteorder="little")


def _env_flag(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _surface_model_fingerprint(surface_model):
    """Stable identity for persisted per-surfel visibility observations."""

    digest = hashlib.sha256()
    for key in ("points", "normals"):
        values = np.ascontiguousarray(
            surface_model[key],
            dtype=np.float32,
        )
        digest.update(str(values.shape).encode("ascii"))
        digest.update(values.tobytes())
    return digest.hexdigest()


def load_empirical_visibility_cache(
    path,
    surface_model,
    log_prefix,
):
    """Load persistent pose -> actual-visible-surfel vote counts."""

    fingerprint = _surface_model_fingerprint(surface_model)
    cache = {
        "path": os.path.abspath(path),
        "fingerprint": fingerprint,
        "hit_counts": {},
        "sample_counts": {},
        "dirty": False,
    }
    if not os.path.isfile(path):
        print(
            log_prefix,
            "[IMSI][EMPIRICAL] 실측 visibility cache가 없어 이번 실행에서 "
            f"학습합니다: {path}",
        )
        return cache

    try:
        with np.load(path, allow_pickle=False) as payload:
            stored_fingerprint = str(payload["surface_fingerprint"].item())
            names = [str(value) for value in payload["candidate_names"]]
            hit_counts = np.asarray(payload["hit_counts"], dtype=np.uint16)
            sample_counts = np.asarray(
                payload["sample_counts"],
                dtype=np.uint16,
            )
        expected_shape = (len(names), len(surface_model["observed"]))
        if stored_fingerprint != fingerprint:
            raise ValueError("surface fingerprint changed")
        if hit_counts.shape != expected_shape:
            raise ValueError(
                f"hit_counts shape={hit_counts.shape}, expected={expected_shape}"
            )
        if sample_counts.shape != (len(names),):
            raise ValueError("sample_counts shape mismatch")
        cache["hit_counts"] = {
            name: hit_counts[index].copy()
            for index, name in enumerate(names)
        }
        cache["sample_counts"] = {
            name: int(sample_counts[index])
            for index, name in enumerate(names)
        }
        print(
            log_prefix,
            "[IMSI][EMPIRICAL] 실측 visibility cache 로드:",
            f"poses={len(names)}, samples={int(sample_counts.sum())},",
            path,
        )
    except Exception as error:
        print(
            log_prefix,
            "[IMSI][EMPIRICAL] 기존 cache를 사용할 수 없어 새로 학습합니다:",
            f"{type(error).__name__}: {error}",
        )
    return cache


def empirical_visibility_mask(cache, pose_name, minimum_hit_ratio):
    samples = int(cache["sample_counts"].get(pose_name, 0))
    hits = cache["hit_counts"].get(pose_name)
    if samples <= 0 or hits is None:
        return None
    required_hits = max(1, int(math.ceil(samples * minimum_hit_ratio - 1e-12)))
    return np.asarray(hits >= required_hits, dtype=bool)


def update_empirical_visibility_cache(cache, pose_name, visible_mask):
    """Add one real capture while preventing uint16 vote overflow."""

    visible = np.asarray(visible_mask, dtype=bool)
    samples = int(cache["sample_counts"].get(pose_name, 0))
    hits = cache["hit_counts"].get(pose_name)
    if hits is None or np.asarray(hits).shape != visible.shape:
        hits = np.zeros(visible.shape, dtype=np.uint16)
        samples = 0
    if samples >= np.iinfo(np.uint16).max - 1:
        hits = (np.asarray(hits, dtype=np.uint32) // 2).astype(np.uint16)
        samples //= 2
    hits = np.asarray(hits, dtype=np.uint16)
    hits += visible.astype(np.uint16)
    cache["hit_counts"][pose_name] = hits
    cache["sample_counts"][pose_name] = samples + 1
    cache["dirty"] = True


def save_empirical_visibility_cache(cache, log_prefix):
    if not cache or not cache.get("dirty"):
        return
    names = sorted(cache["hit_counts"])
    if not names:
        return
    path = cache["path"]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary_path = path + ".tmp.npz"
    np.savez_compressed(
        temporary_path,
        schema_version=np.int32(1),
        surface_fingerprint=np.asarray(cache["fingerprint"]),
        candidate_names=np.asarray(names),
        hit_counts=np.stack(
            [cache["hit_counts"][name] for name in names],
            axis=0,
        ),
        sample_counts=np.asarray(
            [cache["sample_counts"][name] for name in names],
            dtype=np.uint16,
        ),
    )
    os.replace(temporary_path, path)
    cache["dirty"] = False
    print(
        log_prefix,
        "[IMSI][EMPIRICAL] 실측 visibility cache 저장:",
        f"poses={len(names)},",
        path,
    )


def route_mask_candidate_results(
    candidate_names,
    visibility_bits,
    observed_mask,
    group_bits,
    top_weight,
    side_weight,
    bottom_weight,
):
    """Build lightweight scorer-compatible records from route bitsets."""

    observed_bits = _boolean_mask_to_bits(observed_mask)
    results = []
    for name in candidate_names:
        visible_bits = int(visibility_bits.get(name, 0))
        new_bits = visible_bits & ~observed_bits
        top_count = int((new_bits & group_bits["top"]).bit_count())
        side_count = int((new_bits & group_bits["side"]).bit_count())
        bottom_count = int((new_bits & group_bits["bottom"]).bit_count())
        new_count = int(new_bits.bit_count())
        visible_count = int(visible_bits.bit_count())
        weighted_score = (
            float(top_weight) * top_count
            + float(side_weight) * side_count
            + float(bottom_weight) * bottom_count
        )
        results.append(
            {
                "candidate_name": name,
                "entropy_score": weighted_score,
                "expected_unknown_visible_count": float(new_count),
                "unknown_visible_count": new_count,
                "free_visible_count": 0,
                "occupied_visible_count": 0,
                "visible_voxel_count": visible_count,
                "mean_visible_entropy": (
                    weighted_score / max(1, visible_count)
                ),
                "surface_unknown_visible_count": new_count,
                "directional_surface_unknown_groups": {
                    "top(+Z)": top_count,
                    "side": side_count,
                    "bottom(-Z)": bottom_count,
                },
                "directional_surface_weighted_unknown_score": weighted_score,
                "score_source": "route_visibility_mask",
            }
        )
    return results


def _coverage_goal_counts(surface_model, targets):
    """Return bit masks and absolute surfel targets for route planning."""

    normals = np.asarray(surface_model["normals"], dtype=np.float64)
    excluded = surface_model.get("coverage_excluded")
    outer = np.ones(len(normals), dtype=bool) if excluded is None else ~np.asarray(excluded, dtype=bool)
    group_arrays = {
        "overall": np.ones(len(normals), dtype=bool),
        "top": (normals[:, 2] >= 0.5) & outer,
        "bottom": normals[:, 2] <= -0.5,
        "side": (np.abs(normals[:, 2]) < 0.5) & outer,
    }
    group_bits = {
        name: _boolean_mask_to_bits(mask)
        for name, mask in group_arrays.items()
    }
    goal_counts = {}
    totals = {}

    for name, mask in group_arrays.items():
        total = int(np.count_nonzero(mask))
        totals[name] = total
        target_fraction = max(0.0, float(targets.get(name, 0.0)))
        goal_counts[name] = min(
            total,
            int(math.ceil(target_fraction * total - 1e-12)),
        )

    return group_bits, goal_counts, totals


def _route_coverage_counts(covered_bits, group_bits):
    return {
        name: int((covered_bits & mask_bits).bit_count())
        for name, mask_bits in group_bits.items()
    }


def cap_route_goals_to_reachable(
    observed_mask,
    candidate_visibility_bits,
    group_bits,
    requested_goal_counts,
    reachable_target_ratio=0.98,
):
    """Cap planning goals to what the reachable candidate graph can see.

    This cap is only for choosing the compact seed route.  The real capture
    termination thresholds remain unchanged and are checked from measured
    coverage after every frame.
    """

    covered_bits = (
        int(observed_mask)
        if isinstance(observed_mask, int)
        else _boolean_mask_to_bits(observed_mask)
    )
    reachable_bits = covered_bits
    for visibility_bits in candidate_visibility_bits.values():
        reachable_bits |= int(visibility_bits)

    ratio = min(1.0, max(0.0, float(reachable_target_ratio)))
    observed_counts = _route_coverage_counts(covered_bits, group_bits)
    reachable_counts = _route_coverage_counts(reachable_bits, group_bits)
    planner_goal_counts = {}

    for name, requested_count in requested_goal_counts.items():
        requested_count = int(requested_count)
        if requested_count <= 0:
            planner_goal_counts[name] = 0
            continue
        if requested_count <= reachable_counts[name]:
            planner_goal_counts[name] = requested_count
            continue
        reachable_target = int(
            math.floor(reachable_counts[name] * ratio + 1e-12)
        )
        # Never make an already achieved group look incomplete merely because
        # the ratio cap was rounded below its measured count.
        reachable_target = max(
            observed_counts[name],
            reachable_target,
        )
        planner_goal_counts[name] = min(
            requested_count,
            reachable_target,
        )

    return planner_goal_counts, reachable_counts


def top_completion_candidates(
    results, view_directions, top_coverage, side_coverage,
    top_target, side_target, min_entropy, min_local_z=0.15,
):
    """Focus on +Z once side is complete; fall back if no useful top view.

    Only candidate eligibility changes. Gains remain current voxel entropy,
    and measured coverage remains the completion criterion.
    """
    if not (
        top_target > 0.0 and side_target > 0.0
        and top_coverage + 1e-12 < top_target
        and side_coverage + 1e-12 >= side_target
    ):
        return results, False
    focused = [
        result for result in results
        if result["entropy_score"] > min_entropy
        and view_directions[result["candidate_name"]][2] >= min_local_z
    ]
    return (focused, True) if focused else (results, False)


def top_view_gain_weights(view_directions, top_coverage, top_target, max_weight=3.0):
    """Prefer front-facing +Z views while measured top coverage is unmet.

    This is a pose-direction prior, not predicted surface coverage. Raw voxel
    entropy and overlap masks remain unchanged; +Z alignment only changes
    gain per motion cost. A side/back view has weight 1, a frontal view W.
    """
    active = float(top_coverage) + 1e-12 < float(top_target)
    bonus = max(0.0, float(max_weight) - 1.0) if active else 0.0
    return {
        name: 1.0 + bonus * float(np.clip(direction[2], 0.0, 1.0)) ** 2
        for name, direction in view_directions.items()
    }


def inspection_first_candidates(
    results, view_directions, captured_directions, top_coverage, side_coverage,
    top_target, side_target, min_entropy, min_separation_deg=20.0,
    top_min_local_z=0.35, side_min_local_z=-0.05,
    side_max_local_z=0.25, azimuth_band_deg=15.0,
):
    """Constrain the NEXT view using measured goal deficits and pose directions.

    This is an inspection-direction prior, not a prediction of unseen surface
    coverage. Node rewards remain UNKNOWN voxel entropy. Do not restrict later
    Beam depths: a top view must still be followed by complementary side views.
    """
    useful = [r["candidate_name"] for r in results
              if r["entropy_score"] > min_entropy]
    if not useful:
        return [], "no_information"
    deficits = {
        "top": (max(1e-9, top_target - top_coverage) / max(top_target, 1e-9)
                if top_target > 0 and top_coverage <= top_target else 0.0),
        "side": (max(1e-9, side_target - side_coverage) / max(side_target, 1e-9)
                 if side_target > 0 and side_coverage <= side_target else 0.0),
    }
    if not any(deficits.values()):
        return useful, "complete"

    def unit(vector):
        vector = np.asarray(vector, dtype=float)
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm > 1e-12 else vector

    directions = {name: unit(view_directions[name]) for name in useful}
    history = [unit(d) for d in captured_directions
               if np.linalg.norm(d) > 1e-12]
    separation = {
        name: min((math.degrees(math.acos(float(np.clip(np.dot(d, h), -1, 1))))
                   for h in history), default=180.0)
        for name, d in directions.items()
    }
    # This is a hard redundancy gate. Re-adding the least-bad pose when every
    # view is too close caused the measured 8.8/10.2/12.8 degree, zero-gain
    # captures. If no distinct view remains, ending early is cheaper and more
    # honest than spending another robot/capture cycle on the same surface.
    useful = [name for name in useful
              if separation[name] + 1e-12 >= min_separation_deg]
    if not useful:
        focus = max(deficits, key=deficits.get)
        return [], focus

    def azimuth_novelty_deg(name, relevant_history):
        horizontal = directions[name][:2]
        horizontal_norm = float(np.linalg.norm(horizontal))
        if not relevant_history:
            return 180.0
        # Azimuth is undefined for a perfectly vertical view. Once one top
        # direction exists, do not let such a view masquerade as novel.
        if horizontal_norm <= 1e-9:
            return 0.0
        horizontal /= horizontal_norm
        return min(
            math.degrees(math.acos(float(np.clip(np.dot(horizontal, prior), -1, 1))))
            for prior in relevant_history
        )

    for focus in sorted(deficits, key=deficits.get, reverse=True):
        if deficits[focus] <= 0:
            continue
        if focus == "top":
            eligible = [name for name in useful
                        if directions[name][2] + 1e-12 >= top_min_local_z]
            if not eligible:
                continue
            top_history = [
                h[:2] / np.linalg.norm(h[:2])
                for h in history
                if h[2] + 1e-12 >= top_min_local_z
                and np.linalg.norm(h[:2]) > 1e-9
            ]
            novelty = {name: azimuth_novelty_deg(name, top_history)
                       for name in eligible}
            best_novelty = max(novelty.values())
            # First preserve the most different top azimuths. The Beam search
            # then chooses among this band using actual UNKNOWN-voxel entropy
            # and motion cost; local-Z itself is no longer the reward.
            eligible = [name for name in eligible
                        if novelty[name] + azimuth_band_deg + 1e-12
                        >= best_novelty]
        else:
            eligible = [name for name in useful
                        if side_min_local_z - 1e-12 <= directions[name][2]
                        <= side_max_local_z + 1e-12]
            if not eligible:
                continue
            # Compare azimuth only against earlier SIDE observations. A frontal
            # top image must not pretend that all side directions were inspected.
            side_history = [
                h[:2] / np.linalg.norm(h[:2])
                for h in history
                if side_min_local_z - 1e-12 <= h[2]
                <= side_max_local_z + 1e-12
                and np.linalg.norm(h[:2]) > 1e-9
            ]
            novelty = {name: azimuth_novelty_deg(name, side_history)
                       for name in eligible}
            best_novelty = max(novelty.values())
            eligible = [name for name in eligible
                        if novelty[name] + azimuth_band_deg + 1e-12
                        >= best_novelty]
        return eligible, focus
    return useful, "direction_unavailable"


def plan_motion_aware_nbv_route(
    observed_mask,
    candidate_visibility_bits,
    candidate_names,
    current_edge_costs,
    pair_edge_costs,
    group_bits,
    goal_counts,
    max_views,
    beam_width=512,
    branch_width=15,
    candidate_gain_weights=None,
    first_candidate_names=None,
):
    """Beam-search a minimum-view, low-motion residual inspection route.

    The first completed beam depth is the minimum number of views found.
    Within that depth the lowest accumulated selection cost wins. Optional
    pose priors divide each edge cost by the destination's gain weight, so
    both branch pruning and final route selection honor the same preference.
    Reported ``cost`` remains the unweighted motion/capture estimate;
    ``selection_cost`` records the weighted optimization objective. This
    function accepts generic coverage bitsets: legacy callers may use surfels,
    while IMSI uses currently visible UNKNOWN occupancy voxels. Rebuilding the
    latter after each capture makes the graph a receding-horizon NBV planner
    rather than a fixed playback sequence. ``first_candidate_names`` constrains
    only the immediate next move; later depths still see the full graph.
    """

    names = [
        name
        for name in candidate_names
        if name in candidate_visibility_bits
        and name in current_edge_costs
    ]
    if not names or max_views <= 0:
        return None
    first_names = None if first_candidate_names is None else set(first_candidate_names)

    gain_weights = {
        name: float((candidate_gain_weights or {}).get(name, 1.0))
        for name in names
    }
    if any(not math.isfinite(w) or w <= 0.0 for w in gain_weights.values()):
        raise ValueError("candidate gain weights must be finite and positive")

    def make_plan(state, complete):
        covered, _, route, selection_cost = state
        motion_cost = sum(
            current_edge_costs[name] if i == 0
            else pair_edge_costs[route[i - 1]][name]
            for i, name in enumerate(route)
        )
        return {
            "route": list(route),
            "cost": float(motion_cost),
            "selection_cost": float(selection_cost),
            "complete": complete,
            "coverage_counts": _route_coverage_counts(covered, group_bits),
        }

    observed_bits = (
        int(observed_mask)
        if isinstance(observed_mask, int)
        else _boolean_mask_to_bits(observed_mask)
    )
    active_goals = {
        name: count
        for name, count in goal_counts.items()
        if count > 0
    }
    if not active_goals:
        return {
            "route": [],
            "cost": 0.0,
            "complete": True,
            "coverage_counts": _route_coverage_counts(
                observed_bits,
                group_bits,
            ),
        }

    def coverage_counts(bits):
        return {
            name: int((bits & group_bits[name]).bit_count())
            for name in active_goals
        }

    def is_complete(counts):
        return all(
            counts[name] >= target
            for name, target in active_goals.items()
        )

    def deficit_score(counts):
        return sum(
            max(0, target - counts[name]) / max(1, target)
            for name, target in active_goals.items()
        )

    initial_counts = coverage_counts(observed_bits)
    if is_complete(initial_counts):
        return {
            "route": [],
            "cost": 0.0,
            "complete": True,
            "coverage_counts": _route_coverage_counts(
                observed_bits,
                group_bits,
            ),
        }

    # covered bits, last candidate, route tuple, accumulated selection cost.
    beam = [(observed_bits, None, tuple(), 0.0)]
    best_partial = None
    beam_width = max(1, int(beam_width))
    branch_width = max(1, int(branch_width))

    for _depth in range(1, max(1, int(max_views)) + 1):
        next_by_signature = {}
        completed = []

        for covered_bits, last_name, route, route_cost in beam:
            current_counts = coverage_counts(covered_bits)
            choices = []

            for name in names:
                if name in route:
                    continue
                if not route and first_names is not None and name not in first_names:
                    continue
                edge_cost = (
                    current_edge_costs[name]
                    if last_name is None
                    else pair_edge_costs[last_name][name]
                )
                if not math.isfinite(edge_cost):
                    continue
                edge_cost /= gain_weights[name]
                new_bits = (
                    candidate_visibility_bits[name] & ~covered_bits
                )
                if not new_bits:
                    continue

                normalized_gain = 0.0
                for group_name, target in active_goals.items():
                    deficit = max(
                        0,
                        target - current_counts[group_name],
                    )
                    if deficit <= 0:
                        continue
                    group_gain = int(
                        (new_bits & group_bits[group_name]).bit_count()
                    )
                    normalized_gain += min(deficit, group_gain) / max(
                        1,
                        target,
                    )
                if normalized_gain <= 0.0:
                    continue
                choices.append(
                    (
                        normalized_gain / max(edge_cost, 1e-9),
                        normalized_gain,
                        -edge_cost,
                        name,
                        edge_cost,
                    )
                )

            choices.sort(reverse=True)
            for _, _, _, name, edge_cost in choices[:branch_width]:
                new_covered = (
                    covered_bits | candidate_visibility_bits[name]
                )
                new_route = route + (name,)
                new_cost = route_cost + edge_cost
                new_counts = coverage_counts(new_covered)
                state = (new_covered, name, new_route, new_cost)
                if is_complete(new_counts):
                    completed.append(state)
                    continue

                signature = (new_covered, name)
                previous = next_by_signature.get(signature)
                if previous is None or new_cost < previous[3]:
                    next_by_signature[signature] = state

                partial_key = (
                    deficit_score(new_counts),
                    len(new_route),
                    new_cost,
                )
                if best_partial is None or partial_key < best_partial[0]:
                    best_partial = (partial_key, state)

        if completed:
            best = min(completed, key=lambda state: state[3])
            return make_plan(best, True)

        if not next_by_signature:
            break

        beam = sorted(
            next_by_signature.values(),
            key=lambda state: (
                deficit_score(coverage_counts(state[0])),
                state[3],
            ),
        )[:beam_width]

    if best_partial is None:
        return None

    best = best_partial[1]
    return make_plan(best, False)


def order_route_by_motion_cost(
    route,
    current_edge_costs,
    pair_edge_costs,
    max_exact_views=10,
):
    """Reorder an already chosen view set into its cheapest open path.

    Beam search decides WHICH views are needed. The union of their
    visibility masks does not depend on visit order, so only motion changes.
    Held-Karp is exact from the current joint state (n=10: 10*2^10 states).
    The Beam order is kept on ties, when a route is longer than
    ``max_exact_views`` or when no finite ordering exists.
    Returns ``(ordered_route, original_cost, ordered_cost)``.
    """

    names = list(dict.fromkeys(route))

    def edge(source, target):
        if source is None:
            return float(current_edge_costs.get(target, float("inf")))
        return float(
            pair_edge_costs.get(source, {}).get(target, float("inf"))
        )

    original_cost = sum(
        edge(names[i - 1] if i else None, name)
        for i, name in enumerate(names)
    )
    count = len(names)
    if count <= 1 or count > max(1, int(max_exact_views)):
        return names, original_cost, original_cost

    # best[(visited mask, last index)] = (cost, previous last index)
    best = {}
    for index, name in enumerate(names):
        cost = edge(None, name)
        if math.isfinite(cost):
            best[(1 << index, index)] = (cost, None)
    for mask in range(1, 1 << count):
        for last in range(count):
            state = best.get((mask, last))
            if state is None:
                continue
            for target in range(count):
                if mask & (1 << target):
                    continue
                step = edge(names[last], names[target])
                if not math.isfinite(step):
                    continue
                key = (mask | (1 << target), target)
                cost = state[0] + step
                if key not in best or cost < best[key][0]:
                    best[key] = (cost, last)

    full_mask = (1 << count) - 1
    endings = [
        (best[(full_mask, last)][0], last)
        for last in range(count)
        if (full_mask, last) in best
    ]
    if not endings:
        return names, original_cost, original_cost
    ordered_cost, last = min(endings)
    if ordered_cost >= original_cost - 1e-9:
        return names, original_cost, original_cost

    ordered = []
    mask = full_mask
    while last is not None:
        ordered.append(names[last])
        previous = best[(mask, last)][1]
        mask &= ~(1 << last)
        last = previous
    ordered.reverse()
    return ordered, original_cost, ordered_cost


def plan_route_with_mandatory(
    mandatory_names,
    candidate_visibility_bits,
    candidate_names,
    current_edge_costs,
    pair_edge_costs,
    group_bits,
    goal_counts,
    max_views,
    observed_mask=0,
    **planner_kwargs,
):
    """Beam-search the remaining views around must-visit nodes.

    Every route contains the pending mandatory views. Their visible UNKNOWN
    voxels are treated as already observed ("reflected") before the search,
    so Beam only spends views on what they cannot see. The returned route is
    ``mandatory + chosen``; visit order is left to order_route_by_motion_cost.
    """

    mandatory = [
        name
        for name in dict.fromkeys(mandatory_names)
        if name in candidate_visibility_bits
    ]
    observed_bits = (
        int(observed_mask)
        if isinstance(observed_mask, int)
        else _boolean_mask_to_bits(observed_mask)
    )
    for name in mandatory:
        observed_bits |= int(candidate_visibility_bits[name])
    others = [name for name in candidate_names if name not in set(mandatory)]
    remaining_views = int(max_views) - len(mandatory)
    plan = None
    if remaining_views > 0 and others:
        plan = plan_motion_aware_nbv_route(
            observed_mask=observed_bits,
            candidate_visibility_bits=candidate_visibility_bits,
            candidate_names=others,
            current_edge_costs=current_edge_costs,
            pair_edge_costs=pair_edge_costs,
            group_bits=group_bits,
            goal_counts=goal_counts,
            max_views=remaining_views,
            **planner_kwargs,
        )
    if plan is None:
        counts = _route_coverage_counts(observed_bits, group_bits)
        plan = {
            "route": [],
            "cost": 0.0,
            "selection_cost": 0.0,
            "complete": all(
                counts.get(name, 0) >= target
                for name, target in goal_counts.items()
                if target > 0
            ),
            "coverage_counts": counts,
        }
    plan = dict(plan)
    plan["route"] = mandatory + [
        name for name in plan["route"] if name not in set(mandatory)
    ]
    plan["mandatory"] = list(mandatory)
    plan["mandatory_coverage_counts"] = _route_coverage_counts(
        observed_bits, group_bits
    )
    plan.setdefault("selection_cost", plan.get("cost", 0.0))
    return plan


def inspection_constraint_lift_reason(
    eligible_names,
    focus,
    min_eligible,
    failed_focus,
):
    """Why the next-view direction constraint should be lifted, if at all.

    The constraint is a pose prior. When it leaves fewer than
    ``min_eligible`` views, or a view it admitted for the same deficient
    group has already produced too little measured gain there, it is no
    longer steering toward new surface: the measured run spent 3 cycles on
    1-2 admitted redundant views while the view that finished side coverage
    (local-Z -0.12) was outside the side band.
    """

    if focus not in {"top", "side"}:
        return None
    if failed_focus == focus:
        return f"low measured {focus} gain from a constrained view"
    if len(eligible_names) < max(0, int(min_eligible)):
        return f"eligible={len(eligible_names)} < {int(min_eligible)}"
    return None


# Object-local face normals for resolution-aware ("quality") coverage.
QUALITY_FACE_NORMALS = {
    "+Z": (0.0, 0.0, 1.0),
    "+X": (1.0, 0.0, 0.0),
    "-X": (-1.0, 0.0, 0.0),
    "+Y": (0.0, 1.0, 0.0),
    "-Y": (0.0, -1.0, 0.0),
}


def quality_face_group_masks(normals, group_names, excluded=None):
    """Split surfels into +Z and four side faces by their dominant normal axis.

    The plain side group mixes the long +/-Y faces with the X end faces, so a
    side percentage can be met without ever facing an end face (the 0/5
    recall run saw both end-face defects only at 66-80 deg incidence).
    ``excluded`` (coverage_exclusion_masks: legs' inner faces, +/-Y faces,
    thin strips, buried top layers) is left out, so +/-Y can come back empty.
    """

    normals = np.asarray(normals, dtype=np.float64)
    keep = np.ones(len(normals), dtype=bool)
    if excluded is not None:
        keep = ~np.asarray(excluded, dtype=bool)
    side = (np.abs(normals[:, 2]) < 0.5) & keep
    x_dominant = np.abs(normals[:, 0]) >= np.abs(normals[:, 1])
    masks = {
        "+Z": (normals[:, 2] >= 0.5) & keep,
        "+X": side & x_dominant & (normals[:, 0] > 0.0),
        "-X": side & x_dominant & (normals[:, 0] <= 0.0),
        "+Y": side & ~x_dominant & (normals[:, 1] > 0.0),
        "-Y": side & ~x_dominant & (normals[:, 1] <= 0.0),
    }
    return {name: masks[name] for name in group_names}


def face_deficit_candidates(
    observed,
    face_masks,
    view_directions,
    min_coverage,
    max_incidence_deg,
    excluded_faces=(),
):
    """Next-view constraint for a face that measured coverage has barely seen.

    Among faces seen below ``min_coverage``, pick the one with the most
    unobserved surfels and return the candidates whose pose direction faces
    it within ``max_incidence_deg``. In battery_rb_20260929_123052 the -Y face
    was 0.4% seen after frame 4 while voxel entropy chose two diagonal views
    that could not see it (+0.3/+0.07%p). Returns ``(face, names, fraction)``
    or ``None``.
    """

    observed = np.asarray(observed, dtype=bool)
    cos_limit = math.cos(math.radians(float(max_incidence_deg)))
    deficient = []
    for face, mask in face_masks.items():
        total = int(np.count_nonzero(mask))
        if total == 0 or face in excluded_faces:
            continue
        fraction = float(np.count_nonzero(observed & mask)) / total
        if fraction + 1e-12 < float(min_coverage):
            deficient.append((total - int(np.count_nonzero(observed & mask)), face, fraction))
    for _, face, fraction in sorted(deficient, reverse=True):
        normal = np.asarray(QUALITY_FACE_NORMALS[face], dtype=np.float64)
        names = []
        for name, direction in view_directions.items():
            direction = np.asarray(direction, dtype=np.float64)
            norm = float(np.linalg.norm(direction))
            if norm > 1e-12 and float(np.dot(direction / norm, normal)) + 1e-12 >= cos_limit:
                names.append(name)
        if names:
            return face, names, fraction
    return None


def surfel_pixel_density(points, normals, sensor_origin, reference_distance):
    """Projected pixels relative to a frontal view at ``reference_distance``.

    ``(d_ref / d)^2 * cos(incidence)`` is the same basis as the defect-recall
    denominator (frontal pixels at the nominal 0.40 m), without using any
    defect ground truth.
    """

    points = np.asarray(points, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)
    view = np.asarray(sensor_origin, dtype=np.float64)[None, :] - points
    distance = np.maximum(np.linalg.norm(view, axis=1), 1e-9)
    facing = np.einsum("ij,ij->i", normals, view) / distance
    return np.where(
        facing > 0.0,
        (float(reference_distance) / distance) ** 2 * facing,
        0.0,
    )


def update_quality_observations(
    surface_model,
    visible_mask,
    sensor_origin,
    reference_distance,
    min_density,
):
    """Mark surfels seen by this capture with enough pixel density."""

    quality = surface_model.get("quality_observed")
    if quality is None or len(quality) != len(surface_model["points"]):
        quality = np.zeros(len(surface_model["points"]), dtype=bool)
    good = np.asarray(visible_mask, dtype=bool) & (
        surfel_pixel_density(
            surface_model["points"],
            surface_model["normals"],
            sensor_origin,
            reference_distance,
        )
        >= float(min_density)
    )
    newly_observed = int(np.count_nonzero(good & ~quality))
    surface_model["quality_observed"] = quality | good
    return newly_observed


def quality_coverage_fractions(surface_model, group_masks):
    quality = surface_model.get("quality_observed")
    if quality is None:
        quality = np.zeros(len(surface_model["points"]), dtype=bool)
    return {
        name: (
            float(np.count_nonzero(quality & mask)) / np.count_nonzero(mask)
            if np.any(mask)
            else 1.0
        )
        for name, mask in group_masks.items()
    }


def predicted_face_density(
    view_direction,
    distance,
    reference_distance,
    face_normal,
):
    """Pose-direction prior for one face: no CAD visibility or occlusion."""

    direction = np.asarray(view_direction, dtype=np.float64)
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-12 or distance <= 1e-9:
        return 0.0
    facing = float(np.dot(direction / norm, face_normal))
    return max(0.0, facing) * (float(reference_distance) / distance) ** 2


def rank_quality_candidates(
    candidate_names,
    view_directions,
    distances,
    edge_costs,
    deficits,
    reference_distance,
    min_density,
):
    """Rank next views for the deficient quality faces by gain per second.

    A candidate serves a face when its predicted face density reaches
    ``min_density``. Returns ``(ranked, uncoverable_faces)`` where ranked is
    ``[(name, served_faces, score), ...]``.
    """

    ranked = []
    coverable = set()
    for name in candidate_names:
        cost = float(edge_costs.get(name, float("inf")))
        if not math.isfinite(cost) or name not in view_directions:
            continue
        served = [
            face
            for face in deficits
            if predicted_face_density(
                view_directions[name],
                float(distances.get(name, float("inf"))),
                reference_distance,
                QUALITY_FACE_NORMALS[face],
            )
            + 1e-12
            >= min_density
        ]
        if not served:
            continue
        coverable.update(served)
        score = sum(deficits[face] for face in served) / max(cost, 1e-9)
        ranked.append((score, -cost, name, served))
    ranked.sort(reverse=True)
    return (
        [(name, served, score) for score, _, name, served in ranked],
        sorted(set(deficits) - coverable),
    )


def proximity_gain_weights(distances, baseline_distance, max_weight, exponent=2.0):
    """Pixel-density prior: a pose held closer than the lift pose gets its
    edge cost divided by ``(baseline / d)^exponent`` (clipped to [1, max])."""

    maximum = max(1.0, float(max_weight))
    return {
        name: float(np.clip(
            (float(baseline_distance) / max(float(distance), 1e-9))
            ** float(exponent),
            1.0,
            maximum,
        ))
        for name, distance in distances.items()
    }


def rotation_toward(normal, target_direction, angle_deg):
    """Rotation turning ``normal`` toward ``target_direction`` by at most
    ``angle_deg`` (Rodrigues). Identity when they are (anti)parallel."""

    normal = np.asarray(normal, dtype=np.float64)
    target = np.asarray(target_direction, dtype=np.float64)
    normal = normal / np.linalg.norm(normal)
    target = target / np.linalg.norm(target)
    axis = np.cross(normal, target)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= 1e-9:
        return np.eye(3)
    axis /= axis_norm
    full_angle = math.atan2(axis_norm, float(np.dot(normal, target)))
    angle = min(math.radians(float(angle_deg)), full_angle)
    skew = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    return (
        np.eye(3)
        + math.sin(angle) * skew
        + (1.0 - math.cos(angle)) * (skew @ skew)
    )


def rotation_about_axis(axis, angle_deg):
    """Rodrigues rotation about ``axis`` by ``angle_deg``."""

    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    angle = math.radians(float(angle_deg))
    skew = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    return (
        np.eye(3)
        + math.sin(angle) * skew
        + (1.0 - math.cos(angle)) * (skew @ skew)
    )


def frontal_top_delta_rotation(top_normal, camera_forward, roll_deg):
    """Delta rotation putting ``top_normal`` exactly against the camera's
    forward axis (top face parallel to the image plane), then rolling the
    part about that axis by ``roll_deg``."""

    toward_camera = -np.asarray(camera_forward, dtype=np.float64)
    align = rotation_toward(top_normal, toward_camera, 180.0)
    if np.dot(align @ np.asarray(top_normal, dtype=np.float64), toward_camera) < 0.0:
        # Exactly anti-parallel: rotation_toward returns identity.
        perpendicular = np.cross(toward_camera, [1.0, 0.0, 0.0])
        if np.linalg.norm(perpendicular) < 1e-6:
            perpendicular = np.cross(toward_camera, [0.0, 1.0, 0.0])
        align = rotation_about_axis(perpendicular, 180.0)
    return rotation_about_axis(toward_camera, roll_deg) @ align


def rotation_xyz_deg_from_matrix(rotation):
    """Inverse of make_object_world_matrix's Rz @ Ry @ Rx (for records)."""

    rotation = np.asarray(rotation, dtype=np.float64)
    ry = math.asin(float(np.clip(-rotation[2, 0], -1.0, 1.0)))
    rx = math.atan2(rotation[2, 1], rotation[2, 2])
    rz = math.atan2(rotation[1, 0], rotation[0, 0])
    return tuple(round(math.degrees(value), 3) for value in (rx, ry, rz))


def make_pose(x, y, z, orientation):
    pose = Pose()
    pose.position.x = float(x)
    pose.position.y = float(y)
    pose.position.z = float(z)
    pose.orientation = orientation
    return pose


class NbvArmMoveRelayClient:
    """Talks to nbv_arm_move_relay.py over plain topics.

    Isaac Sim's bundled rclpy has no moveit_msgs, so the MoveGroup goal
    itself is sent by that separate process; this just publishes a target
    PoseStamped and waits for a Bool success/failure result. Advances the
    sim one frame per poll while waiting, since Isaac Sim only steps
    physics when simulation_app.update() is called and the arm's actual
    motion (which the relay's goal is waiting on) depends on that.
    """

    def __init__(
        self,
        node,
        simulation_app,
        request_topic="/nbv/move_request",
        result_topic="/nbv/move_result",
        reason_topic="/nbv/move_reason",
        frustum_config_topic="/nbv/frustum_guard_config",
        preflight_request_topic="/nbv/preflight_request",
        preflight_result_topic="/nbv/preflight_result",
        joint_request_topic="/nbv/joint_request",
        frame_id="world",
        result_timeout=60.0,
        log_prefix="[NBV-ARM]",
    ):
        self.node = node
        self.simulation_app = simulation_app
        self.frame_id = frame_id
        self.result_timeout = result_timeout
        self.log_prefix = log_prefix
        self._latest_result = None
        self._results_received = 0
        # relay가 reason 문자열에 실어 보내는 MoveIt 계측값.
        # Isaac의 rclpy에는 moveit_msgs가 없어 이 경로로만 받을 수 있다.
        self.last_move_plan_s = None
        self.last_move_traj_s = None
        self.last_move_ik_s = None
        self.last_move_accept_s = None
        self.last_move_guard_s = None
        self.last_move_exec_s = None
        self._latest_reason = None
        self._frustum_guard_json = None
        self._latest_preflight_result = None
        self._preflight_results_received = 0

        self.request_publisher = node.create_publisher(
            PoseStamped,
            request_topic,
            10,
        )
        self.frustum_config_publisher = node.create_publisher(
            String,
            frustum_config_topic,
            10,
        )
        self.preflight_request_publisher = node.create_publisher(
            String,
            preflight_request_topic,
            10,
        )
        self.joint_request_publisher = node.create_publisher(
            String,
            joint_request_topic,
            10,
        )
        self.result_subscription = node.create_subscription(
            Bool,
            result_topic,
            self._handle_result,
            10,
        )
        self.reason_subscription = node.create_subscription(
            String,
            reason_topic,
            self._handle_reason,
            10,
        )
        self.preflight_result_subscription = node.create_subscription(
            String,
            preflight_result_topic,
            self._handle_preflight_result,
            10,
        )

    def _handle_result(self, message):
        self._latest_result = bool(message.data)
        self._results_received += 1

    def _handle_reason(self, message):
        """relay의 reason 문자열에서 plan=/traj= 를 뽑는다.

        형식: "succeeded plan=0.123 traj=1.450"
        계측값이 없는 실패 경로에서는 토큰이 아예 없다.
        """
        text = str(message.data)
        self._latest_reason = text
        self._parse_timing_tokens(text)

    def _handle_preflight_result(self, message):
        try:
            self._latest_preflight_result = json.loads(message.data)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            self._latest_preflight_result = {
                "success": False,
                "reason": f"invalid_preflight_result:{exc}",
            }
        self._preflight_results_received += 1

    def _parse_timing_tokens(self, text):
        """Extract relay timing fields from either result channel."""

        for token in text.split():
            key, _, value = token.partition("=")

            if not value:
                continue

            try:
                seconds = float(value)
            except ValueError:
                continue

            if key == "plan":
                self.last_move_plan_s = seconds
            elif key == "traj":
                self.last_move_traj_s = seconds
            elif key == "ik":
                self.last_move_ik_s = seconds
            elif key == "accept":
                self.last_move_accept_s = seconds
            elif key == "guard":
                self.last_move_guard_s = seconds
            elif key == "exec":
                self.last_move_exec_s = seconds

    def wait_until_ready(self, timeout_sec=20.0):
        deadline = time.monotonic() + timeout_sec
        while rclpy.ok() and time.monotonic() < deadline:
            if self.request_publisher.get_subscription_count() > 0:
                return True
            self.simulation_app.update()
            rclpy.spin_once(self.node, timeout_sec=0.1)
        return self.request_publisher.get_subscription_count() > 0

    def wait_until_preflight_ready(self, timeout_sec=20.0):
        deadline = time.monotonic() + timeout_sec
        while rclpy.ok() and time.monotonic() < deadline:
            ready = (
                self.preflight_request_publisher.get_subscription_count() > 0
                and self.joint_request_publisher.get_subscription_count() > 0
            )
            if ready:
                return True
            self.simulation_app.update()
            rclpy.spin_once(self.node, timeout_sec=0.1)
        return (
            self.preflight_request_publisher.get_subscription_count() > 0
            and self.joint_request_publisher.get_subscription_count() > 0
        )

    def configure_frustum_guard(
        self,
        camera_matrix,
        intrinsic_matrix,
        image_size,
        corners_local,
        grasp_offset,
        margin_px,
        min_in_frame_fraction,
    ):
        """Send the geometry needed to validate MoveIt's final stopped pose.

        All transforms use MoveIt's robot-base/world frame and column-vector
        convention.  The payload is republished before every pose request so
        a restarted relay cannot accidentally execute without the guard.
        """
        payload = {
            "schema_version": 1,
            "camera_matrix": np.asarray(
                camera_matrix,
                dtype=np.float64,
            ).reshape(4, 4).tolist(),
            "intrinsic_matrix": np.asarray(
                intrinsic_matrix,
                dtype=np.float64,
            ).reshape(3, 3).tolist(),
            "image_size": [int(image_size[0]), int(image_size[1])],
            "corners_local": np.asarray(
                corners_local,
                dtype=np.float64,
            ).reshape(-1, 3).tolist(),
            "grasp_offset": np.asarray(
                grasp_offset,
                dtype=np.float64,
            ).reshape(4, 4).tolist(),
            "margin_px": float(margin_px),
            "min_in_frame_fraction": float(min_in_frame_fraction),
        }
        self._frustum_guard_json = json.dumps(
            payload,
            separators=(",", ":"),
        )
        self._publish_frustum_guard_config()

    def _publish_frustum_guard_config(self):
        if self._frustum_guard_json is None:
            return

        message = String()
        message.data = self._frustum_guard_json
        self.frustum_config_publisher.publish(message)

    def _reset_move_result_state(self):
        results_before = self._results_received
        self.last_move_plan_s = None
        self.last_move_traj_s = None
        self.last_move_ik_s = None
        self.last_move_accept_s = None
        self.last_move_guard_s = None
        self.last_move_exec_s = None
        self._latest_reason = None
        return results_before

    def _wait_for_move_result(self, label, results_before):
        started_at = time.monotonic()
        deadline = started_at + self.result_timeout
        next_progress_at = started_at + 5.0
        while (
            rclpy.ok()
            and self._results_received == results_before
            and time.monotonic() < deadline
        ):
            self.simulation_app.update()
            rclpy.spin_once(self.node, timeout_sec=0.0)
            now = time.monotonic()
            if now >= next_progress_at:
                elapsed = now - started_at
                self.node.get_logger().info(
                    f"{self.log_prefix} Waiting for MoveIt: {label}, "
                    f"elapsed={elapsed:.1f}s / "
                    f"timeout={self.result_timeout:.1f}s"
                )
                next_progress_at = now + 5.0

        elapsed = time.monotonic() - started_at

        # relay는 reason을 result보다 먼저 보내지만, spin_once는 콜백을
        # 하나씩 처리하므로 Bool에서 루프를 빠져나온 시점에 String이
        # 아직 큐에 남아 있을 수 있다. 짧게만 더 돌려 받아둔다.
        if self._results_received != results_before:
            drain_deadline = time.monotonic() + 0.05

            while (
                rclpy.ok()
                and self._latest_reason is None
                and time.monotonic() < drain_deadline
            ):
                rclpy.spin_once(self.node, timeout_sec=0.0)

        if self._results_received == results_before:
            self.node.get_logger().warning(
                f"{self.log_prefix} No result from nbv_arm_move_relay.py "
                f"within {self.result_timeout:.0f}s: {label} "
                f"(elapsed={elapsed:.1f}s)"
            )
            return False

        if not self._latest_result:
            self.node.get_logger().warning(
                f"{self.log_prefix} Relay reported move failure: {label} "
                f"(reason={self._latest_reason or 'unknown'}, "
                f"elapsed={elapsed:.1f}s)"
            )
            return False

        self.node.get_logger().info(
            f"{self.log_prefix} Move succeeded: {label} "
            f"(elapsed={elapsed:.1f}s)"
        )
        return True

    def move_to_pose(self, label, pose):
        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = self.frame_id
        pose_stamped.header.stamp = self.node.get_clock().now().to_msg()
        pose_stamped.pose = pose

        results_before = self._reset_move_result_state()
        # The relay fails closed when the final-pose guard is enabled but no
        # geometry has arrived. Republish immediately before the request as
        # well as when configured, so normal DDS packet timing cannot leave a
        # pose request paired with stale/missing geometry.
        self._publish_frustum_guard_config()
        self.request_publisher.publish(pose_stamped)
        return self._wait_for_move_result(label, results_before)

    def preflight_pose(self, label, pose):
        """Plan without execution and return the planned final joints."""
        payload = {
            "schema_version": 1,
            "label": str(label),
            "frame_id": self.frame_id,
            "position": [
                float(pose.position.x),
                float(pose.position.y),
                float(pose.position.z),
            ],
            "orientation_xyzw": [
                float(pose.orientation.x),
                float(pose.orientation.y),
                float(pose.orientation.z),
                float(pose.orientation.w),
            ],
        }
        results_before = self._preflight_results_received
        self._latest_preflight_result = None
        self._publish_frustum_guard_config()
        message = String()
        message.data = json.dumps(payload, separators=(",", ":"))
        self.preflight_request_publisher.publish(message)

        started_at = time.monotonic()
        deadline = started_at + self.result_timeout
        next_progress_at = started_at + 5.0
        while (
            rclpy.ok()
            and self._preflight_results_received == results_before
            and time.monotonic() < deadline
        ):
            self.simulation_app.update()
            rclpy.spin_once(self.node, timeout_sec=0.0)
            now = time.monotonic()
            if now >= next_progress_at:
                elapsed = now - started_at
                self.node.get_logger().info(
                    f"{self.log_prefix} Preflight planning: {label}, "
                    f"elapsed={elapsed:.1f}s / "
                    f"timeout={self.result_timeout:.1f}s"
                )
                next_progress_at = now + 5.0

        elapsed = time.monotonic() - started_at

        if self._preflight_results_received == results_before:
            self.node.get_logger().warning(
                f"{self.log_prefix} No preflight result within "
                f"{self.result_timeout:.0f}s: {label} "
                f"(elapsed={elapsed:.1f}s)"
            )
            return None

        result = self._latest_preflight_result or {}
        if not result.get("success", False):
            self.node.get_logger().warning(
                f"{self.log_prefix} Preflight rejected: {label} "
                f"(reason={result.get('reason', 'unknown')}, "
                f"elapsed={elapsed:.1f}s)"
            )
            return result

        self.node.get_logger().info(
            f"{self.log_prefix} Preflight succeeded: {label}, "
            f"joints={len(result.get('joint_names', []))} "
            f"(elapsed={elapsed:.1f}s)"
        )
        return result

    def move_to_joint_positions(self, label, joint_names, positions):
        if len(joint_names) != len(positions) or not joint_names:
            raise ValueError("joint_names/positions must be non-empty and equal")

        payload = {
            "schema_version": 1,
            "label": str(label),
            "joint_names": [str(name) for name in joint_names],
            "positions": [float(value) for value in positions],
        }
        results_before = self._reset_move_result_state()
        self._publish_frustum_guard_config()
        message = String()
        message.data = json.dumps(payload, separators=(",", ":"))
        self.joint_request_publisher.publish(message)
        return self._wait_for_move_result(label, results_before)

    def destroy(self):
        if self.node is not None:
            self.node.destroy_node()
            self.node = None


def create_nbv_move_client(
    simulation_app,
    request_topic="/nbv/move_request",
    result_topic="/nbv/move_result",
    frustum_config_topic="/nbv/frustum_guard_config",
    frame_id="world",
    result_timeout=60.0,
    log_prefix="[NBV-ARM]",
):
    if not rclpy.ok():
        rclpy.init(args=None)

    node = rclpy.create_node("battery_nbv_arm_controller")
    client = NbvArmMoveRelayClient(
        node=node,
        simulation_app=simulation_app,
        request_topic=request_topic,
        result_topic=result_topic,
        frustum_config_topic=frustum_config_topic,
        frame_id=frame_id,
        result_timeout=result_timeout,
        log_prefix=log_prefix,
    )

    if not client.wait_until_ready():
        node.get_logger().warning(
            f"{log_prefix} nbv_arm_move_relay.py was not detected on "
            f"{request_topic} - make sure it is running."
        )

    return client


def read_joint_positions(rb_robot):
    """Current DOF positions as a {joint_name: radians} dict."""
    positions = np.asarray(
        rb_robot.get_dof_positions()
    ).reshape(-1)
    return {
        name: float(value)
        for name, value in zip(rb_robot.dof_names, positions)
    }


def quat_to_xyzw_list(quat):
    imaginary = quat.GetImaginary()
    return [
        float(imaginary[0]),
        float(imaginary[1]),
        float(imaginary[2]),
        float(quat.GetReal()),
    ]


def measure_grasp_offset(stage, object_prim_path, tcp_prim_path):
    """Object pose expressed relative to tcp - constant while rigidly held."""
    return compute_relative_pose(
        stage=stage,
        child_prim_path=object_prim_path,
        parent_prim_path=tcp_prim_path,
    )


def predict_object_matrix(tcp_matrix_np, grasp_offset_np):
    return tcp_matrix_np @ grasp_offset_np


def _load_surface_coverage_tools():
    """표면 커버리지 계산에 필요한 함수들을 늦게 import한다.

    Battery_defect_utils는 pxr/PIL을 끌어오므로 모듈 최상단에서 받으면
    순환 import가 생길 수 있다.
    """
    try:
        from rb_ws.Battery_coverage_metrics import (
            surface_coverage,
            surface_voxel_mask,
        )
        from rb_ws.Battery_defect_utils import collect_object_frame_mesh
    except ModuleNotFoundError:
        from Battery_coverage_metrics import (
            surface_coverage,
            surface_voxel_mask,
        )
        from Battery_defect_utils import collect_object_frame_mesh

    return surface_voxel_mask, surface_coverage, collect_object_frame_mesh


def _load_directional_surface_tools():
    try:
        from rb_ws.Battery_coverage_metrics import (
            directional_surface_coverage,
            sample_directional_surfels,
            update_directional_surfel_observations,
        )
        from rb_ws.Battery_defect_utils import collect_object_frame_mesh
    except ModuleNotFoundError:
        from Battery_coverage_metrics import (
            directional_surface_coverage,
            sample_directional_surfels,
            update_directional_surfel_observations,
        )
        from Battery_defect_utils import collect_object_frame_mesh

    return (
        sample_directional_surfels,
        update_directional_surfel_observations,
        directional_surface_coverage,
        collect_object_frame_mesh,
    )


def build_directional_surface_model(
    stage,
    target_prim_path,
    sample_count,
    log_prefix,
):
    sample_surfels, _, _, collect_mesh = _load_directional_surface_tools()
    object_mesh, _ = collect_mesh(
        stage,
        target_prim_path,
        log_prefix=log_prefix,
    )
    model = sample_surfels(
        object_mesh.vertices,
        object_mesh.triangles,
        samples=sample_count,
        seed=0,
    )
    try:
        from rb_ws.Battery_coverage_metrics import coverage_exclusion_masks
    except ModuleNotFoundError:
        from Battery_coverage_metrics import coverage_exclusion_masks
    # side counts the legs' outer faces only (the legs' inner faces, the thin
    # +/-Y lines and other < 8 mm strips leave it); top counts only surfaces
    # not buried under another CAD layer.
    start = time.perf_counter()
    masks = coverage_exclusion_masks(model["points"], model["normals"])
    model["coverage_excluded"] = masks["excluded"]
    normals = np.asarray(model["normals"])
    side = np.abs(normals[:, 2]) < 0.5
    top = normals[:, 2] >= 0.5
    print(
        log_prefix,
        "방향성 surface 분모:",
        f"{len(model['points'])} surfels,",
        f"실제 표면적 {model['total_area'] * 1e4:.1f} cm2,",
        f"top은 묻힌 층 제외 {int((top & ~masks['excluded']).sum())}/{int(top.sum())}",
        f"(묻힌 층 {int(masks['buried_top'].sum())}),",
        f"side는 다리 바깥면만 {int((side & ~masks['excluded']).sum())}/{int(side.sum())}",
        f"(제외: 다리 안쪽 {int(masks['inner'].sum())},",
        f"+/-Y 얇은 면 {int(masks['y_facing'].sum())},",
        f"기타 8mm 미만 띠 {int(masks['thin'].sum())});",
        f"{time.perf_counter() - start:.2f}s",
    )
    return model


def _load_frontier_tools():
    try:
        import rb_ws.Battery_frontier_map_20261027 as module
    except ModuleNotFoundError:
        import Battery_frontier_map_20261027 as module
    return module


def _last_surface_observation():
    try:
        from rb_ws.Zivid_capture_utils import LAST_TARGET_SURFACE_OBSERVATION
    except ModuleNotFoundError:
        from Zivid_capture_utils import LAST_TARGET_SURFACE_OBSERVATION
    return LAST_TARGET_SURFACE_OBSERVATION


def update_directional_surface_from_last_capture(
    surface_model,
    frame_id,
    occupancy_map_dir,
    match_distance,
    min_facing_cos,
    log_prefix,
    save_debug_artifacts=False,
):
    try:
        from rb_ws.Zivid_capture_utils import (
            LAST_TARGET_SURFACE_OBSERVATION,
        )
    except ModuleNotFoundError:
        from Zivid_capture_utils import LAST_TARGET_SURFACE_OBSERVATION

    _, update_surfels, coverage_fn, _ = _load_directional_surface_tools()
    observation = LAST_TARGET_SURFACE_OBSERVATION

    if int(observation.get("frame_id", -1)) != int(frame_id):
        raise RuntimeError(
            "직전 surface observation frame이 현재 capture와 다릅니다: "
            f"expected={frame_id}, actual={observation.get('frame_id')}"
        )

    target_points = observation.get("target_points_object")
    sensor_origin = observation.get("sensor_origin_object")
    if target_points is None or sensor_origin is None:
        raise RuntimeError(
            f"frame {frame_id}의 조밀한 surface observation이 없습니다."
        )
    # Quality coverage needs this capture's sensor-surfel distance.
    surface_model["last_sensor_origin_object"] = np.asarray(
        sensor_origin, dtype=np.float64
    ).reshape(3)

    newly_observed, visible_mask = update_surfels(
        surface_model,
        target_points_object=target_points,
        sensor_origin_object=sensor_origin,
        match_distance=match_distance,
        min_facing_cos=min_facing_cos,
        return_visible_mask=True,
    )
    coverage = coverage_fn(surface_model)
    if save_debug_artifacts:
        save_directional_surface_model(
            surface_model=surface_model,
            occupancy_map_dir=occupancy_map_dir,
            frame_id=frame_id,
            save_frame_snapshot=True,
        )
    print(
        log_prefix,
        f"frame {frame_id} 방향성 surfel 갱신:",
        f"new={newly_observed},",
        f"누적={coverage['observed']}/{coverage['total']}",
    )
    return coverage, visible_mask


def save_directional_surface_model(
    surface_model,
    occupancy_map_dir,
    frame_id,
    save_frame_snapshot=False,
):
    """Persist the current directional state once, plus history in debug."""

    os.makedirs(occupancy_map_dir, exist_ok=True)
    payload = {
        "points": np.asarray(surface_model["points"], dtype=np.float32),
        "normals": np.asarray(surface_model["normals"], dtype=np.float32),
        "observed": np.asarray(surface_model["observed"], dtype=bool),
        "coverage_excluded": np.asarray(
            surface_model.get("coverage_excluded", np.zeros(len(surface_model["points"]))),
            dtype=bool,
        ),
        "total_area": np.float64(surface_model["total_area"]),
        "frame_id": np.int32(frame_id),
    }
    np.savez_compressed(
        os.path.join(occupancy_map_dir, "directional_surface_latest.npz"),
        **payload,
    )
    if save_frame_snapshot:
        np.savez_compressed(
            os.path.join(
                occupancy_map_dir,
                f"directional_surface_{int(frame_id):06d}.npz",
            ),
            **payload,
        )


def report_directional_surface_coverage(
    surface_model,
    elapsed_s=None,
    elapsed_excl_render_s=None,
    log_prefix="[NBV-ARM]",
):
    _, _, coverage_fn, _ = _load_directional_surface_tools()
    result = coverage_fn(surface_model)
    if elapsed_s is None:
        suffix = ""
    elif elapsed_excl_render_s is None:
        suffix = f"  (t+{elapsed_s:.1f}s)"
    else:
        suffix = (
            f"  (t+{elapsed_s:.1f}s, render 제외 "
            f"{elapsed_excl_render_s:.1f}s)"
        )
    group_text = ", ".join(
        f"{name}={entry['fraction'] * 100:.1f}%"
        for name, entry in result["groups"].items()
    )
    # Inspection targets only top and side; bottom(-Z) is never scored, so
    # also report coverage over top+side surfels alone.
    top_side_observed = sum(
        result["groups"][name]["observed"] for name in ("top(+Z)", "side")
    )
    top_side_total = sum(
        result["groups"][name]["total"] for name in ("top(+Z)", "side")
    )
    result["top_side"] = {
        "observed": top_side_observed,
        "total": top_side_total,
        "fraction": top_side_observed / top_side_total if top_side_total else 0.0,
    }
    print(
        f"{log_prefix} 방향성 표면 커버리지: "
        f"{result['observed']}/{result['total']} = "
        f"{result['fraction'] * 100:.1f}% "
        f"[{group_text}] "
        f"top+side={top_side_observed}/{top_side_total} = "
        f"{result['top_side']['fraction'] * 100:.1f}%{suffix}"
    )
    return result


def build_surface_voxel_mask(
    stage,
    target_prim_path,
    occupancy_path,
    log_prefix="[NBV-ARM]",
):
    """물체 실제 표면을 occupancy 격자에 이산화한 mask를 만든다.

    격자 전체를 분모로 쓰면 95%가 물체와 무관한 원거리 공기라 수치가
    의미를 잃는다. 검사 성공률은 '물체 표면을 얼마나 봤는가'이므로
    실제 메시를 분모로 삼는다.
    """
    if not os.path.isfile(occupancy_path):
        print(
            f"{log_prefix} occupancy map이 아직 없어 표면 mask 계산을 "
            "건너뜁니다."
        )

        return None

    voxelize, _, collect_mesh = _load_surface_coverage_tools()
    object_mesh, _ = collect_mesh(
        stage,
        target_prim_path,
        log_prefix=log_prefix,
    )

    with np.load(occupancy_path) as data:
        origin = np.asarray(data["origin"], dtype=np.float64)
        leaf_size = float(data["leaf_size"])
        grid_shape = tuple(
            np.asarray(data["grid_shape"], dtype=np.int32).tolist()
        )

    mask, area = voxelize(
        object_mesh.vertices,
        object_mesh.triangles,
        origin,
        leaf_size,
        grid_shape,
    )
    print(
        f"{log_prefix} 표면 기준 분모: {int(mask.sum())} voxel "
        f"(실제 표면적 {area * 1e4:.1f} cm2, leaf {leaf_size * 1000:.0f}mm)"
    )

    return mask


def report_surface_coverage(
    occupancy_path,
    surface_mask,
    elapsed_s=None,
    elapsed_excl_render_s=None,
    log_prefix="[NBV-ARM]",
):
    """현재 occupancy map의 표면 커버리지를 계산해 출력한다."""
    if surface_mask is None or not os.path.isfile(occupancy_path):
        return None

    _, coverage_fn, _ = _load_surface_coverage_tools()

    with np.load(occupancy_path) as data:
        state = np.asarray(data["state"], dtype=np.int8)

    fraction, observed, total = coverage_fn(state, surface_mask)
    if elapsed_s is None:
        suffix = ""
    elif elapsed_excl_render_s is None:
        suffix = f"  (t+{elapsed_s:.1f}s)"
    else:
        # render는 Isaac 전용 비용이라 실기 예산에서는 빠진다.
        suffix = (
            f"  (t+{elapsed_s:.1f}s, render 제외 "
            f"{elapsed_excl_render_s:.1f}s)"
        )
    print(
        f"{log_prefix} 표면 커버리지: {observed}/{total} = "
        f"{fraction * 100:.1f}%{suffix}"
    )

    return fraction


def camera_direction_in_object_frame(
    object_world_np,
    camera_world_matrix,
    reference_local_point=None,
):
    """object frame에서 본 '물체 기준점 -> 카메라' 단위 방향.

    월드 카메라는 고정이므로, 이 방향이 곧 '물체의 어느 면이 보이는가'다.
    물체를 시선축 둘레로 아무리 돌려도 이 방향은 거의 안 바뀌고, 그런
    뷰는 새 표면을 드러내지 못한다.

    reference_local_point는 그 '기준점'을 object frame에서 지정한다.
    기본값(원점)은 authored origin이 형상 안에 있을 때만 맞다.
    batterycap.usd는 origin이 형상보다 0.4668 m 아래에 있어서, 카메라가
    0.5 m 거리일 때 origin->카메라 방향이 부품->카메라 방향과 전혀 달랐다.
    실측: 90도 롤(local_x_neg_90deg)이 이격 23.1도로 계산됐고(실제 92.4도),
    그 결과 멀쩡한 후보가 매 사이클 29~39개씩 '중복 뷰'로 버려져 스윕이
    5프레임 만에 no_reachable_candidates로 끝났다. 형상 중심을 넘기면
    실제 각도가 나온다.
    """
    object_world_np = np.asarray(object_world_np, dtype=np.float64)
    camera_world_matrix = np.asarray(
        camera_world_matrix, dtype=np.float64
    )

    if reference_local_point is None:
        reference_world = object_world_np[:3, 3]
    else:
        reference_world = object_world_np[:3, 3] + (
            object_world_np[:3, :3]
            @ np.asarray(reference_local_point, dtype=np.float64)
        )

    offset = camera_world_matrix[:3, 3] - reference_world
    direction = object_world_np[:3, :3].T @ offset
    norm = np.linalg.norm(direction)

    if norm < 1e-9:
        return np.array([0.0, 0.0, 1.0])

    return direction / norm


def object_local_bbox_corners(stage, target_prim_path, log_prefix):
    """The target's 8 bbox corners, in its own frame.

    Taken from the same object-frame mesh the surface-coverage denominator
    uses, so no pxr import is needed at module scope.
    """
    _, _, collect_mesh = _load_surface_coverage_tools()
    object_mesh, _ = collect_mesh(
        stage,
        target_prim_path,
        log_prefix=log_prefix,
    )
    vertices = np.asarray(object_mesh.vertices, dtype=np.float64)
    minimum = vertices.min(axis=0)
    maximum = vertices.max(axis=0)
    return np.array(
        [
            [x, y, z]
            for x in (minimum[0], maximum[0])
            for y in (minimum[1], maximum[1])
            for z in (minimum[2], maximum[2])
        ],
        dtype=np.float64,
    )


def project_object_framing(
    object_world_np,
    camera_world_matrix,
    intrinsic_matrix,
    image_size,
    corners_local,
    margin_px=0.0,
):
    """How much of the part a candidate pose would actually put on the sensor.

    Batterycap candidates rotate around the geometry centre; legacy targets
    can still rotate around a fixed TCP. Either way IK/workspace limits can
    put an endpoint partly outside the sensor. Scoring alone does not catch
    that, hence this check.

    Returns (in_frame_fraction, off_axis_deg, min_depth).
    """
    object_world_np = np.asarray(object_world_np, dtype=np.float64)
    camera_world_matrix = np.asarray(camera_world_matrix, dtype=np.float64)
    intrinsic_matrix = np.asarray(intrinsic_matrix, dtype=np.float64)

    position = camera_world_matrix[:3, 3]
    right = camera_world_matrix[:3, 0]
    up = camera_world_matrix[:3, 1]
    view = -camera_world_matrix[:3, 2]

    fx = float(intrinsic_matrix[0, 0])
    fy = float(intrinsic_matrix[1, 1])
    cx = float(intrinsic_matrix[0, 2])
    cy = float(intrinsic_matrix[1, 2])
    width, height = (int(image_size[0]), int(image_size[1]))

    corners_world = (
        np.asarray(corners_local, dtype=np.float64)
        @ object_world_np[:3, :3].T
        + object_world_np[:3, 3]
    )

    inside = 0
    depths = []

    for point in corners_world:
        offset = point - position
        depth = float(offset @ view)
        depths.append(depth)

        if depth <= 1e-6:
            continue

        u = cx + fx * float(offset @ right) / depth
        v = cy + fy * float(-(offset @ up)) / depth

        if (
            margin_px <= u < width - margin_px
            and margin_px <= v < height - margin_px
        ):
            inside += 1

    centre_world = (
        object_world_np[:3, :3] @ np.asarray(corners_local).mean(axis=0)
        + object_world_np[:3, 3]
    )
    centre_offset = centre_world - position
    norm = np.linalg.norm(centre_offset)
    off_axis = (
        float(np.degrees(np.arccos(np.clip((centre_offset / norm) @ view,
                                           -1.0, 1.0))))
        if norm > 1e-9
        else 0.0
    )

    return inside / len(corners_world), off_axis, min(depths)


def min_view_separation_deg(direction, captured_directions):
    """이미 찍은 뷰들과의 최소 각도 이격(도). 이력이 없으면 180."""
    if not captured_directions:
        return 180.0

    dots = [
        float(np.clip(np.dot(direction, other), -1.0, 1.0))
        for other in captured_directions
    ]

    return float(np.degrees(np.arccos(max(dots))))


def solve_tcp_pose_for_candidate(
    initial_object_rotation_base_np,
    grasp_offset_np,
    reference_tcp_position,
    rotation_xyz_deg,
    initial_object_base_matrix=None,
    object_pivot_local_point=None,
    delta_rotation_np=None,
    pivot_base_override=None,
):
    """``delta_rotation_np`` replaces the XYZ Euler delta; with a pivot,
    ``pivot_base_override`` moves the geometry centre (e.g. nearer the camera).
    """
    if delta_rotation_np is None:
        delta_rotation_np = make_object_world_matrix(
            position=(0.0, 0.0, 0.0),
            rotation_xyz_deg=rotation_xyz_deg,
        )[:3, :3]
    delta_rotation_np = np.asarray(delta_rotation_np, dtype=np.float64)
    desired_object_rotation_base_np = (
        delta_rotation_np @ initial_object_rotation_base_np
    )

    if object_pivot_local_point is None:
        # Legacy mustard behavior: rotate around the held TCP position.
        grasp_offset_rotation_np = grasp_offset_np[:3, :3]
        tcp_rotation_np = (
            desired_object_rotation_base_np
            @ np.linalg.inv(grasp_offset_rotation_np)
        )
        tcp_matrix_np = np.eye(4, dtype=np.float64)
        tcp_matrix_np[:3, :3] = tcp_rotation_np
        tcp_matrix_np[:3, 3] = reference_tcp_position
    else:
        # batterycap.usd's authored origin is about 0.467 m below its actual
        # geometry. Keeping TCP translation fixed while rotating that origin
        # makes the part swing out of view. Rotate about the supplied geometry
        # centre instead, then derive TCP from object = TCP @ grasp_offset.
        if initial_object_base_matrix is None:
            raise ValueError(
                "object pivot을 쓰려면 initial_object_base_matrix가 "
                "필요합니다."
            )
        initial_object_base_matrix = np.asarray(
            initial_object_base_matrix,
            dtype=np.float64,
        )
        pivot_local = np.asarray(
            object_pivot_local_point,
            dtype=np.float64,
        )
        pivot_base = (
            initial_object_base_matrix[:3, :3] @ pivot_local
            + initial_object_base_matrix[:3, 3]
        )
        if pivot_base_override is not None:
            pivot_base = np.asarray(pivot_base_override, dtype=np.float64)
        desired_object_base_np = np.eye(4, dtype=np.float64)
        desired_object_base_np[:3, :3] = desired_object_rotation_base_np
        desired_object_base_np[:3, 3] = (
            pivot_base - desired_object_rotation_base_np @ pivot_local
        )
        tcp_matrix_np = (
            desired_object_base_np @ np.linalg.inv(grasp_offset_np)
        )

    predicted_object_np = predict_object_matrix(tcp_matrix_np, grasp_offset_np)
    return tcp_matrix_np, predicted_object_np


def run_nbv_capture_sequence_with_arm(
    stage,
    simulation_app,
    move_client,
    rb_robot,
    target_prim,
    target_prim_path,
    tcp_prim_path,
    robot_base_prim_path,
    capture_poses,
    pose_settle_update_count,
    camera_prim,
    zivid_render_product,
    zivid_rgb_annotator,
    zivid_depth_annotator,
    zivid_semantic_annotator,
    zivid_instance_annotator,
    zivid_bbox_annotator,
    orchestrator,
    zivid_save_dir,
    intrinsic_matrix,
    image_size,
    nbv_roi_min,
    nbv_roi_max,
    nbv_pixel_stride,
    nbv_max_capture_count,
    nbv_min_expected_entropy,
    object_grid_leaf_size,
    object_grid_root_size,
    object_grid_origin,
    is_still_gripped,
    nbv_min_view_separation_deg=20.0,
    nbv_view_reference_local_point=None,
    nbv_min_in_frame_fraction=1.0,
    nbv_frame_margin_px=0.0,
    nbv_surface_coverage_target=0.93,
    nbv_top_surface_coverage_target=0.0,
    nbv_side_surface_coverage_target=0.0,
    nbv_min_surface_coverage_gain=0.0,
    nbv_min_surface_coverage_gain_consecutive=1,
    nbv_bottom_surface_coverage_target=0.0,
    nbv_max_consecutive_move_failures=0,
    nbv_background_score_weight=0.001,
    nbv_surface_top_score_weight=1.0,
    nbv_surface_bottom_score_weight=1.0,
    nbv_surface_side_score_weight=1.0,
    nbv_surfel_sample_count=50000,
    nbv_surfel_match_distance=0.003,
    nbv_surfel_min_facing_cos=0.05,
    nbv_surfel_pixel_bin_size=8,
    nbv_surfel_occlusion_tolerance=0.003,
    log_prefix="[NBV-ARM]",
):
    # 물체를 들어올린 직후 호출되므로, 여기가 검사 동작의 t=0이다.
    sequence_t0 = time.monotonic()

    # Replicator 렌더에 쓴 시간의 누적.
    #
    # 실기 Zivid는 이 비용을 치르지 않으므로(하드웨어 캡처는 수백 ms에
    # 끝나고 Isaac의 오프스크린 렌더와 무관하다), 검사 시간 예산을 볼
    # 때는 이 몫을 뺀 값도 같이 본다. 벽시계 시간(t+)은 그대로 둔다.
    render_total_s = [0.0]

    def accumulate_render_time():
        """직전 캡처의 render 구간을 누적한다."""
        try:
            from rb_ws.Zivid_capture_utils import LAST_CAPTURE_SPANS
        except ModuleNotFoundError:
            from Zivid_capture_utils import LAST_CAPTURE_SPANS

        render_total_s[0] += float(LAST_CAPTURE_SPANS.get("render", 0.0))

    def elapsed_pair():
        """(벽시계 경과, render 제외 경과)를 돌려준다."""
        wall = time.monotonic() - sequence_t0

        return wall, wall - render_total_s[0]
    save_debug_artifacts = _env_flag(
        "BATTERY_SAVE_DEBUG_ARTIFACTS",
        default=False,
    )
    empirical_visibility_cache = None
    imsi_empirical_visibility_enabled = False
    imsi_empirical_minimum_hit_ratio = 0.5
    imsi_empirical_bootstrap_min_cached_poses = 0
    imsi_empirical_bootstrap_pose_order = []
    precomputed_joint_view_limit = max(
        0,
        int(os.environ.get("BATTERY_IMSI_PRECOMPUTE_VIEW_LIMIT", "0")),
    )
    imsi_top_side_only_termination = (
        precomputed_joint_view_limit > 0
        and os.environ.get(
            "BATTERY_IMSI_TOP_SIDE_ONLY_TERMINATION",
            "1",
        ).strip().lower() not in {"0", "false", "no", "off"}
    )
    imsi_inspection_focus = imsi_top_side_only_termination and _env_flag(
        "BATTERY_IMSI_INSPECTION_FOCUS", True,
    )
    imsi_inspection_min_view_separation_deg = max(
        0.0,
        float(os.environ.get(
            "BATTERY_IMSI_INSPECTION_MIN_VIEW_SEPARATION_DEG", "20.0"
        )),
    )
    imsi_inspection_top_min_local_z = min(
        1.0,
        max(-1.0, float(os.environ.get(
            "BATTERY_IMSI_INSPECTION_TOP_MIN_LOCAL_Z", "0.35"
        ))),
    )
    imsi_inspection_side_min_local_z = min(
        1.0,
        max(-1.0, float(os.environ.get(
            "BATTERY_IMSI_INSPECTION_SIDE_MIN_LOCAL_Z", "-0.05"
        ))),
    )
    imsi_inspection_side_max_local_z = min(
        1.0,
        max(imsi_inspection_side_min_local_z, float(os.environ.get(
            "BATTERY_IMSI_INSPECTION_SIDE_MAX_LOCAL_Z", "0.25"
        ))),
    )
    imsi_inspection_azimuth_band_deg = max(
        0.0,
        float(os.environ.get(
            "BATTERY_IMSI_INSPECTION_AZIMUTH_BAND_DEG", "15.0"
        )),
    )
    # 제약이 이 개수 미만의 후보만 남기거나, 같은 focus로 허용한 뷰가
    # 실측 gain을 못 내면 다음 뷰 제약을 풀고 전체 후보로 되돌린다.
    imsi_inspection_min_eligible = max(0, int(os.environ.get(
        "BATTERY_IMSI_INSPECTION_MIN_ELIGIBLE", "3"
    )))
    imsi_inspection_fallback_min_gain = max(0.0, float(os.environ.get(
        "BATTERY_IMSI_INSPECTION_FALLBACK_MIN_GAIN", "0.005"
    )))
    # Beam이 고른 뷰 집합을 관절 이동 최소 순서로 재배열하고 전체를
    # 커밋한다. 재계획은 실측 gain 부족/경로 소진/목표 변화 때만 한다.
    imsi_route_order_tsp = _env_flag("BATTERY_IMSI_ROUTE_ORDER_TSP", False)
    imsi_route_replan_min_gain = max(0.0, float(os.environ.get(
        "BATTERY_IMSI_ROUTE_REPLAN_MIN_GAIN", "0.005"
    )))

    def _env_float_list(name, default):
        return [
            float(value)
            for value in os.environ.get(name, default).split(",")
            if value.strip()
        ]

    # 실측 side 커버리지에서 거의 안 보인 면(+/-X, +/-Y)이 있으면 그 면을
    # 향하는 후보로 다음 뷰를 제한한다(측정 결손 + pose 방향 prior).
    imsi_face_deficit_focus = _env_flag("BATTERY_IMSI_FACE_DEFICIT_FOCUS", False)
    # top/side 목표를 넘으면 필수 node 잔여나 quality 부족과 무관하게 즉시
    # 종료한다. ACCEPTANCE도 coverage + 시간만 보고 나머지는 참고로 출력.
    imsi_stop_at_coverage = _env_flag("BATTERY_IMSI_STOP_AT_COVERAGE", False)
    # Mesh-free decisions (Battery_frontier_map_20261027): NBV gain = visible
    # frontier voxels of a 2.5 mm map carved with every depth ray; stop when
    # no remaining candidate sees enough frontier. GT coverage is reported.
    imsi_frontier_enabled = _env_flag("BATTERY_IMSI_FRONTIER_MAP", False)
    imsi_frontier_leaf_m = max(5e-4, float(os.environ.get(
        "BATTERY_IMSI_FRONTIER_LEAF_M", "0.0025"
    )))
    imsi_frontier_pixel_stride = max(1, int(os.environ.get(
        "BATTERY_IMSI_FRONTIER_PIXEL_STRIDE", "6"
    )))
    imsi_frontier_margin_m = max(0.0, float(os.environ.get(
        "BATTERY_IMSI_FRONTIER_MARGIN_M", "0.01"
    )))
    imsi_frontier_stop_ratio = max(0.0, float(os.environ.get(
        "BATTERY_IMSI_FRONTIER_STOP_RATIO", "0.05"
    )))
    imsi_frontier_stop_min_voxels = max(0, int(os.environ.get(
        "BATTERY_IMSI_FRONTIER_STOP_MIN_VOXELS", "30"
    )))
    imsi_frontier_replan_min_resolved = min(1.0, max(0.0, float(os.environ.get(
        "BATTERY_IMSI_FRONTIER_REPLAN_MIN_RESOLVED", "0.30"
    ))))
    # Underside frontier (adjacent part above it, within acos(cos) of -Z) is
    # not counted: the bottom is not inspected. Empty = count everything.
    _frontier_down = os.environ.get("BATTERY_IMSI_FRONTIER_EXCLUDE_DOWN_COS", "").strip()
    imsi_frontier_exclude_down_cos = float(_frontier_down) if _frontier_down else None
    imsi_face_deficit_min_coverage = min(1.0, max(0.0, float(os.environ.get(
        "BATTERY_IMSI_FACE_DEFICIT_MIN_COVERAGE", "0.50"
    ))))
    imsi_face_deficit_max_incidence_deg = min(89.0, max(0.0, float(os.environ.get(
        "BATTERY_IMSI_FACE_DEFICIT_MAX_INCIDENCE_DEG", "60.0"
    ))))
    # 카메라 쪽 접근: 각 후보를 이 pivot-카메라 거리들에서 먼저 preflight
    # 하고, 모두 실패하면 원래(들어올린) 거리로 되돌아간다. 비우면 끈다.
    # NBV 작업점(회전 중심) 평행이동, world 좌표 x,y,z [m]. 들어올린
    # 위치 대신 이 점을 기준으로 모든 후보/접근 거리를 만든다.
    imsi_workspace_offset_world_m = np.zeros(3)
    workspace_offset_values = _env_float_list(
        "BATTERY_IMSI_WORKSPACE_OFFSET_M", "0,0,0"
    )
    if len(workspace_offset_values) == 3:
        imsi_workspace_offset_world_m = np.asarray(
            workspace_offset_values, dtype=np.float64
        )
    imsi_approach_distances_m = sorted(
        value for value in _env_float_list(
            "BATTERY_IMSI_APPROACH_DISTANCES_M", ""
        ) if value > 0.0
    )
    # 윗면을 카메라 쪽으로 기울인 근접 pose를 첫 NBV 뷰로 강제한다.
    imsi_top_closeup_enabled = _env_flag("BATTERY_IMSI_TOP_CLOSEUP", False)
    imsi_top_closeup_tilts_deg = [
        value for value in _env_float_list(
            "BATTERY_IMSI_TOP_CLOSEUP_TILTS_DEG", "25,20,15"
        ) if value > 0.0
    ]
    # "tilt": +Z를 카메라 쪽으로 tilt만큼 기울임(현재 동작).
    # "frontal": +Z를 광축과 정확히 반대로(윗면이 영상면과 평행) 두고,
    # 형상 중심을 광축 위에 놓아 화면 정중앙에 온다. roll은 광축 둘레 회전.
    imsi_top_closeup_mode = os.environ.get(
        "BATTERY_IMSI_TOP_CLOSEUP_MODE", "tilt"
    ).strip().lower()
    imsi_top_closeup_rolls_deg = _env_float_list(
        "BATTERY_IMSI_TOP_CLOSEUP_ROLLS_DEG", "0"
    ) or [0.0]
    imsi_top_closeup_frontal_distances_m = sorted(
        value for value in _env_float_list(
            "BATTERY_IMSI_TOP_CLOSEUP_FRONTAL_DISTANCES_M", ""
        ) if value > 0.0
    )
    # 사용자가 직접 맞춘 첫 pose. /joint_states 관절값(아래 이름 순서)과
    # MoveIt world(=link0) 기준 TCP pose x,y,z,qx,qy,qz,qw. 둘 다 있으면
    # 이 pose가 강제 첫 뷰가 되고 top close-up보다 우선한다.
    imsi_first_pose_joint_names = [
        name.strip()
        for name in os.environ.get(
            "BATTERY_IMSI_FIRST_POSE_JOINT_NAMES",
            "base,shoulder,elbow,wrist1,wrist2,wrist3",
        ).split(",")
        if name.strip()
    ]
    imsi_first_pose_joints = _env_float_list(
        "BATTERY_IMSI_FIRST_POSE_JOINTS", ""
    )
    imsi_first_pose_tcp = _env_float_list("BATTERY_IMSI_FIRST_POSE_TCP", "")
    imsi_first_pose_enabled = (
        len(imsi_first_pose_joints) == len(imsi_first_pose_joint_names) > 0
        and len(imsi_first_pose_tcp) == 7
    )
    if (imsi_first_pose_joints or imsi_first_pose_tcp) and not imsi_first_pose_enabled:
        print(
            log_prefix,
            "[IMSI][FIRST-POSE] 무시: joints",
            f"{len(imsi_first_pose_joints)}/{len(imsi_first_pose_joint_names)}개,",
            f"tcp {len(imsi_first_pose_tcp)}/7개 값이 필요합니다.",
        )
    # 반드시 방문할 사용자 pose들. JOINTS/TCP는 '|'로 pose를 구분하고
    # 각 pose는 FIRST_POSE와 같은 형식(관절 6개, TCP 7개)이다. 이 pose들의
    # 가시 UNKNOWN voxel을 관측된 것으로 반영한 뒤 남은 영역만 Beam으로
    # 탐색하고, 전체 경로(필수 + 선택)는 관절 이동 최소 순서로 방문한다.
    def _env_pose_groups(name):
        return [
            [float(value) for value in group.split(",") if value.strip()]
            for group in os.environ.get(name, "").split("|")
            if group.strip()
        ]

    mandatory_joint_groups = _env_pose_groups("BATTERY_IMSI_MANDATORY_POSE_JOINTS")
    mandatory_tcp_groups = _env_pose_groups("BATTERY_IMSI_MANDATORY_POSE_TCP")
    mandatory_labels = [
        label.strip()
        for label in os.environ.get("BATTERY_IMSI_MANDATORY_POSE_NAMES", "").split(",")
        if label.strip()
    ]
    imsi_mandatory_poses = []
    if mandatory_joint_groups or mandatory_tcp_groups:
        if len(mandatory_joint_groups) != len(mandatory_tcp_groups) or any(
            len(joints) != len(imsi_first_pose_joint_names) or len(tcp) != 7
            for joints, tcp in zip(mandatory_joint_groups, mandatory_tcp_groups)
        ):
            print(
                log_prefix,
                "[IMSI][MANDATORY] 무시: pose마다 관절",
                f"{len(imsi_first_pose_joint_names)}개와 TCP 7개가 필요합니다",
                f"(joints groups={[len(g) for g in mandatory_joint_groups]},",
                f"tcp groups={[len(g) for g in mandatory_tcp_groups]})",
            )
        else:
            for index, (joints, tcp) in enumerate(
                zip(mandatory_joint_groups, mandatory_tcp_groups)
            ):
                label = (
                    mandatory_labels[index]
                    if index < len(mandatory_labels)
                    else str(index + 1)
                )
                imsi_mandatory_poses.append({
                    "name": f"mandatory_{label}",
                    "joints": joints,
                    "tcp": tcp,
                })
    if imsi_mandatory_poses and imsi_first_pose_enabled:
        print(
            log_prefix,
            "[IMSI][MANDATORY] 필수 pose가 있어 FIRST_POSE 설정은 무시합니다.",
        )
        imsi_first_pose_enabled = False
    imsi_mandatory_names = [pose["name"] for pose in imsi_mandatory_poses]
    mandatory_name_set = set(imsi_mandatory_names)
    if imsi_mandatory_poses and not imsi_route_order_tsp:
        print(
            log_prefix,
            "[IMSI][MANDATORY] 필수 경로 전체를 커밋해야 하므로",
            "BATTERY_IMSI_ROUTE_ORDER_TSP를 켭니다.",
        )
        imsi_route_order_tsp = True
    # 디버그: 강제 첫 뷰만 preflight하고, NBV 뷰를 이 개수만큼 찍고 끝낸다.
    imsi_preflight_only_forced = _env_flag(
        "BATTERY_IMSI_PREFLIGHT_ONLY_FORCED", False
    )
    imsi_max_nbv_views = max(0, int(os.environ.get(
        "BATTERY_IMSI_MAX_NBV_VIEWS", "0"
    )))
    # 가까운 후보의 Beam edge cost를 (lift 거리 / 후보 거리)^2로 나눈다.
    imsi_proximity_weight_max = max(1.0, float(os.environ.get(
        "BATTERY_IMSI_PROXIMITY_WEIGHT_MAX", "1.0"
    )))
    # 해상도 기준 quality coverage: surfel이 (d_ref/d)^2 cos(입사각)
    # >= min density로 보인 적이 있어야 한다. 면(+Z, +/-X, +/-Y)별 목표.
    imsi_quality_enabled = _env_flag("BATTERY_IMSI_QUALITY_COVERAGE", False)
    imsi_quality_groups = [
        name.strip()
        for name in os.environ.get(
            "BATTERY_IMSI_QUALITY_GROUPS", "+Z,+X,-X,+Y,-Y"
        ).split(",")
        if name.strip() in QUALITY_FACE_NORMALS
    ]
    imsi_quality_target = min(1.0, max(0.0, float(os.environ.get(
        "BATTERY_IMSI_QUALITY_COVERAGE_TARGET", "0.50"
    ))))
    imsi_quality_min_density = max(0.0, float(os.environ.get(
        "BATTERY_IMSI_QUALITY_MIN_DENSITY", "0.50"
    )))
    imsi_quality_reference_distance_m = max(1e-3, float(os.environ.get(
        "BATTERY_IMSI_QUALITY_REFERENCE_DISTANCE_M", "0.40"
    )))
    imsi_quality_max_views = max(0, int(os.environ.get(
        "BATTERY_IMSI_QUALITY_MAX_VIEWS", "4"
    )))
    imsi_quality_min_gain = max(0.0, float(os.environ.get(
        "BATTERY_IMSI_QUALITY_MIN_GAIN", "0.005"
    )))
    # quality density는 거리/입사각 기준이다. zoom으로 fx가 커지면 같은
    # 거리에서도 픽셀이 (fx/fx_ref)^2 배가 되므로 임계값을 그만큼 낮춘다.
    imsi_quality_reference_fx_px = max(1.0, float(os.environ.get(
        "BATTERY_IMSI_QUALITY_REFERENCE_FX_PX", "2120.0"
    )))
    imsi_quality_fx_scale = (
        float(intrinsic_matrix[0, 0]) / imsi_quality_reference_fx_px
    ) ** 2
    imsi_quality_min_density = imsi_quality_min_density / imsi_quality_fx_scale
    if imsi_top_side_only_termination:
        top_side_target = min(
            1.0,
            max(
                0.0,
                float(
                    os.environ.get(
                        "BATTERY_IMSI_TOP_SIDE_COVERAGE_TARGET",
                        "0.90",
                    )
                ),
            ),
        )
        # 호출자가 legacy bottom/overall 목표를 넘겨도 IMSI의 성공
        # 종료는 top/side 실측 coverage가 모두 이 값에 도달했을 때다.
        # Route 목표는 이와 독립적인 UNKNOWN voxel entropy로 계산한다.
        nbv_surface_coverage_target = 0.0
        nbv_top_surface_coverage_target = top_side_target
        nbv_side_surface_coverage_target = top_side_target
        nbv_bottom_surface_coverage_target = 0.0
        nbv_surface_bottom_score_weight = 0.0

    print(
        log_prefix,
        "Starting arm-driven NBV capture sequence (t=0, 물체 들어올림 직후)",
    )
    print(
        log_prefix,
        "방향성 표면 추적/legacy 점수 가중치:",
        f"top(+Z)={nbv_surface_top_score_weight:.2f},",
        f"side={nbv_surface_side_score_weight:.2f},",
        f"bottom(-Z)={nbv_surface_bottom_score_weight:.2f};",
        "종료 목표:",
        f"전체={nbv_surface_coverage_target * 100:.0f}%,",
        f"top={nbv_top_surface_coverage_target * 100:.0f}%,",
        f"side={nbv_side_surface_coverage_target * 100:.0f}%,",
        f"bottom={nbv_bottom_surface_coverage_target * 100:.0f}%,",
        "판정=실측 top/side 동시 달성"
        if imsi_top_side_only_termination
        else "판정=활성화된 전체 방향 목표 동시 달성",
    )
    if precomputed_joint_view_limit:
        imsi_trajectory_top_k = max(
            1,
            int(
                os.environ.get(
                    "BATTERY_IMSI_TRAJECTORY_TOP_K",
                    "15",
                )
            ),
        )
        imsi_trajectory_min_score_ratio = min(
            1.0,
            max(
                0.0,
                float(
                    os.environ.get(
                        "BATTERY_IMSI_TRAJECTORY_MIN_SCORE_RATIO",
                        "0.20",
                    )
                ),
            ),
        )
        imsi_cycle_fixed_overhead_s = max(
            0.0,
            float(
                os.environ.get(
                    "BATTERY_IMSI_CYCLE_FIXED_OVERHEAD_S",
                    "0.95",
                )
            ),
        )
        imsi_joint_path_penalty_s_per_rad = max(
            0.0,
            float(
                os.environ.get(
                    "BATTERY_IMSI_JOINT_PATH_PENALTY_S_PER_RAD",
                    "0.35",
                )
            ),
        )
        # max(|dq|/vmax)에 곱하는 보정값. 1.0이면 기존 추정과 같다.
        imsi_joint_time_scale = max(
            1e-3,
            float(os.environ.get("BATTERY_IMSI_JOINT_TIME_SCALE", "1.0")),
        )
        imsi_min_view_direction_local_z = min(
            1.0,
            max(
                -1.0,
                float(
                    os.environ.get(
                        "BATTERY_IMSI_MIN_VIEW_DIRECTION_LOCAL_Z",
                        "-1.0",
                    )
                ),
            ),
        )
        imsi_route_planner_enabled = os.environ.get(
            "BATTERY_IMSI_ROUTE_PLANNER",
            "1",
        ).strip().lower() not in {"0", "false", "no", "off"}
        imsi_route_beam_width = max(
            1,
            int(os.environ.get("BATTERY_IMSI_ROUTE_BEAM_WIDTH", "512")),
        )
        imsi_route_branch_width = max(
            1,
            int(os.environ.get("BATTERY_IMSI_ROUTE_BRANCH_WIDTH", "15")),
        )
        imsi_runtime_route_beam_width = max(
            1,
            int(
                os.environ.get(
                    "BATTERY_IMSI_RUNTIME_ROUTE_BEAM_WIDTH",
                    "128",
                )
            ),
        )
        imsi_runtime_route_branch_width = max(
            1,
            int(
                os.environ.get(
                    "BATTERY_IMSI_RUNTIME_ROUTE_BRANCH_WIDTH",
                    "10",
                )
            ),
        )
        imsi_route_reachable_target_ratio = min(
            1.0,
            max(
                0.0,
                float(
                    os.environ.get(
                        "BATTERY_IMSI_ROUTE_REACHABLE_TARGET_RATIO",
                        "0.98",
                    )
                ),
            ),
        )
        imsi_route_commit_views = max(
            1,
            int(os.environ.get("BATTERY_IMSI_ROUTE_COMMIT_VIEWS", "2")),
        )
        imsi_top_view_max_weight = max(
            1.0,
            float(os.environ.get("BATTERY_IMSI_TOP_VIEW_MAX_WEIGHT", "3.0")),
        )
        imsi_finish_top_focus = _env_flag("BATTERY_IMSI_FINISH_TOP_FOCUS", True)
        imsi_finish_top_min_local_z = min(1.0, max(0.0, float(
            os.environ.get("BATTERY_IMSI_FINISH_TOP_MIN_LOCAL_Z", "0.15")
        )))
        imsi_route_max_deferrals = max(
            1,
            int(os.environ.get("BATTERY_IMSI_ROUTE_MAX_DEFERRALS", "2")),
        )
        imsi_joint_tolerance_rad = max(
            0.0,
            float(
                os.environ.get(
                    "BATTERY_IMSI_JOINT_TOLERANCE_RAD",
                    "0.05",
                )
            ),
        )
        imsi_final_pose_timeout_s = max(
            0.0,
            float(
                os.environ.get(
                    "BATTERY_IMSI_FINAL_POSE_TIMEOUT_S",
                    "3.0",
                )
            ),
        )
        imsi_final_pose_stable_updates = max(
            1,
            int(
                os.environ.get(
                    "BATTERY_IMSI_FINAL_POSE_STABLE_UPDATES",
                    "3",
                )
            ),
        )
        imsi_empirical_visibility_enabled = _env_flag(
            "BATTERY_IMSI_EMPIRICAL_VISIBILITY_CACHE",
            default=True,
        )
        imsi_empirical_minimum_hit_ratio = min(
            1.0,
            max(
                0.0,
                float(
                    os.environ.get(
                        "BATTERY_IMSI_EMPIRICAL_MIN_HIT_RATIO",
                        "0.50",
                    )
                ),
            ),
        )
        imsi_empirical_bootstrap_min_cached_poses = max(
            0,
            int(
                os.environ.get(
                    "BATTERY_IMSI_EMPIRICAL_BOOTSTRAP_MIN_CACHED_POSES",
                    "6",
                )
            ),
        )
        imsi_empirical_bootstrap_pose_order = [
            name.strip()
            for name in os.environ.get(
                "BATTERY_IMSI_EMPIRICAL_BOOTSTRAP_POSE_ORDER",
                "",
            ).split(",")
            if name.strip()
        ]
        print(
            log_prefix,
            "[IMSI] joint preflight mode enabled:",
            f"successful views limit={precomputed_joint_view_limit},",
            "legacy endpoint acceptance:",
            f"joint<={imsi_joint_tolerance_rad:.3f}rad,",
            "actual object in-frame, gripped,",
            "TCP/object pose error=diagnostic only,",
            f"trajectory optimize=information top {imsi_trajectory_top_k},",
            f"score>={imsi_trajectory_min_score_ratio * 100:.0f}% of best,",
            "selection=max information/motion-adjusted cost,",
            f"fixed overhead={imsi_cycle_fixed_overhead_s:.2f}s,",
            "joint-path penalty="
            f"{imsi_joint_path_penalty_s_per_rad:.2f}s/rad,",
            f"joint-time scale={imsi_joint_time_scale:.3f},",
            f"route order=min-joint-time(Held-Karp)={imsi_route_order_tsp},",
            "route replan min gain="
            f"{imsi_route_replan_min_gain * 100:.2f}%p,",
            "inspection lift: eligible<"
            f"{imsi_inspection_min_eligible} or gain<"
            f"{imsi_inspection_fallback_min_gain * 100:.2f}%p,",
            "approach distances="
            f"{','.join(f'{d:.3f}' for d in imsi_approach_distances_m) or 'off'},",
            f"top close-up={imsi_top_closeup_enabled}"
            f"(tilts={','.join(f'{t:.0f}' for t in imsi_top_closeup_tilts_deg)}),",
            f"proximity weight max={imsi_proximity_weight_max:.2f},",
            f"quality coverage={imsi_quality_enabled}"
            f"(faces={','.join(imsi_quality_groups)}, "
            f"target>={imsi_quality_target * 100:.0f}%, "
            f"density>={imsi_quality_min_density:.2f}@"
            f"{imsi_quality_reference_distance_m:.2f}m, "
            f"max views={imsi_quality_max_views}),",
            "minimum local view Z="
            f"{imsi_min_view_direction_local_z:.2f},",
            f"route planner={imsi_route_planner_enabled},",
            "beam="
            f"{imsi_route_beam_width}/{imsi_route_branch_width},",
            "runtime beam="
            f"{imsi_runtime_route_beam_width}/"
            f"{imsi_runtime_route_branch_width},",
            "route commit/defer="
            f"{imsi_route_commit_views}/{imsi_route_max_deferrals},",
            "reachable target ratio="
            f"{imsi_route_reachable_target_ratio * 100:.0f}%,",
            "empirical visibility="
            f"{imsi_empirical_visibility_enabled},",
            "empirical votes>="
            f"{imsi_empirical_minimum_hit_ratio * 100:.0f}%,",
            "inspection focus="
            f"{imsi_inspection_focus},",
            "distinct view>="
            f"{imsi_inspection_min_view_separation_deg:.0f}deg,",
            "top local-Z>="
            f"{imsi_inspection_top_min_local_z:.2f},",
            "side local-Z="
            f"[{imsi_inspection_side_min_local_z:.2f},"
            f"{imsi_inspection_side_max_local_z:.2f}],",
            "azimuth band="
            f"{imsi_inspection_azimuth_band_deg:.0f}deg,",
            f"timeout={imsi_final_pose_timeout_s:.1f}s,",
            f"stable={imsi_final_pose_stable_updates} updates",
        )

    # MoveIt's static world->link0 TF is identity (see the
    # static_transform_publisher in moveit.launch.py), but the robot is
    # actually mounted with a translation offset in the Isaac stage (stand
    # height plus a small Battery_env parent offset). A goal pose sent to
    # MoveGroup must be expressed relative to the robot base, not Isaac's
    # raw stage world - otherwise every candidate silently targets a
    # position shifted by that mount offset and MoveIt reports every
    # candidate unreachable regardless of the requested rotation.
    robot_base_relative_pos, robot_base_relative_quat = compute_relative_pose(
        stage=stage,
        child_prim_path=tcp_prim_path,
        parent_prim_path=robot_base_prim_path,
    )
    reference_tcp_position = np.array(
        [
            robot_base_relative_pos[0],
            robot_base_relative_pos[1],
            robot_base_relative_pos[2],
        ],
        dtype=np.float64,
    )
    initial_joint_positions = read_joint_positions(rb_robot)
    initial_tcp_position = [
        float(robot_base_relative_pos[0]),
        float(robot_base_relative_pos[1]),
        float(robot_base_relative_pos[2]),
    ]
    initial_tcp_orientation_xyzw = quat_to_xyzw_list(
        robot_base_relative_quat
    )

    grasp_offset_pos, grasp_offset_quat = measure_grasp_offset(
        stage=stage,
        object_prim_path=target_prim_path,
        tcp_prim_path=tcp_prim_path,
    )
    grasp_offset_np = matrix_np_from_pos_quat(
        grasp_offset_pos,
        grasp_offset_quat,
    )

    robot_base_world_matrix = matrix_np_from_gf(
        omni.usd.get_world_transform_matrix(
            stage.GetPrimAtPath(robot_base_prim_path)
        )
    )
    initial_object_world_matrix = matrix_np_from_gf(
        omni.usd.get_world_transform_matrix(target_prim)
    )
    initial_object_base_matrix = (
        np.linalg.inv(robot_base_world_matrix)
        @ initial_object_world_matrix
    )
    initial_object_rotation_base_np = initial_object_base_matrix[:3, :3]

    initial_pose_name, _initial_rotation = capture_poses[0]

    camera_world_matrix, _ = capture_current_zivid_frame(
        frame_id=0,
        pose_name=initial_pose_name,
        camera_prim=camera_prim,
        object_prim=target_prim,
        zivid_render_product=zivid_render_product,
        zivid_rgb_annotator=zivid_rgb_annotator,
        zivid_depth_annotator=zivid_depth_annotator,
        zivid_semantic_annotator=zivid_semantic_annotator,
        zivid_instance_annotator=zivid_instance_annotator,
        zivid_bbox_annotator=zivid_bbox_annotator,
        orchestrator=orchestrator,
        zivid_save_dir=zivid_save_dir,
        intrinsic_matrix=intrinsic_matrix,
        pause_timeline=True,
        voxel_size=object_grid_leaf_size,
        object_grid_root_size=object_grid_root_size,
        object_grid_origin=object_grid_origin,
    )
    accumulate_render_time()

    # capture_current_zivid_frame() captures via Replicator's
    # orchestrator.step(pause_timeline=True), which pauses the Kit timeline
    # as a side effect (for a clean, motion-free capture) and leaves it
    # paused. Nothing else resumes it, so without this, physics never ticks
    # again and every subsequent MoveGroup-commanded trajectory streams to
    # Isaac Sim with no effect - the arm silently never moves.
    omni.timeline.get_timeline_interface().play()

    occupancy_map_dir = os.path.join(zivid_save_dir, "object_occupancy")
    occupancy_latest_path = os.path.join(
        occupancy_map_dir, "object_occupancy_latest.npz"
    )
    nbv_manifest_path = os.path.join(
        occupancy_map_dir, "nbv_sequence_manifest.json"
    )
    max_capture_count = min(int(nbv_max_capture_count), len(capture_poses))

    sequence_started_at = datetime.now().isoformat()
    frame_records = [
        make_nbv_frame_record(
            frame_id=0,
            pose_name=initial_pose_name,
            rotation_xyz_deg=(0.0, 0.0, 0.0),
            occupancy_snapshot_path=os.path.join(
                occupancy_map_dir, "object_occupancy_000000.npz"
            ),
            joint_positions=initial_joint_positions,
            tcp_position=initial_tcp_position,
            tcp_orientation_xyzw=initial_tcp_orientation_xyzw,
            elapsed_s=elapsed_pair()[0],
            elapsed_excl_render_s=elapsed_pair()[1],
        )
    ]
    save_nbv_sequence_manifest(
        manifest_path=nbv_manifest_path,
        frame_records=frame_records,
        status="running",
        started_at=sequence_started_at,
        max_capture_count=max_capture_count,
    )

    remaining_pose_rotations = {
        pose_name: rotation_xyz_deg
        for pose_name, rotation_xyz_deg in capture_poses[1:]
    }

    # 방향성 표면을 분모로 한 커버리지 판정 준비. 10 mm occupancy
    # voxel은 얇은 batterycap의 윗면/아랫면을 한 칸으로 합치므로,
    # 검사 여부는 위치+법선을 가진 surfel로 따로 추적한다.
    surface_model = build_directional_surface_model(
        stage=stage,
        target_prim_path=target_prim_path,
        sample_count=nbv_surfel_sample_count,
        log_prefix=log_prefix,
    )
    initial_coverage_result, _initial_visible_mask = (
        update_directional_surface_from_last_capture(
        surface_model=surface_model,
        frame_id=0,
        occupancy_map_dir=occupancy_map_dir,
        match_distance=nbv_surfel_match_distance,
        min_facing_cos=nbv_surfel_min_facing_cos,
        log_prefix=log_prefix,
        save_debug_artifacts=save_debug_artifacts,
        )
    )
    imsi_quality_enabled = (
        imsi_quality_enabled
        and precomputed_joint_view_limit > 0
        and bool(imsi_quality_groups)
        and nbv_view_reference_local_point is not None
    )
    # Faces left with no side surfel (+/-Y) are not quality targets.
    quality_group_masks = {
        name: mask
        for name, mask in quality_face_group_masks(
            surface_model["normals"],
            imsi_quality_groups,
            surface_model.get("coverage_excluded"),
        ).items()
        if np.any(mask)
    }
    quality_exhausted_groups = set()
    quality_views_taken = 0
    quality_phase_active = False
    mandatory_finish_active = False
    side_face_masks = quality_face_group_masks(
        surface_model["normals"],
        ["+X", "-X", "+Y", "-Y"],
        surface_model.get("coverage_excluded"),
    )
    face_failed = set()

    def quality_report(label):
        fractions = quality_coverage_fractions(
            surface_model, quality_group_masks
        )
        print(
            log_prefix,
            f"[IMSI][QUALITY] {label}:",
            ", ".join(
                f"{name}={fraction * 100:.1f}%"
                + ("(exhausted)" if name in quality_exhausted_groups else "")
                for name, fraction in fractions.items()
            ),
            f"(density=(d_ref/d)^2*cos>={imsi_quality_min_density:.2f},",
            f"d_ref={imsi_quality_reference_distance_m:.2f}m,",
            f"target>={imsi_quality_target * 100:.0f}%)",
        )
        return fractions

    def quality_deficits():
        fractions = quality_coverage_fractions(
            surface_model, quality_group_masks
        )
        return {
            name: imsi_quality_target - fraction
            for name, fraction in fractions.items()
            if fraction + 1e-12 < imsi_quality_target
            and name not in quality_exhausted_groups
        }

    if imsi_quality_enabled:
        update_quality_observations(
            surface_model,
            _initial_visible_mask,
            surface_model["last_sensor_origin_object"],
            imsi_quality_reference_distance_m,
            imsi_quality_min_density,
        )
        quality_report("frame 0")

    frontier_map = None
    frontier_state = {
        "visible": {}, "frontier": None, "frame": -1,
        "initial": None, "best": None, "best_name": None, "done": False,
        "last_resolved": None, "stats": None,
    }

    def frontier_threshold():
        return max(
            float(imsi_frontier_stop_min_voxels),
            imsi_frontier_stop_ratio * float(frontier_state["initial"] or 0),
        )

    def frontier_integrate(frame_id):
        observation = _last_surface_observation()
        if int(observation.get("frame_id", -1)) != int(frame_id) or "depth" not in observation:
            print(log_prefix, "[IMSI][FRONTIER] WARNING: frame", frame_id,
                  "의 depth 관측이 없어 frontier map을 갱신하지 않습니다.")
            return None
        before = frontier_state["stats"]
        started = time.monotonic()
        info = frontier_map.integrate_depth(
            observation["depth"],
            observation["intrinsic"],
            observation["object_from_camera_cv"],
            observation.get("part_pixel_mask"),
        )
        stats = frontier_map.stats()
        frontier_state["stats"] = stats
        if frontier_state["initial"] is None:
            frontier_state["initial"] = stats["frontier"]
        print(
            log_prefix, f"[IMSI][FRONTIER] frame {frame_id}: rays={info['rays']},",
            f"part voxels={stats['part_occupied']}, UNKNOWN={stats['unknown']},",
            f"frontier={stats['frontier']}"
            + (f" ({stats['frontier'] - before['frontier']:+d})" if before else "")
            + (
                f", underside excluded={stats['frontier_down_excluded']}"
                if imsi_frontier_exclude_down_cos is not None else ""
            ),
            f"[{time.monotonic() - started:.2f}s]",
        )
        return stats

    def frontier_visible_for(names):
        frontier_flat = frontier_map.frontier_indices()
        visible = {}
        for name in names:
            object_world = candidate_object_world_matrices.get(name)
            if object_world is None:
                continue
            visible[name] = frontier_map.visible_frontier(
                frontier_flat,
                np.linalg.inv(object_world) @ camera_world_matrix,
                intrinsic_matrix,
                image_size,
            )
        frontier_state.update(frontier=frontier_flat, visible=visible, frame=len(frame_records))
        return frontier_flat, visible

    if imsi_frontier_enabled and precomputed_joint_view_limit:
        frontier_tools = _load_frontier_tools()
        frontier_map = frontier_tools.FrontierMap(
            np.asarray(nbv_roi_min, dtype=np.float64) - imsi_frontier_margin_m,
            np.asarray(nbv_roi_max, dtype=np.float64) + imsi_frontier_margin_m,
            leaf_size=imsi_frontier_leaf_m,
            pixel_stride=imsi_frontier_pixel_stride,
            exclude_down_cos=imsi_frontier_exclude_down_cos,
        )
        print(
            log_prefix, "[IMSI][FRONTIER] mesh-free 결정 모드: grid",
            frontier_map.shape, f"leaf={imsi_frontier_leaf_m * 1000:.1f}mm,",
            f"pixel stride={imsi_frontier_pixel_stride},",
            f"stop: 남은 후보 최대 가시 frontier < max({imsi_frontier_stop_min_voxels},",
            f"{imsi_frontier_stop_ratio * 100:.0f}% x frame-0 frontier)",
            "(GT coverage는 보고/ACCEPTANCE에만 사용),",
            "underside frontier 제외="
            + (
                f"-Z {math.degrees(math.acos(min(1.0, max(-1.0, imsi_frontier_exclude_down_cos)))):.0f}deg 이내"
                if imsi_frontier_exclude_down_cos is not None else "off"
            ) + ",",
            f"후보 시점 local view Z >= {imsi_min_view_direction_local_z:.2f}",
        )
        frontier_integrate(0)
    initial_coverage_result = report_directional_surface_coverage(
        surface_model,
        elapsed_s=elapsed_pair()[0],
        elapsed_excl_render_s=elapsed_pair()[1],
        log_prefix=log_prefix,
    )
    previous_coverage_fraction = float(
        initial_coverage_result["fraction"]
    )
    previous_top_coverage_fraction = float(
        initial_coverage_result["groups"]["top(+Z)"]["fraction"]
    )
    previous_side_coverage_fraction = float(
        initial_coverage_result["groups"]["side"]["fraction"]
    )
    previous_bottom_coverage_fraction = float(
        initial_coverage_result["groups"]["bottom(-Z)"]["fraction"]
    )
    consecutive_low_coverage_gains = 0

    if imsi_empirical_visibility_enabled:
        empirical_cache_path = os.environ.get(
            "BATTERY_IMSI_EMPIRICAL_VISIBILITY_CACHE_PATH",
            os.path.join(zivid_save_dir, "nbv_empirical_visibility.npz"),
        )
        empirical_visibility_cache = load_empirical_visibility_cache(
            path=empirical_cache_path,
            surface_model=surface_model,
            log_prefix=log_prefix,
        )

    gain_goal_names = []
    if nbv_surface_coverage_target > 0.0:
        gain_goal_names.append("전체")
    if nbv_top_surface_coverage_target > 0.0:
        gain_goal_names.append("top")
    if nbv_side_surface_coverage_target > 0.0:
        gain_goal_names.append("side")
    if nbv_bottom_surface_coverage_target > 0.0:
        gain_goal_names.append("bottom")
    if not gain_goal_names:
        gain_goal_names.append("전체")

    def surface_targets_reached(coverage):
        """실측 directional coverage가 현재 검사의 완료 목표인지 판정."""
        top_reached = (
            coverage["groups"]["top(+Z)"]["fraction"]
            >= nbv_top_surface_coverage_target
        )
        side_reached = (
            coverage["groups"]["side"]["fraction"]
            >= nbv_side_surface_coverage_target
        )
        if imsi_top_side_only_termination:
            return (
                coverage["groups"]["top(+Z)"]["fraction"]
                > nbv_top_surface_coverage_target
                and coverage["groups"]["side"]["fraction"]
                > nbv_side_surface_coverage_target
            )

        return (
            coverage["fraction"] >= nbv_surface_coverage_target
            and top_reached
            and side_reached
            and coverage["groups"]["bottom(-Z)"]["fraction"]
            >= nbv_bottom_surface_coverage_target
        )

    if nbv_min_surface_coverage_gain > 0.0:
        print(
            log_prefix,
            "표면 커버리지 증가 진단:"
            if imsi_top_side_only_termination
            else "표면 커버리지 증가 종료 조건:",
            "/".join(gain_goal_names),
            "목표의 직전 촬영 대비 증가가 모두",
            f"< {nbv_min_surface_coverage_gain * 100:.2f}%p인 상태가",
            f"{nbv_min_surface_coverage_gain_consecutive}회 연속",
            "(IMSI는 목표 미달 시 계속)"
            if imsi_top_side_only_termination
            else "",
        )

    if surface_targets_reached(initial_coverage_result):
        if imsi_top_side_only_termination:
            print(
                f"{log_prefix} 첫 뷰만으로 top/side 목표 "
                f"{nbv_top_surface_coverage_target * 100:.0f}%/"
                f"{nbv_side_surface_coverage_target * 100:.0f}% "
                "도달 - 종료"
            )
        else:
            print(
                f"{log_prefix} 첫 뷰만으로 목표 "
                f"{nbv_surface_coverage_target * 100:.0f}% 도달 - 종료"
            )
        save_nbv_sequence_manifest(
            manifest_path=nbv_manifest_path,
            frame_records=frame_records,
            status="coverage_reached",
            started_at=sequence_started_at,
            max_capture_count=max_capture_count,
        )
        save_directional_surface_model(
            surface_model=surface_model,
            occupancy_map_dir=occupancy_map_dir,
            frame_id=0,
            save_frame_snapshot=save_debug_artifacts,
        )

        return frame_records

    # 이미 관측한 시선 방향들(object frame). 새 후보가 이것들과 충분히
    # 떨어져 있어야 새 표면이 드러난다.
    # Candidate framing is checked against this, see project_object_framing.
    try:
        framing_corners_local = object_local_bbox_corners(
            stage,
            target_prim_path,
            log_prefix,
        )
    except Exception as exc:
        framing_corners_local = None
        print(
            log_prefix,
            "ERROR: object bbox를 못 구해 경로 화각 검사를 시작할 수 "
            "없습니다:",
            f"{type(exc).__name__}: {exc}",
        )

    if framing_corners_local is None:
        save_nbv_sequence_manifest(
            manifest_path=nbv_manifest_path,
            frame_records=frame_records,
            status="frustum_guard_unavailable",
            started_at=sequence_started_at,
            max_capture_count=max_capture_count,
        )
        return frame_records

    # MoveIt의 FK 결과와 목표 자세는 robot-base 좌표계(world==link0)다.
    # 카메라도 같은 좌표계로 변환해서 relay에 넘기면, relay가 계획된
    # joint trajectory의 최종 정지 자세에서 TCP->object 강체 변환을
    # 적용한 뒤 bbox를 센서에 직접 투영할 수 있다. 이동 중간 자세는
    # 촬영하지 않으므로 화각 밖으로 나가도 허용한다.
    camera_base_matrix = (
        np.linalg.inv(robot_base_world_matrix) @ camera_world_matrix
    )
    move_client.configure_frustum_guard(
        camera_matrix=camera_base_matrix,
        intrinsic_matrix=intrinsic_matrix,
        image_size=image_size,
        corners_local=framing_corners_local,
        grasp_offset=grasp_offset_np,
        margin_px=nbv_frame_margin_px,
        min_in_frame_fraction=nbv_min_in_frame_fraction,
    )
    print(
        log_prefix,
        "MoveIt 최종 정지 자세 화각 guard 설정:",
        f"bbox corners={len(framing_corners_local)},",
        f"margin={nbv_frame_margin_px:.0f}px,",
        f"required={nbv_min_in_frame_fraction * 100:.0f}%",
    )

    captured_view_directions = [
        camera_direction_in_object_frame(
            matrix_np_from_gf(
                omni.usd.get_world_transform_matrix(target_prim)
            ),
            camera_world_matrix,
            reference_local_point=nbv_view_reference_local_point,
        )
    ]

    status = "completed"

    # 모든 가상 카메라/물체 자세는 최초로 들어 올린 물체 자세와 grasp
    # offset 기준의 절대 자세다. 변하지 않는 후보 행렬은 한 번만 만든다.
    # MoveIt 검사는 여기서 미리 하지 않고, NBV가 선택한 후보만 실제
    # 이동 직전에 계획한다.
    candidate_tcp_matrices = {}
    candidate_object_world_matrices = {}
    # IMSI preflight tries each (distance, rotation) variant in order and
    # keeps the first that passes; the lift distance is always the last one.
    candidate_variants = {}
    candidate_pivot_distance_m = {}
    candidate_proximity_weights = {}
    camera_base_position = camera_base_matrix[:3, 3]
    lift_pivot_distance_m = None
    lift_pivot_base = None
    if nbv_view_reference_local_point is not None:
        lift_pivot_base = (
            initial_object_base_matrix[:3, :3]
            @ np.asarray(nbv_view_reference_local_point, dtype=np.float64)
            + initial_object_base_matrix[:3, 3]
        )
        if precomputed_joint_view_limit and np.any(
            imsi_workspace_offset_world_m
        ):
            lift_pivot_world_before = (
                robot_base_world_matrix[:3, :3] @ lift_pivot_base
                + robot_base_world_matrix[:3, 3]
            )
            lift_pivot_base = lift_pivot_base + (
                robot_base_world_matrix[:3, :3].T
                @ imsi_workspace_offset_world_m
            )
            print(
                log_prefix,
                "[IMSI][WORKSPACE] 작업점 이동(world):",
                np.round(imsi_workspace_offset_world_m, 4).tolist(),
                "m, pivot world",
                np.round(lift_pivot_world_before, 4).tolist(),
                "->",
                np.round(
                    robot_base_world_matrix[:3, :3] @ lift_pivot_base
                    + robot_base_world_matrix[:3, 3],
                    4,
                ).tolist(),
                ", camera 거리",
                f"{np.linalg.norm(lift_pivot_world_before - camera_world_matrix[:3, 3]):.3f}"
                "->"
                f"{np.linalg.norm(lift_pivot_base - camera_base_position):.3f}m",
            )
        lift_pivot_distance_m = float(
            np.linalg.norm(lift_pivot_base - camera_base_position)
        )
    approach_distances_m = (
        [
            distance
            for distance in imsi_approach_distances_m
            if distance < lift_pivot_distance_m - 1e-3
        ]
        if precomputed_joint_view_limit and lift_pivot_distance_m is not None
        else []
    )

    def pivot_at_distance(distance):
        if distance is None:
            # The (possibly offset) work point; None keeps the lift pivot.
            return lift_pivot_base
        return camera_base_position + distance * (
            (lift_pivot_base - camera_base_position) / lift_pivot_distance_m
        )

    def build_variant(
        rotation_xyz_deg,
        distance,
        delta_rotation_np=None,
        pivot_base=None,
    ):
        """``pivot_base`` places the geometry centre explicitly (frontal
        close-up on the optical axis); otherwise it moves along the ray."""
        tcp_matrix_np, predicted_object_base_np = (
            solve_tcp_pose_for_candidate(
                initial_object_rotation_base_np=(
                    initial_object_rotation_base_np
                ),
                grasp_offset_np=grasp_offset_np,
                reference_tcp_position=reference_tcp_position,
                rotation_xyz_deg=rotation_xyz_deg,
                initial_object_base_matrix=initial_object_base_matrix,
                object_pivot_local_point=nbv_view_reference_local_point,
                delta_rotation_np=delta_rotation_np,
                pivot_base_override=(
                    pivot_at_distance(distance)
                    if pivot_base is None
                    else pivot_base
                ),
            )
        )
        return {
            "rotation_xyz_deg": tuple(rotation_xyz_deg),
            "distance_m": (
                lift_pivot_distance_m if distance is None else distance
            ),
            "tcp": tcp_matrix_np,
            "object_world": robot_base_world_matrix @ predicted_object_base_np,
        }

    for pose_name, rotation_xyz_deg in remaining_pose_rotations.items():
        variants = [
            build_variant(rotation_xyz_deg, distance)
            for distance in approach_distances_m + [None]
        ]
        candidate_variants[pose_name] = variants
        candidate_tcp_matrices[pose_name] = variants[-1]["tcp"]
        candidate_object_world_matrices[pose_name] = (
            variants[-1]["object_world"]
        )
        candidate_pivot_distance_m[pose_name] = variants[-1]["distance_m"]

    forced_first_view_name = None
    if (
        imsi_top_closeup_enabled
        and precomputed_joint_view_limit
        and lift_pivot_base is not None
        and (imsi_top_closeup_tilts_deg or imsi_top_closeup_mode == "frontal")
    ):
        # tilt: turn the object's +Z toward the camera about the geometry
        # centre. frontal: +Z exactly against the optical axis, centre on
        # the axis. Preflight order = predicted top pixel density, best first.
        top_normal_base = initial_object_rotation_base_np[:, 2]
        camera_forward_base = -camera_base_matrix[:3, 2]
        closeup_specs = []
        if imsi_top_closeup_mode == "frontal":
            frontal_distances = (
                imsi_top_closeup_frontal_distances_m
                or approach_distances_m + [lift_pivot_distance_m]
            )
            for roll_deg in imsi_top_closeup_rolls_deg:
                delta_rotation = frontal_top_delta_rotation(
                    top_normal_base, camera_forward_base, roll_deg
                )
                for distance in frontal_distances:
                    closeup_specs.append((
                        f"frontal roll {roll_deg:.0f}deg", roll_deg,
                        delta_rotation, distance,
                        camera_base_position + distance * camera_forward_base,
                    ))
        else:
            for tilt_deg in imsi_top_closeup_tilts_deg:
                delta_rotation = rotation_toward(
                    top_normal_base,
                    camera_base_position - lift_pivot_base,
                    tilt_deg,
                )
                for distance in approach_distances_m + [None]:
                    closeup_specs.append((
                        f"tilt {tilt_deg:.0f}deg", tilt_deg,
                        delta_rotation, distance, None,
                    ))
        closeup_variants = []
        for label, tilt_deg, delta_rotation, distance, pivot_base in (
            closeup_specs
        ):
            rotation_xyz_deg = rotation_xyz_deg_from_matrix(
                delta_rotation
            )
            variant = build_variant(
                rotation_xyz_deg, distance, delta_rotation, pivot_base
            )
            variant["label"] = label
            pivot_world = (
                variant["object_world"][:3, :3]
                @ np.asarray(nbv_view_reference_local_point)
                + variant["object_world"][:3, 3]
            )
            variant["predicted_top_density"] = predicted_face_density(
                variant["object_world"][:3, :3].T
                @ (camera_world_matrix[:3, 3] - pivot_world),
                variant["distance_m"],
                imsi_quality_reference_distance_m,
                QUALITY_FACE_NORMALS["+Z"],
            )
            variant["tilt_deg"] = tilt_deg
            closeup_variants.append(variant)
        closeup_variants.sort(
            key=lambda variant: -variant["predicted_top_density"]
        )
        forced_first_view_name = "top_closeup"
        remaining_pose_rotations = {
            forced_first_view_name: closeup_variants[0]["rotation_xyz_deg"],
            **remaining_pose_rotations,
        }
        candidate_variants[forced_first_view_name] = closeup_variants
        candidate_tcp_matrices[forced_first_view_name] = (
            closeup_variants[0]["tcp"]
        )
        candidate_object_world_matrices[forced_first_view_name] = (
            closeup_variants[0]["object_world"]
        )
        candidate_pivot_distance_m[forced_first_view_name] = (
            closeup_variants[0]["distance_m"]
        )
        print(
            log_prefix,
            "[IMSI][TOP-CLOSEUP] 후보",
            f"{len(closeup_variants)}개 (mode={imsi_top_closeup_mode}), 예상 top density 순:",
            ", ".join(
                f"{v['label']}@{v['distance_m']:.3f}m="
                f"{v['predicted_top_density']:.2f}"
                for v in closeup_variants
            ),
        )
    direct_joint_targets = {}

    def register_direct_joint_pose(name, joints, tcp, log_label):
        """Hand-set pose: exact joints (no IK preflight) + its tf TCP."""
        qx, qy, qz, qw = tcp[3:7]
        tcp_np = matrix_np_from_pos_quat(
            tcp[:3],
            SimpleNamespace(GetReal=lambda: qw, GetImaginary=lambda: (qx, qy, qz)),
        )
        object_np = robot_base_world_matrix @ predict_object_matrix(
            tcp_np, grasp_offset_np
        )
        candidate_tcp_matrices[name] = tcp_np
        candidate_object_world_matrices[name] = object_np
        direct_joint_targets[name] = {
            "joint_names": list(imsi_first_pose_joint_names),
            "positions": [float(value) for value in joints],
            "planning_time_s": 0.0,
            "trajectory_duration_s": 0.0,
            "ik_time_s": 0.0,
            "joint_source": "user_joint_states",
        }
        in_frame, off_axis, _ = project_object_framing(
            object_np, camera_world_matrix, intrinsic_matrix, image_size,
            framing_corners_local, margin_px=nbv_frame_margin_px,
        )
        facing = ""
        if nbv_view_reference_local_point is not None:
            pivot = (
                object_np[:3, :3] @ np.asarray(nbv_view_reference_local_point)
                + object_np[:3, 3]
            )
            to_camera = camera_world_matrix[:3, 3] - pivot
            candidate_pivot_distance_m[name] = float(np.linalg.norm(to_camera))
            local = object_np[:3, :3].T @ (to_camera / np.linalg.norm(to_camera))
            face, normal = max(
                QUALITY_FACE_NORMALS.items(),
                key=lambda item: float(np.dot(local, item[1])),
            )
            facing = (
                f"정면 면={face} 입사각="
                f"{math.degrees(math.acos(float(np.clip(np.dot(local, normal), -1, 1)))):.1f}deg, "
                f"camera 거리={candidate_pivot_distance_m[name]:.3f}m,"
            )
        print(
            log_prefix, log_label, name,
            f"joints={[round(v, 4) for v in joints]},",
            f"tcp={[round(v, 4) for v in tcp]},",
            "예측:", facing,
            f"광축 이탈={off_axis:.1f}deg, in-frame={in_frame * 100:.0f}%",
        )
        return object_np

    if imsi_mandatory_poses and precomputed_joint_view_limit:
        mandatory_rotations = {}
        for pose in imsi_mandatory_poses:
            object_np = register_direct_joint_pose(
                pose["name"], pose["joints"], pose["tcp"],
                "[IMSI][MANDATORY] 필수 방문 node:",
            )
            mandatory_rotations[pose["name"]] = rotation_xyz_deg_from_matrix(
                (np.linalg.inv(robot_base_world_matrix) @ object_np)[:3, :3]
                @ initial_object_rotation_base_np.T
            )
        remaining_pose_rotations = {
            **mandatory_rotations,
            **remaining_pose_rotations,
        }

    user_first_pose_target = None
    if imsi_first_pose_enabled and precomputed_joint_view_limit:
        # A pose the user set by hand (read from /joint_states and tf). The
        # joints are exact, so no IK preflight: the runtime move plans from
        # the current state to this joint goal with the same endpoint check.
        if forced_first_view_name is not None:
            remaining_pose_rotations.pop(forced_first_view_name, None)
            candidate_variants.pop(forced_first_view_name, None)
            candidate_tcp_matrices.pop(forced_first_view_name, None)
            candidate_object_world_matrices.pop(forced_first_view_name, None)
            candidate_pivot_distance_m.pop(forced_first_view_name, None)
        qx, qy, qz, qw = imsi_first_pose_tcp[3:7]
        tcp_matrix_np = matrix_np_from_pos_quat(
            imsi_first_pose_tcp[:3],
            SimpleNamespace(
                GetReal=lambda: qw,
                GetImaginary=lambda: (qx, qy, qz),
            ),
        )
        object_world_np = robot_base_world_matrix @ predict_object_matrix(
            tcp_matrix_np, grasp_offset_np
        )
        rotation_xyz_deg = rotation_xyz_deg_from_matrix(
            (np.linalg.inv(robot_base_world_matrix) @ object_world_np)[:3, :3]
            @ initial_object_rotation_base_np.T
        )
        forced_first_view_name = "user_first_pose"
        remaining_pose_rotations = {
            forced_first_view_name: rotation_xyz_deg,
            **remaining_pose_rotations,
        }
        candidate_tcp_matrices[forced_first_view_name] = tcp_matrix_np
        candidate_object_world_matrices[forced_first_view_name] = (
            object_world_np
        )
        user_first_pose_target = {
            "joint_names": list(imsi_first_pose_joint_names),
            "positions": [float(value) for value in imsi_first_pose_joints],
            "planning_time_s": 0.0,
            "trajectory_duration_s": 0.0,
            "ik_time_s": 0.0,
            "joint_source": "user_joint_states",
        }
        pivot_world = None
        if nbv_view_reference_local_point is not None:
            pivot_world = (
                object_world_np[:3, :3]
                @ np.asarray(nbv_view_reference_local_point)
                + object_world_np[:3, 3]
            )
            candidate_pivot_distance_m[forced_first_view_name] = float(
                np.linalg.norm(camera_world_matrix[:3, 3] - pivot_world)
            )
        in_frame, off_axis, _ = project_object_framing(
            object_world_np,
            camera_world_matrix,
            intrinsic_matrix,
            image_size,
            framing_corners_local,
            margin_px=nbv_frame_margin_px,
        )
        top_incidence = float("nan")
        if pivot_world is not None:
            to_camera = camera_world_matrix[:3, 3] - pivot_world
            top_incidence = math.degrees(math.acos(float(np.clip(
                object_world_np[:3, 2] @ to_camera
                / np.linalg.norm(to_camera), -1.0, 1.0,
            ))))
        print(
            log_prefix,
            "[IMSI][FIRST-POSE] 사용자 pose를 강제 첫 뷰로 사용:",
            f"joints={[round(v, 4) for v in imsi_first_pose_joints]},",
            f"tcp={[round(v, 4) for v in imsi_first_pose_tcp]},",
            "예측: camera 거리="
            f"{candidate_pivot_distance_m.get(forced_first_view_name, float('nan')):.3f}m,",
            f"top 입사각={top_incidence:.1f}deg,",
            f"광축 이탈={off_axis:.1f}deg,",
            f"in-frame={in_frame * 100:.0f}%",
        )
    if approach_distances_m:
        print(
            log_prefix,
            "[IMSI][APPROACH] lift pivot-camera 거리",
            f"{lift_pivot_distance_m:.3f}m, 먼저 시도할 거리=",
            ",".join(f"{d:.3f}" for d in approach_distances_m),
            "(실패 시 lift 거리로 fallback)",
        )

    precomputed_joint_targets = None
    precomputed_route_graph = None
    initial_route_plan = None
    initial_voxel_results = None
    voxel_ray_cache = RayGeometryCache()
    if precomputed_joint_view_limit:
        if not move_client.wait_until_preflight_ready():
            print(
                log_prefix,
                "[IMSI] relay의 preflight/joint topic을 찾지 못했습니다. "
                "갱신된 nbv_arm_move_relay.py를 재시작하세요.",
            )
            save_nbv_sequence_manifest(
                manifest_path=nbv_manifest_path,
                frame_records=frame_records,
                status="preflight_relay_unavailable",
                started_at=sequence_started_at,
                max_capture_count=max_capture_count,
            )
            return frame_records

        precomputed_joint_targets = {}
        preflight_attempts = []
        preflight_started_at = time.monotonic()
        print(
            log_prefix,
            "[IMSI] 후보 pose 사전 IK + motion plan 검사를 시작합니다. "
            "로봇은 움직이지 않습니다.",
        )

        preflight_jobs = [
            (pose_name, variant_index, variant)
            for pose_name in remaining_pose_rotations
            if pose_name in candidate_variants
            and (
                not imsi_preflight_only_forced
                or pose_name == forced_first_view_name
            )
            for variant_index, variant in enumerate(
                candidate_variants[pose_name]
            )
        ]
        for name, target in direct_joint_targets.items():
            if name == "user_first_pose":
                continue
            precomputed_joint_targets[name] = target
            preflight_attempts.append({
                "pose_name": name,
                "rotation_xyz_deg": [
                    float(value) for value in remaining_pose_rotations[name]
                ],
                "success": True,
                "reason": "user_joint_states_no_preflight",
                "pivot_camera_distance_m": float(
                    candidate_pivot_distance_m.get(name, float("nan"))
                ),
                **target,
            })
        if user_first_pose_target is not None:
            precomputed_joint_targets[forced_first_view_name] = (
                user_first_pose_target
            )
            preflight_attempts.append({
                "pose_name": forced_first_view_name,
                "rotation_xyz_deg": [
                    float(value)
                    for value in remaining_pose_rotations[forced_first_view_name]
                ],
                "success": True,
                "reason": "user_joint_states_no_preflight",
                "pivot_camera_distance_m": float(
                    candidate_pivot_distance_m.get(
                        forced_first_view_name, float("nan")
                    )
                ),
                **user_first_pose_target,
            })
        if imsi_preflight_only_forced:
            print(
                log_prefix,
                "[IMSI][DEBUG] 강제 첫 뷰만 preflight:",
                forced_first_view_name,
                f"({len(preflight_jobs)} variants)",
            )
        preflight_done_names = set()
        for pose_name, variant_index, variant in preflight_jobs:
            if pose_name in preflight_done_names:
                continue
            rotation_xyz_deg = variant["rotation_xyz_deg"]
            tcp_matrix_np = variant["tcp"]
            tcp_rotation_quaternion = quaternion_from_rotation_matrix(
                tcp_matrix_np[:3, :3]
            )
            target_pose = make_pose(
                tcp_matrix_np[0, 3],
                tcp_matrix_np[1, 3],
                tcp_matrix_np[2, 3],
                tcp_rotation_quaternion,
            )
            result = move_client.preflight_pose(
                label=pose_name,
                pose=target_pose,
            )
            result = result or {
                "success": False,
                "reason": "preflight_timeout",
            }
            attempt_record = {
                "pose_name": pose_name,
                "rotation_xyz_deg": [
                    float(value) for value in rotation_xyz_deg
                ],
                "target_tcp_pose": {
                    "position": [
                        float(tcp_matrix_np[0, 3]),
                        float(tcp_matrix_np[1, 3]),
                        float(tcp_matrix_np[2, 3]),
                    ],
                    "orientation_xyzw": [
                        float(tcp_rotation_quaternion.x),
                        float(tcp_rotation_quaternion.y),
                        float(tcp_rotation_quaternion.z),
                        float(tcp_rotation_quaternion.w),
                    ],
                },
                "success": bool(result.get("success", False)),
                "reason": str(result.get("reason", "unknown")),
                "pivot_camera_distance_m": float(variant["distance_m"]),
                "variant_index": int(variant_index),
            }

            if result.get("success", False):
                joint_names = [
                    str(name) for name in result.get("joint_names", [])
                ]
                joint_positions = [
                    float(value) for value in result.get("positions", [])
                ]
                if joint_names and len(joint_names) == len(joint_positions):
                    cached_target = {
                        "joint_names": joint_names,
                        "positions": joint_positions,
                        "planning_time_s": float(
                            result.get("planning_time_s", 0.0)
                        ),
                        "trajectory_duration_s": float(
                            result.get("trajectory_duration_s", 0.0)
                        ),
                        "ik_time_s": float(result.get("ik_time_s", 0.0)),
                        "joint_source": str(
                            result.get(
                                "joint_source",
                                "planned_trajectory_final_state",
                            )
                        ),
                    }
                    if "in_frame_fraction" in result:
                        cached_target["in_frame_fraction"] = float(
                            result["in_frame_fraction"]
                        )
                    precomputed_joint_targets[pose_name] = cached_target
                    attempt_record.update(cached_target)
                    # This variant is now the candidate's pose.
                    preflight_done_names.add(pose_name)
                    remaining_pose_rotations[pose_name] = rotation_xyz_deg
                    candidate_tcp_matrices[pose_name] = tcp_matrix_np
                    candidate_object_world_matrices[pose_name] = (
                        variant["object_world"]
                    )
                    candidate_pivot_distance_m[pose_name] = (
                        variant["distance_m"]
                    )
                else:
                    attempt_record["success"] = False
                    attempt_record["reason"] = "empty_joint_solution"

            preflight_attempts.append(attempt_record)
            print(
                log_prefix,
                "[IMSI] preflight",
                f"{len(preflight_attempts)}회 시도,",
                f"성공 {len(precomputed_joint_targets)}/"
                f"{precomputed_joint_view_limit}:",
                pose_name,
                variant.get("label", ""),
                f"d={variant['distance_m']:.3f}m",
                attempt_record["reason"],
            )

            if (
                len(precomputed_joint_targets)
                >= precomputed_joint_view_limit
            ):
                break

        preflight_elapsed_s = time.monotonic() - preflight_started_at
        cache_path = os.environ.get(
            "BATTERY_IMSI_JOINT_CACHE_PATH",
            os.path.join(
                occupancy_map_dir,
                "nbv_precomputed_joint_views.json",
            ),
        )
        cache_payload = {
            "schema_version": 1,
            "created_at": datetime.now().isoformat(),
            "requested_success_limit": precomputed_joint_view_limit,
            "successful_view_count": len(precomputed_joint_targets),
            "attempt_count": len(preflight_attempts),
            "elapsed_s": preflight_elapsed_s,
            "joint_source": "planned_trajectory_final_state",
            "attempts": preflight_attempts,
        }
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        temporary_cache_path = cache_path + ".tmp"
        with open(temporary_cache_path, "w", encoding="utf-8") as stream:
            json.dump(cache_payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        os.replace(temporary_cache_path, cache_path)
        print(
            log_prefix,
            "[IMSI] 사전 검사 완료:",
            f"성공={len(precomputed_joint_targets)},",
            f"시도={len(preflight_attempts)},",
            f"elapsed={preflight_elapsed_s:.1f}s,",
            f"cache={cache_path}",
        )

        remaining_pose_rotations = {
            name: rotation
            for name, rotation in remaining_pose_rotations.items()
            if name in precomputed_joint_targets
        }
        candidate_tcp_matrices = {
            name: matrix
            for name, matrix in candidate_tcp_matrices.items()
            if name in precomputed_joint_targets
        }
        candidate_object_world_matrices = {
            name: matrix
            for name, matrix in candidate_object_world_matrices.items()
            if name in precomputed_joint_targets
        }
        candidate_pivot_distance_m = {
            name: distance
            for name, distance in candidate_pivot_distance_m.items()
            if name in precomputed_joint_targets
        }
        if forced_first_view_name is not None:
            if forced_first_view_name in precomputed_joint_targets:
                print(
                    log_prefix,
                    "[IMSI][TOP-CLOSEUP] 선택:",
                    f"rotation={remaining_pose_rotations[forced_first_view_name]},",
                    "distance="
                    f"{candidate_pivot_distance_m[forced_first_view_name]:.3f}m",
                    "- 첫 NBV 뷰로 강제합니다.",
                )
            else:
                failure_reasons = sorted({
                    attempt["reason"]
                    for attempt in preflight_attempts
                    if attempt["pose_name"] == forced_first_view_name
                })
                print(
                    log_prefix,
                    "[IMSI][TOP-CLOSEUP]",
                    f"mode={imsi_top_closeup_mode}의 모든 조합",
                    f"({len(candidate_variants[forced_first_view_name])}개)이",
                    "preflight 실패, 사유=" + ",".join(failure_reasons),
                    "- 로봇을 움직이지 않고 종료합니다."
                    if imsi_preflight_only_forced
                    else "- 강제 첫 뷰 없이 일반 NBV로 진행합니다.",
                )
                forced_first_view_name = None
        if approach_distances_m and lift_pivot_distance_m is not None:
            closer_count = sum(
                distance < lift_pivot_distance_m - 1e-3
                for distance in candidate_pivot_distance_m.values()
            )
            print(
                log_prefix,
                "[IMSI][APPROACH] 가까운 거리로 성공한 후보",
                f"{closer_count}/{len(candidate_pivot_distance_m)}:",
                ", ".join(
                    f"{name}={distance:.3f}m"
                    for name, distance in candidate_pivot_distance_m.items()
                ),
            )
        if (
            lift_pivot_distance_m is not None
            and imsi_proximity_weight_max > 1.0
        ):
            candidate_proximity_weights = proximity_gain_weights(
                candidate_pivot_distance_m,
                lift_pivot_distance_m,
                imsi_proximity_weight_max,
            )
            print(
                log_prefix,
                "[IMSI][PROXIMITY] Beam edge cost weight "
                f"(lift {lift_pivot_distance_m:.3f}m / d)^2, "
                f"max={imsi_proximity_weight_max:.2f}:",
                ", ".join(
                    f"{name}={weight:.2f}"
                    for name, weight in candidate_proximity_weights.items()
                    if weight > 1.0
                ) or "none",
            )

        if not remaining_pose_rotations:
            print(
                log_prefix,
                "[IMSI] 실행 가능한 사전 계산 관절 후보가 없습니다.",
            )
            save_nbv_sequence_manifest(
                manifest_path=nbv_manifest_path,
                frame_records=frame_records,
                status="no_precomputed_joint_candidates",
                started_at=sequence_started_at,
                max_capture_count=max_capture_count,
            )
            return frame_records

        if imsi_route_planner_enabled:
            route_precompute_started_at = time.monotonic()
            route_candidate_names = []
            route_view_directions = {}

            for pose_name in remaining_pose_rotations:
                predicted_object_world_np = (
                    candidate_object_world_matrices[pose_name]
                )
                in_frame_fraction, _, _ = project_object_framing(
                    predicted_object_world_np,
                    camera_world_matrix,
                    intrinsic_matrix,
                    image_size,
                    framing_corners_local,
                    margin_px=nbv_frame_margin_px,
                )
                view_direction = camera_direction_in_object_frame(
                    predicted_object_world_np,
                    camera_world_matrix,
                    reference_local_point=(
                        nbv_view_reference_local_point
                    ),
                )
                if pose_name not in mandatory_name_set and (
                    in_frame_fraction + 1e-12
                    < nbv_min_in_frame_fraction
                    or view_direction[2] + 1e-12
                    < imsi_min_view_direction_local_z
                ):
                    continue
                route_candidate_names.append(pose_name)
                route_view_directions[pose_name] = view_direction

            # The route graph's node gain is occupancy entropy, not a
            # known-mesh surfel prediction. UNKNOWN voxels have p=0.5 and
            # exactly one bit of entropy. Keep their visible flat indices as
            # bitsets so Beam search removes overlap between successive views.
            if frontier_map is not None:
                route_frontier, route_frontier_visible = frontier_visible_for(
                    route_candidate_names
                )
                initial_voxel_results = frontier_tools.frontier_candidate_results(
                    route_frontier_visible
                )
                voxel_route_problem = frontier_tools.frontier_route_problem(
                    route_frontier, route_frontier_visible
                )
                print(
                    log_prefix, "[IMSI][FRONTIER] 후보 가시 frontier (상위 5):",
                    ", ".join(
                        f"{name}={len(visible)}"
                        for name, visible in sorted(
                            route_frontier_visible.items(),
                            key=lambda item: -len(item[1]),
                        )[:5]
                    ),
                    f"/ frontier {len(route_frontier)}",
                )
            else:
                initial_voxel_results = score_candidate_poses(
                    occupancy_map=occupancy_latest_path,
                    candidates=[
                        (
                            name,
                            candidate_object_world_matrices[name],
                        )
                        for name in route_candidate_names
                    ],
                    camera_world_matrix=camera_world_matrix,
                    intrinsic_matrix=intrinsic_matrix,
                    progress_label=f"{log_prefix} [IMSI][VOXEL-PRECOMPUTE]",
                    image_size=image_size,
                    pixel_stride=nbv_pixel_stride,
                    roi_min=nbv_roi_min,
                    roi_max=nbv_roi_max,
                    return_visible_indices=True,
                    score_unknown_only=True,
                    ray_geometry_cache=voxel_ray_cache,
                    surface_model=None,
                    surface_mask=None,
                    background_weight=1.0,
                )
                for result in initial_voxel_results:
                    result["entropy_score"] = float(
                        result["unknown_entropy_sum"]
                    )
                    result["score_source"] = "unknown_voxel_entropy"
                voxel_route_problem = (
                    build_unknown_voxel_visibility_bitsets(
                        occupancy_latest_path,
                        initial_voxel_results,
                    )
                )
            route_visibility_bits = voxel_route_problem[
                "visibility_bits"
            ]
            route_visibility_sources = {
                name: "occupancy-unknown-entropy"
                for name in route_candidate_names
            }

            current_joint_positions = read_joint_positions(rb_robot)
            max_velocity_values = np.asarray(
                rb_robot.get_dof_max_velocities()
            ).reshape(-1)
            max_velocity_by_name = {
                name: abs(float(value))
                for name, value in zip(
                    rb_robot.dof_names,
                    max_velocity_values,
                )
            }

            def route_edge_cost(source_positions, target):
                normalized_joint_times = []
                joint_path_length_rad = 0.0
                for joint_name, target_position in zip(
                    target["joint_names"],
                    target["positions"],
                ):
                    source_position = source_positions.get(joint_name)
                    max_velocity = max_velocity_by_name.get(joint_name)
                    if (
                        source_position is None
                        or max_velocity is None
                        or max_velocity <= 1e-9
                    ):
                        return float("inf")
                    distance = abs(
                        float(target_position) - source_position
                    )
                    joint_path_length_rad += distance
                    normalized_joint_times.append(
                        distance / max_velocity
                    )
                return (
                    max(normalized_joint_times, default=0.0)
                    * imsi_joint_time_scale
                    + imsi_cycle_fixed_overhead_s
                    + joint_path_length_rad
                    * imsi_joint_path_penalty_s_per_rad
                )

            candidate_joint_positions = {
                name: {
                    joint_name: float(position)
                    for joint_name, position in zip(
                        precomputed_joint_targets[name]["joint_names"],
                        precomputed_joint_targets[name]["positions"],
                    )
                }
                for name in route_candidate_names
            }
            initial_edge_costs = {
                name: route_edge_cost(
                    current_joint_positions,
                    precomputed_joint_targets[name],
                )
                for name in route_candidate_names
            }
            pair_edge_costs = {
                source_name: {
                    target_name: (
                        float("inf")
                        if source_name == target_name
                        else route_edge_cost(
                            candidate_joint_positions[source_name],
                            precomputed_joint_targets[target_name],
                        )
                    )
                    for target_name in route_candidate_names
                }
                for source_name in route_candidate_names
            }
            reachable_entropy_bits = voxel_route_problem[
                "reachable_entropy_bits"
            ]
            group_bits = {
                "voxel_entropy": voxel_route_problem["reachable_bits"]
            }
            requested_goal_counts = {
                "voxel_entropy": reachable_entropy_bits
            }
            planner_goal_counts = {
                "voxel_entropy": int(
                    math.ceil(
                        reachable_entropy_bits
                        * imsi_route_reachable_target_ratio
                        - 1e-12
                    )
                )
            }
            reachable_counts = {
                "voxel_entropy": reachable_entropy_bits
            }
            group_totals = {
                "voxel_entropy": reachable_entropy_bits
            }
            initial_mandatory_names = [
                name for name in imsi_mandatory_names
                if name in route_visibility_bits
            ]
            if initial_mandatory_names:
                entropy_by_name = voxel_route_problem[
                    "entropy_bits_by_candidate"
                ]
                mandatory_union = 0
                for name in initial_mandatory_names:
                    mandatory_union |= int(route_visibility_bits[name])
                print(
                    log_prefix,
                    "[IMSI][MANDATORY] 필수 node NBV(UNKNOWN voxel entropy):",
                    ", ".join(
                        f"{name}={entropy_by_name.get(name, 0)} bits"
                        for name in initial_mandatory_names
                    ),
                    f"-> 합집합 {mandatory_union.bit_count()} /",
                    f"reachable {reachable_entropy_bits} bits를 관측된 것으로",
                    "반영, 남은 탐색 영역="
                    f"{reachable_entropy_bits - mandatory_union.bit_count()} bits",
                )
            initial_route_plan = (
                plan_route_with_mandatory
                if initial_mandatory_names
                else plan_motion_aware_nbv_route
            )(
                **(
                    {"mandatory_names": initial_mandatory_names}
                    if initial_mandatory_names
                    else {}
                ),
                observed_mask=0,
                candidate_visibility_bits=route_visibility_bits,
                candidate_names=route_candidate_names,
                current_edge_costs=initial_edge_costs,
                pair_edge_costs=pair_edge_costs,
                group_bits=group_bits,
                goal_counts=planner_goal_counts,
                max_views=max_capture_count - len(frame_records),
                beam_width=imsi_route_beam_width,
                branch_width=imsi_route_branch_width,
                candidate_gain_weights={
                    name: weight
                    * candidate_proximity_weights.get(name, 1.0)
                    for name, weight in (
                        {name: 1.0 for name in route_view_directions}
                        if imsi_inspection_focus
                        else top_view_gain_weights(
                            route_view_directions,
                            previous_top_coverage_fraction,
                            nbv_top_surface_coverage_target,
                            imsi_top_view_max_weight,
                        )
                    ).items()
                },
                first_candidate_names=(
                    inspection_first_candidates(
                        initial_voxel_results, route_view_directions,
                        captured_view_directions,
                        previous_top_coverage_fraction, previous_side_coverage_fraction,
                        nbv_top_surface_coverage_target, nbv_side_surface_coverage_target,
                        nbv_min_expected_entropy,
                        imsi_inspection_min_view_separation_deg,
                        imsi_inspection_top_min_local_z,
                        imsi_inspection_side_min_local_z,
                        imsi_inspection_side_max_local_z,
                        imsi_inspection_azimuth_band_deg,
                    )[0]
                    if imsi_inspection_focus and not initial_mandatory_names
                    else None
                ),
            )
            precomputed_route_graph = {
                "score_basis": "unknown_voxel_entropy",
                "candidate_names": route_candidate_names,
                "visibility_bits": route_visibility_bits,
                "node_entropy_bits": voxel_route_problem[
                    "entropy_bits_by_candidate"
                ],
                "joint_positions": candidate_joint_positions,
                "pair_edge_costs": pair_edge_costs,
                "group_bits": group_bits,
                "goal_counts": planner_goal_counts,
                "requested_goal_counts": requested_goal_counts,
                "reachable_counts": reachable_counts,
                "group_totals": group_totals,
                "visibility_sources": route_visibility_sources,
                "voxel_count": voxel_route_problem["voxel_count"],
                "grid_shape": voxel_route_problem["grid_shape"],
                "top_bias_active": (
                    previous_top_coverage_fraction + 1e-12
                    < nbv_top_surface_coverage_target
                ),
            }

            route_cache_path = os.environ.get(
                "BATTERY_IMSI_ROUTE_CACHE_PATH",
                os.path.join(
                    occupancy_map_dir,
                    "nbv_precomputed_route_graph.npz",
                ),
            )
            os.makedirs(
                os.path.dirname(os.path.abspath(route_cache_path)),
                exist_ok=True,
            )
            if route_candidate_names:
                visibility_byte_count = int(
                    math.ceil(voxel_route_problem["voxel_count"] / 8)
                )
                packed_visibility = np.stack(
                    [
                        np.frombuffer(
                            int(route_visibility_bits[name]).to_bytes(
                                visibility_byte_count,
                                byteorder="little",
                            ),
                            dtype=np.uint8,
                        )
                        for name in route_candidate_names
                    ],
                    axis=0,
                )
            else:
                packed_visibility = np.empty(
                    (
                        0,
                        int(
                            math.ceil(
                                voxel_route_problem["voxel_count"] / 8
                            )
                        ),
                    ),
                    dtype=np.uint8,
                )
            edge_cost_matrix = np.asarray(
                [
                    [
                        pair_edge_costs[source][target]
                        for target in route_candidate_names
                    ]
                    for source in route_candidate_names
                ],
                dtype=np.float32,
            )
            np.savez_compressed(
                route_cache_path,
                schema_version=np.int32(3),
                score_basis=np.asarray("unknown_voxel_entropy"),
                candidate_names=np.asarray(route_candidate_names),
                visibility_packed=packed_visibility,
                visibility_size=np.int32(
                    voxel_route_problem["voxel_count"]
                ),
                grid_shape=np.asarray(
                    voxel_route_problem["grid_shape"],
                    dtype=np.int32,
                ),
                edge_cost_matrix=edge_cost_matrix,
                initial_edge_costs=np.asarray(
                    [initial_edge_costs[name] for name in route_candidate_names],
                    dtype=np.float32,
                ),
                node_entropy_bits=np.asarray(
                    [
                        voxel_route_problem[
                            "entropy_bits_by_candidate"
                        ][name]
                        for name in route_candidate_names
                    ],
                    dtype=np.int32,
                ),
                group_names=np.asarray(["voxel_entropy"]),
                requested_goal_counts=np.asarray(
                    [requested_goal_counts["voxel_entropy"]],
                    dtype=np.int32,
                ),
                planner_goal_counts=np.asarray(
                    [planner_goal_counts["voxel_entropy"]],
                    dtype=np.int32,
                ),
                reachable_counts=np.asarray(
                    [reachable_counts["voxel_entropy"]],
                    dtype=np.int32,
                ),
            )
            route_summary_path = (
                os.path.splitext(route_cache_path)[0] + ".json"
            )
            route_summary = {
                "schema_version": 3,
                "created_at": datetime.now().isoformat(),
                "score_basis": "unknown_voxel_entropy",
                "candidate_count": len(route_candidate_names),
                "top_view_max_weight": imsi_top_view_max_weight,
                "candidate_local_view_z": {
                    name: float(direction[2])
                    for name, direction in route_view_directions.items()
                },
                "beam_width": imsi_route_beam_width,
                "branch_width": imsi_route_branch_width,
                "reachable_target_ratio": (
                    imsi_route_reachable_target_ratio
                ),
                "requested_goal_counts": requested_goal_counts,
                "planner_goal_counts": planner_goal_counts,
                "goal_counts": planner_goal_counts,
                "reachable_counts": reachable_counts,
                "group_totals": group_totals,
                "node_entropy_bits": voxel_route_problem[
                    "entropy_bits_by_candidate"
                ],
                "voxel_count": voxel_route_problem["voxel_count"],
                "grid_shape": voxel_route_problem["grid_shape"],
                "initial_plan": initial_route_plan,
                "graph_path": route_cache_path,
            }
            temporary_route_summary_path = route_summary_path + ".tmp"
            with open(
                temporary_route_summary_path,
                "w",
                encoding="utf-8",
            ) as stream:
                json.dump(
                    route_summary,
                    stream,
                    ensure_ascii=False,
                    indent=2,
                )
                stream.write("\n")
            os.replace(
                temporary_route_summary_path,
                route_summary_path,
            )
            route_precompute_elapsed_s = (
                time.monotonic() - route_precompute_started_at
            )
            if initial_route_plan is None:
                print(
                    log_prefix,
                    "[IMSI][ROUTE] 초기 entropy 경로를 만들 수 없습니다. "
                    "실행 중에도 voxel entropy 기준을 유지하고 greedy "
                    "선택 또는 no-information 종료를 사용합니다.",
                )
            else:
                predicted_entropy_bits = initial_route_plan[
                    "coverage_counts"
                ]["voxel_entropy"]
                predicted_percentage = (
                    predicted_entropy_bits
                    / max(1, reachable_entropy_bits)
                    * 100.0
                )
                requested_complete = all(
                    requested_goal_counts[name] <= 0
                    or initial_route_plan["coverage_counts"][name]
                    >= requested_goal_counts[name]
                    for name in requested_goal_counts
                )
                print(
                    log_prefix,
                    "[IMSI][ROUTE] UNKNOWN voxel entropy 목표:",
                    f"reachable={reachable_entropy_bits} bits,",
                    "planner="
                    f"{planner_goal_counts['voxel_entropy']} bits "
                    f"({imsi_route_reachable_target_ratio * 100:.0f}%)",
                )
                print(
                    log_prefix,
                    "[IMSI][ROUTE] 사전 entropy 최소-view 경로:",
                    " -> ".join(initial_route_plan["route"]),
                    f"views={len(initial_route_plan['route'])},",
                    f"cost={initial_route_plan['cost']:.2f},",
                    f"requested-complete={requested_complete},",
                    "예측 unique entropy="
                    f"{predicted_entropy_bits} bits/"
                    f"{reachable_entropy_bits} bits "
                    f"({predicted_percentage:.1f}%),",
                    f"elapsed={route_precompute_elapsed_s:.2f}s,",
                    f"cache={route_cache_path},",
                    f"summary={route_summary_path}",
                )

    # Separate runtime clock requested for the cached-joint experiment.
    # This boundary is after every preflight IK/plan attempt, cache write and
    # candidate filtering step, immediately before the first NBV selection.
    post_preflight_t0 = time.monotonic()
    post_preflight_render_t0 = render_total_s[0]

    def post_preflight_elapsed_pair():
        wall = time.monotonic() - post_preflight_t0
        render = render_total_s[0] - post_preflight_render_t0
        return wall, wall - render

    if precomputed_joint_view_limit:
        print(
            log_prefix,
            "[IMSI][TIMING] 사전 계산 이후 실행 시간 측정을 시작합니다.",
        )

    # 매 촬영마다 전체 경로를 갈아끼우면, 다음에 반드시 필요하다고
    # 계산된 후보도 계속 두 번째로 밀릴 수 있다. 계획의 앞 N개를 짧게
    # 커밋하고 실제 관측 후 다시 계획하여 안정성과 적응성을 함께 둔다.
    committed_route_names = []
    route_deferral_counts = {}
    # 같은 focus에서 제약이 허용한 뷰가 실측 gain을 못 낸 경우 그 focus.
    # focus가 바뀔 때까지 다음 뷰 제약을 풀어 둔다.
    inspection_failed_focus = None
    cached_empirical_pose_count = (
        sum(
            int(count) > 0
            for name, count in empirical_visibility_cache[
                "sample_counts"
            ].items()
            if name in remaining_pose_rotations
        )
        if empirical_visibility_cache is not None
        else 0
    )
    bootstrap_route_names = [
        name
        for name in imsi_empirical_bootstrap_pose_order
        if name in remaining_pose_rotations
        and precomputed_route_graph is not None
        and name in precomputed_route_graph["visibility_bits"]
    ]
    if (
        bootstrap_route_names
        and precomputed_route_graph.get("score_basis")
        != "unknown_voxel_entropy"
        and cached_empirical_pose_count
        < imsi_empirical_bootstrap_min_cached_poses
    ):
        # A brand-new empirical cache has no trustworthy bit masks yet. Use
        # the small set that repeatedly produced the largest measured top/
        # side gains, then let subsequent episodes plan from their real masks.
        committed_route_names = bootstrap_route_names
        print(
            log_prefix,
            "[IMSI][EMPIRICAL] cache bootstrap 경로:",
            " -> ".join(committed_route_names),
            f"(cached={cached_empirical_pose_count}/"
            f"{imsi_empirical_bootstrap_min_cached_poses})",
        )
    elif (
        precomputed_route_graph is not None
        and initial_route_plan is not None
        and initial_route_plan.get("route")
    ):
        if imsi_route_order_tsp:
            committed_route_names, beam_order_cost, ordered_cost = (
                order_route_by_motion_cost(
                    initial_route_plan["route"],
                    initial_edge_costs,
                    pair_edge_costs,
                )
            )
            print(
                log_prefix,
                "[IMSI][ROUTE] 최초 경로 전체 커밋(관절 이동 최소 순서):",
                " -> ".join(committed_route_names),
                f"motion cost {beam_order_cost:.2f} -> {ordered_cost:.2f}",
            )
        else:
            committed_route_names = list(
                initial_route_plan["route"][:imsi_route_commit_views]
            )
            print(
                log_prefix,
                "[IMSI][ROUTE] 최초 경로 앞부분 커밋:",
                " -> ".join(committed_route_names),
            )
    if forced_first_view_name is not None:
        # The route after the close-up is planned from its joint state and
        # the occupancy it produced, not from the lift pose.
        committed_route_names = []
        print(
            log_prefix,
            "[IMSI][TOP-CLOSEUP] 첫 뷰",
            forced_first_view_name,
            "촬영 후 경로를 다시 계획합니다.",
        )

    def run_full_candidate_scoring(candidate_poses, next_frame_id):
        return score_candidate_poses(
            occupancy_map=occupancy_latest_path,
            candidates=candidate_poses,
            camera_world_matrix=camera_world_matrix,
            intrinsic_matrix=intrinsic_matrix,
            progress_label=(
                f"{log_prefix} [capture {next_frame_id + 1}/"
                f"{max_capture_count}]"
            ),
            image_size=image_size,
            pixel_stride=nbv_pixel_stride,
            roi_min=nbv_roi_min,
            roi_max=nbv_roi_max,
            surface_model=surface_model,
            surface_pixel_bin_size=nbv_surfel_pixel_bin_size,
            surface_occlusion_tolerance=nbv_surfel_occlusion_tolerance,
            surface_min_facing_cos=nbv_surfel_min_facing_cos,
            surface_top_weight=nbv_surface_top_score_weight,
            surface_bottom_weight=nbv_surface_bottom_score_weight,
            surface_side_weight=nbv_surface_side_score_weight,
            background_weight=nbv_background_score_weight,
        )

    def run_voxel_entropy_scoring(candidate_poses, next_frame_id):
        nonlocal initial_voxel_results
        # No capture/map update occurs between graph construction and the
        # first selection. Reuse exactly those ray results once; subsequent
        # captures always trigger fresh occlusion and UNKNOWN evaluation.
        cached = (
            {
                result["candidate_name"]: result
                for result in (initial_voxel_results or [])
            }
            if next_frame_id == 1 else {}
        )
        initial_voxel_results = None
        missing_poses = [
            (name, matrix) for name, matrix in candidate_poses
            if name not in cached
        ]
        results = score_candidate_poses(
            occupancy_map=occupancy_latest_path,
            candidates=missing_poses,
            camera_world_matrix=camera_world_matrix,
            intrinsic_matrix=intrinsic_matrix,
            progress_label=(
                f"{log_prefix} [capture {next_frame_id + 1}/"
                f"{max_capture_count}][VOXEL]"
            ),
            image_size=image_size,
            pixel_stride=nbv_pixel_stride,
            roi_min=nbv_roi_min,
            roi_max=nbv_roi_max,
            return_visible_indices=True,
            score_unknown_only=True,
            ray_geometry_cache=voxel_ray_cache,
            surface_model=None,
            surface_mask=None,
            background_weight=1.0,
        ) if missing_poses else []
        results.extend(
            dict(cached[name]) for name, _ in candidate_poses if name in cached
        )
        if cached:
            print(
                log_prefix,
                "[IMSI][VOXEL] 최초 graph 채점 재사용:",
                f"{len(results) - len(missing_poses)} candidates",
            )
        for result in results:
            # The route objective is new information. In this occupancy map,
            # UNKNOWN means p=0.5, hence one entropy bit per visible cell.
            # Preserve raw_entropy_sum for diagnostics, but do not reward
            # repeatedly viewing cells that already have an observation.
            result["entropy_score"] = float(
                result["unknown_entropy_sum"]
            )
            result["score_source"] = "unknown_voxel_entropy"
        return results

    while (
        len(frame_records) < max_capture_count
        and remaining_pose_rotations
    ):
        if not is_still_gripped():
            print(
                log_prefix,
                "Object no longer gripped - stopping NBV sequence early.",
            )
            status = "aborted_grip_lost"
            break

        next_frame_id = len(frame_records)
        cycle_timer = StageTimer(f"nbv_cycle[{next_frame_id}]")
        candidate_poses = [
            (pose_name, candidate_object_world_matrices[pose_name])
            for pose_name in remaining_pose_rotations
        ]

        # 화각을 벗어나는 후보를 먼저 제거한다. 후보 이름은 회전이지만
        # batterycap은 형상 중심을 고정하도록 TCP 위치까지 보정하지만,
        # IK/작업공간 한계와 mustard의 legacy TCP-pivot 회전 때문에 후보
        # endpoint가 화각을 벗어날 수 있다. 엔트로피 점수 전에 거른다.
        if framing_corners_local is not None and candidate_poses:
            framing = {}

            for pose_name, predicted_object_world_np in candidate_poses:
                framing[pose_name] = project_object_framing(
                    predicted_object_world_np,
                    camera_world_matrix,
                    intrinsic_matrix,
                    image_size,
                    framing_corners_local,
                    margin_px=nbv_frame_margin_px,
                )

            framed_poses = [
                (pose_name, matrix)
                for pose_name, matrix in candidate_poses
                if framing[pose_name][0] >= nbv_min_in_frame_fraction
                or pose_name in mandatory_name_set
            ]
            dropped = len(candidate_poses) - len(framed_poses)

            if framed_poses:
                candidate_poses = framed_poses
            else:
                # 전부 걸리면 교착되므로 가장 잘 담기는 하나는 남긴다.
                best_name = max(
                    framing,
                    key=lambda name: (
                        framing[name][0],
                        -framing[name][1],
                    ),
                )
                candidate_poses = [
                    (name, matrix)
                    for name, matrix in candidate_poses
                    if name == best_name
                ]
                dropped = 0
                print(
                    log_prefix,
                    "화각 안에 온전히 담기는 후보가 없어 가장 나은 "
                    f"하나만 남깁니다: {best_name} "
                    f"(in-frame {framing[best_name][0] * 100:.0f}%, "
                    f"광축 이탈 {framing[best_name][1]:.1f}deg)",
                )

            if dropped:
                print(
                    log_prefix,
                    f"화각을 벗어나는 후보 {dropped}개 제외 "
                    f"(in-frame < {nbv_min_in_frame_fraction * 100:.0f}%): "
                    + ", ".join(
                        f"{name}={framing[name][0] * 100:.0f}%/"
                        f"{framing[name][1]:.0f}deg"
                        for name, _ in remaining_pose_rotations.items()
                        if name in framing
                        and framing[name][0] < nbv_min_in_frame_fraction
                    ),
                )

        # 시선축 둘레로만 도는 후보는 걸러낸다. 물체를 45도 돌려도
        # 회전축이 시선과 가까우면 object frame의 시선 방향은 10도쯤
        # 밖에 안 바뀌고, 그런 뷰는 이미 본 면을 다시 보는 것이라
        # 해소되는 voxel이 거의 없다 (실측: 신규 0~20개).
        separations = {}
        candidate_view_directions = {}

        for pose_name, predicted_object_world_np in candidate_poses:
            candidate_view_directions[pose_name] = (
                camera_direction_in_object_frame(
                    predicted_object_world_np,
                    camera_world_matrix,
                    reference_local_point=nbv_view_reference_local_point,
                )
            )
            separations[pose_name] = min_view_separation_deg(
                candidate_view_directions[pose_name],
                captured_view_directions,
            )

        # IMSI의 top/side 전용 검사는 카메라가 물체 local -Z 쪽에
        # 위치하는 후보를 애초에 허용하지 않는다. bottom 점수만 0으로
        # 두면 side를 조금 함께 보는 뒤집힌 자세가 여전히 선택될 수 있다.
        if precomputed_joint_targets is not None:
            inspection_side_poses = [
                (pose_name, matrix)
                for pose_name, matrix in candidate_poses
                if candidate_view_directions[pose_name][2] + 1e-12
                >= imsi_min_view_direction_local_z
                or pose_name in mandatory_name_set
            ]
            dropped_backside = len(candidate_poses) - len(
                inspection_side_poses
            )
            if dropped_backside:
                print(
                    log_prefix,
                    f"[IMSI] 뒷면(-Z) 시야 후보 {dropped_backside}개 제외 "
                    f"(local view Z < "
                    f"{imsi_min_view_direction_local_z:.2f}):",
                    ", ".join(
                        f"{name}="
                        f"{candidate_view_directions[name][2]:.2f}"
                        for name, _ in candidate_poses
                        if candidate_view_directions[name][2] + 1e-12
                        < imsi_min_view_direction_local_z
                    ),
                )
            candidate_poses = inspection_side_poses
            if not candidate_poses:
                print(
                    log_prefix,
                    "검사 가능한 top/side 후보가 남지 않아 시퀀스를 "
                    "종료합니다.",
                )
                status = "no_top_side_candidates"
                break

        # 일반 route planner는 voxel bitset 중복을 직접 계산한다. 다만
        # IMSI의 시간 제한 검사는 실측/예측 시선이 20도 미만인 재촬영을
        # hard reject한다. 최근 실행에서 그런 세 프레임이 모두 gain=0인
        # 채로 약 3.5초를 소비했기 때문이다.
        if imsi_inspection_focus:
            effective_min_view_separation_deg = (
                imsi_inspection_min_view_separation_deg
            )
        else:
            effective_min_view_separation_deg = (
                0.0
                if precomputed_route_graph is not None
                else nbv_min_view_separation_deg
            )
        diverse_poses = [
            (pose_name, matrix)
            for pose_name, matrix in candidate_poses
            if separations[pose_name]
            >= effective_min_view_separation_deg
            # A close-up repeats a direction at higher pixel density.
            or pose_name == forced_first_view_name
            or pose_name in mandatory_name_set
        ]
        skipped = len(candidate_poses) - len(diverse_poses)

        if diverse_poses:
            candidate_poses = diverse_poses
        elif candidate_poses and imsi_inspection_focus:
            candidate_poses = []
            print(
                log_prefix,
                "[IMSI][INSPECTION] 모든 후보가 기존 실측 시야와 "
                f"{effective_min_view_separation_deg:.0f}deg 미만입니다. "
                "중복 촬영 없이 종료합니다.",
            )
        elif candidate_poses:
            # 전부 걸리면 교착되므로 가장 이격이 큰 하나는 남긴다.
            # separations에는 위에서 뒷면 필터로 제거된 이름도 남아 있다.
            # 반드시 현재 허용된 top/side 후보 안에서만 복구해야 한다.
            allowed_candidate_names = [
                name for name, _ in candidate_poses
            ]
            best_name = max(
                allowed_candidate_names,
                key=separations.get,
            )
            candidate_poses = [
                (name, matrix)
                for name, matrix in candidate_poses
                if name == best_name
            ]
            skipped = 0
            print(
                log_prefix,
                "All candidates are near-redundant views; keeping the "
                f"most separated one: {best_name} "
                f"({separations[best_name]:.1f}deg)",
            )

        if skipped:
            print(
                log_prefix,
                f"Skipped {skipped} near-redundant candidate(s) "
                "(view separation < "
                f"{effective_min_view_separation_deg:.0f}deg):",
                ", ".join(
                    f"{name}={separations[name]:.0f}deg"
                    for name, _ in remaining_pose_rotations.items()
                    if name in separations
                    and separations[name]
                    < effective_min_view_separation_deg
                ),
            )

        print(
            log_prefix,
            "Selecting pose for frame:",
            next_frame_id,
            "- remaining candidates:",
            len(candidate_poses),
        )

        # 향후 필터 조합이 바뀌더라도 빈 목록을 점수 함수에 넘겨
        # 예외로 시뮬레이터 전체가 종료되는 일은 막는다.
        if not candidate_poses:
            print(
                log_prefix,
                "선택 가능한 후보가 남지 않아 NBV 시퀀스를 정상 "
                "종료합니다.",
            )
            status = "no_reachable_candidates"
            break

        using_voxel_entropy_graph = (
            precomputed_route_graph is not None
            and precomputed_route_graph.get("score_basis")
            == "unknown_voxel_entropy"
        )
        inspection_first_names = None
        face_focus = None
        face_focus_names = None
        if using_voxel_entropy_graph:
            ray_cache_hits_before = voxel_ray_cache.hits
            if frontier_map is not None:
                cycle_names = [name for name, _ in candidate_poses]
                if frontier_state["frame"] == len(frame_records) and all(
                    name in frontier_state["visible"] for name in cycle_names
                ):
                    cycle_visible = {
                        name: frontier_state["visible"][name] for name in cycle_names
                    }
                else:
                    _, cycle_visible = frontier_visible_for(cycle_names)
                candidate_results = frontier_tools.frontier_candidate_results(
                    cycle_visible
                )
            else:
                candidate_results = run_voxel_entropy_scoring(
                    candidate_poses,
                    next_frame_id,
                )
            finish_top_active = False
            if (imsi_top_side_only_termination and imsi_finish_top_focus
                    and not imsi_inspection_focus):
                candidate_results, finish_top_active = top_completion_candidates(
                    candidate_results, candidate_view_directions,
                    previous_top_coverage_fraction, previous_side_coverage_fraction,
                    nbv_top_surface_coverage_target, nbv_side_surface_coverage_target,
                    nbv_min_expected_entropy, imsi_finish_top_min_local_z,
                )
            if finish_top_active != precomputed_route_graph.get("finish_top_active", False):
                committed_route_names = []
                route_deferral_counts.clear()
            precomputed_route_graph["finish_top_active"] = finish_top_active
            if finish_top_active:
                print(
                    log_prefix, "[IMSI][TOP-FINISH] side 목표 달성: +Z 후보로 재계획",
                    f"candidates={len(candidate_results)},",
                    f"local-Z>={imsi_finish_top_min_local_z:.2f}",
                )
            candidate_gain_weights = top_view_gain_weights(
                candidate_view_directions,
                previous_top_coverage_fraction,
                nbv_top_surface_coverage_target,
                imsi_top_view_max_weight,
            )
            if imsi_inspection_focus:
                # Direction eligibility is decided above from the measured
                # deficit and azimuth novelty. Inside that eligible band use
                # raw UNKNOWN-voxel entropy (plus real edge cost), not another
                # local-Z multiplier.
                candidate_gain_weights = {
                    name: 1.0 for name in candidate_view_directions
                }
                inspection_first_names, inspection_focus = inspection_first_candidates(
                    candidate_results, candidate_view_directions,
                    captured_view_directions,
                    previous_top_coverage_fraction, previous_side_coverage_fraction,
                    nbv_top_surface_coverage_target, nbv_side_surface_coverage_target,
                    nbv_min_expected_entropy,
                    imsi_inspection_min_view_separation_deg,
                    imsi_inspection_top_min_local_z,
                    imsi_inspection_side_min_local_z,
                    imsi_inspection_side_max_local_z,
                    imsi_inspection_azimuth_band_deg,
                )
                if inspection_failed_focus != inspection_focus:
                    inspection_failed_focus = None
                if not imsi_route_order_tsp:
                    # Each measured capture can change the deficient
                    # direction. A previously committed path may not
                    # override that feedback. The TSP-ordered route instead
                    # replans on low measured gain (see after capture).
                    committed_route_names = []
                    route_deferral_counts.clear()
                print(
                    log_prefix, "[IMSI][INSPECTION] 다음 뷰 방향 제약:",
                    f"focus={inspection_focus},",
                    f"top={previous_top_coverage_fraction * 100:.1f}%,",
                    f"side={previous_side_coverage_fraction * 100:.1f}%,",
                    "eligible=" + ",".join(inspection_first_names),
                    "(gain=UNKNOWN voxel entropy)",
                )
                lift_reason = inspection_constraint_lift_reason(
                    inspection_first_names,
                    inspection_focus,
                    imsi_inspection_min_eligible,
                    inspection_failed_focus,
                )
                if lift_reason is not None:
                    print(
                        log_prefix,
                        "[IMSI][INSPECTION] 제약 해제 -> 전체 후보:",
                        lift_reason,
                    )
                    inspection_first_names = None
            face_focus = None
            face_focus_names = None
            if (
                imsi_face_deficit_focus
                and previous_side_coverage_fraction
                <= nbv_side_surface_coverage_target
            ):
                picked = face_deficit_candidates(
                    surface_model["observed"],
                    side_face_masks,
                    {
                        result["candidate_name"]: candidate_view_directions[
                            result["candidate_name"]
                        ]
                        for result in candidate_results
                        if result["candidate_name"] in candidate_view_directions
                    },
                    imsi_face_deficit_min_coverage,
                    imsi_face_deficit_max_incidence_deg,
                    face_failed,
                )
                if picked is not None:
                    face_focus, face_focus_names, face_fraction = picked
                    inspection_first_names = list(face_focus_names)
                    print(
                        log_prefix,
                        "[IMSI][FACE] 실측",
                        f"{face_focus} 면 {face_fraction * 100:.1f}% <",
                        f"{imsi_face_deficit_min_coverage * 100:.0f}% - 다음 뷰를",
                        f"그 면을 {imsi_face_deficit_max_incidence_deg:.0f}deg 이내로",
                        "향하는 후보로 제한:",
                        ",".join(face_focus_names),
                    )
                    if committed_route_names and not any(
                        name in face_focus_names for name in committed_route_names
                    ):
                        print(
                            log_prefix,
                            "[IMSI][FACE] 커밋 경로에 해당 면 후보가 없어 재계획:",
                            "dropped=" + " -> ".join(committed_route_names),
                        )
                        committed_route_names = []
            candidate_gain_weights = {
                name: weight * candidate_proximity_weights.get(name, 1.0)
                for name, weight in candidate_gain_weights.items()
            }
            top_bias_active = (
                previous_top_coverage_fraction + 1e-12
                < nbv_top_surface_coverage_target
            )
            if frontier_map is None and top_bias_active != precomputed_route_graph["top_bias_active"]:
                # A route committed under top preference must be reconsidered
                # when top succeeds, so the remaining side goal can recover.
                committed_route_names = []
                route_deferral_counts.clear()
            precomputed_route_graph["top_bias_active"] = top_bias_active
            for result in candidate_results:
                name = result["candidate_name"]
                result["top_view_weight"] = candidate_gain_weights[name]
                result["local_view_z"] = float(candidate_view_directions[name][2])
                result["selection_gain"] = (
                    result["entropy_score"] * candidate_gain_weights[name]
                )
            if frontier_map is not None:
                result_names = {result["candidate_name"] for result in candidate_results}
                voxel_route_problem = frontier_tools.frontier_route_problem(
                    frontier_state["frontier"],
                    {
                        name: visible
                        for name, visible in frontier_state["visible"].items()
                        if name in result_names
                    },
                )
            else:
                voxel_route_problem = (
                    build_unknown_voxel_visibility_bitsets(
                        occupancy_latest_path,
                        candidate_results,
                    )
                )
            reachable_entropy_bits = voxel_route_problem[
                "reachable_entropy_bits"
            ]
            precomputed_route_graph["visibility_bits"] = (
                voxel_route_problem["visibility_bits"]
            )
            precomputed_route_graph["node_entropy_bits"] = (
                voxel_route_problem["entropy_bits_by_candidate"]
            )
            precomputed_route_graph["group_bits"] = {
                "voxel_entropy": voxel_route_problem["reachable_bits"]
            }
            precomputed_route_graph["goal_counts"] = {
                "voxel_entropy": int(
                    math.ceil(
                        reachable_entropy_bits
                        * imsi_route_reachable_target_ratio
                        - 1e-12
                    )
                )
            }
            precomputed_route_graph["reachable_counts"] = {
                "voxel_entropy": reachable_entropy_bits
            }
            precomputed_route_graph["group_totals"] = {
                "voxel_entropy": reachable_entropy_bits
            }
            cycle_timer.mark("voxel_entropy_scoring")
            top_candidate = print_candidate_scores(candidate_results)
            print(
                log_prefix,
                "[IMSI][VOXEL] 현재 occupancy로 graph gain 갱신:",
                f"candidates={len(candidate_results)},",
                f"top={top_candidate['candidate_name']},",
                f"top-gain={top_candidate['entropy_score']:.1f} bits,",
                f"reachable-unique={reachable_entropy_bits} bits",
                f"top-bias={'on' if top_bias_active else 'off'},",
                f"max-weight={max(candidate_gain_weights.values()):.2f}",
                f"ray-cache-hits={voxel_ray_cache.hits - ray_cache_hits_before},",
                f"cache-MiB={voxel_ray_cache.bytes_used / (1024 ** 2):.1f}",
            )
        else:
            candidate_results = run_full_candidate_scoring(
                candidate_poses,
                next_frame_id,
            )
            cycle_timer.mark("scoring")
            top_candidate = print_candidate_scores(candidate_results)

        ranked_candidates = sorted(
            candidate_results,
            key=lambda result: result.get("selection_gain", result["entropy_score"]),
            reverse=True,
        )
        mandatory_pending = [
            name for name in imsi_mandatory_names
            if name in remaining_pose_rotations
        ]
        if (
            top_candidate["entropy_score"] <= nbv_min_expected_entropy
            and not mandatory_pending
        ):
            print(
                log_prefix,
                "Stopping: remaining visible entropy below threshold:",
                top_candidate["entropy_score"],
            )
            status = "no_expected_information"
            break
        runtime_route_plan = None
        route_order_applied = False

        # Forced top close-up, then the quality phase, bypass the entropy
        # route: their order is decided by pixel density and motion time.
        override_ranked_names = None
        quality_served_by_name = {}
        candidate_result_names = {
            result["candidate_name"] for result in ranked_candidates
        }
        if (
            forced_first_view_name is not None
            and forced_first_view_name not in remaining_pose_rotations
        ):
            forced_first_view_name = None
        if (
            forced_first_view_name is not None
            and forced_first_view_name in candidate_result_names
        ):
            override_ranked_names = [forced_first_view_name]
            print(
                log_prefix,
                "[IMSI][TOP-CLOSEUP] 강제 첫 뷰:",
                forced_first_view_name,
            )
        elif (
            mandatory_finish_active
            and mandatory_pending
            and precomputed_route_graph is not None
        ):
            route_current_joints = read_joint_positions(rb_robot)
            pending_names = [
                name for name in mandatory_pending
                if name in candidate_result_names
            ]
            override_ranked_names, _, _ = order_route_by_motion_cost(
                pending_names,
                {
                    name: route_edge_cost(
                        route_current_joints, precomputed_joint_targets[name]
                    )
                    for name in pending_names
                },
                precomputed_route_graph["pair_edge_costs"],
            )
            if not override_ranked_names:
                override_ranked_names = None
            else:
                print(
                    log_prefix,
                    "[IMSI][MANDATORY] 커버리지 목표 도달 - 남은 필수 node만 방문:",
                    " -> ".join(override_ranked_names),
                )
        elif quality_phase_active and precomputed_route_graph is not None:
            deficits = quality_deficits()
            route_current_joints = read_joint_positions(rb_robot)
            quality_ranked, uncoverable = rank_quality_candidates(
                [
                    name for name in candidate_result_names
                    if name in precomputed_joint_targets
                ],
                candidate_view_directions,
                {
                    name: float(np.linalg.norm(
                        camera_world_matrix[:3, 3]
                        - (
                            candidate_object_world_matrices[name][:3, :3]
                            @ np.asarray(nbv_view_reference_local_point)
                            + candidate_object_world_matrices[name][:3, 3]
                        )
                    ))
                    for name in candidate_result_names
                    if name in candidate_object_world_matrices
                },
                {
                    name: route_edge_cost(
                        route_current_joints,
                        precomputed_joint_targets[name],
                    )
                    for name in candidate_result_names
                    if name in precomputed_joint_targets
                },
                deficits,
                imsi_quality_reference_distance_m,
                imsi_quality_min_density,
            )
            if uncoverable:
                quality_exhausted_groups.update(uncoverable)
                print(
                    log_prefix,
                    "[IMSI][QUALITY] 남은 후보로 density 기준을 만족할 수 "
                    "없는 면 - 제외:",
                    ",".join(uncoverable),
                )
            if not quality_ranked:
                quality_report("후보 소진")
                print(
                    log_prefix,
                    "[IMSI][QUALITY] 부족한 면을 볼 후보가 없어 종료합니다.",
                )
                status = "coverage_reached"
                break
            override_ranked_names = [name for name, _, _ in quality_ranked]
            quality_served_by_name = {
                name: served for name, served, _ in quality_ranked
            }
            print(
                log_prefix,
                "[IMSI][QUALITY] 다음 뷰 후보(gain/s 순):",
                ", ".join(
                    f"{name}[{'/'.join(served)}]={score:.3f}"
                    for name, served, score in quality_ranked[:6]
                ),
            )

        if override_ranked_names is not None:
            result_by_name = {
                result["candidate_name"]: result
                for result in ranked_candidates
            }
            override_set = set(override_ranked_names)
            ranked_candidates = [
                result_by_name[name] for name in override_ranked_names
            ] + [
                result for result in ranked_candidates
                if result["candidate_name"] not in override_set
                # The quality phase only tries views that serve a face.
                and not quality_phase_active
            ]
            committed_route_names = []
            route_order_applied = True
            cycle_timer.mark("route_rank")

        if (
            precomputed_route_graph is not None
            and ranked_candidates
            and override_ranked_names is None
        ):
            result_by_name = {
                result["candidate_name"]: result
                for result in ranked_candidates
            }
            allowed_route_names = [
                result["candidate_name"]
                for result in ranked_candidates
                if result["candidate_name"]
                in precomputed_route_graph["visibility_bits"]
                and result["candidate_name"]
                in precomputed_route_graph["pair_edge_costs"]
            ]

            committed_route_names = [
                name
                for name in committed_route_names
                if name in allowed_route_names
                and (
                    name in mandatory_name_set
                    or (
                        result_by_name[name]["entropy_score"]
                        > nbv_min_expected_entropy
                        and precomputed_route_graph["visibility_bits"][name]
                    )
                )
            ]

            if committed_route_names:
                committed_name_set = set(committed_route_names)
                ranked_candidates = [
                    result_by_name[name]
                    for name in committed_route_names
                ] + [
                    result
                    for result in ranked_candidates
                    if result["candidate_name"]
                    not in committed_name_set
                ]
                route_order_applied = True
                print(
                    log_prefix,
                    "[IMSI][ROUTE] 커밋한 경로 앞부분 유지:",
                    " -> ".join(committed_route_names),
                )
                cycle_timer.mark("route_rank")
            else:
                route_current_joints = read_joint_positions(rb_robot)
                current_edge_costs = {
                    name: route_edge_cost(
                        route_current_joints,
                        precomputed_joint_targets[name],
                    )
                    for name in allowed_route_names
                }
                runtime_mandatory_names = [
                    name for name in mandatory_pending
                    if name in allowed_route_names
                ]
                runtime_route_plan = (
                    plan_route_with_mandatory
                    if runtime_mandatory_names
                    else plan_motion_aware_nbv_route
                )(
                    **(
                        {"mandatory_names": runtime_mandatory_names}
                        if runtime_mandatory_names
                        else {}
                    ),
                    # The masks were rebuilt from only the occupancy map's
                    # currently UNKNOWN voxels, so there is no separate
                    # observed mask in this bit-index space. Pending
                    # mandatory views are reflected inside the helper.
                    observed_mask=0,
                    candidate_visibility_bits=(
                        precomputed_route_graph["visibility_bits"]
                    ),
                    candidate_names=allowed_route_names,
                    current_edge_costs=current_edge_costs,
                    pair_edge_costs=(
                        precomputed_route_graph["pair_edge_costs"]
                    ),
                    group_bits=precomputed_route_graph["group_bits"],
                    goal_counts=precomputed_route_graph["goal_counts"],
                    max_views=max_capture_count - len(frame_records),
                    beam_width=imsi_runtime_route_beam_width,
                    branch_width=imsi_runtime_route_branch_width,
                    candidate_gain_weights=candidate_gain_weights,
                    first_candidate_names=(
                        face_focus_names
                        if face_focus_names
                        else None if runtime_mandatory_names
                        else inspection_first_names
                    ),
                )
                if runtime_mandatory_names:
                    print(
                        log_prefix,
                        "[IMSI][MANDATORY] 미방문 필수 node 반영 후 재계획:",
                        ",".join(runtime_mandatory_names),
                        "+ 선택",
                        ",".join(
                            name for name in runtime_route_plan["route"]
                            if name not in mandatory_name_set
                        ) or "(없음)",
                    )

            if (
                not route_order_applied
                and runtime_route_plan
                and runtime_route_plan["route"]
            ):
                planned_names = [
                    name
                    for name in runtime_route_plan["route"]
                    if name in result_by_name
                ]
                pinned_face_name = next(
                    (
                        name for name in planned_names
                        if face_focus_names and name in face_focus_names
                    ),
                    None,
                )
                if imsi_route_order_tsp and pinned_face_name is not None:
                    # The deficient face is served first; the rest follow
                    # in minimum joint-travel order from that pose.
                    beam_order_names = list(planned_names)
                    rest_names, rest_before, rest_after = (
                        order_route_by_motion_cost(
                            [n for n in planned_names if n != pinned_face_name],
                            precomputed_route_graph["pair_edge_costs"][
                                pinned_face_name
                            ],
                            precomputed_route_graph["pair_edge_costs"],
                        )
                    )
                    planned_names = [pinned_face_name] + rest_names
                    beam_order_cost = rest_before + current_edge_costs.get(
                        pinned_face_name, 0.0
                    )
                    ordered_cost = rest_after + current_edge_costs.get(
                        pinned_face_name, 0.0
                    )
                elif imsi_route_order_tsp:
                    beam_order_names = list(planned_names)
                    planned_names, beam_order_cost, ordered_cost = (
                        order_route_by_motion_cost(
                            planned_names,
                            current_edge_costs,
                            precomputed_route_graph["pair_edge_costs"],
                        )
                    )
                    print(
                        log_prefix,
                        "[IMSI][ROUTE] 관절 이동 최소 순서:",
                        " -> ".join(beam_order_names),
                        f"({beam_order_cost:.2f}) =>",
                        " -> ".join(planned_names),
                        f"({ordered_cost:.2f})",
                    )
                for deferred_name in (
                    [] if imsi_route_order_tsp else planned_names[1:]
                ):
                    route_deferral_counts[deferred_name] = (
                        route_deferral_counts.get(deferred_name, 0) + 1
                    )
                if planned_names:
                    route_deferral_counts.pop(planned_names[0], None)

                overdue_names = [
                    name
                    for name in planned_names[1:]
                    if route_deferral_counts.get(name, 0)
                    >= imsi_route_max_deferrals
                    and (inspection_first_names is None
                         or name in inspection_first_names)
                ]
                if overdue_names:
                    overdue_name = max(
                        overdue_names,
                        key=lambda name: (
                            route_deferral_counts.get(name, 0),
                            result_by_name[name]["entropy_score"],
                        ),
                    )
                    planned_names.remove(overdue_name)
                    planned_names.insert(0, overdue_name)
                    print(
                        log_prefix,
                        "[IMSI][ROUTE] 반복 지연 후보를 선두로 승격:",
                        overdue_name,
                        f"deferred={route_deferral_counts[overdue_name]}",
                    )
                    route_deferral_counts.pop(overdue_name, None)

                committed_route_names = (
                    list(planned_names)
                    if imsi_route_order_tsp
                    else planned_names[:imsi_route_commit_views]
                )
                planned_name_set = set(planned_names)
                ranked_candidates = [
                    result_by_name[name] for name in planned_names
                ] + [
                    result
                    for result in ranked_candidates
                    if result["candidate_name"] not in planned_name_set
                ]
                route_order_applied = True
                planned_entropy_bits = runtime_route_plan[
                    "coverage_counts"
                ]["voxel_entropy"]
                reachable_entropy_bits = precomputed_route_graph[
                    "reachable_counts"
                ]["voxel_entropy"]
                planned_entropy_percent = (
                    planned_entropy_bits
                    / max(1, reachable_entropy_bits)
                    * 100.0
                )
                print(
                    log_prefix,
                    "[IMSI][ROUTE] occupancy 갱신 반영 entropy 경로:",
                    " -> ".join(planned_names),
                    f"views={len(planned_names)},",
                    f"cost={runtime_route_plan['cost']:.2f},",
                    f"selection-cost={runtime_route_plan['selection_cost']:.2f},",
                    f"complete={runtime_route_plan['complete']},",
                    "unique entropy="
                    f"{planned_entropy_bits}/"
                    f"{reachable_entropy_bits} bits "
                    f"({planned_entropy_percent:.1f}%)",
                )
                cycle_timer.mark("route_rank")
            elif (
                not route_order_applied
                and runtime_route_plan
                and runtime_route_plan["complete"]
            ):
                # Compact seed-route goals are intentionally capped to the
                # candidate graph's reachable coverage. Once those goals are
                # met, return to measured residual NBV below so the unchanged
                # real termination thresholds can still be reached.
                runtime_route_plan = None
                print(
                    log_prefix,
                    "[IMSI][ROUTE] 현재 reachable voxel entropy 목표를 "
                    "달성했습니다. 실측 top/side가 미달이면 residual "
                    "voxel NBV로 보충합니다.",
                )
                cycle_timer.mark("route_rank")
            elif not route_order_applied:
                runtime_route_plan = None
                print(
                    log_prefix,
                    "[IMSI][ROUTE] 잔여 목표 경로를 찾지 못해 이번 "
                    "사이클은 greedy NBV를 사용합니다.",
                )

        if (
            inspection_first_names is not None
            and override_ranked_names is None
            and not (imsi_route_order_tsp and route_order_applied)
        ):
            # Also constrain greedy fallback and retries after a failed move.
            # A later path node is not automatically eligible as the next view.
            # A TSP-ordered route already contains an eligible view (Beam's
            # first-depth constraint); its visit order is decided by motion.
            first_name_set = set(inspection_first_names)
            ranked_candidates = [r for r in ranked_candidates
                                 if r["candidate_name"] in first_name_set]

        # 사전 계산된 관절 목표가 있는 IMSI 모드에서는 정보량과 이동을
        # 따로 우선순위화하지 않고, 예상 정보량 / 이동 보정 비용으로
        # 한 번에 비교한다. 실제 사이클 시간에 더해 6축 관절 이동량의
        # 합에도 비용을 부과한다. 한 축의 최대 이동시간만 비슷하고 여러
        # 관절이 크게 뒤집히는 후보가 과소평가되는 것을 막기 위함이다.
        # Legacy 모드만 top/side surfel 우선순위를 사용한다. IMSI voxel
        # graph에서는 node entropy gain과 motion cost만 비교한다.
        if (
            precomputed_joint_targets is not None
            and ranked_candidates
            and not route_order_applied
        ):
            current_joint_positions = read_joint_positions(rb_robot)
            max_velocity_values = np.asarray(
                rb_robot.get_dof_max_velocities()
            ).reshape(-1)
            max_velocity_by_name = {
                name: abs(float(value))
                for name, value in zip(
                    rb_robot.dof_names,
                    max_velocity_values,
                )
            }
            voxel_entropy_selection = using_voxel_entropy_graph
            top_priority_active = (
                not voxel_entropy_selection
                and previous_top_coverage_fraction + 1e-12
                < nbv_top_surface_coverage_target
            )
            side_priority_active = (
                not voxel_entropy_selection
                and previous_side_coverage_fraction + 1e-12
                < nbv_side_surface_coverage_target
            )
            bottom_priority_active = (
                not voxel_entropy_selection
                and previous_bottom_coverage_fraction + 1e-12
                < nbv_bottom_surface_coverage_target
            )
            if top_priority_active:
                information_ranked_candidates = sorted(
                    ranked_candidates,
                    key=lambda result: (
                        result.get(
                            "directional_surface_unknown_groups",
                            {},
                        ).get("top(+Z)", 0),
                        result["entropy_score"],
                    ),
                    reverse=True,
                )
            elif side_priority_active:
                information_ranked_candidates = sorted(
                    ranked_candidates,
                    key=lambda result: (
                        result.get(
                            "directional_surface_unknown_groups",
                            {},
                        ).get("side", 0),
                        result["entropy_score"],
                    ),
                    reverse=True,
                )
            elif bottom_priority_active:
                information_ranked_candidates = sorted(
                    ranked_candidates,
                    key=lambda result: (
                        result.get(
                            "directional_surface_unknown_groups",
                            {},
                        ).get("bottom(-Z)", 0),
                        result["entropy_score"],
                    ),
                    reverse=True,
                )
            else:
                information_ranked_candidates = ranked_candidates

            information_top_count = min(
                imsi_trajectory_top_k,
                len(information_ranked_candidates),
            )
            information_top = information_ranked_candidates[
                :information_top_count
            ]

            for information_rank, result in enumerate(
                information_top,
                start=1,
            ):
                cached_target = precomputed_joint_targets[
                    result["candidate_name"]
                ]
                normalized_joint_times = []
                joint_path_length_rad = 0.0

                for joint_name, target_position in zip(
                    cached_target["joint_names"],
                    cached_target["positions"],
                ):
                    current_position = current_joint_positions.get(
                        joint_name
                    )
                    max_velocity = max_velocity_by_name.get(joint_name)

                    if (
                        current_position is None
                        or max_velocity is None
                        or max_velocity <= 1e-9
                    ):
                        normalized_joint_times = [float("inf")]
                        joint_path_length_rad = float("inf")
                        break

                    joint_distance = abs(
                        float(target_position) - current_position
                    )
                    joint_path_length_rad += joint_distance
                    normalized_joint_times.append(
                        joint_distance / max_velocity
                    )

                result["trajectory_information_rank"] = information_rank
                result["trajectory_estimate_s"] = max(
                    normalized_joint_times,
                    default=float("inf"),
                ) * imsi_joint_time_scale
                result["joint_path_length_rad"] = joint_path_length_rad
                predicted_cycle_time_s = (
                    result["trajectory_estimate_s"]
                    + imsi_cycle_fixed_overhead_s
                )
                joint_path_penalty_s = (
                    joint_path_length_rad
                    * imsi_joint_path_penalty_s_per_rad
                )
                selection_cost_s = (
                    predicted_cycle_time_s + joint_path_penalty_s
                )
                weighted_gain = float(
                    result.get("selection_gain", result["entropy_score"])
                )
                if not voxel_entropy_selection:
                    weighted_gain = float(
                        result.get(
                            "directional_surface_weighted_unknown_score",
                            result["entropy_score"],
                        )
                    )
                surface_groups = result.get(
                    "directional_surface_unknown_groups",
                    {},
                )
                top_gain = float(surface_groups.get("top(+Z)", 0))
                side_gain = float(surface_groups.get("side", 0))
                bottom_gain = float(surface_groups.get("bottom(-Z)", 0))
                if math.isfinite(predicted_cycle_time_s):
                    information_per_second = (
                        weighted_gain
                        / max(predicted_cycle_time_s, 1e-9)
                    )
                    top_information_per_second = (
                        nbv_surface_top_score_weight
                        * top_gain
                        / max(predicted_cycle_time_s, 1e-9)
                    )
                    side_information_per_second = (
                        nbv_surface_side_score_weight
                        * side_gain
                        / max(predicted_cycle_time_s, 1e-9)
                    )
                    bottom_information_per_second = (
                        nbv_surface_bottom_score_weight
                        * bottom_gain
                        / max(predicted_cycle_time_s, 1e-9)
                    )
                else:
                    information_per_second = 0.0
                    top_information_per_second = 0.0
                    side_information_per_second = 0.0
                    bottom_information_per_second = 0.0
                if math.isfinite(selection_cost_s):
                    motion_adjusted_information_score = (
                        weighted_gain / max(selection_cost_s, 1e-9)
                    )
                    motion_adjusted_top_score = (
                        nbv_surface_top_score_weight
                        * top_gain
                        / max(selection_cost_s, 1e-9)
                    )
                    motion_adjusted_side_score = (
                        nbv_surface_side_score_weight
                        * side_gain
                        / max(selection_cost_s, 1e-9)
                    )
                    motion_adjusted_bottom_score = (
                        nbv_surface_bottom_score_weight
                        * bottom_gain
                        / max(selection_cost_s, 1e-9)
                    )
                else:
                    motion_adjusted_information_score = 0.0
                    motion_adjusted_top_score = 0.0
                    motion_adjusted_side_score = 0.0
                    motion_adjusted_bottom_score = 0.0
                result["predicted_cycle_time_s"] = predicted_cycle_time_s
                result["joint_path_penalty_s"] = joint_path_penalty_s
                result["selection_cost_s"] = selection_cost_s
                result["information_per_second"] = information_per_second
                result["top_information_per_second"] = (
                    top_information_per_second
                )
                result["side_information_per_second"] = (
                    side_information_per_second
                )
                result["bottom_information_per_second"] = (
                    bottom_information_per_second
                )
                result["motion_adjusted_information_score"] = (
                    motion_adjusted_information_score
                )
                result["motion_adjusted_top_score"] = (
                    motion_adjusted_top_score
                )
                result["motion_adjusted_side_score"] = (
                    motion_adjusted_side_score
                )
                result["motion_adjusted_bottom_score"] = (
                    motion_adjusted_bottom_score
                )
                result["predicted_top_gain"] = top_gain
                result["predicted_side_gain"] = side_gain
                result["predicted_bottom_gain"] = bottom_gain

            maximum_top_gain = max(
                result["predicted_top_gain"]
                for result in information_top
            )
            maximum_side_gain = max(
                result["predicted_side_gain"]
                for result in information_top
            )
            maximum_bottom_gain = max(
                result["predicted_bottom_gain"]
                for result in information_top
            )
            best_information_score = max(
                float(result.get("selection_gain", result["entropy_score"]))
                if voxel_entropy_selection
                else float(
                    result.get(
                        "directional_surface_weighted_unknown_score",
                        result["entropy_score"],
                    )
                )
                for result in information_top
            )

            if top_priority_active and maximum_top_gain > 0.0:
                minimum_information_score = (
                    maximum_top_gain
                    * imsi_trajectory_min_score_ratio
                )
                travel_candidates = [
                    result
                    for result in information_top
                    if result["predicted_top_gain"] + 1e-12
                    >= minimum_information_score
                ]
                selection_mode = "top-information/motion-cost"
                travel_candidates.sort(
                    key=lambda result: (
                        -result["motion_adjusted_top_score"],
                        -result["motion_adjusted_information_score"],
                        result["trajectory_estimate_s"],
                    )
                )
            elif side_priority_active and maximum_side_gain > 0.0:
                minimum_information_score = (
                    maximum_side_gain
                    * imsi_trajectory_min_score_ratio
                )
                travel_candidates = [
                    result
                    for result in information_top
                    if result["predicted_side_gain"] + 1e-12
                    >= minimum_information_score
                ]
                selection_mode = "side-information/motion-cost"
                travel_candidates.sort(
                    key=lambda result: (
                        -result["motion_adjusted_side_score"],
                        -result["motion_adjusted_information_score"],
                        result["trajectory_estimate_s"],
                    )
                )
            elif bottom_priority_active and maximum_bottom_gain > 0.0:
                minimum_information_score = (
                    maximum_bottom_gain
                    * imsi_trajectory_min_score_ratio
                )
                travel_candidates = [
                    result
                    for result in information_top
                    if result["predicted_bottom_gain"] + 1e-12
                    >= minimum_information_score
                ]
                selection_mode = "bottom-information/motion-cost"
                travel_candidates.sort(
                    key=lambda result: (
                        -result["motion_adjusted_bottom_score"],
                        -result["motion_adjusted_information_score"],
                        result["trajectory_estimate_s"],
                    )
                )
            else:
                minimum_information_score = (
                    best_information_score
                    * imsi_trajectory_min_score_ratio
                )
                travel_candidates = [
                    result
                    for result in information_top
                    if (
                        float(result.get("selection_gain", result["entropy_score"]))
                        if voxel_entropy_selection
                        else float(
                            result.get(
                                "directional_surface_weighted_unknown_score",
                                result["entropy_score"],
                            )
                        )
                    )
                    + 1e-12
                    >= minimum_information_score
                ]
                selection_mode = (
                    "voxel-entropy/motion-cost"
                    if voxel_entropy_selection
                    else "weighted-information/motion-cost"
                )
                travel_candidates.sort(
                    key=lambda result: (
                        -result["motion_adjusted_information_score"],
                        result["trajectory_estimate_s"],
                        -result["entropy_score"],
                    )
                )

            travel_candidate_names = {
                result["candidate_name"]
                for result in travel_candidates
            }
            deferred_candidates = [
                result
                for result in information_top
                if result["candidate_name"] not in travel_candidate_names
            ]
            ranked_candidates = (
                travel_candidates
                + deferred_candidates
                + information_ranked_candidates[information_top_count:]
            )
            print(
                log_prefix,
                f"[IMSI][TRAVEL] 정보량 상위 {information_top_count}개 중 "
                f"{selection_mode}, "
                f"최고 기준의 {imsi_trajectory_min_score_ratio * 100:.0f}% "
                f"이상 {len(travel_candidates)}개 효율 순서 ",
                f"(threshold={minimum_information_score:.4f}):",
            )
            for travel_rank, result in enumerate(
                travel_candidates,
                start=1,
            ):
                print(
                    log_prefix,
                    f"[IMSI][TRAVEL] {travel_rank:02d}:",
                    result["candidate_name"],
                    f"info-rank={result['trajectory_information_rank']},",
                    f"estimate={result['trajectory_estimate_s']:.3f}s,",
                    f"cycle={result['predicted_cycle_time_s']:.3f}s,",
                    f"info/s={result['information_per_second']:.1f},",
                    f"top/s={result['top_information_per_second']:.1f},",
                    f"side/s={result['side_information_per_second']:.1f},",
                    "bottom/s="
                    f"{result['bottom_information_per_second']:.1f},",
                    f"path-penalty={result['joint_path_penalty_s']:.3f}s,",
                    f"select-cost={result['selection_cost_s']:.3f},",
                    "adjusted="
                    f"{result['motion_adjusted_information_score']:.1f},",
                    f"top-adjusted={result['motion_adjusted_top_score']:.1f},",
                    "side-adjusted="
                    f"{result['motion_adjusted_side_score']:.1f},",
                    "bottom-adjusted="
                    f"{result['motion_adjusted_bottom_score']:.1f},",
                    f"top-new={result['predicted_top_gain']:.0f},",
                    f"side-new={result['predicted_side_gain']:.0f},",
                    f"bottom-new={result['predicted_bottom_gain']:.0f},",
                    f"joint-path={result['joint_path_length_rad']:.3f}rad,",
                    f"score={result['entropy_score']:.4f}",
                )
            cycle_timer.mark("travel_rank")

        selected_nbv = None
        selected_pose_name = None
        selected_rotation = None
        selected_trajectory_estimate_s = None
        selected_information_rank = None
        selected_information_per_second = None
        selected_predicted_cycle_time_s = None
        selected_joint_path_length_rad = None
        selected_joint_path_penalty_s = None
        selected_selection_cost_s = None
        selected_motion_adjusted_information_score = None
        moved = False
        consecutive_failures = 0
        settled_in_candidate_loop = False
        grip_lost_during_candidate = False

        for attempt_index, candidate_result in enumerate(
            ranked_candidates,
            start=1,
        ):
            if (
                candidate_result["entropy_score"]
                <= nbv_min_expected_entropy
                and candidate_result["candidate_name"] not in mandatory_name_set
            ):
                # Direction-weighted ranking need not be sorted by raw bits.
                # A later candidate may still exceed the information floor.
                continue

            candidate_name = candidate_result["candidate_name"]
            candidate_rotation = remaining_pose_rotations.pop(
                candidate_name
            )
            tcp_matrix_np = candidate_tcp_matrices[candidate_name]
            tcp_rotation_quaternion = quaternion_from_rotation_matrix(
                tcp_matrix_np[:3, :3]
            )
            target_pose = make_pose(
                tcp_matrix_np[0, 3],
                tcp_matrix_np[1, 3],
                tcp_matrix_np[2, 3],
                tcp_rotation_quaternion,
            )

            print(
                log_prefix,
                f"Trying cached candidate {attempt_index}/"
                f"{len(ranked_candidates)}:",
                candidate_name,
                candidate_rotation,
                f"score={candidate_result['entropy_score']:.4f}",
            )
            if using_voxel_entropy_graph:
                print(
                    log_prefix,
                    "[IMSI][TOP-VIEW]",
                    f"local-Z={candidate_result['local_view_z']:.3f},",
                    f"weight={candidate_result['top_view_weight']:.2f},",
                    f"selection-gain={candidate_result['selection_gain']:.1f},",
                    f"measured-top={previous_top_coverage_fraction * 100:.1f}%",
                )

            if precomputed_joint_targets is None:
                moved = move_client.move_to_pose(
                    label=candidate_name,
                    pose=target_pose,
                )
            else:
                cached_target = precomputed_joint_targets[candidate_name]
                print(
                    log_prefix,
                    "[IMSI] pose goal 대신 저장된 관절값으로 이동:",
                    candidate_name,
                    [
                        round(value, 5)
                        for value in cached_target["positions"]
                    ],
                )
                moved = move_client.move_to_joint_positions(
                    label=candidate_name,
                    joint_names=cached_target["joint_names"],
                    positions=cached_target["positions"],
                )

                if moved:
                    # Match the legacy NBV acceptance policy: allow modest
                    # joint/TCP tracking error, but require the actually held
                    # object to be in frame and still gripped. TCP/object pose
                    # errors remain in the log for diagnosis only; the USD
                    # object origin is far from the battery geometry and is
                    # therefore unsuitable as a strict position gate.
                    pose_deadline = (
                        time.monotonic() + imsi_final_pose_timeout_s
                    )
                    stable_updates = 0
                    validation_updates = 0
                    endpoint_accepted = False

                    while True:
                        simulation_app.update()
                        validation_updates += 1
                        actual_tcp_pos, actual_tcp_quat = (
                            compute_relative_pose(
                                stage=stage,
                                child_prim_path=tcp_prim_path,
                                parent_prim_path=robot_base_prim_path,
                            )
                        )
                        actual_tcp_matrix = matrix_np_from_pos_quat(
                            actual_tcp_pos,
                            actual_tcp_quat,
                        )
                        actual_object_matrix = matrix_np_from_gf(
                            omni.usd.get_world_transform_matrix(target_prim)
                        )
                        tcp_position_error, tcp_orientation_error = (
                            pose_matrix_errors(
                                actual_tcp_matrix,
                                tcp_matrix_np,
                            )
                        )
                        (
                            object_position_error,
                            object_orientation_error,
                        ) = pose_matrix_errors(
                            actual_object_matrix,
                            candidate_object_world_matrices[candidate_name],
                        )
                        actual_fraction, actual_off_axis, _ = (
                            project_object_framing(
                                actual_object_matrix,
                                camera_world_matrix,
                                intrinsic_matrix,
                                image_size,
                                framing_corners_local,
                                margin_px=nbv_frame_margin_px,
                            )
                        )
                        actual_joint_positions = read_joint_positions(
                            rb_robot
                        )
                        joint_errors = []
                        for joint_name, target_position in zip(
                            cached_target["joint_names"],
                            cached_target["positions"],
                        ):
                            actual_position = actual_joint_positions.get(
                                joint_name
                            )
                            if actual_position is None:
                                joint_errors.append(float("inf"))
                                continue
                            delta = actual_position - target_position
                            joint_errors.append(
                                abs(math.atan2(math.sin(delta), math.cos(delta)))
                            )
                        max_joint_error = max(
                            joint_errors,
                            default=float("inf"),
                        )
                        still_gripped = bool(is_still_gripped())
                        endpoint_usable = (
                            max_joint_error <= imsi_joint_tolerance_rad
                            and actual_fraction + 1e-12
                            >= nbv_min_in_frame_fraction
                            and still_gripped
                        )
                        if (
                            validation_updates
                            >= max(1, int(pose_settle_update_count))
                            and endpoint_usable
                        ):
                            stable_updates += 1
                        else:
                            stable_updates = 0

                        if (
                            stable_updates
                            >= imsi_final_pose_stable_updates
                        ):
                            endpoint_accepted = True
                            break
                        if not still_gripped:
                            break
                        if time.monotonic() >= pose_deadline:
                            break

                    print(
                        log_prefix,
                        "[IMSI][FINAL-POSE]",
                        candidate_name,
                        "PASS" if endpoint_accepted else "REJECT",
                        f"joint-max={max_joint_error:.4f}rad/"
                        f"{imsi_joint_tolerance_rad:.4f}rad,",
                        f"TCP={tcp_position_error * 1000:.1f}mm/"
                        f"{tcp_orientation_error:.2f}deg(info),",
                        f"object={object_position_error * 1000:.1f}mm/"
                        f"{object_orientation_error:.2f}deg(info),",
                        f"in-frame={actual_fraction * 100:.0f}%,",
                        f"off-axis={actual_off_axis:.1f}deg,",
                        f"gripped={still_gripped},",
                        f"stable={stable_updates}/"
                        f"{imsi_final_pose_stable_updates},",
                        f"updates={validation_updates}",
                    )
                    moved = endpoint_accepted
                    if moved:
                        settled_in_candidate_loop = True
                    else:
                        move_client._latest_reason = (
                            "imsi_endpoint_not_accepted "
                            f"joint_max_rad={max_joint_error:.5f} "
                            f"tcp_pos_mm={tcp_position_error * 1000:.1f} "
                            f"tcp_rot_deg={tcp_orientation_error:.2f} "
                            "object_pos_mm="
                            f"{object_position_error * 1000:.1f} "
                            "object_rot_deg="
                            f"{object_orientation_error:.2f} "
                            f"in_frame={actual_fraction:.3f} "
                            f"gripped={still_gripped}"
                        )
                        grip_lost_during_candidate = not still_gripped
            cycle_timer.mark(
                "move_ok" if moved else "move_rejected"
            )

            if moved:
                selected_nbv = candidate_result
                selected_pose_name = candidate_name
                selected_rotation = candidate_rotation
                committed_route_names = [
                    name
                    for name in committed_route_names
                    if name != candidate_name
                ]
                route_deferral_counts.pop(candidate_name, None)
                selected_trajectory_estimate_s = candidate_result.get(
                    "trajectory_estimate_s"
                )
                selected_information_rank = candidate_result.get(
                    "trajectory_information_rank"
                )
                selected_information_per_second = candidate_result.get(
                    "information_per_second"
                )
                selected_predicted_cycle_time_s = candidate_result.get(
                    "predicted_cycle_time_s"
                )
                selected_joint_path_length_rad = candidate_result.get(
                    "joint_path_length_rad"
                )
                selected_joint_path_penalty_s = candidate_result.get(
                    "joint_path_penalty_s"
                )
                selected_selection_cost_s = candidate_result.get(
                    "selection_cost_s"
                )
                selected_motion_adjusted_information_score = (
                    candidate_result.get(
                        "motion_adjusted_information_score"
                    )
                )
                break

            if grip_lost_during_candidate:
                break

            consecutive_failures += 1
            # 0 이하는 '상한 없음'. 남은 후보를 끝까지 시도한다.
            failure_limit = int(nbv_max_consecutive_move_failures)
            limit_text = (
                "무제한" if failure_limit <= 0 else str(failure_limit)
            )
            # relay가 보낸 사유를 같이 남긴다. ik_unreachable(자세 자체가
            # 도달 불가)과 plan_failed(경로를 못 찾음)는 처방이 다르다.
            failure_reason = getattr(
                move_client, "_latest_reason", None
            )
            reason_text = (
                f" reason={failure_reason}" if failure_reason else ""
            )
            print(
                log_prefix,
                "Candidate move rejected; trying next cached candidate:",
                candidate_name,
                candidate_rotation,
                f"({consecutive_failures}/{limit_text} consecutive)"
                f"{reason_text}",
            )

            if 0 < failure_limit <= consecutive_failures:
                # 실패가 연달아 나면 남은 후보를 전부 시도해봐야
                # 시간만 쓴다. 실측: 12개 시도에 63.6초.
                print(
                    log_prefix,
                    f"{consecutive_failures}회 연속 실패 - 남은 후보 "
                    f"{len(ranked_candidates) - attempt_index}개 시도를 "
                    "건너뜁니다.",
                )
                break

        if grip_lost_during_candidate:
            print(
                log_prefix,
                "Object no longer gripped while validating the IMSI "
                "endpoint - stopping NBV sequence early.",
            )
            status = "aborted_grip_lost"
            break

        if not moved:
            print(
                log_prefix,
                "No reachable candidate remains for the current "
                "occupancy map.",
            )
            status = "no_reachable_candidates"
            break

        if not settled_in_candidate_loop:
            update_simulation_frames(
                simulation_app=simulation_app,
                update_count=pose_settle_update_count,
            )
            cycle_timer.mark("settle")

        if not is_still_gripped():
            print(
                log_prefix,
                "Object no longer gripped after move - stopping NBV "
                "sequence early.",
            )
            status = "aborted_grip_lost"
            break

        # MoveIt 최종 자세는 계획된 FK 기준으로 검사하지만 Isaac의 실제
        # 관절 추종 오차나 흡착 물리 오차까지 완전히 예측할 수는 없다.
        # 촬영 직전 실제 USD 자세를 같은 기준으로 한 번 더 확인하여,
        # 화면 밖 결과를 정상 NBV 프레임으로 저장하지 않는다.
        actual_object_world_np = matrix_np_from_gf(
            omni.usd.get_world_transform_matrix(target_prim)
        )
        actual_fraction, actual_off_axis, _ = project_object_framing(
            actual_object_world_np,
            camera_world_matrix,
            intrinsic_matrix,
            image_size,
            framing_corners_local,
            margin_px=nbv_frame_margin_px,
        )
        if actual_fraction + 1e-12 < nbv_min_in_frame_fraction:
            print(
                log_prefix,
                "ERROR: 계획 경로는 통과했지만 실제 도달 자세가 카메라 "
                "화각을 벗어났습니다 - 촬영 없이 NBV를 중단합니다:",
                f"in-frame={actual_fraction * 100:.0f}% < "
                f"{nbv_min_in_frame_fraction * 100:.0f}%,",
                f"off-axis={actual_off_axis:.1f}deg",
            )
            status = "aborted_actual_pose_out_of_frame"
            break

        # Measured post-settle state, not the commanded target - reflects
        # what the arm actually reached rather than what it was asked for.
        selected_joint_positions = read_joint_positions(rb_robot)
        settled_tcp_pos, settled_tcp_quat = compute_relative_pose(
            stage=stage,
            child_prim_path=tcp_prim_path,
            parent_prim_path=robot_base_prim_path,
        )
        selected_tcp_position = [
            float(settled_tcp_pos[0]),
            float(settled_tcp_pos[1]),
            float(settled_tcp_pos[2]),
        ]
        selected_tcp_orientation_xyzw = quat_to_xyzw_list(
            settled_tcp_quat
        )

        camera_world_matrix, _ = capture_current_zivid_frame(
            frame_id=next_frame_id,
            pose_name=selected_pose_name,
            camera_prim=camera_prim,
            object_prim=target_prim,
            zivid_render_product=zivid_render_product,
            zivid_rgb_annotator=zivid_rgb_annotator,
            zivid_depth_annotator=zivid_depth_annotator,
            zivid_semantic_annotator=zivid_semantic_annotator,
            zivid_instance_annotator=zivid_instance_annotator,
            zivid_bbox_annotator=zivid_bbox_annotator,
            orchestrator=orchestrator,
            zivid_save_dir=zivid_save_dir,
            intrinsic_matrix=intrinsic_matrix,
            pause_timeline=True,
            voxel_size=object_grid_leaf_size,
            object_grid_root_size=object_grid_root_size,
            object_grid_origin=object_grid_origin,
        )
        omni.timeline.get_timeline_interface().play()
        accumulate_render_time()
        cycle_timer.mark("capture")

        # 예측이 아니라 실제로 도달한 자세의 시선 방향을 기록한다.
        achieved_view_direction = camera_direction_in_object_frame(
            matrix_np_from_gf(
                omni.usd.get_world_transform_matrix(target_prim)
            ),
            camera_world_matrix,
            reference_local_point=nbv_view_reference_local_point,
        )
        print(
            log_prefix,
            f"View separation achieved: "
            f"{min_view_separation_deg(achieved_view_direction, captured_view_directions):.1f}deg "
            f"(history {len(captured_view_directions)} views)",
        )
        captured_view_directions.append(achieved_view_direction)

        frame_record = make_nbv_frame_record(
            frame_id=next_frame_id,
            pose_name=selected_pose_name,
            rotation_xyz_deg=selected_rotation,
            occupancy_snapshot_path=os.path.join(
                occupancy_map_dir,
                f"object_occupancy_{next_frame_id:06d}.npz",
            ),
            candidate_result=selected_nbv,
            joint_positions=selected_joint_positions,
            tcp_position=selected_tcp_position,
            tcp_orientation_xyzw=selected_tcp_orientation_xyzw,
            elapsed_s=elapsed_pair()[0],
            elapsed_excl_render_s=elapsed_pair()[1],
        )
        frame_records.append(frame_record)

        save_nbv_sequence_manifest(
            manifest_path=nbv_manifest_path,
            frame_records=frame_records,
            status="running",
            started_at=sequence_started_at,
            max_capture_count=max_capture_count,
        )

        # 같은 위치의 반대쪽 면은 별도 surfel이다. 윗면만 촬영해서는
        # 아랫면 커버리지가 올라가지 않으며, bottom(-Z) 수치도 함께
        # 출력해 실제 뒤집기 검사가 수행됐는지 바로 확인할 수 있다.
        _updated_coverage, measured_visible_mask = (
            update_directional_surface_from_last_capture(
            surface_model=surface_model,
            frame_id=next_frame_id,
            occupancy_map_dir=occupancy_map_dir,
            match_distance=nbv_surfel_match_distance,
            min_facing_cos=nbv_surfel_min_facing_cos,
            log_prefix=log_prefix,
            save_debug_artifacts=save_debug_artifacts,
            )
        )
        if empirical_visibility_cache is not None:
            update_empirical_visibility_cache(
                empirical_visibility_cache,
                selected_pose_name,
                measured_visible_mask,
            )
            if (
                precomputed_route_graph is not None
                and precomputed_route_graph.get("score_basis")
                != "unknown_voxel_entropy"
            ):
                empirical_bits = _boolean_mask_to_bits(
                    measured_visible_mask
                )
                precomputed_route_graph["visibility_bits"][
                    selected_pose_name
                ] = empirical_bits
                precomputed_route_graph["visibility_sources"][
                    selected_pose_name
                ] = "empirical-live"
        coverage_result = report_directional_surface_coverage(
            surface_model,
            elapsed_s=elapsed_pair()[0],
            elapsed_excl_render_s=elapsed_pair()[1],
            log_prefix=log_prefix,
        )
        if frontier_map is not None:
            predicted = frontier_state["visible"].get(selected_pose_name)
            frontier_integrate(next_frame_id)
            resolved_ratio = None
            if predicted is not None and len(predicted):
                still_unknown = int(np.count_nonzero(
                    ~frontier_map.observed.reshape(-1)[predicted]
                ))
                resolved_ratio = 1.0 - still_unknown / len(predicted)
            frontier_state["last_resolved"] = resolved_ratio
            remaining_names = [
                name for name in remaining_pose_rotations
                if name in (precomputed_joint_targets or {})
                and name in candidate_object_world_matrices
                and (
                    name in mandatory_name_set
                    or camera_direction_in_object_frame(
                        candidate_object_world_matrices[name],
                        camera_world_matrix,
                        reference_local_point=nbv_view_reference_local_point,
                    )[2] + 1e-12 >= imsi_min_view_direction_local_z
                )
            ]
            _, remaining_visible = frontier_visible_for(remaining_names)
            ranked_gain = sorted(
                ((len(visible), name) for name, visible in remaining_visible.items()),
                reverse=True,
            )
            best_gain = ranked_gain[0][0] if ranked_gain else 0
            threshold = frontier_threshold()
            frontier_state["best"] = best_gain
            frontier_state["best_name"] = ranked_gain[0][1] if ranked_gain else None
            frontier_state["done"] = best_gain < threshold
            print(
                log_prefix, "[IMSI][FRONTIER] 남은 후보 최대 가시 frontier",
                f"{best_gain} ({frontier_state['best_name']}),",
                "다음: " + ", ".join(f"{name}={gain}" for gain, name in ranked_gain[1:4]) + ",",
                f"stop < {threshold:.0f} -> {'종료' if frontier_state['done'] else '계속'}",
                (
                    f"| 이번 뷰 예측 frontier {len(predicted)}개 중 "
                    f"{resolved_ratio * 100:.0f}% 관측"
                    if resolved_ratio is not None else ""
                ),
                f"| GT top={coverage_result['groups']['top(+Z)']['fraction'] * 100:.1f}%"
                f" side={coverage_result['groups']['side']['fraction'] * 100:.1f}%",
            )
        if imsi_quality_enabled:
            quality_before = quality_coverage_fractions(
                surface_model, quality_group_masks
            )
            update_quality_observations(
                surface_model,
                measured_visible_mask,
                surface_model["last_sensor_origin_object"],
                imsi_quality_reference_distance_m,
                imsi_quality_min_density,
            )
            quality_after = quality_report(
                f"frame {next_frame_id} {selected_pose_name} "
                f"(d={candidate_pivot_distance_m.get(selected_pose_name, float('nan')):.3f}m)"
            )
            if quality_phase_active:
                quality_views_taken += 1
                for face in quality_served_by_name.get(selected_pose_name, []):
                    face_gain = quality_after[face] - quality_before[face]
                    if face_gain < imsi_quality_min_gain:
                        # Predicted from the pose direction only; occlusion
                        # can make a face unreachable. Do not retry it.
                        quality_exhausted_groups.add(face)
                        print(
                            log_prefix,
                            f"[IMSI][QUALITY] {face} 목표 뷰의 실측 gain",
                            f"{face_gain * 100:.2f}%p <",
                            f"{imsi_quality_min_gain * 100:.2f}%p - 제외",
                        )
        coverage_gain = (
            float(coverage_result["fraction"])
            - previous_coverage_fraction
        )
        top_coverage_fraction = float(
            coverage_result["groups"]["top(+Z)"]["fraction"]
        )
        top_coverage_gain = (
            top_coverage_fraction - previous_top_coverage_fraction
        )
        side_coverage_fraction = float(
            coverage_result["groups"]["side"]["fraction"]
        )
        side_coverage_gain = (
            side_coverage_fraction - previous_side_coverage_fraction
        )
        bottom_coverage_fraction = float(
            coverage_result["groups"]["bottom(-Z)"]["fraction"]
        )
        bottom_coverage_gain = (
            bottom_coverage_fraction - previous_bottom_coverage_fraction
        )
        coverage_gains_by_goal = {
            "전체": coverage_gain,
            "top": top_coverage_gain,
            "side": side_coverage_gain,
            "bottom": bottom_coverage_gain,
        }
        active_coverage_gains = [
            coverage_gains_by_goal[name] for name in gain_goal_names
        ]
        low_coverage_gain = (
            nbv_min_surface_coverage_gain > 0.0
            and all(
                gain < nbv_min_surface_coverage_gain
                for gain in active_coverage_gains
            )
        )
        # Gain only in groups that were still below target before this view.
        deficient_group_gains = [
            gain
            for gain, before, target in (
                (top_coverage_gain, previous_top_coverage_fraction,
                 nbv_top_surface_coverage_target),
                (side_coverage_gain, previous_side_coverage_fraction,
                 nbv_side_surface_coverage_target),
            )
            if target > 0.0 and before <= target
        ]
        if (
            inspection_first_names is not None
            and override_ranked_names is None
            and selected_pose_name in inspection_first_names
            and inspection_focus in {"top", "side"}
        ):
            focus_gain = (
                top_coverage_gain
                if inspection_focus == "top"
                else side_coverage_gain
            )
            if focus_gain < imsi_inspection_fallback_min_gain:
                inspection_failed_focus = inspection_focus
                print(
                    log_prefix,
                    "[IMSI][INSPECTION] 제약 허용 뷰의 실측",
                    f"{inspection_focus} gain={focus_gain * 100:.2f}%p <",
                    f"{imsi_inspection_fallback_min_gain * 100:.2f}%p -",
                    "focus가 바뀔 때까지 제약을 해제합니다.",
                )
        if face_focus is not None and selected_pose_name in (face_focus_names or []):
            observed_now = np.asarray(surface_model["observed"], dtype=bool)
            face_now = float(
                np.count_nonzero(observed_now & side_face_masks[face_focus])
            ) / max(1, int(np.count_nonzero(side_face_masks[face_focus])))
            print(
                log_prefix,
                f"[IMSI][FACE] {face_focus} 면 실측 {face_now * 100:.1f}%",
                f"({selected_pose_name})",
            )
            if face_now + 1e-12 < imsi_face_deficit_min_coverage:
                face_failed.add(face_focus)
        if (
            frontier_map is not None
            and imsi_route_order_tsp
            and committed_route_names
            and frontier_state["last_resolved"] is not None
            and frontier_state["last_resolved"] < imsi_frontier_replan_min_resolved
        ):
            print(
                log_prefix,
                "[IMSI][FRONTIER] 예측 frontier의",
                f"{frontier_state['last_resolved'] * 100:.0f}%만 관측 (<",
                f"{imsi_frontier_replan_min_resolved * 100:.0f}%) - 커밋 경로 폐기 후 재계획:",
                " -> ".join(committed_route_names),
            )
            committed_route_names = []
        if (
            frontier_map is None
            and imsi_route_order_tsp
            and committed_route_names
            and deficient_group_gains
            and max(deficient_group_gains) < imsi_route_replan_min_gain
        ):
            print(
                log_prefix,
                "[IMSI][ROUTE] 실측 gain 부족으로 커밋 경로 폐기 후 재계획:",
                f"{selected_pose_name} max deficient gain="
                f"{max(deficient_group_gains) * 100:.2f}%p <",
                f"{imsi_route_replan_min_gain * 100:.2f}%p,",
                "dropped=" + " -> ".join(committed_route_names),
            )
            committed_route_names = []
        previous_coverage_fraction = float(coverage_result["fraction"])
        previous_top_coverage_fraction = top_coverage_fraction
        previous_side_coverage_fraction = side_coverage_fraction
        previous_bottom_coverage_fraction = bottom_coverage_fraction
        if low_coverage_gain:
            consecutive_low_coverage_gains += 1
        else:
            consecutive_low_coverage_gains = 0

        cycle_timer.mark("coverage")
        wall_s, excl_render_s = elapsed_pair()
        post_preflight_wall_s, post_preflight_excl_render_s = (
            post_preflight_elapsed_pair()
        )
        cycle_extra = {
            "t_from_lift": f"{wall_s:.2f}",
            "t_excl_render": f"{excl_render_s:.2f}",
            "t_after_preflight": f"{post_preflight_wall_s:.2f}",
            "t_after_preflight_excl_render": (
                f"{post_preflight_excl_render_s:.2f}"
            ),
            "coverage_gain_pct": f"{coverage_gain * 100:.2f}",
            "top_coverage_gain_pct": f"{top_coverage_gain * 100:.2f}",
            "side_coverage_gain_pct": f"{side_coverage_gain * 100:.2f}",
            "bottom_coverage_gain_pct": (
                f"{bottom_coverage_gain * 100:.2f}"
            ),
            "low_gain_consecutive": consecutive_low_coverage_gains,
        }

        if selected_trajectory_estimate_s is not None:
            cycle_extra["trajectory_estimate"] = (
                f"{selected_trajectory_estimate_s:.3f}"
            )

        if selected_information_rank is not None:
            cycle_extra["information_rank"] = selected_information_rank

        if selected_information_per_second is not None:
            cycle_extra["information_per_second"] = (
                f"{selected_information_per_second:.1f}"
            )

        if selected_predicted_cycle_time_s is not None:
            cycle_extra["predicted_cycle_time"] = (
                f"{selected_predicted_cycle_time_s:.3f}"
            )

        if selected_joint_path_length_rad is not None:
            cycle_extra["joint_path_rad"] = (
                f"{selected_joint_path_length_rad:.3f}"
            )

        if selected_joint_path_penalty_s is not None:
            cycle_extra["joint_path_penalty"] = (
                f"{selected_joint_path_penalty_s:.3f}"
            )

        if selected_selection_cost_s is not None:
            cycle_extra["selection_cost"] = (
                f"{selected_selection_cost_s:.3f}"
            )

        if selected_motion_adjusted_information_score is not None:
            cycle_extra["motion_adjusted_score"] = (
                f"{selected_motion_adjusted_information_score:.1f}"
            )

        # relay가 알려준 MoveIt 내역. move_ok 안에서 플래닝이 얼마고
        # 실제 주행이 얼마인지 여기서만 갈린다.
        plan_s = getattr(move_client, "last_move_plan_s", None)
        traj_s = getattr(move_client, "last_move_traj_s", None)

        if plan_s is not None:
            cycle_extra["moveit_plan"] = f"{plan_s:.3f}"

        if traj_s is not None:
            cycle_extra["moveit_traj"] = f"{traj_s:.3f}"

        for key, attribute in (
            ("moveit_ik", "last_move_ik_s"),
            ("moveit_accept", "last_move_accept_s"),
            ("moveit_guard", "last_move_guard_s"),
            ("moveit_exec", "last_move_exec_s"),
        ):
            value = getattr(move_client, attribute, None)

            if value is not None:
                cycle_extra[key] = f"{value:.3f}"

        cycle_timer.summary(extra=cycle_extra)

        if (
            imsi_max_nbv_views
            and len(frame_records) - 1 >= imsi_max_nbv_views
        ):
            if imsi_quality_enabled:
                quality_report("디버그 종료 시점")
            print(
                log_prefix,
                "[IMSI][DEBUG] NBV 뷰",
                f"{len(frame_records) - 1}/{imsi_max_nbv_views}개 촬영 -",
                "시퀀스 종료",
            )
            status = "debug_view_limit"
            break
        quality_pending = (
            imsi_quality_enabled
            and bool(quality_deficits())
            and quality_views_taken < imsi_quality_max_views
        )
        mandatory_pending = [
            name for name in imsi_mandatory_names
            if name in remaining_pose_rotations
        ]
        if imsi_stop_at_coverage:
            mandatory_pending = []
            quality_pending = False
        stop_reached = (
            frontier_state["done"]
            if frontier_map is not None
            else surface_targets_reached(coverage_result)
        )
        if stop_reached and mandatory_pending:
            if not mandatory_finish_active:
                print(
                    log_prefix,
                    "[IMSI][MANDATORY] top/side 커버리지 목표 도달, 미방문 필수",
                    "node를 마저 방문합니다:",
                    ",".join(mandatory_pending),
                )
            mandatory_finish_active = True
            committed_route_names = []
        elif stop_reached and quality_pending:
            if not quality_phase_active:
                print(
                    log_prefix,
                    "[IMSI][QUALITY] top/side 커버리지 목표 도달, 해상도 "
                    "기준 미달 면 보충 단계로 전환:",
                    ", ".join(
                        f"{name}(-{deficit * 100:.1f}%p)"
                        for name, deficit in quality_deficits().items()
                    ),
                    f"(최대 {imsi_quality_max_views}뷰)",
                )
            quality_phase_active = True
            committed_route_names = []
        elif stop_reached and frontier_map is not None:
            if imsi_quality_enabled:
                quality_report("종료 시점")
            print(
                log_prefix,
                "[IMSI][FRONTIER] 남은 후보의 최대 가시 frontier",
                f"{frontier_state['best']} < {frontier_threshold():.0f} - 시퀀스 종료",
                f"(GT top={coverage_result['groups']['top(+Z)']['fraction'] * 100:.1f}%,",
                f"side={coverage_result['groups']['side']['fraction'] * 100:.1f}%)",
            )
            status = "frontier_exhausted"
            break
        elif stop_reached:
            if imsi_quality_enabled:
                quality_report("종료 시점")
            print(
                f"{log_prefix} 표면 커버리지 목표 "
                "도달 "
                f"({coverage_result['fraction'] * 100:.1f}%, "
                "top="
                f"{coverage_result['groups']['top(+Z)']['fraction'] * 100:.1f}%, "
                "side="
                f"{coverage_result['groups']['side']['fraction'] * 100:.1f}%, "
                "bottom="
                f"{coverage_result['groups']['bottom(-Z)']['fraction'] * 100:.1f}%) "
                "- 시퀀스 종료"
            )
            status = "coverage_reached"
            break

        if low_coverage_gain:
            required_low_gains = max(
                1,
                int(nbv_min_surface_coverage_gain_consecutive),
            )
            if consecutive_low_coverage_gains >= required_low_gains:
                gain_text = ", ".join(
                    f"{name}="
                    f"{coverage_gains_by_goal[name] * 100:.2f}%p"
                    for name in gain_goal_names
                )
                if imsi_top_side_only_termination:
                    print(
                        f"{log_prefix} 직전 촬영 대비 표면 커버리지 증가 "
                        f"{gain_text}가 모두 < "
                        f"{nbv_min_surface_coverage_gain * 100:.2f}%p가 "
                        f"{consecutive_low_coverage_gains}회 연속 - "
                        "top/side 90% 목표 미달이므로 다음 후보를 계속 검사"
                    )
                else:
                    print(
                        f"{log_prefix} 직전 촬영 대비 표면 커버리지 증가 "
                        f"{gain_text}가 모두 < "
                        f"{nbv_min_surface_coverage_gain * 100:.2f}%p가 "
                        f"{consecutive_low_coverage_gains}회 연속 - "
                        "시퀀스 종료"
                    )
                    status = "surface_gain_below_threshold"
                    break
            else:
                print(
                    f"{log_prefix} 표면 커버리지 저증가 "
                    f"{consecutive_low_coverage_gains}/"
                    f"{required_low_gains}: "
                    + ", ".join(
                        f"{name}="
                        f"{coverage_gains_by_goal[name] * 100:.2f}%p"
                        for name in gain_goal_names
                    )
                    + " - "
                    "다음 뷰를 계속 검사합니다."
                )

    if (
        status == "completed"
        and len(frame_records) >= max_capture_count
    ):
        status = "max_capture_count_reached"

    save_nbv_sequence_manifest(
        manifest_path=nbv_manifest_path,
        frame_records=frame_records,
        status=status,
        started_at=sequence_started_at,
        max_capture_count=max_capture_count,
    )
    save_directional_surface_model(
        surface_model=surface_model,
        occupancy_map_dir=occupancy_map_dir,
        frame_id=len(frame_records) - 1,
        save_frame_snapshot=save_debug_artifacts,
    )
    if empirical_visibility_cache is not None:
        save_empirical_visibility_cache(
            empirical_visibility_cache,
            log_prefix,
        )

    report_directional_surface_coverage(
        surface_model,
        elapsed_s=elapsed_pair()[0],
        elapsed_excl_render_s=elapsed_pair()[1],
        log_prefix=log_prefix + " [최종]",
    )

    # 심어둔 결함 중 몇 개를 실제로 봤는가. 표면 커버리지는 '면적'을,
    # 이쪽은 '결함 개수'를 센다. 결함을 끄고 돌리면 조용히 건너뛴다.
    try:
        try:
            from rb_ws.Battery_defect_recall import report_defect_recall
        except ModuleNotFoundError:
            from Battery_defect_recall import report_defect_recall

        report_defect_recall(
            zivid_save_dir=zivid_save_dir,
            log_prefix=log_prefix + " [RECALL]",
        )
    except Exception as error:
        # 평가 지표 하나 때문에 시퀀스 결과를 잃지 않는다.
        print(log_prefix, "[RECALL] 계산 실패:", error)

    final_wall_s, final_excl_render_s = elapsed_pair()
    final_post_preflight_s, final_post_preflight_excl_render_s = (
        post_preflight_elapsed_pair()
    )
    if imsi_top_side_only_termination:
        final_quality = (
            quality_coverage_fractions(surface_model, quality_group_masks)
            if imsi_quality_enabled
            else {}
        )
        captured_pose_names = {
            str(record.get("pose_name")) for record in frame_records
        }
        mandatory_visited = [
            name for name in imsi_mandatory_names
            if name in captured_pose_names
        ]
        # An exhausted face still counts as NOT met: it was not inspected.
        quality_passed = all(
            fraction + 1e-12 >= imsi_quality_target
            for fraction in final_quality.values()
        )
        inspection_passed = (
            previous_top_coverage_fraction > nbv_top_surface_coverage_target
            and previous_side_coverage_fraction > nbv_side_surface_coverage_target
            and (imsi_stop_at_coverage or quality_passed)
            and (
                imsi_stop_at_coverage
                or len(mandatory_visited) == len(imsi_mandatory_names)
            )
            and final_post_preflight_excl_render_s < 10.0
        )
        print(
            log_prefix, "[IMSI][ACCEPTANCE]",
            "PASS" if inspection_passed else "NOT_MET",
            f"top={previous_top_coverage_fraction * 100:.2f}% "
            f"(>{nbv_top_surface_coverage_target * 100:.1f}%),",
            f"side={previous_side_coverage_fraction * 100:.2f}% "
            f"(>{nbv_side_surface_coverage_target * 100:.1f}%),",
            (
                "quality("
                + ", ".join(
                    f"{name}={fraction * 100:.1f}%"
                    for name, fraction in final_quality.items()
                )
                + f") (>={imsi_quality_target * 100:.0f}% each"
                + (", info only" if imsi_stop_at_coverage else "")
                + "),"
                if final_quality
                else "quality=off,"
            ),
            (
                f"mandatory={len(mandatory_visited)}/{len(imsi_mandatory_names)}"
                + (
                    " missing=" + ",".join(
                        name for name in imsi_mandatory_names
                        if name not in mandatory_visited
                    )
                    if len(mandatory_visited) < len(imsi_mandatory_names)
                    else ""
                )
                + ","
                if imsi_mandatory_names
                else ""
            ),
            (
                f"decision=frontier(best remaining {frontier_state['best']}, "
                f"stop<{frontier_threshold():.0f}, frontier "
                f"{(frontier_state['stats'] or {}).get('frontier')}/"
                f"{frontier_state['initial']}),"
                if frontier_map is not None
                else ""
            ),
            "사전 계산 이후 render 제외="
            f"{final_post_preflight_excl_render_s:.3f}s (<10.0s)",
        )
    print(
        log_prefix,
        "Arm-driven NBV capture sequence finished:",
        len(frame_records),
        "frames, status=",
        status,
        f"총 소요 {final_wall_s:.1f}s (들어올림 기준), "
        f"render 제외 {final_excl_render_s:.1f}s, "
        f"사전 계산 이후 {final_post_preflight_s:.1f}s, "
        "사전 계산 이후 render 제외 "
        f"{final_post_preflight_excl_render_s:.1f}s "
        f"(Replicator 렌더 누적 {render_total_s[0]:.1f}s)",
    )
    return frame_records
