"""Debug: three hand-set mandatory views + NBV/Beam search for the rest.

Same settings as Battery_rb_imsi.py, except that three poses set by hand
(top frontal, -X end, +X end) are must-visit route nodes. Their predicted
voxel observations are reflected before the remaining candidates are scored,
preflighted and planned with Beam search; the full route is visited in
minimum joint-travel order and only ends after all three are captured.
"""

import os
import runpy


os.environ.setdefault("BATTERY_IMSI_PRECOMPUTE_VIEW_LIMIT", "200")
os.environ.setdefault("BATTERY_BATTERYCAP_ARM_CAPTURE_POSE_LIMIT", "200")
os.environ.setdefault("BATTERY_SAVE_DEBUG_ARTIFACTS", "0")
os.environ.setdefault("BATTERY_SAVE_FRAME_IMAGES", "1")
os.environ.setdefault("BATTERY_PNG_COMPRESSION_LEVEL", "1")
os.environ.setdefault("BATTERY_IMSI_INSPECTION_FOCUS", "1")
os.environ.setdefault(
    "BATTERY_IMSI_INSPECTION_MIN_VIEW_SEPARATION_DEG", "20.0"
)
os.environ.setdefault("BATTERY_IMSI_INSPECTION_TOP_MIN_LOCAL_Z", "0.35")
# Same as the backside exclusion below. At -0.05, local_x_pos_150deg
# (local-Z -0.12), the view that took side from 83.7% to 98.5%, was
# excluded from side focus for 3 cycles (battery_rb_20260929_094132).
os.environ.setdefault("BATTERY_IMSI_INSPECTION_SIDE_MIN_LOCAL_Z", "-0.35")
os.environ.setdefault("BATTERY_IMSI_INSPECTION_SIDE_MAX_LOCAL_Z", "0.25")
os.environ.setdefault("BATTERY_IMSI_INSPECTION_AZIMUTH_BAND_DEG", "15.0")
# Lift the next-view constraint if it leaves < 3 views or a view it admitted
# gains < 0.5%p in its focus group (until the focus changes).
os.environ.setdefault("BATTERY_IMSI_INSPECTION_MIN_ELIGIBLE", "3")
os.environ.setdefault("BATTERY_IMSI_INSPECTION_FALLBACK_MIN_GAIN", "0.005")
os.environ.setdefault("BATTERY_IMSI_TRAJECTORY_TOP_K", "15")
os.environ.setdefault("BATTERY_IMSI_TRAJECTORY_MIN_SCORE_RATIO", "0.20")
# Edge cost = measured post-preflight seconds per view (excluding render).
# Fit on 8 moves of battery_rb_20260929_094132 (r=0.989, residual <= 0.15 s):
#   move_ok = 0.264 + 1.336 * max(|dq| / vmax);  capture/scoring = 0.43 s.
# The L1 joint penalty predicted worse (r=0.967) and inflated costs ~2.8x.
os.environ.setdefault("BATTERY_IMSI_JOINT_TIME_SCALE", "1.336")
os.environ.setdefault("BATTERY_IMSI_CYCLE_FIXED_OVERHEAD_S", "0.70")
os.environ.setdefault("BATTERY_IMSI_JOINT_PATH_PENALTY_S_PER_RAD", "0.0")
# Beam picks the view set; Held-Karp orders it by motion; the whole route is
# committed and replanned only after a view gains < 0.5%p in unmet groups.
os.environ.setdefault("BATTERY_IMSI_ROUTE_ORDER_TSP", "1")
os.environ.setdefault("BATTERY_IMSI_ROUTE_REPLAN_MIN_GAIN", "0.005")
# Recall: the 4.0 s run (battery_rb_20260929_101032) got 0/5 because the
# top was only seen at 57 deg from 0.50 m and the X end faces at 66-80 deg.
# Preflight each candidate at these pivot-camera distances first (the lift
# pose is 0.496 m); on IK/plan/frame failure fall back to the next one.
os.environ.setdefault("BATTERY_IMSI_APPROACH_DISTANCES_M", "0.42,0.45")
# Shift the NBV work point (rotation centre) in WORLD x,y,z [m]. World +Y
# points from the robot (y=-0.05) toward the camera (y=0.91); the robot base
# is yawed 180 deg, so this is base -Y. +0.01: camera 0.496 -> ~0.488 m.
os.environ.setdefault("BATTERY_IMSI_WORKSPACE_OFFSET_M", "0,0.01,0")
# The automatic top close-up is replaced by the hand-set mandatory views.
os.environ.setdefault("BATTERY_IMSI_TOP_CLOSEUP", "0")
os.environ.setdefault("BATTERY_IMSI_PROXIMITY_WEIGHT_MAX", "1.5")
# --- debug: hand-set mandatory views + NBV for the rest --------------------
# Three poses set by hand and read over ROS with rb_ws/capture_first_pose.py
# (joints: /joint_states, TCP: tf world->tcp); see rb_ws/first_pose_records.txt.
#   top_frontal  (11:34): +Z at  5.0 deg, camera 0.573 m, 0.7 deg off axis
#   minus_x_end  (11:37): -X at  3.0 deg, camera 0.536 m, 1.5 deg off axis
#   plus_x_end   (12:21): +X at  2.9 deg, camera 0.543 m, 0.7 deg off axis
# Their joints are used as-is (no IK preflight). Their visible UNKNOWN voxels
# are reflected as observed before the other candidates are scored, so Beam
# search only plans the remaining region; every route contains the unvisited
# mandatory views and the whole route is ordered by joint travel. The run
# does not end until all three are captured.
os.environ.setdefault(
    "BATTERY_IMSI_MANDATORY_POSE_NAMES", "top_frontal,minus_x_end,plus_x_end"
)
os.environ.setdefault(
    "BATTERY_IMSI_MANDATORY_POSE_JOINTS",
    "|".join((
        "-1.456600,0.688500,0.456300,1.148900,1.643600,-0.092400",
        "-1.363300,-0.163800,1.886800,-0.892500,1.429700,-1.758100",
        "-1.356600,-0.196000,1.888900,-0.841800,1.427400,1.428100",
    )),
)
os.environ.setdefault(
    "BATTERY_IMSI_MANDATORY_POSE_TCP",
    "|".join((
        "-0.029356,-0.653815,0.510113,0.912079,-0.000072,-0.003721,0.409998",
        "-0.025947,-0.480111,0.529364,0.276826,-0.659670,-0.286605,0.637229",
        "-0.025458,-0.468771,0.535613,0.285891,0.651600,0.290408,0.639802",
    )),
)
# All remaining candidates are preflighted and the run is not truncated.
os.environ.setdefault("BATTERY_IMSI_PREFLIGHT_ONLY_FORCED", "0")
os.environ.setdefault("BATTERY_IMSI_MAX_NBV_VIEWS", "0")
# Stop as soon as measured top and side both exceed 90%, even with a
# mandatory view or a quality face left; those are reported for info only.
os.environ.setdefault("BATTERY_IMSI_STOP_AT_COVERAGE", "1")
# battery_rb_20260929_123052: after frame 4 the -Y face was 0.4% seen and
# only local_x_pos_150/120deg face it; voxel entropy spent 2 views (~2.7 s
# excl. render, +0.3/+0.07%p) on diagonal poses first. A face seen < 50%
# restricts the next view to candidates facing it within 60 deg.
os.environ.setdefault("BATTERY_IMSI_FACE_DEFICIT_FOCUS", "1")
os.environ.setdefault("BATTERY_IMSI_FACE_DEFICIT_MIN_COVERAGE", "0.50")
os.environ.setdefault("BATTERY_IMSI_FACE_DEFICIT_MAX_INCIDENCE_DEG", "60")
# Simulation-only zoom of Zivid_camera_01 (focal length x1.2 -> fx 2544 px,
# pixels per area x1.44). All three mandatory views stay in frame at x1.2. Quality density thresholds are rescaled by (fx / 2120)^2.
os.environ.setdefault("BATTERY_ZIVID_ZOOM", "1.2")
# Recall is reported with the frontal-pixel denominator at the configured
# nominal distance (Battery_defect_config DEFECT_REFERENCE_DISTANCE_M = 0.400 m,
# GSD 0.1887 mm/px) and, side by side, at these distances. Only the
# denominator changes: ratio(d) = ratio(0.40) * (d / 0.40)^2.
os.environ.setdefault("BATTERY_RECALL_COMPARE_DISTANCES_M", "0.41,0.45")
# Resolution-aware coverage per face (+Z, +/-X, +/-Y): a surfel counts once
# seen with (0.40/d)^2 * cos(incidence) >= 0.5, the recall denominator's
# basis. The run continues after top/side 90% until every face reaches the
# target, a targeted face gains < 0.5%p (excluded), or 4 extra views.
os.environ.setdefault("BATTERY_IMSI_QUALITY_COVERAGE", "1")
os.environ.setdefault("BATTERY_IMSI_QUALITY_COVERAGE_TARGET", "0.50")
os.environ.setdefault("BATTERY_IMSI_QUALITY_MIN_DENSITY", "0.50")
os.environ.setdefault("BATTERY_IMSI_QUALITY_REFERENCE_DISTANCE_M", "0.40")
os.environ.setdefault("BATTERY_IMSI_QUALITY_MAX_VIEWS", "4")
os.environ.setdefault("BATTERY_IMSI_MIN_VIEW_DIRECTION_LOCAL_Z", "-0.35")
os.environ.setdefault("BATTERY_IMSI_ROUTE_PLANNER", "1")
os.environ.setdefault("BATTERY_IMSI_ROUTE_BEAM_WIDTH", "512")
os.environ.setdefault("BATTERY_IMSI_ROUTE_BRANCH_WIDTH", "15")
os.environ.setdefault("BATTERY_IMSI_RUNTIME_ROUTE_BEAM_WIDTH", "128")
os.environ.setdefault("BATTERY_IMSI_RUNTIME_ROUTE_BRANCH_WIDTH", "10")
os.environ.setdefault("BATTERY_IMSI_ROUTE_REACHABLE_TARGET_RATIO", "0.98")
os.environ.setdefault("BATTERY_IMSI_ROUTE_COMMIT_VIEWS", "2")
os.environ.setdefault("BATTERY_IMSI_ROUTE_MAX_DEFERRALS", "2")
os.environ.setdefault("BATTERY_IMSI_TOP_VIEW_MAX_WEIGHT", "3.0")
os.environ.setdefault("BATTERY_IMSI_FINISH_TOP_FOCUS", "1")
os.environ.setdefault("BATTERY_IMSI_FINISH_TOP_MIN_LOCAL_Z", "0.15")
os.environ.setdefault("BATTERY_IMSI_EMPIRICAL_VISIBILITY_CACHE", "1")
os.environ.setdefault("BATTERY_IMSI_EMPIRICAL_MIN_HIT_RATIO", "0.50")
os.environ.setdefault(
    "BATTERY_IMSI_EMPIRICAL_BOOTSTRAP_MIN_CACHED_POSES",
    "6",
)
os.environ.setdefault(
    "BATTERY_IMSI_EMPIRICAL_BOOTSTRAP_POSE_ORDER",
    ",".join(
        (
            "tilt_x_pos_30_z_neg_45",
            "tilt_x_pos_45_z_pos_90",
            "local_x_pos_120deg",
            "local_x_neg_150deg",
            "tilt_y_pos_45_z_pos_90",
            "tilt_y_neg_30_z_neg_45",
        )
    ),
)
os.environ.setdefault("BATTERY_IMSI_JOINT_TOLERANCE_RAD", "0.05")
os.environ.setdefault("BATTERY_IMSI_FINAL_POSE_TIMEOUT_S", "3.0")
os.environ.setdefault("BATTERY_IMSI_FINAL_POSE_STABLE_UPDATES", "3")
os.environ.setdefault("BATTERY_IMSI_TOP_SIDE_ONLY_TERMINATION", "1")
os.environ.setdefault("BATTERY_IMSI_TOP_SIDE_COVERAGE_TARGET", "0.90")
os.environ.setdefault("BATTERY_NBV_SURFACE_COVERAGE_TARGET", "0.0")
os.environ.setdefault("BATTERY_NBV_TOP_SURFACE_COVERAGE_TARGET", "0.90")
os.environ.setdefault("BATTERY_NBV_SIDE_SURFACE_COVERAGE_TARGET", "0.90")
os.environ.setdefault(
    "BATTERY_BATTERYCAP_BOTTOM_SURFACE_COVERAGE_TARGET",
    "0.0",
)
os.environ.setdefault("BATTERY_NBV_SURFACE_TOP_SCORE_WEIGHT", "3.0")
os.environ.setdefault("BATTERY_NBV_SURFACE_SIDE_SCORE_WEIGHT", "1.0")
os.environ.setdefault("BATTERY_NBV_SURFACE_BOTTOM_SCORE_WEIGHT", "0.0")
os.environ.setdefault("BATTERY_NBV_MIN_SURFACE_COVERAGE_GAIN", "0.005")
os.environ.setdefault(
    "BATTERY_NBV_MIN_SURFACE_COVERAGE_GAIN_CONSECUTIVE",
    "2",
)

if os.environ["BATTERY_IMSI_TOP_SIDE_ONLY_TERMINATION"].strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}:
    # 성공 종료는 기존 실측 top/side 기준을 유지한다. Route/Beam의
    # node gain은 arm runner에서 UNKNOWN voxel entropy로 별도 계산한다.
    _top_side_target = os.environ["BATTERY_IMSI_TOP_SIDE_COVERAGE_TARGET"]
    os.environ["BATTERY_NBV_SURFACE_COVERAGE_TARGET"] = "0.0"
    os.environ["BATTERY_NBV_TOP_SURFACE_COVERAGE_TARGET"] = _top_side_target
    os.environ["BATTERY_NBV_SIDE_SURFACE_COVERAGE_TARGET"] = _top_side_target
    os.environ["BATTERY_BATTERYCAP_BOTTOM_SURFACE_COVERAGE_TARGET"] = "0.0"
    os.environ["BATTERY_NBV_SURFACE_BOTTOM_SCORE_WEIGHT"] = "0.0"

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ORIGINAL_SCRIPT = os.path.join(_THIS_DIR, "Battery_rb.py")

print(
    "[IMSI][DEBUG] mandatory views: "
    f"{os.environ['BATTERY_IMSI_MANDATORY_POSE_NAMES']}, "
    f"zoom x{os.environ['BATTERY_ZIVID_ZOOM']}"
)
print(
    "[IMSI] Cached-joint NBV experiment enabled: "
    f"success limit={os.environ['BATTERY_IMSI_PRECOMPUTE_VIEW_LIMIT']}, "
    f"travel optimization=information top "
    f"{os.environ['BATTERY_IMSI_TRAJECTORY_TOP_K']}, "
    f"score floor="
    f"{float(os.environ['BATTERY_IMSI_TRAJECTORY_MIN_SCORE_RATIO']) * 100:.0f}% of best, "
    f"cycle overhead="
    f"{float(os.environ['BATTERY_IMSI_CYCLE_FIXED_OVERHEAD_S']):.2f}s, "
    f"joint-path penalty="
    f"{float(os.environ['BATTERY_IMSI_JOINT_PATH_PENALTY_S_PER_RAD']):.2f}s/rad, "
    f"joint-time scale={os.environ['BATTERY_IMSI_JOINT_TIME_SCALE']}, "
    f"min-joint route order={os.environ['BATTERY_IMSI_ROUTE_ORDER_TSP']}, "
    "side local-Z>="
    f"{os.environ['BATTERY_IMSI_INSPECTION_SIDE_MIN_LOCAL_Z']}, "
    f"approach={os.environ['BATTERY_IMSI_APPROACH_DISTANCES_M']}m, "
    f"top close-up={os.environ['BATTERY_IMSI_TOP_CLOSEUP']}, "
    "quality coverage="
    f"{os.environ['BATTERY_IMSI_QUALITY_COVERAGE']}"
    f"(>={float(os.environ['BATTERY_IMSI_QUALITY_COVERAGE_TARGET']) * 100:.0f}%/face), "
    "minimum local view Z="
    f"{float(os.environ['BATTERY_IMSI_MIN_VIEW_DIRECTION_LOCAL_Z']):.2f}, "
    f"route planner={os.environ['BATTERY_IMSI_ROUTE_PLANNER']}, "
    "node score=visible UNKNOWN voxel entropy, "
    f"measured top/side next-view constraint={os.environ['BATTERY_IMSI_INSPECTION_FOCUS']}, "
    f"+Z selection weight<= {os.environ['BATTERY_IMSI_TOP_VIEW_MAX_WEIGHT']} "
    "until measured top target, "
    f"finish-top focus={os.environ['BATTERY_IMSI_FINISH_TOP_FOCUS']}, "
    "ray geometry cache=64 MiB, "
    "graph update=latest occupancy after every capture, "
    f"beam={os.environ['BATTERY_IMSI_ROUTE_BEAM_WIDTH']}/"
    f"{os.environ['BATTERY_IMSI_ROUTE_BRANCH_WIDTH']}, "
    f"runtime beam={os.environ['BATTERY_IMSI_RUNTIME_ROUTE_BEAM_WIDTH']}/"
    f"{os.environ['BATTERY_IMSI_RUNTIME_ROUTE_BRANCH_WIDTH']}, "
    "route commit/defer="
    f"{os.environ['BATTERY_IMSI_ROUTE_COMMIT_VIEWS']}/"
    f"{os.environ['BATTERY_IMSI_ROUTE_MAX_DEFERRALS']}, "
    "empirical visibility="
    f"{os.environ['BATTERY_IMSI_EMPIRICAL_VISIBILITY_CACHE']}, "
    "empirical votes>="
    f"{float(os.environ['BATTERY_IMSI_EMPIRICAL_MIN_HIT_RATIO']) * 100:.0f}%, "
    "reachable target ratio="
    f"{float(os.environ['BATTERY_IMSI_ROUTE_REACHABLE_TARGET_RATIO']) * 100:.0f}%, "
    "completion target(top/side)="
    f"{float(os.environ['BATTERY_IMSI_TOP_SIDE_COVERAGE_TARGET']) * 100:.0f}%/"
    f"{float(os.environ['BATTERY_IMSI_TOP_SIDE_COVERAGE_TARGET']) * 100:.0f}%, "
    "top-side-only termination="
    f"{os.environ['BATTERY_IMSI_TOP_SIDE_ONLY_TERMINATION']}, "
    "surface weights(top/side/bottom)="
    f"{os.environ['BATTERY_NBV_SURFACE_TOP_SCORE_WEIGHT']}/"
    f"{os.environ['BATTERY_NBV_SURFACE_SIDE_SCORE_WEIGHT']}/"
    f"{os.environ['BATTERY_NBV_SURFACE_BOTTOM_SCORE_WEIGHT']}, "
    f"minimum coverage gain="
    f"{float(os.environ['BATTERY_NBV_MIN_SURFACE_COVERAGE_GAIN']) * 100:.2f}%p, "
    f"consecutive="
    f"{os.environ['BATTERY_NBV_MIN_SURFACE_COVERAGE_GAIN_CONSECUTIVE']}, "
    "success criterion=joint tolerance + actual camera framing + grip"
)
runpy.run_path(_ORIGINAL_SCRIPT, run_name="__main__")
