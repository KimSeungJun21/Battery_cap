"""IMSI run with mesh-free NBV decisions (2026-10-27).

Same scenario and settings as Battery_rb_imsi_debug.py (mandatory views,
zoom, approach distances, route planner), except that every DECISION uses
only depth observations:

    2.5 mm voxels + background ray carving + frontier counting
    (rb_ws/Battery_frontier_map_20261027.py)

* NBV gain: number of frontier voxels (UNKNOWN next to part voxels) a
  candidate sees inside its image with a clear line of sight.
* Stop: when no remaining candidate sees at least
  max(30, 15% of the frame-0 frontier) frontier voxels.
* Nothing from below: candidate views with the camera below the part
  (object-frame view Z < 0) and frontier on the underside are ignored.
* Route replan: when a view observed < 30% of the frontier it predicted.

The CAD mesh (directional surfels) is still used for coverage, quality and
the ACCEPTANCE report, but no longer decides anything: the GT-driven rules
of the debug wrapper (next-view constraint, face-deficit rule, quality
phase, +Z weighting, finish-top focus, top-bias replanning) are disabled.

Offline replay of battery_rb_20260929_165701 with this map: the best
remaining gain stayed 880-1397 until local_x_pos_150deg was captured, then
fell to 59 (GT top+side 92.8%; top alone 88.3%).
"""

import math
import os
import runpy

# --- mesh-free decisions ----------------------------------------------------
os.environ.setdefault("BATTERY_IMSI_FRONTIER_MAP", "1")
os.environ.setdefault("BATTERY_IMSI_FRONTIER_LEAF_M", "0.0025")
os.environ.setdefault("BATTERY_IMSI_FRONTIER_PIXEL_STRIDE", "6")
os.environ.setdefault("BATTERY_IMSI_FRONTIER_MARGIN_M", "0.01")
# 15% of the frame-0 frontier. Excluding the underside cut the frame-0
# frontier of battery_rb_20261027_112032 from 1647 to 516 voxels, so 5%
# (=30) would no longer stop after local_x_pos_120deg (best remaining 38 at
# GT top+side 95.5%); the view before it still scored 150. 15% = 77 sits
# between the two, close to the 82 that stopped that run.
os.environ.setdefault("BATTERY_IMSI_FRONTIER_STOP_RATIO", "0.15")
os.environ.setdefault("BATTERY_IMSI_FRONTIER_STOP_MIN_VOXELS", "30")
os.environ.setdefault("BATTERY_IMSI_FRONTIER_REPLAN_MIN_RESOLVED", "0.30")
# Nothing from below: the bottom is not inspected.
#  * candidate views with the camera below the part's horizontal plane
#    (object-frame view direction Z < 0) are dropped for NBV, route and stop
#    (the debug wrapper allows down to -0.35); hand-set mandatory views exempt.
#  * frontier on the underside (adjacent part above it, within 60 deg of -Z)
#    is not counted, so it neither attracts views nor inflates the stop
#    threshold (5% of the frame-0 frontier).
os.environ.setdefault("BATTERY_IMSI_MIN_VIEW_DIRECTION_LOCAL_Z", "0.0")
os.environ.setdefault("BATTERY_IMSI_FRONTIER_EXCLUDE_DOWN_COS", "0.5")

# --- GT-driven decision rules off (GT is reported only) ---------------------
os.environ.setdefault("BATTERY_IMSI_INSPECTION_FOCUS", "0")       # GT top/side deficit
os.environ.setdefault("BATTERY_IMSI_FACE_DEFICIT_FOCUS", "0")     # CAD face groups
os.environ.setdefault("BATTERY_IMSI_TOP_VIEW_MAX_WEIGHT", "1.0")  # GT top coverage
os.environ.setdefault("BATTERY_IMSI_FINISH_TOP_FOCUS", "0")       # GT side reached
os.environ.setdefault("BATTERY_IMSI_QUALITY_MAX_VIEWS", "0")      # quality: report only
# The frontier stop ends the run even if a mandatory view is still pending.
os.environ.setdefault("BATTERY_IMSI_STOP_AT_COVERAGE", "1")

print(
    "[IMSI][FRONTIER] mesh-free decisions: "
    f"leaf={float(os.environ['BATTERY_IMSI_FRONTIER_LEAF_M']) * 1000:.1f}mm, "
    f"pixel stride={os.environ['BATTERY_IMSI_FRONTIER_PIXEL_STRIDE']}, "
    "stop: best remaining visible frontier < max("
    f"{os.environ['BATTERY_IMSI_FRONTIER_STOP_MIN_VOXELS']}, "
    f"{float(os.environ['BATTERY_IMSI_FRONTIER_STOP_RATIO']) * 100:.0f}% of frame-0 frontier), "
    "GT coverage = report/ACCEPTANCE only, "
    f"views from below dropped (local Z >= {os.environ['BATTERY_IMSI_MIN_VIEW_DIRECTION_LOCAL_Z']}), "
    "underside frontier excluded (within "
    f"{math.degrees(math.acos(float(os.environ['BATTERY_IMSI_FRONTIER_EXCLUDE_DOWN_COS']))):.0f} deg of -Z)"
)

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
runpy.run_path(os.path.join(_THIS_DIR, "Battery_rb_imsi_debug.py"), run_name="__main__")
