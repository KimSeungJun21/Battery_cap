# IMSI inspection correction — 2026-09-29

Acceptance is measured top(+Z) > 90%, measured side > 90%, and
post-preflight elapsed time excluding Replicator render < 10 seconds.
Preflight duration is not an optimization target. The runtime timing boundary
and 3 mm / facing-cosine 0.05 surface-observation test are unchanged.

## Camera calibration

The 2448 × 2048 render product used a camera with horizontal aperture
20.955 and authored vertical aperture 15.2908. With focal length 18.147562,
the previous conversion produced fx=2120.03, fy=2430.63. Isaac's pinhole
camera uses square pixels: the effective vertical aperture is 17.53098 and
fy=fx=2120.03. The incorrect fy distorted depth points in object coordinates,
affecting both occupancy scoring and the measured surfel matches.

`compute_zivid_intrinsic` now synchronizes the live camera's vertical aperture
with render resolution before capture. Frame 0 also refreshes intrinsic.npy
and intrinsic.txt, so an older run's calibration cannot remain beside new depth.
The authored USD asset on disk is not edited.

Reference: [NVIDIA camera API, square pixels and aperture synchronization](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/py/source/deprecated/isaacsim.sensors.camera/docs/index.html).

Diagnostic replay of `run_logs/battery_rb_20260929_091120.log`, saved depth,
camera/object transforms, and `directional_surface_latest.npz`:

| Cumulative through frame | Old K top / side | Square-pixel K top / side |
| --- | --- | --- |
| 0 | 0.00% / 22.51% | 76.49% / 44.02% |
| 2 | 15.39% / 57.74% | 91.27% / 82.35% |
| 5 | 22.18% / 72.66% | 93.01% / 99.52% |

These are diagnostic estimates, not a rerun of the robot. Full semantic masks
were not saved, so the replay used all valid depth points cropped to the
reference surface bounding box plus 10 mm. The original run's measured values
therefore differ slightly even with the old K. They must not be replaced by
these replay numbers. No recorded results were overwritten.

## Selection and runtime

The next-view constraint uses measured top/side deficits, pose directions, and
previous actual viewing directions. It reserves frontal +Z views when top is
the more deficient group, and favors unexplored side azimuths when side is more
deficient. This is a pose prior, not a prediction from hidden CAD surfaces.
Beam node scores and overlap masks remain UNKNOWN voxel entropy. Only its first
depth is constrained; later depths retain the full candidate graph. Each
capture invalidates the prior commitment, and retries/greedy fallback honor
the same next-view constraint.

IMSI defaults now retain the full configured candidate pool, disable debug
geometry/PLY output, keep RGB/depth output, and use lossless PNG compression 1.
On saved rgb_000001.png, three in-memory encoding trials gave median 0.758 s
at compression 6 versus 0.137 s at compression 1. Decoded arrays were identical;
file size increased from 2.32 MiB to 3.00 MiB. This is an encoding benchmark,
not a full pipeline timing guarantee.

The final `[IMSI][ACCEPTANCE]` line reports PASS only if both coverage thresholds
and the <10 s runtime criterion hold. No 10-second cutoff is used to manufacture
a successful result. The 10-frame safety cap still applies.

## Verification

`python3 -m unittest discover -s rb_ws/tests -p 'test_nbv*.py' -v`

16 tests passed, covering camera aspect correction, next-view constraints,
unseen side azimuths, entropy overlap, and cached ray geometry. A fresh Isaac
arm/capture sequence is still required to establish the measured acceptance
result; the saved-data replay does not certify it.
