"""Fine occupancy + frontier voxels for mesh-free NBV decisions (2026-10-27).

The legacy object occupancy (Occupancy_utils, 1 cm) carves FREE space only
along rays that end on the segmented part. Space next to the part, crossed
only by rays that end on the background, therefore stays UNKNOWN forever
(1195 of the 1203 ROI voxels still UNKNOWN at the end of
battery_rb_20260929_165701 were empty), and UNKNOWN entropy can neither rank
views nor stop the sequence.

This map is used only for NBV gain and termination:

* 2.5 mm voxels on a box around the part (the thin batterycap needs it;
  at 1 cm all air is resolved after 2-3 views while GT top+side is 77%).
* Every valid depth ray carves the voxels it crosses (background included).
  A ray endpoint inside the box is OCCUPIED; endpoints of part pixels also
  mark the voxel as part.
* Frontier = UNKNOWN voxels 26-adjacent to part voxels: the estimate of
  surface that has not been observed yet. With ``exclude_down_cos`` set,
  frontier on the underside of the part (the direction from its adjacent
  part voxels points within acos(exclude_down_cos) of object -Z) is not
  counted: the bottom is not inspected and is visible only from below.
* A candidate's gain is the number of frontier voxels inside its image with
  a clear line of sight (OCCUPIED voxels block; UNKNOWN and FREE do not).

Offline on battery_rb_20260929_165701 the frontier went 1659 -> 1462 ->
1228 -> 1237 -> 1048 -> 490 -> 483 while GT top+side went 61.5 -> 69.4 ->
77.1 -> 77.7 -> 85.8 -> 92.8 -> 94.2%.

Coverage evaluation stays on the CAD mesh (Battery_coverage_metrics); this
module never reads the mesh. It is pure numpy so it runs in Isaac's python.
"""

from __future__ import annotations

import math

import numpy as np

UNKNOWN, FREE, OCCUPIED = -1, 0, 1


def _dilate26(mask):
    """Binary dilation with a 3x3x3 cube, numpy only."""
    padded = np.pad(mask, 1)
    out = np.zeros_like(mask)
    sx, sy, sz = mask.shape
    for dx in (0, 1, 2):
        for dy in (0, 1, 2):
            for dz in (0, 1, 2):
                out |= padded[dx:dx + sx, dy:dy + sy, dz:dz + sz]
    return out


_NEIGHBOUR_OFFSETS = [
    (dx, dy, dz)
    for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
    if (dx, dy, dz) != (0, 0, 0)
]


def _away_from_neighbours(mask):
    """Per voxel, sum of unit offsets pointing away from 26-neighbours in mask."""
    padded = np.pad(mask, 1)
    sx, sy, sz = mask.shape
    away = np.zeros(mask.shape + (3,), dtype=np.float32)
    for dx, dy, dz in _NEIGHBOUR_OFFSETS:
        neighbour = padded[1 + dx:1 + dx + sx, 1 + dy:1 + dy + sy, 1 + dz:1 + dz + sz]
        offset = -np.array((dx, dy, dz), dtype=np.float32) / math.sqrt(dx * dx + dy * dy + dz * dz)
        away += neighbour[..., None] * offset
    return away


def _mask_to_bits(mask):
    packed = np.packbits(np.asarray(mask, dtype=np.uint8), bitorder="little")
    return int.from_bytes(packed.tobytes(), byteorder="little")


class FrontierMap:
    """Log-odds occupancy on a box in the object frame."""

    def __init__(
        self,
        box_min,
        box_max,
        leaf_size=0.0025,
        pixel_stride=6,
        hit_probability=0.7,
        miss_probability=0.4,
        clamp=(-2.0, 3.5),
        max_range_m=3.0,
        exclude_down_cos=None,
    ):
        self.leaf = float(leaf_size)
        self.origin = np.asarray(box_min, dtype=np.float64)
        extent = np.asarray(box_max, dtype=np.float64) - self.origin
        if np.any(extent <= 0.0):
            raise ValueError(f"frontier box must have positive extent: {extent}")
        self.shape = tuple(int(v) for v in np.ceil(extent / self.leaf - 1e-9))
        self.upper = self.origin + np.asarray(self.shape) * self.leaf
        self.pixel_stride = max(1, int(pixel_stride))
        self.hit = math.log(hit_probability / (1.0 - hit_probability))
        self.miss = math.log(miss_probability / (1.0 - miss_probability))
        self.clamp = clamp
        self.max_range = float(max_range_m)
        self.exclude_down_cos = None if exclude_down_cos is None else float(exclude_down_cos)
        self.log_odds = np.zeros(self.shape, dtype=np.float32)
        self.observed = np.zeros(self.shape, dtype=bool)
        self.part_hits = np.zeros(self.shape, dtype=np.uint16)
        self.frames = 0

    # ----------------------------------------------------------- state
    @property
    def voxel_count(self):
        return int(np.prod(self.shape))

    def state(self):
        state = np.full(self.shape, UNKNOWN, dtype=np.int8)
        state[self.observed & (self.log_odds <= 0.0)] = FREE
        state[self.observed & (self.log_odds > 0.0)] = OCCUPIED
        return state

    def occupied(self):
        return self.observed & (self.log_odds > 0.0)

    def part_occupied(self):
        return self.occupied() & (self.part_hits > 0)

    def frontier_mask(self, include_down=False):
        part = self.part_occupied()
        frontier = (~self.observed) & _dilate26(part)
        if self.exclude_down_cos is None or include_down:
            return frontier
        return frontier & ~self._downward(part, frontier)

    def _downward(self, part, frontier):
        """Frontier voxels whose adjacent part lies above them (object +Z up)."""
        away = _away_from_neighbours(part)
        norm = np.linalg.norm(away, axis=-1)
        return frontier & (away[..., 2] < -self.exclude_down_cos * np.maximum(norm, 1e-9))

    def frontier_indices(self):
        return np.flatnonzero(self.frontier_mask().reshape(-1))

    def centers(self, flat_indices):
        idx = np.stack(np.unravel_index(np.asarray(flat_indices, dtype=np.int64), self.shape), axis=1)
        return self.origin + (idx + 0.5) * self.leaf

    def stats(self):
        state = self.state()
        return {
            "unknown": int(np.count_nonzero(state == UNKNOWN)),
            "free": int(np.count_nonzero(state == FREE)),
            "occupied": int(np.count_nonzero(state == OCCUPIED)),
            "part_occupied": int(np.count_nonzero(self.part_occupied())),
            "frontier": int(np.count_nonzero(self.frontier_mask())),
            "frontier_down_excluded": int(
                np.count_nonzero(self.frontier_mask(include_down=True))
                - np.count_nonzero(self.frontier_mask())
            ),
        }

    # ------------------------------------------------------- integration
    def integrate_depth(self, depth, intrinsic, object_from_camera_cv, part_pixel_mask=None):
        """Carve with every valid depth pixel (stride-sampled).

        ``object_from_camera_cv`` maps OpenCV camera coordinates (x right,
        y down, z forward) to the object frame, as in Zivid_capture_utils.
        """
        s = self.pixel_stride
        depth = np.asarray(depth, dtype=np.float64)[::s, ::s]
        rows, cols = np.mgrid[0:depth.shape[0], 0:depth.shape[1]]
        valid = np.isfinite(depth) & (depth > 0.0) & (depth < self.max_range)
        d = depth[valid]
        K = np.asarray(intrinsic, dtype=np.float64)
        u = cols[valid] * s
        v = rows[valid] * s
        cam = np.stack([(u - K[0, 2]) / K[0, 0] * d, (v - K[1, 2]) / K[1, 1] * d, d, np.ones_like(d)])
        T = np.asarray(object_from_camera_cv, dtype=np.float64)
        points = (T @ cam)[:3].T
        part = (
            np.asarray(part_pixel_mask, dtype=bool)[::s, ::s][valid]
            if part_pixel_mask is not None
            else np.zeros(len(points), dtype=bool)
        )
        return self.integrate_rays(T[:3, 3], points, part)

    def integrate_rays(self, sensor_origin, endpoints, part_flags):
        sensor = np.asarray(sensor_origin, dtype=np.float64)
        endpoints = np.asarray(endpoints, dtype=np.float64).reshape(-1, 3)
        part_flags = np.asarray(part_flags, dtype=bool).reshape(-1)
        n_total = self.voxel_count
        free = np.zeros(n_total, dtype=bool)
        occ = np.zeros(n_total, dtype=bool)
        part = np.zeros(n_total, dtype=bool)
        if len(endpoints):
            direction = endpoints - sensor
            length = np.linalg.norm(direction, axis=1)
            keep = length > 1e-9
            direction, length, endpoints, part_flags = (
                direction[keep] / length[keep, None], length[keep], endpoints[keep], part_flags[keep])
            with np.errstate(divide="ignore", invalid="ignore"):
                t1 = (self.origin - sensor) / direction
                t2 = (self.upper - sensor) / direction
            t_enter = np.maximum(np.nanmax(np.minimum(t1, t2), axis=1), 0.0)
            t_exit = np.nanmin(np.maximum(t1, t2), axis=1)
            t_stop = np.minimum(t_exit, length - 0.5 * self.leaf)
            step = 0.5 * self.leaf
            crossing = np.flatnonzero(t_stop > t_enter)
            for chunk in np.array_split(crossing, max(1, len(crossing) // 4000)):
                if len(chunk) == 0:
                    continue
                count = int(np.ceil((t_stop[chunk] - t_enter[chunk]).max() / step)) + 1
                ts = t_enter[chunk, None] + np.arange(count)[None, :] * step
                inside = ts <= t_stop[chunk, None]
                samples = sensor + direction[chunk, None, :] * ts[..., None]
                free[self._flat(samples[inside])] = True
            in_box = np.all((endpoints >= self.origin) & (endpoints < self.upper), axis=1)
            end_flat = self._flat(endpoints[in_box])
            occ[end_flat] = True
            part[end_flat[part_flags[in_box]]] = True
        free &= ~occ
        free = free.reshape(self.shape)
        occ = occ.reshape(self.shape)
        self.log_odds[free] += self.miss
        self.log_odds[occ] += self.hit
        np.clip(self.log_odds, self.clamp[0], self.clamp[1], out=self.log_odds)
        self.observed |= free | occ
        self.part_hits[part.reshape(self.shape)] += 1
        self.frames += 1
        return {"rays": int(len(endpoints)), "carved": int(free.sum()), "hits": int(occ.sum())}

    def _flat(self, points):
        idx = np.floor((np.asarray(points) - self.origin) / self.leaf).astype(np.int64)
        good = np.all((idx >= 0) & (idx < np.asarray(self.shape)), axis=1)
        return np.ravel_multi_index(idx[good].T, self.shape)

    # --------------------------------------------------------- visibility
    def visible_frontier(
        self,
        frontier_flat,
        object_from_camera_usd,
        intrinsic,
        image_size,
        step=None,
        chunk=3000,
    ):
        """Frontier voxels a camera at ``object_from_camera_usd`` would see.

        ``object_from_camera_usd`` is the USD camera pose in the object frame
        (camera looks along -Z, +Y up). Visible = projects inside the image
        and no OCCUPIED voxel between the voxel and the camera inside the box.
        """
        frontier_flat = np.asarray(frontier_flat, dtype=np.int64)
        if len(frontier_flat) == 0:
            return frontier_flat
        T = np.asarray(object_from_camera_usd, dtype=np.float64)
        sensor = T[:3, 3]
        centers = self.centers(frontier_flat)
        cam = (centers - sensor) @ T[:3, :3]  # object -> camera (rotation^T)
        depth = -cam[:, 2]
        K = np.asarray(intrinsic, dtype=np.float64)
        width, height = int(image_size[0]), int(image_size[1])
        with np.errstate(divide="ignore", invalid="ignore"):
            u = K[0, 2] + K[0, 0] * cam[:, 0] / depth
            v = K[1, 2] - K[1, 1] * cam[:, 1] / depth
        in_view = (depth > 0.05) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
        candidates = np.flatnonzero(in_view)
        if len(candidates) == 0:
            return frontier_flat[:0]
        occupied = self.occupied().reshape(-1)
        step = 0.5 * self.leaf if step is None else float(step)
        visible = np.zeros(len(frontier_flat), dtype=bool)
        shape = np.asarray(self.shape)
        for part in np.array_split(candidates, max(1, len(candidates) // chunk)):
            start = centers[part]
            direction = sensor - start
            length = np.linalg.norm(direction, axis=1)
            direction /= length[:, None]
            with np.errstate(divide="ignore", invalid="ignore"):
                t1 = (self.origin - start) / direction
                t2 = (self.upper - start) / direction
            t_exit = np.minimum(np.nanmin(np.maximum(t1, t2), axis=1), length)
            count = int(np.ceil(t_exit.max() / step))
            ts = (np.arange(1, count + 1) * step)[None, :]
            samples = start[:, None, :] + direction[:, None, :] * ts[..., None]
            idx = np.floor((samples - self.origin) / self.leaf).astype(np.int64)
            inside = np.all((idx >= 0) & (idx < shape), axis=2) & (ts <= t_exit[:, None])
            flat = np.where(inside, np.ravel_multi_index(np.moveaxis(np.clip(idx, 0, shape - 1), 2, 0), self.shape), -1)
            own = frontier_flat[part][:, None]
            blocked = inside & (flat != own) & occupied[np.maximum(flat, 0)]
            visible[part] = ~blocked.any(axis=1)
        return frontier_flat[visible]


def frontier_route_problem(frontier_flat, visible_by_name):
    """Route-planner input with one bit per current frontier voxel.

    Same keys as NBV_utils.build_unknown_voxel_visibility_bitsets, so beam
    search, mandatory-node reflection and Held-Karp ordering are unchanged.
    """
    frontier_flat = np.asarray(frontier_flat, dtype=np.int64)
    position = {int(flat): i for i, flat in enumerate(frontier_flat)}
    bits, counts, reachable = {}, {}, 0
    for name, visible in visible_by_name.items():
        mask = np.zeros(len(frontier_flat), dtype=bool)
        for flat in np.asarray(visible, dtype=np.int64):
            mask[position[int(flat)]] = True
        value = _mask_to_bits(mask) if len(mask) else 0
        bits[name] = value
        counts[name] = int(mask.sum())
        reachable |= value
    return {
        "visibility_bits": bits,
        "entropy_bits_by_candidate": counts,
        "reachable_bits": reachable,
        "reachable_entropy_bits": int(reachable.bit_count()),
        "voxel_count": int(len(frontier_flat)),
        "grid_shape": (int(len(frontier_flat)),),
    }


def frontier_candidate_results(visible_by_name):
    """Scorer-compatible records whose score is the visible frontier count."""
    results = []
    for name, visible in visible_by_name.items():
        count = int(len(visible))
        results.append({
            "candidate_name": name,
            "entropy_score": float(count),
            "unknown_entropy_sum": float(count),
            "expected_unknown_visible_count": float(count),
            "unknown_visible_count": count,
            "free_visible_count": 0,
            "occupied_visible_count": 0,
            "visible_voxel_count": count,
            "mean_visible_entropy": 1.0 if count else 0.0,
            "surface_unknown_visible_count": count,
            "directional_surface_unknown_groups": {},
            "score_source": "frontier_voxels",
        })
    return results
