#!/usr/bin/env python3
"""Visualize one Battery IMSI NBV run as PNG figures and GIF animations.

Everything is rebuilt from what the run left on disk; nothing is re-planned:

    run_logs/battery_rb_*.log            per-frame camera/object poses,
                                         coverage, timing, route decisions
    zivid_capture/depth_*.npy, rgb_*.png the captures themselves
    zivid_capture/intrinsic.npy          K used by that run (includes zoom)
    zivid_capture/object_occupancy/      occupancy snapshots, surfel model,
                                         preflight joint/TCP cache, manifest

Per-frame surfel observations are re-derived from the saved depth with the
same surfel test as the runner (3 mm / facing-cos 0.05), using the depth
points inside the part's bbox + 10 mm. The script prints the replayed
top/side coverage next to the logged values so any mismatch is visible.

Candidate view directions need each candidate's object pose. The robot base
pose and grasp offset are fitted from the run's own frames (TCP from the
manifest vs. object pose from the log), then applied to the preflight TCP
targets.

    python3 rb_ws/visualize_battery_nbv_system.py
    python3 rb_ws/visualize_battery_nbv_system.py --log run_logs/battery_rb_X.log
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import re
import sys
from pathlib import Path

import matplotlib
import matplotlib.font_manager

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Log strings (ACCEPTANCE/RECALL) are Korean; fall back to NanumGothic. The
# matplotlib font cache may predate the font install, so register it here.
for _font in glob.glob("/usr/share/fonts/truetype/nanum/NanumGothic*.ttf"):
    matplotlib.font_manager.fontManager.addfont(_font)
plt.rcParams["font.family"] = ["DejaVu Sans", "NanumGothic"]
plt.rcParams["axes.unicode_minus"] = False
import numpy as np  # noqa: E402
from matplotlib import colors as mcolors  # noqa: E402
from PIL import Image  # noqa: E402
from scipy.optimize import least_squares  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "rb_ws"))
from Battery_coverage_metrics import update_directional_surfel_observations  # noqa: E402

PIVOT_LOCAL = np.array([0.0, 0.0, 0.46675])  # batterycap geometry centre
BBOX_LO = np.array([-0.145, -0.015, 0.430])
BBOX_HI = np.array([0.145, 0.015, 0.5035])
FACES = {"+Z": (0, 0, 1), "+X": (1, 0, 0), "-X": (-1, 0, 0), "+Y": (0, 1, 0), "-Y": (0, -1, 0)}
TARGET = 0.90
# Occupancy_utils: OCCUPANCY_UNKNOWN=-1, FREE=0, OCCUPIED=1
UNKNOWN, FREE, OCCUPIED = -1, 0, 1


# ----------------------------------------------------------------- loading
def parse_matrix(line):
    return np.array(eval(line.split("] ")[-1]), dtype=float).T  # USD row-major -> column


def load_run(log_path, capture_dir):
    lines = Path(log_path).read_text(encoding="utf-8", errors="replace").splitlines()
    manifest = json.loads((capture_dir / "object_occupancy" / "nbv_sequence_manifest.json").read_text())
    frames = []
    for i, line in enumerate(lines):
        match = re.search(r"\[POSE\] Frame: (\d+) (\S+)", line)
        if match:
            frames.append({
                "id": int(match.group(1)), "pose": match.group(2),
                "camera": parse_matrix(lines[i + 2]), "object": parse_matrix(lines[i + 4]),
            })
    # Only frames of this run: the manifest is rewritten by the same run.
    names = [f["pose_name"] for f in manifest["frames"]]
    if [f["pose"] for f in frames] != names:
        raise SystemExit(f"log frames {[f['pose'] for f in frames]} != manifest {names}; "
                         "pass the log of the run that wrote the manifest (--log).")
    coverage = [tuple(float(v) for v in m.groups()) for m in (
        re.search(r"방향성 표면 커버리지: \d+/\d+ = ([\d.]+)% \[top\(\+Z\)=([\d.]+)%, bottom\(-Z\)=([\d.]+)%, side=([\d.]+)%\]", l)
        for l in lines) if m and "[최종]" not in lines[0]]
    final_lines = [l for l in lines if "[최종] 방향성 표면 커버리지" in l]
    if final_lines:
        coverage = coverage[:len(frames)]
    timing = {}
    for line in lines:
        m = re.search(r"nbv_cycle\[(\d+)\] (.*)$", line)
        if m:
            timing[int(m.group(1))] = dict(re.findall(r"(\w+)=([-\d.]+)", m.group(2)))
    capture_timing = {}
    for line in lines:
        m = re.search(r"TIMING\] capture\[(\d+)\] (.*)$", line)
        if m:
            capture_timing[int(m.group(1))] = dict(re.findall(r"(\w+)=([-\d.]+)", m.group(2)))
    defects = {}
    for line in lines:
        m = re.search(r"\[DEFECT GT\] frame (\d+): (.*)$", line)
        if m:
            defects[int(m.group(1))] = dict(
                (k, int(v)) for k, v in re.findall(r"batterycap_(\S+?)=(\d+)px", m.group(2)))
    face_events = [re.sub(r"^.*\[IMSI\]\[FACE\] ", "", l) for l in lines if "[IMSI][FACE]" in l]
    acceptance = next((re.sub(r"^.*\[IMSI\]\[ACCEPTANCE\] ", "", l) for l in lines if "[ACCEPTANCE]" in l), "")
    recall = next((re.sub(r"^.*\[RECALL\] ", "", l) for l in lines if "[RECALL] 전체 Recall" in l), "")
    recall_rows = [re.sub(r"^.*\[RECALL\]\s+", "", l) for l in lines
                   if re.search(r"\[RECALL\]\s+\[[OX]\]", l)]
    return {
        "lines": lines, "frames": frames, "manifest": manifest, "coverage": coverage,
        "timing": timing, "capture_timing": capture_timing, "defects": defects,
        "face_events": face_events, "acceptance": acceptance, "recall": recall,
        "recall_rows": recall_rows,
    }


def reconstruct_surfels(run, capture_dir):
    surf = np.load(capture_dir / "object_occupancy" / "directional_surface_latest.npz")
    points, normals = surf["points"].astype(float), surf["normals"].astype(float)
    K = np.load(capture_dir / "intrinsic.npy")
    visible = []
    for frame in run["frames"]:
        depth = np.load(capture_dir / f"depth_{frame['id']:06d}.npy").astype(float)
        h, w = depth.shape
        v, u = np.mgrid[0:h, 0:w]
        ok = np.isfinite(depth) & (depth > 0) & (depth < 2.0)
        pts_cam = np.stack([((u - K[0, 2]) / K[0, 0] * depth)[ok], (-(v - K[1, 2]) / K[1, 1] * depth)[ok],
                            -depth[ok], np.ones(ok.sum())])
        to_obj = np.linalg.inv(frame["object"]) @ frame["camera"]
        obj_pts = (to_obj @ pts_cam)[:3].T
        obj_pts = obj_pts[np.all((obj_pts >= BBOX_LO - 0.01) & (obj_pts <= BBOX_HI + 0.01), axis=1)]
        sensor = (np.linalg.inv(frame["object"]) @ frame["camera"][:, 3])[:3]
        model = {"points": points, "normals": normals, "observed": np.zeros(len(points), bool)}
        _, mask = update_directional_surfel_observations(model, obj_pts, sensor, 0.003, 0.05, return_visible_mask=True)
        visible.append(mask)
        frame["sensor_object"] = sensor
        frame["view_dir"] = (sensor - PIVOT_LOCAL) / np.linalg.norm(sensor - PIVOT_LOCAL)
        frame["distance"] = float(np.linalg.norm(sensor - PIVOT_LOCAL))
    return points, normals, visible


def fit_base_and_grasp(run):
    """object_world = B @ T_tcp(base) @ G, fitted over the run's frames."""
    tcps = []
    for record in run["manifest"]["frames"]:
        T = np.eye(4)
        T[:3, :3] = Rotation.from_quat(record["tcp_orientation_xyzw"]).as_matrix()
        T[:3, 3] = record["tcp_position"]
        tcps.append(T)
    objects = [f["object"] for f in run["frames"]]

    def mk(p):
        M = np.eye(4)
        M[:3, :3] = Rotation.from_rotvec(p[:3]).as_matrix()
        M[:3, 3] = p[3:]
        return M

    def residual(p):
        B, G = mk(p[:6]), mk(p[6:])
        return np.concatenate([((B @ T @ G) - O)[:3].ravel() for T, O in zip(tcps, objects)])

    best = min((least_squares(residual, np.r_[0, 0, yaw, 0, 0, 0.5, 0, 0, 0, 0, 0, -0.5])
                for yaw in (0, math.pi, math.pi / 2, -math.pi / 2)), key=lambda r: r.cost)
    return mk(best.x[:6]), mk(best.x[6:]), float(np.abs(best.fun).max())


def candidate_directions(capture_dir, camera, B, G):
    data = json.loads((capture_dir / "object_occupancy" / "nbv_precomputed_joint_views.json").read_text())
    out = {}
    for attempt in data["attempts"]:
        name = attempt["pose_name"]
        pose = attempt.get("target_tcp_pose")
        if pose is None:  # hand-set mandatory views carry no TCP target
            continue
        T = np.eye(4)
        T[:3, :3] = Rotation.from_quat(pose["orientation_xyzw"]).as_matrix()
        T[:3, 3] = pose["position"]
        O = B @ T @ G
        pivot = O[:3, :3] @ PIVOT_LOCAL + O[:3, 3]
        d = O[:3, :3].T @ (camera[:3, 3] - pivot)
        entry = out.setdefault(name, {"success": False, "dirs": []})
        entry["dirs"].append(d / np.linalg.norm(d))
        entry["success"] |= bool(attempt["success"])
        if attempt["success"]:
            entry["dir"] = d / np.linalg.norm(d)
    for entry in out.values():
        entry.setdefault("dir", entry["dirs"][-1])
    return out


# --------------------------------------------------------------- plotting
def short(name):
    return (name.replace("mandatory_", "M:").replace("local_", "").replace("diag_", "d:")
            .replace("tilt_", "t:").replace("deg", "°"))


def frame_colors(n):
    cmap = plt.get_cmap("turbo")
    return [cmap(0.08 + 0.84 * i / max(1, n - 1)) for i in range(n)]


def draw_view_sphere(ax, run, candidates, upto=None, title=True, zoom=1.0):
    n = len(run["frames"]) if upto is None else upto + 1
    u, v = np.mgrid[0:2 * np.pi:40j, 0:np.pi:20j]
    ax.plot_wireframe(np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v), color="0.85", lw=0.3)
    # the part, scaled to the unit sphere
    s = 1.0 / 0.30
    lo, hi = (BBOX_LO - PIVOT_LOCAL) * s, (BBOX_HI - PIVOT_LOCAL) * s
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    for a in range(8):
        for b in range(a + 1, 8):
            if np.sum(np.abs(corners[a] - corners[b]) > 1e-9) == 1:
                ax.plot(*zip(corners[a], corners[b]), color="0.3", lw=0.8)
    executed = {f["pose"] for f in run["frames"]}
    ok = [c["dir"] for k, c in candidates.items() if c["success"] and k not in executed]
    bad = [c["dir"] for k, c in candidates.items() if not c["success"]]
    if ok:
        ok = np.array(ok)
        ax.scatter(*ok.T, s=14, c="#4c78a8", alpha=0.55, depthshade=False, label="feasible candidate")
    if bad:
        bad = np.array(bad)
        ax.scatter(*bad.T, s=14, c="0.6", marker="x", alpha=0.7, depthshade=False, label="IK/plan failed")
    cols = frame_colors(len(run["frames"]))
    dirs = [f["view_dir"] for f in run["frames"]]
    for i in range(n):
        d = dirs[i]
        mand = run["frames"][i]["pose"].startswith("mandatory")
        ax.scatter(*d, s=150 if mand else 90, c=[cols[i]], marker="*" if mand else "o",
                   edgecolors="k", linewidths=0.6, depthshade=False, zorder=5)
        ax.text(*(d * 1.13), str(i), fontsize=9, weight="bold", color="k")
        if i > 0:
            a, b = dirs[i - 1], d
            omega = math.acos(float(np.clip(a @ b, -1, 1)))
            ts = np.linspace(0, 1, 30)
            arc = (np.array([a * math.sin((1 - t) * omega) + b * math.sin(t * omega) for t in ts])
                   / max(math.sin(omega), 1e-9)) if omega > 1e-6 else np.array([a, b])
            ax.plot(*arc.T, color=cols[i], lw=2.0)
    ax.scatter([], [], s=90, c="w", edgecolors="k", label="executed view (order)")
    ax.scatter([], [], s=150, c="w", marker="*", edgecolors="k", label="mandatory (hand-set)")
    ax.set_box_aspect((1, 1, 1), zoom=zoom)
    ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1); ax.set_zlim(-1.1, 1.1)
    ax.set_xlabel("object X"); ax.set_ylabel("object Y"); ax.set_zlabel("object Z")
    ax.view_init(elev=24, azim=-58)
    if title:
        ax.set_title("View sphere (object frame): camera direction per view", fontsize=11)


def draw_surfels(ax, points, normals, visible, upto, highlight_new=True, stride=4, zoom=1.0):
    first_seen = np.full(len(points), -1)
    for i, mask in enumerate(visible[:upto + 1]):
        first_seen[(first_seen < 0) & mask] = i
    sel = np.arange(0, len(points), stride)
    p = points[sel] - PIVOT_LOCAL
    fs = first_seen[sel]
    cols = frame_colors(len(visible))
    rgba = np.tile(mcolors.to_rgba("0.88"), (len(sel), 1))
    for i in range(upto + 1):
        rgba[fs == i] = cols[i]
        if highlight_new and i < upto:
            rgba[fs == i, :3] = 0.55 * np.array(cols[i][:3]) + 0.45  # older frames faded
    ax.scatter(p[:, 0], p[:, 1], p[:, 2], c=rgba, s=1.2, depthshade=False, linewidths=0)
    ax.set_box_aspect((0.30, 0.09, 0.075), zoom=1.15 * zoom)  # Y (30 mm) drawn 3x for visibility
    ax.set_xlim(-0.15, 0.15); ax.set_ylim(-0.015, 0.015); ax.set_zlim(-0.0375, 0.0375)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])


def crop_on_part(img, frame, K, zoom):
    """Crop ``img`` by ``zoom`` around the part's projected geometry centre."""
    if zoom <= 1.0:
        return img
    centre = frame["object"][:3, :3] @ PIVOT_LOCAL + frame["object"][:3, 3]
    local = np.linalg.inv(frame["camera"]) @ np.r_[centre, 1.0]
    depth = -local[2]
    sx = img.width / (2.0 * K[0, 2])  # 1.0 for the full-resolution capture
    u = (K[0, 2] + K[0, 0] * local[0] / depth) * sx
    v = (K[1, 2] - K[1, 1] * local[1] / depth) * sx
    w, h = img.width / zoom, img.height / zoom
    left = float(np.clip(u - w / 2, 0, img.width - w))
    top = float(np.clip(v - h / 2, 0, img.height - h))
    return img.crop((round(left), round(top), round(left + w), round(top + h)))


def cumulative_coverage(normals, visible):
    top = normals[:, 2] >= 0.5
    side = np.abs(normals[:, 2]) < 0.5
    rows, cum = [], np.zeros(len(normals), bool)
    for mask in visible:
        cum |= mask
        rows.append((cum.mean(), cum[top].mean(), cum[side].mean()))
    return np.array(rows)


def post_preflight_times(run):
    t = [0.0]
    for i in range(1, len(run["frames"])):
        t.append(float(run["timing"].get(i, {}).get("t_after_preflight_excl_render", np.nan)))
    return np.array(t)


# ---------------------------------------------------------------- outputs
def overview_png(run, candidates, points, normals, visible, out_path, fit_err):
    fig = plt.figure(figsize=(18, 11))
    ax1 = fig.add_subplot(2, 2, 1, projection="3d")
    draw_view_sphere(ax1, run, candidates)
    ax1.legend(loc="upper left", fontsize=8, bbox_to_anchor=(-0.05, 1.0))

    ax2 = fig.add_subplot(2, 2, 2)
    cov = np.array([(c[0], c[1], c[3]) for c in run["coverage"]]) / 100.0
    t = post_preflight_times(run)
    for idx, (label, color) in enumerate([("overall", "0.4"), ("top (+Z)", "#e45756"), ("side", "#4c78a8")]):
        ax2.step(t, cov[:, idx] * 100, where="post", color=color, lw=2, label=label)
        ax2.plot(t, cov[:, idx] * 100, "o", color=color, ms=5)
    ax2.axhline(TARGET * 100, ls="--", color="k", lw=1)
    ax2.text(0.05, TARGET * 100 + 1, "stop when top & side > 90%", ha="left", fontsize=9)
    for i, f in enumerate(run["frames"]):
        ax2.annotate(f"{i}:{short(f['pose'])}", (t[i], cov[i, 2] * 100), textcoords="offset points",
                     xytext=(4, -14 if i % 2 else 8), fontsize=7.5, rotation=0)
    ax2.set_xlabel("time after preflight, excluding render [s]")
    ax2.set_ylabel("measured directional surface coverage [%]")
    ax2.set_ylim(0, 102)
    ax2.set_title("Coverage per capture (frame 0 = lift pose, before preflight)", fontsize=11)
    ax2.legend(loc="lower right")
    ax2.grid(alpha=0.3)

    ax3 = fig.add_subplot(2, 2, 3, projection="3d")
    draw_surfels(ax3, points, normals, visible, len(visible) - 1, highlight_new=False, stride=3)
    ax3.view_init(elev=28, azim=-60)
    cols = frame_colors(len(visible))
    handles = [plt.Line2D([], [], marker="o", ls="", color=cols[i], label=f"{i}: {short(f['pose'])}")
               for i, f in enumerate(run["frames"])]
    handles.append(plt.Line2D([], [], marker="o", ls="", color="0.88", label="never observed"))
    ax3.legend(handles=handles, fontsize=7.5, loc="upper left", title="first observed in frame",
               title_fontsize=8, bbox_to_anchor=(-0.05, 1.02))
    ax3.set_title("Surfels (50k on the CAD surface) coloured by the frame that first saw them", fontsize=11)

    ax4 = fig.add_subplot(2, 2, 4)
    ids = list(range(1, len(run["frames"])))
    move = np.array([float(run["timing"][i].get("move_ok", 0)) for i in ids])
    render = np.array([float(run["capture_timing"][i].get("render", 0)) for i in ids])
    capture = np.array([float(run["timing"][i].get("capture", 0)) for i in ids]) - render
    total = np.array([float(run["timing"][i].get("total", 0)) for i in ids])
    other = total - move - capture - render
    ax4.bar(ids, move, color="#f58518", label="robot move + settle")
    ax4.bar(ids, capture, bottom=move, color="#54a24b", label="capture processing")
    ax4.bar(ids, other, bottom=move + capture, color="#b279a2", label="NBV scoring / route")
    ax4.bar(ids, render, bottom=move + capture + other, color="none", edgecolor="0.4",
            hatch="///", label="Replicator render (excluded)")
    ax4.set_xticks(ids, [f"{i}\n{short(run['frames'][i]['pose'])}" for i in ids], fontsize=7.5)
    ax4.set_ylabel("cycle time [s]")
    occ = []
    for f in run["frames"]:
        z = np.load(CAPTURE_DIR / "object_occupancy" / f"object_occupancy_{f['id']:06d}.npz")
        occ.append(int(np.count_nonzero(z["state"] == UNKNOWN)))
    twin = ax4.twinx()
    twin.plot(range(len(occ)), occ, "k-o", ms=4, lw=1.2, label="UNKNOWN voxels (32³ grid)")
    twin.set_ylabel("UNKNOWN voxels")
    lines = ax4.get_legend_handles_labels()[0] + twin.get_legend_handles_labels()[0]
    ax4.legend(lines, [l.get_label() for l in lines], fontsize=8, loc="upper left")
    t_end = float(run["timing"][ids[-1]].get("t_after_preflight_excl_render", np.nan))
    ax4.set_title(f"Per-cycle time (sum excl. render = {t_end:.2f} s) and remaining UNKNOWN voxels",
                  fontsize=11)

    acceptance = run["acceptance"].replace(", quality", "\nquality")
    fig.suptitle("Battery IMSI next-best-view run   |   " + acceptance, fontsize=10.5)
    fig.text(0.01, 0.005, f"{run['recall']}   |   base/grasp fit residual {fit_err * 1000:.1f} mm   |   "
             f"log: {Path(run['log']).name}", fontsize=8, color="0.3")
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def frames_png(run, capture_dir, out_path):
    n = len(run["frames"])
    cols = min(4, n)
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.4 * cols, 5.5 * rows))
    axes = np.atleast_1d(axes).ravel()
    colors = frame_colors(n)
    for i, f in enumerate(run["frames"]):
        ax = axes[i]
        img = Image.open(capture_dir / f"rgb_{f['id']:06d}.png").convert("RGB")
        img.thumbnail((900, 760))
        ax.imshow(img)
        c = run["coverage"][i]
        seen = run["defects"].get(i, {})
        defect_text = "\n".join(
            f"{k.replace('_00', '').replace('_outer', '').replace('_scratch', ' scratch').replace('_stain', ' stain')}: {v} px"
            for k, v in seen.items()) or "no defect in view"
        ax.set_title(f"frame {i}: {f['pose']}\nd={f['distance']:.3f} m | top {c[1]:.1f}% side {c[3]:.1f}%",
                     fontsize=9, color=colors[i] if i else "k")
        ax.text(0.015, 0.015, defect_text, transform=ax.transAxes, fontsize=7.5, color="w",
                va="bottom", ha="left", bbox=dict(fc="k", alpha=0.6, lw=0))
        ax.axis("off")
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("Zivid captures in NBV order (cumulative coverage after each frame, "
                 "defect pixels in that frame)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=2.2, w_pad=0.4)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _render_panel(draw, size_in, dpi):
    """Render one 3D panel on its own canvas, cropped tight to its content."""
    fig = plt.figure(figsize=size_in, dpi=dpi)
    ax = fig.add_axes((0, 0, 1, 1), projection="3d")
    draw(ax)
    fig.canvas.draw()
    img = Image.frombuffer("RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba()).convert("RGB")
    plt.close(fig)
    ink = np.argwhere(np.asarray(img).min(axis=2) < 235)
    if len(ink):
        (y0, x0), (y1, x1) = ink.min(axis=0), ink.max(axis=0)
        pad = int(0.02 * max(img.size))
        img = img.crop((max(0, x0 - pad), max(0, y0 - pad), min(img.width, x1 + pad), min(img.height, y1 + pad)))
    return img


def _fit_box(img, width, height):
    """Scale ``img`` to fill a width x height white panel, keeping its aspect."""
    scale = min(width / img.width, height / img.height)
    img = img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))), Image.LANCZOS)
    panel = Image.new("RGB", (width, height), "white")
    panel.paste(img, ((width - img.width) // 2, (height - img.height) // 2))
    return panel


def sequence_gif(run, candidates, capture_dir, points, normals, visible, out_path, tmp_dir, zoom=1.0):
    """Per step: view path | surface seen from that camera | the capture.

    ``zoom`` magnifies all three panels: 3D views are scaled inside their own
    canvas (clipped at its edge) and the RGB is cropped around the part.
    """
    from PIL import ImageDraw, ImageFont

    replay = cumulative_coverage(normals, visible)
    t = post_preflight_times(run)
    K = np.load(capture_dir / "intrinsic.npy")
    font_path = next(iter(glob.glob("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")), None)
    title_font = ImageFont.truetype(font_path, 26) if font_path else ImageFont.load_default()
    label_font = ImageFont.truetype(font_path, 18) if font_path else ImageFont.load_default()
    panel_h = 700
    dpi = int(120 * max(1.0, zoom))  # sharper 3D panels when magnified
    paths = []
    for i, f in enumerate(run["frames"]):
        def sphere(ax):
            draw_view_sphere(ax, run, candidates, upto=i, title=False)
            ax.set_axis_off()
            if ax.get_legend():
                ax.get_legend().remove()

        def surfels(ax):
            draw_surfels(ax, points, normals, visible, i, highlight_new=True, stride=3)
            view = f["view_dir"]
            ax.view_init(elev=math.degrees(math.asin(float(np.clip(view[2], -1, 1)))),
                         azim=math.degrees(math.atan2(view[1], view[0])))
            ax.set_axis_off()

        # 3D panels always fill their box (tight crop); zoom sharpens them and
        # magnifies the capture around the part.
        left = _fit_box(_render_panel(sphere, (5.6, 5.6), dpi), panel_h, panel_h)
        middle = _fit_box(_render_panel(surfels, (7.0, 5.6), dpi), 900, panel_h)
        img = Image.open(capture_dir / f"rgb_{f['id']:06d}.png").convert("RGB")
        right = _fit_box(crop_on_part(img, f, K, zoom), 840, panel_h)  # from 2448x2048

        pad, top = 12, 96
        canvas = Image.new("RGB", (left.width + middle.width + right.width + 4 * pad, panel_h + top + pad), "white")
        x = pad
        new = visible[i] & ~np.logical_or.reduce(visible[:i]) if i else visible[0]
        labels = ["camera direction on the view sphere (object frame)",
                  f"surface seen from this camera: bright = new (+{new.mean() * 100:.1f}%)",
                  f"frame {i}: {f['pose']}  (d = {f['distance']:.3f} m)"]
        draw = ImageDraw.Draw(canvas)
        for panel, label in zip((left, middle, right), labels):
            canvas.paste(panel, (x, top))
            draw.text((x + 4, top - 28), label, fill=(40, 40, 40), font=label_font)
            x += panel.width + pad
        c = run["coverage"][i]
        top_ok = "OK" if c[1] > 90 else "--"
        side_ok = "OK" if c[3] > 90 else "--"
        clock = "lift pose (before preflight)" if i == 0 else f"t = {t[i]:.2f} s after preflight (excl. render)"
        title = (f"NBV step {i}/{len(run['frames']) - 1}    top {c[1]:.1f}% [{top_ok}]    "
                 f"side {c[3]:.1f}% [{side_ok}]    overall {c[0]:.1f}%    |    {clock}")
        draw.text((pad, 14), title, fill=(0, 0, 0), font=title_font)
        path = tmp_dir / f"seq_{zoom:g}_{i:02d}.png"
        canvas.save(path)
        paths.append((path, (x - right.width - pad, top, x - pad, top + panel_h)))
    durations = [1600] * (len(paths) - 1) + [3500]
    save_animation([(Image.open(p).convert("RGB"), box) for p, box in paths], durations, out_path)
    return replay


def _gif_frame(frame, photo_box):
    """GIF allows 256 colours per frame. A plain adaptive palette is spent on
    the plots and posterizes the photo, so weight the palette toward the
    photo region and Floyd-Steinberg dither against it."""
    photo = frame.crop(photo_box)
    source = Image.new("RGB", (frame.width + 2 * photo.width, max(frame.height, photo.height)), "white")
    source.paste(frame, (0, 0))
    source.paste(photo, (frame.width, 0))
    source.paste(photo, (frame.width + photo.width, 0))
    palette = source.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
    return frame.quantize(palette=palette, dither=Image.Dither.FLOYDSTEINBERG)


def save_animation(frames, durations, gif_path):
    """Write the GIF plus full-colour animated WebP and APNG next to it."""
    gif = [_gif_frame(frame, box) for frame, box in frames]
    gif[0].save(gif_path, save_all=True, append_images=gif[1:], duration=durations, loop=0)
    rgb = [frame for frame, _ in frames]
    rgb[0].save(gif_path.with_suffix(".webp"), save_all=True, append_images=rgb[1:],
                duration=durations, loop=0, quality=95, method=6)
    rgb[0].save(gif_path.with_suffix(".apng"), format="PNG", save_all=True, append_images=rgb[1:],
                duration=durations, loop=0)


def occupancy_gif(run, capture_dir, out_path, tmp_dir):
    paths = []
    all_states = [np.load(capture_dir / "object_occupancy" / f"object_occupancy_{g['id']:06d}.npz")["state"].ravel()
                  for g in run["frames"]]
    for i, f in enumerate(run["frames"]):
        z = np.load(capture_dir / "object_occupancy" / f"object_occupancy_{f['id']:06d}.npz")
        state, origin, leaf = z["state"], z["origin"].astype(float), float(z["leaf_size"])
        idx = np.argwhere(np.ones_like(state, bool))
        centers = origin + (idx + 0.5) * leaf - PIVOT_LOCAL
        flat = state.ravel()
        near = np.all(np.abs(centers) <= np.array([0.17, 0.05, 0.06]), axis=1)
        fig = plt.figure(figsize=(14, 5.6))
        grid = fig.add_gridspec(1, 2, width_ratios=(1.5, 1))
        ax = fig.add_subplot(grid[0], projection="3d")
        occ = flat == OCCUPIED
        unk = (flat == UNKNOWN) & near
        free = (flat == FREE) & near
        newly = (flat != UNKNOWN) & (all_states[i - 1] == UNKNOWN) & near if i else np.zeros_like(near)
        ax.scatter(*centers[unk].T, s=3, c="0.6", alpha=0.14, depthshade=False, label="UNKNOWN")
        ax.scatter(*centers[free].T, s=4, c="#9ecae9", alpha=0.35, depthshade=False, label="FREE")
        ax.scatter(*centers[occ].T, s=10, c="#e45756", depthshade=False, label="OCCUPIED")
        if np.any(newly):
            ax.scatter(*centers[newly].T, s=22, facecolors="none", edgecolors="k", linewidths=0.7,
                       depthshade=False, label=f"newly observed in frame {i} ({int(newly.sum())})")
        ax.set_box_aspect((0.34, 0.1, 0.12), zoom=1.35)
        ax.set_xlim(-0.17, 0.17); ax.set_ylim(-0.05, 0.05); ax.set_zlim(-0.06, 0.06)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.view_init(elev=22, azim=-62)
        ax.legend(loc="upper left", fontsize=8)
        ax.set_title("1 cm occupancy voxels within 17 x 5 x 6 cm of the part centre", fontsize=10)
        ax2 = fig.add_subplot(grid[1])
        counts = np.array([[np.count_nonzero(all_states[j] == v) for v in (UNKNOWN, FREE, OCCUPIED)]
                           for j in range(len(run["frames"]))])
        shown = counts[: i + 1]
        ax2.plot(range(i + 1), shown[:, 0], "o-", color="0.35", label="UNKNOWN (entropy, 1 bit each)")
        ax2.plot(range(i + 1), shown[:, 1], "o-", color="#4c78a8", label="FREE")
        ax2.set_xlim(-0.3, len(run["frames"]) - 0.7)
        ax2.set_ylim(0, counts[:, :2].max() * 1.08)
        ax2.set_xticks(range(len(run["frames"])), [str(k) for k in range(len(run["frames"]))])
        ax2.set_xlabel("frame")
        ax2.set_ylabel("voxels in 32³ grid")
        ax2b = ax2.twinx()
        ax2b.plot(range(i + 1), shown[:, 2], "s-", color="#e45756", label="OCCUPIED")
        ax2b.set_ylabel("OCCUPIED voxels", color="#e45756")
        ax2b.set_ylim(0, counts[:, 2].max() * 1.15)
        ax2.set_title(f"UNKNOWN {shown[-1, 0]}  FREE {shown[-1, 1]}  OCCUPIED {shown[-1, 2]}", fontsize=10)
        ax2.legend(loc="center left", fontsize=8)
        fig.suptitle(f"frame {i}: {f['pose']}", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        path = tmp_dir / f"occ_{i:02d}.png"
        fig.savefig(path, dpi=85)
        plt.close(fig)
        paths.append(path)
    frames = [Image.open(p).convert("P", palette=Image.ADAPTIVE, colors=255) for p in paths]
    frames[0].save(out_path, save_all=True, append_images=frames[1:],
                   duration=[1400] * (len(frames) - 1) + [3000], loop=0, optimize=True)


CAPTURE_DIR = WORKSPACE / "zivid_capture"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log", type=Path, default=None, help="run log (default: newest)")
    parser.add_argument("--capture-dir", type=Path, default=CAPTURE_DIR)
    parser.add_argument("--out-dir", type=Path, default=WORKSPACE / "seungjun")
    parser.add_argument("--sequence-zooms", type=float, nargs="+", default=[1.0, 1.5, 2.0],
                        help="one nbv_sequence GIF per zoom (1 = no zoom)")
    args = parser.parse_args()
    log = args.log or Path(sorted(glob.glob(str(WORKSPACE / "run_logs" / "battery_rb_*.log")))[-1])
    run = load_run(log, args.capture_dir)
    run["log"] = str(log)
    stamp = re.search(r"battery_rb_(\d+_\d+)", log.name).group(1)
    out_dir = args.out_dir / f"nbv_{stamp}"
    tmp_dir = out_dir / "_frames"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    points, normals, visible = reconstruct_surfels(run, args.capture_dir)
    replay = cumulative_coverage(normals, visible)
    print("frame  pose                        logged top/side    replayed top/side")
    for i, f in enumerate(run["frames"]):
        c = run["coverage"][i]
        print(f"{i:>5}  {f['pose']:26s}  {c[1]:5.1f} / {c[3]:5.1f}      {replay[i, 1] * 100:5.1f} / {replay[i, 2] * 100:5.1f}")
    B, G, fit_err = fit_base_and_grasp(run)
    candidates = candidate_directions(args.capture_dir, run["frames"][0]["camera"], B, G)
    print(f"base/grasp fit residual {fit_err * 1000:.2f} mm; candidates {len(candidates)} "
          f"({sum(c['success'] for c in candidates.values())} feasible)")

    overview_png(run, candidates, points, normals, visible, out_dir / "nbv_overview.png", fit_err)
    frames_png(run, args.capture_dir, out_dir / "nbv_frames.png")
    for zoom in args.sequence_zooms:
        name = "nbv_sequence.gif" if zoom == 1.0 else f"nbv_sequence_zoom{zoom:g}x.gif"
        sequence_gif(run, candidates, args.capture_dir, points, normals, visible,
                     out_dir / name, tmp_dir, zoom=zoom)
    occupancy_gif(run, args.capture_dir, out_dir / "nbv_occupancy.gif", tmp_dir)
    for p in sorted(tmp_dir.glob("*.png")):
        p.unlink()
    tmp_dir.rmdir()
    for p in sorted(out_dir.iterdir()):
        print(f"wrote {p}  ({p.stat().st_size / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
