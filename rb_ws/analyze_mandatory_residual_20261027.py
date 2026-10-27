"""Residual-region analysis after the three mandatory views (2026-10-27).

Question: if the three hand-set mandatory views ran first, how much of the
remaining top/side surface does each further view observe, how fast, and
does the mesh-free frontier gain track that GT change?

Works offline on the run that wrote zivid_capture/ (depth + manifest):

* GT: per-frame directional surfel masks (CAD surfels matched to depth,
  same rule as the runner). Coverage is a union, so the state after
  "frame 0 + 3 mandatory views" does not depend on their order.
* Residual = top+side surfels still unseen after that baseline; side is
  the legs' outer faces only (as in the runner).
* Frontier: Battery_frontier_map_20261027 replayed in mandatory-first order
  (part pixels from the part bbox; the runner uses the semantic mask).
* Uncaptured candidates: GT visibility predicted with a surfel z-buffer
  (no gripper occlusion), validated on the captured frames first.
* Time: the planner's edge cost 1.336 * max|dq|/vmax + 0.70 s per view.

    python3 rb_ws/analyze_mandatory_residual_20261027.py --log run_logs/<run>.log
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from Battery_coverage_metrics import coverage_exclusion_masks  # noqa: E402
from Battery_frontier_map_20261027 import FrontierMap  # noqa: E402

_spec = importlib.util.spec_from_file_location("viz", HERE / "visualize_battery_nbv_system.py")
viz = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(viz)

CAPTURE = viz.CAPTURE_DIR
IMAGE = (2448, 2048)
CV2USD = np.diag([1.0, -1.0, -1.0, 1.0])
FRONTIER_BOX = ([-0.17, -0.04, 0.41], [0.17, 0.04, 0.525])  # NBV ROI +- 1 cm, grid (136, 32, 46)
PART_LO, PART_HI = viz.BBOX_LO - 0.003, viz.BBOX_HI + 0.003
TARGET = 0.90
STOP_RATIO, STOP_MIN = 0.15, 30
VMAX, TIME_SCALE, OVERHEAD = 3.14, 1.336, 0.70
MIN_LOCAL_Z = 0.0
FACE_ORDER = ("+Z", "+X", "-X", "+Y", "-Y")
MANDATORY_TCP = {
    "mandatory_top_frontal": [-0.029356, -0.653815, 0.510113, 0.912079, -0.000072, -0.003721, 0.409998],
    "mandatory_minus_x_end": [-0.025947, -0.480111, 0.529364, 0.276826, -0.659670, -0.286605, 0.637229],
    "mandatory_plus_x_end": [-0.025458, -0.468771, 0.535613, 0.285891, 0.651600, 0.290408, 0.639802],
}


def face_labels(points, normals):
    """+Z for top (n_z >= 0.5), dominant horizontal axis for side, '' otherwise.

    As in the runner, coverage_exclusion_masks surfels get no label: side
    keeps the legs' outer faces only, top drops buried CAD layers.
    """
    labels = np.full(len(normals), "", dtype=object)
    keep = ~coverage_exclusion_masks(points, normals)["excluded"]
    side = (np.abs(normals[:, 2]) < 0.5) & keep
    labels[(normals[:, 2] >= 0.5) & keep] = "+Z"
    x_major = np.abs(normals[:, 0]) >= np.abs(normals[:, 1])
    labels[side & x_major & (normals[:, 0] > 0)] = "+X"
    labels[side & x_major & (normals[:, 0] <= 0)] = "-X"
    labels[side & ~x_major & (normals[:, 1] > 0)] = "+Y"
    labels[side & ~x_major & (normals[:, 1] <= 0)] = "-Y"
    return labels


def tcp_matrix(position, quat_xyzw):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(quat_xyzw).as_matrix()
    T[:3, 3] = position
    return T


def move_time(q_from, q_to):
    return TIME_SCALE * float(np.max(np.abs(np.subtract(q_to, q_from)))) / VMAX + OVERHEAD


def project(points, object_from_camera_usd, K):
    cam_from_object = np.linalg.inv(object_from_camera_usd)
    pc = points @ cam_from_object[:3, :3].T + cam_from_object[:3, 3]
    depth = -pc[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = K[0, 0] * pc[:, 0] / depth + K[0, 2]
        v = K[1, 2] - K[1, 1] * pc[:, 1] / depth
    inside = (depth > 0.05) & (u >= 0) & (u < IMAGE[0]) & (v >= 0) & (v < IMAGE[1])
    return depth, u, v, inside


def predict_surfel_visibility(points, normals, object_from_camera_usd, K, occluders=None, down=8, tol=0.003):
    """Front-facing surfels in the image that win a coarse z-buffer of the part and the gripper."""
    depth, u, v, inside = project(points, object_from_camera_usd, K)
    sensor = object_from_camera_usd[:3, 3]
    view = sensor - points
    cos = np.einsum("ij,ij->i", normals, view) / np.linalg.norm(view, axis=1)
    facing = cos > 0.05
    # An oblique surface spans more than `tol` of depth inside one coarse cell.
    slope = np.sqrt(np.clip(1 - cos**2, 0, 1)) / np.clip(cos, 0.1, 1)
    tol = tol + down * np.clip(depth, 0, None) / K[0, 0] * slope
    w, h = IMAGE[0] // down, IMAGE[1] // down
    cell = np.full(len(points), -1)
    cell[inside] = (v[inside] // down).astype(int) * w + (u[inside] // down).astype(int)
    zbuf = np.full(w * h, np.inf)
    np.minimum.at(zbuf, cell[inside], depth[inside])
    if occluders is not None and len(occluders):
        od, ou, ov, oin = project(occluders, object_from_camera_usd, K)
        np.minimum.at(zbuf, (ov[oin] // down).astype(int) * w + (ou[oin] // down).astype(int), od[oin])
    visible = np.zeros(len(points), bool)
    visible[inside] = depth[inside] <= zbuf[cell[inside]] + tol[inside]
    return visible & facing


def depth_points_object(depth, K, object_from_camera_cv):
    d = depth.astype(float)
    vv, uu = np.mgrid[0:d.shape[0], 0:d.shape[1]]
    cam = np.stack([(uu - K[0, 2]) / K[0, 0] * d, (vv - K[1, 2]) / K[1, 1] * d, d, np.ones_like(d)]).reshape(4, -1)
    return (object_from_camera_cv @ cam)[:3].T.reshape(d.shape + (3,))


def part_pixel_mask(depth, K, object_from_camera_cv):
    p = depth_points_object(depth, K, object_from_camera_cv)
    return np.isfinite(depth) & np.all((p >= PART_LO) & (p <= PART_HI), axis=-1)


def gripper_points(frames, K, obj_from_cam, margin=0.05, leaf=0.0015):
    """Non-part depth points within `margin` of the part: the gripper, fixed in the object frame."""
    cells = []
    for f in frames:
        depth = np.load(CAPTURE / f"depth_{f['id']:06d}.npy")
        p = depth_points_object(depth, K, obj_from_cam[f["pose"]] @ CV2USD)[np.isfinite(depth)]
        near = np.all((p >= PART_LO - margin) & (p <= PART_HI + margin), axis=1)
        part = np.all((p >= PART_LO) & (p <= PART_HI), axis=1)
        cells.append(np.unique(np.floor(p[near & ~part] / leaf).astype(np.int64), axis=0))
    return (np.unique(np.concatenate(cells), axis=0) + 0.5) * leaf


BLUE, ORANGE, GRAY, INK, MUTED = "#2a78d6", "#eb6834", "#c9c8c2", "#0b0b0b", "#52514e"


def plot(path, stamp, face_totals, baseline_faces, timeline, gt_plan, names, frontier, gt, sees_minus_y, rho):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4), gridspec_kw={"width_ratios": [1, 1.15, 1.15]})
    fig.patch.set_facecolor("#fcfcfb")
    for ax in axes:
        ax.set_facecolor("#fcfcfb")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_color(GRAY)
        ax.spines["bottom"].set_color(GRAY)
        ax.tick_params(colors=MUTED)
        ax.grid(axis="y", color="#ecebe7", lw=0.8)
        ax.set_axisbelow(True)

    ax = axes[0]
    seen = np.divide(100 * baseline_faces, face_totals, out=np.zeros(len(face_totals)), where=face_totals > 0)
    x = np.arange(len(FACE_ORDER))
    ax.bar(x, seen, width=0.6, color=BLUE, label="관측됨")
    ax.bar(x, 100 - seen, bottom=seen, width=0.6, color=GRAY, label="남은 영역", edgecolor="#fcfcfb", lw=2)
    for xi, s_, left in zip(x, seen, face_totals - baseline_faces):
        ax.text(xi, 101.5, f"{left}", ha="center", va="bottom", color=INK, fontsize=9)
    ax.set_xticks(x, FACE_ORDER)
    ax.set_ylim(0, 125)
    ax.set_ylabel("면별 관측률 (%)", color=MUTED)
    ax.set_title("(a) frame 0 + 필수 3개 직후\n막대 위 숫자 = 남은 surfel 수", color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, loc="upper right", ncol=2, fontsize=9)

    ax = axes[1]
    for (label, rows, style) in (("실제 실행", timeline, "-"), ("GT 기준 최선 (예측)", gt_plan, "--")):
        t = [r[3] for r in rows]
        ax.plot(t, [r[1] for r in rows], style, color=BLUE, lw=2, marker="o", ms=6, label=f"top · {label}")
        ax.plot(t, [r[2] for r in rows], style, color=ORANGE, lw=2, marker="o", ms=6, label=f"side · {label}")
    for name, tp, sp, t in timeline[1:]:
        ax.annotate(name.replace("local_", ""), (t, sp), textcoords="offset points", xytext=(4, -12),
                    fontsize=8, color=MUTED)
    for name, tp, sp, t in gt_plan[1:]:
        ax.annotate(name.replace("local_", ""), (t, sp), textcoords="offset points", xytext=(4, 5),
                    fontsize=8, color=MUTED)
    ax.axhline(100 * TARGET, color=MUTED, lw=1, ls=":")
    ax.text(0.02, 100 * TARGET + 0.4, "목표 90%", color=MUTED, fontsize=8, transform=ax.get_yaxis_transform())
    ax.set_xlabel("필수 3개 완료 후 경과 시간 (s, 이동 비용 모델)", color=MUTED)
    ax.set_ylabel("누적 GT coverage (%)", color=MUTED)
    ax.set_ylim(76, 98)
    ax.set_title("(b) 필수 3개 이후 coverage 변화", color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=2)

    ax = axes[2]
    for flag, color, label in ((False, BLUE, "-Y 면 못 봄"), (True, ORANGE, "-Y 면을 봄")):
        sel = sees_minus_y == flag
        ax.scatter(frontier[sel], gt[sel], s=60, color=color, edgecolor="#fcfcfb", lw=2, label=label, zorder=3)
    for n, fx, gy, flag in zip(names, frontier, gt, sees_minus_y):
        if flag or fx >= 70 or gy >= 2000:
            ax.annotate(n.replace("local_", ""), (fx, gy), textcoords="offset points", xytext=(6, 3),
                        fontsize=8, color=INK)
    ax.set_xlabel("frontier 가시 voxel 수 (NBV 점수)", color=MUTED)
    ax.set_ylabel("남은 영역 GT surfel 예측 관측 수", color=MUTED)
    ax.set_title(f"(c) 필수 3개 직후 후보 {len(names)}개: frontier vs GT (Spearman {rho:.2f})",
                 color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="center left", bbox_to_anchor=(0.55, 0.62))

    fig.suptitle(f"필수 3개 뷰 이후 남은 영역 관측 분석 · run {stamp}", color=INK, fontsize=13, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log", required=True)
    parser.add_argument("--out", default=None, help="output dir (default seungjun/nbv_analysis/residual_<log stamp>)")
    parser.add_argument("--exclude-down-cos", type=float, default=0.5,
                        help="underside frontier rule of the run (<= 0: count the underside too)")
    args = parser.parse_args()

    run = viz.load_run(args.log, CAPTURE)
    frames = run["frames"]
    stamp = Path(args.log).stem.replace("battery_rb_", "")
    out = Path(args.out) if args.out else viz.WORKSPACE / "seungjun" / "nbv_analysis" / f"residual_{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    K = np.load(CAPTURE / "intrinsic.npy")
    lines = []

    def say(text=""):
        print(text)
        lines.append(text)

    # ---------------------------------------------------------------- GT masks
    points, normals, visible = viz.reconstruct_surfels(run, CAPTURE)
    faces = face_labels(points, normals)
    top = faces == "+Z"
    scored = faces != ""
    side = scored & ~top
    face_masks = {f: faces == f for f in FACE_ORDER}

    def pct(mask, group):
        return 100.0 * np.count_nonzero(mask & group) / np.count_nonzero(group)

    say(f"run {stamp}: {len(frames)} frames, top {top.sum()} / side {side.sum()} surfels")
    say("[검증] 오프라인 재구성 누적 커버리지 vs 로그 (top / side; top은 묻힌 CAD 층 제외, side는 "
        "다리 바깥면만 - 이 기준 이전 로그와는 다를 수 있음)")
    seen = np.zeros(len(points), bool)
    for f, vis, cov in zip(frames, visible, run["coverage"]):
        seen |= vis
        say(f"  {f['id']} {f['pose']:24s} offline {pct(seen, top):5.1f} / {pct(seen, side):5.1f}"
            f"   log {cov[1]:5.1f} / {cov[3]:5.1f}")

    is_mandatory = [f["pose"].startswith("mandatory_") for f in frames]
    base_ids = [0] + [i for i, m in enumerate(is_mandatory) if m]
    rest_ids = [i for i in range(1, len(frames)) if not is_mandatory[i]]
    baseline = np.zeros(len(points), bool)
    for i in base_ids:
        baseline |= visible[i]
    residual = scored & ~baseline

    say()
    say("[1] 필수 3개 뷰의 기여 (frame 0 대비 추가 / 다른 뷰가 대신 못 보는 고유분)")
    only0 = visible[0] & scored
    for i in base_ids[1:]:
        others = np.zeros(len(points), bool)
        for j in base_ids:
            if j != i:
                others |= visible[j]
        add0 = visible[i] & scored & ~only0
        unique = visible[i] & scored & ~others
        say(f"  {frames[i]['pose']:24s} frame0 대비 +{add0.sum():5d}  고유 +{unique.sum():5d}  "
            + "  ".join(f"{f}:{np.count_nonzero(unique & face_masks[f]):4d}" for f in FACE_ORDER))
    say(f"  기준선(frame 0 + 필수 3): top {pct(baseline, top):.1f}%  side {pct(baseline, side):.1f}%  "
        f"top+side {pct(baseline, scored):.1f}%")
    say("  남은 영역(residual): " + ", ".join(
        f"{f} {np.count_nonzero(residual & face_masks[f])}/{face_masks[f].sum()}" for f in FACE_ORDER)
        + f"  = {residual.sum()} surfels")
    need_top = max(0, int(np.ceil(TARGET * top.sum())) - np.count_nonzero(baseline & top))
    need_side = max(0, int(np.ceil(TARGET * side.sum())) - np.count_nonzero(baseline & side))
    say(f"  90% 도달까지 필요: top +{need_top}, side +{need_side} surfels")

    # ---------------------------------------------------------- candidate poses
    B, G, fit_err = viz.fit_base_and_grasp(run)
    cam_w = frames[0]["camera"]
    joints = {}
    obj_from_cam = {}
    for a in json.loads((CAPTURE / "object_occupancy" / "nbv_precomputed_joint_views.json").read_text())["attempts"]:
        if not a["success"]:
            continue
        joints[a["pose_name"]] = a["positions"]
        tcp = a.get("target_tcp_pose")
        T = (tcp_matrix(tcp["position"], tcp["orientation_xyzw"]) if tcp
             else tcp_matrix(MANDATORY_TCP[a["pose_name"]][:3], MANDATORY_TCP[a["pose_name"]][3:]))
        obj_from_cam[a["pose_name"]] = np.linalg.inv(B @ T @ G) @ cam_w
    for f, rec in zip(frames, run["manifest"]["frames"]):
        obj_from_cam[f["pose"]] = np.linalg.inv(f["object"]) @ f["camera"]
        joints.setdefault(f["pose"], list(rec["joint_positions_rad"].values()))
    frame_joints = [list(r["joint_positions_rad"].values()) for r in run["manifest"]["frames"]]

    def local_z(name):
        d = obj_from_cam[name][:3, 3] - viz.PIVOT_LOCAL
        return d[2] / np.linalg.norm(d)

    # ------------------------------------------------- z-buffer predictor check
    # Gripper points (fixed in the object frame) occlude; surfels predicted
    # visible in some captured frame yet never observed (under the gripper
    # pad, depth dropouts) are treated as unobservable for new views.
    occluders = gripper_points(frames, K, obj_from_cam)
    predicted = [predict_surfel_visibility(points, normals, obj_from_cam[f["pose"]], K, occluders) for f in frames]

    def hidden_from(ids):
        pred_any = np.zeros(len(points), bool)
        seen_any = np.zeros(len(points), bool)
        for j in ids:
            pred_any |= predicted[j]
            seen_any |= visible[j]
        return pred_any & ~seen_any

    hidden = hidden_from(range(len(frames)))
    say()
    say(f"[검증] z-buffer GT 예측 (gripper 점 {len(occluders)}개 포함, base/grasp fit {fit_err * 1000:.1f} mm)")
    say(f"  모든 촬영 뷰에서 예측상 보이는데 한 번도 관측 안 된 surfel(관측 불가로 처리): "
        f"top {np.count_nonzero(hidden & top)}, side {np.count_nonzero(hidden & side)}")
    say("  남은 영역 기준 leave-one-out: 실제 Δ vs 예측 Δ (예측은 그 뷰를 뺀 나머지로 관측 불가 판정)")
    for i, f in enumerate(frames):
        act = visible[i] & residual
        pred = predicted[i] & residual & ~hidden_from([j for j in range(len(frames)) if j != i])
        allp = predicted[i] & scored
        say(f"  {f['pose']:24s} 전체 recall {np.count_nonzero(allp & visible[i]) / max(1, np.count_nonzero(visible[i] & scored)):.2f} "
            f"precision {np.count_nonzero(allp & visible[i]) / max(1, allp.sum()):.2f} | "
            f"남은영역 실제 {act.sum():5d} 예측 {pred.sum():5d}")

    # ------------------------------------------------- frontier replay (mandatory first)
    fmap = FrontierMap(*FRONTIER_BOX, leaf_size=0.0025, pixel_stride=6,
                       exclude_down_cos=args.exclude_down_cos if args.exclude_down_cos > 0 else None)

    def integrate(i):
        depth = np.load(CAPTURE / f"depth_{frames[i]['id']:06d}.npy")
        T = obj_from_cam[frames[i]["pose"]] @ CV2USD
        fmap.integrate_depth(depth, K, T, part_pixel_mask(depth, K, T))
        return fmap.stats()["frontier"]

    frontier_after = {0: integrate(0)}
    frontier0 = frontier_after[0]
    for i in base_ids[1:]:
        frontier_after[i] = integrate(i)
    stop_threshold = max(STOP_MIN, STOP_RATIO * frontier0)
    say()
    say(f"[2] frontier (필수 우선 순서 재현): frame 0 = {frontier0}, 필수 3개 후 = {frontier_after[base_ids[-1]]}, "
        f"종료 기준 < {stop_threshold:.0f}")

    visited = {frames[i]["pose"] for i in base_ids}
    pool = [n for n in obj_from_cam
            if n not in visited and not n.startswith("mandatory_") and local_z(n) >= MIN_LOCAL_Z and n in joints]
    frontier_ids = fmap.frontier_indices()
    fvis = {n: set(fmap.visible_frontier(frontier_ids, obj_from_cam[n], K, IMAGE).tolist()) for n in pool}
    gt_pred = {n: predict_surfel_visibility(points, normals, obj_from_cam[n], K, occluders) & residual & ~hidden
               for n in pool}

    # ------------------------------------------------- [3] captured views on the residual
    say()
    say("[3] 필수 3개 이후 실제 촬영된 뷰의 남은 영역 관측 (실제 순서, 누적)")
    say("  뷰                        frontier예측 frontier해소 | GT남은영역 Δ   %잔여  " + " ".join(f"{f:>5s}" for f in FACE_ORDER)
        + " | top   side  | 시간(s) Δ/s")
    cum = baseline.copy()
    q_prev = frame_joints[base_ids[-1]]
    t = 0.0
    timeline = [("필수3 완료", pct(cum, top), pct(cum, side), 0.0)]
    for i in rest_ids:
        name = frames[i]["pose"]
        predicted = len(set(fmap.visible_frontier(fmap.frontier_indices(), obj_from_cam[name], K, IMAGE).tolist()))
        before = fmap.stats()["frontier"]
        after = integrate(i)
        gain = visible[i] & scored & ~cum
        dt = move_time(q_prev, frame_joints[i])
        t += dt
        q_prev = frame_joints[i]
        cum |= visible[i]
        timeline.append((name, pct(cum, top), pct(cum, side), t))
        say(f"  {name:24s} {predicted:10d} {before - after:10d} | {gain.sum():10d} {100 * gain.sum() / residual.sum():6.1f}%  "
            + " ".join(f"{np.count_nonzero(gain & face_masks[f]):5d}" for f in FACE_ORDER)
            + f" | {pct(cum, top):5.1f} {pct(cum, side):5.1f} | {dt:5.2f} {gain.sum() / dt:6.0f}")
    say(f"  -> 최종 top {pct(cum, top):.1f}% side {pct(cum, side):.1f}%, 남은 frontier {fmap.stats()['frontier']}")

    # ------------------------------------------------- [4] all candidates after the mandatory views
    names = sorted(pool, key=lambda n: -gt_pred[n].sum())
    fg = np.array([len(fvis[n]) for n in names])
    gg = np.array([gt_pred[n].sum() for n in names])
    rho = spearmanr(fg, gg).correlation
    say()
    say(f"[4] 필수 3개 직후 후보 {len(names)}개: frontier 가시 수 vs GT 남은 영역 예측 (Spearman rho = {rho:.2f})")
    say("  후보                       localZ frontier  GT남은 " + " ".join(f"{f:>5s}" for f in FACE_ORDER) + "  이동(s)")
    q_base = frame_joints[base_ids[-1]]
    for n in names:
        say(f"  {n:26s} {local_z(n):5.2f} {len(fvis[n]):8d} {gt_pred[n].sum():6d} "
            + " ".join(f"{np.count_nonzero(gt_pred[n] & face_masks[f]):5d}" for f in FACE_ORDER)
            + f"  {move_time(q_base, joints[n]):5.2f}")

    # ------------------------------------------------- [5] greedy plans from the residual
    def greedy(score_fn, stop_fn):
        chosen, covered, fcovered, q, t = [], baseline.copy(), set(), q_base, 0.0
        rows = []
        while True:
            left = [n for n in pool if n not in chosen]
            if not left:
                break
            best = max(left, key=lambda n: score_fn(n, covered, fcovered, q))
            if stop_fn(best, covered, fcovered):
                break
            t += move_time(q, joints[best])
            q = joints[best]
            chosen.append(best)
            covered |= gt_pred[best]
            fcovered |= fvis[best]
            rows.append((best, pct(covered, top), pct(covered, side), t, len(fvis[best] - fcovered)))
        return rows

    def gt_reached(covered):
        return pct(covered, top) >= 100 * TARGET and pct(covered, side) >= 100 * TARGET

    gt_rate = greedy(
        lambda n, c, fc, q: np.count_nonzero(gt_pred[n] & ~c) / move_time(q, joints[n]),
        lambda n, c, fc: gt_reached(c) or not np.count_nonzero(gt_pred[n] & ~c))
    fr_rate = greedy(
        lambda n, c, fc, q: len(fvis[n] - fc) / move_time(q, joints[n]),
        lambda n, c, fc: len(fvis[n] - fc) < stop_threshold)
    fr_gain = greedy(
        lambda n, c, fc, q: len(fvis[n] - fc),
        lambda n, c, fc: len(fvis[n] - fc) < stop_threshold)
    say()
    say("[5] 필수 3개 이후 greedy 계획 비교 (GT는 z-buffer 예측, 시간은 필수 3개 완료 시점부터)")
    for label, rows in (("GT 남은영역 / 시간 (상한선)", gt_rate),
                        ("frontier / 시간", fr_rate),
                        ("frontier 개수만 (현재 방식에 가까움)", fr_gain)):
        say(f"  {label}:")
        for name, tp, sp, tt, _ in rows:
            say(f"    {name:26s} top {tp:5.1f}% side {sp:5.1f}%  t={tt:5.2f}s")
        if not rows:
            say("    (추가 뷰 없음)")

    (out / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    np.savez_compressed(
        out / "residual_analysis.npz",
        timeline=np.array([(tp, sp, tt) for _, tp, sp, tt in timeline]),
        timeline_names=np.array([n for n, *_ in timeline]),
        candidate_names=np.array(names), candidate_frontier=fg, candidate_gt=gg,
        candidate_faces=np.array([[np.count_nonzero(gt_pred[n] & face_masks[f]) for f in FACE_ORDER] for n in names]),
        residual_faces=np.array([np.count_nonzero(residual & face_masks[f]) for f in FACE_ORDER]),
        face_totals=np.array([face_masks[f].sum() for f in FACE_ORDER]),
        baseline_faces=np.array([np.count_nonzero(baseline & face_masks[f]) for f in FACE_ORDER]),
        plans=np.array(json.dumps({"gt_rate": gt_rate, "frontier_rate": fr_rate, "frontier_gain": fr_gain})),
    )
    # first two GT-greedy views: past them the predicted top is saturated (gripper)
    gt_timeline = [timeline[0]] + [(n, tp, sp, tt) for n, tp, sp, tt, _ in gt_rate[:2]]
    plot(out / "residual_after_mandatory.png", stamp,
         np.array([face_masks[f].sum() for f in FACE_ORDER]),
         np.array([np.count_nonzero(baseline & face_masks[f]) for f in FACE_ORDER]),
         timeline, gt_timeline, names, fg, gg,
         np.array([np.count_nonzero(gt_pred[n] & face_masks["-Y"]) > 500 for n in names]), rho)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
