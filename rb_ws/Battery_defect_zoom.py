#!/usr/bin/env python3
"""Zoomed-in views of each observed defect, from a finished capture run.

Reads what save_zivid_capture() left in the capture directory and writes, per
frame, a labelled overview plus one magnified crop per defect.

Inputs it looks for, per frame id:
    rgb_{id:06d}.png          the captured image - optional
    defect_mask_{id:06d}.png  union of all defect pixels (always written)
    defect_observations.json  per-frame, per-defect pixel counts, and
                              bbox_xyxy / centroid_xy on runs recorded after
                              that field was added

Defect identity is the awkward part. defect_mask_*.png is a union, so a blob
in it carries no name. When the observations file has bbox_xyxy the crop is
exact. Otherwise this falls back to matching each mask blob's pixel count
against the per-defect counts for that frame, which is why the run is
reported as "exact" or "matched by pixel count" in the summary.

Each defect gets a clean crop, which is the captured pixels untouched apart
from a bbox outline drawn clear of the defect, and - unless --overlay says
otherwise - a second "_mask" crop with the defect pixels tinted. The clean one
is what shows whether a scratch is actually visible to a detector; the tint
answers "where exactly did the GT mask land", which is a different question.

Defects observed at fewer than MIN_DEFECT_PIXELS pixels are left out of both
the crops and the overview (--min-pixels), since a 3-pixel sliver magnifies
into nothing useful. They are still counted wherever recall is computed.

Without rgb_*.png there is nothing photographic to magnify and the crops show
the mask alone, which locates a defect but says nothing about its appearance.
Re-run the capture with BATTERY_SAVE_FRAME_IMAGES=1 to get real images.

An rgb_*.png left over from an earlier run is worse than none at all: the file
names carry only a frame id, so a stale image pairs silently with the current
mask and the crop shows the wrong object. Frame images are written in the same
breath as the mask, so anything much older than its mask is from another run
and is rejected. --stale-rgb-seconds sets the tolerance.

    python3 rb_ws/Battery_defect_zoom.py
    python3 rb_ws/Battery_defect_zoom.py --capture-dir ... --out-dir ...

Output layout:
    <out-dir>/YYYY-MM-DD_HH-MM-SS/
        frame_000000_overview.png
        frame_000000_01_DEFECT_NAME.png
        frame_000000_01_DEFECT_NAME_mask.png
        summary.json
        README.md

Frame별 하위 폴더는 만들지 않는다. 모든 이미지가 실행 시각 폴더 한
곳에 모이므로 파일명 앞의 frame_XXXXXX 접두사로 촬영 프레임을 구분한다.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime

import numpy as np
from PIL import Image, ImageDraw

try:
    from scipy import ndimage
except ImportError:
    ndimage = None


DEFAULT_CAPTURE_DIR = "/home/seungjun/isaac_ws/zivid_capture"
DEFAULT_OUT_DIRNAME = "seungjun"
# The overviews also go here as a set, so the whole sweep can be flipped
# through without the per-defect crops in between. Same filenames as the
# copies beside the crops, which stay where they are.
OVERVIEW_DIRNAME = "overviews"

# A scratch is a few hundred pixels in a 2448x2048 frame, so the crop is
# padded out to a readable window and then magnified. The window is square so
# every crop carries the same magnification for a given bbox size; the minimum
# only has to keep a very small defect from filling the whole frame, so it is
# kept tight - at 220 a typical 66x10 scratch came out at 4x, at 120 it is
# 7.5x with the surrounding surface still in view.
CROP_MIN_SIZE = 120
CROP_PAD_RATIO = 1.8
ZOOM_LONG_EDGE = 900
OVERVIEW_LONG_EDGE = 1400

# Defects observed at only a handful of pixels are grazing-angle slivers or
# the last few pixels before an occlusion edge. Magnifying one shows a blob
# a few pixels across and tells you nothing, and they crowd the overview, so
# they are left out of the visualisation. They stay in the recall figures,
# which are computed elsewhere from defect_observations.json.
MIN_DEFECT_PIXELS = 30

# save_zivid_capture() writes rgb_*.png and defect_mask_*.png within the same
# frame, so a gap this large means the two files come from different runs.
STALE_RGB_SECONDS = 600.0

MASK_TINT = (255, 64, 64)
MASK_ALPHA = 0.45
BOX_COLOR = (255, 210, 0)
BOX_COLOR_ALT = (0, 220, 255)


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Write magnified per-defect views from a finished capture run."
        )
    )
    parser.add_argument(
        "--capture-dir",
        default=DEFAULT_CAPTURE_DIR,
        help="Directory save_zivid_capture() wrote to.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help=(
            "Base directory for dated report folders. Defaults to "
            f"<capture-dir>/{DEFAULT_OUT_DIRNAME}. Each execution creates "
            "a YYYY-MM-DD_HH-MM-SS child directory."
        ),
    )
    parser.add_argument(
        "--frames",
        default=None,
        help="Comma-separated frame ids. Default: every frame found.",
    )
    parser.add_argument(
        "--zoom-long-edge",
        type=int,
        default=ZOOM_LONG_EDGE,
        help="Long edge of each magnified crop, in pixels.",
    )
    parser.add_argument(
        "--overlay",
        choices=("both", "clean", "mask"),
        default="both",
        help=(
            "both (default) writes NAME.png untinted and NAME_mask.png with "
            "the defect pixels tinted; clean writes only the untinted crop; "
            "mask only the tinted one."
        ),
    )
    parser.add_argument(
        "--stale-rgb-seconds",
        type=float,
        default=STALE_RGB_SECONDS,
        help=(
            "Reject rgb_*.png older than its defect mask by more than this "
            "many seconds (it is then from an earlier run). Negative "
            "disables the check."
        ),
    )
    parser.add_argument(
        "--min-pixels",
        type=int,
        default=MIN_DEFECT_PIXELS,
        help=(
            "Leave out defects observed at fewer pixels than this "
            f"(default {MIN_DEFECT_PIXELS}). Pass 1 to draw every one."
        ),
    )
    return parser.parse_args(argv)


def create_dated_output_dir(output_root, started_at=None):
    """실행 시각 폴더를 만들고, 같은 초에 겹치면 번호를 붙인다."""
    if started_at is None:
        started_at = datetime.now().astimezone()

    timestamp = started_at.strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(output_root, exist_ok=True)

    for sequence in range(1, 1000):
        suffix = "" if sequence == 1 else f"_{sequence:02d}"
        run_dir = os.path.join(output_root, timestamp + suffix)

        try:
            os.makedirs(run_dir, exist_ok=False)
        except FileExistsError:
            continue

        return run_dir, started_at.isoformat()

    raise RuntimeError(
        f"실행 시각 출력 폴더를 만들 수 없습니다: {output_root}/{timestamp}"
    )


def discover_frame_ids(capture_dir):
    pattern = re.compile(r"^defect_mask_(\d{6})\.png$")
    ids = []

    for name in os.listdir(capture_dir):
        match = pattern.match(name)
        if match:
            ids.append(int(match.group(1)))

    return sorted(ids)


def load_observations(capture_dir):
    path = os.path.join(capture_dir, "defect_observations.json")

    if not os.path.isfile(path):
        return {}

    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (ValueError, OSError):
        return {}

    return {
        int(frame.get("frame_id", -1)): frame.get("defects", {})
        for frame in payload.get("frames", [])
    }


def load_frame_images(capture_dir, frame_id):
    mask_path = os.path.join(
        capture_dir, f"defect_mask_{frame_id:06d}.png"
    )
    rgb_path = os.path.join(capture_dir, f"rgb_{frame_id:06d}.png")

    mask = None
    mask_mtime = None
    if os.path.isfile(mask_path):
        mask = np.array(Image.open(mask_path).convert("L")) > 127
        mask_mtime = os.path.getmtime(mask_path)

    rgb = None
    rgb_mtime = None
    if os.path.isfile(rgb_path):
        rgb = np.array(Image.open(rgb_path).convert("RGB"))
        rgb_mtime = os.path.getmtime(rgb_path)

    return mask, mask_mtime, rgb, rgb_mtime, mask_path, rgb_path


def blobs_from_mask(mask):
    """Connected components of the union mask, largest first."""
    if mask is None or not mask.any():
        return []

    if ndimage is not None:
        labels, count = ndimage.label(mask)
        found = []
        for index in range(1, count + 1):
            component = labels == index
            found.append((int(component.sum()), bbox_of(component)))
        found.sort(key=lambda item: item[0], reverse=True)
        return found

    # No scipy: treat the whole mask as one blob rather than guessing.
    return [(int(mask.sum()), bbox_of(mask))]


def bbox_of(component):
    rows = np.any(component, axis=1)
    columns = np.any(component, axis=0)
    y_indices = np.flatnonzero(rows)
    x_indices = np.flatnonzero(columns)
    return (
        int(x_indices[0]),
        int(y_indices[0]),
        int(x_indices[-1]) + 1,
        int(y_indices[-1]) + 1,
    )


def resolve_defect_boxes(frame_defects, mask, min_pixels):
    """Give every observed defect a box, and say how it was obtained."""
    entries = []
    skipped_small = []
    exact = True
    unmatched_blobs = blobs_from_mask(mask)

    for name, entry in sorted(
        frame_defects.items(),
        key=lambda item: -int(item[1].get("pixels", 0)),
    ):
        pixels = int(entry.get("pixels", 0))

        if pixels < min_pixels:
            skipped_small.append({"name": name, "pixels": pixels})
            continue

        box = entry.get("bbox_xyxy")

        if box is not None:
            entries.append(
                {
                    "name": name,
                    "pixels": pixels,
                    "box": tuple(int(v) for v in box),
                    "centroid": entry.get("centroid_xy"),
                    "source": "bbox_xyxy",
                }
            )
            continue

        exact = False
        best_index = None
        best_delta = None

        for index, (blob_pixels, _) in enumerate(unmatched_blobs):
            delta = abs(blob_pixels - pixels)
            if best_delta is None or delta < best_delta:
                best_delta = delta
                best_index = index

        if best_index is None:
            entries.append(
                {
                    "name": name,
                    "pixels": pixels,
                    "box": None,
                    "centroid": None,
                    "source": "unlocated",
                }
            )
            continue

        blob_pixels, blob_box = unmatched_blobs.pop(best_index)
        entries.append(
            {
                "name": name,
                "pixels": pixels,
                "box": blob_box,
                "centroid": None,
                "source": (
                    f"pixel-count match (blob {blob_pixels}px "
                    f"vs recorded {pixels}px)"
                ),
            }
        )

    return entries, exact, skipped_small


def padded_window(box, image_shape, min_size, pad_ratio):
    height, width = image_shape[:2]
    x0, y0, x1, y1 = box
    center_x = 0.5 * (x0 + x1)
    center_y = 0.5 * (y0 + y1)
    size = max(
        (x1 - x0) * pad_ratio,
        (y1 - y0) * pad_ratio,
        float(min_size),
    )
    half = 0.5 * size

    left = int(round(center_x - half))
    top = int(round(center_y - half))
    right = int(round(center_x + half))
    bottom = int(round(center_y + half))

    # Shift rather than shrink, so the magnification stays constant.
    if left < 0:
        right -= left
        left = 0
    if top < 0:
        bottom -= top
        top = 0
    if right > width:
        left -= right - width
        right = width
    if bottom > height:
        top -= bottom - height
        bottom = height

    return max(0, left), max(0, top), min(width, right), min(height, bottom)


def base_canvas(rgb, mask, shape, tint=True):
    """RGB, optionally with the defect pixels tinted.

    With no RGB the mask has to be drawn whatever `tint` says, otherwise the
    crop would be a blank rectangle.
    """
    if rgb is not None:
        canvas = rgb.astype(np.float32)
    else:
        canvas = np.full(
            (shape[0], shape[1], 3), 18.0, dtype=np.float32
        )
        tint = True

    if tint and mask is not None and mask.any():
        tint = np.array(MASK_TINT, dtype=np.float32)
        if rgb is not None:
            canvas[mask] = (
                (1.0 - MASK_ALPHA) * canvas[mask] + MASK_ALPHA * tint
            )
        else:
            canvas[mask] = tint

    return np.clip(canvas, 0, 255).astype(np.uint8)


def fit_long_edge(image, long_edge):
    width, height = image.size
    scale = long_edge / float(max(width, height))

    if scale <= 1.0:
        return image, 1.0

    return (
        image.resize(
            (max(1, int(round(width * scale))),
             max(1, int(round(height * scale)))),
            Image.NEAREST,
        ),
        scale,
    )


def write_zoom(
    canvas,
    entry,
    window,
    out_path,
    zoom_long_edge,
    has_rgb,
    tinted=False,
):
    left, top, right, bottom = window
    crop = Image.fromarray(canvas[top:bottom, left:right]).convert("RGB")
    crop, scale = fit_long_edge(crop, zoom_long_edge)

    draw = ImageDraw.Draw(crop)
    x0, y0, x1, y1 = entry["box"]
    draw.rectangle(
        [
            (x0 - left) * scale,
            (y0 - top) * scale,
            (x1 - left) * scale,
            (y1 - top) * scale,
        ],
        outline=BOX_COLOR,
        width=3,
    )

    label = (
        f"{entry['name']}  {entry['pixels']}px  "
        f"x{scale:.1f}  bbox={x1 - x0}x{y1 - y0}px"
    )
    if not has_rgb:
        label += "  [MASK ONLY - no rgb_*.png]"
    elif tinted:
        label += "  [GT mask tinted]"
    draw.rectangle([0, 0, crop.size[0], 22], fill=(0, 0, 0))
    draw.text((6, 5), label, fill=(255, 255, 255))
    crop.save(out_path)
    return scale


def write_overview(canvas, entries, out_paths, has_rgb, frame_id):
    """Render the overview once and save it to every given path."""
    image = Image.fromarray(canvas).convert("RGB")
    image, scale = fit_long_edge(image, OVERVIEW_LONG_EDGE)
    if scale == 1.0:
        width, height = image.size
        scale = OVERVIEW_LONG_EDGE / float(max(width, height))
        image = image.resize(
            (max(1, int(round(width * scale))),
             max(1, int(round(height * scale)))),
            Image.LANCZOS,
        )

    draw = ImageDraw.Draw(image)

    for index, entry in enumerate(entries):
        if entry["box"] is None:
            continue
        x0, y0, x1, y1 = entry["box"]
        color = BOX_COLOR if index % 2 == 0 else BOX_COLOR_ALT
        # Pad the drawn box so a 20px scratch is still visible when the
        # whole 2448px frame is squeezed onto 1400px.
        margin = 14
        draw.rectangle(
            [
                x0 * scale - margin,
                y0 * scale - margin,
                x1 * scale + margin,
                y1 * scale + margin,
            ],
            outline=color,
            width=3,
        )
        draw.text(
            (x0 * scale - margin, max(0, y0 * scale - margin - 12)),
            f"{index + 1}",
            fill=color,
        )

    header = f"frame {frame_id}  -  {len(entries)} defect(s) observed"
    if not has_rgb:
        header += "  [MASK ONLY - no rgb_*.png]"
    draw.rectangle([0, 0, image.size[0], 22], fill=(0, 0, 0))
    draw.text((6, 5), header, fill=(255, 255, 255))

    if isinstance(out_paths, (str, bytes, os.PathLike)):
        out_paths = [out_paths]

    for path in out_paths:
        image.save(path)


def main(argv=None):
    options = parse_arguments(argv)
    capture_dir = options.capture_dir
    output_root = options.out_dir or os.path.join(
        capture_dir, DEFAULT_OUT_DIRNAME
    )

    if not os.path.isdir(capture_dir):
        raise SystemExit(f"capture dir이 없습니다: {capture_dir}")

    if options.frames:
        frame_ids = [
            int(token) for token in options.frames.split(",") if token.strip()
        ]
    else:
        frame_ids = discover_frame_ids(capture_dir)

    if not frame_ids:
        raise SystemExit(
            f"defect_mask_*.png를 찾지 못했습니다: {capture_dir}"
        )

    out_dir, started_at = create_dated_output_dir(output_root)
    overview_dir = os.path.join(out_dir, OVERVIEW_DIRNAME)
    os.makedirs(overview_dir, exist_ok=True)

    observations = load_observations(capture_dir)
    if ndimage is None:
        print("[ZOOM] scipy가 없어 blob 분리를 건너뜁니다.")

    summary = {
        "capture_dir": capture_dir,
        "output_root": output_root,
        "out_dir": out_dir,
        "started_at": started_at,
        "frames": [],
    }
    total_crops = 0
    frames_without_rgb = []

    for frame_id in frame_ids:
        (
            mask,
            mask_mtime,
            rgb,
            rgb_mtime,
            mask_path,
            rgb_path,
        ) = load_frame_images(capture_dir, frame_id)

        if mask is None:
            print(f"[ZOOM] frame {frame_id}: mask 없음, 건너뜁니다.")
            continue

        has_rgb = rgb is not None
        stale_age = None

        if has_rgb and rgb.shape[:2] != mask.shape[:2]:
            print(
                f"[ZOOM] frame {frame_id}: rgb {rgb.shape[:2]} != mask "
                f"{mask.shape[:2]} - rgb를 무시합니다 (다른 런의 파일)."
            )
            rgb = None
            has_rgb = False

        if (
            has_rgb
            and options.stale_rgb_seconds >= 0.0
            and mask_mtime is not None
            and rgb_mtime is not None
            and mask_mtime - rgb_mtime > options.stale_rgb_seconds
        ):
            stale_age = mask_mtime - rgb_mtime
            print(
                f"[ZOOM] frame {frame_id}: rgb가 mask보다 "
                f"{stale_age / 3600.0:.1f}시간 오래됐습니다 - 이전 런의 "
                "파일이므로 무시합니다."
            )
            rgb = None
            has_rgb = False

        if not has_rgb:
            frames_without_rgb.append(frame_id)

        frame_defects = observations.get(frame_id, {})
        entries, exact, skipped_small = resolve_defect_boxes(
            frame_defects, mask, options.min_pixels
        )

        frame_prefix = f"frame_{frame_id:06d}"
        clean_canvas = base_canvas(rgb, mask, mask.shape, tint=False)
        mask_canvas = base_canvas(rgb, mask, mask.shape, tint=True)

        overview_name = f"{frame_prefix}_overview.png"
        write_overview(
            clean_canvas,
            entries,
            [
                os.path.join(out_dir, overview_name),
                os.path.join(overview_dir, overview_name),
            ],
            has_rgb,
            frame_id,
        )

        frame_record = {
            "frame_id": frame_id,
            "has_rgb": has_rgb,
            "rgb_path": rgb_path if has_rgb else None,
            "rgb_mtime": rgb_mtime,
            "rgb_rejected_stale_seconds": stale_age,
            "mask_path": mask_path,
            "mask_pixels": int(mask.sum()),
            "identity_exact": exact,
            "overview": overview_name,
            "overview_copy": os.path.join(OVERVIEW_DIRNAME, overview_name),
            "min_pixels": options.min_pixels,
            "skipped_below_min_pixels": skipped_small,
            "defects": [],
        }

        for index, entry in enumerate(entries, start=1):
            if entry["box"] is None:
                frame_record["defects"].append(
                    {
                        "name": entry["name"],
                        "pixels": entry["pixels"],
                        "crop": None,
                        "source": entry["source"],
                    }
                )
                continue

            window = padded_window(
                entry["box"], mask.shape, CROP_MIN_SIZE, CROP_PAD_RATIO
            )
            crop_name = (
                f"{frame_prefix}_{index:02d}_{entry['name']}.png"
            )
            mask_crop_name = (
                f"{frame_prefix}_{index:02d}_{entry['name']}_mask.png"
            )
            scale = None
            written = {}

            if options.overlay in ("both", "clean"):
                scale = write_zoom(
                    clean_canvas,
                    entry,
                    window,
                    os.path.join(out_dir, crop_name),
                    options.zoom_long_edge,
                    has_rgb,
                )
                written["crop"] = crop_name
                total_crops += 1

            if options.overlay in ("both", "mask") and has_rgb:
                scale = write_zoom(
                    mask_canvas,
                    entry,
                    window,
                    os.path.join(out_dir, mask_crop_name),
                    options.zoom_long_edge,
                    has_rgb,
                    tinted=True,
                )
                written["crop_mask"] = mask_crop_name
                total_crops += 1
            elif options.overlay == "mask":
                # No RGB: the clean crop already is the mask.
                scale = write_zoom(
                    mask_canvas,
                    entry,
                    window,
                    os.path.join(out_dir, crop_name),
                    options.zoom_long_edge,
                    has_rgb,
                )
                written["crop"] = crop_name
                total_crops += 1

            record = {
                "name": entry["name"],
                "pixels": entry["pixels"],
                "bbox_xyxy": list(entry["box"]),
                "zoom": round(scale, 2) if scale else None,
                "source": entry["source"],
            }
            record.update(written)
            frame_record["defects"].append(record)

        summary["frames"].append(frame_record)
        skipped_text = ""
        if skipped_small:
            skipped_text = (
                f", {len(skipped_small)} under {options.min_pixels}px "
                "left out ("
                + ", ".join(
                    f"{item['pixels']}px" for item in skipped_small
                )
                + ")"
            )
        print(
            f"[ZOOM] frame {frame_id}: {len(entries)} defect(s), "
            f"rgb={'yes' if has_rgb else 'NO'}, "
            f"identity={'exact' if exact else 'pixel-count match'}"
            f"{skipped_text}"
        )

    summary["total_crops"] = total_crops
    summary["frames_without_rgb"] = sorted(set(frames_without_rgb))

    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    write_report(summary, os.path.join(out_dir, "README.md"))

    print(f"\n[ZOOM] {total_crops} crop(s) -> {out_dir}")
    print(f"[ZOOM] summary: {summary_path}")

    if summary["frames_without_rgb"]:
        print(
            "[ZOOM] rgb_*.png가 없는 frame: "
            + ", ".join(str(v) for v in summary["frames_without_rgb"])
            + " - 마스크만 그렸습니다. 실제 이미지를 보려면 "
            "BATTERY_SAVE_FRAME_IMAGES=1로 다시 촬영하세요."
        )


def write_report(summary, path):
    dropped = sum(
        len(frame.get("skipped_below_min_pixels", []))
        for frame in summary["frames"]
    )
    threshold = next(
        (
            frame["min_pixels"]
            for frame in summary["frames"]
            if "min_pixels" in frame
        ),
        None,
    )
    lines = [
        "# Defect zoom report",
        "",
        f"- started at: `{summary['started_at']}`",
        f"- capture dir: `{summary['capture_dir']}`",
        f"- output dir: `{summary['out_dir']}`",
        f"- crops: {summary['total_crops']}",
    ]

    if threshold is not None:
        lines.append(
            f"- defects under {threshold} px are not drawn; "
            f"{dropped} observation(s) left out on that rule"
        )

    if summary["frames_without_rgb"]:
        lines += [
            "- **rgb_\\*.png missing** for frames "
            + ", ".join(str(v) for v in summary["frames_without_rgb"])
            + "; those crops show the defect mask only. Re-capture with "
            "`BATTERY_SAVE_FRAME_IMAGES=1` for real imagery.",
        ]

    lines += [
        "",
        "Each row's `crop` is the untouched capture; the `_mask.png` beside "
        "it has the GT defect pixels tinted.",
        "",
        "| frame | rgb | defect | pixels | bbox | zoom | identity |",
        "|---|---|---|---|---|---|---|",
    ]

    for frame in summary["frames"]:
        for defect in frame["defects"]:
            box = defect.get("bbox_xyxy")
            box_text = (
                f"{box[2] - box[0]}x{box[3] - box[1]}" if box else "-"
            )
            lines.append(
                f"| {frame['frame_id']} "
                f"| {'yes' if frame['has_rgb'] else 'no'} "
                f"| `{defect['name']}` | {defect['pixels']} "
                f"| {box_text} | x{defect.get('zoom', '-')} "
                f"| {defect['source']} |"
            )

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
