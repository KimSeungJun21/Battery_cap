"""Procedural texture generation for surface defect decals.

Battery_defect_config.py의 DefectSpec을 받아 decal 머티리얼이 쓸
albedo / roughness / normal / opacity 맵을 절차적으로 생성한다.

Isaac Sim에 의존하지 않는다 (numpy + PIL만 사용). 단독 실행하면
활성 시나리오의 텍스처를 굽고, 서로 다른 조명 각도에서 셰이딩한
프리뷰를 함께 저장하므로 시뮬레이터를 켜기 전에 "이 흠집이 각도에
따라 보였다 안 보였다 하는가"를 눈으로 검증할 수 있다.

    python3 Battery_defect_texture_utils.py

decal은 alpha cutout으로 합성한다. 즉 흠집 바깥은 opacity 0이라
아래에 있는 원래 물체 표면이 그대로 보이고, 흠집 픽셀만 이 텍스처
값으로 그려진다. 사각 패치가 통째로 보이면 검출기가 흠집이 아니라
"사각형"을 찾아버리므로 cutout이 필수다.
"""

from __future__ import annotations

import json
import math
import os
import re

import numpy as np
from PIL import Image, ImageDraw

from Battery_defect_config import (
    DEFECT_DEBUG_EMISSIVE_GAIN,
    DEFECT_DEBUG_VISIBLE,
    DEFECT_DEBUG_WIDTH_SCALE,
    DEFECT_DECAL_MARGIN_SCALE,
    DEFECT_TEXTURE_DIR,
    DEFECT_TEXTURE_RESOLUTION_PX,
    debug_color_for,
    debug_visible_spec,
    describe_active_defects,
    get_active_defect_specs,
    resolve_seed,
)


# ============================================================
# Base material 근사
# ============================================================
#
# decal은 별도 머티리얼이라 아래 표면의 색을 샘플링할 수 없다.
# 흠집 픽셀에 쓸 기준 색/거칠기를 여기서 근사한다.
#
# scratch/scuff는 albedo_scale이 0.95~1.0이라 이 근사가 거의
# 영향을 주지 않는다 (시각 효과는 normal + roughness에서 나온다).
# albedo_scale이 0.55인 stain은 어차피 눈에 띄라고 만든 대조군이다.
# 정밀도가 필요해지면 나중에 아래 표면 텍스처를 샘플링해서 이
# 상수를 대체하면 된다.

# Tuned per target, because the fallback has to resemble the surface the
# decal sits on. The mustard bottle is yellow; the batterycap is not - its
# defect-bearing meshes (mainboard2, the side walls, the caps) have no
# material bound at all and render in Isaac's neutral default, measured at
# sRGB (153,154,155)..(230,230,230) around the defects of a real capture.
# Painting the mustard yellow onto that turned every scratch into a bright
# yellow stripe: defect pixels came out at sRGB (211,203,122) against a
# (126,127,129) surround, which makes a detection experiment meaningless.
# Mostly the hue matters, not the level - scratch albedo_scale is 0.95-1.0,
# so the visual effect is meant to come from the normal and roughness maps.
_DEFECT_BASE_ALBEDO_PRESETS = {
    "mustard": (0.86, 0.72, 0.16),
    "batterycap": (0.50, 0.50, 0.50),
}
_DEFECT_TARGET_ALIASES = {
    "mustard": "mustard",
    "mustard_bottle": "mustard",
    "batterycap": "batterycap",
    "battery_cap": "batterycap",
}


def _resolve_defect_base_albedo_rgb():
    """BATTERY_DEFECT_BASE_ALBEDO_RGB, else the target's preset."""
    raw = os.environ.get(
        "BATTERY_DEFECT_BASE_ALBEDO_RGB",
        "",
    ).strip()

    if raw:
        parts = [part for part in re.split(r"[,\s]+", raw) if part]

        if len(parts) != 3:
            raise ValueError(
                "BATTERY_DEFECT_BASE_ALBEDO_RGB는 'r,g,b' 형식의 선형 "
                f"RGB 3개 값이어야 합니다: {raw!r}"
            )

        return tuple(float(part) for part in parts)

    # Battery_config를 import하면 pxr까지 끌어오므로 alias만 따로 둔다.
    target = os.environ.get(
        "BATTERY_TARGET_OBJECT",
        "batterycap",
    ).strip().lower()
    key = _DEFECT_TARGET_ALIASES.get(target, "batterycap")
    return _DEFECT_BASE_ALBEDO_PRESETS[key]


DEFECT_BASE_ALBEDO_RGB = _resolve_defect_base_albedo_rgb()
DEFECT_BASE_ROUGHNESS = float(
    os.environ.get(
        "BATTERY_DEFECT_BASE_ROUGHNESS",
        "0.38",
    )
)
DEFECT_BASE_METALLIC = float(
    os.environ.get(
        "BATTERY_DEFECT_BASE_METALLIC",
        "0.0",
    )
)

# 물리 깊이를 건드리지 않고 normal map 기울기만 증폭하는 렌더링 노브.
# 시뮬에서 흠집이 너무 안 보이면 이 값을 올려 먼저 확인한다.
DEFECT_NORMAL_STRENGTH = float(
    os.environ.get(
        "BATTERY_DEFECT_NORMAL_STRENGTH",
        "1.0",
    )
)

# opacity cutout 임계값. UsdPreviewSurface opacityThreshold와 맞춘다.
DEFECT_ALPHA_THRESHOLD = float(
    os.environ.get(
        "BATTERY_DEFECT_ALPHA_THRESHOLD",
        "0.1",
    )
)

# 패치 여백 하한(mm). 얇은 스크래치는 margin scale만으로는 패치가
# 1mm도 안 되게 잡히므로 절대 하한이 필요하다.
DEFECT_DECAL_MIN_PAD_MM = 1.0

# alpha cutout이라 패치 사각형 자체는 렌더되지 않는다. 여백을 더 줘도
# 얻는 것 없이 px/mm만 떨어지므로 상한을 둔다.
DEFECT_DECAL_MAX_PAD_MM = 2.5

# 홈 단면 지수. 1.0 = 포물선, <1.0 = 평평한 바닥 + 가파른 벽.
DEFECT_GROOVE_PROFILE_EXPONENT = float(
    os.environ.get(
        "BATTERY_DEFECT_GROOVE_PROFILE_EXPONENT",
        "0.5",
    )
)

# 홈 가장자리에 밀려 올라온 둔덕의 높이 (depth 대비 비율).
# 실제로 정반사 글린트를 만드는 건 홈 바닥이 아니라 이 둔덕이다.
DEFECT_GROOVE_LIP_RATIO = float(
    os.environ.get(
        "BATTERY_DEFECT_GROOVE_LIP_RATIO",
        "0.5",
    )
)

# 가장 얇은 특징이 텍스처에서 최소 몇 px를 차지해야 하는가.
# 센서 GSD(0.189mm/px) 대비 충분한 오버샘플링을 보장한다.
DEFECT_TARGET_PX_PER_FEATURE = 8.0
DEFECT_TEXTURE_MAX_RESOLUTION_PX = 1024

# 거리장 계산 시 한 번에 처리할 선분 수 (메모리 상한).
_SEGMENT_CHUNK = 64

# scuff의 length_mm/width_mm는 스트로크 굵기가 아니라 마모 '영역'의
# 크기를 뜻한다. 개별 미세 스크래치의 굵기는 여기서 정한다.
DEFECT_SCUFF_STROKE_WIDTH_MM = float(
    os.environ.get(
        "BATTERY_DEFECT_SCUFF_STROKE_WIDTH_MM",
        "0.25",
    )
)


def stroke_width_mm(spec):
    """거리장에서 홈 굵기로 쓸 값."""
    if spec.defect_type == "scuff":
        if DEFECT_DEBUG_VISIBLE:
            # 영역 크기는 그대로 두고 개별 미세 스크래치만 굵게.
            return (
                DEFECT_SCUFF_STROKE_WIDTH_MM
                * DEFECT_DEBUG_WIDTH_SCALE
            )

        return DEFECT_SCUFF_STROKE_WIDTH_MM

    return spec.width_mm


# ============================================================
# 기하 생성
# ============================================================


def _bezier_polyline(p0, p1, p2, sample_count):
    t = np.linspace(0.0, 1.0, sample_count, dtype=np.float32)
    t = t[:, None]
    one_minus_t = 1.0 - t

    return (
        one_minus_t * one_minus_t * np.asarray(p0, dtype=np.float32)
        + 2.0 * one_minus_t * t * np.asarray(p1, dtype=np.float32)
        + t * t * np.asarray(p2, dtype=np.float32)
    )


def _sample_count_for_length(length_mm):
    return int(np.clip(round(length_mm * 6.0), 8, 128))


def _build_strokes(spec, rng):
    """결함의 스트로크 폴리라인들을 mm 좌표로 만든다.

    주 방향은 항상 +x다. 표면 접평면 안에서의 회전(tangent_deg)은
    decal prim을 돌려서 처리하므로 텍스처는 관여하지 않는다.
    """
    strokes = []

    if spec.stroke_count <= 1:
        half_length = 0.5 * spec.length_mm
        bulge = spec.curvature * spec.length_mm * 0.25
        strokes.append(
            _bezier_polyline(
                (-half_length, 0.0),
                (0.0, bulge),
                (half_length, 0.0),
                _sample_count_for_length(spec.length_mm),
            )
        )

        return strokes

    if spec.defect_type == "scuff":
        # 마모면: 짧은 스트로크가 패치 전체에 흩뿌려진다.
        for _ in range(spec.stroke_count):
            stroke_length = spec.length_mm * rng.uniform(0.25, 0.75)
            # 스트로크 전체가 선언한 length x width 영역 안에 들어오도록
            # 중심 범위를 스트로크 길이만큼 좁힌다.
            max_center_x = max(
                0.5 * (spec.length_mm - stroke_length),
                0.0,
            )
            center_x = rng.uniform(-max_center_x, max_center_x)
            center_y = rng.uniform(
                -0.45 * spec.width_mm,
                0.45 * spec.width_mm,
            )
            angle = math.radians(rng.uniform(-40.0, 40.0))
            direction = np.array(
                [math.cos(angle), math.sin(angle)],
                dtype=np.float32,
            )
            normal = np.array(
                [-direction[1], direction[0]],
                dtype=np.float32,
            )
            center = np.array(
                [center_x, center_y],
                dtype=np.float32,
            )
            bulge = (
                spec.curvature
                * stroke_length
                * rng.uniform(-0.3, 0.3)
            )
            strokes.append(
                _bezier_polyline(
                    center - 0.5 * stroke_length * direction,
                    center + bulge * normal,
                    center + 0.5 * stroke_length * direction,
                    _sample_count_for_length(stroke_length),
                )
            )

        return strokes

    # 다발 스크래치: 거의 평행하게 이웃한 선들.
    spacing = max(spec.width_mm * 2.2, 0.35)

    for index in range(spec.stroke_count):
        offset = (
            index - 0.5 * (spec.stroke_count - 1)
        ) * spacing
        offset += rng.normal(0.0, spacing * 0.15)
        stroke_length = spec.length_mm * rng.uniform(0.8, 1.0)
        half_length = 0.5 * stroke_length
        tilt = math.radians(rng.uniform(-6.0, 6.0))
        bulge = (
            spec.curvature
            * stroke_length
            * 0.25
            * rng.uniform(0.6, 1.2)
        )
        start = np.array(
            [
                -half_length,
                offset - half_length * math.tan(tilt),
            ],
            dtype=np.float32,
        )
        end = np.array(
            [
                half_length,
                offset + half_length * math.tan(tilt),
            ],
            dtype=np.float32,
        )
        control = 0.5 * (start + end) + np.array(
            [0.0, bulge],
            dtype=np.float32,
        )
        strokes.append(
            _bezier_polyline(
                start,
                control,
                end,
                _sample_count_for_length(stroke_length),
            )
        )

    return strokes


def _strokes_bbox(strokes):
    points = np.concatenate(strokes, axis=0)

    return (
        float(points[:, 0].min()),
        float(points[:, 1].min()),
        float(points[:, 0].max()),
        float(points[:, 1].max()),
    )


# ============================================================
# 패치 크기 / 해상도
# ============================================================


def _resolve_patch(spec, strokes):
    """정사각 패치 한 변의 길이(mm)와 텍스처 해상도를 정한다.

    정사각으로 두면 px/mm가 두 축에서 동일해져 스트로크 폭이
    왜곡되지 않는다.
    """
    if strokes is None:
        half_extent = 0.5 * max(spec.length_mm, spec.width_mm)
        min_x, min_y = -half_extent, -half_extent
        max_x, max_y = half_extent, half_extent
    else:
        min_x, min_y, max_x, max_y = _strokes_bbox(strokes)

    # 스트로크 폭의 절반만큼은 양쪽으로 더 필요하다.
    half_width = 0.5 * stroke_width_mm(spec)
    min_x -= half_width
    min_y -= half_width
    max_x += half_width
    max_y += half_width

    content_extent = max(max_x - min_x, max_y - min_y)
    pad_mm = float(
        np.clip(
            0.5 * (DEFECT_DECAL_MARGIN_SCALE - 1.0) * content_extent,
            DEFECT_DECAL_MIN_PAD_MM,
            DEFECT_DECAL_MAX_PAD_MM,
        )
    )
    patch_mm = content_extent + 2.0 * pad_mm

    if spec.defect_type in ("stain", "scuff"):
        min_feature_mm = max(0.2 * spec.width_mm, 0.3)
    else:
        min_feature_mm = max(stroke_width_mm(spec), 0.1)

    required_px = (
        patch_mm
        / (min_feature_mm / DEFECT_TARGET_PX_PER_FEATURE)
    )
    resolution = DEFECT_TEXTURE_RESOLUTION_PX

    while (
        resolution < required_px
        and resolution < DEFECT_TEXTURE_MAX_RESOLUTION_PX
    ):
        resolution *= 2

    resolution = int(
        min(resolution, DEFECT_TEXTURE_MAX_RESOLUTION_PX)
    )

    center_x = 0.5 * (min_x + max_x)
    center_y = 0.5 * (min_y + max_y)

    return patch_mm, resolution, (center_x, center_y)


def compute_patch_geometry(spec):
    """맵을 만들지 않고 패치 크기/해상도만 먼저 구한다.

    decal을 표면에 정합시키려면 patch_mm이 먼저 필요하고, 아래 표면
    색을 샘플링하려면 정합이 먼저 끝나야 한다. 그래서 맵 생성보다
    앞서 이 값만 뽑아 쓴다.
    """
    if spec.defect_type in ("stain", "scuff"):
        strokes = None
    else:
        strokes = _build_strokes(
            spec,
            np.random.default_rng(resolve_seed(spec)),
        )

    return _resolve_patch(spec, strokes)


def _make_grid(patch_mm, resolution, center):
    half = 0.5 * patch_mm
    axis = np.linspace(
        -half,
        half,
        resolution,
        dtype=np.float32,
    )
    grid_x, grid_y = np.meshgrid(
        axis + center[0],
        axis + center[1],
        indexing="xy",
    )

    return grid_x, grid_y


# ============================================================
# 거리장
# ============================================================


def _min_distance_to_strokes(strokes, grid_x, grid_y):
    """모든 스트로크 선분까지의 최소 거리(mm)를 구한다."""
    segments = []

    for stroke in strokes:
        segments.append(
            np.stack(
                [stroke[:-1], stroke[1:]],
                axis=1,
            )
        )

    segments = np.concatenate(segments, axis=0).astype(np.float32)
    points = np.stack(
        [grid_x.ravel(), grid_y.ravel()],
        axis=1,
    ).astype(np.float32)
    best = np.full(points.shape[0], np.inf, dtype=np.float32)

    for start_index in range(0, len(segments), _SEGMENT_CHUNK):
        chunk = segments[start_index:start_index + _SEGMENT_CHUNK]
        seg_a = chunk[:, 0, :]
        seg_b = chunk[:, 1, :]
        seg_dir = seg_b - seg_a
        seg_len_sq = np.maximum(
            (seg_dir * seg_dir).sum(axis=1),
            1e-12,
        )
        delta = points[:, None, :] - seg_a[None, :, :]
        projection = (
            delta * seg_dir[None, :, :]
        ).sum(axis=2) / seg_len_sq[None, :]
        projection = np.clip(projection, 0.0, 1.0)
        closest = (
            seg_a[None, :, :]
            + projection[:, :, None] * seg_dir[None, :, :]
        )
        distance = np.linalg.norm(
            points[:, None, :] - closest,
            axis=2,
        )
        best = np.minimum(best, distance.min(axis=1))

    return best.reshape(grid_x.shape)


def _smoothstep(edge0, edge1, value):
    """edge0 > edge1인 하강 구간과 배열 edge를 모두 지원한다.

    stain/scuff의 blob 경계는 각도마다 다른 배열이라 edge가 스칼라가
    아니다. 반대로 홈 단면은 edge0 > edge1인 하강 구간이다. 둘 다
    같은 함수로 처리한다.
    """
    edge0 = np.asarray(edge0, dtype=np.float32)
    edge1 = np.asarray(edge1, dtype=np.float32)
    denominator = edge1 - edge0
    degenerate = np.abs(denominator) < 1e-12
    safe_denominator = np.where(degenerate, 1.0, denominator)
    t = np.clip(
        (value - edge0) / safe_denominator,
        0.0,
        1.0,
    )
    t = np.where(
        degenerate,
        (value >= edge1).astype(np.float32),
        t,
    )

    return t * t * (3.0 - 2.0 * t)


# ============================================================
# 맵 생성
# ============================================================


def _stain_fields(spec, grid_x, grid_y, rng):
    """얼룩: 높이 변화 없이 불규칙한 blob 커버리지만 만든다."""
    radius = 0.5 * max(spec.length_mm, spec.width_mm)
    aspect = max(spec.width_mm, 1e-6) / max(spec.length_mm, 1e-6)
    scaled_y = grid_y / max(aspect, 1e-6)
    angle = np.arctan2(scaled_y, grid_x)
    distance = np.sqrt(grid_x * grid_x + scaled_y * scaled_y)

    boundary = np.full_like(distance, radius)

    for harmonic in (2, 3, 5, 7):
        phase = rng.uniform(0.0, 2.0 * math.pi)
        amplitude = radius * rng.uniform(0.04, 0.13)
        boundary = boundary + amplitude * np.cos(
            harmonic * angle + phase
        )

    softness = max(0.08 * radius, 0.15)
    coverage = 1.0 - _smoothstep(
        boundary - softness,
        boundary + softness,
        distance,
    )
    height = np.zeros_like(coverage)

    return height, coverage, coverage


def _box_blur(array, kernel_x, kernel_y):
    """cumsum 기반 분리형 이동평균 (scipy 없이)."""
    result = array.astype(np.float32)

    for axis, kernel in ((1, kernel_x), (0, kernel_y)):
        kernel = int(max(1, kernel))

        if kernel <= 1:
            continue

        padding = kernel // 2
        padded = np.pad(
            result,
            [(padding, padding) if index == axis else (0, 0)
             for index in range(2)],
            mode="reflect",
        )
        cumulative = np.cumsum(padded, axis=axis)
        cumulative = np.concatenate(
            [
                np.zeros_like(
                    np.take(cumulative, [0], axis=axis)
                ),
                cumulative,
            ],
            axis=axis,
        )
        upper = np.take(
            cumulative,
            np.arange(kernel, kernel + result.shape[axis]),
            axis=axis,
        )
        lower = np.take(
            cumulative,
            np.arange(0, result.shape[axis]),
            axis=axis,
        )
        result = (upper - lower) / float(kernel)

    return result


def _scuff_fields(spec, grid_x, grid_y, rng, mm_per_px):
    """마모면: 개별 홈이 아니라 '거칠기만 다른 불규칙 영역'.

    scuff를 discrete groove 다발로 만들면 사실상 스크래치 14개가 되어
    모든 각도에서 글린트가 잡힌다. 그러면 이 시나리오의 존재 이유인
    '정반사 각도가 맞는 뷰에서만 드러나는 결함'이 사라진다.

    그래서 형상 변화는 거의 0으로 두고(미세 요철만), 불규칙한 경계의
    영역 안에서 roughness만 올린다. bright field에서는 주변이 밝고
    이 영역만 둔탁하게 빠져 대비가 크지만, 정반사에서 벗어나면 주변과
    구분되지 않는다.
    """
    _, coverage, _ = _stain_fields(spec, grid_x, grid_y, rng)

    # 연마 방향이 있는 이방성 미세 노이즈. 주 방향은 +x.
    noise = rng.random(grid_x.shape).astype(np.float32)
    streak_length_px = max(
        int(round(0.8 * spec.length_mm / spec.stroke_count / mm_per_px)),
        3,
    )
    noise = _box_blur(noise, streak_length_px, 2)
    noise = noise - noise.mean()
    noise_scale = np.std(noise)

    if noise_scale > 1e-9:
        noise = noise / noise_scale

    # 거칠기는 영역 전체에 걸리되 연마 자국만큼 세기가 들쭉날쭉하다.
    roughness_coverage = np.clip(
        coverage * (0.75 + 0.25 * noise),
        0.0,
        1.0,
    )

    # 형상은 미세 요철만. 법선 기울기가 몇 도를 넘지 않아야
    # "거칠기만 다른 결함"이라는 전제가 유지된다.
    depth_mm = spec.depth_um * 1e-3
    height = depth_mm * noise * coverage

    return height, coverage, roughness_coverage


def _stroke_fields(spec, grid_x, grid_y, rng):
    """스크래치/마모/눌림: 거리장에서 높이와 커버리지를 만든다."""
    strokes = _build_strokes(spec, rng)
    distance = _min_distance_to_strokes(strokes, grid_x, grid_y)
    half_width = max(0.5 * stroke_width_mm(spec), 1e-4)
    depth_mm = spec.depth_um * 1e-3

    # 홈 단면. 순수 포물선(지수 1.0)은 바닥이 넓고 벽이 완만해서
    # 정반사 로브에 걸릴 만한 기울기가 안 나온다. 지수를 낮춰
    # 바닥을 평평하게, 벽을 가파르게 만든다.
    profile = np.clip(
        1.0 - (distance / half_width) ** 2,
        0.0,
        1.0,
    ) ** DEFECT_GROOVE_PROFILE_EXPONENT
    height = -depth_mm * profile

    # 실제 스크래치는 소성 변형으로 가장자리에 둔덕이 생긴다.
    # 이 둔덕이 정반사 글린트를 만들어 시점 의존성을 강화한다.
    lip = _smoothstep(
        half_width * 1.55,
        half_width * 1.05,
        distance,
    ) * _smoothstep(
        half_width * 0.95,
        half_width * 1.15,
        distance,
    )
    height = height + depth_mm * DEFECT_GROOVE_LIP_RATIO * lip

    # 홈 바닥의 미세 거칠기. 매끈한 홈은 비현실적이다.
    if depth_mm > 0.0:
        noise = rng.normal(
            0.0,
            1.0,
            size=height.shape,
        ).astype(np.float32)
        noise = (
            noise
            + np.roll(noise, 1, axis=0)
            + np.roll(noise, 1, axis=1)
        ) / 3.0
        height = height + depth_mm * 0.18 * noise * profile

    coverage = _smoothstep(
        half_width * 1.25,
        half_width * 0.55,
        distance,
    )
    # 거칠기는 홈보다 약간 넓은 헤일로까지 번진다 (연마 자국).
    roughness_coverage = _smoothstep(
        half_width * 2.0,
        half_width * 0.7,
        distance,
    )

    return height, coverage, roughness_coverage


def _upsample_grid(grid, resolution):
    """(n, n, C) 격자를 (resolution, resolution, C)로 이중선형 확대."""
    grid = np.asarray(grid, dtype=np.float32)
    source_n = grid.shape[0]

    if source_n == resolution:
        return grid

    coords = np.linspace(0.0, source_n - 1.0, resolution)
    lower = np.floor(coords).astype(np.int32)
    upper = np.minimum(lower + 1, source_n - 1)
    weight = (coords - lower).astype(np.float32)

    rows = (
        grid[lower] * (1.0 - weight)[:, None, None]
        + grid[upper] * weight[:, None, None]
    )

    return (
        rows[:, lower] * (1.0 - weight)[None, :, None]
        + rows[:, upper] * weight[None, :, None]
    )


def _normal_from_height(height, mm_per_px):
    gradient_y, gradient_x = np.gradient(
        height.astype(np.float32),
        mm_per_px,
    )
    normal = np.stack(
        [
            -gradient_x * DEFECT_NORMAL_STRENGTH,
            -gradient_y * DEFECT_NORMAL_STRENGTH,
            np.ones_like(height, dtype=np.float32),
        ],
        axis=2,
    )
    norm = np.linalg.norm(normal, axis=2, keepdims=True)

    return normal / np.maximum(norm, 1e-9)


def generate_defect_maps(
    spec,
    base_albedo_grid=None,
    base_roughness=None,
    debug_color=None,
):
    """DefectSpec 하나에 대한 맵들과 패치 메타데이터를 만든다.

    base_albedo_grid가 주어지면 (n, n, 3) 선형 RGB 격자를 텍스처
    해상도로 확대해 base albedo로 쓴다. decal은 별도 머티리얼이라
    아래 표면 색을 알 수 없는데, 상수로 근사하면 인쇄 라벨 위에서
    색이 완전히 어긋난다. 특히 albedo_scale=1.0인 scuff는 '색 변화
    없음'이 설계인데도 노란 덩어리로 칠해져 시나리오가 무효가 된다.
    """
    rng = np.random.default_rng(resolve_seed(spec))

    if spec.defect_type in ("stain", "scuff"):
        strokes = None
    else:
        strokes = _build_strokes(
            spec,
            np.random.default_rng(resolve_seed(spec)),
        )

    patch_mm, resolution, center = _resolve_patch(spec, strokes)
    grid_x, grid_y = _make_grid(patch_mm, resolution, center)
    mm_per_px = patch_mm / resolution

    if spec.defect_type == "stain":
        height, coverage, roughness_coverage = _stain_fields(
            spec,
            grid_x,
            grid_y,
            rng,
        )
    elif spec.defect_type == "scuff":
        height, coverage, roughness_coverage = _scuff_fields(
            spec,
            grid_x,
            grid_y,
            rng,
            mm_per_px,
        )
    else:
        height, coverage, roughness_coverage = _stroke_fields(
            spec,
            grid_x,
            grid_y,
            rng,
        )

    if base_albedo_grid is None:
        base_albedo = np.broadcast_to(
            np.asarray(DEFECT_BASE_ALBEDO_RGB, dtype=np.float32),
            (resolution, resolution, 3),
        )
        base_albedo_source = "constant"
    else:
        base_albedo = _upsample_grid(base_albedo_grid, resolution)
        base_albedo_source = "sampled"

    albedo = base_albedo * (
        1.0
        + (spec.albedo_scale - 1.0) * coverage[:, :, None]
    )
    albedo = np.clip(albedo, 0.0, 1.0)
    emissive = None

    if debug_color is not None:
        # 디버그 가시화: 흠집 영역만 형광색으로 덮고 자체발광시킨다.
        # 패치 바깥은 opacity cutout으로 잘려나가므로 영향이 없다.
        neon = np.asarray(debug_color, dtype=np.float32)
        blend = coverage[:, :, None]
        albedo = (
            albedo * (1.0 - blend) + neon[None, None, :] * blend
        )
        emissive = (
            neon[None, None, :]
            * blend
            * DEFECT_DEBUG_EMISSIVE_GAIN
        )
        base_albedo_source = f"{base_albedo_source}+debug_neon"

    if base_roughness is None:
        base_roughness = DEFECT_BASE_ROUGHNESS

    roughness = np.clip(
        float(base_roughness)
        + spec.roughness_delta * roughness_coverage,
        0.02,
        1.0,
    )
    normal = _normal_from_height(height, mm_per_px)
    opacity = np.clip(coverage, 0.0, 1.0)

    visible_px = int(
        np.count_nonzero(opacity >= DEFECT_ALPHA_THRESHOLD)
    )
    # 법선이 기준면에서 얼마나 기울었는가. 정반사 로브에 걸리려면
    # 이 값이 충분히 커야 하므로 흠집 가시성의 1차 진단 지표다.
    tilt_deg = np.degrees(
        np.arccos(np.clip(normal[:, :, 2], -1.0, 1.0))
    )
    core = opacity >= DEFECT_ALPHA_THRESHOLD
    max_tilt_deg = float(tilt_deg.max()) if core.any() else 0.0
    p95_tilt_deg = (
        float(np.percentile(tilt_deg[core], 95.0))
        if core.any()
        else 0.0
    )

    return {
        "spec": spec,
        "patch_mm": float(patch_mm),
        "resolution": int(resolution),
        "mm_per_px": float(mm_per_px),
        "patch_center_mm": (
            float(center[0]),
            float(center[1]),
        ),
        "base_albedo_source": base_albedo_source,
        "base_roughness": float(base_roughness),
        "height": height,
        "albedo": albedo,
        "roughness": roughness,
        "normal": normal,
        "opacity": opacity,
        "footprint_area_mm2": float(
            visible_px * mm_per_px * mm_per_px
        ),
        "max_normal_tilt_deg": max_tilt_deg,
        "p95_normal_tilt_deg": p95_tilt_deg,
        "base_albedo_flat": base_albedo,
        "emissive": emissive,
        "debug_visible": debug_color is not None,
    }


# ============================================================
# 저장
# ============================================================


def _to_uint8(array):
    """[0,1] 배열을 8bit 이미지 배열로 바꾼다.

    USD의 UV 규약은 v=0이 이미지 맨 아래 행이다. 우리 배열의 row 0은
    y=-half(즉 -bitangent 쪽)이므로 그대로 저장하면 v축이 bitangent와
    반대로 뒤집힌다. 여기서 상하 반전해 두면 v축이 bitangent와 같은
    방향이 되고, normal map의 G 채널(= +v 방향 성분) 부호도 그대로
    맞는다.
    """
    flipped = np.flipud(array)

    return np.clip(flipped * 255.0 + 0.5, 0, 255).astype(np.uint8)


def _linear_to_srgb(linear):
    linear = np.clip(linear, 0.0, 1.0)

    return np.where(
        linear <= 0.0031308,
        linear * 12.92,
        1.055 * np.power(linear, 1.0 / 2.4) - 0.055,
    )


def write_defect_textures(maps, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    spec = maps["spec"]
    paths = {}

    Image.fromarray(
        _to_uint8(_linear_to_srgb(maps["albedo"])),
        mode="RGB",
    ).save(os.path.join(out_dir, f"{spec.name}_albedo.png"))
    paths["albedo"] = os.path.join(
        out_dir,
        f"{spec.name}_albedo.png",
    )

    Image.fromarray(
        _to_uint8(maps["roughness"]),
        mode="L",
    ).save(os.path.join(out_dir, f"{spec.name}_roughness.png"))
    paths["roughness"] = os.path.join(
        out_dir,
        f"{spec.name}_roughness.png",
    )

    Image.fromarray(
        _to_uint8(maps["normal"] * 0.5 + 0.5),
        mode="RGB",
    ).save(os.path.join(out_dir, f"{spec.name}_normal.png"))
    paths["normal"] = os.path.join(
        out_dir,
        f"{spec.name}_normal.png",
    )

    Image.fromarray(
        _to_uint8(maps["opacity"]),
        mode="L",
    ).save(os.path.join(out_dir, f"{spec.name}_opacity.png"))
    paths["opacity"] = os.path.join(
        out_dir,
        f"{spec.name}_opacity.png",
    )

    if maps.get("emissive") is not None:
        emissive_path = os.path.join(
            out_dir,
            f"{spec.name}_emissive.png",
        )
        Image.fromarray(
            _to_uint8(_linear_to_srgb(maps["emissive"])),
            mode="RGB",
        ).save(emissive_path)
        paths["emissive"] = emissive_path

    metadata = {
        "name": spec.name,
        "defect_type": spec.defect_type,
        "seed": resolve_seed(spec),
        "patch_mm": maps["patch_mm"],
        "resolution_px": maps["resolution"],
        "mm_per_px": maps["mm_per_px"],
        "patch_center_mm": list(maps["patch_center_mm"]),
        "footprint_area_mm2": maps["footprint_area_mm2"],
        "max_normal_tilt_deg": maps["max_normal_tilt_deg"],
        "p95_normal_tilt_deg": maps["p95_normal_tilt_deg"],
        "alpha_threshold": DEFECT_ALPHA_THRESHOLD,
        "base_albedo_source": maps["base_albedo_source"],
        "base_roughness": maps["base_roughness"],
        "debug_visible": maps["debug_visible"],
        "base_metallic": DEFECT_BASE_METALLIC,
        "normal_strength": DEFECT_NORMAL_STRENGTH,
        "length_mm": spec.length_mm,
        "width_mm": spec.width_mm,
        "depth_um": spec.depth_um,
        "textures": paths,
    }
    metadata_path = os.path.join(
        out_dir,
        f"{spec.name}_meta.json",
    )

    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)

    metadata["metadata_path"] = metadata_path

    return metadata


# ============================================================
# 프리뷰 (조명 각도 의존성 검증용)
# ============================================================


# 조명 방위각만 돌리면 세 장 모두 확산광 영역에 머물러서 아무 차이도
# 안 난다. 검사 기하(bright field / dark field / off-specular) 자체를
# 바꿔야 "각도에 따라 보였다 안 보였다"가 드러난다.
#
#   bright field : 광원이 시선의 정반사 방향 근처. 주변이 밝고 흠집이
#                  어둡게 빠진다. 거칠기 변화(scuff)를 가장 잘 드러낸다.
#   dark field   : 광원이 거의 스치듯 들어온다. 주변은 어둡고 홈의
#                  가파른 벽/둔덕만 빛을 튕겨 밝게 보인다.
#   off-specular : 정반사에서 벗어난 일반 조명. 흠집이 거의 안 보인다.
#                  NBV 없이 한 장만 찍었을 때의 최악 케이스.
# 유전체(플라스틱) 기본 반사율.
PREVIEW_F0 = 0.04
PREVIEW_LIGHT_INTENSITY = 3.2
PREVIEW_AMBIENT = 0.10

PREVIEW_GEOMETRIES = (
    ("bright field", 80.0, 0.0),
    ("dark field", 18.0, 90.0),
    ("off-specular", 48.0, 200.0),
)


def _shade(normal, albedo, roughness, light_direction):
    """GGX microfacet BRDF로 셰이딩한다.

    에너지 정규화가 없는 Blinn-Phong을 쓰면 거친 영역이 모든 각도에서
    더 밝게 나와, roughness-only 결함의 시점 의존성을 완전히 잘못
    읽게 된다. Isaac의 RTX 렌더러도 GGX를 쓰므로 프리뷰도 맞춘다.
    """
    view = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    half_vector = light_direction + view
    half_vector = half_vector / np.linalg.norm(half_vector)

    n_dot_l = np.clip(
        (normal * light_direction[None, None, :]).sum(axis=2),
        1e-4,
        1.0,
    )
    n_dot_v = np.clip(
        (normal * view[None, None, :]).sum(axis=2),
        1e-4,
        1.0,
    )
    n_dot_h = np.clip(
        (normal * half_vector[None, None, :]).sum(axis=2),
        0.0,
        1.0,
    )
    v_dot_h = np.clip(
        float(np.dot(view, half_vector)),
        0.0,
        1.0,
    )

    alpha = np.clip(roughness, 0.02, 1.0) ** 2
    alpha_sq = alpha * alpha

    denominator = n_dot_h * n_dot_h * (alpha_sq - 1.0) + 1.0
    distribution = alpha_sq / (
        math.pi * np.maximum(denominator * denominator, 1e-9)
    )

    k = alpha * 0.5
    geometry = (
        n_dot_l / (n_dot_l * (1.0 - k) + k)
    ) * (
        n_dot_v / (n_dot_v * (1.0 - k) + k)
    )

    fresnel = PREVIEW_F0 + (1.0 - PREVIEW_F0) * (
        (1.0 - v_dot_h) ** 5
    )

    specular = (
        distribution * geometry * fresnel
    ) / (4.0 * n_dot_l * n_dot_v)
    diffuse = albedo * (1.0 - fresnel) / math.pi

    color = (
        diffuse + specular[:, :, None]
    ) * (PREVIEW_LIGHT_INTENSITY * n_dot_l)[:, :, None]
    color = color + albedo * PREVIEW_AMBIENT

    # Reinhard. 없으면 bright field가 포화되어 대비를 읽을 수 없다.
    color = color / (1.0 + color)

    return _linear_to_srgb(color)


def write_defect_preview(maps, out_dir):
    """같은 흠집을 조명 방위각만 바꿔 셰이딩한 비교 이미지.

    주변 정상 표면(평평한 법선 + base roughness)과 합성해서
    "decal이 아니라 흠집만 보이는가"도 함께 확인한다.
    """
    spec = maps["spec"]
    opacity = maps["opacity"][:, :, None]
    flat_normal = np.zeros_like(maps["normal"])
    flat_normal[:, :, 2] = 1.0
    base_albedo = maps.get("base_albedo_flat")

    if base_albedo is None:
        base_albedo = np.asarray(
            DEFECT_BASE_ALBEDO_RGB,
            dtype=np.float32,
        )[None, None, :]

    cutout = (
        maps["opacity"] >= DEFECT_ALPHA_THRESHOLD
    )[:, :, None].astype(np.float32)
    composite_normal = (
        flat_normal * (1.0 - cutout) + maps["normal"] * cutout
    )
    composite_albedo = (
        base_albedo * (1.0 - cutout)
        + maps["albedo"] * cutout
    )
    composite_roughness = (
        maps["base_roughness"] * (1.0 - cutout[:, :, 0])
        + maps["roughness"] * cutout[:, :, 0]
    )

    tiles = []
    labels = []

    for label, elevation_deg, azimuth_deg in PREVIEW_GEOMETRIES:
        elevation = math.radians(elevation_deg)
        azimuth = math.radians(azimuth_deg)
        labels.append(label)
        light_direction = np.array(
            [
                math.cos(elevation) * math.cos(azimuth),
                math.cos(elevation) * math.sin(azimuth),
                math.sin(elevation),
            ],
            dtype=np.float32,
        )
        tiles.append(
            _to_uint8(
                _shade(
                    composite_normal,
                    composite_albedo,
                    composite_roughness,
                    light_direction,
                )
            )
        )

    gap = 8
    height, width = tiles[0].shape[:2]
    canvas = np.full(
        (
            height,
            width * len(tiles) + gap * (len(tiles) - 1),
            3,
        ),
        24,
        dtype=np.uint8,
    )

    for index, tile in enumerate(tiles):
        offset = index * (width + gap)
        # 프리뷰는 화면으로 보는 용도이므로 _to_uint8의 상하 반전을
        # 되돌려 원본 방향으로 둔다.
        canvas[:, offset:offset + width] = np.flipud(tile)

    image = Image.fromarray(canvas, mode="RGB")
    draw = ImageDraw.Draw(image)

    for index, label in enumerate(labels):
        offset = index * (width + gap)
        draw.rectangle(
            [offset, 0, offset + width - 1, 16],
            fill=(16, 16, 16),
        )
        draw.text(
            (offset + 5, 4),
            f"{spec.name} | {label}",
            fill=(235, 235, 235),
        )

    preview_path = os.path.join(
        out_dir,
        f"{spec.name}_preview.png",
    )
    image.save(preview_path)

    return preview_path


# ============================================================
# 배치 실행
# ============================================================


def generate_all_defect_textures(
    specs=None,
    out_dir=None,
    write_preview=True,
    base_albedo_grids=None,
    log_prefix="[DEFECT-TEX]",
):
    """활성 결함 전부의 텍스처를 굽고 메타데이터 목록을 돌려준다."""
    if specs is None:
        specs = get_active_defect_specs()

    if out_dir is None:
        out_dir = DEFECT_TEXTURE_DIR

    os.makedirs(out_dir, exist_ok=True)
    results = []

    for spec in specs:
        maps = generate_defect_maps(
            debug_visible_spec(spec),
            base_albedo_grid=(
                base_albedo_grids.get(spec.name)
                if base_albedo_grids
                else None
            ),
            debug_color=debug_color_for(spec),
        )
        metadata = write_defect_textures(maps, out_dir)

        if write_preview:
            metadata["preview_path"] = write_defect_preview(
                maps,
                out_dir,
            )

        results.append(metadata)
        print(
            f"{log_prefix} {spec.name:<22} "
            f"patch={metadata['patch_mm']:5.2f}mm "
            f"res={metadata['resolution_px']:<4} "
            f"{1.0 / metadata['mm_per_px']:5.1f}px/mm "
            f"footprint={metadata['footprint_area_mm2']:6.2f}mm2 "
            f"tilt p95={metadata['p95_normal_tilt_deg']:5.1f}deg "
            f"albedo={metadata['base_albedo_source']}"
        )

    return results


if __name__ == "__main__":
    print(describe_active_defects())
    print()
    generate_all_defect_textures()
    print()
    print(f"[DEFECT-TEX] 출력 위치: {DEFECT_TEXTURE_DIR}")
