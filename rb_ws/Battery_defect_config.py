"""Surface defect catalog for the NBV inspection test scenario.

이 모듈은 "어떤 흠집이 물체 표면 어디에 존재하는가"만 선언한다.
텍스처 생성, raycast를 통한 표면 점/법선 해석, decal prim 스폰,
semantic label 부여는 전부 Battery_defect_utils.py의 몫이다.

의존성은 numpy와 표준 라이브러리뿐이다. Isaac Sim 밖에서 도는
검출 성능 평가 스크립트도 이 카탈로그를 그대로 import해서 GT로
쓸 수 있어야 하기 때문에 pxr을 끌어오지 않는다.

Object frame 규약 (YCB 006_mustard_bottle, Axis_Aligned_Physics):
    +X = 폭   (약 0.096 m)
    +Y = 장축 (약 0.191 m)  <- 병이 서 있는 방향
    +Z = 두께 (약 0.060 m)
    단면은 원기둥이 아니라 납작한 둥근 사각형이다.

    object origin은 bbox 중심이 아니다. 실측 occupancy map 기준
    Y가 [-0.05, +0.10] 범위에 걸쳐 있었다. 따라서 이 파일은 절대
    좌표를 하드코딩하지 않고 정규화 좌표(theta, axial_ratio)만
    선언하며, 실제 좌표는 런타임에 bbox + raycast로 해석한다.
"""

from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass, replace


# ============================================================
# Object frame 규약
# ============================================================

# 정규화 배치의 기준이 되는 장축 인덱스 (0=X, 1=Y, 2=Z).
DEFECT_LONG_AXIS_INDEX = 1

# theta_deg = 0일 때 바라보는 기준 축 인덱스 (+Z면 = 병의 앞면).
DEFECT_THETA_ZERO_AXIS_INDEX = 2

# theta_deg = 90일 때 바라보는 축 인덱스 (+X면 = 병의 옆면).
DEFECT_THETA_NINETY_AXIS_INDEX = 0

# 참고용 공칭 치수(m). 실제 배치는 런타임 bbox를 쓰므로 이 값은
# 카탈로그를 읽는 사람을 위한 문서이자 sanity check 기준일 뿐이다.
DEFECT_NOMINAL_EXTENT_M = (
    0.096,
    0.191,
    0.060,
)


# ============================================================
# 센서 해상도 제약
# ============================================================

# intrinsic.npy (fx=2120.0, fy=2430.6) + 공칭 작업거리 0.400 m 기준.
# 가로 0.189 mm/px, 세로 0.165 mm/px 중 보수적인 가로값을 쓴다.
DEFECT_REFERENCE_DISTANCE_M = 0.400
DEFECT_REFERENCE_FX_PX = 2120.0
DEFECT_REFERENCE_GSD_MM_PER_PX = (
    1000.0 * DEFECT_REFERENCE_DISTANCE_M / DEFECT_REFERENCE_FX_PX
)

# 이 폭 미만이면 subpixel이라 "픽셀 내 정반사 응답 변화"로만 잡힌다.
# 카탈로그 검증에서 경고를 띄우기 위한 기준선 (2 px 상당).
DEFECT_MIN_RESOLVABLE_WIDTH_MM = 2.0 * DEFECT_REFERENCE_GSD_MM_PER_PX


# ============================================================
# 파지 keep-out
# ============================================================

# 흡착컵이 물체 상단에 붙으므로 그 근처에 결함을 두면 decal이
# 가려지거나 흡착 접촉면을 방해한다. axial_ratio 상한을 둔다.
DEFECT_SUCTION_KEEPOUT_AXIAL_RATIO = 0.88


# ============================================================
# Decal / 텍스처 설정
# ============================================================

DEFECT_CLASS_LABEL = "defect"
DEFECT_PRIM_GROUP_NAME = "Defects"

DEFECT_TEXTURE_DIR = os.environ.get(
    "BATTERY_DEFECT_TEXTURE_DIR",
    "/home/seungjun/isaac_ws/zivid_capture/defect_textures",
)
DEFECT_GT_EXPORT_DIR = os.environ.get(
    "BATTERY_DEFECT_GT_DIR",
    "/home/seungjun/isaac_ws/zivid_capture/defect_gt",
)

# 8 mm 패치 기준 32 px/mm. 센서 GSD(5.3 px/mm) 대비 ~6x 오버샘플링.
DEFECT_TEXTURE_RESOLUTION_PX = int(
    os.environ.get(
        "BATTERY_DEFECT_TEXTURE_RESOLUTION_PX",
        "256",
    )
)

# decal mesh를 표면 법선 방향으로 띄우는 거리(m). 너무 작으면
# z-fighting, 너무 크면 grazing view에서 패치가 떠 보인다.
DEFECT_DECAL_OFFSET_M = float(
    os.environ.get(
        "BATTERY_DEFECT_DECAL_OFFSET_M",
        "0.0002",
    )
)

# decal patch를 결함 bounding box보다 얼마나 키울지 (배율).
# 스크래치 주변의 정상 표면이 같이 들어가야 검출기가 대비를 본다.
DEFECT_DECAL_MARGIN_SCALE = float(
    os.environ.get(
        "BATTERY_DEFECT_DECAL_MARGIN_SCALE",
        "1.6",
    )
)


# ============================================================
# 디버그 가시화
# ============================================================
#
# 실제 흠집은 폭 0.5~0.7mm라 191mm 병에서는 머리카락 한 올이고,
# scratch/scuff는 설계상 조명 각도에 따라 보였다 안 보였다 한다.
# 사람이 "결함이 제대로 붙었는지" 눈으로 확인하려면 이게 불편하다.
#
# 디버그 모드는 흠집을 두껍게 키우고 형광색 + 자체발광으로 칠해서
# 조명과 무관하게 눈에 띄게 만든다. 검출 난이도를 완전히 파괴하므로
# 기본값은 꺼져 있다. 실험을 돌릴 때는 절대 켜지 말 것.
#
#   BATTERY_DEFECT_DEBUG_VISIBLE=1 ./run_battery_rb_humble.sh

DEFECT_DEBUG_VISIBLE = (
    os.environ.get(
        "BATTERY_DEFECT_DEBUG_VISIBLE",
        "0",
    )
    .strip()
    .lower()
    not in (
        "0",
        "false",
        "no",
        "off",
        "",
    )
)

# 디버그 모드에서 흠집 폭/면적을 몇 배로 키울지.
DEFECT_DEBUG_WIDTH_SCALE = float(
    os.environ.get(
        "BATTERY_DEFECT_DEBUG_WIDTH_SCALE",
        "12.0",
    )
)
DEFECT_DEBUG_LENGTH_SCALE = float(
    os.environ.get(
        "BATTERY_DEFECT_DEBUG_LENGTH_SCALE",
        "1.8",
    )
)

# 자체발광 세기. 0이면 형광색으로 칠하되 빛나지는 않는다.
DEFECT_DEBUG_EMISSIVE_GAIN = float(
    os.environ.get(
        "BATTERY_DEFECT_DEBUG_EMISSIVE_GAIN",
        "2.5",
    )
)

# 타입별 형광색 (선형 RGB). 종류를 색으로 구분할 수 있게 한다.
# 머스타드 병은 노랑 본체 + 빨강(French's) + 파랑(영양성분표) 라벨이라
# 마젠타는 빨간 라벨 위에서, 시안은 파란 라벨 위에서 대비가 죽는다.
# 어느 배경에도 겹치지 않는 색으로 배정한다.
# 채널을 '순수하게' 둬야 한다. 자체발광 배율(2.5배)이 곱해지면
# 0이 아닌 채널은 전부 포화로 밀려 올라가 색이 흰색으로 바랜다.
# 예: (0.1, 1.0, 0.15) x2.5 -> R,B가 각각 0.25/0.375로 올라가
# 실제 렌더 색이 (212, 244, 204), 즉 거의 백색이 된다.
# 0인 채널은 아무리 곱해도 0이므로 채도가 유지된다.
DEFECT_DEBUG_COLORS = {
    "scratch": (0.0, 1.0, 0.0),    # 순수 초록
    "scuff": (1.0, 0.0, 1.0),      # 순수 마젠타
    "stain": (0.0, 1.0, 1.0),      # 순수 시안
    "dent": (1.0, 0.0, 0.0),       # 순수 빨강
}


# scuff/stain의 length_mm/width_mm는 스트로크 굵기가 아니라 결함
# '영역'의 크기다. 여기에 스트로크용 배율을 걸면 8mm 마모면이 48mm가
# 되어 병 두께를 넘는다. 영역형은 따로 완만하게 키운다.
DEFECT_AREA_TYPES = (
    "scuff",
    "stain",
)
DEFECT_DEBUG_AREA_SCALE = float(
    os.environ.get(
        "BATTERY_DEFECT_DEBUG_AREA_SCALE",
        "1.8",
    )
)


def debug_visible_spec(spec):
    """디버그 모드용으로 크기를 키운 DefectSpec 사본."""
    if not DEFECT_DEBUG_VISIBLE:
        return spec

    if spec.defect_type in DEFECT_AREA_TYPES:
        # 영역형은 이미 충분히 크다. 가시성 문제는 크기가 아니라
        # '거칠기만 바뀐다'는 점이므로 형광색으로 해결된다.
        return replace(
            spec,
            length_mm=spec.length_mm * DEFECT_DEBUG_AREA_SCALE,
            width_mm=spec.width_mm * DEFECT_DEBUG_AREA_SCALE,
        )

    # 다발 스크래치는 스트로크 간격이 폭에 비례해 벌어지므로, 폭을
    # 키우면 패치가 병 두께를 넘어 표면 정합이 깨진다. 디버그에서는
    # 어차피 '어디에 결함이 있는가'만 보면 되니 한 줄로 합친다.
    return replace(
        spec,
        length_mm=spec.length_mm * DEFECT_DEBUG_LENGTH_SCALE,
        width_mm=spec.width_mm * DEFECT_DEBUG_WIDTH_SCALE,
        stroke_count=1,
        curvature=spec.curvature * 0.5,
    )


def debug_color_for(spec):
    """디버그 모드에서 쓸 형광색. 꺼져 있으면 None."""
    if not DEFECT_DEBUG_VISIBLE:
        return None

    return DEFECT_DEBUG_COLORS.get(
        spec.defect_type,
        (1.0, 0.0, 0.85),
    )


# ============================================================
# 시나리오 선택
# ============================================================

ENABLE_DEFECTS = (
    os.environ.get(
        "BATTERY_ENABLE_DEFECTS",
        "1",
    )
    .strip()
    .lower()
    not in (
        "0",
        "false",
        "no",
        "off",
    )
)
DEFECT_SCENARIO_NAME = os.environ.get(
    "BATTERY_DEFECT_SCENARIO",
    "nbv",
).strip()
DEFECT_GLOBAL_SEED = int(
    os.environ.get(
        "BATTERY_DEFECT_SEED",
        "20260917",
    )
)


# ============================================================
# 선언 스키마
# ============================================================


@dataclass(frozen=True)
class SurfacePlacement:
    """Object frame 정규화 배치.

    실제 표면 점과 법선은 Battery_defect_utils.py가 런타임에
    bbox 밖에서 장축을 향해 ray를 쏘아 메시와 교차시켜 얻는다.
    theta는 "대충 어느 면인가"를 고르는 노브일 뿐이고, 정확한
    위치/법선은 메시가 결정한다.
    """

    # 장축(+Y) 기준 방위각. 0 = +Z면(앞), 90 = +X면(옆), 180 = 뒷면.
    theta_deg: float

    # 장축 방향 위치. 0 = bbox 최하단, 1 = bbox 최상단.
    axial_ratio: float

    # 표면 접평면 안에서의 결함 방향. 0 = 장축 평행.
    tangent_deg: float = 0.0

    # 시드 기반 미세 위치 랜덤화 반경(mm). 0이면 결정적.
    jitter_mm: float = 0.0

    # 특정 부품 표면만 허용해야 할 때 대상 Mesh prim 경로의 suffix.
    # 빈 문자열이면 대상 물체 아래 모든 Mesh를 합친 기존 방식이다.
    target_mesh_path_suffix: str = ""

    # 특정 부품의 장축/방위축이 대상 물체 기본축과 다를 때만 쓴다.
    # None이면 setup_defects()에 전달된 대상 물체 기본축을 유지한다.
    long_axis_index_override: int | None = None
    theta_zero_axis_index_override: int | None = None
    theta_ninety_axis_index_override: int | None = None


@dataclass(frozen=True)
class DefectSpec:
    """단일 결함 선언.

    형상 파라미터는 절차적 텍스처 생성기가 읽고, 재질 변조
    파라미터는 decal 머티리얼이 읽는다. 평가용 필드는 검출
    성능 채점 스크립트가 읽는다.
    """

    name: str
    defect_type: str
    placement: SurfacePlacement

    # --- 형상 ---
    length_mm: float = 10.0
    width_mm: float = 0.6
    # normal map 높이 진폭(um). dent 타입에서는 실제 변위량으로 쓴다.
    depth_um: float = 15.0
    # 다발 스크래치 개수. 1이면 단일 선.
    stroke_count: int = 1
    # 0 = 직선, 1 = 강한 곡선.
    curvature: float = 0.0

    # --- 재질 변조 (base material 대비 delta) ---
    albedo_scale: float = 1.0
    roughness_delta: float = 0.0
    metallic_delta: float = 0.0

    # --- 평가용 GT ---
    difficulty: str = "medium"
    # 집기 전 초기 자세에서 고정 카메라에 보이는가.
    # False면 팔이 물체를 재배향해야만 발견 가능하다 -> NBV의 존재 이유.
    visible_at_rest: bool = True
    seed: int = 0
    notes: str = ""


DEFECT_TYPES = (
    "stain",
    "scratch",
    "scuff",
    "dent",
)

# 이번 과제의 고려 대상. scuff(거칠기만 다른 마모면)와 dent(실제 변위)는
# 범위에서 제외했다. 타입 구현과 프리셋은 그대로 남겨두었으므로
# 시나리오에 한 줄 추가하면 다시 쓸 수 있다. 다만 scuff를 되살릴 때는
# 조명 감쇠(STAGE_LIGHT_DIM_FACTOR)를 0.3으로 되돌려야 한다.
ACTIVE_DEFECT_TYPES = (
    "stain",
    "scratch",
)

DEFECT_DIFFICULTIES = (
    "easy",
    "medium",
    "hard",
)


# ============================================================
# 타입별 프리셋
# ============================================================
#
# 1번 제안의 albedo / roughness / geometry 3축이 여기서 파라미터
# 조합으로 표현된다. 핵심은 scuff다 - 색도 형상도 안 바뀌고 오직
# 거칠기만 다르므로 정반사 각도가 맞는 뷰에서만 존재가 드러난다.
# 즉 NBV 없이는 원리적으로 놓칠 수 있는 결함이다.
#
#   stain   : 색만 변함        -> 어디서 봐도 보임 (대조군)
#   scratch : 법선 + 정반사    -> 시점 의존 강함 (주력)
#   scuff   : 거칠기만         -> 가장 어려움
#   dent    : 실제 변위        -> depth에도 신호 (1차 시나리오에서는 미사용)

DEFECT_TYPE_PRESETS = {
    "stain": {
        "length_mm": 9.0,
        "width_mm": 7.0,
        "depth_um": 0.0,
        "stroke_count": 1,
        "curvature": 0.0,
        "albedo_scale": 0.55,
        "roughness_delta": 0.05,
        "metallic_delta": 0.0,
        "difficulty": "easy",
    },
    "scratch": {
        "length_mm": 12.0,
        "width_mm": 0.6,
        # 폭 0.6mm에 깊이 50um이면 종횡비 1:12로, 벽면 최대 기울기가
        # 약 19도가 나온다. 15um(1:40)은 접시에 가까워 어떤 조명에서도
        # 정반사 로브에 걸리지 않았다.
        "depth_um": 50.0,
        "stroke_count": 1,
        "curvature": 0.15,
        # 플라스틱 스크래치는 소성 변형으로 하얗게 일어난다.
        # 금속의 광택 스크래치(매끈해짐)와 반대 방향이다.
        "albedo_scale": 1.08,
        "roughness_delta": 0.28,
        "metallic_delta": 0.0,
        "difficulty": "medium",
    },
    "scuff": {
        # scuff의 length/width는 스트로크 굵기가 아니라 마모 '영역'의
        # 크기다. 개별 미세 스크래치 굵기는 texture_utils가 정한다.
        "length_mm": 12.0,
        "width_mm": 8.0,
        # 형상 변화는 거의 0이어야 한다. 홈을 파면 그건 스크래치 다발이
        # 되어 모든 각도에서 글린트가 잡히고, '정반사 각도에서만 드러나는
        # 결함'이라는 이 타입의 존재 이유가 사라진다.
        "depth_um": 2.0,
        # 연마 자국의 이방성 노이즈 밀도로 쓰인다.
        "stroke_count": 14,
        "curvature": 0.4,
        "albedo_scale": 1.0,
        "roughness_delta": 0.18,
        "metallic_delta": 0.0,
        "difficulty": "hard",
    },
    # --- 아래 두 타입은 이번 과제 범위 밖 (ACTIVE_DEFECT_TYPES 참고) ---
    "dent": {
        "length_mm": 8.0,
        "width_mm": 8.0,
        "depth_um": 800.0,
        "stroke_count": 1,
        "curvature": 0.0,
        "albedo_scale": 1.0,
        "roughness_delta": 0.0,
        "metallic_delta": 0.0,
        "difficulty": "medium",
    },
}


def make_defect(
    name,
    defect_type,
    placement,
    **overrides,
):
    """타입 프리셋 위에 overrides를 얹어 DefectSpec을 만든다."""
    if defect_type not in DEFECT_TYPE_PRESETS:
        raise ValueError(
            f"[{name}] 알 수 없는 defect_type: {defect_type} "
            f"(가능: {sorted(DEFECT_TYPE_PRESETS)})"
        )

    unknown_keys = set(overrides) - {
        field_name
        for field_name in DefectSpec.__dataclass_fields__
    }

    if unknown_keys:
        raise ValueError(
            f"[{name}] DefectSpec에 없는 필드: {sorted(unknown_keys)}"
        )

    values = dict(DEFECT_TYPE_PRESETS[defect_type])
    values.update(overrides)

    return DefectSpec(
        name=name,
        defect_type=defect_type,
        placement=placement,
        **values,
    )


# ============================================================
# 시나리오 카탈로그
# ============================================================
#
# 흡착컵이 상단에 붙으므로 axial_ratio는 DEFECT_SUCTION_KEEPOUT_AXIAL_RATIO
# 미만으로 둔다. theta 180 부근(뒷면)과 axial_ratio 0.2 미만(하단)은
# 초기 자세에서 고정 카메라에 안 보이므로 visible_at_rest=False로 표시하고,
# 평가 때 "재배향해야만 발견되는 결함"으로 따로 집계한다.

# 난이도 축
# ------------------------------------------------------------
# scuff(거칠기만)와 dent(실제 변위)를 과제에서 제외하면서, 정반사
# 각도라는 물리적 난이도 축이 사라졌다. 남은 두 타입으로 난이도를
# 만드는 축은 다음과 같다.
#
#   1) 폭          0.38mm(2px, 해상도 하한) ~ 1.2mm(6px)
#   2) 배경 복잡도  민무늬 노란 몸통 vs 인쇄 라벨 위
#                  - 라벨 위 얇은 선은 글씨 획과 구분하기 어렵다
#                  - decal이 아래 표면 색을 샘플링하므로 실제로 어렵다
#   3) 초기 가시성  앞면(보임) vs 뒷면/아랫면(재배향해야 발견)
#                  - 이제 이것이 NBV의 주된 동기다
#   4) 대비        stain의 albedo_scale 0.55(뚜렷) ~ 0.85(희미)
#   5) 방향/개수    장축 평행 vs 수직, 단일 vs 다발
#
# 병 표면의 배경 구역 (정점 UV로 텍스처를 샘플링해 실측한 지도)
#   휘도 표준편차 기준, 0.06 이상을 '인쇄'로 본다.
#
#   인쇄 구역:  theta 315~45   axial 0.25~0.85  (앞 French's 라벨)
#              theta 150~225  axial 0.30~0.85  (뒤 영양성분표, 글씨 촘촘)
#   민무늬 구역: theta  60~120  거의 전 구간      (오른쪽 옆면)
#              theta 240~285  거의 전 구간      (왼쪽 옆면)
#              axial 0.20 띠   한 바퀴 전부      (라벨 아래)
#
# 안착 자세 실측 입사각 (고정 Zivid 카메라 기준):
#   theta   0 ->  46도 (보임)      theta  90 ->  96도 (edge-on)
#   theta 180 -> 136도 (뒷면)      theta 200 -> 147도 (뒷면)
#
# 두 제약이 겹치는 게 중요하다. 초기 자세에서 보이려면 theta가 0
# 근처여야 하는데, 그 구역은 axial 0.25 위로 전부 라벨이다. 따라서
# '보임 x 민무늬'는 theta 0 / axial 0.20 띠에서만 만들 수 있다.

_SCENARIO_SMOKE = [
    make_defect(
        "front_stain",
        "stain",
        SurfacePlacement(
            theta_deg=0.0,
            axial_ratio=0.20,
        ),
        seed=1,
        notes="민무늬 띠 위 뚜렷한 얼룩. 파이프라인 동작 확인용.",
    ),
]

# 전부 초기 자세에서 보이고 배경도 민무늬다. 검출기 baseline용.
_SCENARIO_BASIC = [
    make_defect(
        "front_stain",
        "stain",
        SurfacePlacement(
            theta_deg=0.0,
            axial_ratio=0.20,
        ),
        seed=1,
        notes="민무늬 띠.",
    ),
    make_defect(
        "front_scratch",
        "scratch",
        SurfacePlacement(
            theta_deg=30.0,
            axial_ratio=0.21,
            tangent_deg=20.0,
        ),
        length_mm=14.0,
        width_mm=0.9,
        seed=2,
        notes="민무늬 띠. 폭 0.9mm로 넉넉하다.",
    ),
    make_defect(
        "front_label_scratch",
        "scratch",
        SurfacePlacement(
            theta_deg=0.0,
            axial_ratio=0.50,
            tangent_deg=-40.0,
        ),
        length_mm=13.0,
        width_mm=1.0,
        seed=3,
        notes="빨간 라벨 위지만 폭 1.0mm라 배경과 구분된다.",
    ),
]

# 주력 시나리오. 초기 가시성과 배경 복잡도를 교차시킨다.
#   보임 x 민무늬  -> front_scratch    (가장 쉬움)
#   보임 x 라벨    -> label_scratch    (배경이 어려움)
#   숨음 x 민무늬  -> rear_scratch     (재배향 필요)
#   숨음 x 라벨    -> rear_label_scratch (둘 다 어려움)
_SCENARIO_NBV = [
    make_defect(
        "front_plain_scratch",
        "scratch",
        SurfacePlacement(
            theta_deg=0.0,
            axial_ratio=0.20,
            tangent_deg=25.0,
        ),
        length_mm=14.0,
        width_mm=0.8,
        difficulty="easy",
        seed=11,
        visible_at_rest=True,
        notes="보임 x 민무늬. 첫 뷰에서 잡혀야 한다.",
    ),
    make_defect(
        "front_label_scratch",
        "scratch",
        SurfacePlacement(
            theta_deg=0.0,
            axial_ratio=0.52,
            tangent_deg=-35.0,
        ),
        length_mm=12.0,
        width_mm=0.6,
        seed=12,
        visible_at_rest=True,
        notes="보임 x 인쇄(std 0.20). 글씨 획과 구분해야 한다.",
    ),
    make_defect(
        "side_plain_cluster",
        "scratch",
        SurfacePlacement(
            theta_deg=90.0,
            axial_ratio=0.45,
            tangent_deg=-55.0,
            jitter_mm=0.4,
        ),
        length_mm=9.0,
        width_mm=0.55,
        stroke_count=3,
        curvature=0.25,
        seed=13,
        visible_at_rest=False,
        notes="숨음 x 민무늬. 안착 자세 입사각 96도(edge-on).",
    ),
    make_defect(
        "rear_label_scratch",
        "scratch",
        SurfacePlacement(
            theta_deg=180.0,
            axial_ratio=0.50,
            tangent_deg=15.0,
        ),
        length_mm=13.0,
        width_mm=0.7,
        difficulty="hard",
        seed=14,
        visible_at_rest=False,
        notes="숨음 x 인쇄(std 0.16). 재배향 + 어려운 배경.",
    ),
    make_defect(
        "bottom_stain",
        "stain",
        SurfacePlacement(
            theta_deg=200.0,
            axial_ratio=0.22,
        ),
        length_mm=8.0,
        width_mm=6.0,
        albedo_scale=0.62,
        seed=15,
        visible_at_rest=False,
        notes="숨음 x 민무늬.",
    ),
]

# 전부 숨어 있고 배경도 어렵다. 폭은 해상도 하한 근처.
_SCENARIO_HARD = [
    make_defect(
        "rear_label_hairline_a",
        "scratch",
        SurfacePlacement(
            theta_deg=180.0,
            axial_ratio=0.50,
            tangent_deg=-15.0,
        ),
        length_mm=13.0,
        width_mm=0.42,
        depth_um=25.0,
        difficulty="hard",
        seed=21,
        visible_at_rest=False,
        notes="영양성분표 글씨 위(std 0.16). 폭 0.42mm = 약 2.2px.",
    ),
    make_defect(
        "rear_label_hairline_b",
        "scratch",
        SurfacePlacement(
            theta_deg=165.0,
            axial_ratio=0.65,
            tangent_deg=75.0,
        ),
        length_mm=10.0,
        width_mm=0.45,
        depth_um=25.0,
        difficulty="hard",
        seed=22,
        visible_at_rest=False,
    ),
    make_defect(
        "rear_label_hairline_c",
        "scratch",
        SurfacePlacement(
            theta_deg=210.0,
            axial_ratio=0.55,
            tangent_deg=-60.0,
        ),
        length_mm=11.0,
        width_mm=0.45,
        depth_um=25.0,
        difficulty="hard",
        seed=23,
        visible_at_rest=False,
    ),
    make_defect(
        "side_plain_hairline",
        "scratch",
        SurfacePlacement(
            theta_deg=270.0,
            axial_ratio=0.55,
            tangent_deg=-70.0,
        ),
        length_mm=12.0,
        width_mm=0.45,
        depth_um=25.0,
        difficulty="hard",
        seed=24,
        visible_at_rest=False,
        notes="민무늬지만 폭이 해상도 하한이라 어렵다.",
    ),
    make_defect(
        "rear_faint_stain",
        "stain",
        SurfacePlacement(
            theta_deg=195.0,
            axial_ratio=0.75,
        ),
        length_mm=8.0,
        width_mm=6.5,
        albedo_scale=0.85,
        difficulty="hard",
        seed=25,
        visible_at_rest=False,
        notes="희미한 얼룩(0.85배)이 인쇄 배경 위(std 0.17).",
    ),
]


def _build_eval_scenario():
    """recall 곡선용. 6방위 x 2높이에 균등 분포시킨 12개."""
    theta_steps = (
        0.0,
        60.0,
        120.0,
        180.0,
        240.0,
        300.0,
    )
    axial_steps = (
        0.32,
        0.68,
    )
    # scratch를 주로 두되 stain을 규칙적으로 섞는다.
    type_cycle = (
        "scratch",
        "scratch",
        "stain",
    )

    specs = []
    index = 0

    for axial_ratio in axial_steps:
        for theta_deg in theta_steps:
            defect_type = type_cycle[index % len(type_cycle)]
            overrides = {}

            if defect_type == "scratch":
                # 폭을 0.5 ~ 0.9mm로 순환시켜 크기 축도 함께 본다.
                overrides["width_mm"] = 0.5 + 0.2 * (index % 3)
                overrides["length_mm"] = 12.0

            specs.append(
                make_defect(
                    f"eval_{index:02d}_{defect_type}",
                    defect_type,
                    SurfacePlacement(
                        theta_deg=theta_deg,
                        axial_ratio=axial_ratio,
                        tangent_deg=(index * 37.0) % 180.0 - 90.0,
                        jitter_mm=0.3,
                    ),
                    seed=100 + index,
                    # 안착 자세에서는 theta가 0 근처일 때만 보인다.
                    visible_at_rest=(
                        theta_deg <= 45.0
                        or theta_deg >= 315.0
                    ),
                    **overrides,
                )
            )
            index += 1

    return specs


# ============================================================
# 랜덤 스크래치 시나리오
# ============================================================

# 배치 난수의 씨앗. 같은 값이면 같은 배치가 재현된다.
DEFECT_RANDOM_SCRATCH_SEED = int(
    os.environ.get(
        "BATTERY_DEFECT_RANDOM_SEED",
        str(DEFECT_GLOBAL_SEED),
    )
)

# 기본 생성 개수는 고정 10개가 아니라 이 범위에서 seed 기반으로
# 결정한다. 최소값은 항상 10 이상으로 강제한다. 배터리캡은 10개짜리
# surface slot 묶음을 순환 재사용하면서 위치를 이동하므로 10개를
# 넘어도 같은 자리에 완전히 겹치지 않는다.
DEFECT_RANDOM_SCRATCH_MIN_COUNT = max(
    10,
    int(
        os.environ.get(
            "BATTERY_DEFECT_RANDOM_MIN_COUNT",
            "10",
        )
    ),
)
DEFECT_RANDOM_SCRATCH_MAX_COUNT = int(
    os.environ.get(
        "BATTERY_DEFECT_RANDOM_MAX_COUNT",
        "15",
    )
)

if DEFECT_RANDOM_SCRATCH_MAX_COUNT < DEFECT_RANDOM_SCRATCH_MIN_COUNT:
    raise ValueError(
        "BATTERY_DEFECT_RANDOM_MAX_COUNT는 "
        "BATTERY_DEFECT_RANDOM_MIN_COUNT 이상이어야 합니다: "
        f"{DEFECT_RANDOM_SCRATCH_MAX_COUNT} < "
        f"{DEFECT_RANDOM_SCRATCH_MIN_COUNT}"
    )

# 기존 변수는 정확한 개수가 필요한 실험을 위해 계속 지원한다.
# 지정하지 않으면 [min, max]에서 뽑고, 지정해도 min 미만은 올린다.
_random_scratch_exact_count = os.environ.get(
    "BATTERY_DEFECT_RANDOM_COUNT"
)

if _random_scratch_exact_count is None:
    DEFECT_RANDOM_SCRATCH_COUNT = random.Random(
        DEFECT_RANDOM_SCRATCH_SEED ^ 0xC0FFEE
    ).randint(
        DEFECT_RANDOM_SCRATCH_MIN_COUNT,
        DEFECT_RANDOM_SCRATCH_MAX_COUNT,
    )
else:
    DEFECT_RANDOM_SCRATCH_COUNT = max(
        DEFECT_RANDOM_SCRATCH_MIN_COUNT,
        int(_random_scratch_exact_count),
    )

# 배치 가능한 axial 띠.
#
# 위: 흡착컵이 상단에 붙으므로 DEFECT_SUCTION_KEEPOUT_AXIAL_RATIO(0.88)
#     아래여야 한다. 여유를 두어 0.78로 끊는다.
# 아래: 병 바닥은 곡률이 심해 평면 decal이 뜬다. 실측으로 axial 0.22
#     자리의 얼룩에서 정점 2개가 0.81mm 떠서 경고가 났다. 0.18 아래는
#     더 심해지므로 피한다.
DEFECT_RANDOM_AXIAL_RANGE = (0.18, 0.78)

# 스크래치 폭/길이 난수 범위.
#
# 폭 하한 0.5mm는 DEFECT_MIN_RESOLVABLE_WIDTH_MM(0.38mm = 2px)보다
# 위다. 그 아래는 센서가 못 푸는 크기라 '검출 실패'가 NBV 탓인지
# 해상도 탓인지 구분되지 않는다.
DEFECT_RANDOM_WIDTH_RANGE_MM = (0.5, 0.9)
DEFECT_RANDOM_LENGTH_RANGE_MM = (9.0, 15.0)

# 결함 중심끼리 최소 이 정도는 떨어뜨린다 (표면 거리, mm).
#
# decal 패치가 한 변 15~20mm이므로 22mm면 두 패치가 겹치지 않는다.
# 겹치면 semantic ID가 서로를 가려 Recall 분모가 깨진다.
DEFECT_RANDOM_MIN_SEPARATION_MM = 22.0

# 병 치수. 표면 거리를 재려고 (theta, axial)을 mm로 바꿀 때 쓴다.
# GT 메시 bounds 실측: X 96.0, Y 191.4(장축), Z 58.2 mm.
# 단면은 반축 48.0 x 29.1mm 타원이고, Ramanujan 근사로 둘레 245.9mm.
DEFECT_BOTTLE_CIRCUMFERENCE_MM = 245.9
DEFECT_BOTTLE_LENGTH_MM = 191.4


def _surface_distance_mm(first, second):
    """(theta_deg, axial_ratio) 두 점 사이의 근사 표면 거리(mm).

    theta는 360도에서 되감기므로 원형 차이를 쓴다. 병을 원기둥으로
    펼친 근사라 정확하지는 않지만, 겹침을 막기 위한 하한으로는 충분하다.
    """
    theta_difference = abs(first[0] - second[0]) % 360.0
    theta_difference = min(theta_difference, 360.0 - theta_difference)
    arc_mm = theta_difference * DEFECT_BOTTLE_CIRCUMFERENCE_MM / 360.0
    axial_mm = abs(first[1] - second[1]) * DEFECT_BOTTLE_LENGTH_MM

    return math.hypot(arc_mm, axial_mm)


def _build_random_scratch_scenario(
    count=None,
    seed=None,
    min_separation_mm=None,
):
    """겹치지 않게 무작위로 흩뿌린 스크래치 시나리오.

    Recall의 분모를 키우려는 것이다. 주력 시나리오(nbv)는 결함이 5개뿐이라
    Recall이 20%p 단위로만 움직여 지표로 거칠다. 스크래치를 10개 이상
    만들면 10%p 이하의 해상도가 나오고, 위치가 한쪽으로 몰리지 않아
    NBV가 실제로 물체를 돌려봐야 한다.

    배치는 거부 샘플링이다. 후보 점을 뽑아 기존 결함 전부와의 표면 거리가
    min_separation_mm 이상일 때만 받는다. 실측 기준 배치 가능 띠가
    28,200mm2이고 결함 1개가 반지름 11mm 원반(380mm2)이라 충전율이 13%에
    그치므로, 거부 샘플링으로 충분히 놓인다.

    seed가 같으면 같은 배치가 나온다.
    """
    if count is None:
        count = DEFECT_RANDOM_SCRATCH_COUNT

    if seed is None:
        seed = DEFECT_RANDOM_SCRATCH_SEED

    if min_separation_mm is None:
        min_separation_mm = DEFECT_RANDOM_MIN_SEPARATION_MM

    count = int(count)

    if count <= 0:
        return []

    rng = random.Random(int(seed))
    axial_low, axial_high = DEFECT_RANDOM_AXIAL_RANGE
    placed = []
    # 후보 하나를 놓기까지의 최대 시도. 넘으면 띠가 포화된 것이다.
    max_attempts = 2000 * count
    attempts = 0

    while len(placed) < count:
        attempts += 1

        if attempts > max_attempts:
            raise RuntimeError(
                f"랜덤 스크래치 {count}개를 "
                f"{min_separation_mm}mm 간격으로 배치하지 못했습니다 "
                f"({len(placed)}개까지 놓음). count를 줄이거나 "
                "BATTERY_DEFECT_RANDOM_SEED를 바꾸세요."
            )

        candidate = (
            rng.uniform(0.0, 360.0),
            rng.uniform(axial_low, axial_high),
        )

        if any(
            _surface_distance_mm(candidate, existing) < min_separation_mm
            for existing in placed
        ):
            continue

        placed.append(candidate)

    specs = []

    for index, (theta_deg, axial_ratio) in enumerate(placed):
        width_mm = rng.uniform(*DEFECT_RANDOM_WIDTH_RANGE_MM)
        length_mm = rng.uniform(*DEFECT_RANDOM_LENGTH_RANGE_MM)
        specs.append(
            make_defect(
                f"rand_{index:02d}_scratch",
                "scratch",
                SurfacePlacement(
                    theta_deg=theta_deg,
                    axial_ratio=axial_ratio,
                    tangent_deg=rng.uniform(-90.0, 90.0),
                    jitter_mm=0.3,
                ),
                length_mm=round(length_mm, 2),
                width_mm=round(width_mm, 2),
                # resolve_seed()는 런타임에 전역 시드와 섞어주므로
                # 여기서는 결함별 정수만 준다. 배치 시드를 섞어 두어야
                # BATTERY_DEFECT_RANDOM_SEED를 바꿀 때 위치뿐 아니라
                # 텍스처 난수까지 함께 바뀐다.
                seed=(int(seed) + index) & 0x7FFFFFFF,
                # 안착 자세에서 고정 카메라에 보이는 것은 theta가 0
                # 근처일 때뿐이다 (실측 입사각: theta 0 -> 46도,
                # theta 90 -> 96도로 edge-on).
                visible_at_rest=(
                    theta_deg <= 45.0 or theta_deg >= 315.0
                ),
                notes=(
                    f"무작위 배치 (seed={seed}). "
                    f"theta={theta_deg:.1f} axial={axial_ratio:.2f}"
                ),
            )
        )

    return specs


DEFECT_SCENARIOS = {
    "none": [],
    "smoke": _SCENARIO_SMOKE,
    "basic": _SCENARIO_BASIC,
    "nbv": _SCENARIO_NBV,
    "hard": _SCENARIO_HARD,
    "eval": _build_eval_scenario(),
    "random_scratch": _build_random_scratch_scenario(),
}


# ============================================================
# 조회 / 검증
# ============================================================


def resolve_seed(spec):
    """전역 시드와 결함별 시드를 합쳐 결정적 시드를 만든다.

    단순 선형 결합은 인접한 spec.seed끼리 시드가 붙어 나와서
    jitter 같은 미세 랜덤화가 결함 간에 상관을 갖는다. splitmix64
    믹싱을 한 번 돌려 서로 독립적인 스트림이 되게 한다.
    """
    value = (
        DEFECT_GLOBAL_SEED * 0x9E3779B97F4A7C15 + spec.seed
    ) & 0xFFFFFFFFFFFFFFFF
    value = (
        (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9
    ) & 0xFFFFFFFFFFFFFFFF
    value = (
        (value ^ (value >> 27)) * 0x94D049BB133111EB
    ) & 0xFFFFFFFFFFFFFFFF
    value = value ^ (value >> 31)

    return value & 0x7FFFFFFF


def validate_defect_spec(spec):
    """카탈로그 선언의 물리적 타당성을 검사한다.

    치명적 오류는 raise하고, 검출 난이도상 주의가 필요한 항목은
    경고 문자열 리스트로 돌려준다.
    """
    warnings = []

    if spec.defect_type not in DEFECT_TYPES:
        raise ValueError(
            f"[{spec.name}] 알 수 없는 defect_type: {spec.defect_type}"
        )

    if spec.defect_type not in ACTIVE_DEFECT_TYPES:
        warnings.append(
            f"[{spec.name}] defect_type '{spec.defect_type}'는 이번 과제 "
            f"범위 밖입니다 (활성: {list(ACTIVE_DEFECT_TYPES)}). "
            "scuff를 쓰려면 조명 감쇠도 0.3으로 되돌려야 합니다."
        )

    if spec.difficulty not in DEFECT_DIFFICULTIES:
        raise ValueError(
            f"[{spec.name}] 알 수 없는 difficulty: {spec.difficulty}"
        )

    if spec.length_mm <= 0.0 or spec.width_mm <= 0.0:
        raise ValueError(
            f"[{spec.name}] length_mm/width_mm는 0보다 커야 합니다: "
            f"{spec.length_mm} x {spec.width_mm}"
        )

    if spec.stroke_count < 1:
        raise ValueError(
            f"[{spec.name}] stroke_count는 1 이상이어야 합니다: "
            f"{spec.stroke_count}"
        )

    if not 0.0 <= spec.placement.axial_ratio <= 1.0:
        raise ValueError(
            f"[{spec.name}] axial_ratio는 [0, 1] 범위여야 합니다: "
            f"{spec.placement.axial_ratio}"
        )

    if not -1.0 <= spec.roughness_delta <= 1.0:
        raise ValueError(
            f"[{spec.name}] roughness_delta는 [-1, 1] 범위여야 합니다: "
            f"{spec.roughness_delta}"
        )

    if spec.albedo_scale <= 0.0:
        raise ValueError(
            f"[{spec.name}] albedo_scale는 0보다 커야 합니다: "
            f"{spec.albedo_scale}"
        )

    if (
        spec.placement.axial_ratio
        > DEFECT_SUCTION_KEEPOUT_AXIAL_RATIO
    ):
        raise ValueError(
            f"[{spec.name}] axial_ratio {spec.placement.axial_ratio}가 "
            "흡착 keep-out 구간에 있습니다 "
            f"(상한 {DEFECT_SUCTION_KEEPOUT_AXIAL_RATIO})."
        )

    if spec.width_mm < DEFECT_MIN_RESOLVABLE_WIDTH_MM:
        warnings.append(
            f"[{spec.name}] width_mm={spec.width_mm}가 해상도 하한 "
            f"{DEFECT_MIN_RESOLVABLE_WIDTH_MM:.2f}mm 미만입니다 "
            f"(GSD {DEFECT_REFERENCE_GSD_MM_PER_PX}mm/px). subpixel "
            "결함이라 정반사 응답 변화로만 검출됩니다."
        )

    if (
        spec.defect_type == "scuff"
        and abs(spec.roughness_delta) < 0.1
    ):
        warnings.append(
            f"[{spec.name}] scuff인데 roughness_delta가 "
            f"{spec.roughness_delta}로 작습니다. 사실상 검출 불가일 수 "
            "있습니다."
        )

    longest_extent_mm = 1000.0 * max(DEFECT_NOMINAL_EXTENT_M)

    if spec.length_mm > 0.5 * longest_extent_mm:
        warnings.append(
            f"[{spec.name}] length_mm={spec.length_mm}가 물체 최장변의 "
            "절반을 넘습니다. decal이 표면을 벗어날 수 있습니다."
        )

    return warnings


def get_scenario_specs(scenario_name=None):
    """시나리오 이름으로 DefectSpec 리스트를 돌려준다."""
    if scenario_name is None:
        scenario_name = DEFECT_SCENARIO_NAME

    if scenario_name not in DEFECT_SCENARIOS:
        raise ValueError(
            f"알 수 없는 defect scenario: {scenario_name} "
            f"(가능: {sorted(DEFECT_SCENARIOS)})"
        )

    return list(DEFECT_SCENARIOS[scenario_name])


def adapt_defect_specs_for_batterycap(specs):
    """배터리캡의 검사 가능한 상면과 외측 벽에만 결함을 배치한다.

    바닥과 가까운 -Z 뒷면은 사용하지 않는다. mainboard2/rightcap/
    leftcap은 +Z 상면, rightsidewall/leftsidewall은 물체 중심에서 먼
    각각 +X/-X 외측면만 사용한다. sidewall은 Z 장축의 중상부만 골라
    테이블과 가까운 아래쪽도 피한다.
    """
    surface_slots = (
        # surface, mesh suffix, theta, axial, long/zero/ninety axes,
        # visible_at_rest, maximum defect length (mm)
        (
            "mainboard2_top",
            "/mainboard2",
            0.0,
            0.22,
            (0, 2, 1),
            True,
            None,
        ),
        (
            "rightcap_top",
            "/Rightblackcap/rightcap",
            0.0,
            0.50,
            (0, 2, 1),
            True,
            6.0,
        ),
        (
            "leftcap_top",
            "/Leftblackcap/leftcap",
            0.0,
            0.50,
            (0, 2, 1),
            True,
            6.0,
        ),
        (
            "rightsidewall_outer",
            "/rightsidewall",
            0.0,
            0.72,
            (2, 0, 1),
            False,
            8.0,
        ),
        (
            "leftsidewall_outer",
            "/leftsidewall",
            180.0,
            0.72,
            (2, 0, 1),
            False,
            8.0,
        ),
        (
            "mainboard2_top",
            "/mainboard2",
            0.0,
            0.42,
            (0, 2, 1),
            True,
            None,
        ),
        (
            "mainboard2_top",
            "/mainboard2",
            0.0,
            0.62,
            (0, 2, 1),
            True,
            None,
        ),
        (
            "mainboard2_top",
            "/mainboard2",
            0.0,
            0.82,
            (0, 2, 1),
            True,
            None,
        ),
        (
            "rightsidewall_outer",
            "/rightsidewall",
            0.0,
            0.48,
            (2, 0, 1),
            False,
            8.0,
        ),
        (
            "leftsidewall_outer",
            "/leftsidewall",
            180.0,
            0.48,
            (2, 0, 1),
            False,
            8.0,
        ),
    )
    surface_counts = {}
    adapted = []

    for index, spec in enumerate(specs):
        (
            surface_name,
            mesh_suffix,
            theta_deg,
            axial_ratio,
            axes,
            visible_at_rest,
            max_length_mm,
        ) = surface_slots[index % len(surface_slots)]
        cycle = index // len(surface_slots)

        # 10개를 넘겨 같은 슬롯을 재사용할 때 완전히 겹치지 않도록
        # 안전 범위 안에서 장축 위치를 조금씩 교차 이동한다.
        if cycle:
            direction = -1.0 if cycle % 2 else 1.0
            axial_ratio = float(
                min(0.84, max(0.20, axial_ratio + direction * 0.06 * cycle))
            )

        surface_rank = surface_counts.get(surface_name, 0)
        surface_counts[surface_name] = surface_rank + 1
        long_axis, theta_zero_axis, theta_ninety_axis = axes
        placement = SurfacePlacement(
            theta_deg=theta_deg,
            axial_ratio=axial_ratio,
            tangent_deg=spec.placement.tangent_deg,
            jitter_mm=(
                0.0
                if max_length_mm is not None
                else spec.placement.jitter_mm
            ),
            target_mesh_path_suffix=mesh_suffix,
            long_axis_index_override=long_axis,
            theta_zero_axis_index_override=theta_zero_axis,
            theta_ninety_axis_index_override=theta_ninety_axis,
        )
        length_mm = spec.length_mm

        if max_length_mm is not None:
            length_mm = min(length_mm, max_length_mm)

        adapted.append(
            replace(
                spec,
                name=(
                    f"batterycap_{surface_name}_{surface_rank:02d}_"
                    f"{spec.defect_type}"
                ),
                placement=placement,
                length_mm=length_mm,
                visible_at_rest=visible_at_rest,
                notes=(
                    f"batterycap {surface_name}; source={spec.name}. "
                    f"{spec.notes}"
                ).strip(),
            )
        )

    return adapted


def get_active_defect_specs(target_object_type=None):
    """현재 환경 설정에 따른 활성 결함 목록.

    BATTERY_ENABLE_DEFECTS=0이면 빈 리스트를 돌려준다.
    """
    if not ENABLE_DEFECTS:
        return []

    specs = get_scenario_specs()

    if target_object_type == "batterycap":
        specs = adapt_defect_specs_for_batterycap(specs)

    names = [spec.name for spec in specs]

    if len(names) != len(set(names)):
        duplicates = sorted(
            {name for name in names if names.count(name) > 1}
        )
        raise ValueError(
            f"결함 이름이 중복되었습니다: {duplicates}"
        )

    for spec in specs:
        for warning in validate_defect_spec(spec):
            print(f"[DEFECT][WARN] {warning}")

    return specs


def describe_active_defects(
    target_object_type=None,
    specs=None,
):
    """활성 결함 카탈로그를 사람이 읽을 요약 문자열로 만든다."""
    if not ENABLE_DEFECTS:
        return "[DEFECT] disabled (BATTERY_ENABLE_DEFECTS=0)"

    if specs is None:
        specs = get_active_defect_specs(
            target_object_type=target_object_type
        )

    target_summary = (
        f" target={target_object_type}"
        if target_object_type is not None
        else ""
    )
    lines = [
        f"[DEFECT] scenario={DEFECT_SCENARIO_NAME} "
        f"seed={DEFECT_GLOBAL_SEED} count={len(specs)}"
        f"{target_summary}",
    ]

    if DEFECT_DEBUG_VISIBLE:
        lines.append(
            "[DEFECT] *** DEBUG VISIBLE 모드 ***: 폭 x"
            f"{DEFECT_DEBUG_WIDTH_SCALE:g}, 길이 x"
            f"{DEFECT_DEBUG_LENGTH_SCALE:g}, 형광색 + 자체발광. "
            "검출 실험에는 쓰지 말 것."
        )

    for spec in specs:
        lines.append(
            f"[DEFECT]   {spec.name:<22} {spec.defect_type:<8} "
            f"theta={spec.placement.theta_deg:6.1f} "
            f"axial={spec.placement.axial_ratio:4.2f} "
            f"{spec.length_mm:5.1f}x{spec.width_mm:4.2f}mm "
            f"rough{spec.roughness_delta:+.2f} "
            f"albedo x{spec.albedo_scale:.2f} "
            f"{spec.difficulty:<6} "
            f"at_rest={'Y' if spec.visible_at_rest else 'N'}"
        )

    type_counts = {}

    for spec in specs:
        type_counts[spec.defect_type] = (
            type_counts.get(spec.defect_type, 0) + 1
        )

    lines.append(
        f"[DEFECT]   타입 분포: {type_counts} "
        f"(활성 타입: {list(ACTIVE_DEFECT_TYPES)})"
    )
    hidden_count = sum(
        1 for spec in specs if not spec.visible_at_rest
    )
    lines.append(
        f"[DEFECT]   초기 자세에서 미관측: {hidden_count}/{len(specs)} "
        "(재배향해야만 발견 가능)"
    )

    return "\n".join(lines)


if __name__ == "__main__":
    print(describe_active_defects())
