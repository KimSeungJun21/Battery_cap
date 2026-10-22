"""Scenario constants for Battery_rb.py."""

import os
from dataclasses import dataclass

import numpy as np
from pxr import Gf


RB_USD_PATH = "/home/seungjun/isaac_ws/Assets/Battery_env2.usd"
RB_PRIM_PATH = (
    "/World/Battery_env/"
    "rb5_850e_u_epick_isaac"
)
# Isaac PhysX only.  link0 and every EPick collider remain enabled; this
# controls just the imported RB5 arm-body collision meshes link1..link6.
RB_ARM_LINK_COLLISION_ENABLED = (
    os.environ.get(
        "BATTERY_RB_ARM_LINK_COLLISION_ENABLED",
        "0",
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

ZIVID_CAMERA_PATH = (
    "/World/Battery_env/Zivid_camera_01/"
    "Geometry/zivid_base_link/"
    "zivid_optical_frame/Zivid_Sensor"
)
# Zivid 2+ full resolution. This script treats the tuple as (width, height).
ZIVID_RESOLUTION = (
    2448,
    2048,
)
ZIVID_SAVE_DIR = (
    "/home/seungjun/isaac_ws/zivid_capture"
)
AUTO_RUN_NBV_CAPTURE = (
    os.environ.get(
        "BATTERY_AUTO_NBV_CAPTURE",
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
PUBLISH_TARGET_OBJECT_POSE = (
    os.environ.get(
        "BATTERY_PUBLISH_TARGET_OBJECT_POSE",
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
TARGET_OBJECT_POSE_TOPIC = os.environ.get(
    "BATTERY_TARGET_OBJECT_POSE_TOPIC",
    "/target_object/pose",
)
TARGET_OBJECT_POSE_FRAME_ID = os.environ.get(
    "BATTERY_TARGET_OBJECT_POSE_FRAME_ID",
    "world",
)
TARGET_OBJECT_POSE_PUBLISH_HZ = float(
    os.environ.get(
        "BATTERY_TARGET_OBJECT_POSE_PUBLISH_HZ",
        "20.0",
    )
)
TARGET_OBJECT_POSE_MODE = os.environ.get(
    "BATTERY_TARGET_OBJECT_POSE_MODE",
    "bbox_top_center",
)
# Whether the operator pinned the mode, so a per-target default can apply
# only when they did not.
TARGET_OBJECT_POSE_MODE_EXPLICIT = (
    "BATTERY_TARGET_OBJECT_POSE_MODE" in os.environ
)
TARGET_OBJECT_GRASP_Z_OFFSET = float(
    os.environ.get(
        "BATTERY_TARGET_OBJECT_GRASP_Z_OFFSET",
        "0.0",
    )
)

ENABLE_SUCTION_GRIPPER = (
    os.environ.get(
        "BATTERY_ENABLE_SUCTION_GRIPPER",
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
SUCTION_GRIPPER_TOPIC = os.environ.get(
    "BATTERY_SUCTION_GRIPPER_TOPIC",
    "/epick/suction_cmd",
)
TCP_PRIM_PATH = (
    RB_PRIM_PATH
    + "/Geometry/link0/link1/link2/link3/link4/link5/link6/tcp"
)
SUCTION_CUP_PRIM_PATH = (
    TCP_PRIM_PATH
    + "/epick_base_link/epick_body/epick_suction_cup"
)
SUCTION_TCP_PRIM_PATH = (
    SUCTION_CUP_PRIM_PATH
    + "/epick_tip/epick_tcp"
)
SUCTION_MAX_GRIP_DISTANCE = float(
    os.environ.get(
        "BATTERY_SUCTION_MAX_GRIP_DISTANCE",
        "0.02",
    )
)
# 흡착이 끊어지는 힘 한계(N).
#
# 머스타드 병은 약 0.6kg이라 정지 상태 하중은 6N 남짓이고, 50N이면
# 8배 여유다. 즉 '힘이 모자라서' 놓치는 상황은 아니었을 가능성이 크고,
# 실제 원인은 프록시 강체의 질량이 지정되지 않아 생긴 조인트 불안정
# 이었다 (SUCTION_PROXY_MASS 주석 참고).
#
# 다만 NBV가 물체를 최대 60도까지 기울이고 손목 회전 가속이 붙으면
# 순간 하중이 정적 하중을 크게 넘을 수 있어 여유를 더 뒀다.
SUCTION_COAXIAL_FORCE_LIMIT = float(
    os.environ.get(
        "BATTERY_SUCTION_COAXIAL_FORCE_LIMIT",
        "200.0",
    )
)
SUCTION_SHEAR_FORCE_LIMIT = float(
    os.environ.get(
        "BATTERY_SUCTION_SHEAR_FORCE_LIMIT",
        "200.0",
    )
)

# 흡착 물체가 붙는 프록시 강체의 질량(kg)과 관성.
#
# 프록시는 excludeFromArticulation=True인 FixedJoint로 손목 링크에
# 매달린 별도 강체다. 따라서 여기 준 질량과 관성이 조인트를 통해 팔에
# 그대로 실린다.
#
# 처음에 1.0kg / 1e-3을 줬다가 팔이 고꾸라졌다. 관성 1e-3은 0.01m
# 정육면체의 물리적 값(약 1.7e-5)의 60배라, 손목이 회전할 때 큰 반작용
# 토크를 만든다. 프록시는 그리퍼 본체를 대신하는 게 아니라 흡착 지점을
# 표시하는 도우미일 뿐이고, 실제 그리퍼 질량은 이미 팔 링크에 들어가
# 있다. 그러니 팔에 부담을 주지 않을 만큼 작아야 한다.
#
# 0으로 두면 MassAPI 자체를 적용하지 않는다 (수정 전 동작). 그 경우
# PhysX가 'negative mass'로 경고하고 작은 구로 근사한다.
SUCTION_PROXY_MASS = float(
    os.environ.get(
        "BATTERY_SUCTION_PROXY_MASS",
        "0.05",
    )
)
SUCTION_PROXY_INERTIA = float(
    os.environ.get(
        "BATTERY_SUCTION_PROXY_INERTIA",
        "1.0e-5",
    )
)

SUCTION_RETRY_INTERVAL = float(
    os.environ.get(
        "BATTERY_SUCTION_RETRY_INTERVAL",
        "1.0",
    )
)

# ============================================================
# Zivid 조명 리그
# ============================================================
#
# 실제 Zivid는 구조광 프로젝터가 카메라와 baseline만큼 떨어져 있고,
# 그 기하가 흠집의 가시성을 결정한다. roughness-only 결함(scuff)은
# 정반사 각도 근처에서만 드러나므로, 조명을 통제하지 않으면 NBV가
# 무엇을 최적화하는지 알 수 없다.
#
# 스테이지 USD의 기존 조명(defaultLight, FlatGrid/SphereLight)은
# 전방향이라 정반사 구조를 씻어낸다. 그래서 프로젝터를 주 광원으로
# 두고 기존 조명은 감쇠시킨다.

ZIVID_PROJECTOR_ENABLE = (
    os.environ.get(
        "BATTERY_ZIVID_PROJECTOR_ENABLE",
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
ZIVID_PROJECTOR_PRIM_NAME = "ZividProjector"

# 카메라 광학 프레임 기준 프로젝터 오프셋(m). Zivid 2+ 실기 baseline에
# 맞춘 값이다. 이 값이 0이면 동축 조명이 되어 정반사가 항상 카메라로
# 돌아오므로 시점 의존성이 사라진다.
ZIVID_PROJECTOR_BASELINE_M = float(
    os.environ.get(
        "BATTERY_ZIVID_PROJECTOR_BASELINE_M",
        "0.10",
    )
)

# 면광원 크기(m). 작을수록 정반사 하이라이트가 날카로워져 거칠기
# 차이가 잘 드러난다. 너무 작으면 노이즈가 심해진다.
ZIVID_PROJECTOR_WIDTH_M = float(
    os.environ.get(
        "BATTERY_ZIVID_PROJECTOR_WIDTH_M",
        "0.03",
    )
)
ZIVID_PROJECTOR_HEIGHT_M = float(
    os.environ.get(
        "BATTERY_ZIVID_PROJECTOR_HEIGHT_M",
        "0.02",
    )
)
ZIVID_PROJECTOR_INTENSITY = float(
    os.environ.get(
        "BATTERY_ZIVID_PROJECTOR_INTENSITY",
        "300000.0",
    )
)
ZIVID_PROJECTOR_COLOR_TEMPERATURE = float(
    os.environ.get(
        "BATTERY_ZIVID_PROJECTOR_COLOR_TEMPERATURE",
        "6500.0",
    )
)

# 조명 기본값은 목적에 따라 갈린다.
#
#   실험 모드: 프로젝터로 방향성 조명을 주되 기존 스테이지 조명은
#             그대로 둔다. scuff(거칠기만 다른 결함)를 과제에서
#             제외하면서 조명을 죽일 이유가 없어졌다.
#
#             감쇠 배율별 실측 (front_scratch 입사각 스윕):
#
#               감쇠   전체휘도  물체휘도  대비최소  대비최대  변동폭
#               0.02     16.7     79.1    0.249    0.341    1.4x
#               0.30     50.0    117.7    0.194    0.307    1.6x
#               1.00     87.4    156.1    0.099    0.259    2.6x  <- 채택
#
#             scratch는 형상이 있어 조명에 둔감하다. 전부 켜도 대비가
#             0.10 밑으로 안 내려가고, 오히려 변동폭이 가장 크며,
#             사람이 RGB를 눈으로 판단하기에 가장 밝다.
#
#             (참고: scuff는 정반대였다. 감쇠 0.3에서 변동폭 13.9배,
#              1.0에서 5.9배. 나중에 scuff를 다시 넣는다면 감쇠를
#              0.3으로 되돌려야 한다.)
#
_DEBUG_VISIBLE = (
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

# 완전한 암부를 피하기 위한 약한 환경광. 0이면 끈다.
ZIVID_AMBIENT_INTENSITY = float(
    os.environ.get(
        "BATTERY_ZIVID_AMBIENT_INTENSITY",
        "200.0" if _DEBUG_VISIBLE else "100.0",
    )
)

# 스테이지 기존 조명 감쇠 배율. 1.0이면 그대로 둔다.
STAGE_LIGHT_DIM_FACTOR = float(
    os.environ.get(
        "BATTERY_STAGE_LIGHT_DIM_FACTOR",
        "1.0",
    )
)

# Battery_rb.py's selectable inspection target.  The local batterycap asset is
# the default; the original YCB target remains available with
# BATTERY_TARGET_OBJECT=mustard or Battery_rb.py --target-object mustard.
TARGET_OBJECT_TYPE_ALIASES = {
    "batterycap": "batterycap",
    "battery_cap": "batterycap",
    "mustard": "mustard",
    "mustard_bottle": "mustard",
}
_target_object_type_raw = os.environ.get(
    "BATTERY_TARGET_OBJECT",
    "batterycap",
).strip().lower()

if _target_object_type_raw not in TARGET_OBJECT_TYPE_ALIASES:
    raise ValueError(
        "BATTERY_TARGET_OBJECT must be one of: "
        + ", ".join(sorted(TARGET_OBJECT_TYPE_ALIASES))
    )

TARGET_OBJECT_TYPE = TARGET_OBJECT_TYPE_ALIASES[_target_object_type_raw]
TARGET_COLLISION_ENABLED = (
    os.environ.get(
        "BATTERY_TARGET_COLLISION_ENABLED",
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
TARGET_GRAVITY_ENABLED = (
    os.environ.get(
        "BATTERY_TARGET_GRAVITY_ENABLED",
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
TARGET_COLLISION_APPROXIMATION = os.environ.get(
    "BATTERY_TARGET_COLLISION_APPROXIMATION",
    "convexHull",
).strip()

BATTERYCAP_USD_PATH = os.environ.get(
    "BATTERY_BATTERYCAP_USD_PATH",
    "/home/seungjun/isaac_ws/Assets/batterycap.usd",
)
BATTERYCAP_PRIM_PATH = "/World/BatteryCap"
# The suction cup aims at this descendant rather than the whole assembly.
# /World/BatteryCap's own bbox centre is dragged around by the side walls and
# the two black caps, whose tops sit 2.5 mm above the board, so the published
# grasp point ends up above a surface the cup never touches. mainboard2 is the
# flat 290 x 30 mm board through the middle - the only face wide enough for
# the 24 mm cup - and its top face is a clean plane at local Z=0.501.
# Empty string grasps the assembly root instead.
BATTERYCAP_GRASP_PRIM_SUFFIX = os.environ.get(
    "BATTERY_BATTERYCAP_GRASP_PRIM_SUFFIX",
    "/mainboard2",
)
# batterycap.usd's local Z extent is [0.4300, 0.5035] m.  The environment
# table top is world Z=0.7000 m, so Z=0.2700 places the lowest point on it.
BATTERYCAP_POSITION = Gf.Vec3d(
    0.0,
    0.5,
    0.27,
)
BATTERYCAP_SCALE = Gf.Vec3d(
    1.0,
    1.0,
    1.0,
)
BATTERYCAP_NEUTRAL_ROTATION = Gf.Vec3d(
    0.0,
    0.0,
    0.0,
)

MUSTARD_BOTTLE_ASSET_PATH = (
    "/Isaac/Props/YCB/Axis_Aligned_Physics/"
    "006_mustard_bottle.usd"
)
MUSTARD_BOTTLE_PRIM_PATH = "/World/MustardBottle"
MUSTARD_BOTTLE_POSITION = Gf.Vec3d(
    0.0,
    0.5,
    1.2,
)
MUSTARD_BOTTLE_SCALE = Gf.Vec3d(
    1.0,
    1.0,
    1.0,
)
MUSTARD_BOTTLE_NEUTRAL_ROTATION = Gf.Vec3d(
    0.0,
    0.0,
    0.0,
)

# 후보 pose는 frame 0의 neutral 이후 NBV score로 선택된다.
# 단일축 tilt, 대각 tilt, yaw가 섞인 tilt를 함께 넣어 후보 관측을 넓힌다.
MUSTARD_BOTTLE_CAPTURE_POSE_DEGREES = [
    (
        "neutral",
        (0.0, 0.0, 0.0),
    ),
    (
        "local_x_pos_15deg",
        (15.0, 0.0, 0.0),
    ),
    (
        "local_x_neg_15deg",
        (-15.0, 0.0, 0.0),
    ),
    (
        "local_x_pos_30deg",
        (30.0, 0.0, 0.0),
    ),
    (
        "local_x_neg_30deg",
        (-30.0, 0.0, 0.0),
    ),
    (
        "local_x_pos_45deg",
        (45.0, 0.0, 0.0),
    ),
    (
        "local_x_neg_45deg",
        (-45.0, 0.0, 0.0),
    ),
    (
        "local_x_pos_60deg",
        (60.0, 0.0, 0.0),
    ),
    (
        "local_x_neg_60deg",
        (-60.0, 0.0, 0.0),
    ),
    (
        "local_y_pos_15deg",
        (0.0, 15.0, 0.0),
    ),
    (
        "local_y_neg_15deg",
        (0.0, -15.0, 0.0),
    ),
    (
        "local_y_pos_30deg",
        (0.0, 30.0, 0.0),
    ),
    (
        "local_y_neg_30deg",
        (0.0, -30.0, 0.0),
    ),
    (
        "local_y_pos_45deg",
        (0.0, 45.0, 0.0),
    ),
    (
        "local_y_neg_45deg",
        (0.0, -45.0, 0.0),
    ),
    (
        "local_y_pos_60deg",
        (0.0, 60.0, 0.0),
    ),
    (
        "local_y_neg_60deg",
        (0.0, -60.0, 0.0),
    ),
    (
        "local_x_pos_90deg",
        (90.0, 0.0, 0.0),
    ),
    (
        "local_x_neg_90deg",
        (-90.0, 0.0, 0.0),
    ),
    (
        "local_y_pos_90deg",
        (0.0, 90.0, 0.0),
    ),
    (
        "local_y_neg_90deg",
        (0.0, -90.0, 0.0),
    ),
    (
        "diag_x_pos_30_y_pos_30",
        (30.0, 30.0, 0.0),
    ),
    (
        "diag_x_pos_30_y_neg_30",
        (30.0, -30.0, 0.0),
    ),
    (
        "diag_x_neg_30_y_pos_30",
        (-30.0, 30.0, 0.0),
    ),
    (
        "diag_x_neg_30_y_neg_30",
        (-30.0, -30.0, 0.0),
    ),
    (
        "diag_x_pos_45_y_pos_30",
        (45.0, 30.0, 0.0),
    ),
    (
        "diag_x_pos_45_y_neg_30",
        (45.0, -30.0, 0.0),
    ),
    (
        "diag_x_neg_45_y_pos_30",
        (-45.0, 30.0, 0.0),
    ),
    (
        "diag_x_neg_45_y_neg_30",
        (-45.0, -30.0, 0.0),
    ),
    (
        "diag_x_pos_30_y_pos_45",
        (30.0, 45.0, 0.0),
    ),
    (
        "diag_x_pos_30_y_neg_45",
        (30.0, -45.0, 0.0),
    ),
    (
        "diag_x_neg_30_y_pos_45",
        (-30.0, 45.0, 0.0),
    ),
    (
        "diag_x_neg_30_y_neg_45",
        (-30.0, -45.0, 0.0),
    ),
    (
        "tilt_x_pos_30_z_pos_45",
        (30.0, 0.0, 45.0),
    ),
    (
        "tilt_x_pos_30_z_neg_45",
        (30.0, 0.0, -45.0),
    ),
    (
        "tilt_x_neg_30_z_pos_45",
        (-30.0, 0.0, 45.0),
    ),
    (
        "tilt_x_neg_30_z_neg_45",
        (-30.0, 0.0, -45.0),
    ),
    (
        "tilt_y_pos_30_z_pos_45",
        (0.0, 30.0, 45.0),
    ),
    (
        "tilt_y_pos_30_z_neg_45",
        (0.0, 30.0, -45.0),
    ),
    (
        "tilt_y_neg_30_z_pos_45",
        (0.0, -30.0, 45.0),
    ),
    (
        "tilt_y_neg_30_z_neg_45",
        (0.0, -30.0, -45.0),
    ),
    (
        "tilt_x_pos_45_z_pos_90",
        (45.0, 0.0, 90.0),
    ),
    (
        "tilt_x_neg_45_z_pos_90",
        (-45.0, 0.0, 90.0),
    ),
    (
        "tilt_y_pos_45_z_pos_90",
        (0.0, 45.0, 90.0),
    ),
    (
        "tilt_y_neg_45_z_pos_90",
        (0.0, -45.0, 90.0),
    ),
]
MUSTARD_BOTTLE_CAPTURE_POSES = [
    (
        pose_name,
        Gf.Vec3d(*rotation_xyz_deg),
    )
    for pose_name, rotation_xyz_deg
    in MUSTARD_BOTTLE_CAPTURE_POSE_DEGREES
]

# Arm-held NBV can't teleport to an absolute pose - only a subset of the
# tilts above are safe to reach by actually reorienting the wrist while
# holding the object, so this keeps the conservative (<=60 deg) ones.
MUSTARD_BOTTLE_ARM_CAPTURE_POSE_MAX_DEGREES = 60.0
# Trims the (already angle-filtered) candidate pool to its first N entries so
# a test run doesn't have to grind through dozens of MoveGroup plan/execute
# attempts - each failed candidate costs ~10-30s (planning timeout + relay
# settle time). List order front-loads single-axis x/y tilts before the
# harder two-axis diag_/tilt_ combos, so a smaller slice also tends to be
# the more reachable subset, not just a shorter one.
# Some candidates in the pool are genuinely unreachable while holding the
# object (skipped as "unreachable" at runtime), so the pool needs enough
# slack beyond NBV_MAX_CAPTURE_COUNT to still land ~10 successful captures.
MUSTARD_BOTTLE_ARM_CAPTURE_POSE_LIMIT = int(
    os.environ.get(
        "BATTERY_ARM_CAPTURE_POSE_LIMIT",
        "20",
    )
)
MUSTARD_BOTTLE_ARM_CAPTURE_POSES = [
    (pose_name, rotation_xyz_deg)
    for pose_name, rotation_xyz_deg in MUSTARD_BOTTLE_CAPTURE_POSE_DEGREES
    if max(abs(component) for component in rotation_xyz_deg)
    <= MUSTARD_BOTTLE_ARM_CAPTURE_POSE_MAX_DEGREES
][:MUSTARD_BOTTLE_ARM_CAPTURE_POSE_LIMIT]

# A flat battery needs poses beyond 90 degrees to expose its underside to the
# fixed camera. These are battery-only candidates: mustard keeps its original
# conservative pose set. X-rolls are listed first because they swing the
# board's short 30 mm axis; Y-pitches remain as alternate IK solutions.
BATTERYCAP_BACKSIDE_CAPTURE_POSE_DEGREES = [
    ("local_x_pos_120deg", (120.0, 0.0, 0.0)),
    ("local_x_neg_120deg", (-120.0, 0.0, 0.0)),
    ("local_x_pos_150deg", (150.0, 0.0, 0.0)),
    ("local_x_neg_150deg", (-150.0, 0.0, 0.0)),
    ("local_x_pos_180deg", (180.0, 0.0, 0.0)),
    ("local_x_neg_180deg", (-180.0, 0.0, 0.0)),
    ("local_y_pos_120deg", (0.0, 120.0, 0.0)),
    ("local_y_neg_120deg", (0.0, -120.0, 0.0)),
    ("local_y_pos_150deg", (0.0, 150.0, 0.0)),
    ("local_y_neg_150deg", (0.0, -150.0, 0.0)),
    ("local_y_pos_180deg", (0.0, 180.0, 0.0)),
    ("local_y_neg_180deg", (0.0, -180.0, 0.0)),
]

# Put underside candidates immediately after the four 90-degree side views so
# the pose limit does not silently truncate them. Remaining diagonal/yaw
# candidates follow afterwards.
BATTERYCAP_CAPTURE_POSE_DEGREES = (
    MUSTARD_BOTTLE_CAPTURE_POSE_DEGREES[:21]
    + BATTERYCAP_BACKSIDE_CAPTURE_POSE_DEGREES
    + MUSTARD_BOTTLE_CAPTURE_POSE_DEGREES[21:]
)
BATTERYCAP_ARM_CAPTURE_POSE_MAX_DEGREES = float(
    os.environ.get(
        "BATTERY_BATTERYCAP_ARM_CAPTURE_POSE_MAX_DEGREES",
        "180.0",
    )
)
BATTERYCAP_ARM_CAPTURE_POSE_LIMIT = int(
    os.environ.get(
        "BATTERY_BATTERYCAP_ARM_CAPTURE_POSE_LIMIT",
        os.environ.get("BATTERY_ARM_CAPTURE_POSE_LIMIT", "36"),
    )
)
BATTERYCAP_ARM_CAPTURE_POSES = [
    (pose_name, rotation_xyz_deg)
    for pose_name, rotation_xyz_deg in BATTERYCAP_CAPTURE_POSE_DEGREES
    if max(abs(component) for component in rotation_xyz_deg)
    <= BATTERYCAP_ARM_CAPTURE_POSE_MAX_DEGREES
][:BATTERYCAP_ARM_CAPTURE_POSE_LIMIT]

POSE_SETTLE_UPDATE_COUNT = 5

# 물체가 '실제로 들어올려졌는가'를 판정하는 기준.
#
# 예전에는 절대 world Z 0.15m를 썼다. 물체가 바닥(z~0.06-0.09)에 놓인다는
# 가정이었는데, 실제 스테이지에는 테이블이 있어 안착 높이가 z=0.7288이다.
# 그래서 0.7288 >= 0.15가 항상 참이 되어, 테이블 위에 가만히 있는 상태도
# '들어올려짐'으로 판정됐다. 안정성 검사도 물체가 안 움직인다는 이유로
# 함께 통과해, 흡착 명령 1.25초 만에 NBV가 시작되며 픽 스크립트의
# 들어올리기 동작을 가로챘다.
#
# 이제는 흡착 직전의 안착 높이를 기준으로 '얼마나 올라갔는가'를 본다.
# 테이블 높이가 바뀌어도 그대로 동작한다.
LIFT_DETECTION_RISE_M = float(
    os.environ.get(
        "BATTERY_LIFT_DETECTION_RISE_M",
        "0.10",
    )
)

# 선택적 절대 높이 하한. 씬에 따라 달라지므로 기본은 비활성(0.0)이다.
# 특정 높이 위에서만 NBV를 돌리고 싶을 때만 설정한다.
LIFT_DETECTION_HEIGHT = float(
    os.environ.get(
        "BATTERY_LIFT_DETECTION_HEIGHT",
        "0.0",
    )
)
# Frame throttle for polling get_gripped_objects() while waiting for a pick.
PICK_DETECTION_CHECK_INTERVAL = int(
    os.environ.get(
        "BATTERY_PICK_DETECTION_CHECK_INTERVAL",
        "15",
    )
)

# Crossing LIFT_DETECTION_HEIGHT only means the suction pick-and-lift motion
# has started, not that it has finished - the selected suction picker's own
# MoveGroup goal can still be executing (or the object still swinging) at
# that instant. Starting the arm-driven NBV sweep then races move_group's
# trajectory_execution_manager ("Cannot push a new trajectory while another
# is being executed") and captures a transient, still-moving TCP pose as the
# NBV reference. So require the object height to stop changing for this many
# consecutive PICK_DETECTION_CHECK_INTERVAL polls (beyond the height
# threshold) before treating the pick as actually settled.
LIFT_STABILITY_CHECK_COUNT = int(
    os.environ.get(
        "BATTERY_LIFT_STABILITY_CHECK_COUNT",
        "3",
    )
)
LIFT_STABILITY_TOLERANCE = float(
    os.environ.get(
        "BATTERY_LIFT_STABILITY_TOLERANCE",
        "0.002",
    )
)

# Object-frame Octree/occupancy grid settings.
# OBJECT_GRID_ORIGIN=None이면 root size 기준으로 object origin 중심에 둔다.
# 예: root_size=0.17, leaf_size=0.01이면 17 x 17 x 17 grid가 된다.
# custom origin이 필요하면 np.array([x, y, z], dtype=np.float32)로 지정한다.
OBJECT_GRID_LEAF_SIZE = 0.01
OBJECT_GRID_ROOT_SIZE = 0.25
OBJECT_GRID_ORIGIN = None

# Attention-driven NBV uses a target-centered object-frame ROI.
NBV_ROI_MIN = np.array(
    [-0.08, -0.12, -0.08],
    dtype=np.float64,
)
NBV_ROI_MAX = np.array(
    [0.08, 0.12, 0.08],
    dtype=np.float64,
)

# batterycap.usd uses an origin below the product. Its meshes occupy roughly
# X=[-0.145, 0.145], Y=[-0.015, 0.015], Z=[0.430, 0.5035] m, so both the
# occupancy cube and attention ROI must be centered around that authored mesh
# instead of (0, 0, 0).
#
# 0.32 m leaves about 15 mm on each end of the 290 mm long product and reduces
# the 10 mm occupancy grid from 35^3=42,875 to 32^3=32,768 voxels (23.6%).
# Keep this an exact multiple of OBJECT_GRID_LEAF_SIZE.
BATTERYCAP_OBJECT_GRID_ROOT_SIZE = float(
    os.environ.get(
        "BATTERY_BATTERYCAP_OBJECT_GRID_ROOT_SIZE",
        "0.32",
    )
)
# Centre of the cube: the authored mesh's Z midpoint, (0.430 + 0.5035) / 2.
BATTERYCAP_OBJECT_GRID_CENTER = np.array(
    [0.0, 0.0, 0.46675],
    dtype=np.float32,
)
# resolve_object_grid_bounds() takes `origin` as the cube's MIN CORNER - its
# None case is -0.5 * root_size, i.e. a cube centred on (0, 0, 0) - so the
# centre above has to be converted.
#
# Passing the centre straight through put the cube at X/Y=[0, 0.35] and
# Z=[0.46675, 0.81675], which holds just one octant of the part (X>=0, Y>=0,
# top ~37 mm). The surface denominator collapsed from a 320 cm2 surface to 36
# voxels, two views covered it, and the arm-driven NBV sweep declared "100%
# coverage" and stopped after a single move with 8 of 10 defects unseen.
BATTERYCAP_OBJECT_GRID_ORIGIN = (
    BATTERYCAP_OBJECT_GRID_CENTER
    - 0.5 * BATTERYCAP_OBJECT_GRID_ROOT_SIZE
).astype(np.float32)
BATTERYCAP_NBV_ROI_MIN = np.array(
    [-0.16, -0.03, 0.42],
    dtype=np.float64,
)
BATTERYCAP_NBV_ROI_MAX = np.array(
    [0.16, 0.03, 0.515],
    dtype=np.float64,
)
# NBV 점수 계산 시 광선을 뽑을 픽셀 간격.
#
# 32이면 2448x2048에서 후보당 77x64 = 4,928개 광선이라 후보 6개에
# 2.6초가 걸렸다. 64로 올리면 광선이 1/4이 되어 0.6초대가 된다.
# occupancy 격자가 1cm이고 물체가 0.45m 거리라 광선 하나가 voxel
# 여러 개를 지나므로, 이 정도 간격에서도 가시 voxel 집합은 거의
# 유지된다.
NBV_PIXEL_STRIDE = int(
    os.environ.get(
        "BATTERY_NBV_PIXEL_STRIDE",
        "64",
    )
)
NBV_MAX_CAPTURE_COUNT = 10
NBV_MIN_EXPECTED_ENTROPY = 1e-6

# NBV 점수에서 '물체 표면이 아닌' voxel의 entropy에 줄 가중치.
#
# 표면 voxel은 항상 1.0이다. 이 값이 1.0이면 격자 전체를 똑같이 세는
# 예전 동작이고, 작을수록 점수가 표면만 본다.
#
# 예전 동작의 문제: 평가 지표는 표면 커버리지인데 점수는 격자 전체
# unknown을 셌다. 물체에서 떨어진 허공을 넓게 스치는 자세가 이겨서,
# 실측 10프레임 중 f5/f7/f9가 각각 +0.8/+1.3/+0.8%p에 그치면서
# 9.6초(전체의 32%)를 썼다.
#
# 0.001인 이유는 실측이다. 저장된 occupancy snapshot 10개에 실제
# ROI를 적용해 '표면 항'과 '배경 항'의 크기를 재보면:
#
#   가중치     점수에서 배경이 차지하는 비중
#   1.0        93 ~ 99%      <- 예전 동작. 사실상 표면을 안 보고 있었다
#   0.1        58 ~ 91%
#   0.01       12 ~ 51%
#   0.001       1 ~ 10%      <- 동점일 때만 갈린다
#
# 한 뷰에서 보이는 voxel이 3100~4100개인데 표면 voxel은 632개뿐이라,
# 가중치를 조금만 남겨도 배경이 점수를 가져간다. 목표가 표면 커버리지
# 하나뿐이므로 배경은 순수 tie-break로만 남긴다.
#
# 0으로 두지 않는 건, 표면이 전부 관측된 뒤에도 후보 간 순서가
# 정해져야 교착되지 않기 때문이다.
#
# 1.0으로 되돌리면 예전 동작과 완전히 같다 (단위 검증으로 확인).
NBV_BACKGROUND_SCORE_WEIGHT = float(
    os.environ.get(
        "BATTERY_NBV_BACKGROUND_SCORE_WEIGHT",
        "0.001",
    )
)

# 방향성 표면 NBV 점수의 object-local 법선 그룹별 가중치. 기본값은
# 기존 동작과 동일한 1:1:1이며, 실험 wrapper에서 batterycap 윗면을
# 우선하도록 덮어쓸 수 있다.
NBV_SURFACE_TOP_SCORE_WEIGHT = max(
    0.0,
    float(os.environ.get("BATTERY_NBV_SURFACE_TOP_SCORE_WEIGHT", "1.0")),
)
NBV_SURFACE_BOTTOM_SCORE_WEIGHT = max(
    0.0,
    float(os.environ.get("BATTERY_NBV_SURFACE_BOTTOM_SCORE_WEIGHT", "1.0")),
)
NBV_SURFACE_SIDE_SCORE_WEIGHT = max(
    0.0,
    float(os.environ.get("BATTERY_NBV_SURFACE_SIDE_SCORE_WEIGHT", "1.0")),
)

# 후보 pose를 거를 최소 시선 이격(도).
#
# 월드 카메라는 고정이고 팔이 물체를 돌린다. 그래서 '어느 면이 보이는가'는
# object frame에서 본 카메라 방향이 결정한다. 그런데 물체의 장축이 시선을
# 향한 상태라 local_y_* 후보들은 사실상 시선축 스핀이다. 실측한 frame 8->9는
# 물체가 44.8도 회전했는데 시선 방향은 10.8도만 바뀌었고, 새로 해소된 voxel이
# 10개에 그쳤다 (frame 5는 0개).
#
# 이미 관측한 방향들과 이 각도 미만으로 붙어 있는 후보는 중복 뷰로 보고
# 건너뛴다. 전부 걸리면 교착되므로 가장 이격이 큰 하나는 남긴다.
# 방향성 표면(surfel) 기준 커버리지 목표. 위치만 저장하는 10 mm voxel은
# 2 mm 두께 batterycap의 윗면과 아랫면을 합치므로 검사 완료 판정에는
# 쓰지 않는다. 점유 격자는 충돌/공간 추론에 그대로 사용하고, 표면 검사는
# 위치와 법선을 함께 가진 surfel을 별도로 추적한다.
NBV_SURFACE_COVERAGE_TARGET = float(
    os.environ.get(
        "BATTERY_NBV_SURFACE_COVERAGE_TARGET",
        "0.93",
    )
)
NBV_TOP_SURFACE_COVERAGE_TARGET = max(
    0.0,
    float(
        os.environ.get(
            "BATTERY_NBV_TOP_SURFACE_COVERAGE_TARGET",
            "0.0",
        )
    ),
)
NBV_SIDE_SURFACE_COVERAGE_TARGET = max(
    0.0,
    float(
        os.environ.get(
            "BATTERY_NBV_SIDE_SURFACE_COVERAGE_TARGET",
            "0.0",
        )
    ),
)

# 직전 촬영 대비 방향성 표면 커버리지 증가가 이 값보다 작은 상태가
# 지정 횟수만큼 연속되면 종료한다. gain이 0이면 비활성화한다.
# IMSI top/side 전용 모드에서는 목표 미달 조기종료에 쓰지 않고 진단만
# 남긴다. 성공 종료는 실측 top/side 동시 목표 달성으로만 판정한다.
NBV_MIN_SURFACE_COVERAGE_GAIN = max(
    0.0,
    float(
        os.environ.get(
            "BATTERY_NBV_MIN_SURFACE_COVERAGE_GAIN",
            "0.0",
        )
    ),
)
NBV_MIN_SURFACE_COVERAGE_GAIN_CONSECUTIVE = max(
    1,
    int(
        os.environ.get(
            "BATTERY_NBV_MIN_SURFACE_COVERAGE_GAIN_CONSECUTIVE",
            "1",
        )
    ),
)

# batterycap은 전체 면적 목표와 별도로 바닥 가까운 -Z 면도 반드시 본다.
# mustard는 아래 TargetObjectConfig에서 0.0을 사용해 기존 동작을 유지한다.
BATTERYCAP_BOTTOM_SURFACE_COVERAGE_TARGET = float(
    os.environ.get(
        "BATTERY_BATTERYCAP_BOTTOM_SURFACE_COVERAGE_TARGET",
        "0.90",
    )
)

NBV_SURFEL_SAMPLE_COUNT = int(
    os.environ.get("BATTERY_NBV_SURFEL_SAMPLE_COUNT", "50000")
)
NBV_SURFEL_MATCH_DISTANCE = float(
    os.environ.get("BATTERY_NBV_SURFEL_MATCH_DISTANCE", "0.003")
)
NBV_SURFEL_MIN_FACING_COS = float(
    os.environ.get("BATTERY_NBV_SURFEL_MIN_FACING_COS", "0.05")
)
NBV_SURFEL_PIXEL_BIN_SIZE = int(
    os.environ.get("BATTERY_NBV_SURFEL_PIXEL_BIN_SIZE", "8")
)
NBV_SURFEL_OCCLUSION_TOLERANCE = float(
    os.environ.get("BATTERY_NBV_SURFEL_OCCLUSION_TOLERANCE", "0.003")
)

# Isaac 쪽에서 relay의 이동 결과를 기다리는 제한 시간(초).
#
# trajectory bridge의 관절 수렴 대기(최대 30초)보다 relay의
# --goal-timeout(기본 40초)이 길고, 이 값은 다시 relay timeout과 취소
# 대기(최대 2초)보다 길어야 한다. 짧으면 실제 로봇이 움직이는 중인데
# Isaac runner가 먼저 포기해 결과가 어긋난다.
# 한 프레임에서 연속 이동 실패를 몇 번까지 허용할지.
#
#   0 (또는 음수) = 무제한. 남은 후보를 끝까지 시도한다
#   1            = 첫 실패에서 그 라운드를 끊는다 (최소 시간)
#   10           = 10번 연달아 실패하면 끊는다 <- 기본값
#
# 환경변수로 바꾼다:
#   BATTERY_NBV_MAX_CONSECUTIVE_MOVE_FAILURES=10 ./run_battery_rb_humble.sh
#
# 무엇을 우선하느냐의 문제다.
#
# 시간을 아끼려면 낮게: 실패 후보가 몰리면 남은 후보를 전부 시도하느라
# 시간을 다 쓴다. 실측으로 후보 12개가 각각 goal-timeout 5.3초에 걸려
# 63.6초를 썼고 전체 105.8초의 60%였다. 1로 두면 이 낭비가 0.16초로
# 줄었다 (실측 41.0초 -> 24.1초 실행).
#
# 커버리지를 끝까지 올리려면 0: 연속 실패가 '더 볼 곳이 없다'를 뜻하는
# 것은 아니다. 도달 불가능한 자세가 우연히 상위에 몰렸을 뿐, 그 아래
# 후보는 갈 수 있을지 모른다. 실측 24.1초 실행은 첫 실패에서 끊느라
# 남은 후보 10개를 시도조차 안 하고 84.3%에서 멈췄다.
#
# 지금은 시간보다 도달 가능한 후보를 충분히 시도하는 것을 우선한다.
# 무제한 대신 10회로 두어 통신/제어 계통의 연속 실패에는 종료할 수 있다.
NBV_MAX_CONSECUTIVE_MOVE_FAILURES = int(
    os.environ.get(
        "BATTERY_NBV_MAX_CONSECUTIVE_MOVE_FAILURES",
        "10",
    )
)

NBV_MOVE_RESULT_TIMEOUT = float(
    os.environ.get(
        "BATTERY_NBV_MOVE_RESULT_TIMEOUT",
        "50.0",
    )
)

# 후보 자세에서 물체 bbox가 센서에 담기는 비율의 하한.
#
# 후보 이름은 회전이지만 실제로는 TCP를 고정하고 방향만 바꾸므로 물체가
# 호를 그리며 움직인다. 실측 런에서 카메라 거리가 0.282~0.488 m로
# 흔들리고 광축 이탈이 26.5도(수직 half-FOV 22.9도)까지 가서, 10프레임
# 중 bbox 8개가 모두 센서에 담긴 것은 2개뿐이었다. 엔트로피 점수는
# 화면 밖을 벌하지 않으므로 여기서 후보 단계에서 잘라낸다.
#
# 1.0은 '화각을 전혀 벗어나지 말 것'이다. 290mm 판이 0.3~0.5m 거리에
# 있어 이 기준을 통과하는 후보가 적을 수 있고, 전부 걸리면 가장 잘
# 담기는 하나만 남긴다(교착 방지). 시퀀스가 너무 짧아지면 0.75 정도로
# 완화하면 된다.
NBV_MIN_IN_FRAME_FRACTION = float(
    os.environ.get(
        "BATTERY_NBV_MIN_IN_FRAME_FRACTION",
        "1.0",
    )
)
# 센서 테두리에서 이만큼 안쪽까지만 '담긴 것'으로 본다.
NBV_FRAME_MARGIN_PX = float(
    os.environ.get(
        "BATTERY_NBV_FRAME_MARGIN_PX",
        "0.0",
    )
)

NBV_MIN_VIEW_SEPARATION_DEG = float(
    os.environ.get(
        "BATTERY_NBV_MIN_VIEW_SEPARATION_DEG",
        "20.0",
    )
)


def get_mustard_bottle_usd_path(
    assets_root_path,
):
    return (
        assets_root_path
        + MUSTARD_BOTTLE_ASSET_PATH
    )


@dataclass(frozen=True)
class TargetObjectConfig:
    """Resolved asset and inspection settings for one selectable target."""

    key: str
    description: str
    object_label: str
    usd_path: str
    prim_path: str
    position: Gf.Vec3d
    neutral_rotation: Gf.Vec3d
    scale: Gf.Vec3d
    log_prefix: str
    arm_capture_poses: tuple
    create_missing_colliders: bool
    object_grid_leaf_size: float
    object_grid_root_size: float
    object_grid_origin: object
    nbv_roi_min: np.ndarray
    nbv_roi_max: np.ndarray
    defect_long_axis_index: int
    defect_theta_zero_axis_index: int
    defect_theta_ninety_axis_index: int
    grasp_prim_path_suffix: str
    grasp_pose_mode: str
    view_reference_local_point: object
    bottom_surface_coverage_target: float


def get_target_object_config(
    assets_root_path,
):
    """Return the configuration selected by BATTERY_TARGET_OBJECT."""
    if TARGET_OBJECT_TYPE == "mustard":
        return TargetObjectConfig(
            key="mustard",
            description="YCB mustard bottle",
            object_label="mustard_bottle",
            usd_path=get_mustard_bottle_usd_path(assets_root_path),
            prim_path=MUSTARD_BOTTLE_PRIM_PATH,
            position=MUSTARD_BOTTLE_POSITION,
            neutral_rotation=MUSTARD_BOTTLE_NEUTRAL_ROTATION,
            scale=MUSTARD_BOTTLE_SCALE,
            log_prefix="[MUSTARD]",
            arm_capture_poses=tuple(MUSTARD_BOTTLE_ARM_CAPTURE_POSES),
            create_missing_colliders=False,
            object_grid_leaf_size=OBJECT_GRID_LEAF_SIZE,
            object_grid_root_size=OBJECT_GRID_ROOT_SIZE,
            object_grid_origin=OBJECT_GRID_ORIGIN,
            nbv_roi_min=NBV_ROI_MIN,
            nbv_roi_max=NBV_ROI_MAX,
            defect_long_axis_index=1,
            defect_theta_zero_axis_index=2,
            defect_theta_ninety_axis_index=0,
            grasp_prim_path_suffix="",
            grasp_pose_mode=TARGET_OBJECT_POSE_MODE,
            # The YCB bottle's authored origin sits inside its mesh, so the
            # origin already is a fair stand-in for the part.
            view_reference_local_point=None,
            bottom_surface_coverage_target=0.0,
        )

    return TargetObjectConfig(
        key="batterycap",
        description="Battery cap assembly",
        object_label="batterycap",
        usd_path=BATTERYCAP_USD_PATH,
        prim_path=BATTERYCAP_PRIM_PATH,
        position=BATTERYCAP_POSITION,
        neutral_rotation=BATTERYCAP_NEUTRAL_ROTATION,
        scale=BATTERYCAP_SCALE,
        log_prefix="[BATTERYCAP]",
        arm_capture_poses=tuple(BATTERYCAP_ARM_CAPTURE_POSES),
        create_missing_colliders=True,
        object_grid_leaf_size=OBJECT_GRID_LEAF_SIZE,
        object_grid_root_size=BATTERYCAP_OBJECT_GRID_ROOT_SIZE,
        object_grid_origin=BATTERYCAP_OBJECT_GRID_ORIGIN,
        nbv_roi_min=BATTERYCAP_NBV_ROI_MIN,
        nbv_roi_max=BATTERYCAP_NBV_ROI_MAX,
        # The product's long direction is +X; +Z is its broad top face.
        defect_long_axis_index=0,
        defect_theta_zero_axis_index=2,
        defect_theta_ninety_axis_index=1,
        grasp_prim_path_suffix=BATTERYCAP_GRASP_PRIM_SUFFIX,
        # NBV view-diversity is measured from here, not from the authored
        # origin, which lies 0.4668 m below the product.
        view_reference_local_point=BATTERYCAP_OBJECT_GRID_CENTER,
        bottom_surface_coverage_target=(
            BATTERYCAP_BOTTOM_SURFACE_COVERAGE_TARGET
        ),
        grasp_pose_mode=(
            TARGET_OBJECT_POSE_MODE
            if TARGET_OBJECT_POSE_MODE_EXPLICIT
            else "local_bbox_top_center"
        ),
    )
