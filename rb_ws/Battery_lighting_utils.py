"""Zivid 프로젝터를 모사한 조명 리그.

실제 Zivid는 구조광 프로젝터가 카메라와 baseline만큼 떨어져 있다.
흠집이 보이느냐는 결국 '프로젝터 -> 표면 -> 카메라'의 half-vector가
그 지점의 법선과 얼마나 정렬되는가로 결정된다. roughness만 다른
결함(scuff)은 정반사 근처에서만 대비가 생기므로, 조명을 통제하지
않으면 NBV가 무엇을 최적화하는지 정의되지 않는다.

스테이지 USD의 기존 조명은 전방향이라 이 구조를 씻어낸다. 그래서
프로젝터를 주 광원으로 세우고 기존 조명은 감쇠시킨다.
"""

from __future__ import annotations

from pxr import Gf, Sdf, UsdGeom, UsdLux


def _iter_light_prims(stage):
    for prim in stage.Traverse():
        if prim.HasAPI(UsdLux.LightAPI):
            yield prim


def dim_stage_lights(
    stage,
    dim_factor,
    exclude_prefixes=(),
    log_prefix="[LIGHT]",
):
    """기존 스테이지 조명의 intensity를 배율로 줄인다.

    원래 값을 돌려주므로 필요하면 복원할 수 있다.
    """
    original = {}

    if dim_factor >= 1.0:
        print(
            f"{log_prefix} 기존 스테이지 조명을 그대로 둡니다 "
            f"(dim_factor={dim_factor}). 뷰포트는 밝지만 전방향 조명이 "
            "정반사 구조를 씻어내므로 검출 실험에는 적합하지 않습니다."
        )

        return original

    for prim in _iter_light_prims(stage):
        path = str(prim.GetPath())

        if any(path.startswith(p) for p in exclude_prefixes):
            continue

        light = UsdLux.LightAPI(prim)
        attr = light.GetIntensityAttr()
        value = attr.Get()

        if value is None:
            continue

        original[path] = float(value)
        attr.Set(float(value) * dim_factor)
        print(
            f"{log_prefix} dim {path}: "
            f"{value:.1f} -> {float(value) * dim_factor:.1f}"
        )

    if not original:
        print(f"{log_prefix} 감쇠할 기존 조명이 없습니다.")

    return original


def restore_stage_lights(stage, original, log_prefix="[LIGHT]"):
    for path, value in original.items():
        prim = stage.GetPrimAtPath(path)

        if not prim.IsValid():
            continue

        UsdLux.LightAPI(prim).GetIntensityAttr().Set(value)

    print(f"{log_prefix} 기존 조명 {len(original)}개 복원")


def setup_zivid_projector(
    stage,
    camera_prim_path,
    prim_name,
    baseline_m,
    width_m,
    height_m,
    intensity,
    color_temperature,
    log_prefix="[LIGHT]",
):
    """카메라 기준 baseline만큼 옆에 둔 면광원을 만든다.

    카메라 prim의 자식으로 붙이므로 카메라 자세를 그대로 따라간다.
    USD 관례상 카메라와 RectLight 모두 로컬 -Z가 전방이라, 회전 없이
    평행이동만 주면 프로젝터가 카메라와 같은 방향을 비춘다.
    """
    camera_prim = stage.GetPrimAtPath(camera_prim_path)

    if not camera_prim.IsValid():
        raise RuntimeError(
            f"Zivid 카메라 prim을 찾지 못했습니다: {camera_prim_path}"
        )

    # 라이트를 카메라 prim의 '자식'으로 붙이면 Hydra가 렌더에 반영하지
    # 않는다(세기를 4배 올려도 휘도가 그대로였다). 카메라의 부모 아래에
    # 두고, 카메라의 로컬 변환에 baseline 평행이동을 합성해서 카메라와
    # 같은 자세를 갖게 한다.
    parent_prim = camera_prim.GetParent()

    if not parent_prim.IsValid():
        raise RuntimeError(
            f"카메라의 부모 prim이 없습니다: {camera_prim_path}"
        )

    camera_local = UsdGeom.Xformable(
        camera_prim
    ).GetLocalTransformation()
    offset = Gf.Matrix4d().SetTranslate(
        Gf.Vec3d(float(baseline_m), 0.0, 0.0)
    )
    # USD는 행벡터 규약이라 '먼저 offset, 그다음 camera_local'이다.
    light_local = offset * camera_local

    light_path = f"{parent_prim.GetPath()}/{prim_name}"
    light = UsdLux.RectLight.Define(stage, light_path)

    light.CreateWidthAttr(float(width_m))
    light.CreateHeightAttr(float(height_m))
    light.CreateIntensityAttr(float(intensity))
    light.CreateColorTemperatureAttr(float(color_temperature))
    light.CreateEnableColorTemperatureAttr(True)
    # normalize=True는 광원 면적으로 세기를 나눈다. 30x20mm처럼 작은
    # 면광원에서는 값이 직관과 크게 어긋나므로 끈다.
    light.CreateNormalizeAttr(False)

    prim = light.GetPrim()
    prim.CreateAttribute(
        "visibleInPrimaryRay",
        Sdf.ValueTypeNames.Bool,
    ).Set(False)

    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()
    xformable.AddTransformOp().Set(light_local)

    print(
        f"{log_prefix} Zivid projector: {light_path} "
        f"baseline={baseline_m * 1000:.0f}mm "
        f"size={width_m * 1000:.0f}x{height_m * 1000:.0f}mm "
        f"intensity={intensity:.0f} "
        f"{color_temperature:.0f}K"
    )

    return light_path


def setup_ambient_dome(
    stage,
    intensity,
    prim_path="/World/DefectAmbientDome",
    log_prefix="[LIGHT]",
):
    """완전한 암부를 피하기 위한 약한 환경광."""
    if intensity <= 0.0:
        print(f"{log_prefix} 환경광 생략 (intensity=0)")

        return None

    dome = UsdLux.DomeLight.Define(stage, prim_path)
    dome.CreateIntensityAttr(float(intensity))
    dome.GetPrim().CreateAttribute(
        "visibleInPrimaryRay",
        Sdf.ValueTypeNames.Bool,
    ).Set(False)
    print(f"{log_prefix} ambient dome: {prim_path} {intensity:.0f}")

    return prim_path


def setup_inspection_lighting(
    stage,
    camera_prim_path,
    projector_enable,
    prim_name,
    baseline_m,
    width_m,
    height_m,
    intensity,
    color_temperature,
    ambient_intensity,
    dim_factor,
    log_prefix="[LIGHT]",
):
    """검사용 조명 리그 전체를 구성한다."""
    if not projector_enable:
        print(
            f"{log_prefix} 프로젝터 비활성 "
            "(BATTERY_ZIVID_PROJECTOR_ENABLE=0) - "
            "스테이지 조명을 그대로 씁니다."
        )

        return {
            "projector_path": None,
            "dome_path": None,
            "original_intensities": {},
        }

    projector_path = setup_zivid_projector(
        stage=stage,
        camera_prim_path=camera_prim_path,
        prim_name=prim_name,
        baseline_m=baseline_m,
        width_m=width_m,
        height_m=height_m,
        intensity=intensity,
        color_temperature=color_temperature,
        log_prefix=log_prefix,
    )
    dome_path = setup_ambient_dome(
        stage=stage,
        intensity=ambient_intensity,
        log_prefix=log_prefix,
    )
    original = dim_stage_lights(
        stage=stage,
        dim_factor=dim_factor,
        exclude_prefixes=tuple(
            p for p in (projector_path, dome_path) if p
        ),
        log_prefix=log_prefix,
    )

    return {
        "projector_path": projector_path,
        "dome_path": dome_path,
        "original_intensities": original,
    }
