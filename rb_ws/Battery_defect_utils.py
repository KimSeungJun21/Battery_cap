"""USD authoring for surface defect decals.

Battery_defect_config.py의 카탈로그를 읽어,
Battery_defect_texture_utils.py가 구운 텍스처를,
Battery_defect_placement_utils.py가 푼 표면 좌표계에 얹어
대상 물체의 자식 prim으로 decal을 생성한다.

decal은 alpha cutout(UsdPreviewSurface opacityThreshold)으로 합성한다.
흠집 바깥은 완전히 투명해서 원래 물체 표면이 그대로 보이고, 흠집
픽셀만 이 텍스처로 그려진다. 사각 패치가 통째로 렌더되면 검출기가
흠집이 아니라 '사각형'을 찾아버리므로 cutout이 전제다.

호출 순서 주의:
    Battery_target_utils.add_target_labels()는 대상 prim 아래의 모든
    UsdGeom.Mesh에 class=target_object 라벨을 덮어쓴다. 따라서
    setup_defects()는 반드시 add_target_labels() '뒤에' 불러야
    decal의 class=defect 라벨이 살아남는다.
"""

from __future__ import annotations

import io
import json
import os

import numpy as np
from PIL import Image
from pxr import Ar, Gf, Sdf, Usd, UsdGeom, UsdShade, Vt

from isaacsim.core.utils.semantics import add_labels

from Battery_defect_config import (
    DEFECT_CLASS_LABEL,
    DEFECT_DECAL_OFFSET_M,
    DEFECT_GT_EXPORT_DIR,
    DEFECT_LONG_AXIS_INDEX,
    DEFECT_PRIM_GROUP_NAME,
    DEFECT_SCENARIO_NAME,
    DEFECT_TEXTURE_DIR,
    DEFECT_THETA_NINETY_AXIS_INDEX,
    DEFECT_THETA_ZERO_AXIS_INDEX,
    DEFECT_DEBUG_VISIBLE,
    ENABLE_DEFECTS,
    debug_color_for,
    describe_active_defects,
    debug_visible_spec,
    get_active_defect_specs,
    resolve_seed,
)
from Battery_defect_placement_utils import (
    TriangleMesh,
    build_conformed_decal,
    decal_quad_corners,
    decal_transform_matrix,
    measure_surface_clearance,
    resolve_placement,
)
from Battery_defect_texture_utils import (
    DEFECT_ALPHA_THRESHOLD,
    _upsample_grid,
    compute_patch_geometry,
    generate_defect_maps,
    write_defect_preview,
    write_defect_textures,
)


# decal 격자 분할 수. 평면 quad를 곡면에 붙이면 중심에서 멀어질수록
# 모서리가 표면 위로 뜬다 (머스타드 병 기준 최대 3mm). 그러면 decal이
# 떠 보일 뿐 아니라 point cloud에 찍히는 결함의 3D 위치가 그만큼
# 틀어진다. 격자를 나눠 각 정점을 표면에 다시 투영한다.
DEFECT_DECAL_SUBDIVISIONS = int(
    os.environ.get(
        "BATTERY_DEFECT_DECAL_SUBDIVISIONS",
        "12",
    )
)

# 법선을 부드럽게 만들 때 섞을 이웃 면의 반경(m).
#
# YCB 머스타드 병은 스캔 메시라 표면에 mm 단위 굴곡이 있다. 반경이
# 작으면 중심 법선이 기울어진 면 한 장에서 잡히고, 그 방향으로 decal
# 격자 전체를 투영하는 바람에 가장자리가 수 mm씩 어긋난다. 실측상
# 4mm에서는 병 아래쪽 배치가 최대 2.7mm 떴고, 12mm면 0.6mm 이하로
# 잡힌다. patch 크기(13~20mm)와 같은 규모로 두는 게 맞다.
DEFECT_SMOOTH_NORMAL_RADIUS_M = float(
    os.environ.get(
        "BATTERY_DEFECT_SMOOTH_NORMAL_RADIUS_M",
        "0.012",
    )
)

# 배경을 '인쇄 라벨'로 볼 휘도 표준편차 기준. 민무늬 노란 몸통은
# 0.02 안팎, 글씨가 있는 라벨은 그보다 훨씬 크다. 실측 후 조정한다.
DEFECT_BACKGROUND_CLUTTER_THRESHOLD = float(
    os.environ.get(
        "BATTERY_DEFECT_BACKGROUND_CLUTTER_THRESHOLD",
        "0.06",
    )
)

# decal 정점이 표면에서 이만큼(mm) 넘게 뜨면 배치가 나쁜 것으로 본다.
# 곡률이 급하거나 굴곡이 있는 자리에 큰 patch를 얹으면 평면 격자를
# 한 방향으로 투영하는 방식이 무너진다.
DEFECT_MAX_CLEARANCE_MM = float(
    os.environ.get(
        "BATTERY_DEFECT_MAX_CLEARANCE_MM",
        "0.5",
    )
)


# ============================================================
# 메시 수집
# ============================================================


def _triangulate_polygon(face_vertex_counts, face_vertex_indices):
    """폴리곤을 fan triangulation으로 삼각형 인덱스로 편다."""
    triangles = []
    cursor = 0

    for count in face_vertex_counts:
        count = int(count)

        if count >= 3:
            base = face_vertex_indices[cursor]

            for offset in range(1, count - 1):
                triangles.append(
                    (
                        base,
                        face_vertex_indices[cursor + offset],
                        face_vertex_indices[cursor + offset + 1],
                    )
                )

        cursor += count

    return triangles


def collect_object_frame_mesh(
    stage,
    target_prim_path,
    time_code=None,
    mesh_path_suffix="",
    log_prefix="[DEFECT]",
):
    """대상 prim 아래의 메시를 object frame 삼각형 메시로 모은다.

    mesh_path_suffix가 있으면 그 suffix와 일치하는 Mesh 하나만 모은다.
    batterycap처럼 특정 부품의 특정 바깥면만 골라야 할 때 사용한다.
    """
    if time_code is None:
        time_code = Usd.TimeCode.Default()

    xform_cache = UsdGeom.XformCache(time_code)
    target_prim = stage.GetPrimAtPath(target_prim_path)

    if not target_prim.IsValid():
        raise RuntimeError(
            f"대상 prim을 찾지 못했습니다: {target_prim_path}"
        )

    target_to_world = xform_cache.GetLocalToWorldTransform(
        target_prim
    )
    world_to_target = target_to_world.GetInverse()

    all_vertices = []
    all_triangles = []
    all_uvs = []
    vertex_offset = 0
    mesh_count = 0
    uv_missing = False

    for prim in Usd.PrimRange(target_prim):
        if not prim.IsA(UsdGeom.Mesh):
            continue

        if is_defect_prim(prim):
            continue

        if (
            mesh_path_suffix
            and not str(prim.GetPath()).endswith(mesh_path_suffix)
        ):
            continue

        mesh = UsdGeom.Mesh(prim)
        points = mesh.GetPointsAttr().Get(time_code)
        face_vertex_counts = (
            mesh.GetFaceVertexCountsAttr().Get(time_code)
        )
        face_vertex_indices = (
            mesh.GetFaceVertexIndicesAttr().Get(time_code)
        )

        if (
            not points
            or not face_vertex_counts
            or not face_vertex_indices
        ):
            continue

        mesh_to_world = xform_cache.GetLocalToWorldTransform(prim)
        mesh_to_target = mesh_to_world * world_to_target

        transformed = np.array(
            [
                mesh_to_target.Transform(Gf.Vec3d(point))
                for point in points
            ],
            dtype=np.float64,
        )
        # 아래 표면 색을 샘플링하려면 정점별 UV가 필요하다.
        uvs = _read_vertex_uvs(prim, len(points), time_code)

        if uvs is None:
            uv_missing = True
            uvs = np.full((len(points), 2), np.nan)
        triangles = _triangulate_polygon(
            face_vertex_counts,
            face_vertex_indices,
        )

        if not triangles:
            continue

        all_vertices.append(transformed)
        all_uvs.append(uvs)
        all_triangles.extend(
            [
                (
                    a + vertex_offset,
                    b + vertex_offset,
                    c + vertex_offset,
                )
                for a, b, c in triangles
            ]
        )
        vertex_offset += len(transformed)
        mesh_count += 1

    if not all_vertices:
        suffix_message = (
            f" (mesh suffix={mesh_path_suffix})"
            if mesh_path_suffix
            else ""
        )
        raise RuntimeError(
            f"{target_prim_path} 아래에서 메시를 찾지 못했습니다."
            f"{suffix_message}"
        )

    mesh = TriangleMesh(
        np.concatenate(all_vertices, axis=0),
        all_triangles,
    )
    vertex_uvs = np.concatenate(all_uvs, axis=0)

    if uv_missing:
        print(
            f"{log_prefix}[WARN] 일부 메시에 UV(st)가 없어 아래 표면 "
            "색 샘플링이 제한됩니다."
        )

    bounds_min, bounds_max = mesh.bounds
    print(
        f"{log_prefix} object frame mesh: {mesh_count} meshes, "
        f"{len(mesh.vertices)} verts, {len(mesh)} tris, "
        f"bounds={np.round(bounds_min, 4)} .. "
        f"{np.round(bounds_max, 4)}"
        + (
            f" suffix={mesh_path_suffix}"
            if mesh_path_suffix
            else ""
        )
    )

    return mesh, vertex_uvs


def _read_vertex_uvs(prim, point_count, time_code):
    """메시의 정점별 UV를 (N, 2)로 읽는다. 없으면 None."""
    primvars = UsdGeom.PrimvarsAPI(prim)

    for name in ("st", "st0", "uv", "map1"):
        primvar = primvars.GetPrimvar(name)

        if not primvar or not primvar.HasValue():
            continue

        values = primvar.Get(time_code)

        if values is None:
            continue

        values = np.asarray(values, dtype=np.float64)

        if primvar.IsIndexed():
            indices = np.asarray(
                primvar.GetIndices(time_code),
                dtype=np.int32,
            )

            if len(indices) == point_count:
                return values[indices]

            continue

        if len(values) == point_count:
            return values

    return None


def _srgb_to_linear(srgb):
    srgb = np.clip(srgb, 0.0, 1.0)

    return np.where(
        srgb <= 0.04045,
        srgb / 12.92,
        ((srgb + 0.055) / 1.055) ** 2.4,
    )


def _find_diffuse_texture_path(stage, target_prim_path):
    """대상 물체 머티리얼의 diffuse 텍스처 경로를 찾는다.

    UsdPreviewSurface(diffuseColor 연결)와 OmniPBR MDL
    (diffuse_texture 입력)을 모두 본다.
    """
    target_prim = stage.GetPrimAtPath(target_prim_path)

    for prim in Usd.PrimRange(target_prim):
        if not prim.IsA(UsdGeom.Mesh) or is_defect_prim(prim):
            continue

        binding = UsdShade.MaterialBindingAPI(prim).GetDirectBinding()
        material = binding.GetMaterial()

        if not material:
            continue

        for shader_prim in Usd.PrimRange(material.GetPrim()):
            shader = UsdShade.Shader(shader_prim)

            if not shader:
                continue

            for input_name in (
                "diffuse_texture",
                "diffuseColor",
                "base_color_texture",
                "albedo_texture",
                "file",
            ):
                shader_input = shader.GetInput(input_name)

                if shader_input is None:
                    continue

                value = shader_input.Get()

                if isinstance(value, Sdf.AssetPath):
                    resolved = (
                        value.resolvedPath or value.path
                    )

                    if resolved:
                        return resolved

    return None


def load_surface_albedo_texture(
    stage,
    target_prim_path,
    log_prefix="[DEFECT]",
):
    """물체의 diffuse 텍스처를 선형 RGB 배열로 읽는다.

    못 읽으면 None을 돌려주고, 호출부는 상수 base albedo로 폴백한다.
    """
    path = _find_diffuse_texture_path(stage, target_prim_path)

    if not path:
        print(
            f"{log_prefix}[WARN] 물체의 diffuse 텍스처를 찾지 "
            "못했습니다. decal은 상수 base albedo를 씁니다."
        )

        return None

    try:
        asset = Ar.GetResolver().OpenAsset(Ar.ResolvedPath(path))
        buffer = asset.GetBuffer()
        image = Image.open(io.BytesIO(bytes(buffer[:asset.GetSize()])))
        image = image.convert("RGB")
    except Exception as exc:
        print(
            f"{log_prefix}[WARN] diffuse 텍스처를 열지 못했습니다 "
            f"({type(exc).__name__}: {exc}). 상수 base albedo를 씁니다."
        )

        return None

    array = np.asarray(image, dtype=np.float32) / 255.0
    print(
        f"{log_prefix} surface albedo texture: "
        f"{image.size[0]}x{image.size[1]} <- {os.path.basename(path)}"
    )

    return _srgb_to_linear(array).astype(np.float32)


def sample_base_uv_grid(mesh, vertex_uvs, decal_data):
    """decal 정점마다 아래 표면의 UV를 보간해 격자로 만든다.

    (subdivisions + 1, subdivisions + 1, 2) 배열을 돌려준다.
    UV를 못 구한 정점은 주변 평균으로 메운다.
    """
    if vertex_uvs is None:
        return None

    triangle_indices = decal_data["triangle_indices"]
    barycentrics = decal_data["barycentrics"]
    stride = decal_data["subdivisions"] + 1
    uvs = np.full((len(triangle_indices), 2), np.nan, dtype=np.float64)

    for index, triangle_index in enumerate(triangle_indices):
        if triangle_index < 0:
            continue

        corner_uvs = vertex_uvs[mesh.triangles[triangle_index]]

        if not np.all(np.isfinite(corner_uvs)):
            continue

        uvs[index] = barycentrics[index] @ corner_uvs

    valid = np.all(np.isfinite(uvs), axis=1)

    if not np.any(valid):
        return None

    if not np.all(valid):
        uvs[~valid] = uvs[valid].mean(axis=0)

    return uvs.reshape(stride, stride, 2)


def _sample_texture_bilinear(texture, uv):
    """(H, W, 3) 텍스처를 (..., 2) UV에서 이중선형 샘플링한다."""
    height, width = texture.shape[:2]
    # UsdUVTexture 규약: v=0이 이미지 아래쪽이다.
    x = np.clip(uv[..., 0] * width - 0.5, 0.0, width - 1.0)
    y = np.clip(
        (1.0 - uv[..., 1]) * height - 0.5,
        0.0,
        height - 1.0,
    )
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    fx = (x - x0)[..., None].astype(np.float32)
    fy = (y - y0)[..., None].astype(np.float32)

    top = texture[y0, x0] * (1.0 - fx) + texture[y0, x1] * fx
    bottom = texture[y1, x0] * (1.0 - fx) + texture[y1, x1] * fx

    return top * (1.0 - fy) + bottom * fy


def build_base_albedo_map(texture, uv_grid, resolution):
    """아래 표면 색을 텍스처 해상도로 샘플링한 (res, res, 3) 맵.

    정점 격자(13x13)에서 색을 직접 뽑아 확대하면 인쇄 글씨 같은
    고주파 디테일이 뭉개진다. UV를 먼저 확대한 뒤 텍셀 단위로
    샘플링하면 raycast 비용을 늘리지 않고 원본 선명도를 유지한다.
    """
    if texture is None or uv_grid is None:
        return None

    uv_full = _upsample_grid(uv_grid, resolution)

    return _sample_texture_bilinear(texture, uv_full).astype(np.float32)


# ============================================================
# 머티리얼
# ============================================================


def _create_texture_reader(
    stage,
    material_path,
    name,
    file_path,
    st_reader_output,
    color_space,
    result_channel,
    scale=None,
    bias=None,
):
    shader = UsdShade.Shader.Define(
        stage,
        f"{material_path}/{name}",
    )
    shader.CreateIdAttr("UsdUVTexture")
    shader.CreateInput(
        "file",
        Sdf.ValueTypeNames.Asset,
    ).Set(file_path)
    shader.CreateInput(
        "st",
        Sdf.ValueTypeNames.Float2,
    ).ConnectToSource(st_reader_output)
    shader.CreateInput(
        "sourceColorSpace",
        Sdf.ValueTypeNames.Token,
    ).Set(color_space)
    shader.CreateInput(
        "wrapS",
        Sdf.ValueTypeNames.Token,
    ).Set("clamp")
    shader.CreateInput(
        "wrapT",
        Sdf.ValueTypeNames.Token,
    ).Set("clamp")

    if scale is not None:
        shader.CreateInput(
            "scale",
            Sdf.ValueTypeNames.Float4,
        ).Set(Gf.Vec4f(*scale))

    if bias is not None:
        shader.CreateInput(
            "bias",
            Sdf.ValueTypeNames.Float4,
        ).Set(Gf.Vec4f(*bias))

    if result_channel == "rgb":
        return shader.CreateOutput(
            "rgb",
            Sdf.ValueTypeNames.Float3,
        )

    return shader.CreateOutput(
        result_channel,
        Sdf.ValueTypeNames.Float,
    )


def create_defect_material(stage, material_path, metadata):
    """decal용 UsdPreviewSurface 머티리얼을 만든다."""
    material = UsdShade.Material.Define(stage, material_path)

    st_reader = UsdShade.Shader.Define(
        stage,
        f"{material_path}/stReader",
    )
    st_reader.CreateIdAttr("UsdPrimvarReader_float2")
    st_reader.CreateInput(
        "varname",
        Sdf.ValueTypeNames.Token,
    ).Set("st")
    st_output = st_reader.CreateOutput(
        "result",
        Sdf.ValueTypeNames.Float2,
    )

    surface = UsdShade.Shader.Define(
        stage,
        f"{material_path}/Surface",
    )
    surface.CreateIdAttr("UsdPreviewSurface")
    surface.CreateInput(
        "metallic",
        Sdf.ValueTypeNames.Float,
    ).Set(float(metadata["base_metallic"]))
    surface.CreateInput(
        "useSpecularWorkflow",
        Sdf.ValueTypeNames.Int,
    ).Set(0)

    textures = metadata["textures"]

    surface.CreateInput(
        "diffuseColor",
        Sdf.ValueTypeNames.Color3f,
    ).ConnectToSource(
        _create_texture_reader(
            stage,
            material_path,
            "AlbedoTexture",
            textures["albedo"],
            st_output,
            "sRGB",
            "rgb",
        )
    )
    surface.CreateInput(
        "roughness",
        Sdf.ValueTypeNames.Float,
    ).ConnectToSource(
        _create_texture_reader(
            stage,
            material_path,
            "RoughnessTexture",
            textures["roughness"],
            st_output,
            "raw",
            "r",
        )
    )
    # normal map은 [0,1] 저장값을 [-1,1] 접선공간 벡터로 되돌려야 한다.
    surface.CreateInput(
        "normal",
        Sdf.ValueTypeNames.Normal3f,
    ).ConnectToSource(
        _create_texture_reader(
            stage,
            material_path,
            "NormalTexture",
            textures["normal"],
            st_output,
            "raw",
            "rgb",
            scale=(2.0, 2.0, 2.0, 1.0),
            bias=(-1.0, -1.0, -1.0, 0.0),
        )
    )
    surface.CreateInput(
        "opacity",
        Sdf.ValueTypeNames.Float,
    ).ConnectToSource(
        _create_texture_reader(
            stage,
            material_path,
            "OpacityTexture",
            textures["opacity"],
            st_output,
            "raw",
            "r",
        )
    )
    if "emissive" in textures:
        # 디버그 가시화: 조명과 무관하게 흠집이 빛나게 한다.
        surface.CreateInput(
            "emissiveColor",
            Sdf.ValueTypeNames.Color3f,
        ).ConnectToSource(
            _create_texture_reader(
                stage,
                material_path,
                "EmissiveTexture",
                textures["emissive"],
                st_output,
                "sRGB",
                "rgb",
            )
        )

    # cutout. blend로 두면 투명한 패치 영역까지 렌더/세그멘테이션에
    # 잡혀 사각형이 드러난다.
    surface.CreateInput(
        "opacityThreshold",
        Sdf.ValueTypeNames.Float,
    ).Set(float(metadata["alpha_threshold"]))

    material.CreateSurfaceOutput().ConnectToSource(
        surface.CreateOutput(
            "surface",
            Sdf.ValueTypeNames.Token,
        )
    )

    return material


# ============================================================
# Decal prim
# ============================================================


def _define_decal_mesh(stage, prim_path, decal_data):
    """표면에 정합된 decal 격자를 object frame 좌표로 저작한다.

    정점이 이미 object frame이므로 prim 변환은 identity다.
    """
    points = decal_data["points"]
    normals = decal_data["normals"]
    st_values = decal_data["st"]

    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    mesh.CreatePointsAttr(
        Vt.Vec3fArray(
            [Gf.Vec3f(*point) for point in points.tolist()]
        )
    )
    mesh.CreateFaceVertexCountsAttr(
        Vt.IntArray(
            decal_data["face_vertex_counts"].tolist()
        )
    )
    mesh.CreateFaceVertexIndicesAttr(
        Vt.IntArray(
            decal_data["face_vertex_indices"].tolist()
        )
    )
    mesh.CreateNormalsAttr(
        Vt.Vec3fArray(
            [Gf.Vec3f(*normal) for normal in normals.tolist()]
        )
    )
    mesh.SetNormalsInterpolation("vertex")
    mesh.CreateSubdivisionSchemeAttr().Set("none")
    mesh.CreateDoubleSidedAttr().Set(False)
    mesh.CreateExtentAttr(
        Vt.Vec3fArray(
            [
                Gf.Vec3f(*points.min(axis=0).tolist()),
                Gf.Vec3f(*points.max(axis=0).tolist()),
            ]
        )
    )

    primvars = UsdGeom.PrimvarsAPI(mesh.GetPrim())
    st_primvar = primvars.CreatePrimvar(
        "st",
        Sdf.ValueTypeNames.TexCoord2fArray,
        UsdGeom.Tokens.vertex,
    )
    # u는 tangent, v는 bitangent 방향. 텍스처는 이미 USD의 v 규약에
    # 맞춰 상하 반전해 저장되어 있다.
    st_primvar.Set(
        Vt.Vec2fArray(
            [Gf.Vec2f(*value) for value in st_values.tolist()]
        )
    )

    xformable = UsdGeom.Xformable(mesh.GetPrim())
    xformable.ClearXformOpOrder()

    return mesh


def is_defect_prim(prim):
    """decal 그룹 아래의 prim인가.

    Battery_target_utils.add_target_labels()처럼 대상 prim 아래를
    훑으며 라벨을 덮어쓰는 코드가 decal을 건너뛰게 할 때 쓴다.
    """
    return f"/{DEFECT_PRIM_GROUP_NAME}/" in f"{prim.GetPath()}/"


def remove_defects(stage, target_prim_path):
    """기존 decal 그룹을 지운다 (재실행용)."""
    group_path = f"{target_prim_path}/{DEFECT_PRIM_GROUP_NAME}"

    if stage.GetPrimAtPath(group_path).IsValid():
        stage.RemovePrim(group_path)

    return group_path


# ============================================================
# 상위 진입점
# ============================================================


def setup_defects(
    stage,
    target_prim_path,
    simulation_app=None,
    specs=None,
    write_previews=False,
    export_gt=True,
    target_object_type=None,
    long_axis_index=DEFECT_LONG_AXIS_INDEX,
    theta_zero_axis_index=DEFECT_THETA_ZERO_AXIS_INDEX,
    theta_ninety_axis_index=DEFECT_THETA_NINETY_AXIS_INDEX,
    log_prefix="[DEFECT]",
):
    """대상 물체 표면에 결함 decal들을 생성하고 GT를 내보낸다.

    Battery_target_utils.add_target_labels() '뒤에' 호출해야 한다.
    그 함수가 대상 prim 아래 모든 메시에 class=target_object를
    덮어쓰기 때문이다.

    생성된 결함들의 GT 레코드 리스트를 돌려준다.
    """
    if not ENABLE_DEFECTS:
        print(
            f"{log_prefix} disabled (BATTERY_ENABLE_DEFECTS=0)"
        )

        return []

    if specs is None:
        specs = get_active_defect_specs(
            target_object_type=target_object_type
        )

    # 어떤 모드로 도는지 로그 앞에 분명히 남긴다. 디버그 가시화를
    # 켰는지 안 켰는지가 '결함이 안 보인다'의 대부분의 원인이다.
    print("=" * 62)
    print(
        describe_active_defects(
            target_object_type=target_object_type,
            specs=specs,
        )
    )

    if DEFECT_DEBUG_VISIBLE:
        print(
            "[DEFECT] 형광 디버그 마커가 켜져 있습니다. "
            "검출 실험 전에 반드시 끄세요."
        )
    else:
        print(
            "[DEFECT] 실제 크기/재질 모드입니다 (폭 0.5~0.7mm). "
            "눈으로 확인하려면 "
            "BATTERY_DEFECT_DEBUG_VISIBLE=1 로 실행하세요."
        )

    print("=" * 62)

    if not specs:
        print(
            f"{log_prefix} scenario '{DEFECT_SCENARIO_NAME}'에 "
            "결함이 없습니다."
        )

        return []

    object_mesh, vertex_uvs = collect_object_frame_mesh(
        stage,
        target_prim_path,
        log_prefix=log_prefix,
    )
    surface_texture = load_surface_albedo_texture(
        stage,
        target_prim_path,
        log_prefix=log_prefix,
    )
    os.makedirs(DEFECT_TEXTURE_DIR, exist_ok=True)

    group_path = remove_defects(stage, target_prim_path)
    UsdGeom.Scope.Define(stage, group_path)
    material_root = f"{group_path}/Looks"
    UsdGeom.Scope.Define(stage, material_root)

    records = []
    placement_mesh_cache = {
        "": (object_mesh, vertex_uvs),
    }

    for spec in specs:
        # 순서가 중요하다: 패치 크기를 먼저 정해야 decal을 표면에
        # 정합시킬 수 있고, 정합이 끝나야 각 정점 아래의 표면 색을
        # 샘플링할 수 있으며, 그 색이 있어야 텍스처를 구울 수 있다.
        render_spec = debug_visible_spec(spec)
        patch_mm, patch_resolution, _ = compute_patch_geometry(
            render_spec
        )

        mesh_suffix = spec.placement.target_mesh_path_suffix

        if mesh_suffix not in placement_mesh_cache:
            placement_mesh_cache[mesh_suffix] = collect_object_frame_mesh(
                stage,
                target_prim_path,
                mesh_path_suffix=mesh_suffix,
                log_prefix=log_prefix,
            )

        placement_mesh, placement_vertex_uvs = (
            placement_mesh_cache[mesh_suffix]
        )
        placement_long_axis_index = (
            spec.placement.long_axis_index_override
            if spec.placement.long_axis_index_override is not None
            else long_axis_index
        )
        placement_theta_zero_axis_index = (
            spec.placement.theta_zero_axis_index_override
            if spec.placement.theta_zero_axis_index_override is not None
            else theta_zero_axis_index
        )
        placement_theta_ninety_axis_index = (
            spec.placement.theta_ninety_axis_index_override
            if spec.placement.theta_ninety_axis_index_override is not None
            else theta_ninety_axis_index
        )

        rng = np.random.default_rng(resolve_seed(spec))
        resolved = resolve_placement(
            placement_mesh,
            spec.placement,
            placement_long_axis_index,
            placement_theta_zero_axis_index,
            placement_theta_ninety_axis_index,
            DEFECT_DECAL_OFFSET_M,
            smooth_normal_radius=DEFECT_SMOOTH_NORMAL_RADIUS_M,
            rng=rng,
        )
        transform = decal_transform_matrix(resolved)
        corners = decal_quad_corners(resolved, patch_mm)

        decal_data = build_conformed_decal(
            placement_mesh,
            resolved,
            patch_mm,
            DEFECT_DECAL_OFFSET_M,
            subdivisions=DEFECT_DECAL_SUBDIVISIONS,
            smooth_normal_radius=DEFECT_SMOOTH_NORMAL_RADIUS_M,
        )
        clearance = measure_surface_clearance(
            placement_mesh,
            decal_data["points"],
            resolved["normal"],
        )

        base_albedo_grid = build_base_albedo_map(
            surface_texture,
            sample_base_uv_grid(
                placement_mesh,
                placement_vertex_uvs,
                decal_data,
            ),
            patch_resolution,
        )
        # 배경 복잡도. 인쇄 라벨(촘촘한 글씨) 위의 얇은 스크래치는
        # 민무늬 몸통 위의 같은 스크래치보다 훨씬 어렵다. 샘플링한
        # base albedo의 휘도 표준편차로 자동 분류해 GT에 남긴다.
        if base_albedo_grid is None:
            background_clutter = None
            background = "unknown"
        else:
            luminance = (
                base_albedo_grid
                @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
            )
            background_clutter = float(np.std(luminance))
            background = (
                "printed"
                if background_clutter
                >= DEFECT_BACKGROUND_CLUTTER_THRESHOLD
                else "plain"
            )

        maps = generate_defect_maps(
            render_spec,
            base_albedo_grid=base_albedo_grid,
            debug_color=debug_color_for(spec),
        )
        metadata = write_defect_textures(maps, DEFECT_TEXTURE_DIR)

        if write_previews:
            metadata["preview_path"] = write_defect_preview(
                maps,
                DEFECT_TEXTURE_DIR,
            )

        decal_path = f"{group_path}/{spec.name}"
        mesh = _define_decal_mesh(
            stage,
            decal_path,
            decal_data,
        )
        material = create_defect_material(
            stage,
            f"{material_root}/{spec.name}",
            metadata,
        )
        UsdShade.MaterialBindingAPI(
            mesh.GetPrim()
        ).Bind(material)

        add_labels(
            mesh.GetPrim(),
            labels=[DEFECT_CLASS_LABEL],
            instance_name="class",
        )
        add_labels(
            mesh.GetPrim(),
            labels=[spec.name],
            instance_name="object",
        )

        record = {
            "name": spec.name,
            "defect_type": spec.defect_type,
            "difficulty": spec.difficulty,
            "visible_at_rest": spec.visible_at_rest,
            "notes": spec.notes,
            "prim_path": decal_path,
            "seed": resolve_seed(spec),
            "placement": {
                "theta_deg": spec.placement.theta_deg,
                "axial_ratio": spec.placement.axial_ratio,
                "tangent_deg": spec.placement.tangent_deg,
                "jitter_mm": spec.placement.jitter_mm,
                "target_mesh_path_suffix": mesh_suffix,
                "long_axis_index": placement_long_axis_index,
                "theta_zero_axis_index": (
                    placement_theta_zero_axis_index
                ),
                "theta_ninety_axis_index": (
                    placement_theta_ninety_axis_index
                ),
            },
            "object_frame": {
                "surface_point": resolved[
                    "surface_point"
                ].tolist(),
                "decal_origin": resolved[
                    "decal_origin"
                ].tolist(),
                "normal": resolved["normal"].tolist(),
                "tangent": resolved["tangent"].tolist(),
                "bitangent": resolved["bitangent"].tolist(),
                "quad_corners": corners.tolist(),
                "transform": transform.tolist(),
            },
            "conformance": {
                "subdivisions": DEFECT_DECAL_SUBDIVISIONS,
                "miss_count": decal_data["miss_count"],
                "max_conform_shift_mm": (
                    decal_data["max_conform_shift_m"] * 1e3
                ),
                "clearance_min_mm": float(
                    np.nanmin(clearance) * 1e3
                ),
                "clearance_max_mm": float(
                    np.nanmax(clearance) * 1e3
                ),
            },
            "size": {
                "length_mm": spec.length_mm,
                "width_mm": spec.width_mm,
                "depth_um": spec.depth_um,
                "patch_mm": metadata["patch_mm"],
                "footprint_area_mm2": metadata[
                    "footprint_area_mm2"
                ],
            },
            "appearance": {
                "albedo_scale": spec.albedo_scale,
                "roughness_delta": spec.roughness_delta,
                "base_albedo_source": metadata[
                    "base_albedo_source"
                ],
                "background": background,
                "background_clutter": background_clutter,
                "debug_visible": metadata["debug_visible"],
                "base_roughness": metadata["base_roughness"],
                "max_normal_tilt_deg": metadata[
                    "max_normal_tilt_deg"
                ],
                "p95_normal_tilt_deg": metadata[
                    "p95_normal_tilt_deg"
                ],
            },
            "textures": metadata["textures"],
        }
        records.append(record)

        print(
            f"{log_prefix} {spec.name:<22} -> "
            f"p={np.round(resolved['surface_point'], 4)} "
            f"n={np.round(resolved['normal'], 3)} "
            f"patch={metadata['patch_mm']:.1f}mm "
            f"bg={background}"
            f"({background_clutter:.3f}) "
            if background_clutter is not None
            else "bg=unknown "
            f"clearance={np.nanmin(clearance) * 1e3:.2f}~"
            f"{np.nanmax(clearance) * 1e3:.2f}mm "
            f"miss={decal_data['miss_count']}"
        )

        if decal_data["miss_count"] > 0:
            print(
                f"{log_prefix}[WARN] {spec.name}: decal 정점 "
                f"{decal_data['miss_count']}개가 표면을 벗어났습니다 "
                "(실루엣 근처). patch가 너무 크거나 곡률이 급한 "
                "위치입니다."
            )

        floating_count = int(
            np.count_nonzero(
                clearance * 1e3 > DEFECT_MAX_CLEARANCE_MM
            )
        )

        if floating_count > 0:
            print(
                f"{log_prefix}[WARN] {spec.name}: decal 정점 "
                f"{floating_count}/{len(clearance)}개가 표면에서 "
                f"{DEFECT_MAX_CLEARANCE_MM}mm 넘게 떠 있습니다 "
                f"(최대 {np.nanmax(clearance) * 1e3:.2f}mm). "
                "굴곡이 심한 자리입니다. axial_ratio를 조금 옮기거나 "
                "결함 크기를 줄이세요."
            )

    if simulation_app is not None:
        for _ in range(3):
            simulation_app.update()

    if export_gt:
        export_defect_gt(
            records,
            target_object_type=target_object_type,
            long_axis_index=long_axis_index,
            theta_zero_axis_index=theta_zero_axis_index,
            theta_ninety_axis_index=theta_ninety_axis_index,
            log_prefix=log_prefix,
        )

    hidden_count = sum(
        1 for record in records if not record["visible_at_rest"]
    )
    print(
        f"{log_prefix} {len(records)}개 생성 완료 "
        f"(초기 자세 미관측 {hidden_count}개)"
    )

    return records


def export_defect_gt(
    records,
    out_dir=None,
    scenario_name=None,
    target_object_type=None,
    long_axis_index=DEFECT_LONG_AXIS_INDEX,
    theta_zero_axis_index=DEFECT_THETA_ZERO_AXIS_INDEX,
    theta_ninety_axis_index=DEFECT_THETA_NINETY_AXIS_INDEX,
    log_prefix="[DEFECT]",
):
    """해석된 결함 GT를 JSON으로 내보낸다.

    검출 성능 평가는 여기 적힌 object frame 좌표를 카메라로 투영해
    "이 뷰에서 이 결함이 보였어야 하는가"를 판정한다.
    """
    if out_dir is None:
        out_dir = DEFECT_GT_EXPORT_DIR

    if scenario_name is None:
        scenario_name = DEFECT_SCENARIO_NAME

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(
        out_dir,
        f"defect_gt_{scenario_name}.json",
    )
    payload = {
        "scenario": scenario_name,
        "target_object_type": target_object_type,
        "long_axis_index": long_axis_index,
        "theta_zero_axis_index": theta_zero_axis_index,
        "theta_ninety_axis_index": theta_ninety_axis_index,
        "decal_offset_m": DEFECT_DECAL_OFFSET_M,
        "alpha_threshold": DEFECT_ALPHA_THRESHOLD,
        "class_label": DEFECT_CLASS_LABEL,
        "defects": records,
    }

    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    print(f"{log_prefix} GT 저장: {path}")

    return path
