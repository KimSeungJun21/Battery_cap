"""Object-frame surface placement math for defect decals.

Battery_defect_config.py의 정규화 배치(theta / axial_ratio / tangent_deg)를
실제 표면 점, 법선, 접평면 기저, decal quad의 4x4 변환으로 바꾼다.

numpy만 쓴다. USD/Isaac에 의존하지 않으므로 합성 메시로 단독 검증이
가능하다 (python3 Battery_defect_placement_utils.py).

배치 절차:
    1. bbox 밖에서 장축을 향해 안쪽으로 ray를 쏜다 (theta가 방위각,
       axial_ratio가 장축 방향 높이를 정한다).
    2. 메시와 교차시켜 가장 가까운 hit을 찾는다 -> 바깥쪽 표면.
    3. hit 주변 면들의 법선을 면적 가중 평균해 부드러운 법선을 얻는다.
    4. 장축을 접평면에 투영해 tangent를 잡고 tangent_deg만큼 돌린다.

theta는 "대충 어느 면인가"를 고르는 노브일 뿐이다. 단면이 원기둥이
아니어도(머스타드 병은 납작한 둥근 사각형) 정확한 위치와 법선은
메시가 결정하므로 decal이 뜨거나 파묻히지 않는다.
"""

from __future__ import annotations

import math

import numpy as np


class TriangleMesh:
    """Object frame 삼각형 메시."""

    def __init__(self, vertices, triangles):
        self.vertices = np.asarray(vertices, dtype=np.float64)
        self.triangles = np.asarray(triangles, dtype=np.int32)

        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3:
            raise ValueError(
                f"vertices shape가 (N, 3)이 아닙니다: "
                f"{self.vertices.shape}"
            )

        if self.triangles.ndim != 2 or self.triangles.shape[1] != 3:
            raise ValueError(
                f"triangles shape가 (M, 3)이 아닙니다: "
                f"{self.triangles.shape}"
            )

        if len(self.triangles) == 0:
            raise ValueError("삼각형이 하나도 없습니다.")

        self._v0 = self.vertices[self.triangles[:, 0]]
        self._edge1 = self.vertices[self.triangles[:, 1]] - self._v0
        self._edge2 = self.vertices[self.triangles[:, 2]] - self._v0
        face_cross = np.cross(self._edge1, self._edge2)
        face_area2 = np.linalg.norm(face_cross, axis=1)
        self.face_areas = 0.5 * face_area2
        self.face_normals = face_cross / np.maximum(
            face_area2[:, None],
            1e-16,
        )
        self.face_centroids = (
            self._v0
            + (self._edge1 + self._edge2) / 3.0
        )

    @property
    def bounds(self):
        return (
            self.vertices.min(axis=0),
            self.vertices.max(axis=0),
        )

    def __len__(self):
        return len(self.triangles)


def ray_mesh_intersect(mesh, origin, direction, epsilon=1e-12):
    """가장 가까운 ray-mesh 교차점을 찾는다 (Moller-Trumbore).

    (t, triangle_index, hit_point, barycentric)을 돌려주고, 교차가
    없으면 None. barycentric은 삼각형 정점 (0, 1, 2) 순서다. 아래
    표면의 UV를 보간해 원래 색을 샘플링할 때 쓴다.
    """
    origin = np.asarray(origin, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)
    direction = direction / np.linalg.norm(direction)

    pvec = np.cross(direction, mesh._edge2)
    determinant = np.einsum("ij,ij->i", mesh._edge1, pvec)
    parallel = np.abs(determinant) < epsilon
    safe_determinant = np.where(parallel, 1.0, determinant)
    inverse_determinant = 1.0 / safe_determinant

    tvec = origin[None, :] - mesh._v0
    u = np.einsum("ij,ij->i", tvec, pvec) * inverse_determinant

    qvec = np.cross(tvec, mesh._edge1)
    v = np.einsum("j,ij->i", direction, qvec) * inverse_determinant

    t = np.einsum("ij,ij->i", mesh._edge2, qvec) * inverse_determinant

    valid = (
        ~parallel
        & (u >= -1e-9)
        & (v >= -1e-9)
        & (u + v <= 1.0 + 1e-9)
        & (t > 1e-9)
    )

    if not np.any(valid):
        return None

    candidate_t = np.where(valid, t, np.inf)
    triangle_index = int(np.argmin(candidate_t))
    hit_t = float(candidate_t[triangle_index])
    hit_point = origin + hit_t * direction
    bary_u = float(u[triangle_index])
    bary_v = float(v[triangle_index])
    barycentric = np.array(
        [1.0 - bary_u - bary_v, bary_u, bary_v],
        dtype=np.float64,
    )

    return hit_t, triangle_index, hit_point, barycentric


def smooth_normal_at(mesh, hit_point, triangle_index, radius):
    """hit 주변 면들의 면적 가중 평균 법선.

    면 법선을 그대로 쓰면 저해상도 메시에서 decal이 눈에 띄게
    각져 보인다. 반경 안의 이웃 면을 섞어 부드럽게 만든다.
    """
    face_normal = mesh.face_normals[triangle_index]

    if radius <= 0.0:
        return face_normal

    distances = np.linalg.norm(
        mesh.face_centroids - np.asarray(hit_point, dtype=np.float64),
        axis=1,
    )
    neighborhood = distances <= radius

    if not np.any(neighborhood):
        return face_normal

    # 반대쪽 면(뒷면)이 섞이면 법선이 상쇄된다. 같은 방향만 남긴다.
    aligned = (
        mesh.face_normals @ face_normal
    ) > 0.3
    selected = neighborhood & aligned

    if not np.any(selected):
        return face_normal

    weights = mesh.face_areas[selected]
    normal = (
        mesh.face_normals[selected] * weights[:, None]
    ).sum(axis=0)
    norm = np.linalg.norm(normal)

    if norm < 1e-12:
        return face_normal

    return normal / norm


def _axis_vector(axis_index):
    vector = np.zeros(3, dtype=np.float64)
    vector[axis_index] = 1.0

    return vector


def build_placement_ray(
    placement,
    bounds_min,
    bounds_max,
    long_axis_index,
    theta_zero_axis_index,
    theta_ninety_axis_index,
    jitter_offset=(0.0, 0.0),
):
    """정규화 배치를 (ray origin, ray direction)으로 바꾼다."""
    bounds_min = np.asarray(bounds_min, dtype=np.float64)
    bounds_max = np.asarray(bounds_max, dtype=np.float64)
    center = 0.5 * (bounds_min + bounds_max)
    extent = bounds_max - bounds_min

    long_axis = _axis_vector(long_axis_index)
    zero_axis = _axis_vector(theta_zero_axis_index)
    ninety_axis = _axis_vector(theta_ninety_axis_index)

    theta = math.radians(placement.theta_deg)
    outward = (
        math.cos(theta) * zero_axis
        + math.sin(theta) * ninety_axis
    )
    outward = outward / np.linalg.norm(outward)

    axial_position = (
        bounds_min[long_axis_index]
        + placement.axial_ratio * extent[long_axis_index]
    )

    # bbox를 확실히 벗어나는 거리에서 출발한다.
    standoff = float(np.linalg.norm(extent)) + 1.0

    origin = center.copy()
    origin[long_axis_index] = axial_position
    origin = origin + outward * standoff

    # jitter는 ray를 접평면 방향으로 평행이동시킨다.
    lateral = np.cross(long_axis, outward)
    lateral_norm = np.linalg.norm(lateral)

    if lateral_norm > 1e-9:
        lateral = lateral / lateral_norm
        origin = (
            origin
            + lateral * jitter_offset[0]
            + long_axis * jitter_offset[1]
        )

    return origin, -outward


def build_tangent_basis(normal, tangent_deg, long_axis_index):
    """법선에 수직인 (tangent, bitangent)를 만든다.

    tangent_deg=0이면 tangent가 물체 장축에 평행하다. 법선이 장축과
    거의 나란한 위/아래 캡에서는 장축 투영이 0이 되므로 다른 축으로
    폴백한다.
    """
    normal = np.asarray(normal, dtype=np.float64)
    normal = normal / np.linalg.norm(normal)
    reference = _axis_vector(long_axis_index)

    tangent = reference - np.dot(reference, normal) * normal

    if np.linalg.norm(tangent) < 1e-6:
        fallback = _axis_vector((long_axis_index + 1) % 3)
        tangent = fallback - np.dot(fallback, normal) * normal

    tangent = tangent / np.linalg.norm(tangent)
    bitangent = np.cross(normal, tangent)
    bitangent = bitangent / np.linalg.norm(bitangent)

    angle = math.radians(tangent_deg)
    rotated_tangent = (
        math.cos(angle) * tangent
        + math.sin(angle) * bitangent
    )
    rotated_bitangent = np.cross(normal, rotated_tangent)

    return (
        rotated_tangent / np.linalg.norm(rotated_tangent),
        rotated_bitangent / np.linalg.norm(rotated_bitangent),
    )


class _AxialOverride:
    """axial_ratio만 바꾼 배치 뷰 (원본 spec은 frozen이다)."""

    __slots__ = (
        "theta_deg",
        "axial_ratio",
        "tangent_deg",
        "jitter_mm",
    )

    def __init__(self, placement, axial_ratio):
        self.theta_deg = placement.theta_deg
        self.axial_ratio = axial_ratio
        self.tangent_deg = placement.tangent_deg
        self.jitter_mm = placement.jitter_mm


def _placement_with_axial_ratio(placement, axial_ratio):
    return _AxialOverride(placement, axial_ratio)


def resolve_placement(
    mesh,
    placement,
    long_axis_index,
    theta_zero_axis_index,
    theta_ninety_axis_index,
    decal_offset_m,
    smooth_normal_radius=0.004,
    rng=None,
):
    """정규화 배치를 실제 표면 좌표계로 해석한다.

    decal quad의 원점(법선 방향으로 offset만큼 띄운 점)과 정규직교
    기저를 담은 dict를 돌려준다.
    """
    bounds_min, bounds_max = mesh.bounds

    jitter_offset = (0.0, 0.0)

    if placement.jitter_mm > 0.0 and rng is not None:
        radius_m = placement.jitter_mm * 1e-3
        angle = rng.uniform(0.0, 2.0 * math.pi)
        magnitude = radius_m * math.sqrt(rng.uniform(0.0, 1.0))
        jitter_offset = (
            magnitude * math.cos(angle),
            magnitude * math.sin(angle),
        )

    origin, direction = build_placement_ray(
        placement,
        bounds_min,
        bounds_max,
        long_axis_index,
        theta_zero_axis_index,
        theta_ninety_axis_index,
        jitter_offset=jitter_offset,
    )
    hit = ray_mesh_intersect(mesh, origin, direction)

    # axial_ratio가 0이나 1에 가까우면 ray가 bbox 면의 정확한 경계를
    # 스쳐 수치적으로 빗나갈 수 있다. 중심 쪽으로 조금씩 당기며
    # 재시도한다.
    if hit is None:
        for inset in (0.01, 0.03, 0.08):
            retry_placement = _placement_with_axial_ratio(
                placement,
                float(
                    np.clip(
                        placement.axial_ratio,
                        inset,
                        1.0 - inset,
                    )
                ),
            )

            if (
                retry_placement.axial_ratio
                == placement.axial_ratio
            ):
                continue

            origin, direction = build_placement_ray(
                retry_placement,
                bounds_min,
                bounds_max,
                long_axis_index,
                theta_zero_axis_index,
                theta_ninety_axis_index,
                jitter_offset=jitter_offset,
            )
            hit = ray_mesh_intersect(mesh, origin, direction)

            if hit is not None:
                break

    if hit is None:
        raise RuntimeError(
            "표면 배치 ray가 메시와 만나지 않았습니다 "
            f"(theta={placement.theta_deg}, "
            f"axial_ratio={placement.axial_ratio}). "
            "theta/axial_ratio가 물체 표면을 향하는지 확인하세요."
        )

    _, triangle_index, hit_point, _ = hit
    normal = smooth_normal_at(
        mesh,
        hit_point,
        triangle_index,
        smooth_normal_radius,
    )

    # 법선이 ray를 향하도록(즉 바깥쪽) 뒤집는다.
    if np.dot(normal, direction) > 0.0:
        normal = -normal

    tangent, bitangent = build_tangent_basis(
        normal,
        placement.tangent_deg,
        long_axis_index,
    )
    decal_origin = hit_point + normal * decal_offset_m

    return {
        "surface_point": hit_point,
        "decal_origin": decal_origin,
        "normal": normal,
        "tangent": tangent,
        "bitangent": bitangent,
        "triangle_index": triangle_index,
        "ray_origin": origin,
        "ray_direction": direction,
    }


def decal_transform_matrix(resolved):
    """decal quad의 object frame 4x4 변환 (행 우선, 마지막 행 [0,0,0,1]).

    quad는 local +x = tangent, +y = bitangent, +z = normal 평면에
    놓인다.
    """
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, 0] = resolved["tangent"]
    matrix[:3, 1] = resolved["bitangent"]
    matrix[:3, 2] = resolved["normal"]
    matrix[:3, 3] = resolved["decal_origin"]

    return matrix


def decal_quad_corners(resolved, patch_mm):
    """object frame에서 decal quad의 네 꼭짓점.

    GT 평가에서 결함 footprint를 이미지로 투영할 때 쓴다.
    """
    half = 0.5 * patch_mm * 1e-3
    tangent = resolved["tangent"]
    bitangent = resolved["bitangent"]
    origin = resolved["decal_origin"]

    return np.stack(
        [
            origin - half * tangent - half * bitangent,
            origin + half * tangent - half * bitangent,
            origin + half * tangent + half * bitangent,
            origin - half * tangent + half * bitangent,
        ],
        axis=0,
    )


# ============================================================
# 합성 메시 (단독 검증용)
# ============================================================


def make_box_mesh(size_x, size_y, size_z, center=(0.0, 0.0, 0.0)):
    half = np.array(
        [size_x, size_y, size_z],
        dtype=np.float64,
    ) * 0.5
    center = np.asarray(center, dtype=np.float64)
    signs = np.array(
        [
            [-1, -1, -1],
            [+1, -1, -1],
            [+1, +1, -1],
            [-1, +1, -1],
            [-1, -1, +1],
            [+1, -1, +1],
            [+1, +1, +1],
            [-1, +1, +1],
        ],
        dtype=np.float64,
    )
    vertices = center[None, :] + signs * half[None, :]
    quads = [
        (0, 3, 2, 1),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (2, 3, 7, 6),
        (1, 2, 6, 5),
        (0, 4, 7, 3),
    ]
    triangles = []

    for a, b, c, d in quads:
        triangles.append((a, b, c))
        triangles.append((a, c, d))

    return TriangleMesh(vertices, triangles)


def make_capsule_like_mesh(
    radius_x,
    radius_z,
    height,
    segments=48,
    rings=24,
):
    """타원 단면 기둥. 부드러운 법선 검증용."""
    vertices = []

    for ring_index in range(rings + 1):
        v = ring_index / rings
        y = -0.5 * height + v * height

        for segment_index in range(segments):
            angle = 2.0 * math.pi * segment_index / segments
            vertices.append(
                (
                    radius_x * math.sin(angle),
                    y,
                    radius_z * math.cos(angle),
                )
            )

    bottom_index = len(vertices)
    vertices.append((0.0, -0.5 * height, 0.0))
    top_index = len(vertices)
    vertices.append((0.0, +0.5 * height, 0.0))

    triangles = []

    for ring_index in range(rings):
        for segment_index in range(segments):
            next_segment = (segment_index + 1) % segments
            a = ring_index * segments + segment_index
            b = ring_index * segments + next_segment
            c = (ring_index + 1) * segments + next_segment
            d = (ring_index + 1) * segments + segment_index
            triangles.append((a, b, c))
            triangles.append((a, c, d))

    for segment_index in range(segments):
        next_segment = (segment_index + 1) % segments
        triangles.append(
            (bottom_index, next_segment, segment_index)
        )
        triangles.append(
            (
                top_index,
                rings * segments + segment_index,
                rings * segments + next_segment,
            )
        )

    return TriangleMesh(vertices, triangles)


# ============================================================
# 단독 검증
# ============================================================


class _Placement:
    """테스트용 최소 배치 객체 (config import 없이 돌리기 위함)."""

    def __init__(
        self,
        theta_deg,
        axial_ratio,
        tangent_deg=0.0,
        jitter_mm=0.0,
    ):
        self.theta_deg = theta_deg
        self.axial_ratio = axial_ratio
        self.tangent_deg = tangent_deg
        self.jitter_mm = jitter_mm


def _run_self_test():
    """머스타드 병 공칭 치수의 합성 메시로 배치 수학을 검증한다."""
    long_axis = 1
    zero_axis = 2
    ninety_axis = 0
    offset = 2e-4
    failures = []

    def check(label, condition, detail=""):
        status = "PASS" if condition else "FAIL"
        print(f"  [{status}] {label} {detail}")

        if not condition:
            failures.append(label)

    size = (0.096, 0.191, 0.060)
    box = make_box_mesh(*size)

    print("box mesh (0.096 x 0.191 x 0.060):")

    expected_faces = {
        0.0: (np.array([0.0, 0.0, 1.0]), 2, +0.030),
        90.0: (np.array([1.0, 0.0, 0.0]), 0, +0.048),
        180.0: (np.array([0.0, 0.0, -1.0]), 2, -0.030),
        270.0: (np.array([-1.0, 0.0, 0.0]), 0, -0.048),
    }

    for theta, (
        expected_normal,
        axis_index,
        expected_coordinate,
    ) in expected_faces.items():
        resolved = resolve_placement(
            box,
            _Placement(theta, 0.5),
            long_axis,
            zero_axis,
            ninety_axis,
            offset,
        )
        normal_error = float(
            np.linalg.norm(resolved["normal"] - expected_normal)
        )
        coordinate = float(
            resolved["surface_point"][axis_index]
        )
        check(
            f"theta={theta:5.1f} 법선",
            normal_error < 1e-9,
            f"err={normal_error:.2e}",
        )
        check(
            f"theta={theta:5.1f} 표면 좌표",
            abs(coordinate - expected_coordinate) < 1e-9,
            f"{coordinate:+.4f} (기대 {expected_coordinate:+.4f})",
        )

    print("axial_ratio -> 장축 좌표:")

    for ratio, expected_y in (
        (0.0, -0.0955),
        (0.25, -0.0478),
        (0.5, 0.0),
        (0.75, +0.0478),
        (1.0, +0.0955),
    ):
        resolved = resolve_placement(
            box,
            _Placement(0.0, ratio),
            long_axis,
            zero_axis,
            ninety_axis,
            offset,
        )
        y = float(resolved["surface_point"][1])
        # 0/1은 면 경계라 중심 쪽으로 당겨 재시도된다.
        tolerance = 0.02 if ratio in (0.0, 1.0) else 1e-4
        check(
            f"axial_ratio={ratio:4.2f}",
            abs(y - expected_y) < tolerance,
            f"y={y:+.4f} (기대 {expected_y:+.4f})",
        )

    print("접평면 기저:")

    resolved = resolve_placement(
        box,
        _Placement(0.0, 0.5, tangent_deg=0.0),
        long_axis,
        zero_axis,
        ninety_axis,
        offset,
    )
    tangent = resolved["tangent"]
    bitangent = resolved["bitangent"]
    normal = resolved["normal"]
    check(
        "tangent_deg=0 -> 장축 평행",
        abs(abs(float(tangent[1])) - 1.0) < 1e-9,
        f"tangent={np.round(tangent, 4)}",
    )
    check(
        "정규직교",
        abs(float(np.dot(tangent, bitangent))) < 1e-12
        and abs(float(np.dot(tangent, normal))) < 1e-12
        and abs(float(np.dot(bitangent, normal))) < 1e-12,
    )
    check(
        "오른손 좌표계",
        float(
            np.dot(np.cross(tangent, bitangent), normal)
        ) > 0.999,
    )
    check(
        "decal이 법선 방향으로 정확히 offset만큼 떠 있음",
        abs(
            float(
                np.dot(
                    resolved["decal_origin"]
                    - resolved["surface_point"],
                    normal,
                )
            )
            - offset
        ) < 1e-15,
    )

    rotated = resolve_placement(
        box,
        _Placement(0.0, 0.5, tangent_deg=90.0),
        long_axis,
        zero_axis,
        ninety_axis,
        offset,
    )
    check(
        "tangent_deg=90 -> 장축과 수직",
        abs(float(rotated["tangent"][1])) < 1e-9,
        f"tangent={np.round(rotated['tangent'], 4)}",
    )

    print("quad 꼭짓점:")

    patch_mm = 20.0
    corners = decal_quad_corners(resolved, patch_mm)
    side = float(np.linalg.norm(corners[1] - corners[0]))
    plane_offsets = [
        abs(
            float(
                np.dot(
                    corner - resolved["decal_origin"],
                    resolved["normal"],
                )
            )
        )
        for corner in corners
    ]
    check(
        "한 변 길이 = patch_mm",
        abs(side - patch_mm * 1e-3) < 1e-12,
        f"{side * 1000:.4f}mm",
    )
    check(
        "네 점이 같은 평면",
        max(plane_offsets) < 1e-15,
    )

    print("타원 기둥 (부드러운 법선):")

    cylinder = make_capsule_like_mesh(0.048, 0.030, 0.191)

    for theta in (0.0, 45.0, 135.0, 220.0):
        resolved = resolve_placement(
            cylinder,
            _Placement(theta, 0.5),
            long_axis,
            zero_axis,
            ninety_axis,
            offset,
        )
        point = resolved["surface_point"]
        # 타원면 f = (x/a)^2 + (z/b)^2 의 해석적 법선.
        analytic = np.array(
            [
                point[0] / 0.048 ** 2,
                0.0,
                point[2] / 0.030 ** 2,
            ]
        )
        analytic = analytic / np.linalg.norm(analytic)
        angle_deg = math.degrees(
            math.acos(
                float(
                    np.clip(
                        np.dot(resolved["normal"], analytic),
                        -1.0,
                        1.0,
                    )
                )
            )
        )
        check(
            f"theta={theta:5.1f} 법선이 해석해와 일치",
            angle_deg < 2.0,
            f"오차 {angle_deg:.2f}deg",
        )
        check(
            f"theta={theta:5.1f} 법선이 장축 성분 없음",
            abs(float(resolved["normal"][1])) < 0.05,
        )

    print("jitter:")

    generator = np.random.default_rng(0)
    max_shift = 0.0

    for _ in range(200):
        jittered = resolve_placement(
            box,
            _Placement(0.0, 0.5, jitter_mm=1.0),
            long_axis,
            zero_axis,
            ninety_axis,
            offset,
            rng=generator,
        )
        shift = float(
            np.linalg.norm(
                jittered["surface_point"]
                - np.array([0.0, 0.0, 0.030])
            )
        )
        max_shift = max(max_shift, shift)

    check(
        "jitter가 선언한 반경 안에 머묾",
        max_shift <= 1.0e-3 + 1e-9,
        f"최대 {max_shift * 1000:.4f}mm",
    )

    print()

    if failures:
        print(f"[FAIL] {len(failures)}개 실패: {failures}")

        return 1

    print("[OK] 배치 수학 검증 통과")

    return 0


if __name__ == "__main__":
    raise SystemExit(_run_self_test())


# ============================================================
# 곡면 정합 decal
# ============================================================
#
# 평면 quad를 곡면에 접하게 붙이면 중심에서 멀어질수록 모서리가 물체
# 안으로 파묻힌다. 머스타드 병의 단면 반경 30mm 기준, 중심에서 7mm
# 떨어진 지점의 sagitta가 0.83mm인데 decal offset은 0.2mm뿐이다. 즉
# 14mm 스크래치의 양 끝이 표면 아래로 사라진다.
#
# 그래서 quad를 격자로 잘게 나누고 각 정점을 표면으로 다시 투영한다.


def build_conformed_decal(
    mesh,
    resolved,
    patch_mm,
    decal_offset_m,
    subdivisions=12,
    smooth_normal_radius=0.004,
):
    """decal 격자를 실제 표면에 정합시킨 메시 데이터를 만든다.

    반환 dict:
        points               (K, 3) object frame 정점
        normals              (K, 3) 정점 법선
        st                   (K, 2) UV
        face_vertex_counts   (F,)   전부 4
        face_vertex_indices  (4F,)
        miss_count           표면을 못 맞춘 정점 수 (평면으로 폴백)
        max_conform_shift_m  평면 대비 최대 보정량
    """
    subdivisions = int(max(1, subdivisions))
    half = 0.5 * patch_mm * 1e-3
    origin = resolved["decal_origin"]
    normal = resolved["normal"]
    tangent = resolved["tangent"]
    bitangent = resolved["bitangent"]

    extent = np.linalg.norm(
        mesh.bounds[1] - mesh.bounds[0]
    )
    standoff = float(extent) + 1.0

    coordinates = np.linspace(
        -half,
        half,
        subdivisions + 1,
    )
    points = []
    normals = []
    st = []
    triangle_indices = []
    barycentrics = []
    miss_count = 0
    max_shift = 0.0

    for row, v in enumerate(coordinates):
        for column, u in enumerate(coordinates):
            planar = origin + u * tangent + v * bitangent
            ray_origin = planar + normal * standoff
            hit = ray_mesh_intersect(mesh, ray_origin, -normal)

            if hit is None:
                # 실루엣 밖이라 표면을 못 맞춘 정점. 평면으로 둔다.
                miss_count += 1
                points.append(planar)
                normals.append(normal)
                triangle_indices.append(-1)
                barycentrics.append(
                    np.array([1.0, 0.0, 0.0], dtype=np.float64)
                )
            else:
                _, triangle_index, hit_point, bary = hit
                local_normal = smooth_normal_at(
                    mesh,
                    hit_point,
                    triangle_index,
                    smooth_normal_radius,
                )

                if np.dot(local_normal, normal) < 0.0:
                    local_normal = -local_normal

                vertex = hit_point + local_normal * decal_offset_m
                max_shift = max(
                    max_shift,
                    float(np.linalg.norm(vertex - planar)),
                )
                points.append(vertex)
                normals.append(local_normal)
                triangle_indices.append(triangle_index)
                barycentrics.append(bary)

            st.append(
                (
                    column / subdivisions,
                    row / subdivisions,
                )
            )

    face_vertex_counts = []
    face_vertex_indices = []
    stride = subdivisions + 1

    for row in range(subdivisions):
        for column in range(subdivisions):
            base = row * stride + column
            face_vertex_counts.append(4)
            face_vertex_indices.extend(
                [
                    base,
                    base + 1,
                    base + stride + 1,
                    base + stride,
                ]
            )

    return {
        "points": np.asarray(points, dtype=np.float64),
        "normals": np.asarray(normals, dtype=np.float64),
        "st": np.asarray(st, dtype=np.float64),
        "face_vertex_counts": np.asarray(
            face_vertex_counts,
            dtype=np.int32,
        ),
        "face_vertex_indices": np.asarray(
            face_vertex_indices,
            dtype=np.int32,
        ),
        "triangle_indices": np.asarray(
            triangle_indices,
            dtype=np.int32,
        ),
        "barycentrics": np.asarray(barycentrics, dtype=np.float64),
        "subdivisions": subdivisions,
        "miss_count": miss_count,
        "max_conform_shift_m": max_shift,
    }


def measure_surface_clearance(mesh, points, normal_hint):
    """각 정점이 표면에서 법선 방향으로 얼마나 떨어져 있는지 잰다.

    음수면 물체 안으로 파묻힌 것이다. decal 정합 검증용.
    """
    extent = float(
        np.linalg.norm(mesh.bounds[1] - mesh.bounds[0])
    )
    standoff = extent + 1.0
    clearances = []

    for point in np.asarray(points, dtype=np.float64):
        ray_origin = point + normal_hint * standoff
        hit = ray_mesh_intersect(mesh, ray_origin, -normal_hint)

        if hit is None:
            clearances.append(np.nan)
            continue

        _, _, hit_point, _ = hit
        clearances.append(
            float(np.dot(point - hit_point, normal_hint))
        )

    return np.asarray(clearances, dtype=np.float64)
