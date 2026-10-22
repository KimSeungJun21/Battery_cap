import isaacsim.core.experimental.utils.stage as stage_utils
import omni.usd
from pxr import UsdGeom, Gf, UsdPhysics,Sdf,UsdShade, Sdf, Vt
from isaacsim.core.experimental.materials import PreviewSurfaceMaterial
from isaacsim.core.experimental.objects import Cube
import numpy as np
from isaacsim.core.experimental.prims import GeomPrim, RigidPrim
import omni.kit.commands


def create_image_material(material_path, image_path):
    import os
    import omni.usd
    from pxr import UsdShade, Sdf

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    stage = omni.usd.get_context().get_stage()

    material = UsdShade.Material.Define(stage, material_path)

    shader = UsdShade.Shader.Define(stage, material_path + "/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.4)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    
    tex = UsdShade.Shader.Define(stage, material_path + "/ImageTexture")
    tex.CreateIdAttr("UsdUVTexture")

    # 중요: 그냥 string 말고 Sdf.AssetPath
    tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(image_path))
    tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
    tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)

    st_reader = UsdShade.Shader.Define(stage, material_path + "/PrimvarReader_st")
    st_reader.CreateIdAttr("UsdPrimvarReader_float2")
    st_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    st_reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)

    tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
        st_reader.ConnectableAPI(),
        "result",
    )

    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        tex.ConnectableAPI(),
        "rgb",
    )

    material.CreateSurfaceOutput().ConnectToSource(
        shader.ConnectableAPI(),
        "surface",
    )

    print("Created image material:", material_path, "from", image_path)
    return material

def add_image_label_to_box_root(
    image_path,
    box_root_path="/World/FlatBox",
    box_scale=(0.25, 0.18, 0.035),
    label_name="Label",
    material_path="/World/Materials/flatbox_label_mat",
    face="top",
    margin=0.90,
    z_offset=0.002,
):
    import os
    import omni.usd
    from pxr import UsdGeom, Gf, UsdShade, Sdf, Vt

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    stage = omni.usd.get_context().get_stage()

    root_prim = stage.GetPrimAtPath(box_root_path)
    if not root_prim.IsValid():
        raise RuntimeError(f"Box root prim not found: {box_root_path}")

    label_path = box_root_path + "/" + label_name

    sx, sy, sz = box_scale
    hx = sx * 0.5 * margin
    hy = sy * 0.5 * margin
    hz = sz * 0.5 * margin

    if face == "top":
        z = sz * 0.5 + z_offset
        points = [
            Gf.Vec3f(-hx, -hy, z),
            Gf.Vec3f( hx, -hy, z),
            Gf.Vec3f( hx,  hy, z),
            Gf.Vec3f(-hx,  hy, z),
        ]

    elif face == "front":  # +Y
        y = sy * 0.5 + z_offset
        points = [
            Gf.Vec3f(-hx, y, -hz),
            Gf.Vec3f( hx, y, -hz),
            Gf.Vec3f( hx, y,  hz),
            Gf.Vec3f(-hx, y,  hz),
        ]

    elif face == "back":  # -Y
        y = -sy * 0.5 - z_offset
        points = [
            Gf.Vec3f( hx, y, -hz),
            Gf.Vec3f(-hx, y, -hz),
            Gf.Vec3f(-hx, y,  hz),
            Gf.Vec3f( hx, y,  hz),
        ]

    elif face == "right":  # +X
        x = sx * 0.5 + z_offset
        points = [
            Gf.Vec3f(x,  hy, -hz),
            Gf.Vec3f(x, -hy, -hz),
            Gf.Vec3f(x, -hy,  hz),
            Gf.Vec3f(x,  hy,  hz),
        ]

    elif face == "left":  # -X
        x = -sx * 0.5 - z_offset
        points = [
            Gf.Vec3f(x, -hy, -hz),
            Gf.Vec3f(x,  hy, -hz),
            Gf.Vec3f(x,  hy,  hz),
            Gf.Vec3f(x, -hy,  hz),
        ]

    else:
        raise ValueError(f"Unknown face: {face}")

    mesh = UsdGeom.Mesh.Define(stage, label_path)
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.CreateDoubleSidedAttr(True)

    st = Vt.Vec2fArray([
        Gf.Vec2f(0.0, 0.0),
        Gf.Vec2f(1.0, 0.0),
        Gf.Vec2f(1.0, 1.0),
        Gf.Vec2f(0.0, 1.0),
    ])

    primvars_api = UsdGeom.PrimvarsAPI(mesh)
    st_primvar = primvars_api.CreatePrimvar(
        "st",
        Sdf.ValueTypeNames.TexCoord2fArray,
        UsdGeom.Tokens.faceVarying,
    )
    st_primvar.Set(st)
    st_primvar.SetIndices(Vt.IntArray([0, 1, 2, 3]))

    material = create_image_material(material_path, image_path)

    # 중요: Label prim에 직접 강한 material binding
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(
        material,
        bindingStrength=UsdShade.Tokens.strongerThanDescendants,
    )

    print("Image label added:", label_path)


def add_static_collider_preset(
    prim_path="/World/ConveyorBelt",
    approximation="convexHull",
):
    
    omni.kit.commands.execute(
        "SetStaticCollider",
        path=Sdf.Path(prim_path),
        approximationShape=approximation,
    )

    print(f"Static collider preset applied to {prim_path} / {approximation}")

def add_collision_preset_to_meshes(
    root_path,
    approximation="convexHull",   # "convexHull", "convexDecomposition", "boundingCube", "none"
):

    stage = omni.usd.get_context().get_stage()
    root_prim = stage.GetPrimAtPath(root_path)

    if not root_prim.IsValid():
        raise RuntimeError(f"Root prim not found: {root_path}")

    count = 0

    for prim in stage.Traverse():
        prim_path = str(prim.GetPath())

        if not prim_path.startswith(root_path):
            continue

        if not prim.IsA(UsdGeom.Mesh):
            continue

        # Collision API 적용
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI.Apply(prim)

        # Mesh collision approximation 적용
        if not prim.HasAPI(UsdPhysics.MeshCollisionAPI):
            mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
        else:
            mesh_collision_api = UsdPhysics.MeshCollisionAPI(prim)

        mesh_collision_api.CreateApproximationAttr().Set(approximation)

        count += 1
        print(f"Collider applied: {prim_path} / approximation={approximation}")

    print(f"Applied collision preset to {count} mesh prims under {root_path}")

def set_prim_transform(
    prim_path,
    translation=(0.0, 0.0, 0.0),
    rotation_deg=(0.0, 0.0, 0.0),
    scale=(1.0, 1.0, 1.0),
):
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)

    if not prim.IsValid():
        raise RuntimeError(f"Prim not found: {prim_path}")

    xformable = UsdGeom.Xformable(prim)

    # 기존 order만 정리. 기존 attribute 자체는 남아있을 수 있음.
    xformable.ClearXformOpOrder()

    translate_op = xformable.AddTranslateOp(
        precision=UsdGeom.XformOp.PrecisionDouble
    )
    rotate_op = xformable.AddRotateXYZOp(
        precision=UsdGeom.XformOp.PrecisionDouble
    )
    scale_op = xformable.AddScaleOp(
        precision=UsdGeom.XformOp.PrecisionDouble
    )

    translate_op.Set(Gf.Vec3d(*translation))
    rotate_op.Set(Gf.Vec3d(*rotation_deg))
    scale_op.Set(Gf.Vec3d(*scale))



def set_objects(assets_root_path=None):
    # UR robot from cloud
    robot_usd_path = assets_root_path + "/Isaac/Robots/UniversalRobots/ur5e/ur5e.usd"

    stage_utils.add_reference_to_stage(
        usd_path=robot_usd_path,
        path="/World/UR5e",
    )

    print("Loaded robot:", robot_usd_path)

    conveyor_belt_usd_path = assets_root_path + "/Isaac/Props/Conveyors/ConveyorBelt_A07.usd"

    stage_utils.add_reference_to_stage(
        usd_path=conveyor_belt_usd_path,
        path="/World/ConveyorBelt",
    )

    print("Loaded conveyor belt:", conveyor_belt_usd_path)

    set_prim_transform(
        "/World/ConveyorBelt",
        translation=(1.5, 0.56, 0.0),
        rotation_deg=(0.0, 0.0, 0.0),
        scale=(1, 0.3, 0.2),
    )

    add_static_collider_preset("/World/ConveyorBelt/Belt/SM_ConveyorBelt_A07_Belt", "convexHull")

    ########box소환
    box_root_path = "/World/FlatBox"
    box_body_path = "/World/FlatBox/Body"
    box_center = (0.0, 0.56, 0.7)
    box_scale = (0.25, 0.18, 0.035)
    box_scale2 = (0.23, 0.16, 0.035)

    # 갈색 박스 material 정의
    box_material = PreviewSurfaceMaterial("/World/Materials/box_brown")
    box_material.set_input_values("diffuseColor", [0.55, 0.32, 0.16])

    # 1. Rigid body root Xform 생성
    stage = omni.usd.get_context().get_stage()
    UsdGeom.Xform.Define(stage, box_root_path)

    set_prim_transform(
        box_root_path,
        translation=box_center,
        rotation_deg=(0.0, 0.0, 0.0),
        scale=(1.0, 1.0, 1.0),
    )

    # 2. Body cube는 root 아래 local origin에 생성
    box_shape = Cube(
        paths=box_body_path,
        positions=np.array([[box_center]]),
        sizes=[1.0],
        scales=np.array([box_scale2]),
        reset_xform_op_properties=True,
    )

    # 3. Collision은 Body에 적용
    GeomPrim(paths=box_shape.paths, apply_collision_apis=True)

    # 4. Rigid body는 root에 적용
    box = RigidPrim(paths=box_root_path, reset_xform_op_properties=True)

    # 5. 갈색 material은 Body에만 적용
    box_shape.apply_visual_materials(box_material)

    # 6. Label은 root의 child로 생성
    add_image_label_to_box_root(
        image_path="/home/seungjun/isaac_ws/label.png",
        box_root_path=box_root_path,
        box_scale=box_scale,
        label_name="Label",
        face="top",
    )

    # FlatBox 전체를 이동 시작점으로 옮긴다.
    box.set_world_poses(
        positions=np.array([[-1.0, box_center[1], box_center[2]]], dtype=np.float32)
    )

    return box


def set_box_conveyor(assets_root_path=None):
    # UR robot from cloud
    # robot_usd_path = assets_root_path + "/Isaac/Robots/UniversalRobots/ur5e/ur5e.usd"

    # stage_utils.add_reference_to_stage(
    #     usd_path=robot_usd_path,
    #     path="/World/UR5e",
    # )

    # print("Loaded robot:", robot_usd_path)

    conveyor_belt_usd_path = assets_root_path + "/Isaac/Props/Conveyors/ConveyorBelt_A07.usd"

    stage_utils.add_reference_to_stage(
        usd_path=conveyor_belt_usd_path,
        path="/World/ConveyorBelt",
    )

    print("Loaded conveyor belt:", conveyor_belt_usd_path)

    set_prim_transform(
        "/World/ConveyorBelt",
        translation=(1.5, 0.56, 0.0),
        rotation_deg=(0.0, 0.0, 0.0),
        scale=(1, 0.3, 0.2),
    )

    add_static_collider_preset("/World/ConveyorBelt/Belt/SM_ConveyorBelt_A07_Belt", "convexHull")

    ########box소환
    box_root_path = "/World/FlatBox"
    box_body_path = "/World/FlatBox/Body"
    box_center = (0.0, 0.56, 0.7)
    box_scale = (0.25, 0.18, 0.035)
    box_scale2 = (0.23, 0.16, 0.035)

    # 갈색 박스 material 정의
    box_material = PreviewSurfaceMaterial("/World/Materials/box_brown")
    box_material.set_input_values("diffuseColor", [0.55, 0.32, 0.16])

    # 1. Rigid body root Xform 생성
    stage = omni.usd.get_context().get_stage()
    UsdGeom.Xform.Define(stage, box_root_path)

    set_prim_transform(
        box_root_path,
        translation=box_center,
        rotation_deg=(0.0, 0.0, 0.0),
        scale=(1.0, 1.0, 1.0),
    )

    # 2. Body cube는 root 아래 local origin에 생성
    box_shape = Cube(
        paths=box_body_path,
        positions=np.array([[box_center]]),
        sizes=[1.0],
        scales=np.array([box_scale2]),
        reset_xform_op_properties=True,
    )

    # 3. Collision은 Body에 적용
    GeomPrim(paths=box_shape.paths, apply_collision_apis=True)

    # 4. Rigid body는 root에 적용
    box = RigidPrim(paths=box_root_path, reset_xform_op_properties=True)

    # 5. 갈색 material은 Body에만 적용
    box_shape.apply_visual_materials(box_material)

    # 6. Label은 root의 child로 생성
    add_image_label_to_box_root(
        image_path="/home/seungjun/isaac_ws/label.png",
        box_root_path=box_root_path,
        box_scale=box_scale,
        label_name="Label",
        face="top",
    )

    # FlatBox 전체를 이동 시작점으로 옮긴다.
    box.set_world_poses(
        positions=np.array([[-1.0, box_center[1], box_center[2]]], dtype=np.float32)
    )

    return box
