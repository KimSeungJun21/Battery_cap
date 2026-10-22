"""Target object setup helpers for Battery_rb.py."""

from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

import isaacsim.core.experimental.utils.stage as stage_utils
from isaacsim.core.utils.semantics import add_labels


def add_referenced_target_prim(
    stage,
    simulation_app,
    usd_path,
    prim_path,
    update_count=10,
    description="Target",
):
    stage_utils.add_reference_to_stage(
        usd_path=usd_path,
        path=prim_path,
    )

    for _ in range(update_count):
        simulation_app.update()

    target_prim = stage.GetPrimAtPath(
        prim_path
    )

    if not target_prim.IsValid():
        raise RuntimeError(
            f"{description} prim을 찾지 못했습니다: {prim_path}"
        )

    return target_prim


def set_target_pose(
    target_prim,
    position,
    rotation_xyz_deg,
    scale,
    log_prefix="[TARGET]",
):
    target_xform = UsdGeom.Xformable(
        target_prim
    )

    target_xform.ClearXformOpOrder()
    target_xform.AddTranslateOp().Set(
        position
    )
    target_xform.AddRotateXYZOp().Set(
        rotation_xyz_deg
    )
    target_xform.AddScaleOp().Set(
        scale
    )

    print(
        log_prefix,
        "Pose set:",
        "position=",
        position,
        "rotation_xyz_deg=",
        rotation_xyz_deg,
    )


def set_target_kinematic(
    stage,
    target_prim,
    target_prim_path,
    enabled,
    log_prefix="[TARGET]",
):
    rigid_body_prims = find_target_rigid_body_prims(
        stage=stage,
        target_prim_path=target_prim_path,
    )

    if not rigid_body_prims:
        UsdPhysics.RigidBodyAPI.Apply(
            target_prim
        )

        rigid_body_prims.append(
            target_prim
        )

    for rigid_body_prim in rigid_body_prims:
        rigid_body_api = UsdPhysics.RigidBodyAPI(
            rigid_body_prim
        )
        did_set = set_schema_bool_attr(
            api=rigid_body_api,
            prim=rigid_body_prim,
            create_method_name="CreateKinematicEnabledAttr",
            attr_name="physics:kinematicEnabled",
            value=enabled,
        )

        print(
            log_prefix,
            "Kinematic:",
            str(rigid_body_prim.GetPath()),
            enabled,
            "set=",
            did_set,
        )


def iter_target_prims(
    stage,
    target_prim_path,
):
    for child_prim in stage.Traverse():
        child_path = str(child_prim.GetPath())

        if child_path.startswith(target_prim_path):
            yield child_prim


def find_target_rigid_body_prims(
    stage,
    target_prim_path,
):
    return [
        prim
        for prim in iter_target_prims(
            stage=stage,
            target_prim_path=target_prim_path,
        )
        if prim.HasAPI(UsdPhysics.RigidBodyAPI)
    ]


def ensure_target_mesh_colliders(
    stage,
    target_prim_path,
    approximation="convexHull",
    log_prefix="[TARGET]",
):
    """Apply an enabled convex collider to every target mesh.

    Dynamic rigid bodies cannot use raw triangle-mesh collision in PhysX, so
    convexHull is the safe default for an imported visual-only USD.
    """
    collider_count = 0

    for prim in iter_target_prims(
        stage=stage,
        target_prim_path=target_prim_path,
    ):
        if not prim.IsA(UsdGeom.Mesh):
            continue

        if prim.HasAPI(UsdPhysics.CollisionAPI):
            collision_api = UsdPhysics.CollisionAPI(prim)
        else:
            collision_api = UsdPhysics.CollisionAPI.Apply(prim)

        collision_api.CreateCollisionEnabledAttr().Set(True)

        if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
            mesh_collision_api = UsdPhysics.MeshCollisionAPI(prim)
        else:
            mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(prim)

        mesh_collision_api.CreateApproximationAttr().Set(
            approximation
        )
        collider_count += 1

    print(
        log_prefix,
        "Mesh colliders configured:",
        collider_count,
        "approximation=",
        approximation,
    )

    if collider_count == 0:
        raise RuntimeError(
            f"{target_prim_path} 아래에서 collider로 사용할 Mesh를 "
            "찾지 못했습니다."
        )

    return collider_count


def set_schema_bool_attr(
    api,
    prim,
    create_method_name,
    attr_name,
    value,
):
    create_method = getattr(
        api,
        create_method_name,
        None,
    )

    if create_method is not None:
        create_method().Set(
            value
        )
        return True

    attr = prim.GetAttribute(
        attr_name
    )

    if attr.IsValid():
        attr.Set(
            value
        )
        return True

    return False


def set_target_rigid_body_enabled(
    stage,
    target_prim,
    target_prim_path,
    enabled,
    log_prefix="[TARGET]",
):
    rigid_body_prims = find_target_rigid_body_prims(
        stage=stage,
        target_prim_path=target_prim_path,
    )

    if not rigid_body_prims:
        UsdPhysics.RigidBodyAPI.Apply(
            target_prim
        )
        rigid_body_prims.append(
            target_prim
        )

    for rigid_body_prim in rigid_body_prims:
        rigid_body_api = UsdPhysics.RigidBodyAPI(
            rigid_body_prim
        )
        did_set = set_schema_bool_attr(
            api=rigid_body_api,
            prim=rigid_body_prim,
            create_method_name="CreateRigidBodyEnabledAttr",
            attr_name="physics:rigidBodyEnabled",
            value=enabled,
        )

        print(
            log_prefix,
            "RigidBody enabled:",
            str(rigid_body_prim.GetPath()),
            enabled,
            "set=",
            did_set,
        )


def set_target_collision_enabled(
    stage,
    target_prim_path,
    enabled,
    log_prefix="[TARGET]",
):
    collision_prims = [
        prim
        for prim in iter_target_prims(
            stage=stage,
            target_prim_path=target_prim_path,
        )
        if prim.HasAPI(UsdPhysics.CollisionAPI)
    ]

    for collision_prim in collision_prims:
        collision_api = UsdPhysics.CollisionAPI(
            collision_prim
        )
        did_set = set_schema_bool_attr(
            api=collision_api,
            prim=collision_prim,
            create_method_name="CreateCollisionEnabledAttr",
            attr_name="physics:collisionEnabled",
            value=enabled,
        )

        print(
            log_prefix,
            "Collision enabled:",
            str(collision_prim.GetPath()),
            enabled,
            "set=",
            did_set,
        )

    if not collision_prims:
        print(
            log_prefix,
            "WARNING: CollisionAPI prim을 찾지 못했습니다:",
            target_prim_path,
        )


def set_target_gravity_enabled(
    stage,
    target_prim,
    target_prim_path,
    enabled,
    log_prefix="[TARGET]",
):
    """Enable or disable PhysX gravity for every target rigid body."""
    rigid_body_prims = find_target_rigid_body_prims(
        stage=stage,
        target_prim_path=target_prim_path,
    )

    if not rigid_body_prims:
        UsdPhysics.RigidBodyAPI.Apply(target_prim)
        rigid_body_prims.append(target_prim)

    for rigid_body_prim in rigid_body_prims:
        if rigid_body_prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
            physx_api = PhysxSchema.PhysxRigidBodyAPI(rigid_body_prim)
        else:
            physx_api = PhysxSchema.PhysxRigidBodyAPI.Apply(
                rigid_body_prim
            )

        # PhysX exposes this setting as disableGravity, so invert the
        # user-facing "gravity enabled" option.
        physx_api.CreateDisableGravityAttr().Set(not bool(enabled))
        print(
            log_prefix,
            "Gravity enabled:",
            str(rigid_body_prim.GetPath()),
            bool(enabled),
        )


def configure_target_physics(
    stage,
    target_prim,
    target_prim_path,
    collision_enabled=True,
    gravity_enabled=True,
    create_missing_colliders=False,
    collision_approximation="convexHull",
    log_prefix="[TARGET]",
):
    """Configure a referenced target as a dynamic PhysX rigid body."""
    set_target_rigid_body_enabled(
        stage=stage,
        target_prim=target_prim,
        target_prim_path=target_prim_path,
        enabled=True,
        log_prefix=log_prefix,
    )
    set_target_kinematic(
        stage=stage,
        target_prim=target_prim,
        target_prim_path=target_prim_path,
        enabled=False,
        log_prefix=log_prefix,
    )

    if collision_enabled and create_missing_colliders:
        ensure_target_mesh_colliders(
            stage=stage,
            target_prim_path=target_prim_path,
            approximation=collision_approximation,
            log_prefix=log_prefix,
        )

    set_target_collision_enabled(
        stage=stage,
        target_prim_path=target_prim_path,
        enabled=collision_enabled,
        log_prefix=log_prefix,
    )
    set_target_gravity_enabled(
        stage=stage,
        target_prim=target_prim,
        target_prim_path=target_prim_path,
        enabled=gravity_enabled,
        log_prefix=log_prefix,
    )

    print(
        log_prefix,
        "Physics configured:",
        "collision=",
        bool(collision_enabled),
        "gravity=",
        bool(gravity_enabled),
    )


def reset_target_rigid_body_velocities(
    stage,
    target_prim_path,
    log_prefix="[TARGET]",
):
    zero_velocity = Gf.Vec3f(
        0.0,
        0.0,
        0.0,
    )
    rigid_body_prims = find_target_rigid_body_prims(
        stage=stage,
        target_prim_path=target_prim_path,
    )

    for rigid_body_prim in rigid_body_prims:
        velocity_attr = rigid_body_prim.GetAttribute(
            "physics:velocity"
        )
        angular_velocity_attr = rigid_body_prim.GetAttribute(
            "physics:angularVelocity"
        )

        if velocity_attr.IsValid():
            velocity_attr.Set(
                zero_velocity
            )

        if angular_velocity_attr.IsValid():
            angular_velocity_attr.Set(
                zero_velocity
            )

        print(
            log_prefix,
            "RigidBody velocity reset:",
            str(rigid_body_prim.GetPath()),
        )


def restore_target_dynamic_state(
    stage,
    target_prim,
    target_prim_path,
    position,
    rotation_xyz_deg,
    scale,
    log_prefix="[TARGET]",
):
    """NBV용 kinematic pose sweep 이후 물리 상호작용 상태로 복귀한다."""
    print(
        log_prefix,
        "Restoring dynamic simulation state...",
    )

    set_target_pose(
        target_prim=target_prim,
        position=position,
        rotation_xyz_deg=rotation_xyz_deg,
        scale=scale,
        log_prefix=log_prefix,
    )
    set_target_rigid_body_enabled(
        stage=stage,
        target_prim=target_prim,
        target_prim_path=target_prim_path,
        enabled=True,
        log_prefix=log_prefix,
    )
    set_target_kinematic(
        stage=stage,
        target_prim=target_prim,
        target_prim_path=target_prim_path,
        enabled=False,
        log_prefix=log_prefix,
    )
    set_target_collision_enabled(
        stage=stage,
        target_prim_path=target_prim_path,
        enabled=True,
        log_prefix=log_prefix,
    )
    reset_target_rigid_body_velocities(
        stage=stage,
        target_prim_path=target_prim_path,
        log_prefix=log_prefix,
    )

    print(
        log_prefix,
        "Dynamic simulation state restored.",
    )


def add_target_labels(
    stage,
    target_prim,
    target_prim_path,
    class_label="target_object",
    object_label=None,
):
    add_labels(
        target_prim,
        labels=[class_label],
        instance_name="class",
    )

    if object_label is not None:
        add_labels(
            target_prim,
            labels=[object_label],
            instance_name="object",
        )

    for child_prim in stage.Traverse():
        child_path = str(child_prim.GetPath())

        if (
            child_path.startswith(target_prim_path)
            and child_prim.IsA(UsdGeom.Mesh)
        ):
            add_labels(
                child_prim,
                labels=[class_label],
                instance_name="class",
            )
