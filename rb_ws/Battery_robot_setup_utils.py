"""RB5 articulation setup helpers."""

from isaacsim.core.experimental.prims import Articulation
from pxr import UsdPhysics


def set_rb5_arm_link_collisions_enabled(
    stage,
    robot_prim_path,
    enabled,
    link_indices=range(1, 7),
    log_prefix="[RB5 PHYSX]",
):
    """Toggle Isaac PhysX colliders for RB5 link1..link6 only.

    The imported RB5 collision meshes live inside instanceable USD references.
    Authoring ``physics:collisionEnabled`` on an instance proxy is illegal, so
    each link's collision-only reference is first made non-instanceable in the
    stage session layer.  EPick lives below ``tcp/epick_base_link`` and is not
    touched by these explicit body-collider paths.
    """
    previous_edit_target = stage.GetEditTarget()
    stage.SetEditTarget(stage.GetSessionLayer())
    current_link_path = f"{robot_prim_path}/Geometry/link0"
    changed_paths = []

    try:
        for link_index in link_indices:
            link_name = f"link{int(link_index)}"
            current_link_path = f"{current_link_path}/{link_name}"
            collider_instance_path = (
                f"{current_link_path}/{link_name}_1"
            )
            collider_instance_prim = stage.GetPrimAtPath(
                collider_instance_path
            )

            if not collider_instance_prim.IsValid():
                print(
                    log_prefix,
                    "WARNING: collision instance prim not found:",
                    collider_instance_path,
                )
                continue

            if collider_instance_prim.IsInstanceable():
                collider_instance_prim.SetInstanceable(False)

            collider_path = (
                f"{collider_instance_path}/{link_name}_1"
            )
            collider_prim = stage.GetPrimAtPath(collider_path)

            if not collider_prim.IsValid():
                print(
                    log_prefix,
                    "WARNING: collider prim not found after de-instancing:",
                    collider_path,
                )
                continue
            if collider_prim.IsInstanceProxy():
                print(
                    log_prefix,
                    "WARNING: collider is still an instance proxy:",
                    collider_path,
                )
                continue
            if not collider_prim.HasAPI(UsdPhysics.CollisionAPI):
                print(
                    log_prefix,
                    "WARNING: CollisionAPI missing:",
                    collider_path,
                )
                continue

            collision_api = UsdPhysics.CollisionAPI(collider_prim)
            collision_api.CreateCollisionEnabledAttr().Set(bool(enabled))
            changed_paths.append(collider_path)
            print(
                log_prefix,
                "Collision enabled:",
                collider_path,
                bool(enabled),
            )
    finally:
        stage.SetEditTarget(previous_edit_target)

    print(
        log_prefix,
        f"RB5 link1-link6 colliders changed: {len(changed_paths)},",
        "EPick collider paths were excluded.",
    )
    return changed_paths


def create_rb_articulation(
    articulation_root,
    before_log=None,
    after_log=None,
    print_dof_names=True,
    flush=False,
):
    if before_log is not None:
        print(
            before_log,
            flush=flush,
        )

    rb_robot = Articulation(
        paths=articulation_root
    )

    if after_log is not None:
        print(
            after_log,
            flush=flush,
        )

    if print_dof_names:
        print(
            "RB5 DOF names:",
            rb_robot.dof_names,
            flush=flush,
        )

    return rb_robot


def hold_current_joint_positions(
    rb_robot,
):
    current_joint_positions = rb_robot.get_dof_positions()

    rb_robot.set_dof_position_targets(
        current_joint_positions
    )

    rb_robot.switch_dof_control_mode("position")


def apply_rb5_drive_settings(
    rb_robot,
):
    # The finite-stiffness PhysX position drives must generate gravity/load
    # holding torque from a non-zero position error.  With the former gains,
    # shoulder sag alone was measured at 0.04645 rad (2.66 deg), producing a
    # 34.9 mm TCP error.  Strengthen the load-bearing pitch joints and scale
    # their damping approximately with sqrt(stiffness ratio) to retain a
    # similar damping ratio without introducing excessive oscillation.
    rb_robot.set_dof_gains(
        stiffnesses=[
            400.0,
            10000.0,
            5000.0,
            1000.0,
            150.0,
            100.0,
        ],
        dampings=[
            40.0,
            525.0,
            270.0,
            45.0,
            15.0,
            10.0,
        ],
    )

    rb_robot.set_dof_max_efforts(
        max_efforts=[
            300.0,
            300.0,
            200.0,
            100.0,
            100.0,
            50.0,
        ]
    )


def print_rb5_drive_settings(
    rb_robot,
):
    stiffnesses, dampings = rb_robot.get_dof_gains()
    max_efforts = rb_robot.get_dof_max_efforts()
    max_velocities = rb_robot.get_dof_max_velocities()

    stiffnesses = stiffnesses.numpy()[0]
    dampings = dampings.numpy()[0]
    max_efforts = max_efforts.numpy()[0]
    max_velocities = max_velocities.numpy()[0]

    print("\n========== RB5 Drive Settings ==========")

    for index, joint_name in enumerate(rb_robot.dof_names):
        print(
            f"{joint_name}: "
            f"stiffness={stiffnesses[index]:.6f}, "
            f"damping={dampings[index]:.6f}, "
            f"max_effort={max_efforts[index]:.6f}, "
            f"max_velocity={max_velocities[index]:.6f}"
        )

    print("========================================")


def configure_rb5_position_control(
    rb_robot,
):
    hold_current_joint_positions(
        rb_robot
    )
    apply_rb5_drive_settings(
        rb_robot
    )
    print_rb5_drive_settings(
        rb_robot
    )
