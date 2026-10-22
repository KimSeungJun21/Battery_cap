"""Epick suction cup simulation using Isaac Sim's Surface Gripper."""

import omni.usd
import rclpy
from isaacsim.robot.surface_gripper import create_surface_gripper
from isaacsim.robot.surface_gripper import _surface_gripper as surface_gripper
from pxr import Gf, Usd, UsdGeom, UsdPhysics
from std_msgs.msg import Bool
from usd.schema.isaac import robot_schema


def compute_relative_pose(stage, child_prim_path, parent_prim_path):
    child_prim = stage.GetPrimAtPath(child_prim_path)
    parent_prim = stage.GetPrimAtPath(parent_prim_path)
    child_world = omni.usd.get_world_transform_matrix(child_prim)
    parent_world = omni.usd.get_world_transform_matrix(parent_prim)
    relative = child_world * parent_world.GetInverse()
    translation = relative.ExtractTranslation()
    rotation = relative.RemoveScaleShear().ExtractRotationQuat()
    local_pos = Gf.Vec3f(
        float(translation[0]),
        float(translation[1]),
        float(translation[2]),
    )
    local_rot = Gf.Quatf(
        float(rotation.GetReal()),
        float(rotation.GetImaginary()[0]),
        float(rotation.GetImaginary()[1]),
        float(rotation.GetImaginary()[2]),
    )
    return local_pos, local_rot


def read_body_mass(stage, prim_path):
    """prim의 명시적 질량을 읽는다. 없으면 None."""
    prim = stage.GetPrimAtPath(prim_path)

    if not prim.IsValid():
        return None

    for candidate in Usd.PrimRange(prim):
        if not candidate.HasAPI(UsdPhysics.MassAPI):
            continue

        value = UsdPhysics.MassAPI(candidate).GetMassAttr().Get()

        if value is not None and value > 0.0:
            return float(value)

    return None


def create_proxy_body(
    stage,
    proxy_path,
    local_pos,
    local_rot,
    mass=0.05,
    inertia=1.0e-5,
):
    """흡착 대상이 붙을 프록시 강체를 만든다.

    질량을 명시하지 않으면 PhysX가 'negative mass'로 경고하고 작은 구로
    근사한다 (콜라이더가 없어 자동 계산도 불가능하다).

    다만 이 프록시는 excludeFromArticulation=True인 FixedJoint로 손목
    링크에 매달린 별도 강체라, 여기 준 질량과 관성이 팔에 그대로
    실린다. 1kg / 1e-3을 줬더니 팔이 고꾸라졌다. 프록시는 그리퍼
    본체를 대신하는 게 아니라 흡착 지점을 표시하는 도우미이고, 실제
    그리퍼 질량은 이미 팔 링크에 들어가 있다. 경고를 없앨 만큼만 작게
    준다.

    mass <= 0이면 MassAPI를 적용하지 않는다 (수정 전 동작).
    """
    proxy_geom = UsdGeom.Cube.Define(stage, proxy_path)
    proxy_geom.CreateSizeAttr(0.01)
    proxy_prim = proxy_geom.GetPrim()

    xformable = UsdGeom.Xformable(proxy_prim)
    xformable.ClearXformOpOrder()
    xformable.AddTranslateOp().Set(Gf.Vec3d(local_pos))
    xformable.AddOrientOp().Set(local_rot)

    UsdGeom.Imageable(proxy_prim).MakeInvisible()
    UsdPhysics.RigidBodyAPI.Apply(proxy_prim)

    if mass > 0.0:
        mass_api = UsdPhysics.MassAPI.Apply(proxy_prim)
        mass_api.CreateMassAttr().Set(float(mass))
        mass_api.CreateDiagonalInertiaAttr().Set(
            Gf.Vec3f(
                float(inertia),
                float(inertia),
                float(inertia),
            )
        )
        mass_api.CreatePrincipalAxesAttr().Set(
            Gf.Quatf(1.0, 0.0, 0.0, 0.0)
        )

    return proxy_prim


def create_weld_joint(stage, joint_path, body0_path, body0_local_pos, body1_path):
    joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
    joint.CreateBody0Rel().SetTargets([body0_path])
    joint.CreateBody1Rel().SetTargets([body1_path])
    joint.CreateLocalPos0Attr().Set(body0_local_pos)
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    joint.CreateJointEnabledAttr().Set(True)
    joint.CreateExcludeFromArticulationAttr().Set(True)
    return joint.GetPrim()


def apply_joint_limit(prim, instance_name, low, high):
    limit_api = UsdPhysics.LimitAPI.Apply(prim, instance_name)
    limit_api.CreateLowAttr().Set(float(low))
    limit_api.CreateHighAttr().Set(float(high))


def apply_joint_drive(prim, instance_name, stiffness, damping, target_position):
    drive_api = UsdPhysics.DriveAPI.Apply(prim, instance_name)
    drive_api.CreateTypeAttr().Set("force")
    drive_api.CreateStiffnessAttr().Set(float(stiffness))
    drive_api.CreateDampingAttr().Set(float(damping))
    drive_api.CreateTargetPositionAttr().Set(float(target_position))
    drive_api.CreateTargetVelocityAttr().Set(0.0)
    drive_api.CreateMaxForceAttr().Set(float("inf"))


def create_attachment_point_joint(stage, joint_path, body0_path):
    joint = UsdPhysics.Joint.Define(
        stage,
        joint_path,
    )
    joint.CreateBody0Rel().SetTargets(
        [body0_path]
    )
    joint.CreateLocalPos0Attr().Set(
        Gf.Vec3f(0.0, 0.0, 0.0)
    )
    joint.CreateLocalRot0Attr().Set(
        Gf.Quatf(1.0, 0.0, 0.0, 0.0)
    )
    joint.CreateLocalPos1Attr().Set(
        Gf.Vec3f(0.0, 0.0, 0.0)
    )
    joint.CreateLocalRot1Attr().Set(
        Gf.Quatf(1.0, 0.0, 0.0, 0.0)
    )
    joint.CreateJointEnabledAttr().Set(True)
    joint.CreateExcludeFromArticulationAttr().Set(True)

    joint_prim = joint.GetPrim()
    robot_schema.ApplyAttachmentPointAPI(joint_prim)
    joint_prim.GetAttribute(
        robot_schema.Attributes.FORWARD_AXIS.name
    ).Set("Z")
    joint_prim.GetAttribute(
        robot_schema.Attributes.CLEARANCE_OFFSET.name
    ).Set(0.008)

    # Matches Isaac Sim's own Surface Gripper example joints exactly:
    # lateral translation locked, approach axis (transZ) and orientation
    # (rotX/Y/Z) spring-driven to hold the gripped object in place.
    apply_joint_limit(joint_prim, "transX", low=1.0, high=-1.0)
    apply_joint_limit(joint_prim, "transY", low=1.0, high=-1.0)
    apply_joint_limit(joint_prim, "transZ", low=0.0, high=0.01)
    apply_joint_limit(joint_prim, "rotX", low=-3.0, high=3.0)
    apply_joint_limit(joint_prim, "rotY", low=-3.0, high=3.0)
    apply_joint_limit(joint_prim, "rotZ", low=-3.0, high=3.0)

    apply_joint_drive(joint_prim, "transZ", stiffness=5000.0, damping=100.0, target_position=0.0)
    apply_joint_drive(joint_prim, "rotX", stiffness=100.0, damping=0.0, target_position=0.0)
    apply_joint_drive(joint_prim, "rotY", stiffness=100.0, damping=0.0, target_position=0.0)
    apply_joint_drive(joint_prim, "rotZ", stiffness=10000.0, damping=0.0, target_position=0.0)

    return joint_prim


def setup_epick_surface_gripper(
    stage,
    suction_cup_prim_path,
    tcp_prim_path,
    max_grip_distance,
    coaxial_force_limit,
    shear_force_limit,
    retry_interval,
    proxy_mass=1.0,
    proxy_inertia=1.0e-3,
    log_prefix="[SUCTION]",
):
    suction_cup_prim = stage.GetPrimAtPath(
        suction_cup_prim_path
    )
    tcp_prim = stage.GetPrimAtPath(
        tcp_prim_path
    )

    if not suction_cup_prim.IsValid():
        raise RuntimeError(
            f"Suction cup prim을 찾지 못했습니다: {suction_cup_prim_path}"
        )
    if not tcp_prim.IsValid():
        raise RuntimeError(
            f"Suction tip prim을 찾지 못했습니다: {tcp_prim_path}"
        )

    # epick_suction_cup is a link inside the arm's articulation. Isaac Sim's
    # Surface Gripper is built/tested against standalone rigid bodies, so
    # attach it to a small proxy body welded to the cup instead of the link
    # itself (positioned at the tip, but keeping the cup's own orientation -
    # already confirmed to point roughly along the approach/-Z direction).
    tip_local_pos, _tip_local_rot = compute_relative_pose(
        stage=stage,
        child_prim_path=tcp_prim_path,
        parent_prim_path=suction_cup_prim_path,
    )
    identity_rot = Gf.Quatf(1.0, 0.0, 0.0, 0.0)

    proxy_path = f"{suction_cup_prim_path}/GripperProxyBody"
    create_proxy_body(
        stage=stage,
        proxy_path=proxy_path,
        local_pos=tip_local_pos,
        local_rot=identity_rot,
        mass=proxy_mass,
        inertia=proxy_inertia,
    )

    weld_joint_path = f"{suction_cup_prim_path}/GripperProxyWeldJoint"
    create_weld_joint(
        stage=stage,
        joint_path=weld_joint_path,
        body0_path=suction_cup_prim_path,
        body0_local_pos=tip_local_pos,
        body1_path=proxy_path,
    )

    joint_path = f"{proxy_path}/SuctionAttachmentJoint"
    joint_prim = create_attachment_point_joint(
        stage=stage,
        joint_path=joint_path,
        body0_path=proxy_path,
    )

    gripper_prim = create_surface_gripper(
        stage,
        proxy_path,
    )
    gripper_prim_path = str(
        gripper_prim.GetPath()
    )

    attachment_points_rel = gripper_prim.GetRelationship(
        robot_schema.Relations.ATTACHMENT_POINTS.name
    )
    attachment_points_rel.SetTargets(
        [joint_prim.GetPath()]
    )

    gripper_prim.GetAttribute(
        robot_schema.Attributes.MAX_GRIP_DISTANCE.name
    ).Set(
        float(max_grip_distance)
    )
    gripper_prim.GetAttribute(
        robot_schema.Attributes.COAXIAL_FORCE_LIMIT.name
    ).Set(
        float(coaxial_force_limit)
    )
    gripper_prim.GetAttribute(
        robot_schema.Attributes.SHEAR_FORCE_LIMIT.name
    ).Set(
        float(shear_force_limit)
    )
    gripper_prim.GetAttribute(
        robot_schema.Attributes.RETRY_INTERVAL.name
    ).Set(
        float(retry_interval)
    )

    print(
        log_prefix,
        "Surface gripper created:",
        gripper_prim_path,
        "attachment_joint=",
        joint_path,
        "max_grip_distance=",
        max_grip_distance,
        "coaxial_force_limit=",
        coaxial_force_limit,
        "shear_force_limit=",
        shear_force_limit,
        "retry_interval=",
        retry_interval,
    )

    return gripper_prim_path


def report_grip_mass_ratio(
    stage,
    payload_prim_path,
    proxy_mass,
    max_ratio=10.0,
    log_prefix="[SUCTION]",
):
    """흡착 물체와 프록시 강체의 질량비를 보고한다.

    그리퍼 설정은 timeline.play() 전에 끝나야 하는데 대상 물체는 그
    뒤에 생성되므로, 진단은 물체가 만들어진 다음에 따로 호출한다.

    PhysX는 조인트 양쪽 질량비가 대략 10:1을 넘으면 풀기 어려워한다.
    이동 중에 물체를 놓치는 전형적인 원인이다.
    """
    payload_mass = read_body_mass(stage, payload_prim_path)

    if payload_mass is None:
        print(
            log_prefix,
            "payload에 명시적 질량이 없습니다 (PhysX가 콜라이더에서",
            "자동 계산). proxy_mass=",
            f"{proxy_mass:.3f}kg.",
            "이동 중 놓치면 BATTERY_SUCTION_PROXY_MASS를 올려보세요.",
        )

        return None

    ratio = payload_mass / max(proxy_mass, 1e-9)
    print(
        log_prefix,
        f"payload_mass={payload_mass:.3f}kg",
        f"proxy_mass={proxy_mass:.3f}kg",
        f"ratio={ratio:.2f}:1",
    )

    if ratio > max_ratio:
        print(
            log_prefix,
            f"참고: 질량비가 {max_ratio:.0f}:1을 넘습니다.",
            "이동 중 물체를 놓친다면 BATTERY_SUCTION_PROXY_MASS를",
            "조금씩 올려보세요. 다만 이 질량은 팔에도 실리므로",
            "크게 올리면 팔이 처집니다.",
        )

    return ratio


class SuctionGripperController:
    def __init__(
        self,
        gripper_prim_path,
        topic_name="/epick/suction_cmd",
        log_prefix="[SUCTION]",
    ):
        if not rclpy.ok():
            rclpy.init(
                args=None
            )

        self.gripper_prim_path = gripper_prim_path
        self.topic_name = topic_name
        self.log_prefix = log_prefix
        self.gripper_interface = (
            surface_gripper.acquire_surface_gripper_interface()
        )
        self.gripper_interface.set_write_to_usd(True)
        self.node = rclpy.create_node(
            "battery_suction_gripper_controller"
        )
        self.subscription = self.node.create_subscription(
            Bool,
            self.topic_name,
            self.handle_command,
            10,
        )

        print(
            self.log_prefix,
            "Suction gripper controller listening on:",
            self.topic_name,
            "gripper=",
            self.gripper_prim_path,
        )

    def handle_command(
        self,
        message,
    ):
        if message.data:
            self.node.get_logger().info(
                "Closing suction gripper."
            )
            self.gripper_interface.close_gripper(
                self.gripper_prim_path
            )
        else:
            self.node.get_logger().info(
                "Opening suction gripper."
            )
            self.gripper_interface.open_gripper(
                self.gripper_prim_path
            )

    def spin_once(
        self,
    ):
        rclpy.spin_once(
            self.node,
            timeout_sec=0.0,
        )

    def destroy(
        self,
    ):
        if self.node is not None:
            self.node.destroy_node()
            self.node = None
