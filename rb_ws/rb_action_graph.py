def print_ros2_joint_graph_summary(
    stage,
    graph_path,
):
    graph_prim = stage.GetPrimAtPath(graph_path)

    if not graph_prim.IsValid():
        print("ROS2 Action Graph missing:", graph_path)
        return

    children = list(graph_prim.GetChildren())

    print("ROS2 Action Graph path:", graph_path)

    if not children:
        print("ROS2 Action Graph nodes: no visible USD children")
        return

    print("ROS2 Action Graph nodes:")

    for child in children:
        print(
            "  -",
            child.GetPath(),
            child.GetTypeName(),
        )


def create_ros2_joint_graph(
    robot_prim_path="/rb5_850e/Geometry",
    graph_path="/ActionGraph",
    joint_state_topic="/isaac_joint_states",
    joint_command_topic="/isaac_joint_commands",
):
    import omni.graph.core as og
    import omni.usd
    import usdrt

    stage = omni.usd.get_context().get_stage()

    # 로봇 Prim 경로 확인
    robot_prim = stage.GetPrimAtPath(robot_prim_path)

    if not robot_prim.IsValid():
        raise RuntimeError(
            f"Robot articulation prim을 찾지 못했습니다: "
            f"{robot_prim_path}"
        )

    print("Using robot articulation:", robot_prim_path)

    # 기존 Action Graph 제거
    graph_prim = stage.GetPrimAtPath(graph_path)

    if graph_prim.IsValid():
        stage.RemovePrim(graph_path)
        print("Removed existing graph:", graph_path)

    # Action Graph 생성
    og.Controller.edit(
        {
            "graph_path": graph_path,
            "evaluator_name": "execution",
        },
        {
            og.Controller.Keys.CREATE_NODES: [
                (
                    "OnPlaybackTick",
                    "omni.graph.action.OnPlaybackTick",
                ),
                (
                    "ReadSimTime",
                    "isaacsim.core.nodes.IsaacReadSimulationTime",
                ),
                (
                    "ReadJointState",
                    "isaacsim.sensors.physics.IsaacReadJointState",
                ),
                (
                    "Context",
                    "isaacsim.ros2.bridge.ROS2Context",
                ),
                (
                    "PublishJointState",
                    "isaacsim.ros2.bridge.ROS2PublishJointState",
                ),
                (
                    "SubscribeJointState",
                    "isaacsim.ros2.bridge.ROS2SubscribeJointState",
                ),
                (
                    "ArticulationController",
                    "isaacsim.core.nodes.IsaacArticulationController",
                ),
                (
                    "PublishClock",
                    "isaacsim.ros2.bridge.ROS2PublishClock",
                ),
            ],

            og.Controller.Keys.CONNECT: [
                (
                    "OnPlaybackTick.outputs:tick",
                    "ReadJointState.inputs:execIn",
                ),
                (
                    "ReadJointState.outputs:execOut",
                    "PublishJointState.inputs:execIn",
                ),
                (
                    "ReadJointState.outputs:jointNames",
                    "PublishJointState.inputs:jointNames",
                ),
                (
                    "ReadJointState.outputs:jointPositions",
                    "PublishJointState.inputs:jointPositions",
                ),
                (
                    "ReadJointState.outputs:jointVelocities",
                    "PublishJointState.inputs:jointVelocities",
                ),
                (
                    "ReadJointState.outputs:jointEfforts",
                    "PublishJointState.inputs:jointEfforts",
                ),
                (
                    "ReadJointState.outputs:jointDofTypes",
                    "PublishJointState.inputs:jointDofTypes",
                ),
                (
                    "ReadJointState.outputs:stageMetersPerUnit",
                    "PublishJointState.inputs:stageMetersPerUnit",
                ),
                (
                    "ReadJointState.outputs:sensorTime",
                    "PublishJointState.inputs:sensorTime",
                ),
                (
                    "OnPlaybackTick.outputs:tick",
                    "SubscribeJointState.inputs:execIn",
                ),
                (
                    "OnPlaybackTick.outputs:tick",
                    "PublishClock.inputs:execIn",
                ),
                (
                    "OnPlaybackTick.outputs:tick",
                    "ArticulationController.inputs:execIn",
                ),
                (
                    "Context.outputs:context",
                    "PublishJointState.inputs:context",
                ),
                (
                    "Context.outputs:context",
                    "SubscribeJointState.inputs:context",
                ),
                (
                    "Context.outputs:context",
                    "PublishClock.inputs:context",
                ),
                (
                    "ReadSimTime.outputs:simulationTime",
                    "PublishClock.inputs:timeStamp",
                ),
                (
                    "SubscribeJointState.outputs:jointNames",
                    "ArticulationController.inputs:jointNames",
                ),
                (
                    "SubscribeJointState.outputs:positionCommand",
                    "ArticulationController.inputs:positionCommand",
                ),
                (
                    "SubscribeJointState.outputs:velocityCommand",
                    "ArticulationController.inputs:velocityCommand",
                ),
                (
                    "SubscribeJointState.outputs:effortCommand",
                    "ArticulationController.inputs:effortCommand",
                ),
            ],

            og.Controller.Keys.SET_VALUES: [
                (
                    "ReadJointState.inputs:prim",
                    [usdrt.Sdf.Path(robot_prim_path)],
                ),
                (
                    "PublishJointState.inputs:topicName",
                    joint_state_topic,
                ),
                (
                    "SubscribeJointState.inputs:topicName",
                    joint_command_topic,
                ),
                (
                    "ArticulationController.inputs:robotPath",
                    robot_prim_path,
                ),
                (
                    "ArticulationController.inputs:targetPrim",
                    [usdrt.Sdf.Path(robot_prim_path)],
                ),
            ],
        },
    )

    print("ROS2 Action Graph created:", graph_path)
    print("Publish topic:", joint_state_topic)
    print("Subscribe topic:", joint_command_topic)
    print_ros2_joint_graph_summary(
        stage=stage,
        graph_path=graph_path,
    )
