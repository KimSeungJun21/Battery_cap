def create_ros2_joint_graph(
    robot_prim_path="/World/UR10e",
    graph_path="/ActionGraph",
    joint_state_topic="/joint_states",
    joint_command_topic="/joint_command",
):
    import omni.graph.core as og
    import omni.usd

    stage = omni.usd.get_context().get_stage()

    if stage.GetPrimAtPath(graph_path):
        stage.RemovePrim(graph_path)

    og.Controller.edit(
        {"graph_path": graph_path, "evaluator_name": "execution"},
        {
            og.Controller.Keys.CREATE_NODES: [
                ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                ("PublishJointState", "isaacsim.ros2.bridge.ROS2PublishJointState"),
                ("SubscribeJointState", "isaacsim.ros2.bridge.ROS2SubscribeJointState"),
                ("ArticulationController", "isaacsim.core.nodes.IsaacArticulationController"),
            ],
            og.Controller.Keys.CONNECT: [
                ("OnPlaybackTick.outputs:tick", "PublishJointState.inputs:execIn"),
                ("OnPlaybackTick.outputs:tick", "SubscribeJointState.inputs:execIn"),
                ("OnPlaybackTick.outputs:tick", "ArticulationController.inputs:execIn"),

                ("ReadSimTime.outputs:simulationTime", "PublishJointState.inputs:timeStamp"),

                ("SubscribeJointState.outputs:jointNames", "ArticulationController.inputs:jointNames"),
                ("SubscribeJointState.outputs:positionCommand", "ArticulationController.inputs:positionCommand"),
                ("SubscribeJointState.outputs:velocityCommand", "ArticulationController.inputs:velocityCommand"),
                ("SubscribeJointState.outputs:effortCommand", "ArticulationController.inputs:effortCommand"),
            ],
            og.Controller.Keys.SET_VALUES: [
                ("PublishJointState.inputs:targetPrim", robot_prim_path),
                ("PublishJointState.inputs:topicName", joint_state_topic),

                ("SubscribeJointState.inputs:topicName", joint_command_topic),

                ("ArticulationController.inputs:robotPath", robot_prim_path),
            ],
        },
    )

    print("ROS2 ActionGraph created:", graph_path)
