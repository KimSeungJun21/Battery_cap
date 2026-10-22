cmake_minimum_required(VERSION 3.22)
project(rbpodo_moveit_config)

find_package(ament_cmake REQUIRED)

ament_package()

install(
  PROGRAMS
    scripts/isaac_joint_trajectory_bridge.py
    scripts/suction_batterycap.py
    scripts/suction_mustard_bottle.py
    scripts/nbv_arm_move_relay.py
  DESTINATION lib/${PROJECT_NAME})
install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
  PATTERN "setup_assistant.launch" EXCLUDE)
install(DIRECTORY config DESTINATION share/${PROJECT_NAME})
