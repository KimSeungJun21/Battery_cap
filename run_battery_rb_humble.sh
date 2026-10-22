#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_ROOT="${WORKSPACE_ROOT:-/home/seungjun/isaac_ws}"
ISAACSIM_ROOT="${ISAACSIM_ROOT:-/home/seungjun/isaacsim}"
ISAAC_ROS_DISTRO="${BATTERY_ISAAC_ROS_DISTRO:-humble}"
ISAAC_ROS2_CORE_EXT="${ISAACSIM_ROOT}/exts/isaacsim.ros2.core"
ISAAC_ROS2_LIB_DIR="${ISAAC_ROS2_CORE_EXT}/${ISAAC_ROS_DISTRO}/lib"
ISAAC_ROS2_RCLPY_DIR="${ISAAC_ROS2_CORE_EXT}/${ISAAC_ROS_DISTRO}/rclpy"
BATTERY_SCRIPT="${BATTERY_SCRIPT:-${WORKSPACE_ROOT}/rb_ws/Battery_rb.py}"

filter_path_var() {
    local var_name="$1"
    shift

    local current_value="${!var_name:-}"
    local filtered_value=""
    local entry
    local pattern
    local skip

    IFS=":" read -r -a entries <<< "${current_value}"

    for entry in "${entries[@]}"; do
        if [[ -z "${entry}" ]]; then
            continue
        fi

        skip=0

        for pattern in "$@"; do
            case "${entry}" in
                ${pattern})
                    skip=1
                    break
                    ;;
            esac
        done

        if [[ "${skip}" -eq 0 ]]; then
            if [[ -z "${filtered_value}" ]]; then
                filtered_value="${entry}"
            else
                filtered_value="${filtered_value}:${entry}"
            fi
        fi
    done

    printf -v "${var_name}" "%s" "${filtered_value}"
    export "${var_name}"
}

if [[ ! -x "${ISAACSIM_ROOT}/python.sh" ]]; then
    echo "[ERROR] Isaac Sim python.sh not found: ${ISAACSIM_ROOT}/python.sh" >&2
    exit 1
fi

if [[ ! -d "${ISAAC_ROS2_LIB_DIR}" ]]; then
    echo "[ERROR] Isaac Sim internal ROS2 libs not found: ${ISAAC_ROS2_LIB_DIR}" >&2
    exit 1
fi

if [[ ! -d "${ISAAC_ROS2_RCLPY_DIR}" ]]; then
    echo "[ERROR] Isaac Sim internal rclpy not found: ${ISAAC_ROS2_RCLPY_DIR}" >&2
    exit 1
fi

if [[ ! -f "${BATTERY_SCRIPT}" ]]; then
    echo "[ERROR] Battery script not found: ${BATTERY_SCRIPT}" >&2
    exit 1
fi

filter_path_var LD_LIBRARY_PATH \
    "/opt/ros/jazzy*" \
    "${WORKSPACE_ROOT}/rb_moveit_jazzy_ws*"

filter_path_var PYTHONPATH \
    "/opt/ros/jazzy*" \
    "${WORKSPACE_ROOT}/rb_moveit_jazzy_ws*"

filter_path_var AMENT_PREFIX_PATH \
    "/opt/ros/jazzy*" \
    "${WORKSPACE_ROOT}/rb_moveit_jazzy_ws*"

filter_path_var CMAKE_PREFIX_PATH \
    "/opt/ros/jazzy*" \
    "${WORKSPACE_ROOT}/rb_moveit_jazzy_ws*"

unset ROS_LOCALHOST_ONLY
unset ROS_PACKAGE_PATH
unset COLCON_PREFIX_PATH

export ROS_DISTRO="${ISAAC_ROS_DISTRO}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
export ROS_DOMAIN_ID="${BATTERY_ROS_DOMAIN_ID:-42}"
export ROS_AUTOMATIC_DISCOVERY_RANGE="${ROS_AUTOMATIC_DISCOVERY_RANGE:-SUBNET}"
export LD_LIBRARY_PATH="${ISAAC_ROS2_LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
# jazzy was filtered out of PYTHONPATH above, so rclpy has to come from the
# matching Isaac-internal ROS2 distro or the script dies on "import rclpy".
export PYTHONPATH="${ISAAC_ROS2_RCLPY_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

echo "[ISAAC ROS] ROS_DISTRO=${ROS_DISTRO}"
echo "[ISAAC ROS] RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION}"
echo "[ISAAC ROS] ROS_DOMAIN_ID=${ROS_DOMAIN_ID}"
echo "[ISAAC ROS] ROS_AUTOMATIC_DISCOVERY_RANGE=${ROS_AUTOMATIC_DISCOVERY_RANGE}"
echo "[ISAAC ROS] ROS_LOCALHOST_ONLY=${ROS_LOCALHOST_ONLY:-<unset>}"
echo "[ISAAC ROS] LD_LIBRARY_PATH starts with ${ISAAC_ROS2_LIB_DIR}"
echo "[ISAAC ROS] PYTHONPATH starts with ${ISAAC_ROS2_RCLPY_DIR}"

exec "${ISAACSIM_ROOT}/python.sh" "${BATTERY_SCRIPT}" "$@"
