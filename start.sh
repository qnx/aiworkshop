#!/bin/bash
# Launch the handtrack node with the environment it needs.
#
# Any argument given here is passed straight through to the node, e.g.
#   ./start.sh --ros-args -p camera_unit:=5

# --- Set Environment Variables ---
# These paths are needed for ROS2 to find its libraries
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE}")" && pwd)
export ROS2_NODES_INSTALL="${SCRIPT_DIR}/install"

# --- Sourcing ROS2 ---
# The packaged ROS2 install has the packager's build path baked into its setup
# scripts; without this it aborts with "The build time path ... doesn't exist".
export COLCON_CURRENT_PREFIX=/opt/ros/jazzy

if [ -f /opt/ros/jazzy/setup.bash ]; then
    . /opt/ros/jazzy/setup.bash
else
    echo "Error: ROS2 global setup file not found!"
    exit 1
fi

# Source your workspace's local setup file to find your custom nodes
if [ -f ${ROS2_NODES_INSTALL}/local_setup.bash ]; then
    . ${ROS2_NODES_INSTALL}/local_setup.bash
else
    echo "Error: ROS2 node install not found! ROS2_NODES_INSTALL=${ROS2_NODES_INSTALL}"
    exit 1
fi

export ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET
ros2 run handtrack handtrack_node "$@"
