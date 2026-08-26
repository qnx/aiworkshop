#!/bin/bash

set -e
trap 'last_command=$current_command; current_command=$BASH_COMMAND' DEBUG

# --- ROS2 Installation Discovery ---
if [ -f "/opt/ros/jazzy/local_setup.bash" ]; then
    ROS2_HOST_INSTALLATION_PATH=/opt/ros/jazzy
    echo "Found ROS2 Installation in $ROS2_HOST_INSTALLATION_PATH"
else
    echo "Failed to find ROS2 in expected locations, please run ./configure.sh"
    exit 1
fi

# Source the ROS2 environment for cross-compilation
. "${ROS2_HOST_INSTALLATION_PATH}/local_setup.bash"

echo "--- ROS Environment Variables ---"
printenv | grep "ROS"
echo "-------------------------------"

# --- Build Execution ---
colcon build --merge-install --cmake-force-configure

rc=$?
if [ $rc -eq 0 ]; then
    echo "Success"
else
    echo "Error: $rc"
    exit $rc
fi

echo " "
