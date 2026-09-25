#!/usr/bin/env bash
set -Eeuo pipefail

WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROS_DISTRO="${ROS_DISTRO:-jazzy}"
ROS_SETUP="/opt/ros/${ROS_DISTRO}/setup.bash"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"

if [[ "${EUID}" -eq 0 ]]; then
  SUDO=""
else
  SUDO="sudo"
fi

if [[ ! -f "${ROS_SETUP}" ]]; then
  printf 'ROS %s is required at %s.\n' "${ROS_DISTRO}" "${ROS_SETUP}" >&2
  printf 'Install ROS 2 %s first: https://docs.ros.org/en/%s/Installation.html\n' \
    "${ROS_DISTRO}" "${ROS_DISTRO}" >&2
  exit 1
fi

if ! command -v apt-get >/dev/null 2>&1; then
  echo 'This installer supports Debian/Ubuntu systems with apt-get.' >&2
  exit 1
fi

echo "Installing system and ROS dependencies..."
${SUDO} apt-get update
${SUDO} apt-get install -y \
  build-essential \
  python3-pip \
  python3-numpy \
  python3-opencv \
  ros-"${ROS_DISTRO}"-ament-cmake \
  ros-"${ROS_DISTRO}"-colcon-common-extensions \
  ros-"${ROS_DISTRO}"-launch \
  ros-"${ROS_DISTRO}"-launch-ros \
  ros-"${ROS_DISTRO}"-ros-gz-bridge \
  ros-"${ROS_DISTRO}"-ros-gz-image \
  ros-"${ROS_DISTRO}"-rqt-image-view \
  ros-"${ROS_DISTRO}"-sensor-msgs

if ! command -v MicroXRCEAgent >/dev/null 2>&1; then
  echo "MicroXRCEAgent was not found."
  echo "Install/build Micro-XRCE-DDS-Agent separately, then rerun this script."
fi

if ! command -v colcon >/dev/null 2>&1; then
  echo "colcon is unavailable after dependency installation." >&2
  exit 1
fi

echo "Installing Ultralytics for ${PYTHON_BIN}..."
"${PYTHON_BIN}" -m pip install --user --break-system-packages --upgrade ultralytics

source "${ROS_SETUP}"
cd "${WORKSPACE}"

echo "Checking the bundled YOLO model..."
"${PYTHON_BIN}" - <<'PY'
from pathlib import Path

from ultralytics import YOLO

model_path = Path("src/uav_vision_follow/models/drone_yolo.pt")
if not model_path.is_file():
    raise SystemExit(f"Missing model: {model_path}")

model = YOLO(str(model_path))
names = {str(name).lower() for name in model.names.values()}
if "drone" not in names and not {"uav", "quadcopter"}.intersection(names):
    raise SystemExit(f"Model has no drone class: {sorted(names)}")
print(f"Model classes: {sorted(names)}")
PY

echo "Building uav_vision_follow..."
colcon build \
  --packages-select uav_vision_follow \
  --cmake-args -DPython3_EXECUTABLE="${PYTHON_BIN}"
source install/setup.bash

cat <<EOF

Installation complete.

Source the workspace:
  source ${WORKSPACE}/install/setup.bash

Start the full tmuxinator simulation:
  tmuxinator start -p ${WORKSPACE}/src/uav_vision_follow/tmuxinator/uav_vision_follow.yml

Or start the vision node after PX4, XRCE-DDS, and camera bridges are running:
  ros2 launch uav_vision_follow vision_interceptor.launch.py

View the annotated detections:
  ros2 run rqt_image_view rqt_image_view /uav_2/camera/detections
EOF
