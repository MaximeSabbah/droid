import os
from cv2 import aruco


def _env_or_default(name, default):
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value


# Robot Params #
# Custom same-PC OpenPI deployment defaults.
# None selects RobotEnv's local FrankaRobot path. Set DROID_NUC_IP to use
# the legacy server-backed path as a fallback.
nuc_ip = _env_or_default("DROID_NUC_IP", None)
robot_ip = _env_or_default("DROID_ROBOT_IP", "SET_DROID_ROBOT_IP")
laptop_ip = _env_or_default("DROID_LAPTOP_IP", "127.0.0.1")
sudo_password = _env_or_default("DROID_SUDO_PASSWORD", "")
robot_type = _env_or_default("DROID_ROBOT_TYPE", "panda")  # 'panda' or 'fr3'
robot_serial_number = _env_or_default("DROID_ROBOT_SERIAL_NUMBER", "SET_ROBOT_SERIAL_NUMBER")

# Camera ID's #
# Semantic IDs used by the OpenPI DROID inference path.
hand_camera_id = _env_or_default("DROID_HAND_CAMERA_ID", "d435_color")
varied_camera_1_id = _env_or_default("DROID_VARIED_CAMERA_1_ID", "arducam_left")
varied_camera_2_id = _env_or_default("DROID_VARIED_CAMERA_2_ID", "arducam_right")

# Charuco Board Params #
CHARUCOBOARD_ROWCOUNT = 9
CHARUCOBOARD_COLCOUNT = 14
CHARUCOBOARD_CHECKER_SIZE = 0.020
CHARUCOBOARD_MARKER_SIZE = 0.016
ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_5X5_100)

# Ubuntu Pro Token (RT PATCH) #
ubuntu_pro_token = ""

# Code Version [DONT CHANGE] #
droid_version = "1.3"
