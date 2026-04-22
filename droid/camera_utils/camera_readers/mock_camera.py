from copy import deepcopy

import numpy as np

from droid.misc.time import time_ms


default_camera_ids = ("arducam_left", "arducam_right", "d435_color")


def gather_mock_cameras(camera_ids=None):
    camera_ids = camera_ids or default_camera_ids
    return [MockCamera(camera_id) for camera_id in camera_ids]


class MockCamera:
    def __init__(self, serial_number):
        self.serial_number = serial_number
        self.high_res_calibration = False
        self.current_mode = "disabled"
        self._intrinsics = {}
        self._frame_index = 0
        self.set_reading_parameters()

    def enable_advanced_calibration(self):
        self.high_res_calibration = True

    def disable_advanced_calibration(self):
        self.high_res_calibration = False

    def set_reading_parameters(
        self,
        image=True,
        depth=False,
        pointcloud=False,
        concatenate_images=False,
        resolution=(0, 0),
        resize_func=None,
        fps=None,
        **kwargs,
    ):
        if depth or pointcloud:
            raise RuntimeError("{0} mock reader only supports RGB image reads".format(self.serial_number))
        if concatenate_images:
            raise RuntimeError("{0} mock reader is monocular; set concatenate_images=False".format(self.serial_number))

        self.image = image
        self.depth = depth
        self.pointcloud = pointcloud
        self.concatenate_images = concatenate_images
        width, height = resolution if resolution != (0, 0) else (640, 480)
        self.width = int(width)
        self.height = int(height)
        self.fps = fps

    def set_calibration_mode(self):
        self.set_trajectory_mode()
        self.current_mode = "calibration"

    def set_trajectory_mode(self):
        self.current_mode = "trajectory"

    def get_intrinsics(self):
        return deepcopy(self._intrinsics)

    def start_recording(self, filename):
        raise NotImplementedError("{0} recording is not implemented for the mock reader".format(self.serial_number))

    def stop_recording(self):
        return None

    def read_camera(self):
        if not self.image:
            return {}, {}
        if not self.is_running():
            raise RuntimeError("{0} is not running".format(self.serial_number))

        read_start = time_ms()
        frame = self._make_frame()
        read_end = time_ms()
        timestamp_dict = {
            self.serial_number + "_read_start": read_start,
            self.serial_number + "_read_end": read_end,
            self.serial_number + "_frame_received": read_end,
        }
        return {"image": {self.serial_number + "_left": frame}}, timestamp_dict

    def _make_frame(self):
        self._frame_index += 1
        x = np.linspace(0, 255, self.width, dtype=np.uint8)
        y = np.linspace(0, 255, self.height, dtype=np.uint8)
        xx = np.tile(x[None, :], (self.height, 1))
        yy = np.tile(y[:, None], (1, self.width))
        offset = (sum(ord(char) for char in self.serial_number) + self._frame_index) % 255

        # OpenCV hardware readers return BGR, so the mock follows that convention.
        blue = ((xx.astype(np.uint16) + offset) % 256).astype(np.uint8)
        green = ((yy.astype(np.uint16) + 2 * offset) % 256).astype(np.uint8)
        red_values = (xx.astype(np.uint16) // 2) + (yy.astype(np.uint16) // 2) + 3 * offset
        red = (red_values % 256).astype(np.uint8)
        return np.stack([blue, green, red], axis=-1)

    def disable_camera(self):
        self.current_mode = "disabled"

    def is_running(self):
        return self.current_mode != "disabled"
