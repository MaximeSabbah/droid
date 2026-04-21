from copy import deepcopy
import os

import cv2

from droid.misc.time import time_ms


resize_func_map = {"cv2": cv2.resize, None: None}

default_devices = {
    "arducam_left": 0,
    "arducam_right": 1,
}


def _env_name(camera_id, suffix):
    return "DROID_" + camera_id.upper() + "_" + suffix


def _coerce_device(value):
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return value


def gather_arducam_cameras(camera_ids=None):
    camera_ids = camera_ids or default_devices.keys()
    return [ArducamCamera(camera_id) for camera_id in camera_ids if camera_id in default_devices]


class ArducamCamera:
    def __init__(self, serial_number):
        if serial_number not in default_devices:
            raise ValueError("Unsupported Arducam id: {0}".format(serial_number))

        self.serial_number = serial_number
        self.high_res_calibration = False
        self.current_mode = "disabled"
        self._cap = None
        self._intrinsics = {}

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
        device=None,
        device_path=None,
        device_index=None,
        backend=None,
        fps=None,
        fourcc=None,
    ):
        self.image = image
        self.depth = depth
        self.pointcloud = pointcloud
        self.concatenate_images = concatenate_images
        self.resizer_resolution = resolution if resize_func is not None else (0, 0)
        self.capture_resolution = resolution if resize_func is None else (0, 0)
        self.resize_func = resize_func_map[resize_func]
        self.device = device if device is not None else device_path
        self.device_index = device_index
        self.backend = backend
        self.fps = fps
        self.fourcc = fourcc

    def set_calibration_mode(self):
        self.set_trajectory_mode()
        self.current_mode = "calibration"

    def set_trajectory_mode(self):
        if self.depth or self.pointcloud:
            raise RuntimeError("{0} only supports RGB image reads in this v1 reader".format(self.serial_number))
        if self.concatenate_images:
            raise RuntimeError("{0} is monocular; set concatenate_images=False".format(self.serial_number))

        self.disable_camera()
        if not self.image:
            self.current_mode = "trajectory"
            return

        device = self._resolve_device()
        api_preference = self._resolve_backend()
        self._cap = cv2.VideoCapture(device, api_preference)
        if not self._cap.isOpened():
            raise RuntimeError(
                "Failed to open {0} at device {1!r}. Override with {2} or {3}.".format(
                    self.serial_number,
                    device,
                    _env_name(self.serial_number, "DEVICE"),
                    _env_name(self.serial_number, "INDEX"),
                )
            )

        self._apply_capture_settings()
        self.current_mode = "trajectory"

    def _resolve_device(self):
        if self.device is not None:
            return _coerce_device(self.device)
        if self.device_index is not None:
            return _coerce_device(self.device_index)

        env_device = os.environ.get(_env_name(self.serial_number, "DEVICE"))
        if env_device:
            return _coerce_device(env_device)

        env_index = os.environ.get(_env_name(self.serial_number, "INDEX"))
        if env_index:
            return _coerce_device(env_index)

        return default_devices[self.serial_number]

    def _resolve_backend(self):
        backend = self.backend or os.environ.get("DROID_OPENCV_VIDEOIO_BACKEND", "CAP_V4L2")
        if not backend:
            return 0
        return getattr(cv2, backend, 0)

    def _apply_capture_settings(self):
        width, height = self.capture_resolution
        if width and height:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if self.fps is not None:
            self._cap.set(cv2.CAP_PROP_FPS, self.fps)
        if self.fourcc is not None:
            fourcc = cv2.VideoWriter_fourcc(*self.fourcc)
            self._cap.set(cv2.CAP_PROP_FOURCC, fourcc)

    def get_intrinsics(self):
        return deepcopy(self._intrinsics)

    def start_recording(self, filename):
        raise NotImplementedError("{0} recording is not implemented for v1 inference".format(self.serial_number))

    def stop_recording(self):
        return None

    def read_camera(self):
        if not self.image:
            return {}, {}
        if self._cap is None or not self._cap.isOpened():
            raise RuntimeError("{0} is not open".format(self.serial_number))

        timestamp_dict = {self.serial_number + "_read_start": time_ms()}
        ok, frame = self._cap.read()
        timestamp_dict[self.serial_number + "_read_end"] = time_ms()
        if not ok or frame is None:
            raise RuntimeError("Failed to read a frame from {0}".format(self.serial_number))

        frame = deepcopy(frame)
        if self.resizer_resolution != (0, 0):
            frame = self.resize_func(frame, self.resizer_resolution)

        timestamp_dict[self.serial_number + "_frame_received"] = timestamp_dict[self.serial_number + "_read_end"]
        return {"image": {self.serial_number + "_left": frame}}, timestamp_dict

    def disable_camera(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self.current_mode = "disabled"

    def is_running(self):
        return self.current_mode != "disabled"
