from copy import deepcopy
import os
import time

import cv2
import numpy as np

from droid.misc.time import time_ms


resize_func_map = {"cv2": cv2.resize, None: None}
default_camera_id = "d435_color"


def gather_realsense_cameras(camera_ids=None):
    camera_ids = camera_ids or [default_camera_id]
    return [RealSenseColorCamera(camera_id) for camera_id in camera_ids if camera_id == default_camera_id]


class RealSenseColorCamera:
    def __init__(self, serial_number=default_camera_id):
        if serial_number != default_camera_id:
            raise ValueError("Unsupported RealSense id: {0}".format(serial_number))

        self.serial_number = serial_number
        self.high_res_calibration = False
        self.current_mode = "disabled"
        self._pipeline = None
        self._profile = None
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
        serial=None,
        fps=None,
        timeout_ms=5000,
    ):
        self.image = image
        self.depth = depth
        self.pointcloud = pointcloud
        self.concatenate_images = concatenate_images
        self.resizer_resolution = resolution if resize_func is not None else (0, 0)
        self.capture_resolution = resolution if resize_func is None else (0, 0)
        self.resize_func = resize_func_map[resize_func]
        self.device_serial = serial
        self.fps = fps
        self.timeout_ms = timeout_ms

    def set_calibration_mode(self):
        self.set_trajectory_mode()
        self.current_mode = "calibration"

    def set_trajectory_mode(self):
        if self.depth or self.pointcloud:
            raise RuntimeError("{0} only exposes the D435 color stream in this v1 reader".format(self.serial_number))
        if self.concatenate_images:
            raise RuntimeError("{0} is monocular; set concatenate_images=False".format(self.serial_number))

        self.disable_camera()
        if not self.image:
            self.current_mode = "trajectory"
            return

        try:
            import pyrealsense2 as rs
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "pyrealsense2 is required for {0}. Install librealsense/pyrealsense2 or use another camera backend.".format(
                    self.serial_number
                )
            ) from exc

        width, height = self.capture_resolution
        width = width or int(os.environ.get("DROID_D435_COLOR_WIDTH", "640"))
        height = height or int(os.environ.get("DROID_D435_COLOR_HEIGHT", "480"))
        fps = self.fps or int(os.environ.get("DROID_D435_COLOR_FPS", "30"))
        device_serial = self.device_serial or os.environ.get("DROID_D435_SERIAL")

        config = rs.config()
        if device_serial:
            config.enable_device(device_serial)
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

        retries = int(os.environ.get("DROID_D435_OPEN_RETRIES", "3"))
        retry_delay_s = float(os.environ.get("DROID_D435_OPEN_RETRY_DELAY_S", "0.5"))
        last_exc = None
        for attempt in range(max(1, retries)):
            self._pipeline = rs.pipeline()
            try:
                self._profile = self._pipeline.start(config)
                break
            except RuntimeError as exc:
                last_exc = exc
                self.disable_camera()
                if attempt + 1 < retries:
                    time.sleep(retry_delay_s)
                    continue
                raise RuntimeError(
                    "Failed to start {0} color stream at {1}x{2}@{3}. "
                    "If this says the device is busy, make sure the external camera path "
                    "does not point at a RealSense /dev/video node; prefer /dev/v4l/by-id symlinks.".format(
                        self.serial_number,
                        width,
                        height,
                        fps,
                    )
                ) from last_exc
        self._save_intrinsics(rs)
        self.current_mode = "trajectory"

    def _save_intrinsics(self, rs):
        color_profile = self._profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = color_profile.get_intrinsics()
        self._intrinsics = {
            self.serial_number + "_left": {
                "cameraMatrix": np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]]),
                "distCoeffs": np.array(intr.coeffs),
            }
        }

    def get_intrinsics(self):
        return deepcopy(self._intrinsics)

    def start_recording(self, filename):
        raise NotImplementedError("{0} recording is not implemented for v1 inference".format(self.serial_number))

    def stop_recording(self):
        return None

    def read_camera(self):
        if not self.image:
            return {}, {}
        if self._pipeline is None:
            raise RuntimeError("{0} is not open".format(self.serial_number))

        timestamp_dict = {self.serial_number + "_read_start": time_ms()}
        frames = self._pipeline.wait_for_frames(self.timeout_ms)
        color_frame = frames.get_color_frame()
        timestamp_dict[self.serial_number + "_read_end"] = time_ms()
        if not color_frame:
            raise RuntimeError("Failed to read a color frame from {0}".format(self.serial_number))

        frame = np.asanyarray(color_frame.get_data()).copy()
        if self.resizer_resolution != (0, 0):
            frame = self.resize_func(frame, self.resizer_resolution)

        timestamp_dict[self.serial_number + "_frame_received"] = color_frame.get_timestamp()
        return {"image": {self.serial_number + "_left": frame}}, timestamp_dict

    def disable_camera(self):
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except RuntimeError:
                pass
            self._pipeline = None
            self._profile = None
        self.current_mode = "disabled"

    def is_running(self):
        return self.current_mode != "disabled"
