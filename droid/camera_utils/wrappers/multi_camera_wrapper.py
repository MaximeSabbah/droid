import os
import random
from collections import defaultdict

from droid.camera_utils.info import get_camera_type
from droid.misc.parameters import hand_camera_id, varied_camera_1_id, varied_camera_2_id


class MultiCameraWrapper:
    custom_camera_ids = {"arducam_left", "arducam_right", "d435_color"}

    def __init__(self, camera_kwargs=None):
        camera_kwargs = dict(camera_kwargs or {})
        camera_backend = camera_kwargs.pop("camera_backend", os.environ.get("DROID_CAMERA_BACKEND", "auto"))
        camera_backend = (camera_backend or "auto").lower()
        requested_camera_ids = camera_kwargs.pop(
            "camera_ids", [hand_camera_id, varied_camera_1_id, varied_camera_2_id]
        )
        if isinstance(requested_camera_ids, str):
            requested_camera_ids = [requested_camera_ids]
        requested_camera_ids = [cam_id for cam_id in requested_camera_ids if cam_id]

        # Open Cameras #
        cameras = self._gather_cameras(camera_backend, requested_camera_ids)
        self.camera_dict = {cam.serial_number: cam for cam in cameras}

        # Set Correct Parameters #
        for cam_id in self.camera_dict.keys():
            curr_cam_kwargs = self._get_camera_kwargs(cam_id, camera_kwargs)
            self.camera_dict[cam_id].set_reading_parameters(**curr_cam_kwargs)

        # Launch Camera #
        self.set_trajectory_mode()

    def _gather_cameras(self, camera_backend, requested_camera_ids):
        if camera_backend == "auto":
            use_custom = any(cam_id in self.custom_camera_ids for cam_id in requested_camera_ids)
            camera_backend = "openpi" if use_custom else "zed"

        if camera_backend == "zed":
            return self._gather_zed_cameras()
        if camera_backend in ("mock", "fake", "synthetic"):
            from droid.camera_utils.camera_readers.mock_camera import gather_mock_cameras

            return gather_mock_cameras(requested_camera_ids)
        if camera_backend in ("openpi", "custom", "opencv_realsense"):
            from droid.camera_utils.camera_readers.arducam_camera import gather_arducam_cameras
            from droid.camera_utils.camera_readers.realsense_camera import gather_realsense_cameras

            cameras = []
            cameras.extend(gather_arducam_cameras(requested_camera_ids))
            cameras.extend(gather_realsense_cameras(requested_camera_ids))
            if not cameras:
                raise RuntimeError(
                    "No OpenPI cameras configured. Expected one or more of: {0}".format(
                        sorted(self.custom_camera_ids)
                    )
                )
            return cameras

        raise ValueError("Unsupported DROID camera backend: {0}".format(camera_backend))

    def _gather_zed_cameras(self):
        try:
            from droid.camera_utils.camera_readers.zed_camera import gather_zed_cameras
        except Exception as exc:
            raise RuntimeError(
                "Failed to import the ZED camera reader. Set DROID_CAMERA_BACKEND=openpi "
                "for Arducam/D435 deployment, or install the ZED SDK for the upstream path."
            ) from exc
        return gather_zed_cameras()

    def _get_camera_kwargs(self, cam_id, camera_kwargs):
        curr_cam_kwargs = {}
        default_kwargs = camera_kwargs.get("default", {})
        cam_type = get_camera_type(cam_id)
        type_kwargs = camera_kwargs.get(cam_type, {}) if cam_type is not None else {}
        id_kwargs = camera_kwargs.get(cam_id, {})
        curr_cam_kwargs.update(default_kwargs)
        curr_cam_kwargs.update(type_kwargs)
        curr_cam_kwargs.update(id_kwargs)
        return curr_cam_kwargs

    ### Calibration Functions ###
    def get_camera(self, camera_id):
        return self.camera_dict[camera_id]

    def enable_advanced_calibration(self):
        for cam in self.camera_dict.values():
            cam.enable_advanced_calibration()

    def disable_advanced_calibration(self):
        for cam in self.camera_dict.values():
            cam.disable_advanced_calibration()

    def set_calibration_mode(self, cam_id):
        # If High Res Calibration, Only One Can Run #
        close_all = any([cam.high_res_calibration for cam in self.camera_dict.values()])

        if close_all:
            for curr_cam_id in self.camera_dict:
                if curr_cam_id != cam_id:
                    self.camera_dict[curr_cam_id].disable_camera()

        self.camera_dict[cam_id].set_calibration_mode()

    def set_trajectory_mode(self):
        # If High Res Calibration, Close All #
        close_all = any(
            [cam.high_res_calibration and cam.current_mode == "calibration" for cam in self.camera_dict.values()]
        )

        if close_all:
            for cam in self.camera_dict.values():
                cam.disable_camera()

        # Put All Cameras In Trajectory Mode #
        for cam in self.camera_dict.values():
            cam.set_trajectory_mode()

    ### Data Storing Functions ###
    def start_recording(self, recording_folderpath):
        subdir = os.path.join(recording_folderpath, "SVO")
        if not os.path.isdir(subdir):
            os.makedirs(subdir)
        for cam in self.camera_dict.values():
            filepath = os.path.join(subdir, cam.serial_number + ".svo")
            cam.start_recording(filepath)

    def stop_recording(self):
        for cam in self.camera_dict.values():
            cam.stop_recording()

    ### Basic Camera Functions ###
    def read_cameras(self):
        full_obs_dict = defaultdict(dict)
        full_timestamp_dict = {}

        # Read Cameras In Randomized Order #
        all_cam_ids = list(self.camera_dict.keys())
        random.shuffle(all_cam_ids)

        for cam_id in all_cam_ids:
            if not self.camera_dict[cam_id].is_running():
                continue
            data_dict, timestamp_dict = self.camera_dict[cam_id].read_camera()

            for key in data_dict:
                full_obs_dict[key].update(data_dict[key])
            full_timestamp_dict.update(timestamp_dict)

        return full_obs_dict, full_timestamp_dict

    def disable_cameras(self):
        for camera in self.camera_dict.values():
            camera.disable_camera()
