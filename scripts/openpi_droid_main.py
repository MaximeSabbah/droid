import argparse
import contextlib
import os
import signal
import time

import numpy as np
from PIL import Image


DROID_CONTROL_FREQUENCY = 15
EXPECTED_ACTION_DIM = 8


@contextlib.contextmanager
def prevent_keyboard_interrupt():
    interrupted = False
    original_handler = signal.getsignal(signal.SIGINT)

    def handler(signum, frame):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, original_handler)
        if interrupted:
            raise KeyboardInterrupt


def parse_args():
    parser = argparse.ArgumentParser(description="Run OpenPI DROID inference with Arducam + D435 cameras.")
    parser.add_argument("--left_camera_id", default=os.environ.get("DROID_VARIED_CAMERA_1_ID", "arducam_left"))
    parser.add_argument("--right_camera_id", default=os.environ.get("DROID_VARIED_CAMERA_2_ID", ""))
    parser.add_argument("--wrist_camera_id", default=os.environ.get("DROID_HAND_CAMERA_ID", "d435_color"))
    parser.add_argument("--external_camera", choices=["left", "right"], default="left")
    parser.add_argument("--camera_backend", default=os.environ.get("DROID_CAMERA_BACKEND", "openpi"))
    parser.add_argument("--left_camera_device", default=os.environ.get("DROID_ARDUCAM_LEFT_DEVICE"))
    parser.add_argument("--right_camera_device", default=os.environ.get("DROID_ARDUCAM_RIGHT_DEVICE"))
    parser.add_argument("--d435_serial", default=os.environ.get("DROID_D435_SERIAL"))
    parser.add_argument("--camera_width", type=int, default=0)
    parser.add_argument("--camera_height", type=int, default=0)
    parser.add_argument("--camera_fps", type=int, default=30)
    parser.add_argument("--mock_cameras", action="store_true")
    parser.add_argument("--remote_host", default=os.environ.get("OPENPI_HOST", "127.0.0.1"))
    parser.add_argument("--remote_port", type=int, default=int(os.environ.get("OPENPI_PORT", "8000")))
    parser.add_argument("--mock_policy", action="store_true")
    parser.add_argument("--mock_policy_bad_shape", action="store_true")
    parser.add_argument("--prompt", default=os.environ.get("OPENPI_PROMPT"))
    parser.add_argument("--max_timesteps", type=int, default=1)
    parser.add_argument("--open_loop_horizon", type=int, default=8)
    parser.add_argument("--dry-run", dest="dry_run", action="store_true", default=True)
    parser.add_argument("--execute", dest="dry_run", action="store_false")
    parser.add_argument("--simulation", action="store_true")
    parser.add_argument("--mock_robot_state", action="store_true")
    parser.add_argument("--no_launch_robot", action="store_true")
    parser.add_argument("--no_reset", action="store_true")
    parser.add_argument("--save_preview", default="robot_camera_views.png")
    parser.add_argument("--no_bgr_to_rgb", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.dry_run and args.mock_robot_state:
        raise ValueError("--mock_robot_state cannot be used with --execute")
    if not args.dry_run and not args.simulation and (args.mock_policy or args.mock_policy_bad_shape or args.mock_cameras):
        raise ValueError("Mock policy/camera options are dry-run only")
    if args.external_camera == "right" and not args.right_camera_id:
        raise ValueError("--external_camera=right requires --right_camera_id")
    if not args.dry_run:
        require_motion_guards(args)

    policy_client = make_policy_client(args)
    observation_source = make_observation_source(args)
    instruction = args.prompt or input("Enter instruction: ")

    actions_from_chunk_completed = 0
    pred_action_chunk = None

    try:
        for t_step in range(args.max_timesteps):
            start_time = time.time()
            obs_dict = observation_source.get_observation()
            curr_obs = extract_observation(args, obs_dict, save_to_disk=(t_step == 0))

            if (
                pred_action_chunk is None
                or actions_from_chunk_completed >= args.open_loop_horizon
                or actions_from_chunk_completed >= pred_action_chunk.shape[0]
            ):
                actions_from_chunk_completed = 0
                request_data = make_policy_request(curr_obs, args.external_camera, instruction)
                validate_policy_request(request_data)
                with prevent_keyboard_interrupt():
                    pred_action_chunk = np.asarray(policy_client.infer(request_data)["actions"])
                validate_action_chunk(pred_action_chunk)

            action = process_action(pred_action_chunk[actions_from_chunk_completed])
            actions_from_chunk_completed += 1

            if args.dry_run:
                print_dry_run_step(t_step, action, pred_action_chunk, curr_obs)
            else:
                observation_source.step(action)

            elapsed_time = time.time() - start_time
            sleep_time = (1 / DROID_CONTROL_FREQUENCY) - elapsed_time
            if sleep_time > 0:
                time.sleep(sleep_time)
    finally:
        observation_source.close()


def require_motion_guards(args):
    if args.simulation:
        if not args.no_launch_robot:
            raise RuntimeError(
                "Simulation execution requires --no_launch_robot and a separately launched Polymetis sim server."
            )
        required = {
            "DROID_ENABLE_SIM_MOTION": "1",
            "DROID_CONFIRM_SIMULATION": "1",
        }
        missing = [name for name, value in required.items() if os.environ.get(name) != value]
        if missing:
            raise RuntimeError(
                "Refusing to execute simulation motion. Set these environment variables first: {0}".format(
                    ", ".join("{0}=1".format(name) for name in missing)
                )
            )
        return

    required = {
        "DROID_ENABLE_ROBOT_MOTION": "1",
        "CONFIRM_REAL_ROBOT": "1",
    }
    missing = [name for name, value in required.items() if os.environ.get(name) != value]
    if missing:
        raise RuntimeError(
            "Refusing to execute robot motion. Set these environment variables first: {0}".format(
                ", ".join("{0}=1".format(name) for name in missing)
            )
        )


def make_policy_client(args):
    if args.mock_policy or args.mock_policy_bad_shape:
        return MockPolicyClient(bad_shape=args.mock_policy_bad_shape)

    try:
        from openpi_client import websocket_client_policy
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "openpi-client is required. Install /workspace/openpi/packages/openpi-client in the DROID env."
        ) from exc
    return websocket_client_policy.WebsocketClientPolicy(args.remote_host, args.remote_port)


class MockPolicyClient:
    def __init__(self, bad_shape=False):
        self.bad_shape = bad_shape

    def infer(self, request_data):
        shape = (10, 7) if self.bad_shape else (10, EXPECTED_ACTION_DIM)
        actions = np.zeros(shape, dtype=np.float32)
        if not self.bad_shape:
            actions[:, :7] = np.linspace(-0.15, 0.15, shape[0], dtype=np.float32)[:, None]
            actions[:, 7] = np.linspace(0.0, 1.0, shape[0], dtype=np.float32)
        return {"actions": actions}


def make_policy_request(curr_obs, external_camera, instruction):
    try:
        from openpi_client import image_tools
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "openpi-client is required. Install /workspace/openpi/packages/openpi-client in the DROID env."
        ) from exc

    external_image_key = "{0}_image".format(external_camera)
    if curr_obs.get(external_image_key) is None:
        raise ValueError("Requested external camera {0!r}, but no image is available".format(external_camera))

    return {
        "observation/exterior_image_1_left": image_tools.resize_with_pad(
            curr_obs[external_image_key], 224, 224
        ),
        "observation/wrist_image_left": image_tools.resize_with_pad(curr_obs["wrist_image"], 224, 224),
        "observation/joint_position": curr_obs["joint_position"],
        "observation/gripper_position": curr_obs["gripper_position"],
        "prompt": instruction,
    }


def validate_policy_request(request_data):
    required_keys = {
        "observation/exterior_image_1_left",
        "observation/wrist_image_left",
        "observation/joint_position",
        "observation/gripper_position",
        "prompt",
    }
    missing_keys = sorted(required_keys.difference(request_data))
    if missing_keys:
        raise ValueError("Policy request is missing keys: {0}".format(missing_keys))

    for image_key in ("observation/exterior_image_1_left", "observation/wrist_image_left"):
        image = np.asarray(request_data[image_key])
        if image.shape != (224, 224, 3):
            raise ValueError("{0} must have shape (224, 224, 3), got {1}".format(image_key, image.shape))

    joint_position = np.asarray(request_data["observation/joint_position"])
    if joint_position.shape != (7,):
        raise ValueError("observation/joint_position must have shape (7,), got {0}".format(joint_position.shape))

    gripper_position = np.asarray(request_data["observation/gripper_position"])
    if gripper_position.shape != (1,):
        raise ValueError(
            "observation/gripper_position must have shape (1,), got {0}".format(gripper_position.shape)
        )

    if not isinstance(request_data["prompt"], str) or not request_data["prompt"]:
        raise ValueError("prompt must be a non-empty string")


def make_observation_source(args):
    camera_kwargs = build_camera_kwargs(args)
    if args.mock_robot_state:
        return CameraOnlyObservationSource(camera_kwargs)

    from droid.robot_env import RobotEnv

    env = RobotEnv(
        action_space="joint_velocity",
        gripper_action_space="position",
        camera_kwargs=camera_kwargs,
        do_reset=not args.no_reset and not args.dry_run,
        launch_robot=not args.no_launch_robot,
    )
    return RobotEnvObservationSource(env)


def build_camera_kwargs(args):
    camera_backend = "mock" if args.mock_cameras else args.camera_backend
    resolution = (args.camera_width, args.camera_height) if args.camera_width and args.camera_height else (0, 0)
    default_kwargs = {
        "image": True,
        "depth": False,
        "pointcloud": False,
        "concatenate_images": False,
        "resolution": resolution,
        "resize_func": None,
        "fps": args.camera_fps,
    }
    camera_ids = [args.left_camera_id, args.wrist_camera_id]
    if args.right_camera_id:
        camera_ids.append(args.right_camera_id)
    camera_kwargs = {
        "camera_backend": camera_backend,
        "camera_ids": camera_ids,
        "default": default_kwargs,
    }
    if args.left_camera_device:
        camera_kwargs[args.left_camera_id] = {"device": args.left_camera_device}
    if args.right_camera_device:
        camera_kwargs[args.right_camera_id] = {"device": args.right_camera_device}
    if args.d435_serial:
        camera_kwargs[args.wrist_camera_id] = {"serial": args.d435_serial}
    return camera_kwargs


class RobotEnvObservationSource:
    def __init__(self, env):
        self.env = env

    def get_observation(self):
        return self.env.get_observation()

    def step(self, action):
        return self.env.step(action)

    def close(self):
        self.env.camera_reader.disable_cameras()


class CameraOnlyObservationSource:
    def __init__(self, camera_kwargs):
        from droid.camera_utils.wrappers.multi_camera_wrapper import MultiCameraWrapper

        self.camera_reader = MultiCameraWrapper(camera_kwargs)

    def get_observation(self):
        camera_obs, camera_timestamp = self.camera_reader.read_cameras()
        obs_dict = {
            "timestamp": {"cameras": camera_timestamp},
            "robot_state": {
                "cartesian_position": [0.0] * 6,
                "joint_positions": [0.0] * 7,
                "gripper_position": 0.0,
            },
        }
        obs_dict.update(camera_obs)
        return obs_dict

    def step(self, action):
        raise RuntimeError("CameraOnlyObservationSource cannot execute robot actions")

    def close(self):
        self.camera_reader.disable_cameras()


def extract_observation(args, obs_dict, save_to_disk=False):
    image_observations = obs_dict["image"]
    left_image = find_camera_image(image_observations, args.left_camera_id)
    right_image = find_camera_image(image_observations, args.right_camera_id) if args.right_camera_id else None
    wrist_image = find_camera_image(image_observations, args.wrist_camera_id)

    left_image = normalize_image(left_image, bgr_to_rgb=not args.no_bgr_to_rgb)
    if right_image is not None:
        right_image = normalize_image(right_image, bgr_to_rgb=not args.no_bgr_to_rgb)
    wrist_image = normalize_image(wrist_image, bgr_to_rgb=not args.no_bgr_to_rgb)

    robot_state = obs_dict["robot_state"]
    curr_obs = {
        "left_image": left_image,
        "right_image": right_image,
        "wrist_image": wrist_image,
        "cartesian_position": np.asarray(robot_state["cartesian_position"]),
        "joint_position": np.asarray(robot_state["joint_positions"]),
        "gripper_position": np.asarray([robot_state["gripper_position"]]),
    }

    if save_to_disk and args.save_preview:
        save_preview_image(args.save_preview, [left_image, wrist_image, right_image])

    return curr_obs


def find_camera_image(image_observations, camera_id):
    if not camera_id:
        raise ValueError("camera_id must be non-empty")
    for key, value in image_observations.items():
        if camera_id in key and "left" in key:
            return value
    raise KeyError(
        "Could not find image for camera id {0!r}. Available image keys: {1}".format(
            camera_id, sorted(image_observations.keys())
        )
    )


def normalize_image(image, bgr_to_rgb=True):
    image = np.asarray(image)
    if image.ndim != 3:
        raise ValueError("Expected an HxWxC image, got shape {0}".format(image.shape))
    if image.shape[-1] < 3:
        raise ValueError("Expected at least 3 image channels, got shape {0}".format(image.shape))
    image = image[..., :3]
    if bgr_to_rgb:
        image = image[..., ::-1]
    return np.ascontiguousarray(image)


def save_preview_image(path, images):
    images = [np.asarray(image) for image in images if image is not None]
    if not images:
        return

    preview_height = min(image.shape[0] for image in images)
    preview_tiles = []
    for image in images:
        height, width = image.shape[:2]
        if height != preview_height:
            preview_width = max(1, int(round(width * (preview_height / height))))
            image = np.asarray(Image.fromarray(image).resize((preview_width, preview_height)))
        preview_tiles.append(image)

    Image.fromarray(np.concatenate(preview_tiles, axis=1)).save(path)


def validate_action_chunk(pred_action_chunk):
    if (
        pred_action_chunk.ndim != 2
        or pred_action_chunk.shape[0] < 1
        or pred_action_chunk.shape[1] != EXPECTED_ACTION_DIM
    ):
        raise ValueError(
            "Expected action chunk shape (N, {0}) with N >= 1, got {1}".format(
                EXPECTED_ACTION_DIM, pred_action_chunk.shape
            )
        )


def process_action(action):
    action = np.asarray(action)
    if action[-1].item() > 0.5:
        action = np.concatenate([action[:-1], np.ones((1,))])
    else:
        action = np.concatenate([action[:-1], np.zeros((1,))])
    return np.clip(action, -1, 1)


def print_dry_run_step(t_step, action, pred_action_chunk, curr_obs):
    print(
        "dry_run step={0} action_chunk={1} action_min={2:.3f} action_max={3:.3f} "
        "left={4} wrist={5} joints={6}".format(
            t_step,
            pred_action_chunk.shape,
            float(action.min()),
            float(action.max()),
            curr_obs["left_image"].shape,
            curr_obs["wrist_image"].shape,
            curr_obs["joint_position"].shape,
        )
    )


if __name__ == "__main__":
    main()
