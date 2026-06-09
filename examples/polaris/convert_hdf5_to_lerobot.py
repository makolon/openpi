"""Convert polaris_real2sim TAMP HDF5 episodes to the openpi (pi0.5 / DROID) LeRobot v2.1 dataset.

Self-contained: depends only on this repo's pinned lerobot (legacy
``lerobot.common.*`` API, codebase v2.1) plus the bundled ``polaris_hdf5``
reader. Run inside the openpi env, fully independent of IsaacLab:

    cd third_party/openpi
    uv sync --frozen          # installs openpi + lerobot (+ h5py/imageio/pyarrow)
    uv run examples/polaris/convert_hdf5_to_lerobot.py \\
        --hdf5-root /path/to/tamp_hdf5_dataset \\
        --repo-id local/my_droid_dataset --action-space joint

Proprio is always joint-space (``joint_position`` 7 + ``gripper_position`` 1,
the DROID convention). ``--action-space joint`` stores the recorded joint
position target (8-D); ``ee`` stores delta-EE pose actions (10-D, needs a custom
``*Inputs/*Outputs`` transform on the openpi side).

Then finetune in this same env via ``uv run scripts/train.py <config>``. Reusing
the pretrained ``asset_id="droid"`` norm stats is NOT valid here (pi05-droid was
pretrained on joint *velocity* actions, this stores joint *position targets*) --
always run ``uv run scripts/compute_norm_stats.py --config-name <name>``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
from polaris_hdf5 import CameraMapping
from polaris_hdf5 import Hdf5DatasetReader
from polaris_hdf5 import Hdf5EpisodeReader
from polaris_hdf5 import ee_delta_action
from polaris_hdf5 import joint_action
from polaris_hdf5 import normalize_gripper

_CAMERA_FEATURES: dict[str, str] = {
    "exterior_1": "exterior_image_1_left",
    "exterior_2": "exterior_image_2_left",
    "wrist": "wrist_image_left",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdf5-root", required=True, type=Path, help="TAMP HDF5 collection root")
    parser.add_argument("--repo-id", required=True, help="output LeRobot repo id")
    parser.add_argument("--root", type=Path, default=None, help="output dir (default: $HF_LEROBOT_HOME/<repo-id>)")
    parser.add_argument("--action-space", choices=("joint", "ee"), default="joint")
    parser.add_argument("--camera-map", default=None, help="JSON {role: hdf5_image_key} override")
    parser.add_argument("--robot-type", default="panda")
    parser.add_argument("--fps", type=int, default=None, help="override fps (default: from episodes)")
    parser.add_argument("--raw-gripper", action="store_true", help="keep raw gripper joint value (no [0,1] norm)")
    parser.add_argument("--push-to-hub", action="store_true")
    return parser.parse_args()


def _action_for(reader: Hdf5EpisodeReader, action_space: str, *, normalize: bool) -> np.ndarray:
    if action_space == "joint":
        return joint_action(reader, normalize_gripper_value=normalize)
    return ee_delta_action(reader, normalize_gripper_value=normalize)


def _build_features(sample: Hdf5EpisodeReader, mapping: CameraMapping, action_dim: int) -> dict[str, dict]:
    features: dict[str, dict] = {}
    for role, feature_name in _CAMERA_FEATURES.items():
        height, width, channels = sample.read_video(mapping.key_for(role)).shape[1:]
        features[feature_name] = {
            "dtype": "image",
            "shape": (int(height), int(width), int(channels)),
            "names": ["height", "width", "channel"],
        }
    features["joint_position"] = {
        "dtype": "float32",
        "shape": (int(sample.arm_joint_position.shape[1]),),
        "names": ["joint_position"],
    }
    features["gripper_position"] = {"dtype": "float32", "shape": (1,), "names": ["gripper_position"]}
    features["actions"] = {"dtype": "float32", "shape": (int(action_dim),), "names": ["actions"]}
    return features


def main() -> None:
    args = _parse_args()
    normalize = not args.raw_gripper
    mapping = CameraMapping.from_json(args.camera_map)

    episode_paths = Hdf5DatasetReader(args.hdf5_root).episode_paths()
    if not episode_paths:
        raise FileNotFoundError(f"no episodes under {args.hdf5_root}")

    sample = Hdf5EpisodeReader(episode_paths[0])
    fps = args.fps if args.fps is not None else sample.fps
    action_dim = int(_action_for(sample, args.action_space, normalize=normalize).shape[1])
    features = _build_features(sample, mapping, action_dim)

    output_root = args.root if args.root is not None else HF_LEROBOT_HOME / args.repo_id
    if Path(output_root).exists():
        raise FileExistsError(f"output already exists: {output_root} (remove it or pick a new --repo-id)")

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=output_root,
        robot_type=args.robot_type,
        fps=int(fps),
        features=features,
        image_writer_threads=10,
        image_writer_processes=5,
    )

    for path in episode_paths:
        reader = Hdf5EpisodeReader(path)
        videos = {feat: reader.read_video(mapping.key_for(role)) for role, feat in _CAMERA_FEATURES.items()}
        gripper = normalize_gripper(
            reader.gripper_joint_position,
            reader.gripper_open_value if normalize else None,
            reader.gripper_closed_value if normalize else None,
        )
        action = _action_for(reader, args.action_space, normalize=normalize)
        for t in range(reader.num_frames):
            frame = {feat: videos[feat][t] for feat in videos}
            frame["joint_position"] = reader.arm_joint_position[t]
            frame["gripper_position"] = np.asarray(gripper[t], dtype=np.float32).reshape(1)
            frame["actions"] = action[t]
            frame["task"] = reader.instruction
            dataset.add_frame(frame)
        dataset.save_episode()
        print(f"[openpi-convert] wrote episode {path.stem} ({reader.num_frames} frames)", flush=True)

    if args.push_to_hub:
        dataset.push_to_hub(tags=["polaris", args.robot_type], private=False, push_videos=True)

    print(f"[openpi-convert] done: {len(episode_paths)} episodes -> {output_root}", flush=True)


if __name__ == "__main__":
    main()
