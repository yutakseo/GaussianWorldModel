import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset
import torch.nn.functional as F
# import zarr
from PIL import Image
import pathlib
from typing import Dict, List, Tuple, Optional, Union, Any
import glob
from pathlib import Path
import tensorflow as tf
import tensorflow_graphics.geometry.transformation as tfg

from gaussianwm.processor.rlds import make_interleaved_dataset, make_single_dataset
from gaussianwm.processor.rlds.oxe import OXE_NAMED_MIXTURES, get_oxe_dataset_kwargs_and_weights
# from gaussianwm.processor.rlds.utils.data_utils import NormalizationType, combine_dataset_statistics
from gaussianwm.processor.rlds.utils.data_utils import NormalizationType
from gaussianwm.processor.rlds.dataset import make_dataset_from_rlds

# Configure Tensorflow with *no GPU devices* (to prevent clobber with PyTorch)
tf.config.set_visible_devices([], "GPU")


def euler_to_rmat(euler):
    """Convert Euler angles to rotation matrix."""
    return tfg.rotation_matrix_3d.from_euler(euler)


def mat_to_rot6d(mat):
    """Convert rotation matrix to 6D rotation representation."""
    r6 = mat[..., :2, :]
    r6_0, r6_1 = r6[..., 0, :], r6[..., 1, :]
    r6_flat = tf.concat([r6_0, r6_1], axis=-1)
    return r6_flat


def droid_dataset_transform(trajectory):
    """Transform DROID dataset trajectory to canonical format."""
    # every input feature is batched, ie has leading batch dimension
    T = trajectory["action_dict"]["cartesian_position"][:, :3]
    R = mat_to_rot6d(euler_to_rmat(trajectory["action_dict"]["cartesian_position"][:, 3:6]))
    trajectory["action"] = tf.concat(
        (
            T,
            R,
            trajectory["action_dict"]["gripper_position"],
        ),
        axis=-1,
    )
    return trajectory


def robomimic_transform(trajectory):
    """Transform trajectory to robomimic format."""
    return {
        "obs": {
            "camera/image/varied_camera_1_left_image": 
                # tf.cast(trajectory["observation"]["image_primary"], tf.float32) / 255.,
                trajectory["observation"]["image_primary"],
            "camera/image/varied_camera_2_left_image": 
                # tf.cast(trajectory["observation"]["image_secondary"], tf.float32) / 255.,
                trajectory["observation"]["image_secondary"],
            "raw_language": trajectory["task"]["language_instruction"],
            "robot_state/cartesian_position": trajectory["observation"]["proprio"][..., :6],
            "robot_state/gripper_position": trajectory["observation"]["proprio"][..., -1:],
            "pad_mask": trajectory["observation"]["pad_mask"][..., None],
        },
        "actions": trajectory["action"][1:],
    }


class AxisScaling(object):
    def __init__(self, interval=(0.75, 1.25), jitter=True):
        assert isinstance(interval, tuple)
        self.interval = interval
        self.jitter = jitter
        
    def __call__(self, surface, point):
        scaling = torch.rand(1, 3) * 0.5 + 0.75
        surface = surface * scaling
        point = point * scaling

        scale = (1 / torch.abs(surface).max().item()) * 0.999999
        surface *= scale
        point *= scale

        if self.jitter:
            surface += 0.005 * torch.randn_like(surface)
            surface.clamp_(min=-1, max=1)

        return surface, point

def build_shape_surface_occupancy_dataset(split, args):
    from .shapenet import ShapeNet
    if split == 'train':
        # transform = #transforms.Compose([
        transform = AxisScaling((0.75, 1.25), True)
        # ])
        return ShapeNet(args.data_path, split=split, transform=transform, sampling=True, num_samples=1024, return_surface=True, surface_sampling=True, pc_size=args.point_cloud_size)
    elif split == 'val':
        # return ShapeNet(args.data_path, split=split, transform=None, sampling=True, num_samples=1024, return_surface=True, surface_sampling=True, pc_size=args.point_cloud_size)
        return ShapeNet(args.data_path, split=split, transform=None, sampling=False, return_surface=True, surface_sampling=True, pc_size=args.point_cloud_size)
    else:
        return ShapeNet(args.data_path, split=split, transform=None, sampling=False, return_surface=True, surface_sampling=True, pc_size=args.point_cloud_size)

class DroidDataset(IterableDataset):
    def __init__(
        self,
        data_path: str,
        segment_length: int = 12,
        context_length: int = 2,
        action_dim: int = 10,
        image_size: int = 128,
        augment: bool = False,
        val_ratio: float = 0.0,
        seed: int = 42,
        split: str = "train",
        camera_keys: List[str] = ["primary", "secondary"],
        action_keys: List[str] = ["actions"],
        future_action_window_size: int = 15,
        subsample_length: int = 100,
        shuffle_buffer_size: int = 100000,
        batch_size: Optional[int] = None,
        traj_transform_threads: int = 48,
        traj_read_threads: int = 48
    ):
        """
        Initialize the DroidDataset using RLDS format.
        
        Args:
            data_path: Path to RLDS datasets
            segment_length: Number of timesteps in each returned segment
            context_length: Number of context frames
            action_dim: Dimension of action vectors (10 for DROID)
            image_size: Size to resize images to (H=W=image_size)
            augment: Whether to use data augmentation
            val_ratio: Fraction of data to use for validation
            seed: Random seed for reproducibility
            split: 'train' or 'val'
            camera_keys: Camera observation keys
            action_keys: Action keys
            window_size: Window size for trajectory transforms
            future_action_window_size: Future action window size
            subsample_length: Subsample trajectory length
            shuffle_buffer_size: Shuffle buffer size
            batch_size: Batch size (None for no batching)
            traj_transform_threads: Number of trajectory transform threads
            traj_read_threads: Number of trajectory read threads
        """
            
        self.data_path = data_path
        self.segment_length = segment_length
        self.context_length = context_length
        self.action_dim = action_dim
        self.image_size = image_size
        self.augment = augment
        self.split = split
        self.camera_keys = camera_keys
        self.action_keys = action_keys
        self.rng = np.random.RandomState(seed)

        # Base dataset configuration
        BASE_DATASET_KWARGS = {
            "data_dir": data_path,
            "image_obs_keys": {"primary": "exterior_image_1_left", "secondary": "exterior_image_2_left"},
            "state_obs_keys": ["cartesian_position", "gripper_position"],
            "language_key": "language_instruction",
            # "norm_skip_keys": ["proprio"],
            "action_proprio_normalization_type": "bounds",
            "absolute_action_mask": [True] * 10,  # droid_dataset_transform uses absolute actions
            "action_normalization_mask": [True] * 9 + [False],  # don't normalize final (gripper) dimension
            "standardize_fn": droid_dataset_transform,
        }

        # Filter for success trajectories only in DROID
        filter_functions = [
            lambda trajectory: tf.strings.regex_full_match(
                        trajectory['traj_metadata']['episode_metadata']['file_path'][0],
                        ".*/success/.*"
                    )
        ]

        dataset_kwargs_list = [{
            # "name": "droid",
            "name": "droid_100",
            # "name": "berkeley_cable_routing",
            # "filter_functions": filter_functions,
            **BASE_DATASET_KWARGS
        }]

        # Compute combined normalization stats
        # combined_dataset_statistics = combine_dataset_statistics(
        #     [make_dataset_from_rlds(**dataset_kwargs, train=(split == "train"))[1] 
        #      for dataset_kwargs in dataset_kwargs_list]
        # )

        # Create the interleaved dataset
        self.dataset, self.dataset_length, self.dataset_statistics = make_interleaved_dataset(
            dataset_kwargs_list,
            sample_weights=[1.0],
            train=(split == "train"),
            shuffle_buffer_size=shuffle_buffer_size,
            batch_size=batch_size,
            balance_weights=False,
            # dataset_statistics=combined_dataset_statistics,
            traj_transform_kwargs=dict(
                window_size=segment_length,
                future_action_window_size=future_action_window_size,
                subsample_length=subsample_length,
                skip_unlabeled=False,  # skip all trajectories without language annotation
            ),
            frame_transform_kwargs=dict(
                image_augment_kwargs=dict() if not augment else dict(
                    # TODO: Add augmentation parameters here
                ),
                resize_size=dict(
                    primary=[image_size, image_size],
                    secondary=[image_size, image_size],
                ),
                num_parallel_calls=200,
            ),
            traj_transform_threads=traj_transform_threads,
            traj_read_threads=traj_read_threads,
        )

        # Apply robomimic transform
        self.dataset = self.dataset.map(robomimic_transform, num_parallel_calls=48)
        
    def __iter__(self):
        """Iterate over the dataset."""
        for sample in self.dataset.as_numpy_iterator():
            # Convert to the expected format (obs_frames, action, reward)
            # print(f"{sample.keys()=}")
            # sample.keys()=dict_keys(['obs', 'actions'])
            # print(f"{sample['obs'].keys()=}")
            # dict_keys(['camera/image/varied_camera_1_left_image', 'camera/image/varied_camera_2_left_image', \
            # 'raw_language', 'robot_state/cartesian_position', 'robot_state/gripper_position', 'pad_mask'])
            obs = {}
            # pad_mask = torch.from_numpy(sample['obs']['pad_mask']).to(torch.bool)

            left_frames = sample['obs']['camera/image/varied_camera_1_left_image']  # Use primary camera
            right_frames = sample['obs']['camera/image/varied_camera_2_left_image']
            action = sample['actions']
            reward = torch.zeros((self.segment_length, 1))  # Dummy reward
            
            # Convert numpy arrays to torch tensors
            left_frames = torch.from_numpy(left_frames)
            right_frames = torch.from_numpy(right_frames)
            action = torch.from_numpy(action)

            # print(f"{left_frames.shape=}, {left_frames.dtype=}, {left_frames.min()=}, {left_frames.max()=}")
            # left_frames.shape=torch.Size([2, 128, 128, 3]), 
            # left_frames.dtype=torch.float32, left_frames.min()=tensor(0.), left_frames.max()=tensor(1.

            # Convert to uint8 range [0, 255] if needed
            if left_frames.dtype == torch.float32 and left_frames.max() <= 1.0:
                left_frames = (left_frames * 255).to(torch.uint8)
            if right_frames.dtype == torch.float32 and right_frames.max() <= 1.0:
                right_frames = (right_frames * 255).to(torch.uint8)
            # print(f"{action.shape=}, {reward.shape=}")
            # action.shape=torch.Size([16, 10]), reward.shape=torch.Size([10, 1])
            # obs = {
            #     "robot0_agentview_left_image": left_frames, # (T, H, W, C)
            #     "robot0_agentview_right_image": right_frames,
            # }
            obs = left_frames
            # yield obs, action, reward, pad_mask
            yield obs, action, reward

    def __len__(self):
        # lengths = np.array(
        #     [
        #         stats["num_transitions"]
        #         for stats in self.dataset.dataset_statistics
        #     ]
        # )
        # if hasattr(self.dataset, "sample_weights"):
        #     lengths *= np.array(self.dataset.sample_weights)
        # total_len = lengths.sum()
        # if self._is_train:
        #     return int(0.95 * total_len)
        # else:
        #     return int(0.05 * total_len)
        return self.dataset_length

class OXEDataset(IterableDataset):
    def __init__(
        self,
        data_root_dir: Path,
        data_mix: str,
        resize_resolution: Tuple[int, int],
        shuffle_buffer_size: int = 256_000,
        train: bool = True,
        image_aug: bool = False,
    ) -> None:
        """Lightweight wrapper around RLDS TFDS Pipeline for use with PyTorch/OpenVLA Data Loaders."""
        self.data_root_dir, self.data_mix = data_root_dir, data_mix

        # Configure RLDS Dataset(s)
        if self.data_mix in OXE_NAMED_MIXTURES:
            mixture_spec = OXE_NAMED_MIXTURES[self.data_mix]
        else:
            # Assume that passed "mixture" name is actually a single dataset -- create single-dataset "mix"
            mixture_spec = [(self.data_mix, 1.0)]

        # fmt: off
        per_dataset_kwargs, weights = get_oxe_dataset_kwargs_and_weights(
            self.data_root_dir,
            mixture_spec,
            load_camera_views=("primary",),
            load_depth=False,
            load_proprio=False,
            load_language=False,
            action_proprio_normalization_type=NormalizationType.BOUNDS_Q99,
        )
        rlds_config = dict(
            traj_transform_kwargs=dict(
                window_size=1,                                      # If we wanted to feed / predict more than one step
                future_action_window_size=0,                        # For action chunking
                skip_unlabeled=True,                                # Skip trajectories without language labels
                goal_relabeling_strategy="uniform",                 # Goals are currently unused
            ),
            frame_transform_kwargs=dict(
                resize_size=resize_resolution,
                num_parallel_calls=16,                          # For CPU-intensive ops (decoding, resizing, etc.)
            ),
            dataset_kwargs_list=per_dataset_kwargs,
            shuffle_buffer_size=shuffle_buffer_size,
            sample_weights=weights,
            balance_weights=True,
            traj_transform_threads=len(mixture_spec),
            traj_read_threads=len(mixture_spec),
            train=train,
        )

        # If applicable, enable image augmentations
        if image_aug:
            rlds_config["frame_transform_kwargs"].update({"image_augment_kwargs" : dict(
                random_resized_crop=dict(scale=[0.9, 0.9], ratio=[1.0, 1.0]),
                random_brightness=[0.2],
                random_contrast=[0.8, 1.2],
                random_saturation=[0.8, 1.2],
                random_hue=[0.05],
                augment_order=[
                    "random_resized_crop",
                    "random_brightness",
                    "random_contrast",
                    "random_saturation",
                    "random_hue",
                ],
            )}),
        # fmt: on

        # Initialize RLDS Dataset
        self.dataset, self.dataset_length, self.dataset_statistics = self.make_dataset(rlds_config)

    def make_dataset(self, rlds_config):
        return make_interleaved_dataset(**rlds_config)

    def __iter__(self):
        for rlds_batch in self.dataset.as_numpy_iterator():
            yield rlds_batch

    def __len__(self) -> int:
        return self.dataset_length

    # === Explicitly Unused ===
    def __getitem__(self, idx: int) -> None:
        raise NotImplementedError("IterableDataset does not implement map-style __getitem__; see __iter__ instead!")


def build_gaussian_splatting_reconstruction_dataset(split, cfg):
    if cfg.dataset_name == 'droid':
        return DroidDataset(
            data_path=cfg.data_path,
            segment_length=cfg.segment_length,
            context_length=cfg.context_length,
            action_dim=cfg.action_dim,
            image_size=cfg.image_size,
            augment=cfg.augment,
            val_ratio=cfg.val_ratio,
            seed=cfg.seed,
            split=split,
            camera_keys=cfg.camera_keys,
            action_keys=cfg.action_keys,
            future_action_window_size=cfg.future_action_window_size,
            subsample_length=cfg.subsample_length,
            shuffle_buffer_size=cfg.shuffle_buffer_size,
            batch_size=None,
            traj_transform_threads=cfg.traj_transform_threads,
            traj_read_threads=cfg.traj_read_threads
        )
    else:
        raise ValueError(f"Dataset {cfg.dataset} not supported")


if __name__ == '__main__':
    from .shapenet import ShapeNet
    m = ShapeNet('./data/', 'train', transform=AxisScaling(), sampling=True, num_samples=1024, return_surface=True, surface_sampling=True)
    p, l, s, c = m[0]
    print(p.shape, l.shape, s.shape, c)
    print(p.max(dim=0)[0], p.min(dim=0)[0])
    print(p[l==1].max(axis=0)[0], p[l==1].min(axis=0)[0])
    print(s.max(axis=0)[0], s.min(axis=0)[0])