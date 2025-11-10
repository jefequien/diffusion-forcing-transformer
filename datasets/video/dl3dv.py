from typing import Dict, Any, Optional, List, Tuple

import torch
from omegaconf import DictConfig
from PIL import Image
import numpy as np
from pathlib import Path
import random
import os
import json
from tqdm import tqdm

from .base_video import (
    BaseVideoDataset,
    BaseSimpleVideoDataset,
    BaseAdvancedVideoDataset,
    SPLIT,
)
from datasets.video.utils import VideoTransform
from .utils import random_bool


class DL3DVBaseVideoDataset(BaseVideoDataset):
    """
    DL3DV base video dataset.

    Expected structure under `save_dir`:
    - metadata/{split}.pt: precomputed split file with keys {video_paths, video_pts, video_fps}
      where each entry in `video_paths` is a directory path to frames, e.g.:
        CUT3R_full/processed_dl3dv_ours/{1K,2K,...}/{hash}/rgb
      and frames are named like `frame_XXXXX.png` inside that directory.
    """

    _ALL_SPLITS = ["training", "validation", "test"]

    def _should_download(self) -> bool:
        # By default, assume users manually stage data in save_dir.
        return False

    def download_dataset(self) -> None:
        # Provide your download logic here if/when needed.
        raise NotImplementedError(
            "DL3DV does not provide an automatic downloader yet. "
            "Please prepare the dataset under cfg.save_dir with the expected structure."
        )
    
    def build_metadata(self, split: SPLIT) -> None:
        data_patterns = {
            "training": ["1K", "2K", "3K", "4K", "5K", "6K", "7K", "8K", "10K"], 
            "validation": ["../dl3dv-evaluation-frames256"],
            "test": ["../dl3dv-evaluation-frames256"],
        }
        default_fps = 5.0

        video_paths = []
        for data_pattern in data_patterns[split]:
            scene_names = os.listdir(os.path.join(self.save_dir, data_pattern))
            for scene_name in scene_names:
                video_paths.append(os.path.join(self.save_dir, data_pattern, scene_name))
        video_paths = sorted(video_paths)

        video_pts: List[torch.Tensor] = []
        video_fps: List[float] = []
        n_frames_list: List[int] = []
        for video_path in tqdm(video_paths):
            transforms_json = Path(video_path) / "transforms.json"
            with open(transforms_json, "r") as f:
                transforms = json.load(f)
            video_pts.append(torch.arange(len(transforms["frames"]), dtype=torch.long))
            video_fps.append(default_fps)
            n_frames_list.append(len(transforms["frames"]))

        metadata: Dict[str, Any] = {
            "video_paths": video_paths,
            "video_pts": video_pts,
            "video_fps": video_fps,
            "n_frames": n_frames_list,
        }
        self.metadata_dir.mkdir(exist_ok=True, parents=True)
        torch.save(metadata, self.metadata_dir / f"{split}.pt")

    def load_metadata(self) -> List[Dict[str, Any]]:
        """
        Load precomputed metadata from save_dir/metadata/{split}.pt.
        Expected keys: {video_paths, video_pts, video_fps}.
        """
        meta_file = self.metadata_dir / f"{self.split}.pt"
        if not meta_file.exists():
            # Build metadata for this split if missing
            self.build_metadata(self.split)
        metadata = torch.load(meta_file, weights_only=False)
        keys = list(metadata.keys())
        return [
            {key: metadata[key][i] for key in metadata.keys()}
            for i in range(len(metadata["video_paths"]))
        ]

    def load_video(
        self,
        video_metadata: Dict[str, Any],
        start_frame: int,
        end_frame: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Load a clip [start_frame:end_frame) by reading sorted PNG frames in the `rgb` directory.
        Returns float tensor in [0, 1] with shape (T, C, H, W).
        """
        if end_frame is None:
            end_frame = self.video_length(video_metadata)
        
        transforms_json = Path(video_metadata["video_paths"]) / "transforms.json"
        with open(transforms_json, "r") as f:
            transforms = json.load(f)
        frames = transforms["frames"]
        
        images: List[torch.Tensor] = []
        for i in range(start_frame, end_frame):
            frame = frames[i]
            fp = Path(video_metadata["video_paths"]) / f"{frame['file_path']}.png"
            img = Image.open(fp).convert("RGB")
            arr = np.array(img, dtype=np.uint8)
            tensor = torch.from_numpy(arr).float() / 255.0  # (H, W, C)
            images.append(tensor.permute(2, 0, 1).contiguous())  # (C, H, W)
        if len(images) == 0:
            return torch.zeros((0, 3, self.resolution, self.resolution), dtype=torch.float32)
        video = torch.stack(images, dim=0)  # (T, C, H, W)
        return video

    def setup(self) -> None:
        # No-op; defined to align with callers that expect `setup()`.
        return

    def build_transform(self):
        """
        Build transform for square resolution.
        """
        return VideoTransform((self.resolution, self.resolution))


class DL3DVSimpleVideoDataset(DL3DVBaseVideoDataset, BaseSimpleVideoDataset):
    """
    Loads full videos and returns video tensors plus target latent paths.
    """

    def __init__(self, cfg: DictConfig, split: SPLIT = "training"):
        BaseSimpleVideoDataset.__init__(self, cfg, split)
        self.setup()


class DL3DVAdvancedVideoDataset(DL3DVBaseVideoDataset, BaseAdvancedVideoDataset):
    """
    Loads variable-length clips with frame skipping. External conditioning is optional.
    """

    def __init__(
        self,
        cfg: DictConfig,
        split: SPLIT = "training",
        current_epoch: Optional[int] = None,
    ):  
        self.maximize_training_data = cfg.maximize_training_data
        self.augmentation = cfg.augmentation
        BaseAdvancedVideoDataset.__init__(self, cfg, split, current_epoch)
    
    @property
    def _training_frame_skip(self) -> int:
        if self.augmentation.frame_skip_increase == 0:
            return self.frame_skip
        assert (
            self.current_subepoch is not None
        ), "Subepoch should be given to the RealEstate10KAdvancedVideoDataset, to use frame skip schedule"
        return self.frame_skip + int(
            self.current_subepoch * self.augmentation.frame_skip_increase
        )

    def on_before_prepare_clips(self) -> None:
        self.setup()

    def load_cond(
        self, video_metadata: Dict[str, Any], start_frame: int, end_frame: int
    ) -> torch.Tensor:
        # Load per-frame camera data from cam/frame_{frame_id}.npz matching RGB frames.
        T = end_frame - start_frame
        D = getattr(self.cfg, "external_cond_dim", 0) or 0
        if D == 0:
            return torch.zeros((T, 0), dtype=torch.float32)

        transforms_json = Path(video_metadata["video_paths"]) / "transforms.json"
        with open(transforms_json, "r") as f:
            transforms = json.load(f)
        frames = transforms["frames"]
        image_height = transforms["h"]
        image_width = transforms["w"]
        fx = transforms["fl_x"]
        fy = transforms["fl_y"]
        cx = transforms["cx"]
        cy = transforms["cy"]

        # 1) Load ALL camera poses and intrinsics for the sequence to compute scene diameter
        all_pose_list: List[torch.Tensor] = []
        all_intrinsic_list: List[torch.Tensor] = []
        for frame in frames:
            intrinsic_matrix = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            # Normalize by image size to be resolution-invariant
            intrinsic_matrix[0, :] /= float(image_width)
            intrinsic_matrix[1, :] /= float(image_height)

            c2w = torch.tensor(frame["transform_matrix"])
            # Convert from OpenGL to OpenCV coordinate system
            c2w[0:3, 1:3] *= -1
            all_pose_list.append(c2w)
            all_intrinsic_list.append(intrinsic_matrix)

        all_poses = torch.stack(all_pose_list, dim=0)              # (N, 4, 4), already camera-to-world
        all_intrinsics = torch.stack(all_intrinsic_list, dim=0)    # (N, 3, 3)

        # 4) Select required camera poses for requested frame range
        sel_poses = all_poses[start_frame:end_frame]
        sel_intrinsics = all_intrinsics[start_frame:end_frame]

        # 5) Convert selected c2w to w2c for output conditioning
        sel_poses_w2c = torch.inverse(sel_poses)

        # 6) Pack conditioning vectors [fx, fy, cx, cy] + flattened 3x4 extrinsics (w2c)
        cams = []
        for pose_w2c, intrinsic_matrix in zip(sel_poses_w2c, sel_intrinsics):
            fx = intrinsic_matrix[0, 0]
            fy = intrinsic_matrix[1, 1]
            cx = intrinsic_matrix[0, 2]
            cy = intrinsic_matrix[1, 2]
            intrinsics_vector = torch.tensor([fx, fy, cx, cy], dtype=torch.float32)

            cond_vector = torch.concatenate([
                intrinsics_vector,
                pose_w2c[:3, :].flatten()
            ])
            cams.append(cond_vector)

        cams_tensor = torch.stack(cams, dim=0)  # (T, 16)
        assert cams_tensor.shape[-1] == 16, f"cams_tensor last dim must be 16, got {cams_tensor.shape}"
        return cams_tensor
    
    def _augment(
        self, video: torch.Tensor, cond: torch.Tensor, video_full: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1) Horizontal flip augmentation
        if random_bool(self.augmentation.horizontal_flip_prob):
            video = video.flip(-1)
            if video_full is not None:
                video_full = video_full.flip(-1)
            # NOTE: extrinsics should also be flipped accordingly - the following is equivalent to:
            # E' = I' @ E @ I' where I' = diag([-1, 1, 1, 1]) (E is 4x4 extrinsics matrix)
            cond[:, [5, 6, 7, 8, 12]] *= -1

        # 2) Back-and-forth video augmentation
        # 0 1 2 ... 2k+1 -> 0 2 4 ... 2k 2k+1 ... 3 1
        if random_bool(self.augmentation.back_and_forth_prob):
            video, cond = map(
                lambda x: torch.cat([x[::2], x[1::2].flip(0)], dim=0).contiguous(),
                (video, cond),
            )
            if video_full is not None:
                video_full = torch.cat(
                    [video_full[::2], video_full[1::2].flip(0)], dim=0
                ).contiguous()
        # 3) Reverse video augmentation
        # 0 ... n -> n ... 0
        if random_bool(self.augmentation.reverse_prob):
            video, cond = map(lambda x: x.flip(0).contiguous(), (video, cond))
            if video_full is not None:
                video_full = video_full.flip(0).contiguous()

        if video_full is None:
            return video, cond
        return video, cond, video_full


    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.split != "training":
            return super().__getitem__(idx)
        video_idx, start_frame = self.get_clip_location(idx)
        video_metadata = self.metadata[video_idx]
        video_length = self.video_length(video_metadata)
        frame_skip = (video_length - start_frame - 1) // (self.cfg.max_frames - 1)
        # For DL3DV, clamp by configured frame_skip during training
        if self.split == "training":
            frame_skip = min(frame_skip, self._training_frame_skip)
        else:
            frame_skip = random.randint(self.frame_skip, frame_skip)

        assert frame_skip > 0, f"Frame skip {frame_skip} should be greater than 0"
        end_frame = start_frame + (self.cfg.max_frames - 1) * frame_skip + 1

        video, cond = self.load_video_and_cond(video_metadata, start_frame, end_frame)
        assert len(video) == len(cond), "Video and cond have different lengths"

        video, cond = video[::frame_skip], self._process_external_cond(cond, frame_skip)
        video, cond = self._augment(video, cond)
        return {
            "videos": self.transform(video),
            "conds": cond,
            "nonterminal": torch.ones(self.cfg.max_frames, dtype=torch.bool),
        }
    
    def exclude_short_videos(
        self, metadata: List[Dict[str, Any]], min_frames: int
    ) -> List[Dict[str, Any]]:
        # if self.maximize_training_data is True,
        # include all videos with at least self.cfg.max_frames frames
        if self.maximize_training_data and self.split == "training":
            min_frames = min(min_frames, self.cfg.max_frames)
        return super().exclude_short_videos(metadata, min_frames)

    def _process_external_cond(
        self, external_cond: torch.Tensor, frame_skip: Optional[int] = None
    ) -> torch.Tensor:
        """
        Converts the raw camera poses to concat-flattened intrinsics and extrinsics.
        Args:
            external_cond (torch.Tensor): Raw camera poses. Shape (T, 16).
            frame_skip (Optional[int]): Frame skip. If None, uses self.frame_skip.
        Returns:
            torch.Tensor: Processed camera poses. Shape (T, 16).
        """
        poses = external_cond[:: frame_skip or self.frame_skip]
        return poses.to(torch.float32)


