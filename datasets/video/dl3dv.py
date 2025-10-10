from typing import Dict, Any, Optional, List

import torch
from omegaconf import DictConfig
from PIL import Image
import numpy as np
from pathlib import Path

from .base_video import (
    BaseVideoDataset,
    BaseSimpleVideoDataset,
    BaseAdvancedVideoDataset,
    SPLIT,
)
from datasets.video.utils import VideoTransform


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
        """
        Build metadata by globbing video directories and counting frames directly.
        Selects shards by split:
          - training: 1K, 2K, 3K, 4K, 5K
          - validation: 6K
          - test: 7K

        Each sequence directory `{shard}/{hash}` contains `dense/rgb` with frames
        named `frame_XXXXX.png`. We glob these directories and count frames directly.
        """
        import glob
        
        root = self.save_dir / "processed_dl3dv_ours"
        
        split_to_shards = {
            "training": {"1K", "2K", "3K", "4K", "5K"},
            "validation": {"6K"},
            "test": {"7K"},
        }
        allowed_shards = split_to_shards.get(split, set())

        video_paths: List[Path] = []
        video_pts: List[torch.Tensor] = []
        video_fps: List[float] = []
        n_frames_list: List[int] = []

        # Default FPS for DL3DV dataset (can be overridden if needed)
        default_fps = 5.0

        # Glob all sequence directories for allowed shards
        for shard in allowed_shards:
            shard_pattern = str(root / shard / "*")
            sequence_dirs = glob.glob(shard_pattern)
            
            for seq_dir in sequence_dirs:
                seq_path = Path(seq_dir)
                rgb_dir = seq_path / "dense" / "rgb"
                
                # Check if RGB directory exists
                if not rgb_dir.exists():
                    continue
                
                # Glob all frame files and count them
                frame_pattern = str(rgb_dir / "frame_*.png")
                frame_files = glob.glob(frame_pattern)
                
                if len(frame_files) == 0:
                    continue
                
                # Sort frame files to ensure correct ordering
                frame_files.sort()
                
                # Extract frame numbers and verify they're consecutive starting from 1
                frame_numbers = []
                for frame_file in frame_files:
                    frame_name = Path(frame_file).name
                    # Extract number from frame_XXXXX.png
                    try:
                        frame_num = int(frame_name.split('_')[1].split('.')[0])
                        frame_numbers.append(frame_num)
                    except (ValueError, IndexError):
                        continue
                
                # Verify frames are consecutive starting from 1
                if not frame_numbers or frame_numbers[0] != 1:
                    continue
                
                expected_frames = list(range(1, len(frame_numbers) + 1))
                if frame_numbers != expected_frames:
                    continue
                
                n_frames_val = len(frame_numbers)
                if n_frames_val <= 0:
                    continue
                
                video_paths.append(rgb_dir)
                video_pts.append(torch.arange(n_frames_val, dtype=torch.long))
                video_fps.append(default_fps)
                n_frames_list.append(n_frames_val)

        if len(video_paths) == 0:
            raise RuntimeError(f"No sequences found for split {split} in {root}")

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
        rgb_dir: Path = video_metadata["video_paths"]
        # Do not glob; construct file paths using 1-based zero-padded indexing
        if end_frame is None:
            end_frame = self.video_length(video_metadata)
        frames: List[torch.Tensor] = []
        for i in range(start_frame, end_frame):
            fp = rgb_dir / f"frame_{i+1:05d}.png"
            if not fp.exists():
                raise FileNotFoundError(f"Missing RGB frame: {fp}")
            img = Image.open(fp).convert("RGB")
            arr = np.array(img, dtype=np.uint8)
            tensor = torch.from_numpy(arr).float() / 255.0  # (H, W, C)
            frames.append(tensor.permute(2, 0, 1).contiguous())  # (C, H, W)
        if len(frames) == 0:
            return torch.zeros((0, 3, self.resolution, self.resolution), dtype=torch.float32)
        video = torch.stack(frames, dim=0)  # (T, C, H, W)
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
        BaseAdvancedVideoDataset.__init__(self, cfg, split, current_epoch)

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

        rgb_dir: Path = video_metadata["video_paths"]
        seq_dir = rgb_dir.parent.parent  # .../{hash}
        cam_dir = seq_dir / "dense" / "cam"
        if not cam_dir.exists():
            raise FileNotFoundError(f"DL3DV cam directory not found: {cam_dir}")

        # Do not glob; construct cam file paths in lockstep with frames
        cams: List[torch.Tensor] = []
        for i in range(start_frame, end_frame):
            cam_fp = cam_dir / f"frame_{i+1:05d}.npz"
            if not cam_fp.exists():
                raise FileNotFoundError(f"Expected cam file not found: {cam_fp}")
            pose = np.load(cam_fp)["pose"]
            cams.append(torch.as_tensor(pose, dtype=torch.float32))

        cams_tensor = torch.stack(cams, dim=0)  # (T, 4, 4)
        cams_tensor = cams_tensor.reshape(T, -1)
        assert cams_tensor.shape[-1] == 16, f"cams_tensor last dim must be 16, got {cams_tensor.shape}"
        return cams_tensor


    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        video_idx, start_frame = self.get_clip_location(idx)
        video_metadata = self.metadata[video_idx]
        video_length = self.video_length(video_metadata)
        frame_skip = (video_length - start_frame - 1) // (self.cfg.max_frames - 1)
        # For DL3DV, clamp by configured frame_skip during training
        frame_skip = min(frame_skip, self.frame_skip)

        assert frame_skip > 0, f"Frame skip {frame_skip} should be greater than 0"
        end_frame = start_frame + (self.cfg.max_frames - 1) * frame_skip + 1

        video, cond = self.load_video_and_cond(video_metadata, start_frame, end_frame)
        assert len(video) == len(cond), "Video and cond have different lengths"

        video, cond = video[::frame_skip], cond[::frame_skip]
        return {
            "videos": self.transform(video),
            "conds": cond,
            "nonterminal": torch.ones(self.cfg.max_frames, dtype=torch.bool),
        }


