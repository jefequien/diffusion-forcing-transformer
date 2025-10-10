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
        Build metadata by scanning `save_dir/processed_dl3dv_ours`.
        Expected per-sequence layout: `{...}/{hash}/dense/rgb/frame_XXXXX.png` and camera
        data under `{shard}/{hash}/cam`.

        Shard directories like `1K`, `2K`, ... are not resolutions; they are groupings
        where each shard contains approximately N=1000, 2000, ... videos respectively.
        If a split subfolder exists (e.g., `processed_dl3dv_ours/training`), only scan that;
        otherwise scan all shard directories under `processed_dl3dv_ours`.
        """
        root = self.save_dir / "processed_dl3dv_ours"
        split_root = root / split
        scan_root = split_root if split_root.exists() else root

        if not scan_root.exists():
            raise FileNotFoundError(
                f"DL3DV root not found: {scan_root}. Expected data under processed_dl3dv_ours."
            )

        video_paths: List[Path] = []
        video_pts: List[torch.Tensor] = []
        video_fps: List[float] = []

        # iterate through all sequences under any shard directory (e.g., 1K, 2K, ...)
        for shard_dir in sorted([p for p in scan_root.iterdir() if p.is_dir()]):
            for seq_dir in sorted([p for p in shard_dir.iterdir() if p.is_dir()]):
                rgb_dir = seq_dir / "dense" / "rgb"
                if not rgb_dir.exists():
                    raise FileNotFoundError(
                        f"DL3DV rgb directory not found: {rgb_dir}. Expected data under {seq_dir}."
                    )
                frames = sorted(rgb_dir.glob("*.png"))
                if len(frames) == 0:
                    raise FileNotFoundError(
                        f"DL3DV frames not found: {frames}. Expected data under {rgb_dir}."
                    )
                video_paths.append(rgb_dir)
                video_pts.append(torch.arange(len(frames), dtype=torch.long))
                video_fps.append(30.0)

        metadata: Dict[str, Any] = {
            "video_paths": video_paths,
            "video_pts": video_pts,
            "video_fps": video_fps,
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
        frame_files = sorted(rgb_dir.glob("*.png"))
        if end_frame is None:
            end_frame = len(frame_files)
        frame_files = frame_files[start_frame:end_frame]

        frames: List[torch.Tensor] = []
        for fp in frame_files:
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
        Override to support non-square resizing using optional cfg fields
        `resolution_height` and `resolution_width`. Falls back to square `resolution`.
        """
        height = getattr(self.cfg, "resolution_height", self.resolution)
        width = getattr(self.cfg, "resolution_width", self.resolution)
        return VideoTransform((height, width))


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

        frame_files = sorted(rgb_dir.glob("*.png"))
        sel_frames = frame_files[start_frame:end_frame]
        if len(sel_frames) != T:
            raise ValueError(f"Selected RGB frames ({len(sel_frames)}) != T ({T})")

        cams: List[torch.Tensor] = []
        for rgb_fp in sel_frames:
            stem = rgb_fp.stem  # e.g., frame_00001
            parts = stem.split("_")
            if len(parts) < 2:
                raise ValueError(f"Unexpected RGB filename: {rgb_fp.name}")
            frame_id = parts[-1]
            cam_fp = cam_dir / f"frame_{frame_id}.npz"
            if not cam_fp.exists():
                raise FileNotFoundError(f"Expected cam file not found: {cam_fp}")
            npz = np.load(cam_fp)
            arrays = [np.asarray(npz[k]).reshape(-1) for k in sorted(npz.files)]
            vec = np.concatenate(arrays, axis=0) if len(arrays) > 0 else np.empty((0,), dtype=np.float32)
            cams.append(torch.as_tensor(vec, dtype=torch.float32))

        cams_tensor = torch.stack(cams, dim=0)
        # Pad/truncate to external_cond_dim
        current_D = cams_tensor.shape[-1]
        if current_D < D:
            cams_tensor = torch.nn.functional.pad(cams_tensor, (0, D - current_D))
        elif current_D > D:
            cams_tensor = cams_tensor[:, :D]
        return cams_tensor


