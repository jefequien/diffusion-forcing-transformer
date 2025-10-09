from typing import Dict, Any, Optional, List

import torch
from omegaconf import DictConfig

from .base_video import (
    BaseVideoDataset,
    BaseSimpleVideoDataset,
    BaseAdvancedVideoDataset,
    SPLIT,
)


class DL3DVBaseVideoDataset(BaseVideoDataset):
    """
    DL3DV base video dataset skeleton.

    Expected folder structure:
    - {save_dir}
        - /training | /validation | /test
            - video files (e.g., .mp4)
        - /metadata
            - {split}.pt (auto-generated)

    Override methods here as the dataset specifics become available
    (e.g., custom download, preprocessing, transforms).
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
        # No on-disk metadata building for the stubbed dataset.
        return

    def setup(self) -> None:
        # Identity transform by default; replace with dataset-specific transforms if needed.
        self.transform = lambda x: x

    def load_metadata(self) -> List[Dict[str, Any]]:
        """
        Provide 100 synthetic videos per split, each with 100 frames.
        """
        num_videos = 100
        frames_per_video = 100  # >= typical n_frames (e.g., 71) so clips exist
        fps = 10.0
        split_dir = self.save_dir / self.split
        metadata: List[Dict[str, Any]] = []
        for i in range(num_videos):
            video_path = split_dir / f"synthetic_{i:05d}.mp4"
            video_pts = torch.arange(frames_per_video, dtype=torch.long)
            metadata.append(
                {
                    "video_paths": video_path,
                    "video_pts": video_pts,
                    "video_fps": fps,
                }
            )
        return metadata

    def load_video(
        self,
        video_metadata: Dict[str, Any],
        start_frame: int,
        end_frame: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Return a zero video tensor of shape (T, C, H, W) in [0, 1].
        """
        if end_frame is None:
            end_frame = len(video_metadata["video_pts"])  # type: ignore[arg-type]
        T = max(0, end_frame - start_frame)
        C, H, W = 3, self.cfg.resolution, self.cfg.resolution
        return torch.zeros((T, C, H, W), dtype=torch.float32)


class DL3DVSimpleVideoDataset(DL3DVBaseVideoDataset, BaseSimpleVideoDataset):
    """
    DL3DV simple dataset: loads full videos and returns video tensors plus target latent paths.
    """

    def __init__(self, cfg: DictConfig, split: SPLIT = "training"):
        BaseSimpleVideoDataset.__init__(self, cfg, split)
        self.setup()


class DL3DVAdvancedVideoDataset(DL3DVBaseVideoDataset, BaseAdvancedVideoDataset):
    """
    DL3DV advanced dataset: loads variable-length clips with frame skipping.

    If external conditioning is used (cfg.external_cond_dim > 0), implement load_cond
    to return a tensor of shape (T, D). For now, returns zeros when requested.
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
        # Skeleton: return zeros if conditioning is expected; otherwise unused.
        T = end_frame - start_frame
        D = getattr(self.cfg, "external_cond_dim", 0) or 0
        if D == 0:
            # Ensure shape compatibility even if not used downstream.
            return torch.zeros((T, 0), dtype=torch.float32)
        return torch.zeros((T, D), dtype=torch.float32)


