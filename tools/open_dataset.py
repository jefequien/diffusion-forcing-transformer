from typing import Optional
from pathlib import Path
import argparse

from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datasets.video.realestate10k import RealEstate10KAdvancedVideoDataset
from datasets.video.dl3dv import DL3DVAdvancedVideoDataset


def _load_dataset_cfg(dataset: Optional[str] = None) -> DictConfig:
    """
    Load dataset config by merging base_video.yaml with the selected dataset YAML.
    Defaults to configurations/dataset/realestate10k.yaml; uses dl3dv.yaml when dataset="dl3dv".
    """
    repo_root = Path(__file__).resolve().parents[1]
    base_cfg_path = repo_root / "configurations" / "dataset" / "base_video.yaml"
    default_cfg_path = repo_root / "configurations" / "dataset" / "realestate10k.yaml"
    dl3dv_cfg_path = repo_root / "configurations" / "dataset" / "dl3dv.yaml"

    cfg_base = OmegaConf.load(base_cfg_path)
    cfg_target_path = dl3dv_cfg_path if (dataset and dataset.lower() == "dl3dv") else default_cfg_path
    cfg_ds = OmegaConf.load(cfg_target_path)
    cfg: DictConfig = OmegaConf.merge(cfg_base, cfg_ds)
    # Clean up Hydra defaults key if present
    if "defaults" in cfg:
        del cfg["defaults"]
    # Ensure required derived fields when not resolved via Hydra interpolation
    # Avoid evaluating ${dataset.max_frames} by setting explicitly
    cfg.n_frames = cfg.max_frames
    return cfg


def _select_dataset_cls(dataset_override: Optional[str]):
    """
    Select dataset class based on override or config filename.
    """
    if dataset_override:
        key = dataset_override.lower()
        if key in ("dl3dv", "dl3d", "dl3dv_dataset"):
            return DL3DVAdvancedVideoDataset
        return RealEstate10KAdvancedVideoDataset
    return RealEstate10KAdvancedVideoDataset


def open_dataset(
    split: str = "training",
    current_epoch: Optional[int] = None,
    dataset: Optional[str] = None,
) -> RealEstate10KAdvancedVideoDataset:
    """
    Minimal helper to open the RealEstate10K dataset with sensible defaults.

    Args:
        split: One of "training", "validation", or "test".
        current_epoch: Optional sub-epoch index for subdataset scheduling.

    Returns:
        An instantiated RealEstate10KAdvancedVideoDataset.
    """
    # Load merged config
    cfg = _load_dataset_cfg(dataset)

    if split == "validation":
        # The dataset implementation internally remaps validation to test.
        split = "validation"

    dataset_cls = _select_dataset_cls(dataset)
    return dataset_cls(cfg=cfg, split=split, current_epoch=current_epoch)


def _tensor_frame_to_pil(frame: torch.Tensor) -> Image.Image:
    """
    Convert a frame tensor (C, H, W) in [0,1] float to a PIL Image.
    """
    assert frame.ndim == 3 and frame.shape[0] in (1, 3), "Expected (C, H, W)"
    frame_uint8 = (frame.clamp(0, 1) * 255).to(torch.uint8)
    if frame_uint8.shape[0] == 1:
        arr = frame_uint8[0].cpu().numpy()
        return Image.fromarray(arr, mode="L").convert("RGB")
    arr = frame_uint8.permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(arr, mode="RGB")


def _video_strip_horizontal(video: torch.Tensor) -> Image.Image:
    """
    Convert a video tensor (T, C, H, W) to a single horizontal strip image (H, T*W).
    """
    assert video.ndim == 4, "Expected (T, C, H, W)"
    T, _, H, W = video.shape
    strip = Image.new("RGB", (T * W, H))
    for t, frame in enumerate(video):
        pil = _tensor_frame_to_pil(frame)
        strip.paste(pil, (t * W, 0))
    return strip


def _tensor_video_to_gif_frames(video: torch.Tensor) -> list[Image.Image]:
    """
    Convert video (T, C, H, W) in [0,1] to list of PIL Images for GIF saving.
    """
    assert video.ndim == 4 and video.shape[1] in (1, 3), "Expected (T, C, H, W)"
    frames: list[Image.Image] = []
    for frame in video:
        frames.append(_tensor_frame_to_pil(frame))
    return frames


def _batch_videos_to_gif_frames(videos: torch.Tensor) -> list[Image.Image]:
    """
    Convert a batch of videos (B, T, C, H, W) into per-time frames by stacking
    samples vertically for each timestep.
    """
    assert videos.ndim == 5 and videos.shape[2] in (1, 3), "Expected (B, T, C, H, W)"
    B, T, C, H, W = videos.shape
    frames: list[Image.Image] = []
    for t in range(T):
        # Build vertical strip of all samples at time t
        canvas = Image.new("RGB", (W, B * H))
        for b in range(B):
            frame = videos[b, t]
            pil = _tensor_frame_to_pil(frame)
            canvas.paste(pil, (0, b * H))
        frames.append(canvas)
    return frames


def _pca_3d(traj: torch.Tensor) -> "tuple[np.ndarray, np.ndarray]":
    """
    Simple PCA to 3D for a (T, D) tensor. Returns (proj(T,3), mean(D)).
    """
    x = traj.detach().cpu().float()
    x = x - x.mean(dim=0, keepdim=True)
    # (D, D) covariance via SVD on (T, D)
    u, s, v = torch.pca_lowrank(x, q=min(3, x.shape[1])) if hasattr(torch, "pca_lowrank") else (None, None, None)
    if v is None:
        # fallback using SVD
        # x = U S V^T, components are V[:,:3]
        _, _, v_full = torch.linalg.svd(x, full_matrices=False)
        v = v_full[:, : min(3, v_full.shape[1])]
    proj = x @ v[:, : min(3, v.shape[1])]
    proj_np = proj.cpu().numpy()
    mean_np = traj.mean(dim=0).cpu().numpy()
    if proj_np.shape[1] == 1:
        proj_np = np.concatenate([proj_np, np.zeros_like(proj_np), np.zeros_like(proj_np)], axis=1)
    elif proj_np.shape[1] == 2:
        proj_np = np.concatenate([proj_np, np.zeros((proj_np.shape[0], 1))], axis=1)
    return proj_np, mean_np


def _save_pose_plot(conds: torch.Tensor, out_path: Path, title: str = "") -> None:
    """
    Visualize pose-like trajectories from a (T, D) tensor using 3D PCA embedding.
    """
    if conds is None or conds.numel() == 0:
        return
    if conds.ndim == 3:
        # pick the first sample
        conds = conds[0]
    proj3d, _ = _pca_3d(conds)
    fig = plt.figure(figsize=(4, 3))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(proj3d[:, 0], proj3d[:, 1], proj3d[:, 2], marker="o", markersize=2, linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_first_batches(
    output_dir: str,
    split: str = "training",
    batch_size: int = 4,
    num_batches: int = 10,
    num_workers: int = 4,
    dataset: Optional[str] = None,
) -> None:
    """
    Open the RealEstate10K dataset and save the first `num_batches` batches to `output_dir` as PNGs.

    For each batch, frames are stacked horizontally per sample, and samples are
    stacked vertically, saved as `batch_{bbb}.png`.
    """
    ds = open_dataset(split=split, dataset=dataset)

    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for batch_idx, batch in enumerate(dl):
        if batch_idx >= num_batches:
            break
        if "videos" not in batch:
            continue
        videos = batch["videos"]  # (B, T, C, H, W)
        print(f"Saving {videos.shape[0]} videos from batch {batch_idx} with shape {videos.shape}")
        assert videos.ndim == 5, "Expected videos of shape (B, T, C, H, W)"
        B, T, C, H, W = videos.shape
        # Build horizontal strip per sample
        strips = [_video_strip_horizontal(videos[i]) for i in range(B)]
        # Compose final image stacking strips vertically
        canvas = Image.new("RGB", (T * W, B * H))
        for i, strip in enumerate(strips):
            canvas.paste(strip, (0, i * H))
        png_path = out_path / f"batch_{batch_idx:03d}.png"
        canvas.save(png_path, format="PNG")

        # Also save the entire batch as a vertically stacked GIF over time
        batch_gif_frames = _batch_videos_to_gif_frames(videos)
        if len(batch_gif_frames) > 0:
            gif_path = out_path / f"batch_{batch_idx:03d}.gif"
            batch_gif_frames[0].save(
                gif_path,
                format="GIF",
                save_all=True,
                append_images=batch_gif_frames[1:],
                duration=1000 // 10,  # ~10 FPS playback
                loop=0,
                optimize=False,
            )

        # Save pose visualization; raise if poses not found
        if "conds" not in batch or batch["conds"] is None or (hasattr(batch["conds"], "numel") and batch["conds"].numel() == 0):
            raise RuntimeError("Pose conditioning `conds` not found or empty in batch. Ensure external_cond_dim > 0 and data provides poses.")
        pose_path = out_path / f"batch_{batch_idx:03d}_poses.png"
        _save_pose_plot(batch["conds"], pose_path, title=f"batch {batch_idx} poses")


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Save first N batches (as PNG grids)")
    parser.add_argument("--output_dir", type=str, default="outputs/open_dataset", help="Directory to save PNGs")
    parser.add_argument("--split", type=str, default="training", choices=["training", "validation", "test"], help="Dataset split")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        choices=["realestate10k", "dl3dv"],
        help="Dataset selector",
    )
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--num_batches", type=int, default=10, help="Number of batches to save as PNGs")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    return parser


if __name__ == "__main__":
    args = _build_argparser().parse_args()
    save_first_batches(
        output_dir=args.output_dir,
        split=args.split,
        batch_size=args.batch_size,
        num_batches=args.num_batches,
        num_workers=args.num_workers,
        dataset=args.dataset,
    )


