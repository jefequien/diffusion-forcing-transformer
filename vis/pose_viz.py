from typing import Optional

import io
import numpy as np
import torch
from PIL import Image
from omegaconf import DictConfig
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vis.camera_pose_visualizer import CameraPoseVisualizer
from einops import rearrange



def save_pose_frustums(mats: torch.Tensor, out_path: str) -> None:
    assert mats is not None and mats.numel() > 0
    if mats.ndim == 3:
        mats = mats.unsqueeze(0)
    # Expecting (B, T, 4, 4)
    assert mats.ndim == 4 and mats.shape[-2:] == (4, 4)
    B, T, _, _ = mats.shape
    per_row_images: list[Image.Image] = []
    for b in range(B):
        mats_np: list[np.ndarray] = []
        for t in range(T):
            m44 = mats[b, t].detach().cpu().float().numpy()
            assert np.all(np.isfinite(m44))
            mats_np.append(m44)
        assert len(mats_np) > 0
        ref_inv = np.linalg.inv(mats_np[0])

        vis = CameraPoseVisualizer(xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1))
        for idx, m44 in enumerate(mats_np):
            m_rel = ref_inv @ m44
            vis.extrinsic2pyramid(
                m_rel,
                color_map=(idx / max(len(mats) - 1, 1)),
                focal_len_scaled=0.2,
                aspect_ratio=0.3,
                plotly_viz=False,
                legend_group='g',
                name=f'f{idx}',
                show_legend=False,
            )
        buf = io.BytesIO()
        vis.fig.tight_layout()
        vis.fig.savefig(buf, format="png")
        plt.close(vis.fig)
        buf.seek(0)
        per_row_images.append(Image.open(buf).convert("RGB"))

    widths = [im.width for im in per_row_images]
    heights = [im.height for im in per_row_images]
    canvas = Image.new("RGB", (max(widths), sum(heights)))
    y = 0
    for im in per_row_images:
        canvas.paste(im, (0, y))
        y += im.height
    canvas.save(out_path)


