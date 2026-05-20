"""
recording.py — Capture simulation frames and write MP4 (or GIF fallback).
"""

from __future__ import annotations

import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.data_layout import SIMULATION_GIF, SIMULATION_MP4


class SimulationRecorder:
    """Accumulates PNG frames from matplotlib figures, then writes a video file."""

    def __init__(self, dpi: int = 80):
        self.dpi = dpi
        self._frames: list[np.ndarray] = []

    def capture_figure(self, fig) -> None:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=self.dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
        buf.seek(0)
        try:
            import imageio.v3 as iio
            frame = iio.imread(buf)
        except Exception:
            buf.seek(0)
            from PIL import Image
            frame = np.array(Image.open(buf).convert("RGB"))
        self._frames.append(frame)

    @property
    def frame_count(self) -> int:
        return len(self._frames)

    def save(self, run_dir: Path, fps: float = 10.0) -> Path | None:
        if not self._frames:
            print("[recording] Warning: no frames captured; video not saved.")
            return None
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        mp4_path = run_dir / SIMULATION_MP4
        gif_path = run_dir / SIMULATION_GIF
        duration_ms = max(1, int(1000.0 / fps))

        try:
            import imageio.v3 as iio
            iio.imwrite(mp4_path, self._frames, fps=fps, codec="libx264", pixelformat="yuv420p")
            return mp4_path
        except Exception as exc:
            print(f"[recording] MP4 save failed ({exc}); trying GIF...")

        try:
            import imageio.v3 as iio
            iio.imwrite(gif_path, self._frames, fps=fps, loop=0)
            return gif_path
        except Exception:
            pass

        try:
            from PIL import Image
            images = [Image.fromarray(frame) for frame in self._frames]
            images[0].save(
                gif_path,
                save_all=True,
                append_images=images[1:],
                duration=duration_ms,
                loop=0,
            )
            return gif_path
        except Exception as exc:
            print(f"[recording] Warning: could not save video ({exc}).")
            print("  Install: pip install imageio imageio-ffmpeg")
            return None

    def clear(self) -> None:
        self._frames.clear()


def render_and_capture(world, title: str, recorder: SimulationRecorder | None, dpi: int = 80):
    """Render one simulation step; optionally append to recorder."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    world.render(ax=ax, title=title)
    if recorder is not None:
        recorder.capture_figure(fig)
    plt.close(fig)
