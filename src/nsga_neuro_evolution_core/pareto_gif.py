import io
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import imageio.v2 as imageio


@dataclass
class ParetoFrame:
    epoch: Optional[int]
    pop_f: Optional[np.ndarray]
    front_f: Optional[np.ndarray]


class ParetoGifRecorder:
    """
    Collects Pareto population/front data per epoch, then renders a GIF.
    """
    def __init__(
        self,
        output_path: str,
        fps: int = 1,
        repeat_last: bool = True,
        xlabel: str = "Physics Loss",
        ylabel: str = "Data Loss",
        title_prefix: str = "Epoch",
        pop_color: str = "#9E9E9E",
        front_color: str = "#00A86B",
        pop_size: int = 18,
        front_size: int = 32,
        figsize: Tuple[float, float] = (6.0, 4.0),
        dpi: int = 120,
        axes_limits: Optional[List[float]] = None,
    ):
        self.output_path = output_path
        self.fps = fps
        self.repeat_last = repeat_last
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title_prefix = title_prefix
        self.pop_color = pop_color
        self.front_color = front_color
        self.pop_size = pop_size
        self.front_size = front_size
        self.figsize = figsize
        self.dpi = dpi
        self.axes_limits = axes_limits

        self._frames: List[ParetoFrame] = []
        self._last_pop: Optional[np.ndarray] = None
        self._last_front: Optional[np.ndarray] = None

    def _to_numpy(self, arr):
        if arr is None:
            return None
        if hasattr(arr, "detach"):
            return arr.detach().cpu().numpy()
        return np.asarray(arr)

    def _sanitize(self, arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if arr is None:
            return None
        arr = np.asarray(arr)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[1] < 2:
            return None
        arr = arr[:, :2]
        mask = np.isfinite(arr).all(axis=1)
        arr = arr[mask]
        if arr.size == 0:
            return None
        return arr

    def record(self, pop_f=None, front_f=None, epoch: Optional[int] = None) -> None:
        pop_f = self._sanitize(self._to_numpy(pop_f))
        front_f = self._sanitize(self._to_numpy(front_f))

        if pop_f is None and front_f is None:
            if self.repeat_last and self._last_front is not None:
                pop_f = self._last_pop
                front_f = self._last_front
            else:
                return

        self._frames.append(ParetoFrame(epoch=epoch, pop_f=pop_f, front_f=front_f))

        if pop_f is not None:
            self._last_pop = pop_f
        if front_f is not None:
            self._last_front = front_f

    def _compute_bounds(self):
        all_pts = []
        for frame in self._frames:
            if frame.pop_f is not None:
                all_pts.append(frame.pop_f)
            if frame.front_f is not None:
                all_pts.append(frame.front_f)
        if not all_pts:
            return None
        pts = np.vstack(all_pts)
        x = pts[:, 0]
        y = pts[:, 1]
        x_min, x_max = float(np.min(x)), float(np.max(x))
        y_min, y_max = float(np.min(y)), float(np.max(y))
        x_pad = (x_max - x_min) * 0.05
        y_pad = (y_max - y_min) * 0.05
        if x_pad == 0.0:
            x_pad = 1.0
        if y_pad == 0.0:
            y_pad = 1.0
        return (x_min - x_pad, x_max + x_pad), (y_min - y_pad, y_max + y_pad)

    def _render_frame(self, frame: ParetoFrame, bounds=None) -> np.ndarray:
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        if frame.pop_f is not None:
            ax.scatter(
                frame.pop_f[:, 0],
                frame.pop_f[:, 1],
                s=self.pop_size,
                c=self.pop_color,
                alpha=0.5,
                edgecolors="none",
            )
        if frame.front_f is not None:
            ax.scatter(
                frame.front_f[:, 0],
                frame.front_f[:, 1],
                s=self.front_size,
                c=self.front_color,
                alpha=0.9,
                edgecolors="none",
            )

        if frame.epoch is not None:
            ax.set_title(f"{self.title_prefix} {frame.epoch}")
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)
        ax.grid(alpha=0.25)

        if self.axes_limits is not None:
            if len(self.axes_limits) != 2:
                raise ValueError("axes_limits must be a list like [x_max, y_max]")
            ax.set_xlim(0.0, float(self.axes_limits[0]))
            ax.set_ylim(0.0, float(self.axes_limits[1]))
        elif bounds is not None:
            (xmin, xmax), (ymin, ymax) = bounds
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)

        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=self.dpi)
        plt.close(fig)
        buf.seek(0)
        return imageio.imread(buf)

    def save_gif(self) -> Optional[str]:
        if not self._frames:
            return None

        bounds = self._compute_bounds()
        frames = [self._render_frame(frame, bounds=bounds) for frame in self._frames]

        out_dir = os.path.dirname(self.output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        imageio.mimsave(self.output_path, frames, fps=self.fps)
        return self.output_path
