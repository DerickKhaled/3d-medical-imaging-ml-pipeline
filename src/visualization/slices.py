"""Static figure: three orthogonal slices of the scan with the segmentation overlaid.

    python -m src.visualization.slices --case INF-... --out docs/images/slices.png

Useful for reports and reviews where an interactive window is not available.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # file output only; no display needed

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from src.lineage.records import ArtifactPaths  # noqa: E402
from src.visualization.scene import CaseScene, load_case  # noqa: E402


def slice_figure(scene: CaseScene, out: Path) -> Path:
    nx, ny, nz = scene.image.dimensions
    image = np.asarray(scene.image.point_data["intensity"], dtype=np.float32).reshape(nz, ny, nx)
    labels = np.asarray(scene.labels.point_data["label"]).reshape(nz, ny, nx)
    foreground = np.argwhere(labels > 0)
    cz, cy, cx = (foreground.mean(axis=0) if len(foreground) else np.array([nz, ny, nx]) / 2).astype(int)

    names = list(scene.meshes) or list(scene.record["structures"])
    colors = [scene.colors.get(n, "#e8743b") for n in names]
    overlay_cmap = ListedColormap(colors)
    low, high = np.percentile(image, [1, 99])

    views = [("axial (z)", image[cz], labels[cz]),
             ("coronal (y)", image[:, cy], labels[:, cy]),
             ("sagittal (x)", image[:, :, cx], labels[:, :, cx])]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.6), facecolor="#10182b")
    for ax, (title, img, lbl) in zip(axes, views, strict=True):
        ax.imshow(img, cmap="gray", vmin=low, vmax=high, origin="lower")
        masked = np.ma.masked_where(lbl == 0, lbl)
        ax.imshow(masked, cmap=overlay_cmap, vmin=1, vmax=max(len(colors), 2), alpha=0.5,
                  origin="lower", interpolation="nearest")
        ax.set_title(title, color="#e8ecf4")
        ax.axis("off")
    fig.legend(handles=[Patch(color=c, label=n) for n, c in zip(names, colors, strict=True)],
               loc="lower center", ncol=len(names), frameon=False, labelcolor="#e8ecf4")
    fig.suptitle(f"{scene.inference_id}  |  model {scene.record['model_version']}  |  "
                 f"{scene.record['input_name']}", color="#e8ecf4")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Save a slice + overlay figure for one case.")
    parser.add_argument("--case", "--inference-id", dest="case", required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--artifacts-dir", default=Path("artifacts"), type=Path)
    args = parser.parse_args(argv)
    paths = ArtifactPaths(args.artifacts_dir)
    out = args.out or paths.inference_run(args.case) / "slices.png"
    print(f"written: {slice_figure(load_case(args.case, paths), out)}")


if __name__ == "__main__":
    main()
