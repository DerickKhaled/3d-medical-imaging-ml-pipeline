"""Interactive viewer: 3D reconstruction next to the scan with its segmentation overlay.

    python -m src.visualization.viewer --case INF-20261003-a1b2c3
    python -m src.visualization.viewer --case INF-... --screenshot docs/images/viewer.png

Left:  3D surface meshes in scanner coordinates, with the scan's orthogonal
       slices for anatomical context. Mouse: rotate (left), zoom (wheel), pan (shift+left).
       Checkboxes show/hide each structure and the scan slices; a slider sets opacity;
       the 'Measure' checkbox enables a click-click distance ruler (in mm).
Right: one slice of the scan with the predicted segmentation overlaid; a slider scrolls
       through the slices.

This is not a reproduction of a clinical viewer. It shows how segmentation output
becomes a usable spatial representation, with the provenance of what is shown.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyvista as pv

from src.lineage.records import ArtifactPaths
from src.visualization.scene import CaseScene, load_case

pv.global_theme.allow_empty_mesh = True  # a slice may contain no segmentation at all

BACKGROUND = "#10182b"
TEXT = "#e8ecf4"


def build_plotter(scene: CaseScene, off_screen: bool = False) -> pv.Plotter:
    plotter = pv.Plotter(shape=(1, 2), window_size=(1600, 860), off_screen=off_screen,
                         title=f"3D Medical Imaging ML Pipeline - {scene.inference_id}")
    plotter.set_background(BACKGROUND)
    _add_3d_view(plotter, scene)
    _add_slice_view(plotter, scene)
    plotter.subplot(0, 0)
    return plotter


def _add_3d_view(plotter: pv.Plotter, scene: CaseScene) -> None:
    plotter.subplot(0, 0)
    actors = {
        name: plotter.add_mesh(mesh, color=scene.colors[name], smooth_shading=True,
                               specular=0.3, name=name)
        for name, mesh in scene.meshes.items()
    }
    center = _segmentation_center(scene)
    slices = scene.image.slice_orthogonal(*center)
    slice_actor = plotter.add_mesh(slices, cmap="gray", show_scalar_bar=False, opacity=0.9,
                                   name="scan_slices")

    # One checkbox per structure, plus one for the scan slices.
    toggles = [(name, actors[name], scene.colors[name]) for name in scene.meshes]
    toggles.append(("scan slices", slice_actor, "#9aa3b5"))
    for row, (label, actor, color) in enumerate(toggles):
        y = 12 + row * 42
        plotter.add_checkbox_button_widget(
            lambda visible, a=actor: a.SetVisibility(visible), value=True,
            position=(12, y), size=30, color_on=color, color_off="#3a4255",
        )
        info = scene.mesh_info.get(label)
        suffix = f"  ({info['mesh_volume_ml']:.2f} mL)" if info else ""
        plotter.add_text(f"{label}{suffix}", position=(52, y + 4), font_size=10, color=TEXT)

    def set_opacity(value: float) -> None:
        for actor in actors.values():
            actor.GetProperty().SetOpacity(value)

    plotter.add_slider_widget(
        set_opacity, rng=[0.1, 1.0], value=1.0, title="Structure opacity",
        pointa=(0.62, 0.08), pointb=(0.95, 0.08), style="modern", color=TEXT,
    )

    distance_text = plotter.add_text("", position="lower_right", font_size=10, color="#ffd166")
    ruler = plotter.add_measurement_widget(
        lambda a, b, d: distance_text.SetText(3, f"distance: {d:.1f} mm"), color="#ffd166"
    )
    ruler.Off()
    y = 12 + len(toggles) * 42
    plotter.add_checkbox_button_widget(
        lambda on: ruler.On() if on else ruler.Off(), value=False,
        position=(12, y), size=30, color_on="#ffd166", color_off="#3a4255",
    )
    plotter.add_text("Measure (click two points)", position=(52, y + 4), font_size=10, color=TEXT)

    plotter.add_text("\n".join(scene.summary_lines()), position="upper_left", font_size=9,
                     color=TEXT, font="courier")
    plotter.add_axes(color=TEXT)
    plotter.camera_position = "iso"
    plotter.reset_camera()


def _add_slice_view(plotter: pv.Plotter, scene: CaseScene) -> None:
    plotter.subplot(0, 1)
    nx, ny, nz = scene.image.dimensions
    colors = [scene.colors[name] for name in scene.meshes] or ["#e8743b"]

    def show_slice(value: float) -> None:
        k = int(round(value))
        extent = (0, nx - 1, 0, ny - 1, k, k)
        plotter.add_mesh(scene.image.extract_subset(extent), cmap="gray", name="slice",
                         show_scalar_bar=False)
        overlay = scene.labels.extract_subset(extent).threshold(0.5, scalars="label")
        plotter.add_mesh(overlay, scalars="label", cmap=colors, clim=[1, max(len(colors), 2)],
                         opacity=0.45, name="overlay", show_scalar_bar=False)

    start = int(round(_segmentation_center_index(scene)[2]))
    show_slice(start)
    plotter.add_slider_widget(
        show_slice, rng=[0, nz - 1], value=start, title="Slice", fmt="%.0f",
        pointa=(0.08, 0.08), pointb=(0.92, 0.08), style="modern", color=TEXT,
    )
    plotter.add_text("Scan + predicted segmentation", position="upper_left", font_size=11, color=TEXT)
    plotter.view_xy()
    plotter.reset_camera()


def _segmentation_center_index(scene: CaseScene) -> np.ndarray:
    nx, ny, nz = scene.labels.dimensions
    labels = np.asarray(scene.labels.point_data["label"]).reshape(nz, ny, nx)
    zyx = np.argwhere(labels > 0)
    center_zyx = zyx.mean(axis=0) if len(zyx) else np.array([nz, ny, nx]) / 2
    return center_zyx[::-1]  # (x, y, z) indices


def _segmentation_center(scene: CaseScene) -> tuple[float, float, float]:
    """Physical point where the three scan slices cross: the centre of the structures."""
    if not scene.meshes:
        return tuple(scene.image.center)  # type: ignore[return-value]
    points = np.vstack([mesh.points for mesh in scene.meshes.values()])
    x, y, z = points.mean(axis=0)
    return float(x), float(y), float(z)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Interactive 3D viewer for one inference case.")
    parser.add_argument("--case", "--inference-id", dest="case", required=True)
    parser.add_argument("--image", type=Path, default=None,
                        help="scan location if it moved since inference (hash-checked)")
    parser.add_argument("--screenshot", type=Path, default=None,
                        help="render off-screen to this PNG instead of opening a window")
    parser.add_argument("--artifacts-dir", default=Path("artifacts"), type=Path)
    args = parser.parse_args(argv)

    scene = load_case(args.case, ArtifactPaths(args.artifacts_dir), args.image)
    plotter = build_plotter(scene, off_screen=args.screenshot is not None)
    if args.screenshot:
        args.screenshot.parent.mkdir(parents=True, exist_ok=True)
        plotter.screenshot(str(args.screenshot))
        print(f"screenshot written: {args.screenshot}")
    else:
        plotter.show()


if __name__ == "__main__":
    main()
