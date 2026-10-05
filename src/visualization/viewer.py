"""3D viewer for one segmented scan.

    python -m src.visualization.viewer --case INF-20261005-abc123
    python -m src.visualization.viewer --case INF-... --screenshot viewer.png

Left: the 3D meshes with three slices of the scan. Rotate with the left mouse
button, zoom with the wheel, pan with shift + left mouse. The boxes turn each
structure and the slices on and off, the slider changes opacity, and "Measure"
lets you click two points to get a distance in mm.

Right: one slice of the scan with the segmentation on top. The slider scrolls
through the slices.
"""

from __future__ import annotations

import argparse
import ctypes
import sys
from pathlib import Path

import numpy as np
import pyvista as pv

from src.lineage.records import ArtifactPaths
from src.visualization.scene import CaseScene, load_case

pv.global_theme.allow_empty_mesh = True  # a slice may contain no segmentation at all

BACKGROUND = "#10182b"
TEXT = "#e8ecf4"


def _screen_setup() -> tuple[float, tuple[int, int]]:
    """Return (display scale, window size in pixels).

    On Windows with display scaling (e.g. 150%), VTK windows are stretched by the
    system and look blurry. Asking for real pixels fixes that; fonts and buttons
    are then multiplied by the scale so they keep their normal size.
    """
    if sys.platform != "win32":
        return 1.0, (1600, 900)
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
        scale = ctypes.windll.user32.GetDpiForSystem() / 96
        width = ctypes.windll.user32.GetSystemMetrics(0)
        height = ctypes.windll.user32.GetSystemMetrics(1)
    except (AttributeError, OSError):
        return 1.0, (1600, 900)
    return scale, (int(width * 0.9), int(height * 0.85))


def build_plotter(scene: CaseScene, off_screen: bool = False) -> pv.Plotter:
    scale, window_size = (1.0, (1600, 900)) if off_screen else _screen_setup()
    plotter = pv.Plotter(
        shape=(1, 2),
        window_size=list(window_size),
        off_screen=off_screen,
        title=f"3D Medical Imaging ML Pipeline - {scene.inference_id}",
    )
    plotter.set_background(BACKGROUND)
    plotter.enable_anti_aliasing("msaa")  # smooth 3D edges, keeps text sharp
    _add_3d_view(plotter, scene, scale)
    _add_slice_view(plotter, scene, scale)
    plotter.subplot(0, 0)
    return plotter


def _add_3d_view(plotter: pv.Plotter, scene: CaseScene, scale: float) -> None:
    plotter.subplot(0, 0)
    actors = {
        name: plotter.add_mesh(
            mesh,
            color=scene.colors[name],
            smooth_shading=True,
            specular=0.5,
            specular_power=20,
            name=name,
        )
        for name, mesh in scene.meshes.items()
    }
    x, y, z = _segmentation_center(scene)
    slices = scene.image.slice_orthogonal(x=x, y=y, z=z)
    slice_actor = plotter.add_mesh(
        # Faint, so the structures stay in front and the slices only give context.
        slices,
        cmap="gray",
        show_scalar_bar=False,
        opacity=0.3,
        name="scan_slices",
    )

    # One checkbox per structure, plus one for the scan slices.
    toggles = [(name, actors[name], scene.colors[name]) for name in scene.meshes]
    toggles.append(("scan slices", slice_actor, "#9aa3b5"))
    box, row_height, margin = int(24 * scale), int(34 * scale), int(12 * scale)
    label_x = margin + box + int(10 * scale)
    font = int(8 * scale)
    for row, (label, actor, color) in enumerate(toggles):
        y = margin + row * row_height
        plotter.add_checkbox_button_widget(
            lambda visible, a=actor: a.SetVisibility(visible),
            value=True,
            position=(margin, y),
            size=box,
            color_on=color,
            color_off="#3a4255",
        )
        info = scene.mesh_info.get(label)
        suffix = f"  ({info['mesh_volume_ml']:.2f} mL)" if info else ""
        plotter.add_text(
            f"{label}{suffix}", position=(label_x, y + box // 6), font_size=font, color=TEXT
        )

    def set_opacity(value: float) -> None:
        for actor in actors.values():
            actor.GetProperty().SetOpacity(value)

    plotter.add_slider_widget(
        set_opacity,
        rng=[0.1, 1.0],
        value=1.0,
        title="Structure opacity",
        pointa=(0.66, 0.08),
        pointb=(0.96, 0.08),
        style="modern",
        color=TEXT,
    )

    distance_text = plotter.add_text("", position="lower_right", font_size=font, color="#ffd166")
    ruler = plotter.add_measurement_widget(
        lambda a, b, d: distance_text.SetText(3, f"distance: {d:.1f} mm"), color="#ffd166"
    )
    ruler.Off()
    y = margin + len(toggles) * row_height
    plotter.add_checkbox_button_widget(
        lambda on: ruler.On() if on else ruler.Off(),
        value=False,
        position=(margin, y),
        size=box,
        color_on="#ffd166",
        color_off="#3a4255",
    )
    plotter.add_text(
        "Measure (click two points)", position=(label_x, y + box // 6), font_size=font, color=TEXT
    )

    info_text = plotter.add_text(
        "\n".join(scene.summary_lines()),
        position="upper_left",
        font_size=int(10 * scale),
        color=TEXT,
        font="courier",
    )
    info_text.GetTextProperty().SetBold(True)  # thin monospace text is hard to read
    plotter.add_axes(color=TEXT, viewport=(0.82, 0.78, 1.0, 1.0))
    plotter.camera_position = "iso"
    if scene.meshes:  # frame the anatomy, not the whole scan
        plotter.reset_camera(bounds=pv.merge(list(scene.meshes.values())).bounds)
        plotter.camera.zoom(0.9)
    else:
        plotter.reset_camera()


def _add_slice_view(plotter: pv.Plotter, scene: CaseScene, scale: float) -> None:
    plotter.subplot(0, 1)
    nx, ny, nz = scene.image.dimensions
    colors = [scene.colors[name] for name in scene.meshes] or ["#e8743b"]

    def show_slice(value: float) -> None:
        k = int(round(value))
        extent = (0, nx - 1, 0, ny - 1, k, k)
        plotter.add_mesh(
            scene.image.extract_subset(extent), cmap="gray", name="slice", show_scalar_bar=False
        )
        voxel_layer = (0, nx, 0, ny, k, k + 1)  # the label grid has one more point per axis
        overlay = scene.labels.extract_subset(voxel_layer).threshold(0.5, scalars="label")
        plotter.add_mesh(
            overlay,
            scalars="label",
            cmap=colors,
            clim=[1, max(len(colors), 2)],
            opacity=0.45,
            name="overlay",
            show_scalar_bar=False,
        )

    start = int(round(_segmentation_center_index(scene)[2]))
    show_slice(start)
    plotter.add_slider_widget(
        show_slice,
        rng=[0, nz - 1],
        value=start,
        title="Slice",
        fmt="%.0f",
        pointa=(0.08, 0.08),
        pointb=(0.92, 0.08),
        style="modern",
        color=TEXT,
    )
    plotter.add_text(
        "Scan + predicted segmentation",
        position="upper_left",
        font_size=int(12 * scale),
        color=TEXT,
    )
    plotter.view_xy()
    plotter.reset_camera()


def _segmentation_center_index(scene: CaseScene) -> np.ndarray:
    zyx = np.argwhere(scene.label_array > 0)
    center_zyx = zyx.mean(axis=0) if len(zyx) else np.array(scene.label_array.shape) / 2
    return center_zyx[::-1]  # (x, y, z) indices


def _segmentation_center(scene: CaseScene) -> tuple[float, float, float]:
    """Physical point where the three scan slices cross: the centre of the structures."""
    if not scene.meshes:
        return tuple(scene.image.center)
    points = np.vstack([mesh.points for mesh in scene.meshes.values()])
    x, y, z = points.mean(axis=0)
    return float(x), float(y), float(z)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Interactive 3D viewer for one inference case.")
    parser.add_argument("--case", "--inference-id", dest="case", required=True)
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="scan location if it moved since inference (hash-checked)",
    )
    parser.add_argument(
        "--screenshot",
        type=Path,
        default=None,
        help="render off-screen to this PNG instead of opening a window",
    )
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
