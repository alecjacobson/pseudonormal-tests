"""Plot a residual skinny-triangle failure with double-built float normals."""

import argparse
from pathlib import Path

import matplotlib
import numpy as np

from benchmark import load_mesh
from dataset_cache import DEFAULT_CACHE, prepare_dataset
from validate_tolerances import read_rows

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def run(path):
    """Show the failing queries and the two different edges chosen for sample 60."""
    _, mesh_dir, _ = prepare_dataset(DEFAULT_CACHE)
    vertices, faces = load_mesh(mesh_dir / "109130.npz")
    rows = read_rows(path / "cases.csv")
    failures = [
        r
        for r in rows
        if r["file_id"] == "109130"
        and r["variant"] == "double_normals_original"
        and r["counted_disagreement"] == "True"
    ]
    points = np.array([[float(r[f"point_{a}"]) for a in "xyz"] for r in failures])
    case = next(r for r in failures if r["sample_index"] == "60")
    face = int(case["closest_face_float"])
    triangle = vertices[faces[face]]
    a, b, c = triangle
    along = (c - a) / np.linalg.norm(c - a)
    up = (b - a) - np.dot(b - a, along) * along
    up /= np.linalg.norm(up)
    local = np.column_stack(((triangle - a) @ along, (triangle - a) @ up))
    closest = np.array([float(case[f"closest_{axis}"]) for axis in "xyz"])
    local_closest = np.array([(closest - a) @ along, (closest - a) @ up])

    fig = plt.figure(figsize=(12, 4.8), constrained_layout=True)
    ax = fig.add_subplot(121, projection="3d")
    ax.add_collection3d(
        Poly3DCollection(vertices[faces], facecolor="steelblue", edgecolor="none", alpha=0.18)
    )
    ax.scatter(
        *points.T,
        color="crimson",
        s=13,
        depthshade=False,
        label=f"{len(points)} disagreeing queries",
    )
    span = np.ptp(vertices, axis=0)
    ax.auto_scale_xyz(vertices[:, 0], vertices[:, 1], vertices[:, 2])
    ax.set_box_aspect(span)
    ax.view_init(elev=35, azim=-65)
    ax.set(xlabel="x", ylabel="y", zlabel="z", title="Model 109130: remaining failures")
    ax.legend(loc="upper left", fontsize=8)

    ax = fig.add_subplot(122)
    for pair, color, label in (
        ([0, 1], "crimson", "Float selects AB"),
        ([0, 2], "royalblue", "Double selects AC"),
        ([1, 2], "gray", None),
    ):
        ax.plot(local[pair, 0], local[pair, 1], color=color, lw=2, label=label)
    ax.scatter(
        local_closest[0], local_closest[1], color="black", zorder=5, label="Float closest point"
    )
    for label, xy in zip("ABC", local, strict=True):
        ax.annotate(label, xy, xytext=(5, 5), textcoords="offset points")
    ax.set(
        xlabel="Along AC (original units)",
        ylabel="Across AC (original units)",
        title="Face 3409: axes use different scales",
    )
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.margins(0.1)
    ax.legend(loc="upper left", fontsize=9)
    fig.suptitle("Double-built normals do not fix float feature selection", fontsize=14)
    fig.savefig(path / "skinny-triangle.png", dpi=180)
    fig.savefig(path / "skinny-triangle.pdf")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    run(parser.parse_args().path)
