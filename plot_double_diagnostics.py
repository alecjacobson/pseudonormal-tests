"""Plot the diagnosed source of every tuned-double disagreement by model."""

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib
import numpy as np

from benchmark import write_csv

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def run(path):
    """Export the per-model diagnosis and a labeled comparison figure."""
    cases = json.loads((path / "cases.json").read_text())
    columns = {
        "wrong_edge": "Classifier: wrong edge",
        "face_instead_of_edge": "Classifier: face instead of edge",
        "nearest_feature": "Nearest-feature roundoff",
        "residual_geometry": "Residual on self-intersecting mesh",
    }
    counts = {}
    for row in cases:
        if row["nearest_encoded_feature_corrects"]:
            label = "wrong_edge" if row["normal_kind"] == "edge" else "face_instead_of_edge"
        elif row["global_feature_stored_normal_corrects"]:
            label = "nearest_feature"
        else:
            label = "residual_geometry"
        counts.setdefault(row["file_id"], Counter())[label] += 1
    ids = sorted(counts, key=lambda k: -sum(counts[k].values()))
    rows = [
        {
            "file_id": file_id,
            "disagreements": sum(counts[file_id].values()),
            **{key: counts[file_id][key] for key in columns},
        }
        for file_id in ids
    ]
    write_csv(path / "diagnosis_by_model.csv", rows)
    fig, ax = plt.subplots(figsize=(10, 7.5), constrained_layout=True)
    left = np.zeros(len(ids))
    colors = ["#3479ac", "#62b3aa", "#e49e3e", "#bd5a64"]
    for (key, label), color in zip(columns.items(), colors, strict=True):
        values = np.array([counts[file_id][key] for file_id in ids])
        ax.barh(ids, values, left=left, label=label, color=color)
        left += values
    for i, total in enumerate(left):
        ax.text(total + 2, i, str(int(total)), va="center", fontsize=9)
    ax.invert_yaxis()
    ax.set(
        xlim=(0, max(left) * 1.12),
        xlabel="Disagreeing samples (same distance cutoff)",
        ylabel="Thingi10K file ID",
        title="441 tuned-double disagreements on 17 models",
    )
    ax.legend(loc="lower right", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(path / "diagnosis.png", dpi=180)
    fig.savefig(path / "diagnosis.pdf")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    run(parser.parse_args().path)
