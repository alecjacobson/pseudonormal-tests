"""Check full-mask coverage, checkpoint exports, and reproducibility of the initial subset."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from dataset_cache import DEFAULT_CACHE, prepare_dataset, write_json


def read_rows(path):
    """Read a result table indexed by its original dataset file ID."""
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    indexed = {row["file_id"]: row for row in rows}
    np.testing.assert_equal(len(rows), len(indexed))
    return indexed


def validate(baseline, warp, initial, initial_warp, cache_dir):
    """Verify every eligible ID and compare the reused subset point for point."""
    config = json.loads((baseline / "config.json").read_text())
    initial_config = json.loads((initial / "config.json").read_text())
    for key in ("dataset_revision", "seed", "points_per_model", "relative_cutoff", "filter"):
        np.testing.assert_equal(config[key], initial_config[key])
    metadata_path, _, _ = prepare_dataset(cache_dir, config["dataset_revision"])
    metadata = read_rows(metadata_path)
    eligible = {
        file_id
        for file_id, row in metadata.items()
        if all(row[key] == "1" for key in ("solid", "vertex_manifold", "edge_manifold"))
    }
    baseline_rows = read_rows(baseline / "per_model.csv")
    warp_rows = read_rows(warp / "per_model.csv")
    np.testing.assert_equal(set(baseline_rows), eligible)
    np.testing.assert_equal(set(warp_rows), eligible)
    np.testing.assert_equal(config["models"], len(eligible))
    for output, rows in ((baseline, baseline_rows), (warp, warp_rows)):
        with (output / "per_model.jsonl").open() as stream:
            journal = [json.loads(line) for line in stream]
        np.testing.assert_equal(len(journal), len(rows))
        np.testing.assert_equal([row["file_id"] for row in journal], list(rows))
        for record in journal:
            exported = rows[record["file_id"]]
            np.testing.assert_equal(record["status"], "ok")
            for key, value in record.items():
                np.testing.assert_equal(exported[key], "" if value is None else str(value))
    initial_rows = read_rows(initial / "per_model.csv")
    initial_warp_rows = read_rows(initial_warp / "per_model.csv")
    np.testing.assert_equal(set(initial_rows), set(initial_warp_rows))
    checks = {
        "libigl": ("points", "d", "sign_p", "winding", "sign_w"),
        "warp": (
            "warp_points",
            "warp_d_p",
            "warp_d_w",
            "warp_d_parity",
            "warp_sign_p",
            "warp_sign_w",
            "warp_sign_parity",
        ),
    }
    for file_id, initial_row in initial_rows.items():
        for method, previous, current, previous_row, current_row in (
            ("libigl", initial, baseline, initial_row, baseline_rows[file_id]),
            ("warp", initial_warp, warp, initial_warp_rows[file_id], warp_rows[file_id]),
        ):
            with np.load(previous / previous_row["points_file"], allow_pickle=False) as old:
                with np.load(current / current_row["points_file"], allow_pickle=False) as new:
                    for key in checks[method]:
                        np.testing.assert_array_equal(
                            new[key], old[key], err_msg=f"{file_id}: {key}"
                        )
    result = {
        "result": "passed",
        "eligible_models": len(eligible),
        "models_in_each_full_run": len(baseline_rows),
        "checkpoint_rows_checked": len(baseline_rows) + len(warp_rows),
        "initial_models_compared": len(initial_rows),
        "initial_points_compared": len(initial_rows) * config["points_per_model"],
        "exactly_reproduced_arrays": checks,
        "seed": config["seed"],
    }
    write_json(warp / "full_set_validation.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=Path("results/full-4963"))
    parser.add_argument("--warp", type=Path, default=Path("results/warp-main-full-4963"))
    parser.add_argument("--initial", type=Path, default=Path("results/initial-100"))
    parser.add_argument("--initial-warp", type=Path, default=Path("results/warp-main-100"))
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    args = parser.parse_args()
    validate(args.baseline, args.warp, args.initial, args.initial_warp, args.cache_dir)
