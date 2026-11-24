"""Recount the saved feature-selection experiment and export changed queries."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from benchmark import write_csv
from dataset_cache import write_json
from validate_tolerances import read_rows, statistics


def run(path):
    """Independently recount every saved sign, validity mask, and baseline transition."""
    config = json.loads((path / "config.json").read_text())
    summary = json.loads((path / "summary.json").read_text())
    reference_path, baseline_path = Path(config["reference"]), Path(config["baseline"])
    rows = read_rows(path / "per_model.csv")
    indexed = {(r["file_id"], r["precision"]): r for r in rows}
    ids = config["selected_ids"]
    np.testing.assert_equal(
        set(indexed), {(fid, name) for fid in ids for name in ("float32", "float64")}
    )
    totals = {name: Counter() for name in ("float32", "float64")}
    all_point_transitions = {name: Counter() for name in totals}
    changes, introduced, remaining = [], [], []
    for count, fid in enumerate(ids):
        with np.load(reference_path / "points" / f"{fid}.npz") as data:
            truth, away = data["sign_w"], data["away_from_surface"]
        with np.load(path / "points" / f"{fid}.npz") as data:
            current = dict(data)
        with np.load(baseline_path / "points" / f"{fid}.npz") as data:
            baseline = {
                k: data[k]
                for k in data.files
                if k.endswith("tuned_sign_p") or k.endswith("tuned_valid_query")
            }
        for name in totals:
            sign, valid = current[f"{name}_sign_p"], current[f"{name}_valid"]
            normal, bary = current[f"{name}_normal"], current[f"{name}_barycentric"]
            old, old_valid = baseline[f"{name}_tuned_sign_p"], baseline[f"{name}_tuned_valid_query"]
            old_wrong, new_wrong = old != truth, sign != truth
            all_point_transitions[name].update(
                {
                    "old_disagreements": int(old_wrong.sum()),
                    "new_disagreements": int(new_wrong.sum()),
                    "introduced": int((~old_wrong & new_wrong).sum()),
                    "resolved": int((old_wrong & ~new_wrong).sum()),
                    "introduced_within_cutoff": int((~old_wrong & new_wrong & ~away).sum()),
                    "resolved_within_cutoff": int((old_wrong & ~new_wrong & ~away).sum()),
                }
            )
            bary_valid = (
                np.isfinite(bary).all(axis=1) & (bary >= 0).all(axis=1) & (bary.sum(axis=1) > 0)
            )
            np.testing.assert_array_equal(valid, bary_valid & np.isfinite(normal).all(axis=1))
            metrics = statistics(sign, valid, truth, away, old, old_valid)
            metrics.update(
                {
                    "baseline_disagreements": int(((old != truth) & away).sum()),
                    "sign_changes": int((old != sign).sum()),
                    "invalid_barycentrics": int((~bary_valid).sum()),
                    "end_to_end_models": int(fid in config["end_to_end_ids"]),
                }
            )
            for key, value in metrics.items():
                np.testing.assert_equal(
                    value, int(indexed[fid, name][key]), err_msg=f"{fid}/{name}/{key}"
                )
            totals[name].update(metrics)
            totals[name]["models"] += 1
            for i in np.flatnonzero((sign != old) | ~valid | ((sign != truth) & away)):
                row = {
                    "file_id": fid,
                    "sample_index": int(i),
                    "precision": name,
                    "away": bool(away[i]),
                    "old_sign": int(old[i]),
                    "new_sign": int(sign[i]),
                    "winding_sign": int(truth[i]),
                    "old_valid": bool(old_valid[i]),
                    "new_valid": bool(valid[i]),
                    "introduced": bool(away[i] and old[i] == truth[i] and sign[i] != truth[i]),
                    "resolved": bool(away[i] and old[i] != truth[i] and sign[i] == truth[i]),
                }
                changes.append(row)
                if row["introduced"]:
                    introduced.append(row)
                if away[i] and sign[i] != truth[i]:
                    remaining.append(row)
        if (count + 1) % 1000 == 0:
            print(f"Audited {count + 1}/{len(ids)}", flush=True)
    for name, values in totals.items():
        np.testing.assert_equal(dict(values), summary["totals"]["all"][name])
    write_csv(path / "changed_and_remaining_points.csv", changes)
    write_json(path / "introduced.json", introduced)
    write_csv(path / "remaining_points.csv", remaining)
    audit = {
        "models": len(ids),
        "all_metrics_recounted": True,
        "totals": totals,
        "all_point_transitions": all_point_transitions,
        "introduced_models": {
            name: sorted({r["file_id"] for r in introduced if r["precision"] == name}, key=int)
            for name in totals
        },
        "remaining_models": {
            name: sorted({r["file_id"] for r in remaining if r["precision"] == name}, key=int)
            for name in totals
        },
    }
    write_json(path / "audit.json", audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    run(parser.parse_args().results)
