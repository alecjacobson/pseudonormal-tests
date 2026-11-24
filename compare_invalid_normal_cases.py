"""Compare all normal variants on the same originally nonfinite float queries."""

import json
from collections import Counter
from pathlib import Path

import numpy as np

from benchmark import write_csv
from dataset_cache import write_json


def run():
    """Join the cached normalization ablation and mixed-normal experiment by sample."""
    baseline = Path("results/libigl-tolerance-validation-4963")
    ablation = Path("results/libigl-normalization-probe-104")
    hybrid = Path("results/libigl-mixed-normals-4963")
    reference = Path("results/full-4963")
    file_ids = json.loads((ablation / "config.json").read_text())["invalid_model_ids"]
    variants = ("normalized_float", "raw_float", "double_rounded", "double_original", "full_double")
    totals = {v: Counter() for v in variants}
    per_model, cases = [], []
    for file_id in file_ids:
        with (
            np.load(baseline / "points" / f"{file_id}.npz") as b,
            np.load(ablation / "points" / f"{file_id}.npz") as n,
            np.load(hybrid / "points" / f"{file_id}.npz") as h,
            np.load(reference / "points" / f"{file_id}.npz") as r,
        ):
            selected = ~b["float32_tuned_valid_query"]
            indices = np.flatnonzero(selected)
            truth = r["sign_w"][selected]
            np.testing.assert_equal(r["away_from_surface"][selected].all(), True)
            np.testing.assert_array_equal(
                n["float32_tuned_normalized_sign"], b["float32_tuned_sign_p"]
            )
            np.testing.assert_array_equal(
                n["float32_tuned_normalized_valid"], b["float32_tuned_valid_query"]
            )
            records = (
                (b, "float32_tuned_sign_p", "float32_tuned_valid_query", None),
                (
                    n,
                    "float32_tuned_raw_edge_sign",
                    "float32_tuned_raw_edge_valid",
                    "float32_tuned_raw_edge_zero_normal",
                ),
                (
                    h,
                    "double_normals_rounded_sign",
                    "double_normals_rounded_valid",
                    "double_normals_rounded_zero",
                ),
                (
                    h,
                    "double_normals_original_sign",
                    "double_normals_original_valid",
                    "double_normals_original_zero",
                ),
                (b, "float64_tuned_sign_p", "float64_tuned_valid_query", None),
            )
            for variant, (archive, sign_key, valid_key, zero_key) in zip(
                variants, records, strict=True
            ):
                sign, valid = archive[sign_key][selected], archive[valid_key][selected]
                if zero_key:
                    zero = archive[zero_key][selected]
                else:
                    precision = "float32" if variant == "normalized_float" else "float64"
                    zero = (b[f"{precision}_tuned_pseudonormal"][selected] == 0).all(axis=1)
                counts = {
                    "queries": len(sign),
                    "inside": int((sign == 1).sum()),
                    "outside": int((sign == -1).sum()),
                    "disagreements": int((sign != truth).sum()),
                    "invalid": int((~valid).sum()),
                    "zero_normals": int(zero.sum()),
                }
                totals[variant].update(counts)
                per_model.append({"file_id": file_id, "variant": variant, **counts})
                for j, index in enumerate(indices):
                    cases.append(
                        {
                            "file_id": file_id,
                            "sample_index": int(index),
                            "variant": variant,
                            "sign": int(sign[j]),
                            "winding_sign": int(truth[j]),
                            "valid": bool(valid[j]),
                            "zero_normal": bool(zero[j]),
                        }
                    )
            np.testing.assert_array_equal(
                h["double_normals_original_sign"][selected], b["float64_tuned_sign_p"][selected]
            )
    write_csv(hybrid / "originally_invalid_cases.csv", cases)
    write_json(
        hybrid / "originally_invalid_comparison.json",
        {
            "selection": "Exactly the queries with nonfinite normals in the saved tuned float baseline",
            "all_selected_queries_pass_fixed_distance_cutoff": True,
            "double_original_and_full_double_signs_match_on_cohort": True,
            "inputs": [str(p) for p in (baseline, ablation, hybrid, reference)],
            "totals": {v: dict(totals[v]) for v in variants},
            "per_model": per_model,
        },
    )
    print(json.dumps(totals, indent=2))


if __name__ == "__main__":
    run()
