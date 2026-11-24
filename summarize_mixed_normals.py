"""Audit mixed-normal point caches and summarize the remaining failure cases."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from benchmark_mixed_normals import VARIANTS
from dataset_cache import write_json
from validate_tolerances import read_rows


def run(path):
    """Independently recount every saved sign and check the diagnostic export."""
    config = json.loads((path / "config.json").read_text())
    summary = json.loads((path / "summary.json").read_text())
    rows = read_rows(path / "per_model.csv")
    indexed = {(r["file_id"], r["variant"]): r for r in rows}
    np.testing.assert_equal(len(indexed), len(rows))
    np.testing.assert_equal(
        set(indexed), {(f, v) for f in config["selected_ids"] for v in VARIANTS}
    )
    cases = read_rows(path / "cases.csv")
    case_index = {(r["file_id"], r["variant"], int(r["sample_index"])): r for r in cases}
    np.testing.assert_equal(len(case_index), len(cases))
    totals = {v: Counter() for v in VARIANTS}
    expected_cases = set()
    for file_id in config["selected_ids"]:
        with np.load(Path(config["reference"]) / "points" / f"{file_id}.npz") as archive:
            truth, away = archive["sign_w"], archive["away_from_surface"]
        with np.load(Path(config["baseline"]) / "points" / f"{file_id}.npz") as archive:
            baseline = archive["float32_tuned_sign_p"]
            baseline_normals = archive["float32_tuned_pseudonormal"]
            double_sign = archive["float64_tuned_sign_p"]
        baseline_valid = np.isfinite(baseline_normals).all(axis=1)
        before = (baseline != truth) & away
        with np.load(path / "points" / f"{file_id}.npz") as archive:
            for variant in VARIANTS:
                sign = baseline if variant == "float" else archive[f"{variant}_sign"]
                normals = baseline_normals if variant == "float" else archive[f"{variant}_normal"]
                valid = np.isfinite(normals).all(axis=1)
                zero = valid & (normals == 0).all(axis=1)
                wrong = (sign != truth) & away
                if variant != "float":
                    np.testing.assert_array_equal(valid, archive[f"{variant}_valid"])
                    np.testing.assert_array_equal(zero, archive[f"{variant}_zero"])
                checks = {
                    "points": len(sign),
                    "away_points": int(away.sum()),
                    "disagreements": int(wrong.sum()),
                    "finite_disagreements": int((wrong & valid).sum()),
                    "invalid_queries": int((~valid).sum()),
                    "invalid_queries_away": int((~valid & away).sum()),
                    "zero_normals": int(zero.sum()),
                    "resolved": int((before & ~wrong).sum()),
                    "introduced": int((wrong & ~before).sum()),
                    "invalid_resolved": int((~baseline_valid & valid).sum()),
                    "invalid_introduced": int((baseline_valid & ~valid).sum()),
                    "sign_changes": int((sign != baseline).sum()),
                    "affected_models": int(wrong.any()),
                    "models_with_invalid": int((~valid).any()),
                }
                for key, value in checks.items():
                    np.testing.assert_equal(int(indexed[file_id, variant][key]), value)
                totals[variant].update(checks)
                totals[variant]["also_wrong_in_full_double"] += int(
                    (wrong & (double_sign != truth)).sum()
                )
                diagnostic = wrong | ~valid | zero | before | ~baseline_valid
                for i in np.flatnonzero(diagnostic):
                    key = (file_id, variant, int(i))
                    expected_cases.add(key)
                    case = case_index[key]
                    np.testing.assert_equal(case["counted_disagreement"], str(bool(wrong[i])))
                    np.testing.assert_equal(case["valid_normal"], str(bool(valid[i])))
                    np.testing.assert_equal(int(case["sign"]), sign[i])
                    np.testing.assert_equal(int(case["winding_sign"]), truth[i])
                    np.testing.assert_array_equal(
                        [float(case[f"normal_{axis}"]) for axis in "xyz"], normals[i]
                    )
    np.testing.assert_equal(expected_cases, set(case_index))
    diagnostics = {}
    for variant in VARIANTS:
        for key, value in totals[variant].items():
            if key != "also_wrong_in_full_double":
                np.testing.assert_equal(summary["groups"]["all"][variant][key], value)
        failures = [
            r for r in cases if r["variant"] == variant and r["counted_disagreement"] == "True"
        ]
        diagnostics[variant] = {
            "totals": dict(totals[variant]),
            "failure_features": dict(Counter(r["normal_kind"] for r in failures)),
            "small_area_failures": sum(r["small_area_branch"] == "True" for r in failures),
            "same_closest_face_as_double": sum(
                r["closest_face_float"] == r["closest_face_double"] for r in failures
            ),
            "dot_sign_changes_in_double": sum(
                r["dot_sign_changes_in_double"] == "True" for r in failures
            ),
            "top_models": Counter(r["file_id"] for r in failures).most_common(20),
        }
    result = {
        "all_point_counts_and_diagnostic_rows_verified": True,
        "diagnostic_rows": len(cases),
        "variants": diagnostics,
    }
    write_json(path / "audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    run(parser.parse_args().path)
