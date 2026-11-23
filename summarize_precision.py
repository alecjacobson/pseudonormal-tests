"""Audit the libigl scalar experiment and export its findings and diagnostic cases."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from benchmark import write_csv
from dataset_cache import write_json

OUTPUT_KEYS = ("signed_distance", "closest_face_index", "closest_point", "pseudonormal", "sign_p")


def run(args):
    """Compare every retained output and keep invalid queries visible in the report."""
    args.output.mkdir(parents=True, exist_ok=True)
    variants = {
        "original_double": args.original,
        "patched_double": args.double,
        "patched_single": args.single,
        "quantized_double": args.quantized,
    }
    with (args.reference / "per_model.csv").open(newline="") as stream:
        reference_rows = list(csv.DictReader(stream))
    intended_ids = {row["file_id"] for row in reference_rows}
    configs = {}
    for name, path in variants.items():
        configs[name] = json.loads((path / "config.json").read_text())
        with (path / "per_model.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != len(reference_rows) or {row["file_id"] for row in rows} != intended_ids:
            raise ValueError(f"{name} did not evaluate the same model set")
        if any(row["status"] == "error" for row in rows):
            raise ValueError(f"{name} has failed models")
    exact = {
        comparison: {key: {"different_values": 0, "different_bytes": 0} for key in OUTPUT_KEYS}
        for comparison in ("original_vs_wheel", "patched_double_vs_original")
    }
    model_rows = []
    cases = []
    totals = {
        name: {
            "points": 0,
            "away_points": 0,
            "raw_disagreements": 0,
            "counted_disagreements": 0,
            "valid_counted_disagreements": 0,
            "invalid_queries": 0,
            "invalid_queries_away": 0,
            "affected_models": 0,
            "new_vs_original": 0,
            "resolved_vs_original": 0,
        }
        for name in variants
    }
    distance_errors = {name: [] for name in variants}
    for source in reference_rows:
        file_id = source["file_id"]
        bbd = float(source["bbd"])
        with np.load(args.reference / source["points_file"], allow_pickle=False) as archive:
            reference = dict(archive)
        outputs = {}
        for name, path in variants.items():
            with np.load(path / "points" / f"{file_id}.npz", allow_pickle=False) as archive:
                outputs[name] = dict(archive)
        for label, left, right in [
            ("original_vs_wheel", outputs["original_double"], reference),
            ("patched_double_vs_original", outputs["patched_double"], outputs["original_double"]),
        ]:
            for key in OUTPUT_KEYS:
                a, b = left[key], right[key]
                if a.shape != b.shape or a.dtype != b.dtype:
                    raise ValueError(f"{label} {file_id} {key}: shape or dtype changed")
                exact[label][key]["different_values"] += int(np.count_nonzero(a != b))
                exact[label][key]["different_bytes"] += int(
                    np.count_nonzero(
                        np.ascontiguousarray(a).view(np.uint8)
                        != np.ascontiguousarray(b).view(np.uint8)
                    )
                )
        away = reference["away_from_surface"]
        np.testing.assert_array_equal(away, reference["d"] > 0.001 * bbd)
        original_disagree = outputs["original_double"]["sign_p"] != reference["sign_w"]
        diagnostic = np.zeros(len(away), dtype=bool)
        model = {
            "file_id": file_id,
            "num_points": len(away),
            "num_away": int(away.sum()),
            "bbd": bbd,
        }
        for name, arrays in outputs.items():
            valid = (
                np.isfinite(arrays["signed_distance"])
                & np.isfinite(arrays["closest_point"]).all(axis=1)
                & np.isfinite(arrays["pseudonormal"]).all(axis=1)
            )
            arrays["valid_query"] = valid
            disagree = arrays["sign_p"] != reference["sign_w"]
            counted = disagree & away
            diagnostic |= disagree | ~valid
            values = {
                "points": len(away),
                "away_points": int(away.sum()),
                "raw_disagreements": int(disagree.sum()),
                "counted_disagreements": int(counted.sum()),
                "valid_counted_disagreements": int((counted & valid).sum()),
                "invalid_queries": int((~valid).sum()),
                "invalid_queries_away": int((~valid & away).sum()),
                "affected_models": int(counted.any()),
                "new_vs_original": int((counted & ~original_disagree).sum()),
                "resolved_vs_original": int((~disagree & original_disagree & away).sum()),
            }
            for key, value in values.items():
                totals[name][key] += value
                model[f"{name}_{key}"] = value
            model[f"{name}_percent_all"] = 100 * int(counted.sum()) / len(away)
            distance_errors[name].append(
                np.abs(np.abs(arrays["signed_distance"].astype(np.float64)) - reference["d"]) / bbd
            )
        model_rows.append(model)
        for sample in np.flatnonzero(diagnostic):
            case = {
                "file_id": file_id,
                "sample_index": int(sample),
                "x": reference["points"][sample, 0],
                "y": reference["points"][sample, 1],
                "z": reference["points"][sample, 2],
                "original_d": reference["d"][sample],
                "d_over_bbd": reference["d"][sample] / bbd,
                "away": bool(away[sample]),
                "raw_winding": reference["winding"][sample],
                "reference_sign_w": int(reference["sign_w"][sample]),
            }
            for name, arrays in outputs.items():
                case[f"{name}_sign_p"] = int(arrays["sign_p"][sample])
                case[f"{name}_face"] = int(arrays["closest_face_index"][sample])
                case[f"{name}_valid"] = bool(arrays["valid_query"][sample])
            cases.append(case)
    for name, totals_row in totals.items():
        errors = np.concatenate(distance_errors[name])
        totals_row["percent_all"] = 100 * totals_row["counted_disagreements"] / totals_row["points"]
        totals_row["percent_away"] = (
            100 * totals_row["counted_disagreements"] / totals_row["away_points"]
        )
        totals_row["distance_error_max_bbd"] = float(np.max(errors))
        totals_row["distance_error_p99_bbd"] = float(np.quantile(errors, 0.99))
        totals_row["distance_error_median_bbd"] = float(np.median(errors))
        with (variants[name] / "per_model.csv").open(newline="") as stream:
            timings = list(csv.DictReader(stream))
        totals_row["query_seconds"] = sum(float(row["query_seconds"]) for row in timings)
        totals_row["mesh_load_and_hash_seconds"] = sum(
            float(row["mesh_load_and_hash_seconds"]) for row in timings
        )
    double = totals["patched_double"]
    single = totals["patched_single"]
    summary = {
        "variants": totals,
        "double_exactness": exact,
        "single_minus_double_percentage_points": single["percent_all"] - double["percent_all"],
        "single_to_double_ratio": single["counted_disagreements"] / double["counted_disagreements"],
        "reference": str(args.reference),
        "configurations": configs,
        "metric": "Returned sign disagrees with frozen original winding sign AND original d > 0.001 * bbd; denominator all samples",
        "invalid_query_policy": "Preserve outputs and returned signs, explicitly count nonfinite-query outputs, and also report disagreements with finite outputs only",
    }
    write_json(args.output / "summary.json", summary)
    write_csv(args.output / "per_model.csv", model_rows)
    write_csv(args.output / "changed_cases.csv", cases)
    lines = [
        "# Libigl scalar precision benchmark",
        "",
        "The same 100 models and 1,000,000 cached query points were evaluated at their original scale and position. "
        "The winding reference and `original_d > 0.001 * original_bbd` mask were frozen. "
        "There are 966,656 points beyond that cutoff. The dataset filter allows multiple components.",
        "",
        "| Variant | Counted returned-sign disagreements | % of all points | Affected models | Invalid queries |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, row in totals.items():
        lines.append(
            f"| {name} | {row['counted_disagreements']:,} | {row['percent_all']:.7g}% | {row['affected_models']} | {row['invalid_queries']} |"
        )
    lines += [
        "",
        f"Single precision changes the returned-sign disagreement rate by **{summary['single_minus_double_percentage_points']:.7g} percentage points** "
        f"(**{summary['single_to_double_ratio']:.3f}×** the double rate).",
        "",
    ]
    for label, values in exact.items():
        count = sum(value["different_bytes"] for value in values.values())
        lines.append(
            f"- `{label}`: **{count} differing bytes** across every signed distance, closest-face index, closest point, pseudonormal, and sign."
        )
    if single["invalid_queries"]:
        invalid = [row for row in cases if not row["patched_single_valid"]]
        lines += [
            "",
            f"Single precision returned nonfinite outputs at **{single['invalid_queries']} query/queries**, "
            f"including {single['invalid_queries_away']} beyond the cutoff. These are retained and flagged, not silently dropped. "
            f"There are **{single['valid_counted_disagreements']} disagreements with finite query outputs**. "
            "The table counts signs returned by the library; invalid normals do not provide trustworthy pseudonormal classifications.",
            "",
        ]
        for case in invalid:
            lines.append(
                f"- Invalid case: `file_id={case['file_id']}`, `sample_index={case['sample_index']}`, `d/bbd={case['d_over_bbd']:.9g}`."
            )
    lines += [
        "",
        "## Distance differences and timings",
        "",
        "Errors below compare unsigned distances with the original double result and divide by original `bbd`.",
        "",
        "| Variant | Median error / bbd | p99 error / bbd | Max error / bbd | Batch query seconds |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, row in totals.items():
        lines.append(
            f"| {name} | {row['distance_error_median_bbd']:.6g} | {row['distance_error_p99_bbd']:.6g} | {row['distance_error_max_bbd']:.6g} | {row['query_seconds']:.3f} |"
        )
    lines += [
        "",
        "Timings sum one batch call per mesh, including AABB and normal preprocessing. "
        "They do not isolate steady-state query throughput. Loading, conversion, writing, and cache access "
        "are recorded separately in each run directory. The dataset was reused locally, with no download. "
        "All source comparisons use the same compiler flags, including disabled floating-point contraction; "
        "the small extension builds the actual upstream `signed_distance.cpp` binding.",
        "",
        "## Reproducibility and interpretation",
        "",
        "Seed `20261123`; dataset revision `2d5d3b2f3cd3711028ad75b12788c13b25559ec6`; "
        "Python 3.12.3; Eigen 3.4.0; nanobind 2.7.0; GCC 13.3.0; four libigl threads. "
        "Model IDs, mesh hashes, original points, source commits, module hashes, and configurations are retained. "
        "See [summary.json](summary.json), [per_model.csv](per_model.csv), and [changed_cases.csv](changed_cases.csv). "
        "Per-model tables join to dataset geometry metadata on `file_id`; diagnostic points additionally use `sample_index`.",
        "",
    ]
    for name, config in configs.items():
        lines.append(
            f"- `{name}`: libigl `{config['libigl']['commit']}`, bindings `{config['bindings']['commit']}`."
        )
    lines += [
        "",
        "The quantized-input control rounds vertices and points to float32 once, then evaluates their exact "
        "float64 representations in double. It separates input quantization from arithmetic precision. "
        "The original winding results remain a comparison reference, not independently proven ground truth. "
        "The scalar patch does not retune the feature classifier or its tolerances; that investigation is tracked in issue 2.",
        "",
        "Prepared by OpenAI Codex.",
    ]
    (args.output / "FINDINGS.md").write_text("\n".join(lines) + "\n")
    print(
        json.dumps(
            {key: value for key, value in summary.items() if key != "configurations"}, indent=2
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=Path("results/initial-100"))
    parser.add_argument("--original", type=Path, default=Path("results/libigl-original-double-100"))
    parser.add_argument("--double", type=Path, default=Path("results/libigl-scalar-double-100"))
    parser.add_argument("--single", type=Path, default=Path("results/libigl-scalar-single-100"))
    parser.add_argument(
        "--quantized", type=Path, default=Path("results/libigl-quantized-double-100")
    )
    parser.add_argument("--output", type=Path, default=Path("results/libigl-scalar-100"))
    run(parser.parse_args())
