"""Audit retained Warp results and generate the comparison report."""

import argparse
import csv
import json
import os
from pathlib import Path

import igl
import numpy as np

from benchmark import load_mesh, write_csv
from dataset_cache import DEFAULT_CACHE, prepare_dataset, write_json
from warp_benchmark import METHODS, comparison_statistics, file_sha256
from warp_queries import WARP_METHODS

LABELS = {
    "igl_p": "libigl pseudonormal",
    "warp_p": "Warp pseudonormal",
    "warp_w": "Warp winding",
    "warp_parity": "Warp parity",
}


def read_csv(path):
    """Load a result table with named columns."""
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def summarize(output, baseline, cache_dir):
    """Check saved samples, investigate distance outliers, and export findings."""
    config = json.loads((output / "config.json").read_text())
    summary = json.loads((output / "summary.json").read_text())
    cache_access = json.loads((output / "cache_access.json").read_text())
    baseline = baseline or Path(config["baseline"])
    np.testing.assert_equal(file_sha256(baseline / "config.json"), config["baseline_config_sha256"])
    np.testing.assert_equal(
        file_sha256(baseline / "per_model.csv"), config["baseline_per_model_sha256"]
    )
    rows = read_csv(output / "per_model.csv")
    original_rows = {row["file_id"]: row for row in read_csv(baseline / "per_model.csv")}
    np.testing.assert_equal(len(rows), config["models"])
    np.testing.assert_equal(len({row["file_id"] for row in rows}), len(rows))
    np.testing.assert_equal({row["file_id"] for row in rows}, set(original_rows))
    if any(row["status"] != "ok" for row in rows):
        raise ValueError("Resolve failed model queries before reporting results.")
    _, mesh_dir, _ = prepare_dataset(cache_dir, config["dataset_revision"])
    totals = {method: 0 for method in METHODS}
    raw_totals = {method: 0 for method in METHODS}
    false_inside = {method: 0 for method in METHODS}
    affected = {method: 0 for method in METHODS}
    point_total = away_total = 0
    errors = {method: [] for method in WARP_METHODS}
    expected_cases, expected_outliers, diagnostics = set(), set(), []
    for row in rows:
        file_id = row["file_id"]
        original = original_rows[file_id]
        reference_path = baseline / original["points_file"]
        np.testing.assert_equal(file_sha256(reference_path), row["reference_points_file_sha256"])
        np.testing.assert_equal(row["mesh_sha256"], original["mesh_sha256"])
        np.testing.assert_equal(float(row["bbd"]), float(original["bbd"]))
        with np.load(reference_path, allow_pickle=False) as reference:
            with np.load(output / row["points_file"], allow_pickle=False) as result:
                np.testing.assert_equal(len(reference["points"]), config["points_per_model"])
                np.testing.assert_array_equal(
                    result["warp_points"], reference["points"].astype(np.float32)
                )
                stats, masks, model_errors = comparison_statistics(
                    reference, result, float(row["bbd"]), config["relative_cutoff"]
                )
                for key, value in stats.items():
                    if value is None:
                        np.testing.assert_equal(row[key], "")
                    else:
                        np.testing.assert_allclose(float(row[key]), value, rtol=1e-12, atol=0)
                point_total += stats["num_points"]
                away_total += stats["num_away"]
                for method in METHODS:
                    totals[method] += stats[f"{method}_disagreements"]
                    raw_totals[method] += stats[f"{method}_raw_disagreements"]
                    false_inside[method] += stats[f"{method}_false_inside"]
                    affected[method] += int(stats[f"{method}_disagreements"] > 0)
                for method in WARP_METHODS:
                    errors[method].append(model_errors[method])
                    closest = result[f"warp_closest_point_{method}"]
                    distance = np.linalg.norm(result["warp_points"] - closest, axis=1)
                    np.testing.assert_allclose(
                        result[f"warp_d_{method}"], distance, rtol=3e-7, atol=0
                    )
                    face = result[f"warp_face_{method}"]
                    if np.any(face < 0) or np.any(face >= int(row["num_faces"])):
                        raise ValueError(f"Invalid closest triangle index for {file_id}.")
                union = np.logical_or.reduce(list(masks.values()))
                expected_cases.update((file_id, int(index)) for index in np.flatnonzero(union))
                outliers = np.maximum.reduce(list(model_errors.values())) > 1e-5
                expected_outliers.update(
                    (file_id, int(index)) for index in np.flatnonzero(outliers)
                )
                if outliers.any():
                    # Recompute the worst case per model with exactly Warp's input rounding,
                    # while retaining libigl's double-precision distance arithmetic.
                    mesh_path = mesh_dir / f"{file_id}.npz"
                    np.testing.assert_equal(file_sha256(mesh_path), row["mesh_sha256"])
                    vertices, faces = load_mesh(mesh_path)
                    index = int(np.argmax(model_errors["p"]))
                    point32 = reference["points"][index : index + 1].astype(np.float32)
                    vertices32 = vertices.astype(np.float32)
                    distance2, nearest, _ = igl.point_mesh_squared_distance(
                        point32.astype(np.float64), vertices32.astype(np.float64), faces
                    )
                    triangle = vertices32[faces[nearest[0]]]
                    edges = np.roll(triangle, -1, axis=0) - triangle
                    quality = np.linalg.norm(np.cross(edges[0], -edges[2])) / np.sum(edges**2)
                    diagnostics.append(
                        {
                            "file_id": file_id,
                            "outlier_count": int(outliers.sum()),
                            "point_index": index,
                            "bbd": float(row["bbd"]),
                            "igl_d_original": float(reference["d"][index]),
                            "igl_d_on_float32_inputs": float(np.sqrt(distance2[0])),
                            "warp_d_p": float(result["warp_d_p"][index]),
                            "error_over_bbd": float(model_errors["p"][index]),
                            "igl_face_original": int(reference["closest_face_index"][index]),
                            "igl_face_on_float32_inputs": int(nearest[0]),
                            "warp_face_p": int(result["warp_face_p"][index]),
                            "igl_face_float32_quality": float(quality),
                        }
                    )
    for name, expected in (
        ("disagreements.csv", expected_cases),
        ("distance_outliers.csv", expected_outliers),
    ):
        exported = read_csv(output / name)
        np.testing.assert_equal(len(exported), len(expected))
        np.testing.assert_equal(
            {(row["file_id"], int(row["point_index"])) for row in exported}, expected
        )
    np.testing.assert_equal(summary["num_points"], point_total)
    np.testing.assert_equal(summary["num_away"], away_total)
    np.testing.assert_equal(summary["models_completed"], len(rows))
    np.testing.assert_equal(summary["models_failed"], 0)
    for method in METHODS:
        values = summary["methods"][method]
        np.testing.assert_equal(values["disagreements"], totals[method])
        np.testing.assert_equal(values["raw_disagreements"], raw_totals[method])
        np.testing.assert_equal(values["false_inside"], false_inside[method])
        np.testing.assert_equal(values["false_outside"], totals[method] - false_inside[method])
        np.testing.assert_equal(values["models_with_disagreements"], affected[method])
        np.testing.assert_allclose(values["percent_all"], 100 * totals[method] / point_total)
        np.testing.assert_allclose(values["percent_away"], 100 * totals[method] / away_total)
    for method in WARP_METHODS:
        combined = np.concatenate(errors[method])
        values = summary["distance_errors"][method]
        for key, value in {
            "max_over_bbd": combined.max(),
            "mean_over_bbd": combined.mean(),
            "rms_over_bbd": np.sqrt(np.mean(combined**2)),
            "p99_over_bbd": np.quantile(combined, 0.99),
            "count_over_1e_minus6_bbd": (combined > 1e-6).sum(),
            "count_over_1e_minus5_bbd": (combined > 1e-5).sum(),
        }.items():
            np.testing.assert_allclose(values[key], value, rtol=1e-12, atol=0)
    diagnostics.sort(key=lambda row: row["error_over_bbd"], reverse=True)
    if diagnostics:
        write_csv(output / "distance_diagnostics.csv", diagnostics)
    validation = {
        "result": "passed",
        "models_checked": len(rows),
        "points_checked": point_total,
        "disagreement_rows_checked": len(expected_cases),
        "distance_outlier_rows_checked": len(expected_outliers),
        "checks": [
            "baseline config, per-model CSV, and retained point hashes",
            "same unique model IDs, mesh hashes, and bounding-box diagonals",
            "identical samples after float32 conversion",
            "all query validity flags, finite distances, binary signs and valid face indices",
            "distance agrees with returned closest point",
            "shared original distance cutoff, per-model counts and rates",
            "aggregate counts, rates, and distance quantiles",
            "exported disagreement and distance outlier IDs",
        ],
        "distance_diagnostic": "worst distance outlier per affected model recomputed with libigl on float32-rounded inputs",
    }
    write_json(output / "validation.json", validation)
    cache_note = (
        "Every dataset asset was reused from the local cache."
        if all(item["cache_hit"] for item in cache_access["files"])
        and cache_access["extraction"]["cache_hit"]
        else "Dataset acquisition details are recorded in cache_access.json."
    )
    exact_methods = [
        LABELS[method] for method in METHODS if summary["methods"][method]["raw_disagreements"] == 0
    ]
    parity_note = (
        "This is a one-ray parity test through the voting API."
        if config["parity_rays"] == 1
        else "The rays are combined by majority voting."
    )
    report = [
        "# Warp comparison findings\n",
        f"Run completed {summary['completed_at_utc']}. Warp `{config['warp_source']['version']}`, "
        f"[main commit `{config['warp_source']['revision'][:12]}`]"
        f"(https://github.com/NVIDIA/warp/commit/{config['warp_source']['revision']}). "
        f"Device: {config['device_name']} (`{config['device']}`).\n",
        f"Reused the libigl baseline's {len(rows)} models and {point_total:,} saved samples, "
        f"with no resampling. {cache_note} All comparisons use saved `igl_sign_w` as the "
        f"reference and the same mask `igl_d > {config['relative_cutoff']} * bbd` "
        f"({away_total:,} points). Signs are inside +1, outside -1. The reference is a numerical "
        "algorithm, not independently established ground truth.\n",
        "| Method | Disagreements | % of all samples | % beyond cutoff | Models affected | Before cutoff |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method in METHODS:
        values = summary["methods"][method]
        report.append(
            f"| {LABELS[method]} | {values['disagreements']} | {values['percent_all']:.7g} | "
            f"{values['percent_away']:.9g} | {values['models_with_disagreements']} | "
            f"{values['raw_disagreements']} |"
        )
    values = summary["methods"]["warp_p"]
    report.extend(
        [
            f"\nWarp pseudonormal's counted disagreements comprise {values['false_inside']} "
            f"inside predictions and {values['false_outside']} outside predictions relative to "
            f"libigl winding. Methods agreeing with the reference even before the cutoff: "
            f"{', '.join(exact_methods) or 'none'}.\n",
            "## Method and parameters\n",
            "Warp calls `mesh_query_point_sign_normal`, `mesh_query_point_sign_winding_number`, "
            "and `mesh_query_point_sign_parity` separately. Each distance is "
            "`length(p - mesh_eval_position(mesh, face, u, v))` from that call. "
            "Mesh winding support is explicitly enabled, preventing the winding API's ray-vote "
            "fallback. Vertices and samples are converted to float32 for Warp without "
            "normalization; libigl uses float64. The original connectivity and orientation remain.\n",
            f"Pseudonormal epsilon: `{config['pseudonormal_epsilon']}`; winding accuracy: "
            f"`{config['winding_accuracy']}`, threshold: `{config['winding_threshold']}`. "
            f"Parity uses **{config['parity_rays']} ray(s) per query**, "
            f"perturbation scale `{config['parity_perturbation']}` and Warp's fixed internal RNG "
            f"seed 42. {parity_note} "
            "Use `--parity-rays 3` or another positive odd count for multi-ray majority voting. "
            "Warp winding uses its native threshold; libigl rounds its raw winding values.\n",
            "## Distance agreement\n",
            "Absolute distance differences are normalized by each model's bounding-box diagonal.\n",
            "| Warp distance path | Mean / bbd | 99th percentile / bbd | Maximum / bbd | Count > 1e-5 bbd |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for method in WARP_METHODS:
        values = summary["distance_errors"][method]
        report.append(
            f"| {method} | {values['mean_over_bbd']:.9g} | {values['p99_over_bbd']:.9g} | "
            f"{values['max_over_bbd']:.9g} | {values['count_over_1e_minus5_bbd']} |"
        )
    report.extend(
        [
            f"\n[distance_outliers.csv](distance_outliers.csv) records all {len(expected_outliers):,} "
            f"samples exceeding `1e-5 * bbd` in any path, across {len(diagnostics)} models. "
            "[distance_diagnostics.csv](distance_diagnostics.csv) recomputes the worst case per "
            "affected model using libigl's double-precision distance query with exactly "
            "Warp's float32-rounded geometry and samples.\n",
            "| file_id | Point index | Original libigl d | libigl d on float32 inputs | Warp d |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in diagnostics[:2]:
        report.append(
            f"| {row['file_id']} | {row['point_index']} | {row['igl_d_original']:.10g} | "
            f"{row['igl_d_on_float32_inputs']:.10g} | {row['warp_d_p']:.10g} |"
        )
    if diagnostics:
        largest = diagnostics[:2]
        persist = all(
            abs(row["igl_d_on_float32_inputs"] - row["warp_d_p"])
            > 0.9 * abs(row["igl_d_original"] - row["warp_d_p"])
            for row in largest
        )
        different_faces = sum(
            row["igl_face_on_float32_inputs"] != row["warp_face_p"] for row in largest
        )
        note = (
            "The largest discrepancies shown above persist with identical rounded inputs, "
            "so input rounding alone does not explain them. "
            if persist
            else "The table separates input rounding from query differences. "
        )
        report.append(
            f"\n{note}The returned nearest triangles differ in {different_faces} of "
            f"these {len(largest)} cases. A root cause in the distance query has not been "
            "established.\n"
        )
    report.extend(
        [
            "## Timings\n",
            f"Sums over {len(rows)} batches of {config['points_per_model']:,} points, "
            "one batch per method per model. "
            "Each GPU query timing includes launch and synchronization plus distance, sign, "
            "and closest-point outputs. No per-model warmup or repetitions. Mesh construction, "
            "transfers, and module loading are measured separately.\n",
            "| Step | Seconds |",
            "| --- | ---: |",
        ]
    )
    for key, value in summary["summed_timings_seconds"].items():
        report.append(f"| {key.removesuffix('_seconds').replace('_', ' ')} | {value:.6f} |")
    for key in ("checkpoint_seconds", "csv_export_seconds"):
        if key in summary:
            report.append(
                f"| {key.removesuffix('_seconds').replace('_', ' ')} | {summary[key]:.6f} |"
            )
    report.extend(
        [
            f"| Module loading (before queries) | {config['module_load_seconds']:.6f} |",
            f"| Whole run | {summary['wall_seconds']:.6f} |",
            "\nWarp kernels had been compiled during validation before this run; module loading "
            "reused the shared kernel cache. Native source build time is outside the benchmark. "
            "The original libigl CPU timings include acceleration-structure construction and "
            "normals inside each call. These timing scopes and devices differ; ratios are not "
            "an isolated algorithmic speed comparison.\n",
            "## Validation and retained data\n",
            "The full point audit passed; see [validation.json](validation.json). "
            "Analytic integration tests exercise CPU and CUDA cube distances, cavity signs, "
            "and the shared cutoff. One analytic case is marked as an expected failure: "
            "at the exact center of a symmetric empty cube cavity, the pinned Warp "
            "pseudonormal query returns inside on both CPU and CUDA. This observed behavior "
            "is preserved in `test_warp_benchmark.py`; it is separate from the random benchmark.\n",
            "Join [per_model.csv](per_model.csv) to the dataset's `geometry_data.csv` on `file_id` "
            "or to `input_summary.csv` on `ID`. Original metadata is retained in "
            f"[selected_metadata.csv]({Path(os.path.relpath(baseline / 'selected_metadata.csv', output)).as_posix()}). "
            "[disagreements.csv](disagreements.csv) contains the union of counted cases with "
            "coordinates, raw libigl winding, signs, distances and Warp closest-face indices. "
            "Full point arrays remain cached locally and are excluded from Git. "
            "[config.json](config.json) records source commit, native library hashes, "
            "baseline hashes, precision, parameters and timing scopes. "
            "[cache_access.json](cache_access.json) confirms reuse of every dataset asset.\n",
            "Prepared with OpenAI Codex.\n",
        ]
    )
    (output / "FINDINGS.md").write_text("\n".join(report))
    print(
        f"Audited {len(rows)} models / {point_total:,} points; {len(diagnostics)} distance diagnostics."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    args = parser.parse_args()
    summarize(args.output, args.baseline, args.cache_dir)
