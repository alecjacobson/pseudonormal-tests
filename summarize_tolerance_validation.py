"""Audit the frozen-tolerance validation and export findings and remaining cases."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

from benchmark import write_csv
from dataset_cache import DEFAULT_CACHE, prepare_dataset, write_json

matplotlib.use("Agg")
import matplotlib.pyplot as plt

VARIANTS = ["float32_default", "float32_tuned", "float64_default", "float64_tuned"]
LABELS = ["Float default", "Float tuned", "Double default", "Double tuned"]


def read_rows(path):
    """Read a complete exported table."""
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def check_bytes(actual, expected, context):
    """Compare dtype, shape, and every byte, including nonfinite values."""
    np.testing.assert_equal(actual.dtype, expected.dtype, err_msg=context)
    np.testing.assert_equal(actual.shape, expected.shape, err_msg=context)
    np.testing.assert_array_equal(
        np.ascontiguousarray(actual).view(np.uint8),
        np.ascontiguousarray(expected).view(np.uint8),
        err_msg=context,
    )


def run(path, cache_dir):
    """Recount every cached point and separate the original and additional models."""
    config = json.loads((path / "config.json").read_text())
    summary = json.loads((path / "summary.json").read_text())
    reference_path, initial_path = Path(config["reference"]), Path(config["initial"])
    sources = read_rows(reference_path / "per_model.csv")
    initial_ids = set(config["initial_ids"])
    rows = read_rows(path / "per_model.csv")
    indexed = {(row["file_id"], row["variant"]): row for row in rows}
    np.testing.assert_equal(len(rows), len(indexed))
    np.testing.assert_equal(
        set(indexed), {(row["file_id"], variant) for row in sources for variant in VARIANTS}
    )
    metadata_path, _, _ = prepare_dataset(cache_dir)
    eligible = {
        row["file_id"]
        for row in read_rows(metadata_path)
        if all(row[key] == "1" for key in ("solid", "vertex_manifold", "edge_manifold"))
    }
    np.testing.assert_equal({row["file_id"] for row in sources}, eligible)
    np.testing.assert_equal(config["full_mask"], True)
    journal = [json.loads(line) for line in (path / "per_model.jsonl").read_text().splitlines()]
    np.testing.assert_equal(len(journal), len(rows))
    for exported, record in zip(rows, journal, strict=True):
        for key, value in record.items():
            np.testing.assert_equal(exported[key], str(value))

    studies = {}
    for precision, name in [
        ("float32", "libigl-float-tolerances-100"),
        ("float64", "libigl-double-tolerances-100"),
    ]:
        study_path = Path("results") / name
        study = json.loads((study_path / "config.json").read_text())
        target = config["settings"][precision]
        matching = [
            setting
            for setting in study["settings"]
            if setting["minimum_double_area"] == target["area"]
            and setting["epsilon"] == target["epsilon"]
        ]
        np.testing.assert_equal(len(matching), 1)
        studies[precision] = study_path, matching[0]["setting_id"]

    totals = {
        split: {variant: {} for variant in VARIANTS}
        for split in ["all", "original_100", "additional_models"]
    }
    cases, comparisons = [], []
    original_points = 0
    for index, source in enumerate(sources):
        file_id = source["file_id"]
        split = "original_100" if file_id in initial_ids else "additional_models"
        with np.load(reference_path / source["points_file"], allow_pickle=False) as archive:
            reference = dict(archive)
        with np.load(path / "points" / f"{file_id}.npz", allow_pickle=False) as archive:
            saved = dict(archive)
        away = reference["away_from_surface"]
        np.testing.assert_array_equal(away, reference["d"] > 0.001 * float(source["bbd"]))
        diagnostic = np.zeros(len(away), dtype=bool)
        comparison = {
            "file_id": file_id,
            "split": split,
            "bbd": source["bbd"],
            "num_vertices": source["num_vertices"],
            "num_faces": source["num_faces"],
            "num_connected_components": source["num_connected_components"],
        }
        for variant in VARIANTS:
            precision, label = variant.split("_")
            row = indexed[file_id, variant]
            sign, valid = saved[f"{variant}_sign_p"], saved[f"{variant}_valid_query"]
            baseline = saved[f"{precision}_default_sign_p"]
            baseline_valid = saved[f"{precision}_default_valid_query"]
            distance = (
                saved["float32_signed_distance"]
                if precision == "float32"
                else reference["signed_distance"]
            )
            closest = (
                saved["float32_closest_point"]
                if precision == "float32"
                else reference["closest_point"]
            )
            if label == "tuned":
                normals = saved[f"{variant}_pseudonormal"]
            else:
                normals = (
                    saved["float32_pseudonormal"]
                    if precision == "float32"
                    else reference["pseudonormal"]
                )
            np.testing.assert_equal(normals.dtype, np.dtype(precision))
            np.testing.assert_array_equal(
                valid,
                np.isfinite(distance)
                & np.isfinite(closest).all(axis=1)
                & np.isfinite(normals).all(axis=1),
            )
            if label == "default":
                np.testing.assert_array_equal(sign, np.where(distance < 0, 1, -1).astype(np.int8))
            disagree = sign != reference["sign_w"]
            counted, default_counted = disagree & away, (baseline != reference["sign_w"]) & away
            invalid = ~valid
            metrics = {
                "models": 1,
                "points": len(away),
                "away_points": int(np.count_nonzero(away)),
                "raw_disagreements": int(np.count_nonzero(disagree)),
                "counted_disagreements": int(np.count_nonzero(counted)),
                "valid_counted_disagreements": int(np.count_nonzero(counted & valid)),
                "invalid_queries": int(np.count_nonzero(invalid)),
                "invalid_queries_away": int(np.count_nonzero(invalid & away)),
                "new_vs_default": int(np.count_nonzero(counted & ~default_counted)),
                "resolved_vs_default": int(np.count_nonzero(~counted & default_counted)),
                "new_invalid_queries": int(np.count_nonzero(invalid & baseline_valid)),
                "resolved_invalid_queries": int(np.count_nonzero(valid & ~baseline_valid)),
                "affected_models": int(counted.any()),
                "models_with_invalid_queries": int(invalid.any()),
            }
            for key, value in metrics.items():
                if key != "models":
                    np.testing.assert_equal(
                        int(row[key]), value, err_msg=f"{file_id} {variant} {key}"
                    )
                for fold in ["all", split]:
                    totals[fold][variant][key] = totals[fold][variant].get(key, 0) + value
            np.testing.assert_equal(row["status"], "nonfinite_output" if invalid.any() else "ok")
            for key in [
                "counted_disagreements",
                "valid_counted_disagreements",
                "invalid_queries",
                "new_vs_default",
                "resolved_vs_default",
            ]:
                comparison[f"{variant}_{key}"] = metrics[key]
            if label == "tuned":
                diagnostic |= counted | invalid
        np.testing.assert_equal(
            int(source["num_counted_disagreements"]),
            comparison["float64_default_counted_disagreements"],
        )
        comparisons.append(comparison)
        if file_id in initial_ids:
            with np.load(
                initial_path / "points" / f"{file_id}.npz", allow_pickle=False
            ) as previous:
                for key in ["points", "d", "sign_w", "away_from_surface"]:
                    check_bytes(reference[key], previous[key], f"{file_id} original {key}")
            previous_single = Path("results/libigl-scalar-single-100/points") / f"{file_id}.npz"
            with np.load(previous_single, allow_pickle=False) as previous:
                for key in [
                    "signed_distance",
                    "closest_face_index",
                    "closest_point",
                    "pseudonormal",
                ]:
                    check_bytes(saved[f"float32_{key}"], previous[key], f"{file_id} single {key}")
            for precision, (study_path, setting_id) in studies.items():
                with np.load(
                    study_path / "points" / f"{file_id}.npz", allow_pickle=False
                ) as previous:
                    for key in ["sign_p", "valid_query"]:
                        np.testing.assert_array_equal(
                            saved[f"{precision}_tuned_{key}"], previous[key][setting_id]
                        )
            original_points += len(away)
        for sample in np.flatnonzero(diagnostic):
            case = {
                "file_id": file_id,
                "sample_index": int(sample),
                "split": split,
                "x": reference["points"][sample, 0],
                "y": reference["points"][sample, 1],
                "z": reference["points"][sample, 2],
                "d_over_bbd": reference["d"][sample] / float(source["bbd"]),
                "float32_d_over_bbd": abs(float(saved["float32_signed_distance"][sample]))
                / float(source["bbd"]),
                "float32_distance_error_bbd": abs(
                    abs(float(saved["float32_signed_distance"][sample])) - reference["d"][sample]
                )
                / float(source["bbd"]),
                "away": bool(away[sample]),
                "raw_winding": reference["winding"][sample],
                "sign_w": int(reference["sign_w"][sample]),
                "float32_closest_face_index": int(saved["float32_closest_face_index"][sample]),
                "float64_closest_face_index": int(reference["closest_face_index"][sample]),
            }
            for variant in VARIANTS:
                case[f"{variant}_sign_p"] = int(saved[f"{variant}_sign_p"][sample])
                case[f"{variant}_valid"] = bool(saved[f"{variant}_valid_query"][sample])
            cases.append(case)
        if (index + 1) % 500 == 0:
            print(f"Recounted {index + 1}/{len(sources)} models", flush=True)
    for split, variants in totals.items():
        for variant, metrics in variants.items():
            for key, value in metrics.items():
                np.testing.assert_equal(
                    value,
                    summary["totals"][split][variant][key],
                    err_msg=f"{split} {variant} {key}",
                )
    write_csv(path / "remaining_cases.csv", cases)
    write_csv(path / "model_comparison.csv", comparisons)
    write_json(
        path / "validation.json",
        {
            "result": "passed",
            "eligible_models": len(eligible),
            "models_audited": len(sources),
            "points_per_variant": totals["all"]["float64_default"]["points"],
            "variants_recounted": VARIANTS,
            "splits_recounted": list(totals),
            "all_per_model_integer_metrics_match": True,
            "all_aggregate_integer_metrics_match": True,
            "validity_recomputed_from_saved_outputs": True,
            "journal_matches_csv": True,
            "original_models_reproduced": len(initial_ids),
            "original_points_reproduced": original_points,
            "original_float_geometry_and_normals_byte_identical": True,
            "original_tuned_signs_and_validity_identical": True,
            "default_reproduction": summary["baseline_reproduction"],
            "diagnostic_cases": len(cases),
        },
    )
    plot_and_report(path, config, summary, comparisons)
    print(json.dumps(summary["totals"], indent=2))


def plot_and_report(path, config, summary, comparisons):
    """Show the frozen-parameter result and its remaining limitations."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), layout="constrained")
    for axis, split, title in zip(
        axes,
        ["all", "additional_models"],
        ["All eligible models", "Additional models"],
        strict=True,
    ):
        rows = summary["totals"][split]
        counts = [rows[name]["counted_disagreements"] for name in VARIANTS]
        bars = axis.bar(LABELS, counts, color=["#90b8d8", "#246d9f", "#e9b987", "#bb631f"])
        axis.set_yscale("symlog", linthresh=1)
        axis.bar_label(bars, labels=[f"{value:,}" for value in counts], padding=4)
        axis.set_ylim(0, max(counts) * 3)
        axis.set_title(f"{title}: {rows['float64_default']['models']:,}")
        axis.tick_params(axis="x", rotation=20)
        axis.set_ylabel("Counted disagreements (logarithmic above 1)")
    fig.suptitle("Frozen tolerance validation against cached winding signs")
    fig.savefig(path / "comparison.png", dpi=170)
    fig.savefig(path / "comparison.pdf")
    plt.close(fig)
    all_rows = summary["totals"]["all"]
    base = all_rows["float64_default"]
    lines = [
        "# Broader validation of frozen pseudonormal tolerances",
        "",
        f"Evaluated **{base['models']:,} models × 10,000 points = {base['points']:,} points**, "
        f"including {summary['totals']['additional_models']['float64_default']['models']:,} models outside "
        "the original tolerance sweeps. Parameters were fixed before this run: "
        "area threshold `1e-4` for both precisions, epsilon `0.003` for float and `1e-9` for double. "
        "The default epsilon is `1e-12`.",
        "",
        "Meshes retain original scale, position, and topology. The reference is the previously saved "
        "double-input libigl fast winding sign. The mask is the saved original double "
        f"`d > 0.001 * bbd` ({base['away_points']:,} away points). The primary percentage uses **all** sampled points. "
        "The additional models were not used in the tolerance sweeps, although their original baseline results "
        "were already available. They are broader regression validation, not a completely uninspected holdout.",
        "",
    ]
    for split, title in [
        ("all", "All eligible models"),
        ("additional_models", "Models outside the original 100"),
        ("original_100", "Original 100 models (reproduction)"),
    ]:
        lines += [
            f"## {title}",
            "",
            "| Method | Disagreements | Percent of all points | Finite-output disagreements | Invalid queries | Affected models |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for variant, label in zip(VARIANTS, LABELS, strict=True):
            row = summary["totals"][split][variant]
            lines.append(
                f"| {label} | {row['counted_disagreements']:,} | {row['percent_all']:.8g}% | "
                f"{row['valid_counted_disagreements']:,} | {row['invalid_queries']:,} | {row['affected_models']:,} |"
            )
        lines.append("")
    lines += [
        "## Improvements and regressions",
        "",
        "| Precision | Resolved disagreements | Introduced disagreements | Net reduction | Invalid queries before → after |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, label in [("float32", "Float"), ("float64", "Double")]:
        default, tuned = all_rows[f"{name}_default"], all_rows[f"{name}_tuned"]
        reduction = 100 * (1 - tuned["counted_disagreements"] / default["counted_disagreements"])
        lines.append(
            f"| {label} | {tuned['resolved_vs_default']:,} | {tuned['new_vs_default']:,} | "
            f"{reduction:.4g}% | {default['invalid_queries']:,} → {tuned['invalid_queries']:,} |"
        )
    lines += [
        "",
        "Counts use the returned sign even when the output normal is nonfinite. "
        "Finite-output disagreement and invalid-query counts are reported separately; "
        "fewer returned-sign disagreements do not by themselves establish valid classification. "
        "A disagreement with fast winding is a benchmark discrepancy, not independently proven ground truth.",
        "",
        "![Comparison](comparison.png)",
        "",
        "[Download PDF](comparison.pdf).",
        "",
        "## Largest remaining model counts",
        "",
        "| Precision | File ID | Split | Default disagreements | Tuned disagreements | Introduced | Invalid tuned queries |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for name, label in [("float32", "Float"), ("float64", "Double")]:
        for model in sorted(
            comparisons,
            key=lambda row: (-row[f"{name}_tuned_counted_disagreements"], int(row["file_id"])),
        )[:10]:
            lines.append(
                f"| {label} | {model['file_id']} | {model['split']} | "
                f"{model[f'{name}_default_counted_disagreements']} | {model[f'{name}_tuned_counted_disagreements']} | "
                f"{model[f'{name}_tuned_new_vs_default']} | {model[f'{name}_tuned_invalid_queries']} |"
            )
    timings = summary["timings_seconds"]
    timing_rows = read_rows(path / "timings.csv")
    worst_distance = max(timing_rows, key=lambda row: float(row["float32_distance_error_max_bbd"]))
    lines += [
        "",
        "## Reproduction, timings, and retained data",
        "",
        f"The runtime default classifier reproduces all **{base['points']:,}** baseline signs and normal bytes "
        "in each precision. Float uses a new full default signed-distance pass; double reuses its cached closest "
        "faces, points, and unsigned distances. Within each precision, default and tuned classification "
        "share the same closest geometry, so changing the tolerance does not change unsigned distances. "
        "Normals are precomputed once per runtime mesh. No recompilation occurs between parameters.",
        "",
        f"The largest float-versus-original-double unsigned-distance difference is "
        f"{float(worst_distance['float32_distance_error_max_bbd']):.8g} × the bounding-box diagonal "
        f"on model `{worst_distance['file_id']}`. These distance differences already exist in the "
        "default float query and are shared by both float sign settings.",
        "",
        "The auditor independently recounts every point in all four variants and all three model groups, "
        "checks every per-model and aggregate integer metric, and recomputes validity from saved outputs. "
        "It also checks the original million inputs and winding masks, exact float baseline geometry/normal bytes, "
        "and both tuned sign/validity results against the original sweeps.",
        "",
        f"Validation run wall time: **{summary['wall_seconds']:.1f} s**. "
        f"New float full default queries: {timings['float32_full_default_query_seconds']:.1f} s. "
        f"Runtime normal preprocessing: {timings['float32_normal_precompute_seconds'] + timings['float64_normal_precompute_seconds']:.1f} s. "
        f"All four runtime classifier calls: {sum(timings[f'{name}_classifier_seconds'] for name in VARIANTS):.1f} s. "
        f"Compressed point output: {timings['output_seconds']:.1f} s. "
        "Cached double nearest-point and winding computations are excluded. This is not an end-to-end timing "
        "comparison between the two precisions. [Per-model timings](timings.csv).",
        "",
        "[Configuration and binary/source provenance](config.json), [summary](summary.json), "
        "[independent audit](validation.json), [per-model metrics](per_model.csv), "
        "[side-by-side model comparison](model_comparison.csv), and [remaining cases](remaining_cases.csv). "
        "Remaining cases include any tuned disagreement beyond the cutoff or any invalid tuned query, "
        "with dataset file ID, point index, coordinates, raw winding, and all four signs and validity flags. "
        "All per-point arrays remain cached locally; meshes and reference points were not downloaded again.",
        "",
        "Epsilon has dimensionless barycentric semantics in one branch and absolute coordinate-distance semantics "
        "in the small-area branch; the area threshold has squared coordinate units. These original-scale results "
        "do not establish behavior under arbitrary scaling or translation. No upstream defaults were changed.",
        "",
        "Prepared by OpenAI Codex.",
    ]
    (path / "FINDINGS.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    args = parser.parse_args()
    run(args.results, args.cache_dir)
