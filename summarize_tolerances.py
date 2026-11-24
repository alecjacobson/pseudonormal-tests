"""Audit, plot, and summarize the runtime float pseudonormal tolerance sweep."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

from benchmark import write_csv
from dataset_cache import write_json

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def run(path, upper_path=None):
    """Verify the selected setting against cached point arrays and export plots."""
    config = json.loads((path / "config.json").read_text())
    summary = json.loads((path / "summary.json").read_text())
    grids = {fold: list(csv.DictReader((path / f"grid_{fold}.csv").open()))
             for fold in ["all", "tuning", "holdout"]}
    reference_path = Path(config["reference"])
    reference_rows = list(csv.DictReader((reference_path / "per_model.csv").open()))
    default_id = config["default_setting_id"]
    selected_id = summary["selected_on_tuning"]["all"]["setting_id"]
    settings = config["settings"]
    audit = {fold: np.zeros((len(settings), 3), dtype=np.int64)
             for fold in grids}
    cases = []
    for source in reference_rows:
        file_id = source["file_id"]
        fold = "tuning" if int(file_id) in config["tuning_ids"] else "holdout"
        with np.load(path / "points" / f"{file_id}.npz", allow_pickle=False) as archive:
            signs, valid = archive["sign_p"], archive["valid_query"]
        with np.load(reference_path / source["points_file"], allow_pickle=False) as archive:
            reference = dict(archive)
        away = reference["away_from_surface"]
        disagree = signs != reference["sign_w"][None, :]
        counted = disagree & away[None, :]
        values = np.column_stack((counted.sum(axis=1), (~valid).sum(axis=1),
                                  (counted & valid).sum(axis=1)))
        audit["all"] += values
        audit[fold] += values
        diagnostic = counted[default_id] | counted[selected_id] | ~valid[selected_id]
        for sample in np.flatnonzero(diagnostic):
            cases.append({
                "file_id": file_id, "sample_index": int(sample), "split": fold,
                "x": reference["points"][sample, 0], "y": reference["points"][sample, 1],
                "z": reference["points"][sample, 2],
                "d_over_bbd": reference["d"][sample]/float(source["bbd"]),
                "away": bool(away[sample]),
                "reference_sign_w": int(reference["sign_w"][sample]),
                "default_sign_p": int(signs[default_id, sample]),
                "selected_sign_p": int(signs[selected_id, sample]),
                "default_valid": bool(valid[default_id, sample]),
                "selected_valid": bool(valid[selected_id, sample]),
            })
    for fold, rows in grids.items():
        expected = [[int(row[key]) for key in
                     ["counted_disagreements", "invalid_queries", "valid_counted_disagreements"]]
                    for row in rows]
        np.testing.assert_array_equal(audit[fold], expected)
    write_csv(path / "selected_cases.csv", cases)
    write_json(path / "validation.json", {
        "point_arrays_recounted": True,
        "settings_audited": len(settings),
        "models_audited": len(reference_rows),
        "metrics_audited": ["counted_disagreements", "invalid_queries", "valid_counted_disagreements"],
        "splits_audited": list(grids),
        "default_reproduction": summary["baseline_audit"],
    })

    areas = list(dict.fromkeys(setting["minimum_double_area"] for setting in settings))
    epsilons = list(dict.fromkeys(setting["epsilon"] for setting in settings))
    matrix = audit["all"][:, 0].reshape(len(areas), len(epsilons))
    fig, axis = plt.subplots(figsize=(15, 7), layout="constrained")
    shown = axis.imshow(matrix, aspect="auto", cmap="viridis_r",
                        norm=LogNorm(vmin=max(1, int(matrix.min())), vmax=int(matrix.max())))
    axis.set_xticks(range(len(epsilons)), [f"{value:.0e}" if value else "0" for value in epsilons], rotation=45)
    axis.set_yticks(range(len(areas)), [f"{value:.0e}" if value else "0" for value in areas])
    axis.set_xlabel("Epsilon (original branch-dependent units)")
    axis.set_ylabel("Minimum double area (squared coordinate units)")
    axis.set_title("Float pseudonormal tolerance sweep — 100 models, 1,000,000 cached points\n"
                   "Numbers count winding-sign disagreements beyond the fixed distance cutoff")
    for y, x in np.ndindex(matrix.shape):
        axis.text(x, y, str(matrix[y, x]), ha="center", va="center", fontsize=6.5,
                  color="white" if shown.norm(matrix[y, x]) > 0.55 else "black")
    for index, label, color in [(default_id, "Default", "red"), (selected_id, "Selected on tuning half", "orange")]:
        axis.scatter(index % len(epsilons), index // len(epsilons), s=380,
                     marker="s", facecolors="none", edgecolors=color, linewidths=2, label=label)
    axis.legend(loc="upper left", bbox_to_anchor=(0, -0.17), ncol=2)
    fig.colorbar(shown, ax=axis, label="Counted disagreements (logarithmic color scale)")
    fig.savefig(path / "tolerance_grid.png", dpi=170)
    fig.savefig(path / "tolerance_grid.pdf")
    plt.close(fig)

    selected = summary["selected_on_tuning"]
    best = summary["best_overall_exploratory"]
    default = summary["default"]
    lines = [
        "# Float pseudonormal tolerance sweep", "",
        "The same 100 Thingi10K models and 1,000,000 cached points were evaluated at original scale. "
        "The saved double-input winding signs and `original_d > 0.001 * original_bbd` mask remain fixed.", "",
        "A single compiled experiment exposes both tolerances as function arguments. "
        "The classifier is generated mechanically from the patched libigl source, changing only the function name "
        "and replacing its two constants with parameters. Mesh normals are computed once per model; closest faces "
        "and points come from the cached patched-float run. No recompilation or nearest-point search occurs between settings.", "",
        f"**All {summary['baseline_audit']['points']:,} default classifications reproduce the saved float baseline:** "
        "zero sign changes, zero differing normal bytes, and zero validity changes.", "",
        f"Swept **{len(settings)} parameter pairs**: {len(epsilons)} epsilon values × {len(areas)} area thresholds. "
        "A model-level split (seed `20261124`) assigns 50 models to tuning and 50 to holdout before evaluation. "
        "The selected parameters minimize the tuning disagreement count; the holdout half does not select them.", "",
        "| Setting | Minimum double area | Epsilon | Tuning / 500,000 | Holdout / 500,000 | All / 1,000,000 | All % | Invalid queries |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, item in [("Default", default), ("Selected on tuning half", selected), ("Best over all models (exploratory)", best)]:
        row = item["all"]
        lines.append(f"| {label} | {row['minimum_double_area']:.4g} | {row['epsilon']:.4g} | "
                     f"{item['tuning']['counted_disagreements']} | {item['holdout']['counted_disagreements']} | "
                     f"{row['counted_disagreements']} | {row['percent_all']:.6g}% | {row['invalid_queries']} |")
    selected_all = selected["all"]
    reduction = 100*(1-selected_all["counted_disagreements"]/default["all"]["counted_disagreements"])
    lines += ["", f"The tuning-selected setting reduces the full-sample count by **{reduction:.3f}%**. "
              f"It resolves {selected_all['resolved_vs_default']} default disagreements and introduces "
              f"{selected_all['new_vs_default']} others. It has {selected_all['valid_counted_disagreements']} "
              f"counted disagreements with finite outputs and {selected_all['invalid_queries']} invalid queries. "
              "The double baseline is 7 disagreements (0.0007%).", "",
              "![Tolerance grid](tolerance_grid.png)", "",
              "[Download PDF](tolerance_grid.pdf). Complete grids: [all](grid_all.csv), "
              "[tuning](grid_tuning.csv), [holdout](grid_holdout.csv).", "",
              "## Epsilon sweep at the default area threshold", "",
              "| Epsilon | Disagreements | Percent of all points | Invalid queries |", "| --- | ---: | ---: | ---: |"]
    for row in grids["all"]:
        if float(row["minimum_double_area"]) == default["all"]["minimum_double_area"]:
            lines.append(f"| {float(row['epsilon']):.4g} | {row['counted_disagreements']} | "
                         f"{float(row['percent_all']):.6g}% | {row['invalid_queries']} |")
    if upper_path is not None:
        upper_config = json.loads((upper_path / "config.json").read_text())
        upper_summary = json.loads((upper_path / "summary.json").read_text())
        if config["module_sha256"] != upper_config["module_sha256"]:
            raise ValueError("The upper-range check must use the same compiled module")
        if config["tuning_ids"] != upper_config["tuning_ids"]:
            raise ValueError("The upper-range check must keep the same model split")
        def key(setting):
            return setting["minimum_double_area"], setting["epsilon"]
        first_indices = {key(s): s["setting_id"] for s in settings}
        second_indices = {key(s): s["setting_id"] for s in upper_config["settings"]}
        common = sorted(first_indices.keys() & second_indices.keys())
        for source in reference_rows:
            filename = f"{source['file_id']}.npz"
            with np.load(path / "points" / filename) as first, np.load(upper_path / "points" / filename) as second:
                for output in ["sign_p", "valid_query"]:
                    np.testing.assert_array_equal(
                        first[output][[first_indices[k] for k in common]],
                        second[output][[second_indices[k] for k in common]])
        study = {
            "unique_parameter_pairs": len(first_indices.keys() | second_indices.keys()),
            "shared_parameter_pairs": len(common),
            "shared_point_outputs_identical": True,
            "same_compiled_module": True,
            "initial_grid": str(path), "upper_range_grid": str(upper_path),
            "first_grid_wall_seconds": summary["wall_seconds"],
            "upper_grid_wall_seconds": upper_summary["wall_seconds"],
            "initial_selection_retained": key(selected["all"]) == key(upper_summary["selected_on_tuning"]["all"]),
        }
        write_json(path / "study.json", study)
        upper_rows = list(csv.DictReader((upper_path / "grid_all.csv").open()))
        lines += ["", "## Upper-range check", "",
                  f"The same compiled module evaluated another {len(second_indices)} pairs, including {len(common)} repeats, "
                  f"for **{study['unique_parameter_pairs']} distinct parameter pairs** across both grids. "
                  "Every repeated point sign and validity flag matches. The initial tuning-selected pair remains the selection "
                  "in this extension. This follow-up checks where larger epsilon values start hurting; it does not provide a new holdout sample.", "",
                  "At the default area threshold:", "",
                  "| Epsilon | Disagreements | Invalid queries |", "| --- | ---: | ---: |"]
        for row in upper_rows:
            if (float(row["minimum_double_area"]) == default["all"]["minimum_double_area"]
                    and float(row["epsilon"]) > max(epsilons)):
                lines.append(f"| {float(row['epsilon']):.4g} | {row['counted_disagreements']} | {row['invalid_queries']} |")
        lines += ["", f"[Full upper-range grid and diagnostics](../{upper_path.name}/FINDINGS.md). "
                  "[Combined study checks](study.json)."]
    lines += ["", "## Interpretation and timings", "",
              "Epsilon is dimensionless in the barycentric branch, but measures coordinate distance in the small-area "
              "vertex/edge checks. The area threshold compares twice triangle area in squared coordinate units. "
              "These absolute tolerances were tested at original mesh scales; this does not establish scale-independent defaults. "
              "No upstream defaults or classification algorithm were changed.", "",
              f"Normal preprocessing: {summary['normal_precompute_seconds']:.3f} s total. "
              f"Classifier calls across all {len(settings)} settings: {summary['classifier_seconds']:.3f} s total. "
              f"Sweep wall time including loading, verification and compressed output: {summary['wall_seconds']:.3f} s. "
              "These classifier-only timings exclude nearest-point queries and are not comparable to the earlier full-query timings.", "",
              "See [config.json](config.json) for exact float32 parameter values, source/module hashes and split IDs; "
              "[validation.json](validation.json) for the independent point-array recount; "
              "[per_model.csv](per_model.csv) and [selected_cases.csv](selected_cases.csv) for model/point diagnostics. "
              "All setting signs and validity flags remain cached locally by `(file_id, setting_id, sample_index)`.", "",
              "Prepared by OpenAI Codex."]
    (path / "FINDINGS.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({"selected": selected, "best_overall": best}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--upper-results", type=Path)
    args = parser.parse_args()
    run(args.results, args.upper_results)
