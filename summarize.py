"""Audit retained point outputs and export a compact, shareable experiment report."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from benchmark import write_csv
from dataset_cache import write_json


def summarize(output):
    """Verify the saved measurements and export every counted disagreement."""
    output = Path(output)
    config = json.loads((output / "config.json").read_text())
    summary = json.loads((output / "summary.json").read_text())
    download = json.loads((output / "download.json").read_text())
    with (output / "per_model.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != config["models"] or len({row["file_id"] for row in rows}) != len(rows):
        raise ValueError("The result rows do not match the requested unique model count.")
    if any(row["status"] == "error" for row in rows):
        raise ValueError("Resolve failed model rows before summarizing the experiment.")
    cases = []
    total_away = 0
    total_points = 0
    total_raw_disagreements = 0
    for row in rows:
        with np.load(output / row["points_file"], allow_pickle=False) as arrays:
            points = arrays["points"]
            if len(points) != config["points_per_model"]:
                raise ValueError(f"Unexpected point count for {row['file_id']}.")
            lower = np.array([float(row[f"bbox_min_{axis}"]) for axis in "xyz"])
            upper = np.array([float(row[f"bbox_max_{axis}"]) for axis in "xyz"])
            if np.any(points < lower) or np.any(points > upper):
                raise ValueError(f"Points outside the bounding box for {row['file_id']}.")
            # Check the saved distance against libigl's closest-point coordinates.
            np.testing.assert_allclose(
                arrays["d"],
                np.linalg.norm(points - arrays["closest_point"], axis=1),
                atol=1e-12 * float(row["bbd"]),
                rtol=1e-10,
            )
            np.testing.assert_array_equal(
                arrays["sign_p"], np.where(arrays["signed_distance"] < 0, 1, -1)
            )
            np.testing.assert_array_equal(
                arrays["sign_w"], 2 * np.rint(arrays["winding"]).astype(np.int64) - 1
            )
            away = arrays["d"] > config["relative_cutoff"] * float(row["bbd"])
            disagree = arrays["sign_p"] != arrays["sign_w"]
            counted = disagree & away
            np.testing.assert_array_equal(arrays["counted_disagreement"], counted)
            np.testing.assert_equal(int(counted.sum()), int(row["num_counted_disagreements"]))
            np.testing.assert_allclose(100 * counted.mean(), float(row["disagreement_percent_all"]))
            total_away += int(away.sum())
            total_points += len(points)
            total_raw_disagreements += int(disagree.sum())
            for index in np.flatnonzero(counted):
                case = {
                    "file_id": row["file_id"],
                    "point_index": int(index),
                    "d": float(arrays["d"][index]),
                    "d_over_bbd": float(arrays["d"][index]) / float(row["bbd"]),
                    "distance_cutoff": float(row["distance_cutoff"]),
                    "sign_p": int(arrays["sign_p"][index]),
                    "sign_w": int(arrays["sign_w"][index]),
                    "winding": float(arrays["winding"][index]),
                    "closest_face_index": int(arrays["closest_face_index"][index]),
                }
                for array_name in ["points", "closest_point", "pseudonormal"]:
                    for axis, value in zip("xyz", arrays[array_name][index], strict=True):
                        case[f"{array_name}_{axis}"] = float(value)
                cases.append(case)
    np.testing.assert_equal(len(cases), summary["num_counted_disagreements"])
    np.testing.assert_equal(total_away, summary["num_away"])
    np.testing.assert_equal(total_points, summary["num_points"])
    if cases:
        write_csv(output / "disagreements.csv", cases)
    else:
        (output / "disagreements.csv").write_text(
            "file_id,point_index,d,d_over_bbd,sign_p,sign_w,winding\n"
        )
    write_json(
        output / "validation.json",
        {
            "models_checked": len(rows),
            "points_checked": total_points,
            "checks": [
                "unique model count",
                "points in bounding boxes",
                "distance agrees with closest point",
                "sign conversions",
                "strict cutoff",
                "per-model primary rates",
                "aggregate counts",
            ],
            "result": "passed",
        },
    )
    affected = [row for row in rows if int(row["num_counted_disagreements"])]
    environment = config["environment"]
    report = [
        "# Benchmark findings\n",
        f"Run completed {summary['completed_at_utc']}. Dataset revision `{config['dataset_revision']}`; "
        f"seed `{config['seed']}`. Sampled {len(rows)} of {config['eligible_models']:,} eligible models "
        f"and {config['points_per_model']:,} points per model. Eligibility requires solid, vertex-manifold, "
        "and edge-manifold metadata flags; multiple components are allowed.\n",
        f"**{len(cases)} disagreements beyond the cutoff across {len(affected)} models**, "
        f"out of {total_points:,} sampled points: **{summary['disagreement_percent_all']:.6g}%**. "
        f"Among the {total_away:,} points with `d > {config['relative_cutoff']} * bbd`, "
        f"the disagreement rate is {summary['disagreement_percent_away']:.9g}%. "
        f"There were {total_raw_disagreements} sign disagreements before applying the cutoff. "
        f"There were {summary['models_failed']} failed models and "
        f"{summary['models_with_nonbinary_winding']} models with nonbinary rounded winding values.\n",
        "## Models with disagreements\n",
        "| file_id | Faces | Components | Points beyond cutoff | Disagreements | % of all points | % beyond cutoff |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in affected:
        report.append(
            f"| {row['file_id']} | {int(row['num_faces']):,} | {row['num_connected_components']} | "
            f"{int(row['num_away']):,} | {row['num_counted_disagreements']} | "
            f"{float(row['disagreement_percent_all']):.4g} | "
            f"{float(row['disagreement_percent_away']):.7g} |"
        )
    report.extend(
        [
            "\nThe complete [per-model CSV](per_model.csv) joins to the dataset geometry CSV on `file_id`, "
            "or to `input_summary.csv` on `ID`. The [selected metadata](selected_metadata.csv) retains all "
            "geometry columns. Every counted case, including coordinates and raw winding value, is in "
            "[disagreements.csv](disagreements.csv).\n",
        ]
    )
    if cases:
        signs = sorted({(case["sign_p"], case["sign_w"]) for case in cases})
        report.append(
            f"Observed `(sign_p, sign_w)` pairs: `{signs}`. The disagreement points have "
            f"`d / bbd` between {min(case['d_over_bbd'] for case in cases):.7g} and "
            f"{max(case['d_over_bbd'] for case in cases):.7g}. "
            "This comparison does not independently establish which method is correct.\n"
        )
    timings = summary["summed_timings_seconds"]
    report.extend(
        [
            "## Timings\n",
            "| Step | Seconds |",
            "| --- | ---: |",
            f"| Initial download (archive and metadata) | {sum(item['seconds'] for item in download['files']):.6f} |",
            f"| Initial archive extraction | {download['extraction']['seconds']:.6f} |",
            f"| Current run cache access | {summary['cache_access_seconds']:.6f} |",
            f"| Metadata CSV loading | {config['metadata_load_seconds']:.6f} |",
            f"| Mesh loading, summed over models | {timings['mesh_load_seconds']:.6f} |",
            f"| Pseudonormal signed-distance calls, summed | {timings['pseudonormal_seconds']:.6f} |",
            f"| Fast winding calls, summed | {timings['fast_winding_seconds']:.6f} |",
            f"| Per-point output compression/writing, summed | {timings['point_output_seconds']:.6f} |",
            f"| Benchmark wall time with dataset cached | {summary['run_wall_seconds']:.6f} |",
            f"\nEnvironment: Python {environment['python']}, libigl {environment['packages']['libigl']}, "
            f"NumPy {environment['packages']['numpy']}, {environment['cpu_model']}; "
            f"`IGL_NUM_THREADS={environment['igl_num_threads']}`, cgroup CPU limit "
            f"`{environment['cgroup_cpu_max']}`. The run uses the libigl CPU implementation.\n",
            "Each method was called once per model, without warmup. Both call timings include internal "
            "acceleration-structure setup; pseudonormal timing also includes the distance query. These "
            "are batch costs, not isolated sign evaluation or precomputed-tree query timings. Mesh "
            "loading excludes the separately timed download and extraction. OS file caching and shared "
            "machine load can affect timings.\n",
            "## Validation and reproducibility\n",
            "The repository's analytic tests cover cube distances and signs, a two-component cavity boundary, "
            "the strict cutoff, denominator choices, and deterministic sampling. The point-output audit "
            "passed for every retained sample; see [validation.json](validation.json). Cache hit records "
            "for this run are in [cache_access.json](cache_access.json).\n",
            "Reproduce with the committed lockfile, default parameters, and a fresh output directory. "
            "Exact selected IDs and source hashes are retained in the result files. Full per-point arrays "
            "are cached locally in `points/`; they are not committed.\n",
            "Prepared with OpenAI Codex.\n",
        ]
    )
    (output / "FINDINGS.md").write_text("\n".join(report))
    print(
        f"Audited {len(rows)} models / {total_points:,} points; exported {len(cases)} disagreements."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    summarize(parser.parse_args().output)
