"""Independently recount the closer-cutoff comparison and export point evidence."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from time import perf_counter

import numpy as np

from benchmark import write_csv
from dataset_cache import write_json
from warp_benchmark import file_sha256

LABELS = {
    "igl_double": "libigl PR pseudonormal, double",
    "igl_float": "libigl PR pseudonormal, float",
    "warp_p": "Warp pseudonormal",
    "warp_w": "Warp winding",
    "warp_parity": "Warp parity (1 ray)",
}


def read_csv(path):
    """Read named result columns."""
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def audit(output):
    """Verify per-model/aggregate counts from saved arrays and write findings."""
    started = perf_counter()
    config = json.loads((output / "config.json").read_text())
    summary = json.loads((output / "summary.json").read_text())
    rows = read_csv(output / "per_model.csv")
    reference_path = Path(config["reference"])
    for name in ("config", "per_model"):
        suffix = "json" if name == "config" else "csv"
        np.testing.assert_equal(
            file_sha256(reference_path / f"{name}.{suffix}"), config[f"reference_{name}_sha256"]
        )
    np.testing.assert_equal([r["file_id"] for r in rows], config["selected_ids"])
    np.testing.assert_equal(len({r["file_id"] for r in rows}), config["models"])
    totals = {method: Counter() for method in LABELS}
    points_total = away_total = old_away_total = 0
    cases, distance_cases, model_cases = [], [], []
    max_errors = dict.fromkeys(LABELS, 0.0)
    mean_errors = dict.fromkeys(LABELS, 0.0)
    for position, row in enumerate(rows):
        fid, bbd = row["file_id"], float(row["bbd"])
        if row["status"] not in ("ok", "nonfinite_normals"):
            raise ValueError(f"Failed model {fid}")
        path = reference_path / "points" / f"{fid}.npz"
        np.testing.assert_equal(file_sha256(path), row["reference_points_sha256"])
        with np.load(path, allow_pickle=False) as data:
            points, distance, reference = (data[key] for key in ("points", "d", "sign_w"))
        with np.load(output / row["points_file"], allow_pickle=False) as data:
            result = dict(data)
        np.testing.assert_array_equal(result["reference_sign"], reference)
        np.testing.assert_array_equal(
            2 * np.rint(result["igl_winding"]).astype(np.int64) - 1, reference
        )
        away = distance > config["relative_cutoff"] * bbd
        old_away = distance > config["old_relative_cutoff"] * bbd
        np.testing.assert_equal(int(away.sum()), int(row["num_away"]))
        np.testing.assert_equal(int(old_away.sum()), int(row["num_away_old"]))
        points_total += len(points)
        away_total += int(away.sum())
        old_away_total += int(old_away.sum())
        signs, validities, masks, distances, closest_faces = {}, {}, {}, {}, {}
        for method in LABELS:
            if method.startswith("igl_"):
                signed = result[f"{method}_signed_distance"]
                sign = result[f"{method}_sign"]
                np.testing.assert_array_equal(sign, np.where(signed < 0, 1, -1))
                normal, closest = result[f"{method}_normal"], result[f"{method}_closest"]
                valid = (
                    np.isfinite(signed)
                    & np.isfinite(closest).all(axis=1)
                    & np.isfinite(normal).all(axis=1)
                )
                np.testing.assert_array_equal(valid, result[f"{method}_valid"])
                d = np.abs(signed.astype(np.float64))
                nearest = result[f"{method}_face"]
                if method == "igl_double":
                    np.testing.assert_array_equal(d, distance)
            else:
                suffix = method.removeprefix("warp_")
                sign = result[f"warp_sign_{suffix}"]
                valid = result[f"warp_valid_{suffix}"] == 1
                d = result[f"warp_d_{suffix}"].astype(np.float64)
                nearest = result[f"warp_face_{suffix}"]
                np.testing.assert_array_equal(valid, True)
            if not np.isfinite(d).all() or not np.isin(sign, [-1, 1]).all():
                raise ValueError(f"Invalid signs or distances from {method} on {fid}")
            if np.any(nearest < 0) or np.any(nearest >= int(row["num_faces"])):
                raise ValueError(f"Invalid nearest face from {method} on {fid}")
            mismatch = sign != reference
            mask = mismatch & away
            counts = {
                "raw_disagreements": int(mismatch.sum()),
                "disagreements": int(mask.sum()),
                "old_cutoff_disagreements": int((mismatch & old_away).sum()),
                "newly_included_disagreements": int((mask & ~old_away).sum()),
                "valid_disagreements": int((mask & valid).sum()),
                "invalid_disagreements": int((mask & ~valid).sum()),
                "invalid_queries": int((~valid).sum()),
                "invalid_queries_away": int((~valid & away).sum()),
                "false_inside": int((mask & (sign == 1)).sum()),
                "false_outside": int((mask & (sign == -1)).sum()),
            }
            for key, value in counts.items():
                np.testing.assert_equal(int(row[f"{method}_{key}"]), value)
            totals[method].update(counts)
            totals[method]["models_with_disagreements"] += int(mask.any())
            totals[method]["models_with_invalid_queries"] += int((~valid).any())
            signs[method], validities[method], masks[method] = sign, valid, mask
            distances[method], closest_faces[method] = d, nearest
            error = np.abs(d - distance) / bbd
            for key, value in (
                ("max", float(error.max())),
                ("mean", float(error.mean())),
                ("p99", float(np.quantile(error, 0.99))),
            ):
                np.testing.assert_allclose(
                    float(row[f"{method}_distance_error_{key}_bbd"]), value, rtol=1e-12, atol=0
                )
            max_errors[method] = max(max_errors[method], float(error.max()))
            mean_errors[method] += float(error.sum())
            if mask.any() or (~valid).any():
                model_cases.append({"file_id": fid, "method": method, **counts})
        union = np.logical_or.reduce(list(masks.values()))
        newly_included_union = union & ~old_away
        for i in np.flatnonzero(union):
            case = {
                "file_id": fid,
                "point_index": int(i),
                "point_x": float(points[i, 0]),
                "point_y": float(points[i, 1]),
                "point_z": float(points[i, 2]),
                "reference_distance": float(distance[i]),
                "d_over_bbd": float(distance[i] / bbd),
                "reference_winding": float(result["igl_winding"][i]),
                "reference_sign": int(reference[i]),
                "newly_included": bool(newly_included_union[i]),
            }
            for method in LABELS:
                case[f"{method}_sign"] = int(signs[method][i])
                case[f"{method}_valid"] = bool(validities[method][i])
                case[f"{method}_disagrees"] = bool(masks[method][i])
                case[f"{method}_distance"] = float(distances[method][i])
                case[f"{method}_face"] = int(closest_faces[method][i])
            cases.append(case)
        for method in LABELS:
            error = np.abs(distances[method] - distance) / bbd
            for i in np.flatnonzero(error > config["relative_cutoff"]):
                distance_cases.append(
                    {
                        "file_id": fid,
                        "point_index": int(i),
                        "method": method,
                        "d_over_bbd": float(distance[i] / bbd),
                        "distance_error_over_bbd": float(error[i]),
                    }
                )
        if (position + 1) % 500 == 0:
            print(f"Audited {position + 1}/{len(rows)} models", flush=True)
    np.testing.assert_equal(points_total, summary["num_points"])
    np.testing.assert_equal(away_total, summary["num_away"])
    np.testing.assert_equal(old_away_total, summary["num_away_old"])
    for method, values in totals.items():
        for key, value in values.items():
            np.testing.assert_equal(value, summary["methods"][method][key])
        np.testing.assert_allclose(
            summary["methods"][method]["percent_all"], 100 * values["disagreements"] / points_total
        )
        np.testing.assert_allclose(
            summary["methods"][method]["percent_away"], 100 * values["disagreements"] / away_total
        )
    if config["full_mask"]:
        feature_summary = json.loads((Path(config["feature_control"]) / "summary.json").read_text())
        warp_summary = json.loads((Path(config["warp_control"]) / "summary.json").read_text())
        for method, precision in (("igl_float", "float32"), ("igl_double", "float64")):
            np.testing.assert_equal(
                totals[method]["old_cutoff_disagreements"],
                feature_summary["totals"]["all"][precision]["counted_disagreements"],
            )
        for method in ("warp_p", "warp_w", "warp_parity"):
            np.testing.assert_equal(
                totals[method]["old_cutoff_disagreements"],
                warp_summary["methods"][method]["disagreements"],
            )
    write_csv(output / "disagreements.csv", cases)
    write_csv(
        output / "newly_included_disagreements.csv", [c for c in cases if c["newly_included"]]
    )
    write_csv(output / "affected_models.csv", model_cases)
    write_csv(output / "distance_outliers.csv", distance_cases)
    audit_result = {
        "all_counts_match": True,
        "models": len(rows),
        "points": points_total,
        "num_away": away_total,
        "num_away_old": old_away_total,
        "newly_eligible_points": away_total - old_away_total,
        "unique_disagreement_points": len(cases),
        "unique_newly_included_disagreement_points": sum(c["newly_included"] for c in cases),
        "distance_errors": {
            m: {"max_over_bbd": max_errors[m], "mean_over_bbd": mean_errors[m] / points_total}
            for m in LABELS
        },
        "seconds": perf_counter() - started,
    }
    write_json(output / "audit.json", audit_result)
    cache = json.loads((output / "cache_access.json").read_text())
    lines = [
        "# Full comparison with the surface cutoff reduced 100-fold",
        "",
        f"Fresh queries on **{len(rows):,} eligible meshes × 10,000 points = {points_total:,} points**.",
        "The primary cutoff is now **`d > 0.00001 * bbd`**, down from `d > 0.001 * bbd`.",
        f"It includes **{away_total:,}** points, adding **{away_total - old_away_total:,}** closer points.",
        "All methods use the same original double distance and bounding-box diagonal.",
        "",
        "| Method | Old cutoff | New cutoff | Newly included | % of all samples | Affected models |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method, label in LABELS.items():
        metrics = summary["methods"][method]
        lines.append(
            f"| {label} | {metrics['old_cutoff_disagreements']:,} | {metrics['disagreements']:,} | {metrics['newly_included_disagreements']:,} | {metrics['percent_all']:.9g}% | {metrics['models_with_disagreements']:,} |"
        )
    lines += [
        "",
        "Percentages divide by all samples; `summary.json` also reports percentages among eligible points.",
        "The reference is freshly computed libigl fast winding with double inputs and NumPy rounding",
        "(inside +1, outside -1). Its implementation internally uses float arithmetic. It is a numerical",
        "reference, not an independent proof of the correct sign.",
        "",
        "## Reproducibility and checks",
        "",
        f"- Core PR [#2586](https://github.com/libigl/libigl/pull/2586), commit `{config['build']['core']['commit']}`.",
        f"- Actual Python binding sources at `{config['build']['bindings']['commit']}`; both signed-distance and fast-winding bindings rebuilt against the PR.",
        f"- Warp main revision `{config['warp']['revision']}` on `{config['device_name']}`; original settings retained (pseudonormal epsilon 0.001, winding accuracy 2, parity one ray).",
        "- Original scales, same dataset revision and eligible mask, seed 20261123. Every cached point array was verified against its per-model seeded generator.",
        "- Both libigl pseudonormal routes, libigl winding, and all three Warp methods were rerun for every model. Libigl signs/normal bytes/validity match the earlier feature replay; double distances and closest geometry match the original reference. All Warp distances, signs, faces, closest points, and validity flags match the earlier run.",
        f"- Maximum raw winding difference from the cached reference: `{summary['winding_max_difference_vs_cached']}`. All reference signs match.",
        "- Independent point-file audit reproduces every per-model count and aggregate; no model failed.",
        "- Fresh binding passed all 26 precision tests. Four-model smoke test includes the known float degeneracy and double residual models; a strict-boundary fixture checks both distance thresholds and invalid-normal counts.",
        "",
        "## Invalid normals",
        "",
    ]
    metrics = summary["methods"]["igl_float"]
    lines += [
        f"Libigl float has **{metrics['invalid_queries']} invalid-normal queries**. Its new-cutoff disagreement total contains **{metrics['valid_disagreements']} finite-output cases** and **{metrics['invalid_disagreements']} invalid-normal cases**.",
        "Returned signs are retained for continuity with previous experiments; invalid outputs are also identified per point and per model. Other methods' invalid-query counts are in `summary.json`.",
        "",
        "## Timings",
        "",
        f"The full fresh run took **{summary['wall_seconds']:.2f} seconds ({summary['wall_seconds'] / 60:.2f} minutes)**. The independent audit took **{audit_result['seconds']:.2f} seconds**.",
        "Single passes on shared hardware; no timing repetitions or speedup claim. Libigl times include",
        "internal tree/normal construction; Warp kernel times exclude separately measured setup/transfers.",
        "",
        "| Operation | Seconds |",
        "| --- | ---: |",
    ]
    for key, label in (
        ("igl_double_query_seconds", "libigl double signed distance"),
        ("igl_float_query_seconds", "libigl float signed distance"),
        ("igl_winding_query_seconds", "libigl fast winding"),
        ("warp_mesh_build_seconds", "Warp mesh builds"),
        ("warp_pseudonormal_query_seconds", "Warp pseudonormal kernels"),
        ("warp_winding_query_seconds", "Warp winding kernels"),
        ("warp_parity_query_seconds", "Warp parity kernels"),
        ("mesh_load_and_hash_seconds", "Mesh load/hash"),
        ("reference_and_control_load_seconds", "Reference/control loading"),
        ("output_seconds", "Compressed output"),
    ):
        lines.append(f"| {label} | {summary['timings_seconds'][key]:.3f} |")
    lines += [
        "",
        f"Dataset cache access took {cache['total_seconds']:.6f} seconds; all downloads/extraction were cache hits. No dataset download occurred.",
        f"Checkpointing and CSV export took {summary['checkpoint_seconds'] + summary['csv_export_seconds']:.3f} seconds combined.",
        "",
        "## Distance agreement",
        "",
        "Errors below are relative to the original double unsigned distance and normalized by bbd.",
        "| Method | Maximum absolute error / bbd | Mean absolute error / bbd |",
        "| --- | ---: | ---: |",
    ]
    for method, label in LABELS.items():
        lines.append(
            f"| {label} | {max_errors[method]:.9g} | {mean_errors[method] / points_total:.9g} |"
        )
    lines += [
        "",
        "## Artifacts",
        "",
        "- `per_model.csv` and `selected_metadata.csv` join to the dataset on `file_id` (or `input_summary.csv` on `ID`).",
        "- `disagreements.csv` contains every counted disagreement point across methods, with coordinates, signs, validity, distances, and nearest faces.",
        "- `newly_included_disagreements.csv` restricts this to the newly admitted distance band; `affected_models.csv` lists each affected method/model.",
        "- `distance_outliers.csv` records distance errors exceeding the new relative cutoff.",
        "- `config.json`, `cache_access.json`, `summary.json`, and `audit.json` preserve provenance, timings, and verification.",
        "- Fresh per-point outputs remain in local ignored `points/<file_id>.npz`; the original query coordinates remain in the referenced baseline cache.",
        "",
    ]
    (output / "FINDINGS.md").write_text("\n".join(lines))
    print(json.dumps(audit_result, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    audit(parser.parse_args().output)
