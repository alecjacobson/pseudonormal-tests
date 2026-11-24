"""Rerun PR libigl and pinned Warp on cached full-mask samples at a new cutoff."""

import argparse
import csv
import importlib.util
import json
import os
import shutil
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import warp as wp

import warp_queries
from benchmark import append_result, environment_info, load_mesh, sample_points, write_csv
from dataset_cache import DEFAULT_CACHE, REVISION, prepare_dataset, write_json
from warp_benchmark import file_sha256, verify_warp_source
from warp_queries import WARP_METHODS, query_warp

METHODS = ("igl_float", "igl_double", "warp_p", "warp_w", "warp_parity")
OLD_CUTOFF = 0.001


def read_rows(path):
    """Read a CSV with named columns."""
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def import_module(path):
    """Import a native module by its filename."""
    spec = importlib.util.spec_from_file_location(path.name.split(".")[0], path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def method_metrics(sign, valid, reference, distance, bbd, cutoff):
    """Count returned-sign disagreements and invalid queries at both cutoffs."""
    if not np.isin(sign, [-1, 1]).all():
        raise ValueError("A method returned nonbinary signs")
    away = distance > cutoff * bbd
    old_away = distance > OLD_CUTOFF * bbd
    disagree = sign != reference
    counted = disagree & away
    return {
        "raw_disagreements": int(disagree.sum()),
        "disagreements": int(counted.sum()),
        "old_cutoff_disagreements": int((disagree & old_away).sum()),
        "newly_included_disagreements": int((counted & ~old_away).sum()),
        "valid_disagreements": int((counted & valid).sum()),
        "invalid_disagreements": int((counted & ~valid).sum()),
        "invalid_queries": int((~valid).sum()),
        "invalid_queries_away": int((~valid & away).sum()),
        "false_inside": int((counted & (sign == 1)).sum()),
        "false_outside": int((counted & (sign == -1)).sum()),
    }


def run(args):
    """Run every query afresh, verify historical controls, and retain outputs."""
    if not 0 <= args.relative_cutoff <= OLD_CUTOFF or not np.isfinite(args.relative_cutoff):
        raise ValueError(f"relative-cutoff must be finite and between 0 and {OLD_CUTOFF}")
    if args.output.exists():
        raise FileExistsError(f"Choose a fresh output directory: {args.output}")
    os.environ["IGL_NUM_THREADS"] = "4"
    started = perf_counter()
    reference_path = Path("results/full-4963")
    feature_path = Path("results/libigl-closest-feature-4963")
    warp_path = Path("results/warp-main-full-4963")
    sources = read_rows(reference_path / "per_model.csv")
    reference_config = json.loads((reference_path / "config.json").read_text())
    if reference_config["dataset_revision"] != REVISION:
        raise ValueError("Unexpected reference dataset revision")
    metadata_path, meshes, cache_access = prepare_dataset(DEFAULT_CACHE)
    eligible = {
        row["file_id"]
        for row in read_rows(metadata_path)
        if all(row[key] == "1" for key in ("solid", "vertex_manifold", "edge_manifold"))
    }
    if {row["file_id"] for row in sources} != eligible:
        raise ValueError("Reference does not cover the full eligible mask")
    if any(row["status"] != "ok" for row in sources):
        raise ValueError("Reference has unsuccessful models")
    if args.file_ids:
        requested = set(args.file_ids)
        if not requested <= eligible:
            raise ValueError("Requested model is outside the eligible mask")
        sources = [row for row in sources if row["file_id"] in requested]
    module = import_module(args.module.resolve())
    build_manifest = json.loads((args.module.parent / "build_provenance.json").read_text())
    if file_sha256(args.module) != build_manifest["module_sha256"]:
        raise ValueError("Native module does not match its build manifest")
    warp_source = verify_warp_source()
    init_start = perf_counter()
    wp.init()
    device = wp.get_device("cuda:0")
    initialization_seconds = perf_counter() - init_start
    init_start = perf_counter()
    wp.load_module(module=warp_queries, device=device)
    wp.synchronize_device(device)
    warp_module_load_seconds = perf_counter() - init_start
    args.output.mkdir(parents=True)
    (args.output / "points").mkdir()
    shutil.copyfile(reference_path / "selected_metadata.csv", args.output / "selected_metadata.csv")
    config = {
        "started_at_utc": datetime.now(UTC).isoformat(),
        "dataset_revision": REVISION,
        "models": len(sources),
        "full_mask": len(sources) == len(eligible),
        "selected_ids": [row["file_id"] for row in sources],
        "filter": reference_config["filter"],
        "points_per_model": reference_config["points_per_model"],
        "seed": reference_config["seed"],
        "relative_cutoff": args.relative_cutoff,
        "old_relative_cutoff": OLD_CUTOFF,
        "mask": "saved original double d > relative_cutoff * original bbd; same for every method",
        "reference_sign": "Fresh libigl PR fast winding on original double inputs, rounded with numpy.rint; verified against cached signs. Internal winding arithmetic is float.",
        "coordinates": "Original scale; original vertices and points rounded once to float32 for libigl float and Warp",
        "reference": str(reference_path),
        "feature_control": str(feature_path),
        "warp_control": str(warp_path),
        "reference_config_sha256": file_sha256(reference_path / "config.json"),
        "reference_per_model_sha256": file_sha256(reference_path / "per_model.csv"),
        "module": str(args.module.resolve()),
        "build": build_manifest,
        "warp": warp_source,
        "device": str(device),
        "device_name": device.name,
        "warp_settings": {
            "epsilon": 0.001,
            "accuracy": 2.0,
            "threshold": 0.5,
            "parity_rays": 1,
            "parity_perturbation": 0.1,
            "support_winding_number": True,
            "max_distance": "2 * bbd",
        },
        "timing": "Fresh full batch per model, no repetitions. Libigl includes tree/normal setup; Warp kernel times exclude separately measured build/transfers. Module load separate.",
        "warp_initialization_seconds": initialization_seconds,
        "warp_module_load_seconds": warp_module_load_seconds,
        "environment": environment_info(),
        "source_sha256": {
            name: file_sha256(Path(name))
            for name in ("compare_methods.py", "warp_queries.py", "warp_benchmark.py")
        },
        "failure_policy": "Stop on controls/query failure; retain and separately count known nonfinite libigl normals",
    }
    write_json(args.output / "config.json", config)
    write_json(args.output / "cache_access.json", cache_access)
    rows, totals = [], {method: Counter() for method in METHODS}
    checkpoint_seconds = 0.0
    for index, source in enumerate(sources):
        model_start = perf_counter()
        fid = source["file_id"]
        row = {
            "file_id": fid,
            "status": "ok",
            "bbd": float(source["bbd"]),
            "num_vertices": int(source["num_vertices"]),
            "num_faces": int(source["num_faces"]),
            "num_connected_components": int(source["num_connected_components"]),
            "mesh_sha256": source["mesh_sha256"],
            "points_file": f"points/{fid}.npz",
        }
        try:
            t = perf_counter()
            mesh_path = meshes / f"{fid}.npz"
            if file_sha256(mesh_path) != row["mesh_sha256"]:
                raise ValueError("Mesh hash differs from the saved reference")
            vertices, faces = load_mesh(mesh_path)
            row["mesh_load_and_hash_seconds"] = perf_counter() - t
            t = perf_counter()
            ref_path = reference_path / source["points_file"]
            row["reference_points_sha256"] = file_sha256(ref_path)
            with np.load(ref_path, allow_pickle=False) as data:
                reference = {
                    key: data[key]
                    for key in (
                        "points",
                        "d",
                        "sign_w",
                        "winding",
                        "closest_face_index",
                        "closest_point",
                    )
                }
            with np.load(feature_path / row["points_file"], allow_pickle=False) as data:
                control = dict(data)
            with np.load(warp_path / row["points_file"], allow_pickle=False) as data:
                warp_control = dict(data)
            row["reference_and_control_load_seconds"] = perf_counter() - t
            t = perf_counter()
            points, _, _, bbd = sample_points(
                vertices, config["points_per_model"], config["seed"], fid
            )
            np.testing.assert_array_equal(points, reference["points"])
            np.testing.assert_equal(bbd, row["bbd"])
            row["seed_verification_seconds"] = perf_counter() - t
            row["num_points"] = len(points)
            row["num_away"] = int((reference["d"] > args.relative_cutoff * bbd).sum())
            row["num_away_old"] = int((reference["d"] > OLD_CUTOFF * bbd).sum())
            saved = {}
            for method, dtype in (("igl_float", np.float32), ("igl_double", np.float64)):
                t = perf_counter()
                v, p = vertices.astype(dtype, copy=False), points.astype(dtype, copy=False)
                row[f"{method}_conversion_seconds"] = perf_counter() - t
                t = perf_counter()
                distance, nearest, closest, normal = module.signed_distance(
                    p, v, faces, module.SIGNED_DISTANCE_TYPE_PSEUDONORMAL
                )
                row[f"{method}_query_seconds"] = perf_counter() - t
                if any(array.dtype != dtype for array in (distance, closest, normal)):
                    raise TypeError("libigl returned the wrong floating-point precision")
                sign = np.where(distance < 0, 1, -1).astype(np.int8)
                valid = (
                    np.isfinite(distance)
                    & np.isfinite(closest).all(axis=1)
                    & np.isfinite(normal).all(axis=1)
                )
                prefix = np.dtype(dtype).name
                np.testing.assert_array_equal(sign, control[f"{prefix}_sign_p"])
                np.testing.assert_array_equal(valid, control[f"{prefix}_valid"])
                np.testing.assert_array_equal(
                    np.ascontiguousarray(normal).view(np.uint8),
                    np.ascontiguousarray(control[f"{prefix}_normal"]).view(np.uint8),
                )
                if dtype == np.float64:
                    np.testing.assert_array_equal(np.abs(distance), reference["d"])
                    np.testing.assert_array_equal(nearest, reference["closest_face_index"])
                    np.testing.assert_array_equal(closest, reference["closest_point"])
                for name, array in (
                    ("signed_distance", distance),
                    ("sign", sign),
                    ("face", nearest),
                    ("closest", closest),
                    ("normal", normal),
                    ("valid", valid),
                ):
                    saved[f"{method}_{name}"] = array
            t = perf_counter()
            winding = module.fast_winding_number(vertices, faces, points)
            row["igl_winding_query_seconds"] = perf_counter() - t
            if not np.isfinite(winding).all():
                raise ValueError("Nonfinite winding reference")
            rounded = np.rint(winding).astype(np.int64)
            if not np.isin(rounded, [0, 1]).all():
                raise ValueError("Nonbinary rounded winding reference")
            sign_w = 2 * rounded - 1
            np.testing.assert_array_equal(sign_w, reference["sign_w"])
            row["winding_max_difference_vs_cached"] = float(
                np.abs(winding - reference["winding"]).max()
            )
            saved.update(igl_winding=winding, reference_sign=sign_w)
            warp_result, warp_timings = query_warp(vertices, faces, points, device)
            row.update({f"warp_{key}": value for key, value in warp_timings.items()})
            for method in WARP_METHODS:
                for field in ("d", "sign", "face", "closest_point", "valid"):
                    key = f"warp_{field}_{method}"
                    np.testing.assert_array_equal(warp_result[key], warp_control[key])
                    saved[key] = warp_result[key]
                if not np.all(warp_result[f"warp_valid_{method}"] == 1):
                    raise ValueError(f"Warp {method} missed a query")
                if not np.isfinite(warp_result[f"warp_d_{method}"]).all():
                    raise ValueError(f"Warp {method} returned nonfinite distances")
            t = perf_counter()
            for method in METHODS:
                if method.startswith("igl_"):
                    sign, valid = saved[f"{method}_sign"], saved[f"{method}_valid"]
                    distances = np.abs(saved[f"{method}_signed_distance"].astype(np.float64))
                else:
                    key = method.removeprefix("warp_")
                    sign = saved[f"warp_sign_{key}"]
                    valid = saved[f"warp_valid_{key}"] == 1
                    distances = saved[f"warp_d_{key}"].astype(np.float64)
                metrics = method_metrics(
                    sign, valid, sign_w, reference["d"], bbd, args.relative_cutoff
                )
                row.update({f"{method}_{key}": value for key, value in metrics.items()})
                totals[method].update(metrics)
                totals[method]["models_with_disagreements"] += int(metrics["disagreements"] > 0)
                totals[method]["models_with_invalid_queries"] += int(metrics["invalid_queries"] > 0)
                error = np.abs(distances - reference["d"]) / bbd
                if not np.isfinite(error).all():
                    raise ValueError(f"Nonfinite distance error from {method}")
                row[f"{method}_distance_error_max_bbd"] = float(error.max())
                row[f"{method}_distance_error_mean_bbd"] = float(error.mean())
                row[f"{method}_distance_error_p99_bbd"] = float(np.quantile(error, 0.99))
            row["metrics_seconds"] = perf_counter() - t
            if any(row[f"{method}_invalid_queries"] for method in METHODS):
                row["status"] = "nonfinite_normals"
            t = perf_counter()
            np.savez_compressed(args.output / row["points_file"], **saved)
            row["output_seconds"] = perf_counter() - t
        except Exception as error:
            row.update(status="error", error=f"{type(error).__name__}: {error}")
            append_result(args.output / "per_model.jsonl", row)
            raise
        row["model_seconds"] = perf_counter() - model_start
        rows.append(row)
        t = perf_counter()
        append_result(args.output / "per_model.jsonl", row)
        checkpoint_seconds += perf_counter() - t
        if index == 0 or (index + 1) % 100 == 0 or index + 1 == len(sources):
            print(
                json.dumps(
                    {
                        "models": index + 1,
                        "total": len(sources),
                        "seconds": round(perf_counter() - started, 2),
                        "counts": {key: value["disagreements"] for key, value in totals.items()},
                    }
                ),
                flush=True,
            )
    t = perf_counter()
    write_csv(args.output / "per_model.csv", rows)
    export_seconds = perf_counter() - t
    points_count = sum(row["num_points"] for row in rows)
    away_count = sum(row["num_away"] for row in rows)
    for metrics in totals.values():
        metrics["percent_all"] = 100 * metrics["disagreements"] / points_count
        metrics["percent_away"] = (
            100 * metrics["disagreements"] / away_count if away_count else None
        )
    summary = {
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "models": len(rows),
        "failed_models": 0,
        "num_points": points_count,
        "num_away": away_count,
        "num_away_old": sum(row["num_away_old"] for row in rows),
        "methods": totals,
        "all_historical_controls_match": True,
        "winding_max_difference_vs_cached": max(
            row["winding_max_difference_vs_cached"] for row in rows
        ),
        "timings_seconds": {
            key: sum(row.get(key, 0) for row in rows) for key in rows[0] if key.endswith("_seconds")
        },
        "checkpoint_seconds": checkpoint_seconds,
        "csv_export_seconds": export_seconds,
        "wall_seconds": perf_counter() - started,
        "invalid_output_policy": "Returned signs counted even with invalid normals; valid/invalid disagreements reported separately",
    }
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--relative-cutoff", type=float, default=1e-5)
    parser.add_argument("--file-ids", nargs="*")
    run(parser.parse_args())
