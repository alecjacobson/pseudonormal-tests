"""Validate frozen float/double tolerances against cached full-set winding results."""

import argparse
import csv
import hashlib
import importlib.util
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import numpy as np

from benchmark import append_result, load_mesh, write_csv
from dataset_cache import DEFAULT_CACHE, REVISION, prepare_dataset, write_json

# Frozen before broader validation; do not tune these using the validation results.
SETTINGS = {"float32": {"area": 1e-4, "epsilon": 0.003}, "float64": {"area": 1e-4, "epsilon": 1e-9}}


def read_rows(path):
    """Read a CSV table without changing its order."""
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def file_hash(path):
    """Hash an input or executable for the provenance record."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def import_module(path):
    """Import the exact cached extension without rebuilding it."""
    spec = importlib.util.spec_from_file_location(path.name.split(".")[0], path.resolve())
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def statistics(sign, valid, reference_sign, away, default_sign, default_valid):
    """Count returned-sign disagreements and invalid outputs separately."""
    counted = (sign != reference_sign) & away
    baseline = (default_sign != reference_sign) & away
    return {
        "points": len(sign),
        "away_points": int(away.sum()),
        "raw_disagreements": int((sign != reference_sign).sum()),
        "counted_disagreements": int(counted.sum()),
        "valid_counted_disagreements": int((counted & valid).sum()),
        "invalid_queries": int((~valid).sum()),
        "invalid_queries_away": int((~valid & away).sum()),
        "new_vs_default": int((counted & ~baseline).sum()),
        "resolved_vs_default": int((~counted & baseline).sum()),
        "new_invalid_queries": int((~valid & default_valid).sum()),
        "resolved_invalid_queries": int((valid & ~default_valid).sum()),
        "affected_models": int(counted.any()),
        "models_with_invalid_queries": int((~valid).any()),
    }


def run(args):
    """Reuse double geometry and evaluate default/tuned classifications in both precisions."""
    started = perf_counter()
    if args.output.exists():
        raise FileExistsError(f"Choose a fresh output directory: {args.output}")
    if args.threads < 1:
        raise ValueError("threads must be positive")
    os.environ["IGL_NUM_THREADS"] = str(args.threads)
    reference_config = json.loads((args.reference / "config.json").read_text())
    if reference_config["dataset_revision"] != REVISION:
        raise ValueError("Dataset revision differs from the reference")
    sources = read_rows(args.reference / "per_model.csv")
    ids = {row["file_id"] for row in sources}
    if len(ids) != len(sources) or any(row["status"] != "ok" for row in sources):
        raise ValueError("Reference must contain unique, successful models")
    initial_ids = {row["file_id"] for row in read_rows(args.initial / "per_model.csv")}
    if not initial_ids <= ids:
        raise ValueError("Reference must include the original tuning sample")
    runtime_config = json.loads((args.runtime_study / "config.json").read_text())
    precision_config = json.loads((args.precision_study / "config.json").read_text())
    for module, provenance in [
        (args.runtime_module, runtime_config),
        (args.precision_module, precision_config),
    ]:
        if file_hash(module) != provenance["module_sha256"]:
            raise ValueError(f"Extension differs from its recorded source provenance: {module}")
    runtime, precision = import_module(args.runtime_module), import_module(args.precision_module)
    metadata_path, meshes, cache_access = prepare_dataset(args.cache_dir)
    eligible = {
        row["file_id"]
        for row in read_rows(metadata_path)
        if all(row[key] == "1" for key in ("solid", "vertex_manifold", "edge_manifold"))
    }
    if not ids <= eligible:
        raise ValueError("Reference includes ineligible models")
    args.output.mkdir(parents=True)
    (args.output / "points").mkdir()
    config = {
        "started_at_utc": datetime.now(UTC).isoformat(),
        "reference": str(args.reference),
        "reference_config": reference_config,
        "initial": str(args.initial),
        "initial_ids": sorted(initial_ids, key=int),
        "models": len(sources),
        "eligible_models": len(eligible),
        "full_mask": ids == eligible,
        "settings": {
            name: {key: np.dtype(name).type(value).item() for key, value in setting.items()}
            for name, setting in SETTINGS.items()
        },
        "default_epsilon": {name: np.dtype(name).type(1e-12).item() for name in SETTINGS},
        "runtime_module": str(args.runtime_module.resolve()),
        "runtime_provenance": runtime_config,
        "precision_module": str(args.precision_module.resolve()),
        "precision_provenance": precision_config,
        "runner_sha256": file_hash(Path(__file__)),
        "threads": args.threads,
        "reference_csv_sha256": file_hash(args.reference / "per_model.csv"),
        "coordinates": "Original scales and positions; float32 rounds original vertices and points once",
        "reference_sign": "Saved double-input libigl fast winding result",
        "mask": "Saved original double d > 0.001 * original bbd",
        "primary_denominator": "All sample points; away-point denominator also reported",
        "validation_split": "Original 100 models versus remaining models not used in the tolerance sweeps; full baseline results were previously observed",
        "queries": "Float: one full default signed-distance call, then runtime default/tuned classifier calls. Double: cached default geometry, then runtime default/tuned classifier calls. Normals precomputed once per runtime mesh.",
        "failure_policy": "Stop on reference or baseline-reproduction failures; retain nonfinite output normals and report them separately",
    }
    write_json(args.output / "config.json", config)
    write_json(args.output / "cache_access.json", cache_access)
    totals = {split: {} for split in ["all", "original_100", "additional_models"]}
    audits = {
        name: {
            "models": 0,
            "points": 0,
            "sign_changes": 0,
            "normal_differing_bytes": 0,
            "validity_changes": 0,
        }
        for name in SETTINGS
    }
    model_rows, timing_rows = [], []
    for index, source in enumerate(sources):
        model_started = perf_counter()
        file_id = source["file_id"]
        split = "original_100" if file_id in initial_ids else "additional_models"
        timing = {"file_id": file_id}
        mesh_path = meshes / f"{file_id}.npz"
        if file_hash(mesh_path) != source["mesh_sha256"]:
            raise ValueError(f"Mesh changed: {file_id}")
        vertices, faces = load_mesh(mesh_path)
        with np.load(args.reference / source["points_file"], allow_pickle=False) as archive:
            reference = dict(archive)
        away, reference_sign = reference["away_from_surface"], reference["sign_w"]
        np.testing.assert_array_equal(away, reference["d"] > 0.001 * float(source["bbd"]))
        np.testing.assert_equal(
            [len(away), int(away.sum())], [int(source["num_points"]), int(source["num_away"])]
        )
        timing["mesh_and_reference_load_seconds"] = perf_counter() - model_started
        saved, current_rows = {}, []
        for name in SETTINGS:
            dtype = np.dtype(name).type
            setting = config["settings"][name]
            converted_at = perf_counter()
            v, points = (
                vertices.astype(dtype, copy=False),
                reference["points"].astype(dtype, copy=False),
            )
            timing[f"{name}_conversion_seconds"] = perf_counter() - converted_at
            query_at = perf_counter()
            if name == "float32":
                signed_distance, indices, closest, baseline_normals = precision.signed_distance(
                    points, v, faces, precision.SIGNED_DISTANCE_TYPE_PSEUDONORMAL
                )
                timing["float32_full_default_query_seconds"] = perf_counter() - query_at
                for key, array in [
                    ("signed_distance", signed_distance),
                    ("closest_point", closest),
                    ("pseudonormal", baseline_normals),
                ]:
                    if array.dtype != dtype:
                        raise TypeError(f"{file_id} {key}: expected {name}, got {array.dtype}")
                    saved[f"float32_{key}"] = array
                saved["float32_closest_face_index"] = indices
            else:
                signed_distance, indices = (
                    reference["signed_distance"],
                    reference["closest_face_index"],
                )
                closest, baseline_normals = reference["closest_point"], reference["pseudonormal"]
            distance = np.abs(signed_distance)
            baseline_sign = np.where(signed_distance < 0, 1, -1).astype(np.int8)
            geometry_valid = np.isfinite(signed_distance) & np.isfinite(closest).all(axis=1)
            baseline_valid = geometry_valid & np.isfinite(baseline_normals).all(axis=1)
            prepare_at = perf_counter()
            prepared_class = (
                runtime.PreparedMesh if name == "float32" else runtime.PreparedMeshDouble
            )
            prepared = prepared_class(v, faces)
            timing[f"{name}_normal_precompute_seconds"] = perf_counter() - prepare_at
            for label, epsilon in [
                ("default", config["default_epsilon"][name]),
                ("tuned", setting["epsilon"]),
            ]:
                query_at = perf_counter()
                native_sign, normals = prepared.classify(
                    points, indices, closest, setting["area"], epsilon
                )
                seconds = perf_counter() - query_at
                sign = np.where(native_sign * distance < 0, 1, -1).astype(np.int8)
                valid = geometry_valid & np.isfinite(normals).all(axis=1)
                if label == "default":
                    changes = {
                        "sign_changes": int(np.count_nonzero(sign != baseline_sign)),
                        "normal_differing_bytes": int(
                            np.count_nonzero(
                                np.ascontiguousarray(normals).view(np.uint8)
                                != np.ascontiguousarray(baseline_normals).view(np.uint8)
                            )
                        ),
                        "validity_changes": int(np.count_nonzero(valid != baseline_valid)),
                    }
                    if any(changes.values()):
                        raise RuntimeError(
                            f"{file_id} {name}: default reproduction failed: {changes}"
                        )
                    audits[name]["models"] += 1
                    audits[name]["points"] += len(points)
                variant = f"{name}_{label}"
                saved[f"{variant}_sign_p"], saved[f"{variant}_valid_query"] = sign, valid
                if label == "tuned":
                    saved[f"{variant}_pseudonormal"] = normals
                metrics = statistics(
                    sign, valid, reference_sign, away, baseline_sign, baseline_valid
                )
                row = {
                    "file_id": file_id,
                    "split": split,
                    "variant": variant,
                    "minimum_double_area": setting["area"],
                    "epsilon": epsilon,
                    "status": "nonfinite_output" if metrics["invalid_queries"] else "ok",
                    "mesh_sha256": source["mesh_sha256"],
                    "bbd": float(source["bbd"]),
                    **metrics,
                    "percent_all": 100 * metrics["counted_disagreements"] / len(points),
                    "percent_away": 100 * metrics["counted_disagreements"] / int(away.sum()),
                    "classifier_seconds": seconds,
                }
                current_rows.append(row)
                for fold in ["all", split]:
                    aggregate = totals[fold].setdefault(
                        variant, {"models": 0, **dict.fromkeys(metrics, 0)}
                    )
                    aggregate["models"] += 1
                    for key, value in metrics.items():
                        aggregate[key] += value
                timing[f"{variant}_classifier_seconds"] = seconds
            del prepared
            error = np.abs(distance.astype(np.float64) - reference["d"]) / float(source["bbd"])
            finite = error[np.isfinite(error)]
            timing[f"{name}_distance_error_max_bbd"] = float(finite.max()) if len(finite) else None
            timing[f"{name}_distance_error_p99_bbd"] = (
                float(np.quantile(finite, 0.99)) if len(finite) else None
            )
            timing[f"{name}_distance_error_nonfinite"] = int((~np.isfinite(error)).sum())
        output_at = perf_counter()
        destination = args.output / "points" / f"{file_id}.npz"
        with destination.with_suffix(".npz.tmp").open("wb") as stream:
            np.savez_compressed(stream, **saved)
        destination.with_suffix(".npz.tmp").replace(destination)
        timing["output_seconds"] = perf_counter() - output_at
        checkpoint_at = perf_counter()
        for row in current_rows:
            append_result(args.output / "per_model.jsonl", row)
        timing["checkpoint_seconds"] = perf_counter() - checkpoint_at
        timing["model_total_seconds"] = perf_counter() - model_started
        append_result(args.output / "timings.jsonl", timing)
        model_rows.extend(current_rows)
        timing_rows.append(timing)
        if (index + 1) % 100 == 0 or index + 1 == len(sources):
            counts = {key: value["counted_disagreements"] for key, value in totals["all"].items()}
            print(
                f"Validated {index + 1}/{len(sources)} models in {perf_counter() - started:.1f}s: {counts}",
                flush=True,
            )
    for variants in totals.values():
        for row in variants.values():
            row["percent_all"] = 100 * row["counted_disagreements"] / row["points"]
            row["percent_away"] = 100 * row["counted_disagreements"] / row["away_points"]
    write_csv(args.output / "per_model.csv", model_rows)
    write_csv(args.output / "timings.csv", timing_rows)
    summary = {
        "models": len(sources),
        "full_mask": ids == eligible,
        "totals": totals,
        "baseline_reproduction": audits,
        "timings_seconds": {
            key: sum(row[key] for row in timing_rows)
            for key in timing_rows[0]
            if key.endswith("seconds")
        },
        "wall_seconds": perf_counter() - started,
    }
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-module", type=Path, required=True)
    parser.add_argument("--precision-module", type=Path, required=True)
    parser.add_argument(
        "--runtime-study", type=Path, default=Path("results/libigl-double-tolerances-100")
    )
    parser.add_argument(
        "--precision-study", type=Path, default=Path("results/libigl-scalar-single-100")
    )
    parser.add_argument("--reference", type=Path, default=Path("results/full-4963"))
    parser.add_argument("--initial", type=Path, default=Path("results/initial-100"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--threads", type=int, default=4)
    run(parser.parse_args())
