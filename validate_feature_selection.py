"""Validate carried closest features against every cached float/double query."""

import argparse
import json
import os
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import numpy as np

from benchmark import append_result, load_mesh, write_csv
from dataset_cache import DEFAULT_CACHE, REVISION, prepare_dataset, write_json
from probe_normalization import same_bytes
from validate_tolerances import file_hash, import_module, read_rows, statistics


def run(args):
    """Replay cached nearest triangles and audit the full pipeline on selected models."""
    if args.output.exists():
        raise FileExistsError(args.output)
    os.environ["IGL_NUM_THREADS"] = "4"
    started = perf_counter()
    reference_path = Path("results/full-4963")
    baseline_path = Path("results/libigl-tolerance-validation-4963")
    sources = read_rows(reference_path / "per_model.csv")
    initial = {r["file_id"] for r in read_rows(Path("results/initial-100/per_model.csv"))}
    affected = {
        r["file_id"]
        for r in read_rows(baseline_path / "per_model.csv")
        if r["variant"] == "float64_tuned" and int(r["counted_disagreements"])
    }
    if args.file_ids:
        sources = [r for r in sources if r["file_id"] in args.file_ids]
        if {r["file_id"] for r in sources} != set(args.file_ids):
            raise ValueError("Unknown requested file ID")
    module = import_module(args.module)
    production = import_module(args.production_module)
    original = import_module(args.original_module)
    metadata, meshes, cache_access = prepare_dataset(DEFAULT_CACHE)
    eligible = {
        r["file_id"]
        for r in read_rows(metadata)
        if all(r[key] == "1" for key in ("solid", "vertex_manifold", "edge_manifold"))
    }
    original_config = json.loads(Path("results/libigl-tuned-defaults-104/config.json").read_text())
    if file_hash(args.original_module) != original_config["module_sha256"]:
        raise ValueError("Original pipeline module differs from its provenance")
    args.output.mkdir(parents=True)
    (args.output / "points").mkdir()
    config = {
        "started_at_utc": datetime.now(UTC).isoformat(),
        "dataset_revision": REVISION,
        "reference": str(reference_path),
        "baseline": str(baseline_path),
        "selected_ids": [r["file_id"] for r in sources],
        "full_mask": {r["file_id"] for r in sources} == eligible,
        "initial_ids": sorted(initial, key=int),
        "end_to_end_ids": sorted((initial | affected) & {r["file_id"] for r in sources}, key=int),
        "runner_sha256": file_hash(Path(__file__)),
        "module": str(args.module.resolve()),
        "module_sha256": file_hash(args.module),
        "production_module": str(args.production_module.resolve()),
        "production_module_sha256": file_hash(args.production_module),
        "original_module": str(args.original_module.resolve()),
        "original_module_sha256": file_hash(args.original_module),
        "source_hashes": {
            name: file_hash(args.libigl / "include/igl" / name)
            for name in (
                "AABB.h",
                "AABB.cpp",
                "pseudonormal_test.h",
                "pseudonormal_test.cpp",
                "signed_distance.cpp",
                "point_simplex_squared_distance.cpp",
            )
        },
        "coordinates": "Original scales; float rounds original vertices and queries once",
        "reference_sign": "Saved double fast winding sign",
        "mask": "Saved original double d > 0.001 * bbd, identical for both precisions",
        "selection": "Use exact zero support of nearest routine's barycentric coordinates",
        "scope": "Replay the cached closest triangle for every point; end-to-end old/new signed_distance on the original100 plus all17 double-failure models",
        "failure_policy": "Stop on baseline, geometry, or integration mismatches; count nonfinite outputs separately",
    }
    write_json(args.output / "config.json", config)
    write_json(args.output / "cache_access.json", cache_access)
    totals = {
        split: {name: Counter() for name in ("float32", "float64")}
        for split in ("all", "original_100", "additional_models")
    }
    rows, timings = [], []
    for index, source in enumerate(sources):
        model_start = perf_counter()
        fid = source["file_id"]
        split = "original_100" if fid in initial else "additional_models"
        mesh_path = meshes / f"{fid}.npz"
        np.testing.assert_equal(file_hash(mesh_path), source["mesh_sha256"])
        vertices, faces = load_mesh(mesh_path)
        with np.load(reference_path / "points" / f"{fid}.npz") as data:
            reference = {
                k: data[k]
                for k in (
                    "points",
                    "d",
                    "sign_w",
                    "away_from_surface",
                    "closest_face_index",
                    "closest_point",
                )
            }
        with np.load(baseline_path / "points" / f"{fid}.npz") as data:
            baseline = dict(data)
        away = reference["away_from_surface"]
        np.testing.assert_array_equal(away, reference["d"] > 0.001 * float(source["bbd"]))
        timing = {"file_id": fid, "load_seconds": perf_counter() - model_start}
        saved, model_rows = {}, []
        for name, cls in (
            ("float32", module.PreparedMeshFloat),
            ("float64", module.PreparedMeshDouble),
        ):
            v = vertices.astype(name, copy=False)
            points = reference["points"].astype(name, copy=False)
            if name == "float32":
                indices = baseline["float32_closest_face_index"]
                closest = baseline["float32_closest_point"]
                distance = np.abs(baseline["float32_signed_distance"])
            else:
                indices, closest, distance = (
                    reference[k] for k in ("closest_face_index", "closest_point", "d")
                )
            before_sign = baseline[f"{name}_tuned_sign_p"]
            before_valid = baseline[f"{name}_tuned_valid_query"]
            t = perf_counter()
            prepared = cls(v, faces)
            timing[f"{name}_normal_setup_seconds"] = perf_counter() - t
            t = perf_counter()
            control, control_normals = prepared.classify(
                points, indices, closest, 1e-4, 0.003 if name == "float32" else 1e-9
            )
            timing[f"{name}_old_classifier_seconds"] = perf_counter() - t
            np.testing.assert_array_equal(np.where(control * distance < 0, 1, -1), before_sign)
            same_bytes(control_normals, baseline[f"{name}_tuned_pseudonormal"])
            t = perf_counter()
            native_sign, normals, new_closest, barycentric, square_distance = prepared.replay(
                points, indices
            )
            timing[f"{name}_nearest_triangle_and_classifier_seconds"] = perf_counter() - t
            # Replaying the winning primitive must retain its exact closest geometry.
            same_bytes(new_closest, closest)
            np.testing.assert_array_equal(np.sqrt(square_distance), distance)
            sign = np.where(native_sign * distance < 0, 1, -1).astype(np.int8)
            valid = np.isfinite(native_sign) & np.isfinite(normals).all(axis=1)
            metrics = statistics(sign, valid, reference["sign_w"], away, before_sign, before_valid)
            metrics.update(
                {
                    "baseline_disagreements": int(
                        ((before_sign != reference["sign_w"]) & away).sum()
                    ),
                    "sign_changes": int((sign != before_sign).sum()),
                    "invalid_barycentrics": int(
                        (
                            ~np.isfinite(barycentric).all(axis=1)
                            | (barycentric < 0).any(axis=1)
                            | (barycentric.sum(axis=1) <= 0)
                        ).sum()
                    ),
                    "end_to_end_models": 0,
                }
            )
            if fid in config["end_to_end_ids"]:
                t = perf_counter()
                old_distance, old_indices, old_closest, old_normals = original.signed_distance(
                    points, v, faces, original.SIGNED_DISTANCE_TYPE_PSEUDONORMAL
                )
                timing[f"{name}_old_full_query_seconds"] = perf_counter() - t
                np.testing.assert_array_equal(old_indices, indices)
                same_bytes(old_closest, closest)
                same_bytes(old_normals, control_normals)
                np.testing.assert_array_equal(np.abs(old_distance), distance)
                t = perf_counter()
                new_distance, new_indices, pipeline_closest, pipeline_normals = (
                    production.signed_distance(
                        points, v, faces, production.SIGNED_DISTANCE_TYPE_PSEUDONORMAL
                    )
                )
                timing[f"{name}_new_full_query_seconds"] = perf_counter() - t
                np.testing.assert_array_equal(new_indices, indices)
                same_bytes(pipeline_closest, closest)
                same_bytes(pipeline_normals, normals)
                np.testing.assert_array_equal(new_distance, native_sign * distance)
                metrics["end_to_end_models"] = 1
            saved[f"{name}_sign_p"] = sign
            saved[f"{name}_valid"] = valid
            saved[f"{name}_normal"] = normals
            saved[f"{name}_barycentric"] = barycentric
            row = {
                "file_id": fid,
                "split": split,
                "precision": name,
                "mesh_sha256": source["mesh_sha256"],
                **metrics,
            }
            model_rows.append(row)
            for fold in ("all", split):
                totals[fold][name].update(metrics)
                totals[fold][name]["models"] += 1
            del prepared
        t = perf_counter()
        np.savez_compressed(args.output / "points" / f"{fid}.npz", **saved)
        timing["output_seconds"] = perf_counter() - t
        timing["total_seconds"] = perf_counter() - model_start
        timings.append(timing)
        rows.extend(model_rows)
        for row in model_rows:
            append_result(args.output / "per_model.jsonl", row)
        append_result(args.output / "timings.jsonl", timing)
        if (index + 1) % 100 == 0 or index + 1 == len(sources):
            print(
                json.dumps(
                    {
                        "models": index + 1,
                        "total": len(sources),
                        "seconds": perf_counter() - started,
                        "counts": {name: dict(value) for name, value in totals["all"].items()},
                    }
                ),
                flush=True,
            )
    write_csv(args.output / "per_model.csv", rows)
    write_csv(args.output / "timings.csv", timings)
    summary = {
        "models": len(sources),
        "totals": totals,
        "wall_seconds": perf_counter() - started,
        "baseline_sign_and_normal_bytes_match": True,
        "all_cached_closest_points_and_distances_match": True,
        "end_to_end_matches_replay": True,
        "timings_seconds": {
            key: sum(r.get(key, 0) for r in timings)
            for key in {k for r in timings for k in r if k.endswith("seconds")}
        },
    }
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", type=Path, required=True)
    parser.add_argument("--production-module", type=Path, required=True)
    parser.add_argument("--original-module", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--libigl", type=Path, default=Path("/home/horde/projects/libigl"))
    parser.add_argument("--file-ids", nargs="*")
    run(parser.parse_args())
