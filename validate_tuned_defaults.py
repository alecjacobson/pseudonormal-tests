"""Check production defaults against the saved runtime-tolerance experiment."""

import argparse
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from benchmark import load_mesh, write_csv
from dataset_cache import DEFAULT_CACHE, prepare_dataset, write_json
from precision_benchmark import source_state
from probe_normalization import same_bytes
from validate_tolerances import file_hash, import_module, read_rows


def run(args):
    """Require pointwise reproduction on the original sample and every invalid model."""
    if args.output.exists():
        raise FileExistsError(f"Choose a fresh output directory: {args.output}")
    os.environ["IGL_NUM_THREADS"] = "4"
    started = perf_counter()
    module = import_module(args.module)
    _, meshes, cache_access = prepare_dataset(DEFAULT_CACHE)
    sources = {r["file_id"]: r for r in read_rows(args.reference / "per_model.csv")}
    initial_ids = {r["file_id"] for r in read_rows(args.initial / "per_model.csv")}
    invalid_ids = {
        r["file_id"]
        for r in read_rows(args.validation / "per_model.csv")
        if int(r["invalid_queries"])
    }
    selected = sorted(initial_ids | invalid_ids, key=int)
    args.output.mkdir(parents=True)
    write_json(
        args.output / "config.json",
        {
            "libigl": source_state(args.libigl_source),
            "bindings": source_state(args.bindings_source),
            "module": str(args.module.resolve()),
            "module_sha256": file_hash(args.module),
            "runner_sha256": file_hash(Path(__file__)),
            "reference": str(args.reference),
            "validation": str(args.validation),
            "initial_ids": sorted(initial_ids, key=int),
            "invalid_ids": sorted(invalid_ids, key=int),
            "selected_ids": selected,
            "selection": "Original100 plus every model with invalid queries in full tolerance validation",
            "comparison": "Production signed_distance defaults versus saved runtime tuned normals/signs, with identical nearest geometry",
        },
    )
    write_json(args.output / "cache_access.json", cache_access)
    rows = []
    for index, file_id in enumerate(selected):
        mesh_path = meshes / f"{file_id}.npz"
        if file_hash(mesh_path) != sources[file_id]["mesh_sha256"]:
            raise ValueError(f"Mesh changed: {file_id}")
        vertices, faces = load_mesh(mesh_path)
        with np.load(args.reference / "points" / f"{file_id}.npz") as archive:
            reference = dict(archive)
        with np.load(args.validation / "points" / f"{file_id}.npz") as archive:
            saved = dict(archive)
        for precision in ("float32", "float64"):
            dtype = np.dtype(precision).type
            points, v = reference["points"].astype(dtype), vertices.astype(dtype)
            query_start = perf_counter()
            signed, indices, closest, normals = module.signed_distance(
                points, v, faces, module.SIGNED_DISTANCE_TYPE_PSEUDONORMAL
            )
            seconds = perf_counter() - query_start
            for output in (signed, closest, normals):
                np.testing.assert_equal(output.dtype, dtype)
            if precision == "float32":
                previous_signed = saved["float32_signed_distance"]
                previous_closest = saved["float32_closest_point"]
                previous_indices = saved["float32_closest_face_index"]
            else:
                previous_signed, previous_closest, previous_indices = (
                    reference["signed_distance"],
                    reference["closest_point"],
                    reference["closest_face_index"],
                )
            same_bytes(np.abs(signed), np.abs(previous_signed))
            same_bytes(closest, previous_closest)
            np.testing.assert_array_equal(indices, previous_indices)
            same_bytes(normals, saved[f"{precision}_tuned_pseudonormal"])
            sign = np.where(signed < 0, 1, -1).astype(np.int8)
            valid = (
                np.isfinite(signed)
                & np.isfinite(closest).all(axis=1)
                & np.isfinite(normals).all(axis=1)
            )
            np.testing.assert_array_equal(sign, saved[f"{precision}_tuned_sign_p"])
            np.testing.assert_array_equal(valid, saved[f"{precision}_tuned_valid_query"])
            wrong = (sign != reference["sign_w"]) & reference["away_from_surface"]
            rows.append(
                {
                    "file_id": file_id,
                    "precision": precision,
                    "original_100": file_id in initial_ids,
                    "points": len(points),
                    "disagreements": int(wrong.sum()),
                    "finite_disagreements": int((wrong & valid).sum()),
                    "invalid_queries": int((~valid).sum()),
                    "query_seconds": seconds,
                }
            )
        if (index + 1) % 20 == 0:
            print(
                f"{index + 1}/{len(selected)} elapsed={perf_counter() - started:.2f}s", flush=True
            )
    write_csv(args.output / "per_model.csv", rows)
    summary = {
        "status": "passed",
        "models": len(selected),
        "wall_seconds": perf_counter() - started,
        "pointwise_normal_bytes_sign_validity_and_nearest_geometry_match": True,
        "groups": {},
    }
    for group, ids in (("all", set(selected)), ("original_100", initial_ids)):
        summary["groups"][group] = {}
        for precision in ("float32", "float64"):
            matching = [r for r in rows if r["file_id"] in ids and r["precision"] == precision]
            summary["groups"][group][precision] = {
                key: sum(r[key] for r in matching)
                for key in (
                    "points",
                    "disagreements",
                    "finite_disagreements",
                    "invalid_queries",
                    "query_seconds",
                )
            }
    write_json(args.output / "summary.json", summary)
    print(summary, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", type=Path, required=True)
    parser.add_argument("--libigl-source", type=Path, default=Path("../libigl"))
    parser.add_argument("--bindings-source", type=Path, default=Path("../libigl-python-bindings"))
    parser.add_argument("--reference", type=Path, default=Path("results/full-4963"))
    parser.add_argument(
        "--validation", type=Path, default=Path("results/libigl-tolerance-validation-4963")
    )
    parser.add_argument("--initial", type=Path, default=Path("results/initial-100"))
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
