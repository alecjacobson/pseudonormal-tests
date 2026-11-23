"""Replay cached samples through a specific libigl signed-distance extension."""

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import numpy as np

from benchmark import append_result, load_mesh, write_csv
from dataset_cache import DEFAULT_CACHE, REVISION, prepare_dataset, write_json

VARIANTS = {
    "original_double": (np.float64, False),
    "patched_double": (np.float64, False),
    "patched_single": (np.float32, False),
    "quantized_double": (np.float64, True),
}


def source_state(path):
    """Record the exact source state, including an uncommitted patch if present."""

    def git(*arguments):
        return subprocess.check_output(
            ["git", "-C", str(path), *arguments], text=True, timeout=30
        ).strip()

    return {"commit": git("rev-parse", "HEAD"), "diff": git("diff", "HEAD")}


def run(args):
    """Query one precision variant without changing saved inputs or the reference."""
    os.environ["IGL_NUM_THREADS"] = str(args.threads)
    if args.output.exists():
        raise FileExistsError(f"Choose a fresh output directory: {args.output}")
    if args.threads <= 0:
        raise ValueError("threads must be positive")
    args.output.mkdir(parents=True)
    (args.output / "points").mkdir()
    started = perf_counter()
    extension = args.module.resolve()
    module_name = extension.name.split(".", 1)[0]
    spec = importlib.util.spec_from_file_location(module_name, extension)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module_load_seconds = perf_counter() - started
    _, meshes, cache_access = prepare_dataset(args.cache_dir)
    write_json(args.output / "cache_access.json", cache_access)
    metadata_started = perf_counter()
    with (args.reference / "per_model.csv").open(newline="") as stream:
        source_rows = list(csv.DictReader(stream))
    reference_config = json.loads((args.reference / "config.json").read_text())
    if reference_config["dataset_revision"] != REVISION:
        raise ValueError("The reference and dataset revisions must match")
    if any(row["status"] != "ok" for row in source_rows):
        raise ValueError("The reference contains failed models")
    metadata_seconds = perf_counter() - metadata_started
    dtype, quantize = VARIANTS[args.variant]
    write_json(
        args.output / "config.json",
        {
            "started_at_utc": datetime.now(UTC).isoformat(),
            "variant": args.variant,
            "dtype": np.dtype(dtype).name,
            "quantize_inputs_to_float32_first": quantize,
            "reference": str(args.reference),
            "reference_config": reference_config,
            "module": str(extension),
            "module_sha256": hashlib.sha256(extension.read_bytes()).hexdigest(),
            "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "cmake_cache_settings": [
                line
                for line in (extension.parent / "CMakeCache.txt").read_text().splitlines()
                if line.startswith(
                    (
                        "CMAKE_BUILD_TYPE:",
                        "CMAKE_CXX_COMPILER:",
                        "CMAKE_CXX_FLAGS",
                        "FETCHCONTENT_SOURCE_DIR_",
                        "PRECISION_MODULE:",
                    )
                )
            ],
            "libigl": source_state(args.libigl_source),
            "bindings": source_state(args.bindings_source),
            "threads": args.threads,
            "module_load_seconds": module_load_seconds,
            "metadata_load_seconds": metadata_seconds,
            "python": os.sys.version,
            "numpy": np.__version__,
            "compiler": subprocess.check_output(["c++", "--version"], text=True).splitlines()[0],
            "query_timing": "One full batch per model, including internal AABB and normal setup",
            "mask": "saved original double d > 0.001 * original bbd",
        },
    )
    rows = []
    for index, source in enumerate(source_rows):
        file_id = source["file_id"]
        row = {
            "file_id": file_id,
            "variant": args.variant,
            "status": "ok",
            "error": "",
            "num_points": int(source["num_points"]),
            "num_away": int(source["num_away"]),
            "bbd": float(source["bbd"]),
            "mesh_sha256": source["mesh_sha256"],
        }
        model_started = perf_counter()
        try:
            path = meshes / f"{file_id}.npz"
            mesh_started = perf_counter()
            actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual_hash != source["mesh_sha256"]:
                raise ValueError("Cached mesh hash differs from reference")
            vertices, faces = load_mesh(path)
            row["mesh_load_and_hash_seconds"] = perf_counter() - mesh_started
            reference_started = perf_counter()
            with np.load(args.reference / source["points_file"], allow_pickle=False) as saved:
                points = saved["points"]
                reference_distance = saved["d"]
                reference_sign = saved["sign_w"]
                reference_p = saved["sign_p"]
                away = saved["away_from_surface"]
            np.testing.assert_array_equal(away, reference_distance > 0.001 * row["bbd"])
            if len(points) != row["num_points"] or int(away.sum()) != row["num_away"]:
                raise ValueError("Reference point count or mask changed")
            row["reference_load_seconds"] = perf_counter() - reference_started
            conversion_started = perf_counter()
            if quantize:
                vertices = vertices.astype(np.float32)
                points = points.astype(np.float32)
            vertices = vertices.astype(dtype, copy=False)
            points = points.astype(dtype, copy=False)
            row["input_conversion_seconds"] = perf_counter() - conversion_started
            query_started = perf_counter()
            signed_distance, face_index, closest, normal = module.signed_distance(
                points, vertices, faces, module.SIGNED_DISTANCE_TYPE_PSEUDONORMAL
            )
            row["query_seconds"] = perf_counter() - query_started
            for name, array in [
                ("distance", signed_distance),
                ("closest", closest),
                ("normal", normal),
            ]:
                if array.dtype != dtype:
                    raise TypeError(f"{name} returned {array.dtype}, expected {np.dtype(dtype)}")
                row[f"nonfinite_{name}_entries"] = int((~np.isfinite(array)).sum())
            valid = (
                np.isfinite(signed_distance)
                & np.isfinite(closest).all(axis=1)
                & np.isfinite(normal).all(axis=1)
            )
            row["invalid_queries"] = int((~valid).sum())
            row["invalid_queries_away"] = int((~valid & away).sum())
            if not valid.all():
                row["status"] = "nonfinite_output"
                row["error"] = (
                    "Nonfinite query outputs retained; returned-sign metrics are diagnostic"
                )
            signs = np.where(signed_distance < 0, 1, -1).astype(np.int8)
            disagree = signs != reference_sign
            counted = disagree & away
            original_counted = (reference_p != reference_sign) & away
            errors = (
                np.abs(np.abs(signed_distance.astype(np.float64)) - reference_distance) / row["bbd"]
            )
            row.update(
                {
                    "num_disagreements_raw": int(disagree.sum()),
                    "num_valid_counted_disagreements": int((counted & valid).sum()),
                    "num_counted_disagreements": int(counted.sum()),
                    "disagreement_percent_all": 100 * float(counted.mean()),
                    "disagreement_percent_away": 100 * int(counted.sum()) / row["num_away"],
                    "new_disagreements_vs_wheel": int((counted & ~original_counted).sum()),
                    "resolved_disagreements_vs_wheel": int((~counted & original_counted).sum()),
                    "sign_changes_vs_wheel_raw": int((signs != reference_p).sum()),
                    "distance_error_max_bbd": float(errors.max()),
                    "distance_error_p99_bbd": float(np.quantile(errors, 0.99)),
                }
            )
            output_started = perf_counter()
            np.savez_compressed(
                args.output / "points" / f"{file_id}.npz",
                signed_distance=signed_distance,
                closest_face_index=face_index,
                closest_point=closest,
                pseudonormal=normal,
                sign_p=signs,
                valid_query=valid,
            )
            row["output_seconds"] = perf_counter() - output_started
        except Exception as error:
            row["status"] = "error"
            row["error"] = f"{type(error).__name__}: {error}"
        row["total_seconds"] = perf_counter() - model_started
        rows.append(row)
        append_result(args.output / "per_model.jsonl", row)
        if (index + 1) % 20 == 0 or row["status"] == "error":
            print(
                f"{args.variant}: {index + 1}/{len(source_rows)} {file_id} {row['status']}",
                flush=True,
            )
    write_csv(args.output / "per_model.csv", rows)
    failed = [row for row in rows if row["status"] == "error"]
    count = sum(row.get("num_counted_disagreements", 0) for row in rows)
    points_count = sum(row["num_points"] for row in rows)
    away_count = sum(row["num_away"] for row in rows)
    summary = {
        "variant": args.variant,
        "models": len(rows),
        "failed_models": len(failed),
        "models_with_nonfinite_outputs": sum(row["status"] == "nonfinite_output" for row in rows),
        "invalid_queries": sum(row.get("invalid_queries", 0) for row in rows),
        "invalid_queries_away": sum(row.get("invalid_queries_away", 0) for row in rows),
        "num_valid_counted_disagreements": sum(
            row.get("num_valid_counted_disagreements", 0) for row in rows
        ),
        "metric_caveat": "Counts use returned signs; consult invalid-query counts before interpreting as valid pseudonormal classifications",
        "intended_points": points_count,
        "intended_away_points": away_count,
        "num_counted_disagreements": count if not failed else None,
        "disagreement_percent_all": 100 * count / points_count if not failed else None,
        "disagreement_percent_away": 100 * count / away_count if not failed else None,
        "affected_models": sum(row.get("num_counted_disagreements", 0) > 0 for row in rows),
        "query_seconds": sum(row.get("query_seconds", 0) for row in rows),
        "wall_seconds": perf_counter() - started,
    }
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    if failed:
        raise RuntimeError(f"{len(failed)} failed models; see per_model.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, choices=VARIANTS)
    parser.add_argument("--module", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=Path("results/initial-100"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--libigl-source", type=Path, required=True)
    parser.add_argument("--bindings-source", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    run(parser.parse_args())
