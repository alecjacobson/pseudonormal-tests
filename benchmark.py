"""Compare libigl pseudonormal and fast winding signs on cached Thingi10K meshes."""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import logging
import os
import platform
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import igl
import numpy as np

from dataset_cache import DEFAULT_CACHE, REVISION, prepare_dataset, write_json


def select_models(metadata, count, seed):
    """Sample without replacement from solid, vertex- and edge-manifold meshes."""
    eligible = sorted(
        (
            row
            for row in metadata
            if all(row[key] == "1" for key in ("solid", "vertex_manifold", "edge_manifold"))
        ),
        key=lambda row: int(row["file_id"]),
    )
    if not 0 < count <= len(eligible):
        raise ValueError(f"Model count must be between 1 and {len(eligible)}.")
    indices = np.random.default_rng(seed).choice(len(eligible), count, replace=False)
    return [eligible[index] for index in indices], len(eligible)


def load_mesh(path):
    """Load the dataset geometry without welding, reorientation, or normalization."""
    with np.load(path, allow_pickle=False) as arrays:
        vertices = np.asarray(arrays["vertices"], dtype=np.float64)
        raw_faces = arrays["facets"]
        if not np.issubdtype(raw_faces.dtype, np.integer):
            raise TypeError("Expected integer triangle indices.")
        faces = np.asarray(raw_faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        raise ValueError("Expected a nonempty (n, 3) vertex array.")
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
        raise ValueError("Expected a nonempty (m, 3) triangle array.")
    if not np.isfinite(vertices).all():
        raise ValueError("Vertex coordinates must be finite.")
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError("Triangle indices are out of range.")
    return vertices, faces


def sample_points(vertices, count, seed, file_id):
    """Sample uniformly in the axis-aligned bounding box using a per-model seed."""
    lower = vertices.min(axis=0)
    upper = vertices.max(axis=0)
    if count <= 0 or np.any(upper <= lower):
        raise ValueError("Require a positive sample count and nonzero box extent on every axis.")
    generator = np.random.default_rng(np.random.SeedSequence([seed, int(file_id)]))
    points = generator.uniform(lower, upper, size=(count, 3))
    return points, lower, upper, float(np.linalg.norm(upper - lower))


def query_mesh(vertices, faces, points):
    """Measure full batch calls, including the internal acceleration structures."""
    started = perf_counter()
    signed_distance, face_index, closest_point, pseudonormal = igl.signed_distance(
        points, vertices, faces, igl.SIGNED_DISTANCE_TYPE_PSEUDONORMAL
    )
    pseudonormal_seconds = perf_counter() - started
    started = perf_counter()
    winding = igl.fast_winding_number(vertices, faces, points)
    winding_seconds = perf_counter() - started
    if not np.isfinite(signed_distance).all() or not np.isfinite(winding).all():
        raise ValueError("libigl returned nonfinite signed distances or winding numbers.")
    rounded_winding = np.rint(winding).astype(np.int64)
    arrays = {
        "points": points,
        "d": np.abs(signed_distance),
        "signed_distance": signed_distance,
        # libigl uses negative inside. Surface ties are assigned -1 (outside).
        "sign_p": np.where(signed_distance < 0, 1, -1).astype(np.int8),
        "winding": winding,
        "rounded_winding": rounded_winding,
        # Preserve unexpected rounded values instead of silently clipping them.
        "sign_w": 2 * rounded_winding - 1,
        "closest_face_index": face_index,
        "closest_point": closest_point,
        "pseudonormal": pseudonormal,
    }
    return arrays, {
        "pseudonormal_seconds": pseudonormal_seconds,
        "fast_winding_seconds": winding_seconds,
    }


def disagreement_statistics(arrays, bbd, relative_cutoff):
    """Measure disagreements beyond the strict cutoff using both denominators."""
    away = arrays["d"] > relative_cutoff * bbd
    disagree = arrays["sign_p"] != arrays["sign_w"]
    counted = away & disagree
    count = int(counted.sum())
    away_count = int(away.sum())
    arrays.update(away_from_surface=away, signs_disagree=disagree, counted_disagreement=counted)
    return {
        "num_points": len(away),
        "num_away": away_count,
        "num_near_or_on_surface": int((~away).sum()),
        "num_sign_disagreements": int(disagree.sum()),
        "num_counted_disagreements": count,
        "disagreement_percent_all": 100.0 * count / len(away),
        "disagreement_percent_away": 100.0 * count / away_count if away_count else None,
        "pseudonormal_inside_count": int((arrays["sign_p"] == 1).sum()),
        "winding_inside_count": int((arrays["sign_w"] == 1).sum()),
        "nonbinary_winding_count": int((~np.isin(arrays["rounded_winding"], [0, 1])).sum()),
        "zero_signed_distance_count": int((arrays["d"] == 0).sum()),
        "winding_min": float(arrays["winding"].min()),
        "winding_max": float(arrays["winding"].max()),
    }


def write_csv(path, rows):
    """Write a complete CSV checkpoint atomically, preserving all available fields."""
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def environment_info():
    """Record versions and CPU limits without collecting host identifiers."""
    cpu_model = (
        next(
            (
                line.split(":", 1)[1].strip()
                for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")
            ),
            "unknown",
        )
        if Path("/proc/cpuinfo").exists()
        else platform.processor()
    )
    cpu_limit = Path("/sys/fs/cgroup/cpu.max")
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=False, capture_output=True, text=True, timeout=10
    )
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_model": cpu_model,
        "logical_cpus": os.cpu_count(),
        "cpu_affinity_count": len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else None,
        "cgroup_cpu_max": cpu_limit.read_text().strip() if cpu_limit.exists() else None,
        "igl_num_threads": os.environ["IGL_NUM_THREADS"],
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("libigl", "numpy", "scipy", "huggingface-hub")
        },
        "git_commit": revision.stdout.strip() if revision.returncode == 0 else None,
        "source_sha256": {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("benchmark.py", "dataset_cache.py", "uv.lock")
        },
    }


def run(args):
    """Run the benchmark, checkpoint every model, and retain per-point outputs."""
    if args.points <= 0 or args.seed < 0 or args.threads <= 0:
        raise ValueError("Points and threads must be positive, and seed must be nonnegative.")
    if not np.isfinite(args.relative_cutoff) or args.relative_cutoff < 0:
        raise ValueError("The relative cutoff must be finite and nonnegative.")
    os.environ["IGL_NUM_THREADS"] = str(args.threads)
    started = perf_counter()
    output = args.output.resolve()
    if (output / "config.json").exists():
        raise FileExistsError(f"A run already exists at {output}; choose a new --output directory.")
    output.mkdir(parents=True, exist_ok=True)
    metadata_path, mesh_dir, preparation = prepare_dataset(args.cache_dir, args.revision)
    write_json(output / "cache_access.json", preparation)
    acquisition = args.cache_dir.resolve() / f"acquisition-{args.revision}.json"
    if not (output / "download.json").exists():
        shutil.copyfile(acquisition, output / "download.json")
    step_started = perf_counter()
    with metadata_path.open(newline="") as stream:
        metadata = list(csv.DictReader(stream))
    metadata_load_seconds = perf_counter() - step_started
    selected, eligible_count = select_models(metadata, args.models, args.seed)
    config = {
        "started_at_utc": datetime.now(UTC).isoformat(),
        "dataset_repository": preparation["repository"],
        "dataset_revision": args.revision,
        "metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
        "metadata_rows": len(metadata),
        "eligible_models": eligible_count,
        "filter": {"solid": 1, "vertex_manifold": 1, "edge_manifold": 1},
        "models": args.models,
        "points_per_model": args.points,
        "seed": args.seed,
        "model_selection": "PCG64 choice without replacement; eligible IDs sorted numerically",
        "point_sampling": "PCG64 SeedSequence([seed, file_id]); uniform original axis-aligned bbox",
        "relative_cutoff": args.relative_cutoff,
        "distance_cutoff": "d > relative_cutoff * norm(bbox_max - bbox_min)",
        "sign_p": "+1 if libigl signed_distance < 0, else -1",
        "sign_w": "2 * numpy.rint(fast_winding_number) - 1; ties to even; no clipping",
        "primary_percent": "100 * count(sign_p != sign_w AND d > cutoff) / num_points",
        "timing_scope": "One full batch call per method per model, including internal setup; no warmup",
        "metadata_load_seconds": metadata_load_seconds,
        "environment": environment_info(),
    }
    write_json(output / "config.json", config)
    write_csv(output / "selected_metadata.csv", selected)
    (output / "points").mkdir(exist_ok=True)
    results = []
    for index, metadata_row in enumerate(selected):
        file_id = metadata_row["file_id"]
        row = {"sample_index": index, "file_id": file_id, "status": "ok", "error": ""}
        model_started = perf_counter()
        try:
            mesh_path = mesh_dir / f"{file_id}.npz"
            step_started = perf_counter()
            vertices, faces = load_mesh(mesh_path)
            row["mesh_load_seconds"] = perf_counter() - step_started
            row.update(
                num_vertices=len(vertices),
                num_faces=len(faces),
                num_connected_components=int(metadata_row["num_connected_components"]),
                mesh_file_bytes=mesh_path.stat().st_size,
            )
            if len(vertices) != int(metadata_row["num_vertices"]) or len(faces) != int(
                metadata_row["num_faces"]
            ):
                raise ValueError("Loaded vertex/face counts differ from the dataset metadata.")
            step_started = perf_counter()
            with mesh_path.open("rb") as stream:
                row["mesh_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
            row["mesh_hash_seconds"] = perf_counter() - step_started
            step_started = perf_counter()
            points, lower, upper, bbd = sample_points(vertices, args.points, args.seed, file_id)
            row["sampling_seconds"] = perf_counter() - step_started
            row.update(bbd=bbd, distance_cutoff=bbd * args.relative_cutoff)
            for axis, minimum, maximum in zip("xyz", lower, upper, strict=True):
                row[f"bbox_min_{axis}"] = float(minimum)
                row[f"bbox_max_{axis}"] = float(maximum)
            arrays, query_timings = query_mesh(vertices, faces, points)
            row.update(query_timings)
            row.update(disagreement_statistics(arrays, bbd, args.relative_cutoff))
            row["points_file"] = f"points/{file_id}.npz"
            step_started = perf_counter()
            np.savez_compressed(output / row["points_file"], **arrays)
            row["point_output_seconds"] = perf_counter() - step_started
            if row["nonbinary_winding_count"]:
                row["status"] = "warning_nonbinary_winding"
        except Exception as error:
            logging.getLogger(__name__).exception("Failed on file_id=%s", file_id)
            row["status"] = "error"
            row["error"] = f"{type(error).__name__}: {error}"
        row["model_total_seconds"] = perf_counter() - model_started
        results.append(row)
        write_csv(output / "per_model.csv", results)
        print(
            f"[{index + 1}/{len(selected)}] file_id={file_id} {row['status']} "
            f"disagreement={row.get('disagreement_percent_all', 'n/a')}% "
            f"seconds={row['model_total_seconds']:.3f} {row['error']}",
            flush=True,
        )
    succeeded = [row for row in results if row["status"] != "error"]
    point_count = sum(row["num_points"] for row in succeeded)
    away_count = sum(row["num_away"] for row in succeeded)
    disagreements = sum(row["num_counted_disagreements"] for row in succeeded)
    summary = {
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "models_requested": len(selected),
        "models_completed": len(succeeded),
        "models_failed": len(results) - len(succeeded),
        "models_with_nonbinary_winding": sum(
            row["nonbinary_winding_count"] > 0 for row in succeeded
        ),
        "models_with_counted_disagreement": sum(
            row["num_counted_disagreements"] > 0 for row in succeeded
        ),
        "num_points": point_count,
        "num_away": away_count,
        "num_counted_disagreements": disagreements,
        "disagreement_percent_all": 100 * disagreements / point_count if point_count else None,
        "disagreement_percent_away": 100 * disagreements / away_count if away_count else None,
        "max_model_disagreement_percent_all": max(
            (row["disagreement_percent_all"] for row in succeeded), default=None
        ),
        "summed_timings_seconds": {
            key: sum(row.get(key, 0) for row in results)
            for key in (
                "mesh_load_seconds",
                "mesh_hash_seconds",
                "sampling_seconds",
                "pseudonormal_seconds",
                "fast_winding_seconds",
                "point_output_seconds",
                "model_total_seconds",
            )
        },
        "cache_access_seconds": preparation["total_seconds"],
        "run_wall_seconds": perf_counter() - started,
        "timing_note": "Download/extraction acquisition timings are in download.json; this run's cache access is in cache_access.json.",
    }
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 1 if summary["models_failed"] else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--output", type=Path, default=Path("results/initial-100"))
    parser.add_argument("--models", type=int, default=100)
    parser.add_argument("--points", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20261123)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--relative-cutoff", type=float, default=0.001)
    raise SystemExit(run(parser.parse_args()))
