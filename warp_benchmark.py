"""Compare pinned Warp main against the saved libigl benchmark, point for point."""

import argparse
import csv
import hashlib
import json
import logging
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import warp as wp

import warp_queries
from benchmark import append_result, load_mesh, write_csv
from dataset_cache import DEFAULT_CACHE, prepare_dataset, write_json
from warp_queries import WARP_METHODS, query_warp

METHODS = ("igl_p", "warp_p", "warp_w", "warp_parity")


def file_sha256(path):
    """Hash a file without loading its entire contents into memory."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_warp_source():
    """Reject a release wheel or checkout that differs from the pinned GitHub main."""
    expected = json.loads(Path(__file__).with_name("warp_source.json").read_text())
    checkout = Path(wp.__file__).resolve().parents[1]
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=checkout,
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    ).stdout.strip()
    if head != expected["revision"] or wp.__version__ != expected["version"]:
        raise ValueError(
            f"Warp must be built from {expected['revision']}; imported {head}, {wp.__version__}."
        )
    subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", "warp", "build_lib.py", "VERSION.md"],
        cwd=checkout,
        check=True,
        timeout=10,
    )
    return {
        **expected,
        "checkout": str(checkout),
        "native_library_sha256": {
            path.name: file_sha256(path) for path in (checkout / "warp/bin").glob("*.so")
        },
    }


def comparison_statistics(reference, result, bbd, relative_cutoff):
    """Compare signs using the original libigl distance mask for every method."""
    for method in WARP_METHODS:
        if not np.all(result[f"warp_valid_{method}"] == 1):
            raise ValueError(
                f"Warp {method} query missed {np.count_nonzero(result[f'warp_valid_{method}'] != 1)} points."
            )
        if not np.isfinite(result[f"warp_d_{method}"]).all():
            raise ValueError(f"Warp {method} distances contain nonfinite values.")
    reference_sign = reference["sign_w"]
    if not np.isin(reference_sign, [-1, 1]).all():
        raise ValueError("The libigl winding reference must have binary signs.")
    away = reference["d"] > relative_cutoff * bbd
    signs = {
        "igl_p": reference["sign_p"],
        "warp_p": result["warp_sign_p"],
        "warp_w": result["warp_sign_w"],
        "warp_parity": result["warp_sign_parity"],
    }
    statistics = {"num_points": len(away), "num_away": int(away.sum())}
    masks = {}
    errors = {}
    for name, sign in signs.items():
        if not np.isin(sign, [-1, 1]).all():
            raise ValueError(f"{name} produced nonbinary signs.")
        mismatch = sign != reference_sign
        counted = mismatch & away
        masks[name] = counted
        statistics.update(
            {
                f"{name}_raw_disagreements": int(mismatch.sum()),
                f"{name}_disagreements": int(counted.sum()),
                f"{name}_percent_all": 100 * float(counted.mean()),
                f"{name}_percent_away": 100 * int(counted.sum()) / int(away.sum())
                if away.any()
                else None,
                f"{name}_false_inside": int((counted & (sign == 1)).sum()),
                f"{name}_false_outside": int((counted & (sign == -1)).sum()),
            }
        )
    for method in WARP_METHODS:
        error = np.abs(result[f"warp_d_{method}"].astype(np.float64) - reference["d"])
        normalized = error / bbd
        errors[method] = normalized
        statistics.update(
            {
                f"warp_d_{method}_max_abs_error": float(error.max()),
                f"warp_d_{method}_mean_abs_error": float(error.mean()),
                f"warp_d_{method}_max_error_over_bbd": float(normalized.max()),
                f"warp_d_{method}_mean_error_over_bbd": float(normalized.mean()),
                f"warp_d_{method}_p99_error_over_bbd": float(np.quantile(normalized, 0.99)),
                f"warp_d_{method}_errors_over_1e_minus6_bbd": int((normalized > 1e-6).sum()),
                f"warp_d_{method}_errors_over_1e_minus5_bbd": int((normalized > 1e-5).sum()),
                f"warp_d_{method}_max_relative_error_away": float(
                    (error[away] / reference["d"][away]).max()
                )
                if away.any()
                else None,
            }
        )
    statistics["warp_p_vs_w_distance_max_over_bbd"] = float(
        np.max(np.abs(result["warp_d_p"].astype(np.float64) - result["warp_d_w"])) / bbd
    )
    return statistics, masks, errors


def run(args):
    """Evaluate the saved experiment and checkpoint every model's comparison."""
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Choose a fresh output directory; {output} already exists.")
    if (
        args.epsilon < 0
        or not np.isfinite(args.epsilon)
        or args.accuracy <= 0
        or not np.isfinite(args.accuracy)
    ):
        raise ValueError(
            "Epsilon must be finite and nonnegative; accuracy must be finite and positive."
        )
    baseline = args.baseline.resolve()
    if args.parity_rays <= 0 or args.parity_rays % 2 == 0:
        raise ValueError("Use a positive, odd parity ray count.")
    if args.parity_perturbation < 0 or not np.isfinite(args.parity_perturbation):
        raise ValueError("Parity perturbation must be finite and nonnegative.")
    if args.progress_every <= 0:
        raise ValueError("Progress interval must be positive.")
    baseline_config = json.loads((baseline / "config.json").read_text())
    with (baseline / "per_model.csv").open(newline="") as stream:
        baseline_rows = list(csv.DictReader(stream))
    if any(row["status"] != "ok" for row in baseline_rows):
        raise ValueError("The reference run must have succeeded without warnings.")
    if len(baseline_rows) != baseline_config["models"]:
        raise ValueError("The reference model count does not match its configuration.")
    for row in baseline_rows:
        if not (baseline / row["points_file"]).is_file():
            raise FileNotFoundError(
                "Retained baseline points are required; rerun benchmark.py with the recorded seed first."
            )
    started = perf_counter()
    source = verify_warp_source()
    provenance_seconds = perf_counter() - started
    step_started = perf_counter()
    wp.init()
    device = wp.get_device(args.device)
    initialization_seconds = perf_counter() - step_started
    step_started = perf_counter()
    wp.load_module(module=warp_queries, device=device)
    wp.synchronize_device(device)
    module_load_seconds = perf_counter() - step_started
    _, mesh_dir, cache_access = prepare_dataset(args.cache_dir, baseline_config["dataset_revision"])
    output.mkdir(parents=True)
    (output / "points").mkdir()
    config = {
        "started_at_utc": datetime.now(UTC).isoformat(),
        "baseline": str(baseline),
        "baseline_config_sha256": file_sha256(baseline / "config.json"),
        "baseline_per_model_sha256": file_sha256(baseline / "per_model.csv"),
        "warp_source": source,
        "device": str(device),
        "device_name": device.name,
        "device_arch": device.arch,
        "numpy_version": np.__version__,
        "source_sha256": {
            name: file_sha256(Path(__file__).with_name(name))
            for name in ("warp_benchmark.py", "warp_queries.py", "benchmark.py")
        },
        "dataset_revision": baseline_config["dataset_revision"],
        "models": len(baseline_rows),
        "points_per_model": baseline_config["points_per_model"],
        "reference": "saved igl_sign_w is treated as ground truth for this experiment",
        "distance_mask": "original igl_d > baseline relative_cutoff * original bbd",
        "relative_cutoff": baseline_config["relative_cutoff"],
        "coordinates": "same saved float64 samples, converted to float32 for Warp; no normalization",
        "distance": "length(query_point - mesh_eval_position(query.face, query.u, query.v))",
        "warp_sign": "+1 if native query.sign < 0, else -1",
        "pseudonormal_epsilon": args.epsilon,
        "winding_accuracy": args.accuracy,
        "parity_rays": args.parity_rays,
        "parity_perturbation": args.parity_perturbation,
        "parity_randomness": "Warp uses a fixed internal seed for deterministic ray directions",
        "winding_threshold": 0.5,
        "support_winding_number": True,
        "max_distance": "2 * original bbd",
        "bvh_constructor": "lbvh" if device.is_cuda else "sah",
        "timing": "one call per method per model, no warmup; synchronous wall time; mesh build, uploads, compilation and downloads separate",
        "checkpoint": "append and flush per_model.jsonl after each model; export CSV once at completion",
        "progress_every": args.progress_every,
        "provenance_seconds": provenance_seconds,
        "initialization_seconds": initialization_seconds,
        "module_load_seconds": module_load_seconds,
    }
    write_json(output / "config.json", config)
    write_json(output / "cache_access.json", cache_access)
    rows, cases, distance_cases = [], [], []
    checkpoint_seconds = 0.0
    errors = {method: [] for method in WARP_METHODS}
    for index, baseline_row in enumerate(baseline_rows):
        model_started = perf_counter()
        file_id = baseline_row["file_id"]
        row = {
            "sample_index": index,
            "file_id": file_id,
            "status": "ok",
            "error": "",
            "bbd": float(baseline_row["bbd"]),
            "num_faces": int(baseline_row["num_faces"]),
            "num_vertices": int(baseline_row["num_vertices"]),
            "num_connected_components": int(baseline_row["num_connected_components"]),
        }
        try:
            step_started = perf_counter()
            point_path = baseline / baseline_row["points_file"]
            with np.load(point_path, allow_pickle=False) as arrays:
                reference = {
                    key: arrays[key] for key in ("points", "d", "sign_p", "sign_w", "winding")
                }
            row["reference_load_seconds"] = perf_counter() - step_started
            mesh_path = mesh_dir / f"{file_id}.npz"
            step_started = perf_counter()
            vertices, faces = load_mesh(mesh_path)
            row["mesh_load_seconds"] = perf_counter() - step_started
            row["mesh_sha256"] = file_sha256(mesh_path)
            row["reference_points_file_sha256"] = file_sha256(point_path)
            if row["mesh_sha256"] != baseline_row["mesh_sha256"]:
                raise ValueError("Mesh bytes differ from the baseline experiment.")
            if len(reference["points"]) != baseline_config["points_per_model"]:
                raise ValueError("Reference sample count differs from the baseline configuration.")
            result, timings = query_warp(
                vertices,
                faces,
                reference["points"],
                device,
                epsilon=args.epsilon,
                accuracy=args.accuracy,
                parity_rays=args.parity_rays,
                parity_perturbation=args.parity_perturbation,
            )
            row.update(timings)
            row["points_file"] = f"points/{file_id}.npz"
            step_started = perf_counter()
            np.savez_compressed(output / row["points_file"], **result)
            row["point_output_seconds"] = perf_counter() - step_started
            stats, masks, normalized_errors = comparison_statistics(
                reference, result, row["bbd"], config["relative_cutoff"]
            )
            row.update(stats)
            row["max_vertex_rounding_over_bbd"] = float(result["max_vertex_rounding_over_bbd"])
            row["max_query_rounding_over_bbd"] = float(result["max_query_rounding_over_bbd"])
            for method in WARP_METHODS:
                errors[method].append(normalized_errors[method])
            mismatch_union = np.logical_or.reduce(list(masks.values()))
            distance_outlier = np.maximum.reduce(list(normalized_errors.values())) > 1e-5
            for point_index in np.flatnonzero(mismatch_union | distance_outlier):
                case = {
                    "file_id": file_id,
                    "point_index": int(point_index),
                    "igl_d": float(reference["d"][point_index]),
                    "igl_winding": float(reference["winding"][point_index]),
                    "igl_sign_p": int(reference["sign_p"][point_index]),
                    "igl_sign_w": int(reference["sign_w"][point_index]),
                }
                for axis, value in zip("xyz", reference["points"][point_index], strict=True):
                    case[f"point_{axis}"] = float(value)
                for method in WARP_METHODS:
                    case[f"warp_d_{method}"] = float(result[f"warp_d_{method}"][point_index])
                    case[f"warp_sign_{method}"] = int(result[f"warp_sign_{method}"][point_index])
                    case[f"warp_face_{method}"] = int(result[f"warp_face_{method}"][point_index])
                    case[f"warp_d_{method}_error_over_bbd"] = float(
                        normalized_errors[method][point_index]
                    )
                case["igl_d_over_bbd"] = case["igl_d"] / row["bbd"]
                if mismatch_union[point_index]:
                    cases.append(case)
                if distance_outlier[point_index]:
                    distance_cases.append(case)
        except Exception as error:
            logging.getLogger(__name__).exception("Warp comparison failed on file_id=%s", file_id)
            row.update(status="error", error=f"{type(error).__name__}: {error}")
        row["model_total_seconds"] = perf_counter() - model_started
        rows.append(row)
        step_started = perf_counter()
        append_result(output / "per_model.jsonl", row)
        checkpoint_seconds += perf_counter() - step_started
        if (
            index == 0
            or (index + 1) % args.progress_every == 0
            or index + 1 == len(baseline_rows)
            or row["status"] != "ok"
        ):
            print(
                f"[{index + 1}/{len(baseline_rows)}] {file_id} {row['status']} "
                f"igl_p={row.get('igl_p_disagreements')} warp_p={row.get('warp_p_disagreements')} "
                f"warp_w={row.get('warp_w_disagreements')} parity={row.get('warp_parity_disagreements')} "
                f"elapsed={perf_counter() - started:.1f}s {row['error']}",
                flush=True,
            )
    step_started = perf_counter()
    write_csv(output / "per_model.csv", rows)
    csv_export_seconds = perf_counter() - step_started
    good = [row for row in rows if row["status"] == "ok"]
    total = sum(row["num_points"] for row in good)
    away = sum(row["num_away"] for row in good)
    summary = {
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "models_completed": len(good),
        "models_failed": len(rows) - len(good),
        "num_points": total,
        "num_away": away,
        "methods": {},
        "distance_errors": {},
        "checkpoint_seconds": checkpoint_seconds,
        "csv_export_seconds": csv_export_seconds,
    }
    for method in METHODS:
        count = sum(row[f"{method}_disagreements"] for row in good)
        summary["methods"][method] = {
            "disagreements": count,
            "percent_all": 100 * count / total if total else None,
            "percent_away": 100 * count / away if away else None,
            "models_with_disagreements": sum(row[f"{method}_disagreements"] > 0 for row in good),
            "raw_disagreements": sum(row[f"{method}_raw_disagreements"] for row in good),
            "false_inside": sum(row[f"{method}_false_inside"] for row in good),
            "false_outside": sum(row[f"{method}_false_outside"] for row in good),
        }
    for method in WARP_METHODS:
        if errors[method]:
            combined = np.concatenate(errors[method])
            summary["distance_errors"][method] = {
                "max_over_bbd": float(combined.max()),
                "mean_over_bbd": float(combined.mean()),
                "rms_over_bbd": float(np.sqrt(np.mean(combined**2))),
                "p99_over_bbd": float(np.quantile(combined, 0.99)),
                "count_over_1e_minus6_bbd": int((combined > 1e-6).sum()),
                "count_over_1e_minus5_bbd": int((combined > 1e-5).sum()),
            }
    summary["summed_timings_seconds"] = {
        key: sum(row.get(key, 0) for row in rows)
        for key in (
            "reference_load_seconds",
            "mesh_load_seconds",
            "input_conversion_upload_seconds",
            "mesh_build_seconds",
            "output_allocation_seconds",
            "pseudonormal_query_seconds",
            "winding_query_seconds",
            "parity_query_seconds",
            "result_download_seconds",
            "point_output_seconds",
            "model_total_seconds",
        )
    }
    for filename, entries in (
        ("disagreements.csv", cases),
        ("distance_outliers.csv", distance_cases),
    ):
        if entries:
            write_csv(output / filename, entries)
        else:
            (output / filename).write_text("file_id,point_index\n")
    summary["wall_seconds"] = perf_counter() - started
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 1 if summary["models_failed"] else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=Path("results/initial-100"))
    parser.add_argument("--output", type=Path, default=Path("results/warp-main-100"))
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epsilon", type=float, default=0.001)
    parser.add_argument("--accuracy", type=float, default=2.0)
    parser.add_argument("--parity-rays", type=int, default=1)
    parser.add_argument("--parity-perturbation", type=float, default=0.1)
    parser.add_argument("--progress-every", type=int, default=50)
    raise SystemExit(run(parser.parse_args()))
