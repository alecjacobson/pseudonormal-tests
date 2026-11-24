"""Sweep runtime pseudonormal tolerances on the original cached 100-model sample."""

import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from benchmark import load_mesh, write_csv
from dataset_cache import DEFAULT_CACHE, prepare_dataset, write_json
from precision_benchmark import source_state

DEFAULT_EPSILONS = "0,1e-14,1e-12,1e-10,1e-9,1e-8,3e-8,1e-7,3e-7,1e-6,3e-6,1e-5,3e-5,1e-4,3e-4,1e-3,3e-3,1e-2"
DEFAULT_AREAS = "0,1e-12,1e-8,1e-6,1e-4,1e-2,1,1e2,1e4,1e8"
DEFAULT_AREA = np.float32(1e-4).item()
DEFAULT_EPSILON = np.float32(1e-12).item()


def parse_values(text):
    """Round parameter values exactly as the float classifier receives them."""
    values = [np.float32(float(value)).item() for value in text.split(",")]
    if any(not np.isfinite(value) or value < 0 for value in values):
        raise ValueError("All sweep values must be finite and nonnegative")
    if len(values) != len(set(values)):
        raise ValueError("Sweep values must remain distinct after conversion to float32")
    return values


def run(args):
    """Reuse closest features and normal preprocessing across all parameter pairs."""
    started = perf_counter()
    os.environ["IGL_NUM_THREADS"] = str(args.threads)
    if args.output.exists():
        raise FileExistsError(f"Choose a fresh output directory: {args.output}")
    if args.threads < 1:
        raise ValueError("threads must be positive")
    areas, epsilons = parse_values(args.areas), parse_values(args.epsilons)
    if DEFAULT_AREA not in areas or DEFAULT_EPSILON not in epsilons:
        raise ValueError("The grid must include the original default parameter pair")
    settings = [
        {"setting_id": i, "minimum_double_area": area, "epsilon": epsilon}
        for i, (area, epsilon) in enumerate((a, e) for a in areas for e in epsilons)
    ]
    default_id = next(s["setting_id"] for s in settings
                      if s["minimum_double_area"] == DEFAULT_AREA and s["epsilon"] == DEFAULT_EPSILON)
    args.output.mkdir(parents=True)
    (args.output / "points").mkdir()
    spec = importlib.util.spec_from_file_location("_tolerance_sweep", args.module.resolve())
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _, meshes, cache_access = prepare_dataset(args.cache_dir)
    write_json(args.output / "cache_access.json", cache_access)
    reference_rows = list(csv.DictReader((args.reference / "per_model.csv").open()))
    ids = sorted(int(row["file_id"]) for row in reference_rows)
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate reference models")
    tuning_ids = set(np.random.default_rng(args.split_seed).permutation(ids)[:len(ids)//2].tolist())
    config = {
        "settings": settings,
        "default_setting_id": default_id,
        "reference": str(args.reference),
        "float_baseline": str(args.float_baseline),
        "threads": args.threads,
        "split_seed": args.split_seed,
        "tuning_ids": sorted(tuning_ids),
        "holdout_ids": sorted(set(ids)-tuning_ids),
        "selection": "Minimize tuning-set counted disagreements, then invalid queries, then log-distance of parameters from defaults (area first). Holdout models do not select parameters.",
        "module": str(args.module.resolve()),
        "module_sha256": hashlib.sha256(args.module.read_bytes()).hexdigest(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "libigl": source_state(args.libigl_source),
        "bindings": source_state(args.bindings_source),
        "generated_classifier": json.loads((args.module.parent / "pseudonormal_runtime.json").read_text()),
        "cpp_wrapper_sha256": hashlib.sha256(Path("native_precision/tolerance_sweep.cpp").read_bytes()).hexdigest(),
        "reference_config": json.loads((args.reference / "config.json").read_text()),
        "float_baseline_config": json.loads((args.float_baseline / "config.json").read_text()),
        "cmake_cache": (args.module.parent / "CMakeCache.txt").read_text(),
        "coordinates": "Original scale and position; parameters are absolute values with original semantics",
        "epsilon_units": "Dimensionless for barycentrics in the area branch; coordinate-distance units in the small-area branch",
        "area_units": "Twice triangle area, in squared coordinate units",
        "mask": "Saved original double d > 0.001 * original bbd",
        "queries": "Classifier only, with original patched-float closest features and once-per-model normal construction",
    }
    if config["libigl"]["diff"] or config["bindings"]["diff"]:
        raise ValueError("Use clean upstream source checkouts")
    write_json(args.output / "config.json", config)
    integer_metrics = (
        "points", "away_points", "raw_disagreements", "counted_disagreements",
        "valid_counted_disagreements", "invalid_queries", "invalid_queries_away",
        "new_vs_default", "resolved_vs_default", "models_with_disagreements",
    )
    totals = {
        fold: [{**setting, **dict.fromkeys(integer_metrics, 0), "query_seconds": 0.0}
               for setting in settings]
        for fold in ["all", "tuning", "holdout"]
    }
    baseline_audit = {"models": 0, "points": 0, "sign_changes": 0,
                      "normal_differing_bytes": 0, "validity_changes": 0}
    model_rows, timing_rows = [], []
    for model_index, source in enumerate(reference_rows):
        file_id = source["file_id"]
        if source["status"] != "ok":
            raise ValueError(f"Invalid reference model {file_id}")
        fold = "tuning" if int(file_id) in tuning_ids else "holdout"
        load_started = perf_counter()
        mesh_path = meshes / f"{file_id}.npz"
        if hashlib.sha256(mesh_path.read_bytes()).hexdigest() != source["mesh_sha256"]:
            raise ValueError(f"Mesh hash mismatch: {file_id}")
        vertices, faces = load_mesh(mesh_path)
        vertices = vertices.astype(np.float32)
        with np.load(args.reference / source["points_file"], allow_pickle=False) as data:
            points = data["points"].astype(np.float32)
            reference_sign = data["sign_w"]
            away = data["away_from_surface"]
            np.testing.assert_array_equal(away, data["d"] > 0.001*float(source["bbd"]))
        with np.load(args.float_baseline / "points" / f"{file_id}.npz", allow_pickle=False) as data:
            baseline = dict(data)
        distance = np.abs(baseline["signed_distance"])
        baseline_counted = (baseline["sign_p"] != reference_sign) & away
        load_seconds = perf_counter() - load_started
        prepare_started = perf_counter()
        prepared = module.PreparedMesh(vertices, faces)
        prepare_seconds = perf_counter() - prepare_started
        default_started = perf_counter()
        default_sign, default_normals = prepared.classify(
            points, baseline["closest_face_index"], baseline["closest_point"],
            DEFAULT_AREA, DEFAULT_EPSILON)
        default_seconds = perf_counter() - default_started
        default_sign = np.where(default_sign*distance < 0, 1, -1).astype(np.int8)
        default_valid = np.isfinite(default_normals).all(axis=1)
        changes = {
            "sign_changes": int(np.count_nonzero(default_sign != baseline["sign_p"])),
            "normal_differing_bytes": int(np.count_nonzero(
                np.ascontiguousarray(default_normals).view(np.uint8)
                != np.ascontiguousarray(baseline["pseudonormal"]).view(np.uint8))),
            "validity_changes": int(np.count_nonzero(default_valid != baseline["valid_query"])),
        }
        if any(changes.values()):
            raise RuntimeError(f"Default classifier does not reproduce {file_id}: {changes}")
        baseline_audit["models"] += 1
        baseline_audit["points"] += len(points)
        for key, value in changes.items():
            baseline_audit[key] += value
        signs = np.empty((len(settings), len(points)), dtype=np.int8)
        validity = np.empty_like(signs, dtype=bool)
        current_rows = []
        for setting in settings:
            index = setting["setting_id"]
            if index == default_id:
                sign, valid, seconds = default_sign, default_valid, default_seconds
            else:
                query_started = perf_counter()
                native_sign, normals = prepared.classify(
                    points, baseline["closest_face_index"], baseline["closest_point"],
                    setting["minimum_double_area"], setting["epsilon"])
                seconds = perf_counter() - query_started
                sign = np.where(native_sign*distance < 0, 1, -1).astype(np.int8)
                valid = np.isfinite(normals).all(axis=1)
            signs[index], validity[index] = sign, valid
            disagree = sign != reference_sign
            counted = disagree & away
            values = {
                "points": len(points), "away_points": int(away.sum()),
                "raw_disagreements": int(disagree.sum()),
                "counted_disagreements": int(counted.sum()),
                "valid_counted_disagreements": int((counted & valid).sum()),
                "invalid_queries": int((~valid).sum()),
                "invalid_queries_away": int((~valid & away).sum()),
                "new_vs_default": int((counted & ~baseline_counted).sum()),
                "resolved_vs_default": int((~counted & baseline_counted).sum()),
                "models_with_disagreements": int(counted.any()),
                "query_seconds": seconds,
            }
            for name in ["all", fold]:
                for key, value in values.items():
                    totals[name][index][key] += value
            current_rows.append({"file_id": file_id, "split": fold, **setting, **values})
        output_started = perf_counter()
        np.savez_compressed(args.output / "points" / f"{file_id}.npz",
                            sign_p=signs, valid_query=validity)
        with (args.output / "per_model.jsonl").open("a") as stream:
            stream.writelines(json.dumps(row)+"\n" for row in current_rows)
        model_rows.extend(current_rows)
        timing_rows.append({"file_id": file_id, "load_seconds": load_seconds,
                            "normal_precompute_seconds": prepare_seconds,
                            "query_seconds": sum(row["query_seconds"] for row in current_rows),
                            "output_seconds": perf_counter()-output_started})
        if (model_index+1) % 10 == 0:
            print(f"Swept {model_index+1}/{len(reference_rows)} models, {len(settings)} parameter pairs", flush=True)
    for fold, rows in totals.items():
        for row in rows:
            row["percent_all"] = 100*row["counted_disagreements"]/row["points"]
            row["percent_away"] = 100*row["counted_disagreements"]/row["away_points"]
        write_csv(args.output / f"grid_{fold}.csv", rows)

    def rank(row):
        area_offset = abs(np.log10(row["minimum_double_area"]/DEFAULT_AREA)) if row["minimum_double_area"] else 100
        epsilon_offset = abs(np.log10(row["epsilon"]/DEFAULT_EPSILON)) if row["epsilon"] else 100
        return row["counted_disagreements"], row["invalid_queries"], area_offset, epsilon_offset

    selected_id = min(totals["tuning"], key=rank)["setting_id"]
    best_id = min(totals["all"], key=rank)["setting_id"]
    summary = {
        "models": len(reference_rows), "settings": len(settings),
        "baseline_audit": baseline_audit,
        "default": {fold: rows[default_id] for fold, rows in totals.items()},
        "selected_on_tuning": {fold: rows[selected_id] for fold, rows in totals.items()},
        "best_overall_exploratory": {fold: rows[best_id] for fold, rows in totals.items()},
        "normal_precompute_seconds": sum(row["normal_precompute_seconds"] for row in timing_rows),
        "classifier_seconds": sum(row["query_seconds"] for row in timing_rows),
        "wall_seconds": perf_counter()-started,
        "interpretation": "This is a tolerance sweep, not an algorithm change. Preserve invalid-query counts when interpreting returned-sign disagreements.",
    }
    write_csv(args.output / "per_model.csv", model_rows)
    write_csv(args.output / "timings.csv", timing_rows)
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", type=Path, required=True)
    parser.add_argument("--libigl-source", type=Path, required=True)
    parser.add_argument("--bindings-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=Path("results/initial-100"))
    parser.add_argument("--float-baseline", type=Path, default=Path("results/libigl-scalar-single-100"))
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--epsilons", default=DEFAULT_EPSILONS)
    parser.add_argument("--areas", default=DEFAULT_AREAS)
    parser.add_argument("--split-seed", type=int, default=20261124)
    parser.add_argument("--threads", type=int, default=4)
    run(parser.parse_args())
