"""Classify in float32 using normals built and normalized in float64."""

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import numpy as np

from benchmark import append_result, load_mesh, write_csv
from dataset_cache import DEFAULT_CACHE, prepare_dataset, write_json
from precision_benchmark import source_state
from probe_normalization import same_bytes
from validate_tolerances import file_hash, import_module, read_rows

VARIANTS = ("float", "double_normals_rounded", "double_normals_original")


def metrics(sign, normals, reference, baseline_sign, baseline_valid):
    """Count discrepancies, invalid normals, and changes with a fixed cutoff."""
    away = reference["away_from_surface"]
    valid = np.isfinite(normals).all(axis=1)
    zero = valid & (normals == 0).all(axis=1)
    wrong = (sign != reference["sign_w"]) & away
    before = (baseline_sign != reference["sign_w"]) & away
    result = {
        "points": len(sign),
        "away_points": int(away.sum()),
        "disagreements": int(wrong.sum()),
        "finite_disagreements": int((wrong & valid).sum()),
        "invalid_queries": int((~valid).sum()),
        "invalid_queries_away": int((~valid & away).sum()),
        "zero_normals": int(zero.sum()),
        "sign_changes": int((sign != baseline_sign).sum()),
        "resolved": int((before & ~wrong).sum()),
        "introduced": int((wrong & ~before).sum()),
        "invalid_resolved": int((~baseline_valid & valid).sum()),
        "invalid_introduced": int((baseline_valid & ~valid).sum()),
        "affected_models": int(wrong.any()),
        "models_with_invalid": int((~valid).any()),
    }
    return result, valid, zero, wrong


def trace_cases(
    mesh,
    variant,
    file_id,
    points,
    closest,
    indices,
    normals,
    sign,
    valid,
    selected,
    reference,
    saved,
    bbd,
    area,
    epsilon,
):
    """Record the selected feature and local normal geometry for diagnostic points."""
    sample_indices = np.flatnonzero(selected)
    if not len(sample_indices):
        return []
    native_sign, traced_normals, kinds, features, small, dot, double_dot = mesh.trace(
        points[sample_indices], indices[sample_indices], closest[sample_indices], area, epsilon
    )
    same_bytes(traced_normals, normals[sample_indices])
    np.testing.assert_array_equal(
        np.where(native_sign * np.abs(saved["float32_signed_distance"])[sample_indices] < 0, 1, -1),
        sign[sample_indices],
    )
    fn, _, _, edges, edge_map, source_fn = mesh.normal_tables()
    if variant == "float":
        source_fn = fn
    selected_edges = np.unique(features[kinds == 1])
    edge_faces = {int(edge): [] for edge in selected_edges}
    for occurrence in np.flatnonzero(np.isin(edge_map, selected_edges)):
        edge_faces[int(edge_map[occurrence])].append(int(occurrence % len(fn)))
    rows = []
    for j, sample_index in enumerate(sample_indices):
        i = int(sample_index)
        kind, feature = int(kinds[j]), int(features[j])
        incident = sorted(edge_faces.get(feature, [])) if kind == 1 else []
        raw_sum_norm, adjacent_dot = float("nan"), float("nan")
        if incident:
            contributions = source_fn[incident]
            raw_sum_norm = float(np.linalg.norm(contributions.sum(axis=0).astype(np.float64)))
            if len(incident) == 2:
                adjacent_dot = float(
                    np.dot(contributions[0].astype(np.float64), contributions[1].astype(np.float64))
                )
        direction = (points[i] - closest[i]).astype(np.float64)
        normal_length = float(np.linalg.norm(normals[i].astype(np.float64)))
        denominator = normal_length * np.linalg.norm(direction)
        cosine = (
            float(dot[j] / denominator)
            if np.isfinite(denominator) and denominator > 0
            else float("nan")
        )
        row = {
            "file_id": file_id,
            "sample_index": i,
            "variant": variant,
            "counted_disagreement": bool(
                sign[i] != reference["sign_w"][i] and reference["away_from_surface"][i]
            ),
            "valid_normal": bool(valid[i]),
            "normal_kind": ("face", "edge", "vertex")[kind],
            "normal_feature_index": feature,
            "small_area_branch": bool(small[j]),
            "sign": int(sign[i]),
            "winding_sign": int(reference["sign_w"][i]),
            "float_baseline_sign": int(saved["float32_tuned_sign_p"][i]),
            "double_reference_sign": int(saved["float64_tuned_sign_p"][i]),
            "baseline_valid": bool(saved["float32_tuned_valid_query"][i]),
            "d_over_bbd": float(reference["d"][i] / bbd),
            "float_d_over_bbd": float(abs(saved["float32_signed_distance"][i]) / bbd),
            "closest_face_float": int(indices[i]),
            "closest_face_double": int(reference["closest_face_index"][i]),
            "closest_point_difference_over_bbd": float(
                np.linalg.norm(closest[i].astype(np.float64) - reference["closest_point"][i]) / bbd
            ),
            "dot_float": float(dot[j]),
            "dot_double_of_float_operands": float(double_dot[j]),
            "dot_sign_changes_in_double": bool(
                np.isfinite(dot[j])
                and np.isfinite(double_dot[j])
                and (dot[j] >= 0) != (double_dot[j] >= 0)
            ),
            "normal_length": normal_length,
            "direction_normal_cosine": cosine,
            "edge_vertices": json.dumps(edges[feature].tolist()) if kind == 1 else "[]",
            "incident_faces": json.dumps(incident),
            "edge_source_sum_norm": raw_sum_norm,
            "incident_face_normal_dot": adjacent_dot,
        }
        for axis, letter in enumerate("xyz"):
            row[f"point_{letter}"] = float(reference["points"][i, axis])
            row[f"closest_{letter}"] = float(closest[i, axis])
            row[f"normal_{letter}"] = float(normals[i, axis])
        rows.append(
            {
                key: str(value) if isinstance(value, float) and not np.isfinite(value) else value
                for key, value in row.items()
            }
        )
    return rows


def run(args):
    """Reuse every point and float nearest feature, changing only normal construction."""
    if args.output.exists():
        raise FileExistsError(f"Choose a fresh output directory: {args.output}")
    os.environ["IGL_NUM_THREADS"] = "4"
    started = perf_counter()
    module = import_module(args.module)
    _, mesh_dir, cache_access = prepare_dataset(DEFAULT_CACHE)
    sources = read_rows(args.reference / "per_model.csv")
    if args.file_ids:
        selected_ids = set(args.file_ids)
        sources = [r for r in sources if r["file_id"] in selected_ids]
        if {r["file_id"] for r in sources} != selected_ids:
            raise ValueError("Requested models are missing from the reference")
    initial_ids = {r["file_id"] for r in read_rows(args.initial / "per_model.csv")}
    area, epsilon = float(np.float32(1e-4)), float(np.float32(0.003))
    args.output.mkdir(parents=True)
    (args.output / "points").mkdir()
    write_json(
        args.output / "config.json",
        {
            "started_at_utc": datetime.now(UTC).isoformat(),
            "models": len(sources),
            "selected_ids": [r["file_id"] for r in sources],
            "initial_ids": sorted(initial_ids, key=int),
            "module": str(args.module.resolve()),
            "module_sha256": file_hash(args.module),
            "runner_sha256": file_hash(Path(__file__)),
            "area": area,
            "epsilon": epsilon,
            "reference": str(args.reference),
            "baseline": str(args.baseline),
            "reference_csv_sha256": file_hash(args.reference / "per_model.csv"),
            "baseline_csv_sha256": file_hash(args.baseline / "per_model.csv"),
            "libigl": source_state(args.libigl_source),
            "bindings": source_state(args.bindings_source),
            "generated_sources": {
                p.name: json.loads(p.read_text()) for p in args.module.parent.glob("*.json")
            },
            "variants": {
                "float": "Float32 vertices, normal construction/normalization, closest points, classifier",
                "double_normals_rounded": "Float32-rounded vertices promoted to float64 for all normal construction/normalization; final FN/VN/EN cast to float32",
                "double_normals_original": "Original float64 vertices for all normal construction/normalization; final FN/VN/EN cast to float32",
            },
            "fixed": "All variants use identical float32 vertices/queries/closest points/closest-face indices and pseudonormal_test arithmetic",
            "reference_sign": "Saved original double fast winding, rounded and scaled to inside+1/outside-1",
            "cutoff": "Saved original double distance > 0.001 * bbd",
            "denominator": "All samples",
            "timing": "Normal setup/cast and classifier are timed separately; cached distance and winding queries excluded",
        },
    )
    write_json(args.output / "cache_access.json", cache_access)
    rows, cases, timings = [], [], []
    for index, source in enumerate(sources):
        file_id = source["file_id"]
        model_started = perf_counter()
        path = mesh_dir / f"{file_id}.npz"
        if file_hash(path) != source["mesh_sha256"]:
            raise ValueError(f"Mesh changed: {file_id}")
        vertices, faces = load_mesh(path)
        with np.load(args.reference / "points" / f"{file_id}.npz") as archive:
            reference = dict(archive)
        with np.load(args.baseline / "points" / f"{file_id}.npz") as archive:
            saved = dict(archive)
        np.testing.assert_array_equal(
            reference["away_from_surface"], reference["d"] > 0.001 * float(source["bbd"])
        )
        v = vertices.astype(np.float32)
        points = reference["points"].astype(np.float32)
        closest, indices = saved["float32_closest_point"], saved["float32_closest_face_index"]
        distance = np.abs(saved["float32_signed_distance"])
        np.testing.assert_equal(np.isfinite(distance).all() and np.isfinite(closest).all(), True)
        baseline_sign, baseline_valid = (
            saved["float32_tuned_sign_p"],
            saved["float32_tuned_valid_query"],
        )
        baseline_wrong = (baseline_sign != reference["sign_w"]) & reference["away_from_surface"]
        load_seconds = perf_counter() - model_started
        output = {}
        model_cases = []
        for variant in VARIANTS:
            setup_at = perf_counter()
            if variant == "float":
                mesh = module.PreparedMesh(v, faces)
            elif variant == "double_normals_rounded":
                mesh.use_double_normals(v.astype(np.float64))
            else:
                mesh.use_double_normals(vertices)
            setup_seconds = perf_counter() - setup_at
            query_at = perf_counter()
            native_sign, normals = mesh.classify(points, indices, closest, area, epsilon)
            query_seconds = perf_counter() - query_at
            np.testing.assert_equal(normals.dtype, np.float32)
            sign = np.where(native_sign * distance < 0, 1, -1).astype(np.int8)
            statistics, valid, zero, wrong = metrics(
                sign, normals, reference, baseline_sign, baseline_valid
            )
            if variant == "float":
                same_bytes(normals, saved["float32_tuned_pseudonormal"])
                np.testing.assert_array_equal(sign, baseline_sign)
                np.testing.assert_array_equal(valid, baseline_valid)
            else:
                output[f"{variant}_sign"] = sign
                output[f"{variant}_normal"] = normals
                output[f"{variant}_valid"] = valid
                output[f"{variant}_zero"] = zero
            diagnostic = wrong | ~valid | zero | baseline_wrong | ~baseline_valid
            diagnostic_at = perf_counter()
            model_cases.extend(
                trace_cases(
                    mesh,
                    variant,
                    file_id,
                    points,
                    closest,
                    indices,
                    normals,
                    sign,
                    valid,
                    diagnostic,
                    reference,
                    saved,
                    float(source["bbd"]),
                    area,
                    epsilon,
                )
            )
            row = {
                "file_id": file_id,
                "variant": variant,
                "group": "original_100" if file_id in initial_ids else "additional_models",
                **statistics,
                "normal_setup_seconds": setup_seconds,
                "classify_seconds": query_seconds,
                "diagnostic_seconds": perf_counter() - diagnostic_at,
            }
            row["percent_all"] = 100 * row["disagreements"] / row["points"]
            rows.append(row)
            append_result(args.output / "per_model.jsonl", row)
        del mesh
        cases.extend(model_cases)
        output_at = perf_counter()
        temporary = args.output / "points" / f"{file_id}.tmp.npz"
        np.savez_compressed(temporary, **output)
        temporary.replace(args.output / "points" / f"{file_id}.npz")
        for case in model_cases:
            append_result(args.output / "cases.jsonl", case)
        timing = {
            "file_id": file_id,
            "load_seconds": load_seconds,
            "output_seconds": perf_counter() - output_at,
            "model_seconds": perf_counter() - model_started,
        }
        timings.append(timing)
        append_result(args.output / "timings.jsonl", timing)
        if (index + 1) % 50 == 0 or len(sources) < 50 or index + 1 == len(sources):
            totals = {
                variant: sum(r["disagreements"] for r in rows if r["variant"] == variant)
                for variant in VARIANTS
            }
            print(
                f"{index + 1}/{len(sources)} elapsed={perf_counter() - started:.1f}s disagreements={totals}",
                flush=True,
            )
    write_csv(args.output / "per_model.csv", rows)
    write_csv(args.output / "cases.csv", cases)
    write_csv(args.output / "timings.csv", timings)
    summary = {
        "models": len(sources),
        "wall_seconds": perf_counter() - started,
        "baseline_normal_bytes_signs_and_validity_match": True,
        "groups": {},
    }
    for group in ("all", "original_100", "additional_models"):
        summary["groups"][group] = {}
        for variant in VARIANTS:
            selected = [
                r
                for r in rows
                if r["variant"] == variant and (group == "all" or r["group"] == group)
            ]
            if not selected:
                continue
            totals = {"models": len(selected)}
            for key in selected[0]:
                if key not in ("file_id", "variant", "group", "percent_all"):
                    totals[key] = sum(r[key] for r in selected)
            totals["percent_all"] = 100 * totals["disagreements"] / totals["points"]
            summary["groups"][group][variant] = totals
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary["groups"]["all"], indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=Path("results/full-4963"))
    parser.add_argument(
        "--baseline", type=Path, default=Path("results/libigl-tolerance-validation-4963")
    )
    parser.add_argument("--initial", type=Path, default=Path("results/initial-100"))
    parser.add_argument("--libigl-source", type=Path, default=Path("../libigl"))
    parser.add_argument("--bindings-source", type=Path, default=Path("../libigl-python-bindings"))
    parser.add_argument("--file-ids", nargs="*")
    run(parser.parse_args())
