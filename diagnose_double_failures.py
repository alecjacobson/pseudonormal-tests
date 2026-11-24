"""Separate normal arithmetic, feature classification, and nearest-geometry errors."""

import argparse
import json
from collections import Counter
from pathlib import Path
from time import perf_counter

import igl
import mpmath as mp
import numpy as np

from benchmark import load_mesh, write_csv
from dataset_cache import DEFAULT_CACHE, prepare_dataset, write_json
from exact_mesh_diagnostics import ExactMesh, direct_winding, rational_point, sign_of_dot
from probe_normalization import same_bytes
from validate_tolerances import file_hash, import_module, read_rows


def clean(value):
    """Keep diagnostic nonfinite values explicit in strict JSON."""
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(x) for x in value]
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


def feature_from_barycentric(face, bary, mesh):
    """Read the feature encoded by exact zeros in the nearest routine's output."""
    positive = np.flatnonzero(bary > 0)
    if np.any(bary < 0) or not np.isfinite(bary).all() or len(positive) == 0:
        return None
    if len(positive) == 1:
        return mesh.feature(face, 2, int(positive[0]))
    if len(positive) == 2:
        return mesh.feature(face, 1, int(np.flatnonzero(bary == 0)[0]))
    return 0, face


def run(args):
    """Trace the saved double failures and compare controlled diagnostic substitutions."""
    if args.output.exists():
        raise FileExistsError(args.output)
    mp.mp.dps = args.digits
    module = import_module(args.module)
    reference_path = Path("results/full-4963")
    baseline_path = Path("results/libigl-tolerance-validation-4963")
    source_rows = [
        r
        for r in read_rows(baseline_path / "per_model.csv")
        if r["variant"] == "float64_tuned" and int(r["counted_disagreements"])
    ]
    source_rows.sort(key=lambda r: -int(r["counted_disagreements"]))
    if args.file_ids:
        source_rows = [r for r in source_rows if r["file_id"] in args.file_ids]
    _, mesh_dir, cache_access = prepare_dataset(DEFAULT_CACHE)
    args.output.mkdir(parents=True)
    started = perf_counter()
    rows, model_rows = [], []
    write_json(
        args.output / "config.json",
        {
            "module": str(args.module.resolve()),
            "module_sha256": file_hash(args.module),
            "runner_sha256": file_hash(Path(__file__)),
            "exact_geometry_source_sha256": file_hash(Path("exact_mesh_diagnostics.py")),
            "generated_sources": {
                p.name: json.loads(p.read_text()) for p in args.module.parent.glob("*.json")
            },
            "selected_ids": [r["file_id"] for r in source_rows],
            "area_threshold": 1e-4,
            "epsilon": 1e-9,
            "normal_decimal_digits": args.digits,
            "mpmath_version": mp.__version__,
            "longdouble_bits": int(np.finfo(np.longdouble).nmant + 1),
            "reference": str(reference_path),
            "baseline": str(baseline_path),
            "selection": "All counted disagreements in the saved tuned full-double pipeline",
            "closest_geometry": "Exact Fraction coordinates, projections, distances and comparisons; outward-rounded bounding-box pruning",
            "normals": "Exact rational face cross products; mpmath normalization, angle weights and sums",
            "cache_access": cache_access,
        },
    )
    for source in source_rows:
        start = perf_counter()
        file_id = source["file_id"]
        vertices, faces = load_mesh(mesh_dir / f"{file_id}.npz")
        np.testing.assert_equal(file_hash(mesh_dir / f"{file_id}.npz"), source["mesh_sha256"])
        with np.load(reference_path / "points" / f"{file_id}.npz") as data:
            reference = dict(data)
        with np.load(baseline_path / "points" / f"{file_id}.npz") as data:
            baseline = dict(data)
        selected = (baseline["float64_tuned_sign_p"] != reference["sign_w"]) & reference[
            "away_from_surface"
        ]
        sample_indices = np.flatnonzero(selected)
        np.testing.assert_equal(len(sample_indices), int(source["counted_disagreements"]))
        prepared = module.PreparedMesh(vertices, faces)
        trace = prepared.trace(
            reference["points"],
            reference["closest_face_index"],
            reference["closest_point"],
            1e-4,
            1e-9,
        )
        (
            native_sign,
            normals,
            kinds,
            features,
            small,
            areas,
            bary,
            dots,
            nearest_bary,
            nearest_points,
        ) = trace
        np.testing.assert_array_equal(
            np.where(native_sign * reference["d"] < 0, 1, -1), baseline["float64_tuned_sign_p"]
        )
        same_bytes(normals, baseline["float64_tuned_pseudonormal"])
        tables = prepared.normal_tables()
        fn, vn, en, edges, edge_map = tables
        normal_tables = (fn, en, vn)
        exact = ExactMesh(vertices, faces, edges, edge_map)
        queries = reference["points"][selected]
        winding = direct_winding(vertices, faces, queries)
        igl_winding = igl.winding_number(vertices, faces, queries)
        exact_sign = np.rint(winding).astype(np.int64) * 2 - 1
        igl_sign = np.rint(igl_winding).astype(np.int64) * 2 - 1
        counts = Counter()
        for j, sample_index in enumerate(sample_indices):
            i = int(sample_index)
            q = rational_point(reference["points"][i])
            cached_c = rational_point(reference["closest_point"][i])
            face = int(reference["closest_face_index"][i])
            original_feature = (int(kinds[i]), int(features[i]))
            truth = int(reference["sign_w"][i])
            hp_normal = exact.normal(*original_feature)
            hp_sign, hp_dot = sign_of_dot(q, cached_c, hp_normal)
            precise_dot_sign, _ = sign_of_dot(q, cached_c, normals[i])
            encoded_feature = feature_from_barycentric(face, nearest_bary[i], exact)
            encoded_sign = (
                None
                if encoded_feature is None
                else sign_of_dot(
                    q, cached_c, normal_tables[encoded_feature[0]][encoded_feature[1]]
                )[0]
            )
            local_candidates = exact.closest_triangle(q, face)
            global_candidates, evaluated = exact.closest_global(q, face)
            candidate_results = {}
            for name, candidates in (("local", local_candidates), ("global", global_candidates)):
                entries = []
                for distance2, c, feature in candidates:
                    stored_normal = normal_tables[feature[0]][feature[1]]
                    if not np.isfinite(stored_normal).all():
                        stored_sign, stored_dot = None, None
                    else:
                        stored_sign, stored_dot = sign_of_dot(q, c, stored_normal)
                    precise_normal = exact.normal(*feature)
                    precise_sign, precise_dot = sign_of_dot(q, c, precise_normal)
                    entries.append(
                        {
                            "kind": ("face", "edge", "vertex")[feature[0]],
                            "index": feature[1],
                            "distance2": float(distance2),
                            "closest": [float(x) for x in c],
                            "stored_normal_sign": stored_sign,
                            "precise_normal_sign": precise_sign,
                            "stored_dot": None if stored_dot is None else float(stored_dot),
                            "precise_dot": None if precise_dot is None else float(precise_dot),
                            "normal": None
                            if precise_normal is None
                            else [float(x) for x in precise_normal],
                        }
                    )
                candidate_results[name] = entries
            minimum_distance2, exact_closest, global_feature = global_candidates[0]
            incident = exact.incident_faces(*original_feature)
            face_geometry = []
            for incident_face in incident:
                tri = vertices[faces[incident_face]].astype(np.longdouble)
                cross_norm = float(np.linalg.norm(np.cross(tri[1] - tri[0], tri[2] - tri[0])))
                edge2 = float(max(np.sum((tri[(k + 1) % 3] - tri[k]) ** 2) for k in range(3)))
                face_geometry.append(
                    {
                        "face": incident_face,
                        "cross_norm": cross_norm,
                        "relative_area": cross_norm / edge2 if edge2 else 0,
                        "normal": fn[incident_face].tolist(),
                        "precise_normal": None
                        if exact.face_normal(incident_face) is None
                        else [float(x) for x in exact.face_normal(incident_face)],
                    }
                )
            normal_error = (
                None
                if hp_normal is None
                else float(np.linalg.norm(normals[i] - np.array([float(x) for x in hp_normal])))
            )
            row = {
                "file_id": file_id,
                "sample_index": i,
                "winding_sign": truth,
                "fast_winding": float(reference["winding"][i]),
                "direct_winding": float(winding[j]),
                "igl_winding": float(igl_winding[j]),
                "direct_winding_matches": bool(exact_sign[j] == truth and igl_sign[j] == truth),
                "point": reference["points"][i].tolist(),
                "closest": reference["closest_point"][i].tolist(),
                "closest_face": face,
                "normal_kind": ("face", "edge", "vertex")[original_feature[0]],
                "normal_feature": original_feature[1],
                "normal": normals[i].tolist(),
                "normal_finite": bool(np.isfinite(normals[i]).all()),
                "small_area_branch": bool(small[i]),
                "area": float(areas[i]),
                "barycentric": bary[i].tolist(),
                "native_dot": float(dots[i]),
                "precise_dot_only_sign": precise_dot_sign,
                "precise_normal": None if hp_normal is None else [float(x) for x in hp_normal],
                "normal_error": normal_error,
                "precise_normal_sign": hp_sign,
                "precise_normal_dot": None if hp_dot is None else float(hp_dot),
                "nearest_barycentric": nearest_bary[i].tolist(),
                "nearest_encoded_feature": encoded_feature,
                "nearest_encoded_sign": encoded_sign,
                "nearest_recalculation_difference": float(
                    np.linalg.norm(nearest_points[i] - reference["closest_point"][i])
                ),
                "local_feature_matches": original_feature in [x[2] for x in local_candidates],
                "global_feature_matches": original_feature in [x[2] for x in global_candidates],
                "local_and_global_distance_equal": local_candidates[0][0] == minimum_distance2,
                "global_candidate_count": len(global_candidates),
                "evaluated_triangles": evaluated + 1,
                "closest_change_over_bbd": float(
                    np.linalg.norm(
                        np.array([float(x) for x in exact_closest]) - reference["closest_point"][i]
                    )
                    / float(source["bbd"])
                ),
                "distance_change_over_bbd": float(
                    (
                        mp.sqrt(mp.mpf(minimum_distance2.numerator) / minimum_distance2.denominator)
                        - reference["d"][i]
                    )
                    / float(source["bbd"])
                ),
                "incident_geometry": face_geometry,
                **candidate_results,
            }
            for key, corrected in {
                "precise_dot_only": precise_dot_sign == truth,
                "precise_normal_only": hp_sign == truth,
                "nearest_encoded_feature": encoded_sign == truth,
                "local_feature_stored_normal": all(
                    e["stored_normal_sign"] == truth for e in candidate_results["local"]
                ),
                "local_feature_precise_normal": all(
                    e["precise_normal_sign"] == truth for e in candidate_results["local"]
                ),
                "global_feature_stored_normal": all(
                    e["stored_normal_sign"] == truth for e in candidate_results["global"]
                ),
                "global_feature_precise_normal": all(
                    e["precise_normal_sign"] == truth for e in candidate_results["global"]
                ),
            }.items():
                row[key + "_corrects"] = bool(corrected)
                counts[key + "_corrects"] += int(corrected)
            counts["points"] += 1
            counts["small_area"] += int(small[i])
            counts["direct_winding_mismatches"] += int(not row["direct_winding_matches"])
            counts["same_local_feature"] += int(row["local_feature_matches"])
            counts["same_global_feature"] += int(row["global_feature_matches"])
            rows.append(clean(row))
        model_rows.append(
            {
                "file_id": file_id,
                "num_vertices": len(vertices),
                "num_faces": len(faces),
                "seconds": perf_counter() - start,
                **dict(counts),
            }
        )
        print(json.dumps(model_rows[-1]), flush=True)
        write_json(args.output / "cases.json", rows)
        write_csv(args.output / "per_model.csv", model_rows)
    totals = Counter()
    for row in model_rows:
        totals.update(
            {
                k: v
                for k, v in row.items()
                if k not in ("file_id", "num_vertices", "num_faces", "seconds")
            }
        )
    write_json(
        args.output / "summary.json",
        {
            "models": len(model_rows),
            "wall_seconds": perf_counter() - started,
            "baseline_sign_and_normal_bytes_match": True,
            "totals": dict(totals),
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--digits", type=int, default=80)
    parser.add_argument("--file-ids", nargs="*")
    run(parser.parse_args())
