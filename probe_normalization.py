"""Trace invalid normals and ablate final vertex/edge normalization on cached points."""

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from benchmark import load_mesh, write_csv
from dataset_cache import DEFAULT_CACHE, prepare_dataset, write_json
from validate_tolerances import SETTINGS, file_hash, import_module, read_rows


def same_bytes(actual, expected):
    """Require identical native outputs, including NaN payloads."""
    np.testing.assert_array_equal(
        np.ascontiguousarray(actual).view(np.uint8),
        np.ascontiguousarray(expected).view(np.uint8),
    )


def run(args):
    """Compare four normalization settings with fixed geometry and tolerances."""
    started = perf_counter()
    if args.output.exists():
        raise FileExistsError(f"Choose a fresh output directory: {args.output}")
    os.environ["IGL_NUM_THREADS"] = "4"
    module = import_module(args.module)
    _, meshes, cache = prepare_dataset(DEFAULT_CACHE)
    validation = read_rows(args.validation / "per_model.csv")
    bad_ids = {r["file_id"] for r in validation if int(r["invalid_queries"])}
    initial_ids = {r["file_id"] for r in read_rows(args.initial / "per_model.csv")}
    ids = bad_ids | initial_ids
    reference_rows = {r["file_id"]: r for r in read_rows(args.reference / "per_model.csv")}
    args.output.mkdir(parents=True)
    (args.output / "points").mkdir()
    config = {
        "module": str(args.module.resolve()),
        "module_sha256": file_hash(args.module),
        "runner_sha256": file_hash(Path(__file__)),
        "reference": str(args.reference),
        "validation": str(args.validation),
        "file_ids": sorted(ids, key=int),
        "invalid_model_ids": sorted(bad_ids, key=int),
        "initial_ids": sorted(initial_ids, key=int),
        "settings": SETTINGS,
        "selection": "Original 100 models plus every model with an invalid output in full validation",
        "changes": "Only final vertex and/or edge normalization removed; face normals and weights unchanged",
        "mask": "Saved original double distance > 0.001 * bbd",
        "denominator": "All samples",
        "source_provenance": {
            p.name: json.loads(p.read_text()) for p in args.module.parent.glob("*.json")
        },
    }
    write_json(args.output / "config.json", config)
    write_json(args.output / "cache_access.json", cache)
    rows, normal_rows, invalid_cases = [], [], []
    modes = {
        "normalized": (True, True),
        "raw_vertex": (False, True),
        "raw_edge": (True, False),
        "raw_both": (False, False),
    }
    audit_points = 0
    for index, file_id in enumerate(sorted(ids, key=int)):
        mesh_path = meshes / f"{file_id}.npz"
        if file_hash(mesh_path) != reference_rows[file_id]["mesh_sha256"]:
            raise ValueError(f"Mesh changed: {file_id}")
        vertices, faces = load_mesh(mesh_path)
        with np.load(args.reference / "points" / f"{file_id}.npz") as archive:
            ref = dict(archive)
        with np.load(args.validation / "points" / f"{file_id}.npz") as archive:
            saved = dict(archive)
        away, sign_w = ref["away_from_surface"], ref["sign_w"]
        output = {}
        for precision, setting in SETTINGS.items():
            dtype = np.dtype(precision).type
            points = ref["points"].astype(dtype)
            v = vertices.astype(dtype)
            if precision == "float32":
                indices = saved["float32_closest_face_index"]
                closest = saved["float32_closest_point"]
                distance = np.abs(saved["float32_signed_distance"])
                cls = module.PreparedMesh
            else:
                indices, closest, distance = (
                    ref["closest_face_index"],
                    ref["closest_point"],
                    ref["d"],
                )
                cls = module.PreparedMeshDouble
            prep_start = perf_counter()
            prepared = cls(v, faces)
            prep_seconds = perf_counter() - prep_start
            face_n, vertex_n, edge_n, raw_v, raw_e, edges, edge_map = prepared.normals()
            renorm_v, renorm_e, norm2_v, norm2_e = prepared.normalization_checks()
            same_bytes(renorm_v, vertex_n)
            same_bytes(renorm_e, edge_n)
            for feature, raw, normalized, norm2 in [
                ("vertex", raw_v, vertex_n, norm2_v),
                ("edge", raw_e, edge_n, norm2_e),
            ]:
                raw_finite = np.isfinite(raw).all(axis=1)
                nonzero = (raw != 0).any(axis=1)
                invalid = ~np.isfinite(normalized).all(axis=1)
                normal_rows.append(
                    {
                        "file_id": file_id,
                        "precision": precision,
                        "feature": feature,
                        "rows": len(raw),
                        "invalid_face_normals": int((~np.isfinite(face_n).all(axis=1)).sum()),
                        "invalid_raw": int((~raw_finite).sum()),
                        "invalid_normalized": int(invalid.sum()),
                        "raw_exact_zero": int((raw_finite & ~nonzero).sum()),
                        "raw_nonzero_norm_underflow": int(
                            (raw_finite & nonzero & (norm2 == 0)).sum()
                        ),
                        "raw_nonzero_squared_norm_subnormal": int(
                            (
                                raw_finite & nonzero & (norm2 > 0) & (norm2 < np.finfo(dtype).tiny)
                            ).sum()
                        ),
                        "normalization_created_invalid": int((raw_finite & invalid).sum()),
                    }
                )
            for label, epsilon in [("default", 1e-12), ("tuned", setting["epsilon"])]:
                baseline_sign = saved[f"{precision}_{label}_sign_p"]
                baseline_valid = saved[f"{precision}_{label}_valid_query"]
                baseline_disagree = (baseline_sign != sign_w) & away
                baseline_normals = (
                    saved[f"{precision}_tuned_pseudonormal"]
                    if label == "tuned"
                    else (
                        saved["float32_pseudonormal"]
                        if precision == "float32"
                        else ref["pseudonormal"]
                    )
                )
                for mode, (normalize_vertex, normalize_edge) in modes.items():
                    query_start = perf_counter()
                    native_sign, normals = prepared.classify(
                        points,
                        indices,
                        closest,
                        setting["area"],
                        epsilon,
                        normalize_vertex,
                        normalize_edge,
                    )
                    seconds = perf_counter() - query_start
                    sign = np.where(native_sign * distance < 0, 1, -1).astype(np.int8)
                    valid = np.isfinite(normals).all(axis=1)
                    zero = valid & (normals == 0).all(axis=1)
                    disagree = (sign != sign_w) & away
                    if mode == "normalized":
                        np.testing.assert_array_equal(sign, baseline_sign)
                        np.testing.assert_array_equal(valid, baseline_valid)
                        same_bytes(normals, baseline_normals)
                        audit_points += len(points)
                    variant = f"{precision}_{label}_{mode}"
                    output[variant + "_sign"] = sign
                    output[variant + "_valid"] = valid
                    output[variant + "_zero_normal"] = zero
                    rows.append(
                        {
                            "file_id": file_id,
                            "precision": precision,
                            "tolerances": label,
                            "mode": mode,
                            "points": len(points),
                            "counted_disagreements": int(disagree.sum()),
                            "finite_disagreements": int((disagree & valid).sum()),
                            "invalid_queries": int((~valid).sum()),
                            "zero_normal_queries": int(zero.sum()),
                            "zero_normal_disagreements": int((zero & disagree).sum()),
                            "sign_changes": int((sign != baseline_sign).sum()),
                            "finite_baseline_sign_changes": int(
                                ((sign != baseline_sign) & baseline_valid).sum()
                            ),
                            "resolved_disagreements": int((baseline_disagree & ~disagree).sum()),
                            "new_disagreements": int((~baseline_disagree & disagree).sum()),
                            "precompute_seconds": prep_seconds,
                            "classify_seconds": seconds,
                        }
                    )
                    if mode == "raw_both" and label == "tuned":
                        for point_index in np.flatnonzero(~baseline_valid):
                            i = int(point_index)
                            raw = normals[i]
                            # Match only candidates of the selected closest face.
                            f = int(indices[i])
                            candidates = [("face", f, face_n[f], face_n[f])]
                            candidates += [
                                ("vertex", int(k), raw_v[k], vertex_n[k]) for k in faces[f]
                            ]
                            candidates += [
                                ("edge", int(k), raw_e[k], edge_n[k])
                                for k in edge_map[f + np.arange(3) * len(faces)]
                            ]
                            matches = [
                                (kind, feature_id)
                                for kind, feature_id, candidate, normalized in candidates
                                if np.array_equal(raw, candidate, equal_nan=True)
                                and np.array_equal(baseline_normals[i], normalized, equal_nan=True)
                            ]
                            invalid_cases.append(
                                {
                                    "file_id": file_id,
                                    "sample_index": i,
                                    "precision": precision,
                                    "closest_face_index": f,
                                    "matched_features": json.dumps(matches),
                                    "normalized_normal": str(baseline_normals[i].tolist()),
                                    "raw_normal": str(raw.tolist()),
                                    "raw_norm_float64": float(
                                        np.linalg.norm(raw.astype(np.float64))
                                    ),
                                    "raw_exact_zero": bool(zero[i]),
                                    "raw_valid": bool(valid[i]),
                                    "reference_sign": int(sign_w[i]),
                                    "normalized_sign": int(baseline_sign[i]),
                                    "raw_sign": int(sign[i]),
                                    "beyond_cutoff": bool(away[i]),
                                }
                            )
            del prepared
        np.savez_compressed(args.output / "points" / f"{file_id}.npz", **output)
        if (index + 1) % 20 == 0 or file_id in bad_ids:
            print(
                f"{index + 1}/{len(ids)} file_id={file_id} elapsed={perf_counter() - started:.2f}s",
                flush=True,
            )
    write_csv(args.output / "per_model.csv", rows)
    write_csv(args.output / "normal_diagnostics.csv", normal_rows)
    write_csv(args.output / "invalid_queries.csv", invalid_cases)
    aggregate = []
    group_ids = {"all": ids, "original_100": initial_ids, "invalid_models": bad_ids}
    for group, members in group_ids.items():
        for precision in SETTINGS:
            for label in ("default", "tuned"):
                for mode in modes:
                    selected = [
                        r
                        for r in rows
                        if r["file_id"] in members
                        and r["precision"] == precision
                        and r["tolerances"] == label
                        and r["mode"] == mode
                    ]
                    total = {
                        "group": group,
                        "models": len(selected),
                        "precision": precision,
                        "tolerances": label,
                        "mode": mode,
                    }
                    for key in selected[0]:
                        if key not in (
                            "file_id",
                            "precision",
                            "tolerances",
                            "mode",
                            "precompute_seconds",
                        ):
                            total[key] = sum(r[key] for r in selected)
                    total["percent_all"] = 100 * total["counted_disagreements"] / total["points"]
                    aggregate.append(total)
    write_csv(args.output / "aggregate.csv", aggregate)
    write_json(
        args.output / "summary.json",
        {
            "models": len(ids),
            "points_per_variant": sum(
                int(r["num_points"]) for k, r in reference_rows.items() if k in ids
            ),
            "wall_seconds": perf_counter() - started,
            "audited_baseline_point_variants": audit_points,
            "normal_accumulator_reproduction": "Every raw matrix, renormalized, matches the original bytes",
            "aggregate": aggregate,
        },
    )
    print(
        json.dumps(
            {
                "models": len(ids),
                "seconds": perf_counter() - started,
                "invalid_queries_traced": len(invalid_cases),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=Path("results/full-4963"))
    parser.add_argument(
        "--validation", type=Path, default=Path("results/libigl-tolerance-validation-4963")
    )
    parser.add_argument("--initial", type=Path, default=Path("results/initial-100"))
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
