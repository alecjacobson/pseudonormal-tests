"""Inspect newly introduced disagreements using exact geometry of the tested inputs."""

import argparse
import json
from pathlib import Path

import mpmath as mp
import numpy as np

from benchmark import load_mesh
from dataset_cache import DEFAULT_CACHE, prepare_dataset, write_json
from exact_mesh_diagnostics import ExactMesh, direct_winding, rational_point, sign_of_dot
from validate_tolerances import file_hash, import_module


def run(args):
    """Record exact closest-feature ties and their normals for each new disagreement."""
    mp.mp.dps = 80
    cases = json.loads((args.results / "introduced.json").read_text())
    _, meshes, _ = prepare_dataset(DEFAULT_CACHE)
    float_module = import_module(args.float_trace)
    double_module = import_module(args.double_trace)
    records = []
    for fid, precision in sorted({(r["file_id"], r["precision"]) for r in cases}):
        selected = [
            r["sample_index"] for r in cases if (r["file_id"], r["precision"]) == (fid, precision)
        ]
        original_vertices, faces = load_mesh(meshes / f"{fid}.npz")
        vertices = original_vertices.astype(precision)
        module = float_module if precision == "float32" else double_module
        prepared = module.PreparedMesh(vertices, faces)
        fn, vn, en, edges, edge_map = prepared.normal_tables()[:5]
        exact = ExactMesh(vertices.astype(np.float64), faces, edges, edge_map)
        with np.load(Path("results/full-4963/points") / f"{fid}.npz") as data:
            reference = {
                k: data[k] for k in ("points", "closest_face_index", "closest_point", "sign_w")
            }
        with np.load(
            Path("results/libigl-tolerance-validation-4963/points") / f"{fid}.npz"
        ) as data:
            baseline = dict(data)
        with np.load(args.results / "points" / f"{fid}.npz") as data:
            current = dict(data)
        if precision == "float32":
            indices, closest = (
                baseline["float32_closest_face_index"],
                baseline["float32_closest_point"],
            )
        else:
            indices, closest = reference["closest_face_index"], reference["closest_point"]
        points = reference["points"].astype(precision)
        trace = prepared.trace(
            points, indices, closest, 1e-4, 0.003 if precision == "float32" else 1e-9
        )
        for i in selected:
            q = rational_point(points[i])
            face = int(indices[i])
            local = exact.closest_triangle(q, face)
            global_features, _ = exact.closest_global(q, face)
            bary = current[f"{precision}_barycentric"][i]
            positive = np.flatnonzero(bary > 0)
            if len(positive) == 1:
                new_feature = exact.feature(face, 2, int(positive[0]))
            elif len(positive) == 2:
                new_feature = exact.feature(face, 1, int(np.flatnonzero(bary == 0)[0]))
            elif len(positive) == 3:
                new_feature = (0, face)
            else:
                raise ValueError("Invalid barycentric output requires separate diagnosis")
            candidates = []
            for d2, c, feature in global_features:
                native_normal = (fn, en, vn)[feature[0]][feature[1]]
                precise_normal = exact.normal(*feature)
                candidates.append(
                    {
                        "feature": feature,
                        "distance_squared_exact": str(d2),
                        "closest_exact": [str(x) for x in c],
                        "stored_normal": native_normal.tolist(),
                        "precise_normal": None
                        if precise_normal is None
                        else [float(x) for x in precise_normal],
                        "stored_sign": sign_of_dot(q, c, native_normal)[0],
                        "precise_sign": sign_of_dot(q, c, precise_normal)[0],
                    }
                )
            records.append(
                {
                    "file_id": fid,
                    "sample_index": i,
                    "precision": precision,
                    "winding_sign": int(reference["sign_w"][i]),
                    "rounded_mesh_direct_winding": float(
                        direct_winding(
                            vertices.astype(float), faces, points[i : i + 1].astype(float)
                        )[0]
                    ),
                    "closest_face": face,
                    "face_vertices": faces[face].tolist(),
                    "rounded_triangle": vertices[faces[face]].tolist(),
                    "triangle_exactly_degenerate": exact.face_normal(face) is None,
                    "old_feature": [int(trace[2][i]), int(trace[3][i])],
                    "new_feature": new_feature,
                    "old_normal": baseline[f"{precision}_tuned_pseudonormal"][i].tolist(),
                    "new_normal": current[f"{precision}_normal"][i].tolist(),
                    "new_barycentric": bary.tolist(),
                    "local_exact_features": [r[2] for r in local],
                    "global_exact_features": candidates,
                }
            )
    result = {
        "normal_decimal_digits": 80,
        "runner_sha256": file_hash(Path(__file__)),
        "float_trace_sha256": file_hash(args.float_trace),
        "double_trace_sha256": file_hash(args.double_trace),
        "feature_encoding": "[0, face_index], [1, edge_index], [2, vertex_index]",
        "cases": records,
    }
    write_json(args.results / "regression_diagnosis.json", result)
    print(f"Diagnosed {len(records)} introduced disagreements")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--float-trace", type=Path, required=True)
    parser.add_argument("--double-trace", type=Path, required=True)
    run(parser.parse_args())
