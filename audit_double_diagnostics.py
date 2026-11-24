"""Audit precision stability and certify strict crossings in the residual mesh."""

import argparse
import json
from collections import Counter
from pathlib import Path

import mpmath as mp
import numpy as np

from benchmark import load_mesh
from dataset_cache import DEFAULT_CACHE, prepare_dataset, write_json
from exact_mesh_diagnostics import ExactMesh, add, cross, dot, precise, scale, sub
from validate_tolerances import file_hash, read_rows


def strict_crossing(start, end, triangle):
    """Certify an edge-interior crossing of a triangle interior using rationals."""
    a, b, c = triangle
    normal = cross(sub(b, a), sub(c, a))
    norm2 = dot(normal, normal)
    if not norm2:
        return None
    d0, d1 = dot(sub(start, a), normal), dot(sub(end, a), normal)
    if d0 * d1 >= 0:
        return None
    t = d0 / (d0 - d1)
    point = add(start, scale(sub(end, start), t))
    beta = dot(cross(sub(point, a), sub(c, a)), normal) / norm2
    gamma = dot(cross(sub(b, a), sub(point, a)), normal) / norm2
    bary = (1 - beta - gamma, beta, gamma)
    if not all(x > 0 for x in bary):
        return None
    np.testing.assert_equal(dot(sub(point, a), normal), 0)
    np.testing.assert_equal(0 < t < 1, True)
    return point, t, bary, d0, d1


def run(args):
    """Cross-check the diagnostic outputs and record exact geometric witnesses."""
    mp.mp.dps = 120
    cases = json.loads((args.results / "cases.json").read_text())
    high_precision = json.loads((args.check / "cases.json").read_text())
    by_id = {(r["file_id"], r["sample_index"]): r for r in high_precision}
    np.testing.assert_equal(len(by_id), len(cases))
    for row in cases:
        comparison = by_id[row["file_id"], row["sample_index"]]
        for field in (
            "precise_dot_only_sign",
            "precise_normal_sign",
            "nearest_encoded_sign",
            "global_feature_precise_normal_corrects",
            "local_feature_precise_normal_corrects",
        ):
            np.testing.assert_equal(row[field], comparison[field])
        for level in ("local", "global"):
            np.testing.assert_equal(len(row[level]), len(comparison[level]))
            for a, b in zip(row[level], comparison[level], strict=True):
                for field in ("kind", "index", "precise_normal_sign", "stored_normal_sign"):
                    np.testing.assert_equal(a[field], b[field])
    metadata_path, mesh_dir, _ = prepare_dataset(DEFAULT_CACHE)
    metadata = next(r for r in read_rows(metadata_path) if r["file_id"] == "1452672")
    vertices, faces = load_mesh(mesh_dir / "1452672.npz")
    exact = ExactMesh(vertices, faces, np.empty((0, 2), dtype=int), np.empty(0, dtype=int))
    witnesses = []
    for first in range(len(faces)):
        candidates = np.flatnonzero(
            (exact.upper >= exact.lower[first]).all(axis=1)
            & (exact.lower <= exact.upper[first]).all(axis=1)
        )
        for second in candidates:
            if second <= first:
                continue
            first_tri, second_tri = exact.triangle(first), exact.triangle(int(second))
            found = False
            for segment_face, target_face, segments, triangle in (
                (first, int(second), first_tri, second_tri),
                (int(second), first, second_tri, first_tri),
            ):
                for edge in range(3):
                    hit = strict_crossing(segments[edge], segments[(edge + 1) % 3], triangle)
                    if hit is None:
                        continue
                    point, t, bary, d0, d1 = hit
                    distances = [
                        dot(sub(point, exact.vertex(i)), sub(point, exact.vertex(i)))
                        for i in range(len(vertices))
                    ]
                    nearest_vertex = min(range(len(vertices)), key=lambda i: distances[i])
                    witnesses.append(
                        {
                            "faces": [first, int(second)],
                            "segment_face": segment_face,
                            "segment_vertices": [
                                int(faces[segment_face, edge]),
                                int(faces[segment_face, (edge + 1) % 3]),
                            ],
                            "target_face": target_face,
                            "segment_t_exact": str(t),
                            "target_barycentrics_exact": [str(x) for x in bary],
                            "signed_plane_distances_exact": [str(d0), str(d1)],
                            "point_exact": [str(x) for x in point],
                            "point_decimal": [mp.nstr(precise(x), 60) for x in point],
                            "nearest_vertex": nearest_vertex,
                            "distance_to_nearest_vertex": float(
                                mp.sqrt(precise(distances[nearest_vertex]))
                            ),
                        }
                    )
                    found = True
                    break
                if found:
                    break
    np.testing.assert_equal(len(witnesses) > 0, True)
    oriented_edges = {}
    for face in faces:
        for i in range(3):
            a, b = int(face[i]), int(face[(i + 1) % 3])
            oriented_edges.setdefault(tuple(sorted((a, b))), []).append((a, b))
    orientation_errors = sum(len(v) != 2 or v[0] != v[1][::-1] for v in oriented_edges.values())
    summary = {
        "cases": len(cases),
        "precision_stability_digits": [80, 120],
        "all_corrected_signs_and_features_stable": True,
        "check_cases_sha256": file_hash(args.check / "cases.json"),
        "check_config": json.loads((args.check / "config.json").read_text()),
        "max_direct_igl_winding_difference": max(
            abs(r["direct_winding"] - r["igl_winding"]) for r in cases
        ),
        "minimum_absolute_native_dot": min(abs(r["native_dot"]) for r in cases),
        "maximum_normal_vector_difference": max(r["normal_error"] or 0 for r in cases),
        "classifier_feature_changes": dict(
            Counter(
                r["normal_kind"] + "_to_" + r["local"][0]["kind"]
                for r in cases
                if not r["local_feature_matches"]
            )
        ),
        "residual_geometry": {
            "file_id": "1452672",
            "metadata": metadata,
            "orientation_or_incidence_errors": orientation_errors,
            "strict_crossing_witnesses": witnesses,
            "scope": "Reports strict noncoplanar crossings; not a complete intersection detector. Coordinates are exact values of the cached binary64 mesh, which need not match the precision used for dataset metadata.",
        },
    }
    write_json(args.results / "audit.json", summary)
    print(
        json.dumps(
            {k: v for k, v in summary.items() if k not in ("check_config", "residual_geometry")},
            indent=2,
        )
    )
    print("Strict crossing face pairs:", [r["faces"] for r in witnesses])
    print("Distances to nearest vertex:", [r["distance_to_nearest_vertex"] for r in witnesses])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--check", type=Path, required=True)
    run(parser.parse_args())
